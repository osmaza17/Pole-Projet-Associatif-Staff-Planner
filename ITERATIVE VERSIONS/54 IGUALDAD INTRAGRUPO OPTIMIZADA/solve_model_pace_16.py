import os
import gc
import logging
from pathlib import Path
os.environ["GRB_LICENSE_FILE"] = str(Path(__file__).parent / "gurobi.lic")

import time
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict, Counter
from itertools import combinations


TRAVEL_LABEL = "TRAVEL"


# Guard against empty var_dict — model.getAttr("X", {}) raises GurobiError
def _collect(model, var_dict, threshold=0.5):
    """Extract active values from a Gurobi VarDict, filtering by threshold."""
    if not var_dict:
        return {}
    return {k: v for k, v in model.getAttr("X", var_dict).items()
            if v > threshold}


def _check_feasibility(days, hours, demand_set_per_day, demand,
                       people_per_task, available_set_per_day,
                       forced_tasks_set_per_day, forced_work_set_per_day,
                       forced_rest_set_per_day,
                       max_hours_per_day=None,
                       hard_enemies=False, social_enemies=(),
                       task_location=None, travel_time=None):
    """
    Unified pre-check before launching Gurobi.
    Verifies that all hard constraints can be satisfied simultaneously.
    Returns a list of (category, message) tuples, or [] if everything is feasible.
    """
    def _match(tasks_demand, task_people):
        """
        Bipartite matching check: can the given demand be covered
        given the available pool per task?
        Returns list of (task, required, available_count, deficit).
        """
        slots, slot_people = [], []
        for t, req in tasks_demand:
            for _ in range(req):
                slots.append(t)
                slot_people.append(task_people.get(t, []))
        n = len(slots)
        match_person = {}
        match_slot   = [-1] * n

        def _dfs(s, visited):
            for person in slot_people[s]:
                if person in visited:
                    continue
                visited.add(person)
                prev = match_person.get(person, -1)
                if prev == -1 or _dfs(prev, visited):
                    match_slot[s]        = person
                    match_person[person] = s
                    return True
            return False

        for s in range(n):
            _dfs(s, set())

        unmatched = Counter(slots[s] for s in range(n) if match_slot[s] == -1)
        return [
            (t, req, len(task_people.get(t, [])), unmatched[t])
            for t, req in tasks_demand
            if unmatched.get(t, 0) > 0
        ]

    # ── Travel structures ────────────────────────────────────────────
    _travel_cost: dict = {}
    _loc_of_task: dict = {}
    if travel_time and task_location:
        for (l1, l2), k in travel_time.items():
            if k > 0:
                _travel_cost[(l1, l2)] = k
                _travel_cost[(l2, l1)] = k
        all_tasks = {t for d_list in demand_set_per_day.values() for t, h, _ in d_list}
        _loc_of_task = {t: task_location.get(t, "__same__") for t in all_tasks}

    # ── Enemy set (bidirectional) ────────────────────────────────────
    enemy_set: set = set()
    if hard_enemies and social_enemies:
        for p1, p2 in social_enemies:
            enemy_set.add((p1, p2))
            enemy_set.add((p2, p1))

    issues = []  # list of (category, message)

    for d in days:
        available_today = available_set_per_day.get(d, set())
        forced_today    = forced_tasks_set_per_day.get(d, [])
        rest_today      = {(p, h) for p, h in forced_rest_set_per_day.get(d, [])}
        work_today      = {(p, h) for p, h in forced_work_set_per_day.get(d, [])}

        # ── 1. Force: availability and qualification ─────────────────
        for p, t, h, _ in forced_today:
            if (p, h, d) not in available_today:
                issues.append(("force",
                    f"❌  {d} · {h}h  —  Force impossible: '{p}' is not marked as "
                    f"available at that time but is forced to do '{t}'.  "
                    f"→ Mark them as available at {h} on {d}, or remove the force."))
            if p not in people_per_task[t]:
                issues.append(("force",
                    f"❌  {d} · {h}h  —  Force impossible: '{p}' is not qualified "
                    f"for '{t}' but is forced to do it.  "
                    f"→ Add '{p}' as qualified for '{t}', or remove the force."))

        # ── 1b. Double force on same person and hour ─────────────────
        force_by_ph: defaultdict = defaultdict(list)
        for p, t, h, _ in forced_today:
            force_by_ph[(p, h)].append(t)
        for (p, h), tasks_forced in force_by_ph.items():
            if len(tasks_forced) > 1:
                issues.append(("force",
                    f"❌  {d} · {h}h  —  Contradictory force: '{p}' is forced to do "
                    f"{len(tasks_forced)} tasks simultaneously "
                    f"({', '.join(tasks_forced)}).  "
                    f"→ Keep only one force for '{p}' at {h} on {d}."))

        # ── 1c. Force exceeds slot demand ────────────────────────────
        force_by_th: defaultdict = defaultdict(list)
        for p, t, h, _ in forced_today:
            force_by_th[(t, h)].append(p)
        for (t, h), forced_people in force_by_th.items():
            dem = demand.get((t, h, d), 0)
            if len(forced_people) > dem:
                issues.append(("force",
                    f"❌  {d} · {h}h  —  Force exceeds demand: "
                    f"{len(forced_people)} person(s) forced to '{t}' "
                    f"({', '.join(forced_people)}) "
                    f"but demand is only {dem}.  "
                    f"→ Reduce forces for '{t}' at {h} on {d} to at most {dem}."))

        # ── 1d. Force + just_work vs max_hours_per_day ───────────────
        if max_hours_per_day:
            forces_per_person = Counter(p for p, t, h, _ in forced_today)
            for p, count in forces_per_person.items():
                max_h = max_hours_per_day.get(p, 0)
                if max_h > 0 and count > max_h:
                    issues.append(("max_day",
                        f"❌  {d}  —  Daily limit exceeded: '{p}' has {count} "
                        f"force(s) assigned but their daily limit is {max_h}h.  "
                        f"→ Reduce forces for '{p}' on {d} to at most {max_h}, "
                        f"or increase their daily limit."))

            # just_work is also hard → combined count
            jw_per_person = Counter(p for p, h in forced_work_set_per_day.get(d, []))
            for p, jw_count in jw_per_person.items():
                max_h   = max_hours_per_day.get(p, 0)
                f_count = forces_per_person.get(p, 0)
                # just_work slots not already covered by a force
                extra_jw = max(0, jw_count - f_count)
                total    = f_count + extra_jw
                if max_h > 0 and total > max_h:
                    issues.append(("max_day",
                        f"❌  {d}  —  Daily limit exceeded: '{p}' has "
                        f"{f_count} force(s) + {extra_jw} additional just_work slot(s) "
                        f"= {total}h mandatory, but their daily limit is {max_h}h.  "
                        f"→ Reduce just_work or force entries for '{p}' on {d}, "
                        f"or increase their daily limit."))

        # ── 1e. Force + travel_time incompatible ─────────────────────
        if _travel_cost and _loc_of_task:
            h_to_idx_d = {h: i for i, h in enumerate(hours[d])}
            forces_by_person: defaultdict = defaultdict(list)
            for p, t, h, _ in forced_today:
                forces_by_person[p].append((h_to_idx_d.get(h, 0), h, t))
            for p, slots in forces_by_person.items():
                for (i1, h1, t1), (i2, h2, t2) in combinations(slots, 2):
                    l1   = _loc_of_task.get(t1, "__same__")
                    l2   = _loc_of_task.get(t2, "__same__")
                    cost = _travel_cost.get((l1, l2), 0)
                    if cost == 0:
                        continue
                    gap = abs(i2 - i1)
                    if 0 < gap <= cost:
                        issues.append(("travel",
                            f"❌  {d}  —  Force + travel impossible: '{p}' is forced to "
                            f"'{t1}' ({l1}) at {h1} and '{t2}' ({l2}) at {h2}, "
                            f"but travel between {l1} and {l2} requires {cost}h "
                            f"and there is only {gap}h of margin.  "
                            f"→ Separate the forces by at least {cost + 1} hours, "
                            f"or assign '{p}' to tasks in the same location."))

        # ── 1f. Force + just_rest on the same slot ───────────────────
        for p, t, h, _ in forced_today:
            if (p, h) in rest_today:
                issues.append(("conflict",
                    f"❌  {d} · {h}h  —  Force + just_rest conflict: '{p}' is "
                    f"forced to do '{t}' but also has just_rest at that hour.  "
                    f"→ Remove the just_rest or the force for '{p}' at {h} on {d}."))

        # ── 1g. Two forced hard enemies on the same slot ─────────────
        if enemy_set:
            for (t, h), forced_people in force_by_th.items():
                for p1, p2 in combinations(forced_people, 2):
                    if (p1, p2) in enemy_set:
                        issues.append(("enemies",
                            f"❌  {d} · {h}h  —  Force + enemies incompatible: "
                            f"'{p1}' and '{p2}' are both forced to '{t}' "
                            f"but are hard enemies.  "
                            f"→ Remove one of the forces, or disable "
                            f"hard_enemies for this pair."))

        # ── 2. just_work + just_rest conflict on same slot ───────────
        for p, h in work_today & rest_today:
            issues.append(("conflict",
                f"❌  {d} · {h}h  —  Conflict: '{p}' is forced to work "
                f"AND to rest at the same hour.  "
                f"→ Remove the just_work or the just_rest for '{p}' at {h} on {d}."))

        # ── 3. just_work: does the person have any task available? ───
        forced_slots = {(p, h) for p, t, h, _ in forced_today}
        for p, h in work_today:
            if (p, h) in rest_today:
                continue  # already reported in step 2
            if (p, h) in forced_slots:
                continue  # force already assigns a task → just_work satisfied
            has_task = any(
                p in people_per_task[t] and (p, h, d) in available_today
                for t in people_per_task
                if demand.get((t, h, d), 0) > 0
            )
            if not has_task:
                issues.append(("just_work",
                    f"❌  {d} · {h}h  —  just_work impossible: '{p}' must work "
                    f"but has no available and qualified task at that hour.  "
                    f"→ Add availability or qualification for '{p}' at {h} on {d}, "
                    f"or remove the just_work."))

        # ── 4. Coverage with force and just_rest applied ─────────────
        forced_at: dict = defaultdict(list)
        for p, t, h, _ in forced_today:
            if (p, h, d) in available_today and p in people_per_task[t]:
                forced_at[(t, h)].append(p)

        for h in hours[d]:
            tasks_demand_h = [
                (t, demand[(t, h, d)])
                for t, hh, _ in demand_set_per_day.get(d, [])
                if hh == h
            ]
            if not tasks_demand_h:
                continue

            resting_h         = {p for p, hh in rest_today if hh == h}
            forced_to_other_h = {
                p
                for (tt, hh), ps in forced_at.items()
                for p in ps
                if hh == h
            }

            effective_demand: list = []
            task_people_h:    dict = {}
            for t, req in tasks_demand_h:
                forced_here = set(forced_at.get((t, h), []))
                rem_req     = req - len(forced_here)
                if rem_req <= 0:
                    continue
                avail = [
                    p for p in people_per_task[t]
                    if (p, h, d) in available_today
                    and p not in resting_h
                    and p not in forced_here
                    and p not in (forced_to_other_h - forced_here)
                    and not (enemy_set and any(
                        (p, fp) in enemy_set for fp in forced_here))
                ]
                effective_demand.append((t, rem_req))
                task_people_h[t] = avail

            for t, req, avail_count, deficit in _match(effective_demand, task_people_h):
                forced_here = set(forced_at.get((t, h), []))
                # Check whether the enemy exclusion caused the deficit
                blocked_by_enemy = enemy_set and any(
                    p for p in people_per_task[t]
                    if (p, h, d) in available_today
                    and p not in resting_h
                    and p not in forced_here
                    and p not in (forced_to_other_h - forced_here)
                    and any((p, fp) in enemy_set for fp in forced_here)
                )
                cause = (
                    f" The forced person(s) ({', '.join(forced_here)}) are "
                    f"hard enemies of the remaining candidates."
                    if blocked_by_enemy else ""
                )
                issues.append(("coverage",
                    f"❌  {d} · {h}h  —  Coverage impossible for '{t}': "
                    f"{deficit} person(s) missing after applying force, just_rest "
                    f"and enemy constraints "
                    f"({avail_count} available and qualified, {req} required).{cause}  "
                    f"→ Add availability or qualification, reduce demand for "
                    f"'{t}' to {req - deficit}, or review force/just_rest/hard_enemies "
                    f"for this slot."))

            # ── 5. Hard enemies block coverage ───────────────────────
            if enemy_set:
                for t, rem_req in effective_demand:
                    pool = task_people_h.get(t, [])
                    if len(pool) < rem_req or rem_req <= 1:
                        continue  # already detected above, or trivially ok
                    feasible = False
                    if len(pool) <= 20:  # guard against combinatorial explosion
                        for combo in combinations(pool, rem_req):
                            if not any((a, b) in enemy_set
                                       for a, b in combinations(combo, 2)):
                                feasible = True
                                break
                    else:
                        feasible = True  # pool too large to check, assume ok
                    if not feasible:
                        issues.append(("enemies",
                            f"❌  {d} · {h}h  —  Enemies block coverage: "
                            f"'{t}' needs {rem_req} person(s) but all available "
                            f"candidates are hard enemies of each other "
                            f"({', '.join(pool)}).  "
                            f"→ Add qualified staff for '{t}' without enemy conflicts, "
                            f"or disable hard_enemies for some pair."))

    return issues


def solve_model(data, ui_update_callback=None, active_model_ref=None):

    # ══════════════════════════════════════════════════════════════════
    # UNPACK
    # ══════════════════════════════════════════════════════════════════
    people       = data.get("people", [])
    tasks        = data.get("tasks", [])
    hours        = data.get("hours", {})
    days         = data.get("days", [])

    demand       = data.get("demand", {})
    availability = data.get("availability", {})
    emergency    = data.get("emergency", {})
    skills       = data.get("skills", {})
    force        = data.get("force", {})
    social       = data.get("social", {})
    pref_cost    = data.get("pref_cost", {})
    X_prev       = data.get("X_prev", {})

    max_consec_hours    = data.get("max_consec_hours", {})
    hard_enemies        = data.get("hard_enemies", False)
    just_work           = data.get("just_work", {})
    solver_params       = data.get("solver_params", {})
    captain_rules       = data.get("captain_rules", [])
    force_rest          = data.get("force_rest", {})
    quota_rules         = data.get("quota_rules", [])
    sticky              = data.get("sticky", {})
    rotation            = data.get("rotation", {})
    task_duration       = data.get("task_duration", {})
    capacity            = data.get("capacity", {})
    max_hours_per_day   = data.get("max_hours_per_day", {})
    max_hours_per_event = data.get("max_hours_per_event", {})

    task_location       = data.get("task_location", {})
    travel_time         = data.get("travel_time", {})

    # ── Objective weights ─────────────────────────────────────────────
    WEIGHTS = data.get("weights")

    W_RULE       = WEIGHTS.get("W_RULE")
    W_EMERG      = WEIGHTS.get("W_EMERG")
    W_STABILITY  = WEIGHTS.get("W_STABILITY")
    W_INTRAGROUP = WEIGHTS.get("W_INTRAGROUP")
    W_SOCIAL     = WEIGHTS.get("W_SOCIAL")
    W_GAP        = WEIGHTS.get("W_GAP")
    W_PREF       = WEIGHTS.get("W_PREF")
    W_INTERGROUP = WEIGHTS.get("W_INTERGROUP")
    W_VARIETY    = WEIGHTS.get("W_VARIETY")
    W_STICKY     = WEIGHTS.get("W_STICKY")
    W_DURATION   = WEIGHTS.get("W_DURATION")
    W_ROTATION   = WEIGHTS.get("W_ROTATION")
    W_TRAVEL     = WEIGHTS.get("W_TRAVEL")
    W_CONSEC     = WEIGHTS.get("W_CONSEC", W_RULE)
    W_EVENT      = WEIGHTS.get("W_EVENT",  W_RULE)

    groups_work_ratios = data.get("groups_work_ratios", {}) or {}

    raw_groups          = data.get("groups", {}) or {"default": list(people)}
    visual_group_list   = list(raw_groups.keys())
    visual_group_people = dict(raw_groups)

    cap = {p: capacity.get(p, 1.0) for p in people}
    del capacity

    if not groups_work_ratios:
        W_INTERGROUP    = 0
        group_list      = ["__all__"]
        group_people    = {"__all__": list(people)}
        person_to_group = {p: "__all__" for p in people}
        group_weights   = {"__all__": 1.0}
    else:
        group_list      = visual_group_list
        group_people    = visual_group_people
        person_to_group = {p: g for g, ms in group_people.items() for p in ms}

        share_vals, missing = {}, []
        for g in group_list:
            try:
                v = float(groups_work_ratios[g])
                if v < 0:
                    raise ValueError
                share_vals[g] = v
            except (TypeError, ValueError, KeyError):
                missing.append(g)

        if missing:
            raise ValueError(f"Shares missing/invalid for groups: {missing}")
        total = sum(share_vals.values())
        if total <= 1e-9:
            raise ValueError("At least one group share must be > 0.")

        group_weights = {g: share_vals[g] / total for g in group_list}

    number_of_groups = len(group_list)

    # ══════════════════════════════════════════════════════════════════
    # STATIC PRECOMPUTATIONS
    # ══════════════════════════════════════════════════════════════════

    # Inverted index: task → list of people qualified to do it
    people_per_task = defaultdict(list)
    for (p, t), v in skills.items():
        if v == 1:
            people_per_task[t].append(p)
    del skills

    # Per-day list of (task, hour, day) tuples where demand > 0
    demand_set_per_day = defaultdict(list)
    for (t, h, d), v in demand.items():
        if v > 0:
            demand_set_per_day[d].append((t, h, d))

    # Per-day set of (person, hour, day) tuples where person is available
    available_set_per_day = defaultdict(set)
    for (p, h, d), v in availability.items():
        if v == 1:
            available_set_per_day[d].add((p, h, d))
    del availability

    # Per-day list of (person, task, hour, day) hard-forced assignments
    forced_tasks_set_per_day = defaultdict(list)
    for (p, t, h, d), v in force.items():
        if v == 1:
            forced_tasks_set_per_day[d].append((p, t, h, d))
    del force

    # Per-day list of (person, hour) slots where person must work something
    forced_work_set_per_day = defaultdict(list)
    for (p, h, d), v in just_work.items():
        if v == 1:
            forced_work_set_per_day[d].append((p, h))
    del just_work

    # Per-day list of (person, hour) slots where person must not work
    forced_rest_set_per_day = defaultdict(list)
    for (p, h, d), v in force_rest.items():
        if v == 1:
            forced_rest_set_per_day[d].append((p, h))
    del force_rest

    # Lookup: (hour, day) → next hour in that day, for consecutive-hour constraints
    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)}

    # Sets of tasks marked as sticky (same person each hour) or rotation (anti-repeat)
    sticky_tasks   = {t for t in tasks if sticky.get(t, 0) == 1}
    rotation_tasks = {t for t in tasks if rotation.get(t, 0) == 1}
    del sticky, rotation

    # Friend and enemy pairs extracted from the social matrix
    social_friends = tuple((p1, p2) for (p1, p2), sv in social.items() if sv == 1)
    social_enemies = tuple((p1, p2) for (p1, p2), sv in social.items() if sv == -1)
    has_enemies    = bool(social_enemies)
    del social

    # Per-day set of (person, hour) slots flagged as emergency (penalised if worked)
    emergency_per_day: dict[str, set] = defaultdict(set)
    for (p, h, d), v in emergency.items():
        if v == 1:
            emergency_per_day[d].add((p, h))
    del emergency

    # Preference cost per (person, task) — zero entries dropped to reduce dict size
    preferences_values = {(p, t): v for (p, t), v in pref_cost.items() if v != 0}
    del pref_cost

    # Tasks that span more than one hour, with their duration as value
    multi_hour_tasks = {t: task_duration[t] for t in tasks if task_duration.get(t, 1) > 1}
    del task_duration

    # Rotation applies only to single-hour, non-sticky tasks
    effective_rotation_tasks = rotation_tasks - set(multi_hour_tasks) - sticky_tasks

    # Per-day list of captain-rule slots: (rule_idx, task, hour, captains, min_required)
    captain_slots_per_day: dict[str, list[tuple]] = defaultdict(list)
    if captain_rules and W_RULE > 0:
        for r_idx, rule in enumerate(captain_rules):
            caps    = tuple(rule["captains"])
            min_req = rule.get("min_required", 1)
            for d in days:
                for t in rule["tasks"]:
                    for h in rule.get("hours", {}).get(d, []):
                        if h in hours.get(d, []) and demand.get((t, h, d), 0) > 0:
                            captain_slots_per_day[d].append((r_idx, t, h, caps, min_req))
    del captain_rules

    # Filter quota rules to only those with valid people, tasks and days
    people_set, tasks_set = set(people), set(tasks)
    quota_rules_valid = []

    for r_idx, rule in enumerate(quota_rules):
        people_of_rule = [p for p in rule.get("people", []) if p in people_set]
        tasks_of_rule  = {t: int(q) for t, q in rule.get("tasks", {}).items()
                          if t in tasks_set and isinstance(q, (int, float)) and int(q) > 0}
        days_of_rule   = [d for d in days if d in set(rule.get("days", []))]

        if people_of_rule and tasks_of_rule and days_of_rule:
            quota_rules_valid.append({"idx": r_idx, "people": people_of_rule,
                                      "tasks": tasks_of_rule, "days": days_of_rule})
    del quota_rules, people_set, tasks_set

    # Cumulative demand up to and including each day — used for inter-group equity targets
    total_demand      = sum(demand.values())
    cumulative_demand = {}
    running = 0

    for d in days:
        running += sum(demand[k] for k in demand_set_per_day[d])
        cumulative_demand[d] = running

    # Drop zero-demand entries to keep the demand dict compact
    demand = {k: v for k, v in demand.items() if v > 0}

    # ── UNIFIED PRE-CHECK ─────────────────────────────────────────────
    feasibility_issues = _check_feasibility(
        days, hours, demand_set_per_day, demand,
        people_per_task, available_set_per_day,
        forced_tasks_set_per_day, forced_work_set_per_day,
        forced_rest_set_per_day,
        max_hours_per_day=max_hours_per_day,
        hard_enemies=hard_enemies,
        social_enemies=social_enemies,
        task_location=task_location,
        travel_time=travel_time)

    if feasibility_issues:
        _CAT_HEADER = {
            "force":    "⚙️  FORCE IMPOSSIBLE",
            "conflict": "⚡  WORK + REST CONFLICT",
            "just_work":"🔧  JUST-WORK IMPOSSIBLE",
            "coverage": "👥  INSUFFICIENT COVERAGE",
            "max_day":  "📅  DAILY LIMIT EXCEEDED",
            "enemies":  "⚔️  ENEMIES BLOCK COVERAGE",
            "travel":   "🚗  FORCE + TRAVEL INCOMPATIBLE",
        }
        seen_headers = set()
        messages = []
        for cat, msg in feasibility_issues:
            header = _CAT_HEADER.get(cat, "❌  PROBLEM")
            if header not in seen_headers:
                messages.append(f"── {header} ──")
                seen_headers.add(header)
            messages.append(msg)

        return {
            "status":                 "Infeasible: pre-check failed",
            "coverage_gaps_messages": messages,
            "assignment":             {},
            "workload":               {p: 0 for p in people},
            "day_statuses":           {d: "Not attempted" for d in days},
        }

    # Running hour totals per person, per (person, task), and per group — updated each day
    accumulated_hours       = {p: 0 for p in people}
    accumulated_task_hours  = defaultdict(int)
    accumulated_group_hours = {g: 0 for g in group_list}

    # Bidirectional travel cost between locations (in hours of transit)
    travel_cost: dict[tuple[str, str], int] = {}
    for (l1, l2), k in travel_time.items():
        if k > 0:
            travel_cost[(l1, l2)] = k
            travel_cost[(l2, l1)] = k
    del travel_time

    # Lookup: task → its location name
    location_of_task = {t: task_location.get(t, "__same__") for t in tasks}
    del task_location

    # All ordered task pairs (t1 < t2) that have a positive travel cost between them
    task_pairs_with_travel: list[tuple[str, str, int]] = []
    for t1 in tasks:
        for t2 in tasks:
            if t1 >= t2:
                continue
            k = travel_cost.get((location_of_task[t1], location_of_task[t2]), 0)
            if k > 0:
                task_pairs_with_travel.append((t1, t2, k))

    # ══════════════════════════════════════════════════════════════════
    # RESULT ACCUMULATORS
    # ══════════════════════════════════════════════════════════════════
    all_x_vals: dict = {}

    consec_relaxations: dict[str, dict[str, set]] = {}
    captain_violations: list[str] = []
    emerg_issues:       list[str] = []
    processed_days:     list[str] = []

    partial_assignment: dict[str, dict[str, dict[str, str | None]]] = {}

    # Per-day status tracking
    day_statuses: dict[str, str]   = {}
    day_mip_gaps: dict[str, float] = {}

    last_ui_update = [0.0]

    status_map = {
        GRB.OPTIMAL:     "Optimal",
        GRB.SUBOPTIMAL:  "Suboptimal",
        GRB.TIME_LIMIT:  "Time Limit Reached",
        GRB.INFEASIBLE:  "Infeasible",
        GRB.INTERRUPTED: "Interrupted by User",
        GRB.INF_OR_UNBD: "Infeasible or Unbounded"}

    solve_start = time.monotonic()

    # ══════════════════════════════════════════════════════════════════
    # MAIN LOOP  (day-by-day decomposition)
    # ══════════════════════════════════════════════════════════════════
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 1)
            env.start()

            for today in days:

                available_set_today  = available_set_per_day.get(today, set())
                demand_set_today     = demand_set_per_day.get(today, [])
                emergency_today      = emergency_per_day.get(today, set())

                forced_tasks_set_today = forced_tasks_set_per_day.get(today, [])
                forced_work_set_today  = forced_work_set_per_day.get(today, [])
                forced_rest_set_today  = forced_rest_set_per_day.get(today, [])

                hours_today = hours[today]
                h_to_idx    = {h: i for i, h in enumerate(hours_today)}

                # Pre-initialise this day's partial assignment for the UI
                partial_assignment[today] = {p: {h: None for h in hours_today} for p in people}

                with gp.Model("StaffScheduler", env=env) as model:

                    if active_model_ref is not None:
                        if active_model_ref[0] is not None:
                            active_model_ref[0].terminate()
                        active_model_ref[0] = model

                    interrupted = False
                    try:
                        obj = gp.LinExpr()

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 0 — CORE: assignment variables            ║
                        # ╚══════════════════════════════════════════════════╝
                        x_by_th:     defaultdict[tuple, list] = defaultdict(list)
                        x_by_ph:     defaultdict[tuple, list] = defaultdict(list)
                        x_by_p:      defaultdict[str,   list] = defaultdict(list)
                        x_by_pt:     defaultdict[tuple, list] = defaultdict(list)
                        x_by_p_hidx: defaultdict[str,   list] = defaultdict(list)

                        x_keys = []
                        for t, h, _ in demand_set_today:
                            for p in people_per_task[t]:
                                if (p, h, today) in available_set_today:
                                    _k = (p, t, h, today)
                                    x_keys.append(_k)
                                    x_by_th[(t, h)].append(_k)
                                    x_by_ph[(p, h)].append(_k)
                                    x_by_p[p].append(_k)
                                    x_by_pt[(p, t)].append(_k)
                                    x_by_p_hidx[p].append((h_to_idx[h], _k))

                        for _p in x_by_p_hidx:
                            x_by_p_hidx[_p].sort()

                        x = model.addVars(x_keys, vtype=GRB.BINARY)
                        x_keys_set = set(x_keys)

                        # Hard coverage: demand must be met exactly
                        model.addConstrs(
                            gp.quicksum(x[k] for k in x_by_th[(t, h)]) == demand[(t, h, today)]
                            for t, h, _ in demand_set_today)

                        # Hard: one task per person per hour
                        model.addConstrs(
                            gp.quicksum(x[k] for k in x_by_ph[(p, h)]) <= 1
                            for p, h in x_by_ph)

                        people_with_x_today = set(x_by_p)

                        # Shared index for travel blocks
                        _pt_idx: defaultdict[tuple, list] = defaultdict(list)
                        if task_pairs_with_travel and (W_TRAVEL > 0 or W_GAP > 0):
                            for _k in x_keys:
                                p, t, h, _ = _k
                                _pt_idx[(p, t)].append((h_to_idx[h], h))

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 0b — MULTI-HOUR TASK DURATION (soft)      ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_DURATION > 0 and multi_hour_tasks:
                            covers: defaultdict[tuple, list] = defaultdict(list)
                            s_set:  set = set()

                            for t, dur in multi_hour_tasks.items():
                                for p in people_per_task[t]:
                                    for si in range(len(hours_today) - dur + 1):
                                        block = hours_today[si : si + dur]
                                        if all((p, bh, today) in available_set_today for bh in block):
                                            start_h = block[0]
                                            s_set.add((p, t, start_h, today))
                                            for bh in block:
                                                covers[(p, t, bh)].append(start_h)

                            if s_set:
                                s_dur = model.addVars(s_set, vtype=GRB.BINARY)

                                dur_slack_keys = [
                                    k
                                    for t in multi_hour_tasks
                                    for p in people_per_task[t]
                                    for k in x_by_pt[(p, t)]
                                ]

                                dur_slack = model.addVars(dur_slack_keys, vtype=GRB.BINARY)
                                for v in dur_slack.values(): v.BranchPriority = -1

                                obj += W_DURATION * dur_slack.sum()

                                for p, t, h, _ in dur_slack_keys:
                                    cover_starts = covers.get((p, t, h), [])
                                    s_sum = gp.quicksum(s_dur[p, t, sh, today] for sh in cover_starts)
                                    model.addConstr(dur_slack[p, t, h, today] >= x[p, t, h, today] - s_sum)
                                    if cover_starts:
                                        model.addConstr(x[p, t, h, today] >= s_sum)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 1 — FORCE MANDATES (hard)                 ║
                        # ║  Pre-check guarantees feasibility.               ║
                        # ╚══════════════════════════════════════════════════╝
                        force_keys = [
                            (p, t, h, today)
                            for p, t, h, _ in forced_tasks_set_today
                            if (t, h, today) in demand
                            and (p, t, h, today) in x_keys_set
                        ]

                        if force_keys:
                            model.addConstrs(
                                x[p, t, h, today] == 1
                                for p, t, h, _ in force_keys)

                        if forced_work_set_today:
                            model.addConstrs(
                                gp.quicksum(x[k] for k in x_by_ph[(p, h)]) >= 1
                                for p, h in forced_work_set_today
                                if x_by_ph[(p, h)])

                        if forced_rest_set_today:
                            model.addConstrs(
                                gp.quicksum(x[k] for k in x_by_ph[(p, h)]) == 0
                                for p, h in forced_rest_set_today
                                if x_by_ph[(p, h)])

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 2 — REST & WORKLOAD LIMITS                ║
                        # ║    2a. Max consecutive hours      (soft)          ║
                        # ║    2b. Max hours per day          (hard)          ║
                        # ║    2c. Max hours per event        (soft)          ║
                        # ╚══════════════════════════════════════════════════╝

                        # 2a — Max consecutive hours (soft, penalised by W_CONSEC)
                        # sc_consec[s] = slack for sliding window s.
                        # ub=min_rest: worst case is working the entire window
                        # (0 rest hours), which requires a slack of min_rest.
                        if W_CONSEC > 0 and max_consec_hours:
                            for person, consec_config in max_consec_hours.items():
                                max_work, min_rest = (
                                    consec_config if isinstance(consec_config, tuple)
                                    else (int(consec_config), 1))
                                if person not in people or max_work <= 0 or min_rest <= 0:
                                    continue

                                window = max_work + min_rest
                                n_hrs  = len(hours_today)
                                if n_hrs < window or person not in people_with_x_today:
                                    continue

                                _pidx  = x_by_p_hidx[person]
                                n_wins = n_hrs - window + 1

                                sc_consec = model.addVars(
                                    range(n_wins), lb=0, ub=min_rest, vtype=GRB.INTEGER)
                                obj += W_CONSEC * sc_consec.sum()
                                model.addConstrs(
                                    gp.quicksum(
                                        x[k] for idx, k in _pidx if s <= idx < s + window)
                                    <= max_work + sc_consec[s]
                                    for s in range(n_wins))

                        # 2b — Max hours per day (hard)
                        for person, max_h in max_hours_per_day.items():
                            if person in people_with_x_today and max_h > 0:
                                model.addConstr(
                                    gp.quicksum(x[k] for k in x_by_p[person]) <= max_h)

                        # 2c — Max hours per event (soft, penalised by W_EVENT)
                        # sc_ev absorbs any overflow beyond the event-wide cap.
                        for person, max_h in max_hours_per_event.items():
                            if person in people_with_x_today and max_h > 0:
                                sc_ev = model.addVar(lb=0, vtype=GRB.INTEGER)
                                obj += W_EVENT * sc_ev
                                model.addConstr(
                                    accumulated_hours[person]
                                    + gp.quicksum(x[k] for k in x_by_p[person])
                                    <= max_h + sc_ev)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 3 — EMERGENCY                             ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_EMERG > 0 and emergency_today:
                            obj += W_EMERG * gp.quicksum(
                                x[p, t, h, today]
                                for p, t, h, _ in x_keys
                                if (p, h) in emergency_today)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 4 — INTRA-GROUP EQUITY      (DECOUPLED)   ║
                        # ╚══════════════════════════════════════════════════╝

                        if W_INTRAGROUP > 0:
                            group_cap_sum = {
                                g: sum(cap[p] for p in group_people[g])
                                for g in group_list
                            }

                            for group in group_list:
                                g_cap = group_cap_sum[group]
                                if g_cap <= 1e-9:
                                    continue

                                all_members = group_people[group]
                                if len(all_members) < 2:
                                    continue

                                active_members = [p for p in all_members if p in people_with_x_today]
                                if not active_members:
                                    continue

                                eq_members = [p for p in active_members if cap[p] > 1e-9]
                                if not eq_members:
                                    continue

                                # ── Auxiliary: total group hours (accumulated + today) ──────
                                acc_sum = sum(accumulated_hours[p] for p in all_members)
                                G_total = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
                                model.addConstr(
                                    G_total == acc_sum + gp.quicksum(
                                        x[k] for p in active_members for k in x_by_p[p]))

                                # ── Per-person deviation (now SPARSE) ───────────────────────
                                delta_plus  = model.addVars(eq_members, lb=0, vtype=GRB.CONTINUOUS)
                                delta_minus = model.addVars(eq_members, lb=0, vtype=GRB.CONTINUOUS)
                                obj += W_INTRAGROUP * (delta_plus.sum() + delta_minus.sum())

                                model.addConstrs(
                                    accumulated_hours[p]
                                    + gp.quicksum(x[k] for k in x_by_p[p])
                                    - (cap[p] / g_cap) * G_total
                                    == delta_plus[p] - delta_minus[p]
                                    for p in eq_members)
                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 5 — INTER-GROUP EQUITY                    ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_INTERGROUP > 0 and number_of_groups > 1 and groups_work_ratios:
                            # gamma_plus/minus: real-valued group deviations
                            gamma_plus  = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                            gamma_minus = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                            obj += W_INTERGROUP * (gamma_plus.sum() + gamma_minus.sum())

                            x_by_group: defaultdict[str, list] = defaultdict(list)
                            for p, t, h, _ in x_keys:
                                x_by_group[person_to_group[p]].append((p, t, h))

                            cum_demand_today = cumulative_demand[today]
                            model.addConstrs(
                                accumulated_group_hours[g]
                                + gp.quicksum(x[p, t, h, today] for p, t, h in x_by_group[g])
                                - gamma_plus[g] + gamma_minus[g]
                                == cum_demand_today * group_weights[g]
                                for g in group_list)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 6 — SOCIAL                                ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_SOCIAL > 0 or hard_enemies:
                            active_th_per_person: defaultdict[str, set] = defaultdict(set)
                            for p, t, h, _ in x_keys:
                                active_th_per_person[p].add((t, h))

                            def _shared_slots(pairs):
                                return [
                                    (p1, p2, t, h, today)
                                    for p1, p2 in pairs
                                    for t, h in active_th_per_person[p1]
                                    if (t, h) in active_th_per_person[p2]
                                ]

                            if W_SOCIAL > 0:
                                friends_keys = _shared_slots(social_friends)
                                # f ∈ {0,1}: |x[p1] - x[p2]| indicator
                                f = model.addVars(friends_keys, vtype=GRB.BINARY)
                                for v in f.values(): v.BranchPriority = -1
                                obj += W_SOCIAL * f.sum()
                                model.addConstrs(
                                    x[p1, t, h, today] - x[p2, t, h, today] <= f[p1, p2, t, h, today]
                                    for p1, p2, t, h, _ in friends_keys)
                                model.addConstrs(
                                    x[p2, t, h, today] - x[p1, t, h, today] <= f[p1, p2, t, h, today]
                                    for p1, p2, t, h, _ in friends_keys)

                                if not hard_enemies:
                                    enemies_keys = _shared_slots(social_enemies)
                                    # e ∈ {0,1}: x[p1] + x[p2] > 1 indicator
                                    e = model.addVars(enemies_keys, vtype=GRB.BINARY)
                                    for v in e.values(): v.BranchPriority = -1
                                    obj += W_SOCIAL * e.sum()
                                    model.addConstrs(
                                        x[p1, t, h, today] + x[p2, t, h, today] - e[p1, p2, t, h, today] <= 1
                                        for p1, p2, t, h, _ in enemies_keys)

                            if hard_enemies:
                                enemies_scope = _shared_slots(social_enemies)
                                model.addConstrs(
                                    x[p1, t, h, today] + x[p2, t, h, today] <= 1
                                    for p1, p2, t, h, _ in enemies_scope)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 7 — QUOTA RULES (soft)                    ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_RULE > 0 and quota_rules_valid:
                            for r in quota_rules_valid:
                                if today not in r["days"]:
                                    continue
                                for task, required_hours in r["tasks"].items():
                                    capped_req = min(required_hours, len(hours_today))
                                    for person in r["people"]:
                                        pvars = [
                                            x[person, task, h, today]
                                            for h in hours_today
                                            if (person, task, h, today) in x_keys_set
                                        ]
                                        qs = model.addVar(lb=0, ub=capped_req, vtype=GRB.INTEGER)
                                        obj += W_RULE * qs
                                        model.addConstr(gp.quicksum(pvars) + qs >= capped_req)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 8 — STABILITY                             ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_STABILITY > 0 and X_prev:
                            # z ∈ {0,1}: absolute difference between two binary schedules
                            z = model.addVars(x_keys, vtype=GRB.BINARY)
                            for v in z.values(): v.BranchPriority = -1
                            obj += W_STABILITY * z.sum()
                            model.addConstrs(z[k] >= X_prev.get(k, 0) - x[k] for k in x_keys)
                            model.addConstrs(z[k] >= x[k] - X_prev.get(k, 0) for k in x_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 9 — TRANSIT DETECTION (before gaps)       ║
                        # ╚══════════════════════════════════════════════════╝
                        in_transit_keys: set = set()
                        in_transit_vars: dict = {}

                        if task_pairs_with_travel and W_GAP > 0:
                            _tlinks: list = []
                            for t1, t2, k in task_pairs_with_travel:
                                for p in people_with_x_today:
                                    pts1 = _pt_idx[(p, t1)]
                                    pts2 = _pt_idx[(p, t2)]
                                    if not pts1 or not pts2:
                                        continue
                                    for i1, h1 in pts1:
                                        for i2, h2 in pts2:
                                            if i1 == i2:
                                                continue
                                            i_lo, h_lo, t_lo = (i1, h1, t1) if i1 < i2 else (i2, h2, t2)
                                            i_hi, h_hi, t_hi = (i2, h2, t2) if i1 < i2 else (i1, h1, t1)
                                            for mi in range(i_lo + 1, min(i_lo + k + 1, i_hi)):
                                                hm = hours_today[mi]
                                                in_transit_keys.add((p, hm, today))
                                                _tlinks.append((p, t_lo, h_lo, t_hi, h_hi, hm))

                            if in_transit_keys:
                                # in_transit_vars ∈ {0,1}: 1 iff person is travelling at hour hm
                                in_transit_vars = model.addVars(in_transit_keys, vtype=GRB.BINARY)
                                for v in in_transit_vars.values(): v.BranchPriority = -1
                                for p, ta, ha, tb, hb, hm in _tlinks:
                                    model.addConstr(
                                        in_transit_vars[p, hm, today]
                                        >= x[p, ta, ha, today] + x[p, tb, hb, today] - 1)

                            del _tlinks

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 10 — GAPS (transit-aware)                 ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_GAP > 0:
                            g_keys = {(p, h, today) for p, t, h, _ in x_keys}

                            # g ∈ {0,1}: 1 iff person "restarts" a work block at hour h
                            g  = model.addVars(g_keys, vtype=GRB.BINARY)
                            for v in g.values(): v.BranchPriority = -1
                            # rr: number of gaps (block-restarts − 1)
                            # Σg is integer → rr is always integer at optimality
                            rr = model.addVars(people_with_x_today, lb=0, vtype=GRB.INTEGER)
                            obj += W_GAP * rr.sum()

                            if in_transit_keys:
                                active_keys  = g_keys | in_transit_keys
                                only_work    = g_keys - in_transit_keys
                                only_transit = in_transit_keys - g_keys
                                both_keys    = g_keys & in_transit_keys

                                # act ∈ {0,1}: 1 iff person is active (working or in transit)
                                act = model.addVars(active_keys, vtype=GRB.BINARY)
                                for v in act.values(): v.BranchPriority = -1

                                model.addConstrs(
                                    act[p, h, today] == gp.quicksum(x[k] for k in x_by_ph[(p, h)])
                                    for p, h, _ in only_work)
                                model.addConstrs(
                                    act[phd] == in_transit_vars[phd]
                                    for phd in only_transit)
                                for p, h, _ in both_keys:
                                    xs = gp.quicksum(x[k] for k in x_by_ph[(p, h)])
                                    model.addConstr(act[p, h, today] >= xs)
                                    model.addConstr(act[p, h, today] >= in_transit_vars[p, h, today])
                                    model.addConstr(act[p, h, today] <= xs + in_transit_vars[p, h, today])

                                def _a(p, h):
                                    k = (p, h, today)
                                    return act[k] if k in active_keys else 0

                                model.addConstrs(
                                    g[p, hours_today[0], today] >= _a(p, hours_today[0])
                                    for p in people_with_x_today
                                    if (p, hours_today[0], today) in g_keys)
                                model.addConstrs(
                                    g[p, hn, today] >= _a(p, hn) - _a(p, hb)
                                    for p in people_with_x_today
                                    for hb, hn in zip(hours_today[:-1], hours_today[1:])
                                    if (p, hn, today) in g_keys)
                            else:
                                model.addConstrs(
                                    g[p, hours_today[0], today] == gp.quicksum(
                                        x[k] for k in x_by_ph[(p, hours_today[0])])
                                    for p in people_with_x_today
                                    if (p, hours_today[0], today) in g_keys)
                                model.addConstrs(
                                    g[p, hn, today] >= gp.quicksum(x[k] for k in x_by_ph[(p, hn)])
                                                     - gp.quicksum(x[k] for k in x_by_ph[(p, hb)])
                                    for p in people_with_x_today
                                    for hb, hn in zip(hours_today[:-1], hours_today[1:])
                                    if (p, hn, today) in g_keys)

                            model.addConstrs(
                                rr[p] >= gp.quicksum(
                                    g[p, h, today] for h in hours_today if (p, h, today) in g_keys) - 1
                                for p in people_with_x_today)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 11 — PREFERENCES                          ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_PREF > 0 and preferences_values:
                            obj += W_PREF * gp.quicksum(
                                preferences_values[pt] * x[k]
                                for pt in preferences_values
                                if pt in x_by_pt
                                for k in x_by_pt[pt])

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 12 — TASK VARIETY                         ║
                        # ╚══════════════════════════════════════════════════╝
                        quota_required_today: defaultdict[tuple, int] = defaultdict(int)
                        for r in quota_rules_valid:
                            if today in r["days"]:
                                for task, req in r["tasks"].items():
                                    for person in r["people"]:
                                        quota_required_today[person, task] = max(
                                            quota_required_today[person, task], req)

                        if W_VARIETY > 0:
                            v_max_keys = [pt for pt in x_by_pt if x_by_pt[pt]]

                            v_max = model.addVars(v_max_keys, lb=0, vtype=GRB.INTEGER)
                            obj += W_VARIETY * v_max.sum()

                            for p, t in v_max_keys:
                                free_pass = max(
                                    0 if accumulated_task_hours[p, t] > 0 else 1,
                                    quota_required_today[p, t],
                                    len(hours_today) if t in sticky_tasks else 0,
                                    multi_hour_tasks.get(t, 0),
                                )
                                model.addConstr(
                                    v_max[p, t] >= gp.quicksum(x[k] for k in x_by_pt[(p, t)]) - free_pass)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 13 — CAPTAIN RULES (soft)                 ║
                        # ╚══════════════════════════════════════════════════╝
                        w = {}
                        slot_eligibles: list[list[str]] = []
                        sole_captain_hours_per_person: dict[str, frozenset] = {}

                        if W_RULE > 0 and captain_slots_per_day.get(today):
                            slots = captain_slots_per_day[today]

                            slot_eligibles = [
                                [c for c in caps if (c, t, h, today) in x_keys_set]
                                for (_, t, h, caps, _) in slots
                            ]

                            n_slots = len(slots)
                            w = model.addVars(n_slots, lb=0, vtype=GRB.INTEGER)
                            obj += W_RULE * w.sum()

                            model.addConstrs(
                                gp.quicksum(x[c, t, h, today] for c in slot_eligibles[i]) + w[i] >= min_req
                                for i, (_, t, h, _, min_req) in enumerate(slots))

                            sole: defaultdict[str, set] = defaultdict(set)
                            for i, (_, _, h, _, _) in enumerate(slots):
                                if len(slot_eligibles[i]) == 1:
                                    sole[slot_eligibles[i][0]].add(h)

                            sole_captain_hours_per_person = {p: frozenset(hrs) for p, hrs in sole.items()}
                            for p, hrs in sole_captain_hours_per_person.items():
                                consec_relaxations.setdefault(p, {}).setdefault(today, set()).update(hrs)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 14 — STICKY TASKS                         ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_STICKY > 0 and sticky_tasks:
                            sb_keys = [
                                (p, t, h, today)
                                for t in sticky_tasks
                                for p in people_per_task[t]
                                for h in hours_today[:-1]
                                if (p, t, h, today) in x_keys_set
                                and (p, t, h_next_all[(h, today)], today) in x_keys_set]

                            if sb_keys:
                                # s_break ∈ {0,1}: 1 iff a different person covers the
                                # next consecutive hour of the same sticky task
                                s_break = model.addVars(sb_keys, vtype=GRB.BINARY)
                                for v in s_break.values(): v.BranchPriority = -1
                                obj += W_STICKY * s_break.sum()
                                model.addConstrs(
                                    x[p, t, h, today] - x[p, t, h_next_all[(h, today)], today]
                                    <= s_break[p, t, h, today]
                                    for p, t, h, _ in sb_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 15 — ROTATION (anti-repetition)           ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_ROTATION > 0 and effective_rotation_tasks:
                            rot_keys = [
                                k
                                for t in effective_rotation_tasks
                                for p in people_per_task[t]
                                for k in x_by_pt[(p, t)]
                                if (k[2], today) in h_next_all
                                and (p, t, h_next_all[(k[2], today)], today) in x_keys_set
                            ]

                            if rot_keys:
                                # c_rot ∈ {0,1}: 1 iff same person covers two consecutive
                                # hours of a rotation task (penalised to encourage handover)
                                c_rot = model.addVars(rot_keys, vtype=GRB.BINARY)
                                for v in c_rot.values(): v.BranchPriority = -1
                                obj += W_ROTATION * c_rot.sum()
                                model.addConstrs(
                                    x[p, t, h, today] + x[p, t, h_next_all[(h, today)], today]
                                    - c_rot[p, t, h, today] <= 1
                                    for p, t, h, _ in rot_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 16 — TRAVEL TIME (hard + soft)            ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_TRAVEL > 0 and task_pairs_with_travel:

                            # Hard: forbid assignments too close in time to allow travel
                            for t1, t2, k in task_pairs_with_travel:
                                for p in people_with_x_today:
                                    pts1 = _pt_idx[(p, t1)]
                                    pts2 = _pt_idx[(p, t2)]
                                    if not pts1 or not pts2:
                                        continue
                                    for i1, h1 in pts1:
                                        for i2, h2 in pts2:
                                            if 0 < abs(i2 - i1) <= k:
                                                model.addConstr(
                                                    x[p, t1, h1, today] + x[p, t2, h2, today] <= 1)

                            # Soft: penalise using more than one location in a day
                            locs_with_travel = {
                                location_of_task[t]
                                for t1, t2, _ in task_pairs_with_travel
                                for t in (t1, t2)}

                            if len(locs_with_travel) > 1:
                                # Inverted index: location → tasks at that location
                                tasks_at_loc: defaultdict[str, list] = defaultdict(list)
                                for t in tasks:
                                    loc = location_of_task[t]
                                    if loc in locs_with_travel:
                                        tasks_at_loc[loc].append(t)

                                # People who have x-slots at each location
                                p_locs: defaultdict[str, set] = defaultdict(set)
                                for p, t, h, _ in x_keys:
                                    loc = location_of_task[t]
                                    if loc in locs_with_travel:
                                        p_locs[p].add(loc)

                                ul_keys = [(p, loc) for p, locs in p_locs.items() for loc in locs]
                                if ul_keys:
                                    uses_loc = model.addVars(ul_keys, vtype=GRB.BINARY)
                                    for v in uses_loc.values(): v.BranchPriority = -1

                                    for p, loc in ul_keys:
                                        for t in tasks_at_loc[loc]:
                                            for k_ in x_by_pt[(p, t)]:
                                                model.addConstr(uses_loc[p, loc] >= x[k_])

                                    pair_keys:   list  = []
                                    pair_weight: dict  = {}
                                    for p, locs in p_locs.items():
                                        for l1, l2 in combinations(sorted(locs), 2):
                                            tc = travel_cost.get((l1, l2), 0)
                                            if tc > 0:
                                                pair_keys.append((p, l1, l2))
                                                pair_weight[p, l1, l2] = tc

                                    if pair_keys:
                                        uses_pair = model.addVars(pair_keys, vtype=GRB.BINARY)
                                        for v in uses_pair.values(): v.BranchPriority = -1
                                        model.addConstrs(
                                            uses_pair[p, l1, l2]
                                            >= uses_loc[p, l1] + uses_loc[p, l2] - 1
                                            for p, l1, l2 in pair_keys)
                                        obj += W_TRAVEL * gp.quicksum(
                                            pair_weight[k_] * uses_pair[k_] for k_ in pair_keys)

                        # ══════════════════════════════════════════════════
                        # OBJECTIVE, SOLVER PARAMS & OPTIMISE
                        # ══════════════════════════════════════════════════
                        for pn, pv in solver_params.items():
                            try:
                                model.setParam(pn, pv)
                            except Exception as err:
                                logging.warning("Could not set solver param %s=%r: %s", pn, pv, err)

                        model.setObjective(obj, GRB.MINIMIZE)

                        def _cb(mdl, where, _x=x, _hours_today=hours_today,
                                _people=people, _partial=partial_assignment, _today=today):
                            if ui_update_callback is None or where != GRB.Callback.MIPSOL:
                                return
                            now = time.monotonic()
                            if now - last_ui_update[0] < 0.5:
                                return
                            last_ui_update[0] = now
                            try:
                                xv = mdl.cbGetSolution(_x)
                                ta = {p: {h: None for h in _hours_today} for p in _people}
                                
                                # 1. Recopilamos todas las tareas activas en el diccionario 'ta'
                                for (p, t, h, _), val in xv.items():
                                    if val > 0.5:
                                        ta[p][h] = t
                                        
                                # 2. ¡CORRECCIÓN! Llamamos a la UI FUERA del bucle for
                                # Una sola actualización con el diccionario 'ta' ya completo
                                ui_update_callback({
                                    "status": "Solving (New Best Found)...",
                                    "assignment": {**_partial, _today: ta},
                                })
                                
                            except Exception as exc:
                                logging.warning("Solver callback error: %s", exc)

                        model.optimize(_cb)

                        # ══════════════════════════════════════════════════
                        # ACCUMULATE RESULTS
                        # ══════════════════════════════════════════════════
                        iter_status = status_map.get(model.Status, f"Code: {model.Status}")
                        day_statuses[today] = iter_status

                        day_mip_gaps[today] = (
                            model.MIPGap
                            if model.SolCount > 0 and model.NumIntVars > 0
                            else 0.0)

                        if model.SolCount == 0:
                            break

                        day_x = _collect(model, x)
                        all_x_vals.update(day_x)

                        # Captain shortfalls — w is an empty tupledict ({}) when no slots exist
                        if W_RULE > 0 and w:
                            slots = captain_slots_per_day[today]
                            for i, (rule_idx, t, h, caps, min_req) in enumerate(slots):
                                if w[i].X > 0.5:
                                    marker = " (no captains available)" if not slot_eligibles[i] else ""
                                    captain_violations.append(
                                        f"Rule #{rule_idx + 1} — '{t}' @ {h}, {today} "
                                        f"(need {min_req}, captains: [{', '.join(caps)}]){marker}")

                        for (p, t, h, _), v in day_x.items():
                            partial_assignment[today][p][h] = t
                            accumulated_hours[p]         += 1
                            accumulated_task_hours[p, t] += 1
                            if (p, h) in emergency_today:
                                emerg_issues.append(f"{p} @ {t}, {h}, {today}")

                        for group in group_list:
                            accumulated_group_hours[group] = sum(
                                accumulated_hours[p] for p in group_people[group])

                        processed_days.append(today)
                        interrupted = model.Status == GRB.INTERRUPTED

                        del day_x, x_keys, x_keys_set
                        del x_by_th, x_by_ph, x_by_p, x_by_pt, x_by_p_hidx, _pt_idx

                    finally:
                        if active_model_ref is not None:
                            active_model_ref[0] = None

                # On exiting the with block, Gurobi has already called model.dispose()
                # safely. Help Python's GC by dropping references to heavy variables.
                if 'x'   in locals(): del x
                if 'act' in locals(): del act

                if interrupted:
                    break

                # Force GC to free RAM before the next day's model is built
                gc.collect()

    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}") from e

    # Mark days that were never attempted
    for d in days:
        if d not in day_statuses:
            day_statuses[d] = "Not attempted"

    # Compute summary status from per-day statuses
    attempted_statuses = [s for s in day_statuses.values() if s != "Not attempted"]
    if not attempted_statuses:
        summary_status = "No days solved"
    elif all(s == "Optimal" for s in attempted_statuses):
        summary_status = "Optimal"
    elif len(set(attempted_statuses)) == 1:
        summary_status = attempted_statuses[0]
    else:
        counts = Counter(attempted_statuses)
        parts  = sorted(counts.items(), key=lambda kv: -kv[1])
        summary_status = "Mixed: " + ", ".join(f"{cnt}× {st}" for st, cnt in parts)

    # ══════════════════════════════════════════════════════════════════
    # POST-PROCESSING: label TRAVEL slots
    # ══════════════════════════════════════════════════════════════════
    assignment = partial_assignment

    if W_TRAVEL > 0 and task_pairs_with_travel:
        for d in processed_days:
            hrs = hours[d]
            for p in people:
                slots = assignment[d][p]
                assigned_list = [
                    (i, h, slots[h])
                    for i, h in enumerate(hrs)
                    if slots[h] is not None and slots[h] != TRAVEL_LABEL
                ]
                for a_pos in range(len(assigned_list) - 1):
                    i_a, h_a, t_a = assigned_list[a_pos]
                    i_b, h_b, t_b = assigned_list[a_pos + 1]
                    l_a = location_of_task.get(t_a, "__same__")
                    l_b = location_of_task.get(t_b, "__same__")
                    k = travel_cost.get((l_a, l_b), 0)
                    if k <= 0:
                        continue
                    for gi in range(i_a + 1, min(i_b, i_a + k + 1)):
                        if slots[hrs[gi]] is None:
                            slots[hrs[gi]] = TRAVEL_LABEL

    # forced_*_set_per_day are no longer read after the main loop
    # (force/just_work/just_rest violation reports were removed when those
    # constraints moved to hard with pre-check guarantees).
    del demand_set_per_day, available_set_per_day, emergency_per_day

    consec_relaxations_out: dict[str, dict[str, list]] = {
        p: {d: sorted(hrs) for d, hrs in by_day.items()}
        for p, by_day in consec_relaxations.items()
    }

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    assigned_counts       = {p: 0 for p in people}
    task_hours_per_person = Counter()
    active_slots          = set()

    for (p, t, h, d) in all_x_vals:
        assigned_counts[p]             += 1
        task_hours_per_person[(p, t)] += 1
        active_slots.add((p, t, h, d))

    sol = {
        "solve_time":         time.monotonic() - solve_start,
        "status":             summary_status,
        "day_statuses":       day_statuses,
        "day_mip_gaps":       day_mip_gaps,
        "mip_gap":            max(day_mip_gaps.values()) if day_mip_gaps else 0.0,
        "enforced_rest":      False,  # max_consec_hours is now soft
        "consec_relaxations": consec_relaxations_out,
        "assignment":         assignment,
        "workload":           assigned_counts,
        "workload_max":       float(max(assigned_counts.values())) if assigned_counts else 0.0,
        "workload_min":       float(min(assigned_counts.values())) if assigned_counts else 0.0,
        "task_workload":      {p: {t: task_hours_per_person.get((p, t), 0) for t in tasks} for p in people},
        "travel_label":       TRAVEL_LABEL if (W_TRAVEL > 0 and task_pairs_with_travel) else None,
    }

    # Empty: if we reach this point the pre-check passed and coverage is guaranteed
    sol["coverage_gaps"] = []

    sol["captain_violations"] = captain_violations

    processed_days_set = set(processed_days)
    quota_violations = []
    for r in quota_rules_valid:
        for task, req in r["tasks"].items():
            for day in r["days"]:
                if day not in processed_days_set:
                    continue
                for person in r["people"]:
                    actual = sum(
                        1 for h in hours[day]
                        if assignment.get(day, {}).get(person, {}).get(h) == task)
                    if actual < req:
                        has_travel = any(
                            assignment.get(day, {}).get(person, {}).get(h) == TRAVEL_LABEL
                            for h in hours[day])
                        msg = (f"Rule #{r['idx'] + 1} — {person} → '{task}' "
                               f"on {day}: {actual}/{req}h")
                        if has_travel:
                            msg += " (travel may have contributed)"
                        quota_violations.append(msg)
    sol["quota_violations"] = quota_violations

    slots_per_person = defaultdict(set)
    for p, t, h, d in active_slots:
        slots_per_person[p].add((t, h, d))

    enemy_violations = []
    if has_enemies and not hard_enemies:
        for p1, p2 in social_enemies:
            for t, h, d in slots_per_person.get(p1, set()):
                if (p2, t, h, d) in active_slots:
                    enemy_violations.append(f"{p1} & {p2} @ {t}, {h}, {d}")
    sol["enemy_violations"] = enemy_violations

    sol["emerg_issues"] = emerg_issues

    # ── Reporting targets — always on VISUAL groups ───────────────────
    _vn = len(visual_group_list)

    group_capacity = {
        g: sum(cap[p] for p in visual_group_people[g])
        for g in visual_group_list}

    if groups_work_ratios:
        _ts = sum(float(groups_work_ratios.get(g, 0) or 0) for g in visual_group_list)
        if _ts > 1e-9:
            _vw = {g: float(groups_work_ratios.get(g, 0) or 0) / _ts for g in visual_group_list}
        else:
            _vw = {g: 1.0 / _vn for g in visual_group_list} if _vn else {}
    else:
        _total_cap = sum(group_capacity.values())
        if _total_cap > 1e-9:
            _vw = {g: group_capacity[g] / _total_cap for g in visual_group_list}
        else:
            _vw = {g: 1.0 / _vn for g in visual_group_list} if _vn else {}

    group_targets = {g: total_demand * _vw[g] for g in visual_group_list}

    sol["eq_group_mode"] = "shares" if groups_work_ratios else "off"
    sol["group_workload"] = {}
    for g in visual_group_list:
        total_h   = sum(assigned_counts[p] for p in visual_group_people[g])
        target_h  = group_targets[g]
        diff      = total_h - target_h
        n_members = max(len(visual_group_people[g]), 1)
        sol["group_workload"][g] = {
            "total_hours":       total_h,
            "target_hours":      round(target_h, 2),
            "deviation":         round(diff, 2),
            "per_person_avg":    round(total_h / n_members, 2),
            "per_person_target": round(target_h / n_members, 2),
            "members":           visual_group_people[g],
            "share_pct":         round(_vw[g] * 100, 1),
            "group_capacity":    round(group_capacity[g], 2),
        }

    variety_report = []
    for t in tasks:
        qualified = people_per_task[t]
        if not qualified:
            continue
        touched  = sum(1 for p in qualified if task_hours_per_person.get((p, t), 0) > 0)
        repeated = sum(1 for p in qualified if task_hours_per_person.get((p, t), 0) > 1)
        total_h  = sum(task_hours_per_person.get((p, t), 0) for p in qualified)
        variety_report.append({
            "task": t, "qualified": len(qualified),
            "touched": touched, "repeated": repeated, "total_hours": total_h,
        })
    sol["variety_report"] = variety_report

    gc.collect()
    return sol