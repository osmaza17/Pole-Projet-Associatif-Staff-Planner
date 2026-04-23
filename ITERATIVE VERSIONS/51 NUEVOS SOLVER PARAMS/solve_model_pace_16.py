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


# FIX #1: Guard against empty var_dict — model.getAttr("X", {}) raises GurobiError
def _collect(model, var_dict, threshold=0.5):
    """Extract active values from a Gurobi VarDict, filtering by threshold."""
    if not var_dict:
        return {}
    return {k: v for k, v in model.getAttr("X", var_dict).items()
            if v > threshold}


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
    task_priority       = data.get("task_priority", {})
    task_location       = data.get("task_location", {})
    travel_time         = data.get("travel_time", {})

    # ── Objective weights ─────────────────────────────────────────────

    WEIGHTS = data.get("weights")

    W_COVERAGE   = WEIGHTS.get("W_COVERAGE")
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
    all_x_vals      = {}
    all_m_vals      = {}
    all_u_vals      = {}
    all_u_any_vals  = {}
    all_u_rest_vals = {}

    consec_relaxations: dict[str, dict[str, set]] = {}
    captain_violations: list[str] = []
    emerg_issues:       list[str] = []
    processed_days:     list[str] = []

    partial_assignment: dict[str, dict[str, dict[str, str | None]]] = {}

    # ── Per-day status tracking ───────────────────────────────────────
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

                # Pre-init this day's partial assignment for the UI
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
                        # ║  BLOCK 0 — CORE: x + coverage m                  ║
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

                        # m: missing coverage — demand is integer, x is binary
                        m = model.addVars(
                            demand_set_today,
                            lb=0,
                            ub={k: demand[k] for k in demand_set_today},
                            vtype=GRB.INTEGER,
                        )

                        obj += W_COVERAGE * gp.quicksum(
                            task_priority.get(t, 1.0) * m[t, h, today]
                            for t, h, _ in demand_set_today
                        )

                        model.addConstrs(
                            gp.quicksum(x[k] for k in x_by_th[(t, h)]) + m[t, h, today] == demand[(t, h, today)]
                            for t, h, _ in demand_set_today
                        )

                        model.addConstrs(
                            gp.quicksum(x[k] for k in x_by_ph[(p, h)]) <= 1
                            for p, h in x_by_ph
                        )

                        people_with_x_today = set(x_by_p)

                        # ── Shared index for travel blocks ────────────────────────────
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
                                obj += W_DURATION * dur_slack.sum()

                                for p, t, h, _ in dur_slack_keys:
                                    cover_starts = covers.get((p, t, h), [])
                                    s_sum = gp.quicksum(s_dur[p, t, sh, today] for sh in cover_starts)
                                    model.addConstr(dur_slack[p, t, h, today] >= x[p, t, h, today] - s_sum)
                                    if cover_starts:
                                        model.addConstr(x[p, t, h, today] >= s_sum)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 1 — FORCE MANDATES                        ║
                        # ╚══════════════════════════════════════════════════╝
                        force_keys  = []
                        u_any_keys  = []
                        u_rest_keys = []
                        u      = {}
                        u_any  = {}
                        u_rest = {}

                        if W_RULE > 0:
                            force_keys = [
                                (p, t, h, today)
                                for p, t, h, _ in forced_tasks_set_today
                                if (t, h, today) in demand
                                and (p, t, h, today) in x_keys_set
                            ]

                            u_any_keys  = [(p, h, today) for p, h in forced_work_set_today]
                            u_rest_keys = [(p, h, today) for p, h in forced_rest_set_today]

                            # u      ∈ {0,1}: violation indicator per forced-task slot
                            # u_any  ∈ {0,1}: 1 iff person not working at a must-work slot
                            # u_rest ∈ {0,1}: 1 iff person working at a must-rest slot
                            u      = model.addVars(force_keys,  vtype=GRB.BINARY)
                            u_any  = model.addVars(u_any_keys,  vtype=GRB.BINARY)
                            u_rest = model.addVars(u_rest_keys, vtype=GRB.BINARY)

                            obj += W_RULE * (u.sum() + u_any.sum() + u_rest.sum())

                            model.addConstrs(
                                x[p, t, h, today] + u[p, t, h, today] >= 1
                                for p, t, h, _ in force_keys
                            )

                            model.addConstrs(
                                gp.quicksum(x[k] for k in x_by_ph[(p, h)]) + u_any[p, h, today] >= 1
                                for p, h in forced_work_set_today
                            )

                            model.addConstrs(
                                gp.quicksum(x[k] for k in x_by_ph[(p, h)]) <= u_rest[p, h, today]
                                for p, h in forced_rest_set_today
                            )

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 2 — EMERGENCY                             ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_EMERG > 0 and emergency_today:
                            obj += W_EMERG * gp.quicksum(
                                x[p, t, h, today]
                                for p, t, h, _ in x_keys
                                if (p, h) in emergency_today
                            )

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 3 — INTRA-GROUP EQUITY                    ║
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

                                # delta_plus/minus are real-valued deviations (can exceed 1)
                                delta_plus  = model.addVars(eq_members, lb=0, vtype=GRB.CONTINUOUS)
                                delta_minus = model.addVars(eq_members, lb=0, vtype=GRB.CONTINUOUS)
                                obj += W_INTRAGROUP * (delta_plus.sum() + delta_minus.sum())

                                acc_sum = sum(accumulated_hours[p] for p in all_members)
                                group_hours_expr = acc_sum + gp.quicksum(
                                    x[k] for p in active_members for k in x_by_p[p])

                                model.addConstrs(
                                    accumulated_hours[p]
                                    + gp.quicksum(x[k] for k in x_by_p[p])
                                    - (cap[p] / g_cap) * group_hours_expr
                                    == delta_plus[p] - delta_minus[p]
                                    for p in eq_members)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 4 — INTER-GROUP EQUITY                    ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_INTERGROUP > 0 and number_of_groups > 1 and groups_work_ratios:
                            # gamma_plus/minus are real-valued group deviations → CONTINUOUS
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
                        # ║  BLOCK 5 — SOCIAL                                ║
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
                                # f ∈ {0,1}: |x[p1] - x[p2]| indicator (diff of binaries → binary)
                                f = model.addVars(friends_keys, vtype=GRB.BINARY)
                                obj += W_SOCIAL * f.sum()
                                model.addConstrs(
                                    x[p1, t, h, today] - x[p2, t, h, today] <= f[p1, p2, t, h, today]
                                    for p1, p2, t, h, _ in friends_keys)
                                model.addConstrs(
                                    x[p2, t, h, today] - x[p1, t, h, today] <= f[p1, p2, t, h, today]
                                    for p1, p2, t, h, _ in friends_keys)

                                if not hard_enemies:
                                    enemies_keys = _shared_slots(social_enemies)
                                    # e ∈ {0,1}: x[p1] + x[p2] > 1 indicator → BINARY
                                    e = model.addVars(enemies_keys, vtype=GRB.BINARY)
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
                        # ║  BLOCK 6 — QUOTA RULES                           ║
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
                        # ║  BLOCK 7 — STABILITY                             ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_STABILITY > 0 and X_prev:
                            # z ∈ {0,1}: absolute difference between two binary schedules
                            z = model.addVars(x_keys, vtype=GRB.BINARY)
                            obj += W_STABILITY * z.sum()
                            model.addConstrs(z[k] >= X_prev.get(k, 0) - x[k] for k in x_keys)
                            model.addConstrs(z[k] >= x[k] - X_prev.get(k, 0) for k in x_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 7b — TRANSIT DETECTION (before gaps)      ║
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
                                for p, ta, ha, tb, hb, hm in _tlinks:
                                    model.addConstr(
                                        in_transit_vars[p, hm, today]
                                        >= x[p, ta, ha, today] + x[p, tb, hb, today] - 1)

                            del _tlinks

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 8 — GAPS (transit-aware)                  ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_GAP > 0:
                            g_keys = {(p, h, today) for p, t, h, _ in x_keys}

                            # g ∈ {0,1}: 1 iff person "restarts" a work block at hour h
                            g  = model.addVars(g_keys, vtype=GRB.BINARY)
                            # rr: number of gaps (block-restarts − 1) — g is BINARY so
                            # Σg is integer → rr is always integer at optimality → INTEGER.
                            rr = model.addVars(people_with_x_today, lb=0, vtype=GRB.INTEGER)
                            obj += W_GAP * rr.sum()

                            if in_transit_keys:
                                active_keys  = g_keys | in_transit_keys
                                only_work    = g_keys - in_transit_keys
                                only_transit = in_transit_keys - g_keys
                                both_keys    = g_keys & in_transit_keys

                                # act ∈ {0,1}: 1 iff person is active (working or in transit)
                                act = model.addVars(active_keys, vtype=GRB.BINARY)

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
                        # ║  BLOCK 9 — PREFERENCES                           ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_PREF > 0 and preferences_values:
                            obj += W_PREF * gp.quicksum(
                                preferences_values[pt] * x[k]
                                for pt in preferences_values
                                if pt in x_by_pt
                                for k in x_by_pt[pt]
                            )

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 10 — TASK VARIETY                         ║
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
                        # ║  BLOCK 11 — CAPTAIN RULES                        ║
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
                        # ║  BLOCK 12 — MAX CONSECUTIVE HOURS                ║
                        # ╚══════════════════════════════════════════════════╝
                        if max_consec_hours:
                            for person, consec_config in max_consec_hours.items():
                                max_work, min_rest = (
                                    consec_config if isinstance(consec_config, tuple)
                                    else (int(consec_config), 1))
                                if person not in people or max_work <= 0 or min_rest <= 0:
                                    continue

                                window = max_work + min_rest
                                n_hrs  = len(hours_today)
                                if n_hrs < window:
                                    continue
                                if person not in people_with_x_today:
                                    continue

                                _pidx    = x_by_p_hidx[person]
                                sole_hrs = sole_captain_hours_per_person.get(person, frozenset())
                                n_wins   = n_hrs - window + 1

                                if sole_hrs:
                                    pfx = [0] * (n_hrs + 1)
                                    for i, h in enumerate(hours_today):
                                        pfx[i + 1] = pfx[i] + (1 if h in sole_hrs else 0)
                                    model.addConstrs(
                                        gp.quicksum(x[k] for idx, k in _pidx if s <= idx < s + window)
                                        <= max_work
                                        for s in range(n_wins)
                                        if pfx[s + window] - pfx[s] == 0)
                                else:
                                    model.addConstrs(
                                        gp.quicksum(x[k] for idx, k in _pidx if s <= idx < s + window)
                                        <= max_work
                                        for s in range(n_wins))

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 12b — MAX HOURS PER DAY / PER EVENT       ║
                        # ╚══════════════════════════════════════════════════╝
                        for person, max_h in max_hours_per_day.items():
                            if person in people_with_x_today and max_h > 0:
                                model.addConstr(gp.quicksum(x[k] for k in x_by_p[person]) <= max_h)

                        for person, max_h in max_hours_per_event.items():
                            if person in people_with_x_today and max_h > 0:
                                model.addConstr(
                                    accumulated_hours[person]
                                    + gp.quicksum(x[k] for k in x_by_p[person]) <= max_h)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 13 — STICKY TASKS                         ║
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
                                # s_break ∈ {0,1}: 1 iff a different person covers the next
                                # consecutive hour of the same sticky task
                                s_break = model.addVars(sb_keys, vtype=GRB.BINARY)
                                obj += W_STICKY * s_break.sum()
                                model.addConstrs(
                                    x[p, t, h, today] - x[p, t, h_next_all[(h, today)], today]
                                    <= s_break[p, t, h, today]
                                    for p, t, h, _ in sb_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 14 — ROTATION (anti-repetition)           ║
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
                                # c_rot ∈ {0,1}: 1 iff same person covers two consecutive hours
                                # of a rotation task (penalised to encourage handover)
                                c_rot = model.addVars(rot_keys, vtype=GRB.BINARY)
                                obj += W_ROTATION * c_rot.sum()
                                model.addConstrs(
                                    x[p, t, h, today] + x[p, t, h_next_all[(h, today)], today]
                                    - c_rot[p, t, h, today] <= 1
                                    for p, t, h, _ in rot_keys)

                        # ╔══════════════════════════════════════════════════╗
                        # ║  BLOCK 15 — TRAVEL TIME (hard + soft)            ║
                        # ╚══════════════════════════════════════════════════╝
                        if W_TRAVEL > 0 and task_pairs_with_travel:

                            # ── Hard: prohíbe asignaciones demasiado cercanas ─────────────
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

                            # ── Soft: penaliza usar más de una localización ───────────────
                            locs_with_travel = {
                                location_of_task[t]
                                for t1, t2, _ in task_pairs_with_travel
                                for t in (t1, t2)}

                            if len(locs_with_travel) > 1:
                                # Índice inverso: localización → tareas en esa loc con viaje
                                tasks_at_loc: defaultdict[str, list] = defaultdict(list)
                                for t in tasks:
                                    loc = location_of_task[t]
                                    if loc in locs_with_travel:
                                        tasks_at_loc[loc].append(t)

                                # Personas que tienen x-slots en cada localización
                                p_locs: defaultdict[str, set] = defaultdict(set)
                                for p, t, h, _ in x_keys:
                                    loc = location_of_task[t]
                                    if loc in locs_with_travel:
                                        p_locs[p].add(loc)

                                ul_keys = [(p, loc) for p, locs in p_locs.items() for loc in locs]
                                if ul_keys:
                                    uses_loc = model.addVars(ul_keys, vtype=GRB.BINARY)

                                    # uses_loc[p, loc] >= x[p, t, h, d]  ∀ t en esa loc
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
                                        model.addConstrs(
                                            uses_pair[p, l1, l2]
                                            >= uses_loc[p, l1] + uses_loc[p, l2] - 1
                                            for p, l1, l2 in pair_keys)
                                        obj += W_TRAVEL * gp.quicksum(
                                            pair_weight[k_] * uses_pair[k_] for k_ in pair_keys)

                        # ══════════════════════════════════════════════════
                        # OBJECTIVE & PARAMS & OPTIMIZE
                        # ══════════════════════════════════════════════════
                        for pn, pv in solver_params.items():
                            try:
                                model.setParam(pn, pv)
                            except Exception as err:
                                logging.warning("Could not set solver param %s=%r: %s", pn, pv, err)

                        model.setObjective(obj, GRB.MINIMIZE)


                        def _cb( mdl,where, _x=x, _hours_today=hours_today,
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
                                for (p, t, h, _), val in xv.items():
                                    if val > 0.5:
                                        ta[p][h] = t
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
                            else 0.0
                        )

                        if model.SolCount == 0:
                            break

                        # ── Variables de decisión principales ────────────────────────────
                        day_x = _collect(model, x)
                        all_x_vals.update(day_x)
                        all_m_vals.update(_collect(model, m, threshold=0.01))

                        # ── Violaciones de reglas de turno ────────────────────────────────
                        if W_RULE > 0:
                            if force_keys:
                                all_u_vals.update(_collect(model, u))
                            if u_any_keys:
                                all_u_any_vals.update(_collect(model, u_any))
                            if u_rest_keys:
                                all_u_rest_vals.update(_collect(model, u_rest))

                            # Captain shortfalls — w es un tupledict vacío ({}) si no hay slots
                            if w:
                                slots = captain_slots_per_day[today]
                                for i, (rule_idx, t, h, caps, min_req) in enumerate(slots):
                                    if w[i].X > 0.5:
                                        marker = " (no captains available)" if not slot_eligibles[i] else ""
                                        captain_violations.append(
                                            f"Rule #{rule_idx + 1} — '{t}' @ {h}, {today} "
                                            f"(need {min_req}, captains: [{', '.join(caps)}]){marker}")

                        # ── Acumulación de contadores y asignación parcial ───────────────
                        for (p, t, h, _), v in day_x.items():
                            # day_x ya está filtrado a v > 0.5
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

                        # FIX #11: Release per-day index structures that are no longer
                        # needed once results are accumulated, consistent with the pattern
                        # used for skills/force/availability/etc. above.
                        del day_x, x_keys, x_keys_set
                        del x_by_th, x_by_ph, x_by_p, x_by_pt, x_by_p_hidx, _pt_idx

                    finally:
                        if active_model_ref is not None:
                            active_model_ref[0] = None

                # ── AL SALIR DEL BLOQUE WITH ──
                # Aquí Gurobi ya ha ejecutado `model.dispose()` de forma segura 
                # automáticamente gracias al 'with'.

                # Ahora ayudamos a Python eliminando las referencias a las variables pesadas
                # para que el recolector de basura pueda liberar la RAM por completo.
                if 'x' in locals(): del x
                if 'm' in locals(): del m
                if 'u' in locals(): del u
                if 'act' in locals(): del act

                if interrupted:
                    break
                
                # Forzamos al Garbage Collector a limpiar la RAM antes del siguiente día
                gc.collect()

    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}") from e

    # ── Mark days that were never attempted ───────────────────────────
    for d in days:
        if d not in day_statuses:
            day_statuses[d] = "Not attempted"

    # ── Compute summary status from per-day statuses ──────────────────
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

    # FIX #11: Release large intermediate structures that are no longer needed
    # after the main loop, consistent with the pre-loop del pattern.
    # NOTE: forced_*_set_per_day are intentionally kept — they are read again
    # below when building the force/just_work/just_rest violation reports.
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

    # all_x_vals only contains active entries (v > 0.5)
    for (p, t, h, d) in all_x_vals:
        assigned_counts[p]            += 1
        task_hours_per_person[(p, t)] += 1
        active_slots.add((p, t, h, d))

    sol = {
        "solve_time":         time.monotonic() - solve_start,
        "status":             summary_status,
        "day_statuses":       day_statuses,
        "day_mip_gaps":       day_mip_gaps,
        "mip_gap":            max(day_mip_gaps.values()) if day_mip_gaps else 0.0,
        "enforced_rest":      bool(max_consec_hours),
        "consec_relaxations": consec_relaxations_out,
        "assignment":         assignment,
        "workload":           assigned_counts,
        "workload_max":       float(max(assigned_counts.values())) if assigned_counts else 0.0,
        "workload_min":       float(min(assigned_counts.values())) if assigned_counts else 0.0,
        "task_workload":      {p: {t: task_hours_per_person.get((p, t), 0) for t in tasks} for p in people},
        "travel_label":       TRAVEL_LABEL if (W_TRAVEL > 0 and task_pairs_with_travel) else None,
    }

    # all_m_vals only contains entries with deficit > 0.01
    sol["missing"] = [f"{t} @ {h}, {d}: {v:.0f} missing"
                      for (t, h, d), v in all_m_vals.items()]

    force_violations = []
    if W_RULE > 0 and all_u_vals:
        for d in days:
            for p, t, h, _ in forced_tasks_set_per_day.get(d, []):
                if all_u_vals.get((p, t, h, d), 0) > 0.5:
                    force_violations.append(f"{p} — '{t}' @ {h}, {d}")
    sol["force_violations"] = force_violations

    jw_violations = []
    if all_u_any_vals:
        for d in days:
            for p, h in forced_work_set_per_day.get(d, []):
                if all_u_any_vals.get((p, h, d), 0) > 0.5:
                    jw_violations.append(f"{p} @ {h}, {d}")
    sol["just_work_violations"] = jw_violations

    jr_violations = []
    if all_u_rest_vals:
        for d in days:
            for p, h in forced_rest_set_per_day.get(d, []):
                if all_u_rest_vals.get((p, h, d), 0) > 0.5:
                    assigned_task = assignment[d].get(p, {}).get(h, "?")
                    jr_violations.append(f"{p} @ {h}, {d} → '{assigned_task}'")
    sol["just_rest_violations"] = jr_violations

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