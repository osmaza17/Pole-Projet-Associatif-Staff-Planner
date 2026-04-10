import os
import gc
from pathlib import Path
os.environ["GRB_LICENSE_FILE"] = str(Path(__file__).parent / "gurobi.lic")

import time
import logging
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict, Counter


def solve_model(data, ui_update_callback=None, active_model_ref=None):

    # ══════════════════════════════════════════════════════════════════
    # UNPACK
    # ══════════════════════════════════════════════════════════════════
    people       = data["people"];      tasks         = data["tasks"]
    hours        = data["hours"];       days          = data["days"]
    demand       = data["demand"];      availability  = data["availability"]
    emergency    = data["emergency"]
    skills       = data["skills"];      force         = data["force"]
    social       = data["social"]
    pref_cost    = data["pref_cost"];   X_prev        = data["X_prev"]

    max_consec_hours = data.get("max_consec_hours", {})
    hard_enemies     = data.get("hard_enemies", False)
    just_work        = data.get("just_work", {})
    solver_params    = data.get("solver_params", {})
    captain_rules    = data.get("captain_rules", [])
    force_rest       = data.get("force_rest", {})
    quota_rules      = data.get("quota_rules", [])
    sticky           = data.get("sticky", {})
    rotation         = data.get("rotation", {})
    task_duration    = data.get("task_duration", {})
    capacity = data.get("capacity", {}) 
    max_hours_per_day   = data.get("max_hours_per_day", {})    # {person: int} or {}
    max_hours_per_event = data.get("max_hours_per_event", {})  # {person: int} or {}
    task_priority = data.get("task_priority", {})

    if not data.get("live_callbacks", 1):
        ui_update_callback = None

    # ── Objective weights ─────────────────────────────────────────────
    W            = data["weights"]
    W_COVERAGE   = W["W_COVERAGE"];   W_FORCE      = W["W_FORCE"]
    W_EMERG      = W["W_EMERG"]
    W_STABILITY  = W["W_STABILITY"];  W_EQ_GLOBAL  = W["W_EQ_GLOBAL"]
    W_SOCIAL     = W["W_SOCIAL"];     W_GAP        = W["W_GAP"]
    W_QUOTA      = W["W_QUOTA"];      W_PREF       = W["W_PREF"]
    W_EQ_GROUP   = W["W_EQ_GROUP"]
    W_VARIETY    = W["W_VARIETY"];     W_CAPTAIN    = W["W_CAPTAIN"]
    W_STICKY     = W.get("W_STICKY", 0)
    W_DURATION   = W.get("W_DURATION", 0)
    W_ROTATION   = W.get("W_ROTATION", 0)

    # ══════════════════════════════════════════════════════════════════
    # GROUPS
    # ══════════════════════════════════════════════════════════════════
    raw_groups = data.get("groups", {}) or {"default": list(people)}

    group_list       = list(raw_groups.keys())

    number_of_groups = len(group_list)

    group_people     = {g: raw_groups[g] for g in group_list}

    person_to_group  = {p: g for g, members in group_people.items() for p in members}

    cap = {p: capacity.get(p, 1.0) for p in people}

    # ══════════════════════════════════════════════════════════════════
    # STATIC PRECOMPUTATIONS
    # ══════════════════════════════════════════════════════════════════
    people_per_task = defaultdict(list)
    for (p, t), v in skills.items():
        if v == 1:
            people_per_task[t].append(p)

    demand_set_per_day = defaultdict(list)
    for (t, h, d), v in demand.items():
        if v > 0:
            demand_set_per_day[d].append((t, h, d))

    available_set_per_day = defaultdict(set)
    for (p, h, d), v in availability.items():
        if v == 1:
            available_set_per_day[d].add((p, h, d))

    forced_tasks_set_per_day = defaultdict(list)
    for (p, t, h, d), v in force.items():
        if v == 1:
            forced_tasks_set_per_day[d].append((p, t, h, d))

    must_work_set_per_day = defaultdict(list)
    for (p, h, d), v in just_work.items():
        if v == 1:
            must_work_set_per_day[d].append((p, h))

    must_rest_set_per_day = defaultdict(list)
    for (p, h, d), v in force_rest.items():
        if v == 1:
            must_rest_set_per_day[d].append((p, h))

    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)}

    sticky_tasks   = {t for t in tasks if sticky.get(t, 0) == 1}
    rotation_tasks = {t for t in tasks if rotation.get(t, 0) == 1}

    social_friends = [(p1, p2) for (p1, p2), sv in social.items() if sv == 1]
    social_enemies = [(p1, p2) for (p1, p2), sv in social.items() if sv == -1]
    has_enemies    = bool(social_enemies)
    emergency_set  = {(p, h, d) for (p, h, d), v in emergency.items() if v == 1}
    preferences_values = {(p, t): v for (p, t), v in pref_cost.items() if v != 0}

    multi_hour_tasks = {t: task_duration[t] for t in tasks if task_duration.get(t, 1) > 1}

    # ── Effective rotation set: drop tasks that require continuity by
    #    design, so BLOCK 14 cannot fight BLOCK 0b or BLOCK 13.
    effective_rotation_tasks = (
        rotation_tasks - set(multi_hour_tasks) - sticky_tasks
    )

    captain_slots_per_day: dict[str, list[tuple]] = defaultdict(list)
    if captain_rules and W_CAPTAIN > 0:
        for r_idx, rule in enumerate(captain_rules):
            caps    = tuple(rule["captains"])
            min_req = rule.get("min_required", 1)
            for d in days:
                for t in rule["tasks"]:
                    for h in rule.get("hours", {}).get(d, []):
                        if h in hours.get(d, []) and demand.get((t, h, d), 0) > 0:
                            captain_slots_per_day[d].append((r_idx, t, h, caps, min_req))

    # ── Quota rules — preprocessing ──────────────────────────────────
    people_set, tasks_set = set(people), set(tasks)
    quota_rules_valid = []
    for r_idx, rule in enumerate(quota_rules):
        r_people = [p for p in rule.get("people", []) if p in people_set]
        r_tasks  = {t: int(q) for t, q in rule.get("tasks", {}).items()
                    if t in tasks_set and isinstance(q, (int, float)) and int(q) > 0}
        r_days   = [d for d in days if d in set(rule.get("days", []))]
        if r_people and r_tasks and r_days:
            quota_rules_valid.append({"idx": r_idx, "people": r_people,
                                      "tasks": r_tasks, "days": r_days})

    # ── Pacing — cumulative demand targets ───────────────────────────
    demand_per_day    = {d: sum(demand[k] for k in demand_set_per_day[d]) for d in days}
    total_demand      = sum(demand_per_day.values())
    cumulative_demand = {}
    running = 0
    for d in days:
        running += demand_per_day[d]
        cumulative_demand[d] = running

    accumulated_hours       = {p: 0 for p in people}
    accumulated_task_hours  = defaultdict(int)
    accumulated_group_hours = {g: 0 for g in group_list}

    # ══════════════════════════════════════════════════════════════════
    # RESULT ACCUMULATORS
    # ══════════════════════════════════════════════════════════════════
    all_x_vals      = {}
    all_m_vals      = {}
    all_u_vals      = {}
    all_u_any_vals  = {}
    all_u_rest_vals = {}
    all_w_vals      = {}
    consec_relaxations: dict[str, dict] = {}

    partial_assignment = {d: {p: {h: None for h in hours[d]} for p in people} for d in days}

    final_status  = "Optimal"
    final_mip_gap = 0.0
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
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 1)
            env.start()

            for today in days:

                available_set_today    = available_set_per_day[today]
                demand_set_today       = demand_set_per_day[today]
                forced_tasks_set_today = forced_tasks_set_per_day[today]
                must_work_set_today    = must_work_set_per_day[today]
                hours_today            = hours[today]

                with gp.Model("StaffScheduler", env=env) as model:

                    if active_model_ref is not None:
                        if active_model_ref[0] is not None:
                            active_model_ref[0].terminate()
                        active_model_ref[0] = model

                    interrupted = False
                    obj = gp.LinExpr()

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 0 — CORE: x + coverage m                 ║
                    # ╚══════════════════════════════════════════════════╝
                    x_set = {(p, t, h, today)
                        for t, h, _ in demand_set_today
                        for p in people_per_task[t]
                        if (p, h, today) in available_set_today}

                    x = model.addVars(x_set, vtype=GRB.BINARY)
                    m = model.addVars(demand_set_today, lb=0, ub=demand, vtype=GRB.INTEGER)

                    obj += W_COVERAGE * gp.quicksum(
                        task_priority.get(t, 1.0) * m[t, h, today]
                        for t, h, _ in demand_set_today
                    )

                    model.addConstrs(
                        x.sum('*', t, h, today) + m[t, h, today] == demand[(t, h, today)]
                        for t, h, _ in demand_set_today)

                    model.addConstrs(
                        x.sum(p, '*', h, today) <= 1
                        for p, h, _ in available_set_today)

                    people_with_x_today = {p for p, t, h, _ in x_set}

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 0b — MULTI-HOUR TASK DURATION (soft)      ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_DURATION > 0 and multi_hour_tasks:
                        covers = defaultdict(list)   # (p,t,h) → [start_hours]
                        s_set  = set()

                        for t, dur in multi_hour_tasks.items():
                            for p in people_per_task[t]:
                                for si in range(len(hours_today) - dur + 1):
                                    start_h = hours_today[si]
                                    block   = hours_today[si : si + dur]
                                    if all((p, bh, today) in available_set_today for bh in block):
                                        s_set.add((p, t, start_h, today))
                                        for bh in block:
                                            covers[(p, t, bh)].append(start_h)

                        if s_set:
                            s_dur = model.addVars(s_set, vtype=GRB.BINARY)

                            dur_slack_keys = [
                                (p, t, h, today)
                                for p, t, h, _ in x_set
                                if t in multi_hour_tasks]

                            dur_slack = model.addVars(dur_slack_keys, vtype=GRB.BINARY)
                            obj += W_DURATION * dur_slack.sum()

                            for p, t, h, _ in dur_slack_keys:
                                cover_starts = covers.get((p, t, h), [])
                                s_sum = (gp.quicksum(s_dur[p, t, sh, today]
                                                     for sh in cover_starts)
                                         if cover_starts else 0)
                                model.addConstr(dur_slack[p, t, h, today] >= x[p, t, h, today] - s_sum)
                                if cover_starts:
                                    model.addConstr(x[p, t, h, today] >= s_sum)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 1 — FORCE MANDATES                       ║
                    # ╚══════════════════════════════════════════════════╝
                    force_keys  = []
                    u_any_keys  = []
                    u_rest_keys = []

                    if W_FORCE > 0:
                        force_keys = [
                            (p, t, h, today)
                            for p, t, h, _ in forced_tasks_set_today
                            if demand.get((t, h, today), 0) > 0]

                        u = model.addVars(force_keys, vtype=GRB.BINARY)

                        u_any_keys  = [(p, h, today) for p, h in must_work_set_today]
                        u_rest_keys = [(p, h, today) for p, h in must_rest_set_per_day[today]]

                        u_any  = model.addVars(u_any_keys,  vtype=GRB.BINARY)
                        u_rest = model.addVars(u_rest_keys, vtype=GRB.BINARY)

                        obj += W_FORCE * (u.sum() + u_any.sum() + u_rest.sum())

                        model.addConstrs(
                            1 - x[p, t, h, today] <= u[p, t, h, today]
                            for p, t, h, _ in force_keys)

                        model.addConstrs(
                            x.sum(p, '*', h, today) + u_any[p, h, today] >= 1
                            for p, h in must_work_set_today)

                        model.addConstrs(
                            x.sum(p, '*', h, today) <= u_rest[p, h, today]
                            for p, h in must_rest_set_per_day[today])

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 2 — EMERGENCY                            ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EMERG > 0 and emergency_set:
                        obj += W_EMERG * gp.quicksum(
                            x[p, t, h, today]
                            for p, t, h, _ in x_set
                            if (p, h, today) in emergency_set)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 3 — GLOBAL EQUITY                        ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EQ_GLOBAL > 0:
                        cum_today = cumulative_demand[today]
                        person_pace_target = {}
                        for group in group_list:
                            members = group_people[group]
                            group_cap_sum = sum(cap[p] for p in members)
                            group_budget = cum_today / number_of_groups
                            for p in members:
                                person_pace_target[p] = group_budget * cap[p] / max(group_cap_sum, 1e-9)

                        delta_plus  = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)
                        delta_minus = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)
                        obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

                        model.addConstrs(
                            accumulated_hours[p] + x.sum(p, '*', '*', today)
                            - delta_plus[p] + delta_minus[p] == person_pace_target[p]
                            for p in people_with_x_today)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 4 — GROUP EQUITY                         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EQ_GROUP > 0 and number_of_groups > 1:
                        group_target_today = cumulative_demand[today] / number_of_groups

                        gamma_plus  = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                        gamma_minus = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                        obj += W_EQ_GROUP * (gamma_plus.sum() + gamma_minus.sum())

                        x_set_per_group = defaultdict(list)
                        for p, t, h, _ in x_set:
                            x_set_per_group[person_to_group[p]].append((p, t, h, today))

                        model.addConstrs(
                            accumulated_group_hours[group]
                            + gp.quicksum(x[p, t, h, today] for p, t, h, _ in x_set_per_group[group])
                            - gamma_plus[group] + gamma_minus[group] == group_target_today
                            for group in group_list)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 5 — SOCIAL                               ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_SOCIAL > 0 or hard_enemies:
                        active_th_per_person = defaultdict(set)
                        for p, t, h, _ in x_set:
                            active_th_per_person[p].add((t, h))

                        friends_keys = [
                            (p1, p2, t, h, today)
                            for p1, p2 in social_friends
                            for t, h in active_th_per_person[p1]
                            if (t, h) in active_th_per_person[p2]]

                        enemies_scope = [
                            (p1, p2, t, h, today)
                            for p1, p2 in social_enemies
                            for t, h in active_th_per_person[p1]
                            if (t, h) in active_th_per_person[p2]]

                        if W_SOCIAL > 0:
                            enemies_keys = [] if hard_enemies else enemies_scope
                            f = model.addVars(friends_keys, vtype=GRB.BINARY)
                            e = model.addVars(enemies_keys, vtype=GRB.BINARY)
                            obj += W_SOCIAL * (f.sum() + e.sum())

                            model.addConstrs(
                                x[p1, t, h, today] - x[p2, t, h, today] <= f[p1, p2, t, h, today]
                                for p1, p2, t, h, _ in friends_keys)
                            model.addConstrs(
                                x[p2, t, h, today] - x[p1, t, h, today] <= f[p1, p2, t, h, today]
                                for p1, p2, t, h, _ in friends_keys)

                            if not hard_enemies:
                                model.addConstrs(
                                    x[p1, t, h, today] + x[p2, t, h, today]
                                    - e[p1, p2, t, h, today] <= 1
                                    for p1, p2, t, h, _ in enemies_keys)

                        if hard_enemies:
                            model.addConstrs(
                                x[p1, t, h, today] + x[p2, t, h, today] <= 1
                                for p1, p2, t, h, _ in enemies_scope)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 6 — QUOTA RULES                          ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_QUOTA > 0 and quota_rules_valid:
                        for r in quota_rules_valid:
                            if today not in r["days"]:
                                continue
                            for task, required_hours in r["tasks"].items():
                                capped_req = min(required_hours, len(hours_today))
                                for person in r["people"]:
                                    pvars = [x[person, task, h, today]
                                             for h in hours_today
                                             if (person, task, h, today) in x_set]
                                    qs = model.addVar(lb=0, ub=capped_req, vtype=GRB.INTEGER,
                                                      name=f"qs_r{r['idx']}_{person}_{task}_{today}")
                                    model.addConstr(gp.quicksum(pvars) + qs >= capped_req)
                                    obj += W_QUOTA * qs

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 7 — STABILITY                            ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_STABILITY > 0 and X_prev:
                        z = model.addVars(x_set, vtype=GRB.BINARY)
                        obj += W_STABILITY * z.sum()
                        model.addConstrs(
                            z[k] >= X_prev.get(k, 0) - x[k] for k in x_set)
                        model.addConstrs(
                            z[k] >= x[k] - X_prev.get(k, 0) for k in x_set)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 8 — GAPS                                 ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_GAP > 0:
                        g_keys = {(p, h, today) for p, t, h, _ in x_set}
                        r_keys = [(p, today) for p in people_with_x_today]

                        g = model.addVars(g_keys, vtype=GRB.BINARY)
                        rr = model.addVars(r_keys, lb=0, vtype=GRB.INTEGER)
                        obj += W_GAP * rr.sum()

                        model.addConstrs(
                            g[p, hours_today[0], today] == x.sum(p, '*', hours_today[0], today)
                            for p in people_with_x_today
                            if (p, hours_today[0], today) in g_keys)

                        model.addConstrs(
                            g[p, hn, today] >= x.sum(p, '*', hn, today) - x.sum(p, '*', hb, today)
                            for p in people_with_x_today
                            for hb, hn in zip(hours_today[:-1], hours_today[1:])
                            if (p, hn, today) in g_keys)

                        model.addConstrs(
                            rr[p, today] >= gp.quicksum(
                                g[p, h, today] for h in hours_today if (p, h, today) in g_keys) - 1
                            for p, _ in r_keys)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 9 — PREFERENCES                          ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_PREF > 0 and preferences_values:
                        obj += W_PREF * gp.quicksum(
                            preferences_values[(p, t)] * x[p, t, h, today]
                            for p, t, h, _ in x_set
                            if (p, t) in preferences_values)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 10 — TASK VARIETY                        ║
                    # ╚══════════════════════════════════════════════════╝
                    quota_required_today = defaultdict(int)
                    for r in quota_rules_valid:
                        if today in r["days"]:
                            for task, req in r["tasks"].items():
                                for person in r["people"]:
                                    quota_required_today[person, task] = max(
                                        quota_required_today[person, task], req)

                    if W_VARIETY > 0:
                        v_max_keys = [
                            (p, t) for p in people for t in tasks
                            if skills.get((p, t), 0) == 1
                            and any((p, t, h, today) in x_set for h in hours_today)]

                        v_max = model.addVars(v_max_keys, lb=0, vtype=GRB.INTEGER)
                        obj += W_VARIETY * v_max.sum()

                        for p, t in v_max_keys:
                            base_free   = 1 if accumulated_task_hours[p, t] == 0 else 0
                            quota_free  = quota_required_today.get((p, t), 0)
                            sticky_free = len(hours_today) if t in sticky_tasks else 0
                            dur_free    = multi_hour_tasks.get(t, 0)
                            free_pass   = max(base_free, quota_free, sticky_free, dur_free)
                            model.addConstr(v_max[p, t] >= x.sum(p, t, '*', today) - free_pass)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 11 — CAPTAIN RULES                       ║
                    # ╚══════════════════════════════════════════════════╝
                    w_keys = []
                    sole_captain_hours_per_person: dict[str, frozenset] = {}

                    if W_CAPTAIN > 0 and captain_slots_per_day[today]:
                        slots = captain_slots_per_day[today]

                        def eligible(task, hour, caps):
                            return [c for c in caps if (c, task, hour, today) in x_set]

                        w = model.addVars(range(len(slots)), lb=0, vtype=GRB.INTEGER)
                        w_keys = list(range(len(slots)))
                        obj += W_CAPTAIN * w.sum()

                        model.addConstrs(
                            gp.quicksum(x[c, t, h, today] for c in eligible(t, h, caps))
                            + w[i] >= min_req
                            for i, (_, t, h, caps, min_req) in enumerate(slots))

                        sole: dict[str, set] = defaultdict(set)
                        for _, t, h, caps, _ in slots:
                            e = eligible(t, h, caps)
                            if len(e) == 1:
                                sole[e[0]].add(h)
                        sole_captain_hours_per_person = {p: frozenset(hrs) for p, hrs in sole.items()}
                        for p, hrs in sole_captain_hours_per_person.items():
                            consec_relaxations.setdefault(p, {}).setdefault(today, []).extend(sorted(hrs))

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 12 — MAX CONSECUTIVE HOURS               ║
                    # ╚══════════════════════════════════════════════════╝
                    if max_consec_hours:
                        for person, consec_config in max_consec_hours.items():
                            if isinstance(consec_config, tuple):
                                max_work, min_rest = consec_config
                            else:
                                max_work, min_rest = int(consec_config), 1
                            if person not in people or max_work <= 0 or min_rest <= 0:
                                continue
                            window = max_work + min_rest
                            if len(hours_today) < window:
                                continue
                            if not any((person, h, today) in available_set_today for h in hours_today):
                                continue

                            sole_hrs = sole_captain_hours_per_person.get(person, frozenset())

                            if sole_hrs:
                                pfx = [0] * (len(hours_today) + 1)
                                for i, h in enumerate(hours_today):
                                    pfx[i + 1] = pfx[i] + (1 if h in sole_hrs else 0)
                                model.addConstrs(
                                    x.sum(person, '*', hours_today[s:s + window], today) <= max_work
                                    for s in range(len(hours_today) - window + 1)
                                    if pfx[s + window] - pfx[s] == 0)
                            else:
                                model.addConstrs(
                                    x.sum(person, '*', hours_today[s:s + window], today) <= max_work
                                    for s in range(len(hours_today) - window + 1))
                                
                    # ╔══════════════════════════════════════════════════════════════╗
                    # ║  BLOCK 12b — MAX HOURS PER DAY / PER EVENT (hard)            ║
                    # ╚══════════════════════════════════════════════════════════════╝
                    for person, max_h in max_hours_per_day.items():
                        if person in people_with_x_today and max_h > 0:
                            model.addConstr(
                                x.sum(person, '*', '*', today) <= max_h)

                    for person, max_h in max_hours_per_event.items():
                        if person in people_with_x_today and max_h > 0:
                            model.addConstr(
                                accumulated_hours[person] + x.sum(person, '*', '*', today) <= max_h)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 13 — STICKY TASKS                        ║
                    # ╚══════════════════════════════════════════════════╝
                    if sticky_tasks and W_STICKY > 0:
                        sb_keys = [
                            (p, t, h, today)
                            for t in sticky_tasks
                            for p in people_per_task[t]
                            for h in hours_today[:-1]
                            if (p, t, h, today) in x_set
                            and (p, t, h_next_all[(h, today)], today) in x_set]

                        s_break = model.addVars(sb_keys, vtype=GRB.BINARY)
                        obj += W_STICKY * s_break.sum()
                        model.addConstrs(
                            x[p, t, h, today] - x[p, t, h_next_all[(h, today)], today]
                            <= s_break[p, t, h, today]
                            for p, t, h, _ in sb_keys)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 14 — ROTATION (anti-repetition)           ║
                    # ╚══════════════════════════════════════════════════╝
                    # Penalize each consecutive-hour pair on the same task
                    # for the same person, to promote rotation and avoid
                    # boredom. Multi-hour and sticky tasks are excluded
                    # upstream (effective_rotation_tasks) so this block
                    # never fights BLOCK 0b or BLOCK 13.
                    if W_ROTATION > 0 and effective_rotation_tasks:
                        rot_keys = [
                            (p, t, h, today)
                            for t in effective_rotation_tasks
                            for p in people_per_task[t]
                            for h in hours_today[:-1]
                            if (p, t, h, today)                       in x_set
                            and (p, t, h_next_all[(h, today)], today) in x_set]

                        if rot_keys:
                            c_rot = model.addVars(rot_keys, vtype=GRB.BINARY)
                            obj += W_ROTATION * c_rot.sum()
                            model.addConstrs(
                                x[p, t, h, today]
                                + x[p, t, h_next_all[(h, today)], today]
                                - c_rot[p, t, h, today] <= 1
                                for p, t, h, _ in rot_keys)

                    # ══════════════════════════════════════════════════
                    # OBJECTIVE & PARAMS & OPTIMIZE
                    # ══════════════════════════════════════════════════
                    model.setObjective(obj, GRB.MINIMIZE)
                    model._x = x

                    for pn, pv in solver_params.items():
                        try:
                            model.setParam(pn, pv)
                        except Exception as err:
                            print(f"Warning: Could not set param {pn}={pv}: {err}")

                    if ui_update_callback is None:
                        model.optimize()
                    else:
                        def _cb(mdl, where):
                            if where != GRB.Callback.MIPSOL:
                                return
                            now = time.monotonic()
                            if now - last_ui_update[0] < 0.005:
                                return
                            last_ui_update[0] = now
                            try:
                                xv = mdl.cbGetSolution(mdl._x)
                                ta = {p: {h: None for h in hours_today} for p in people}
                                for (p, t, h, d), val in xv.items():
                                    if val > 0.5:
                                        ta[p][h] = t
                                ui_update_callback({"status": "Solving (New Best Found)...",
                                                    "assignment": {**partial_assignment, today: ta}})
                            except Exception as exc:
                                logging.warning(f"Solver callback error: {exc}")
                        model.optimize(_cb)

                    # ══════════════════════════════════════════════════
                    # ACCUMULATE RESULTS
                    # ══════════════════════════════════════════════════
                    iter_status = status_map.get(model.Status, f"Code: {model.Status}")

                    if model.SolCount == 0:
                        if all_x_vals:
                            final_status = iter_status
                            break
                        raise Exception("No feasible solution found.")

                    if iter_status != "Optimal":
                        final_status = iter_status
                    final_mip_gap = model.MIPGap if model.NumIntVars > 0 else 0.0

                    all_x_vals.update(model.getAttr('X', x))
                    all_m_vals.update(model.getAttr('X', m))
                    if W_FORCE > 0:
                        if force_keys:  all_u_vals.update(model.getAttr('X', u))
                        if u_any_keys:  all_u_any_vals.update(model.getAttr('X', u_any))
                        if u_rest_keys: all_u_rest_vals.update(model.getAttr('X', u_rest))
                    if W_CAPTAIN > 0 and w_keys:
                        all_w_vals.update(model.getAttr('X', w))

                    partial_assignment[today] = {p: {h: None for h in hours_today} for p in people}
                    for p, t, h, _ in x_set:
                        if x[p, t, h, today].X > 0.5:
                            partial_assignment[today][p][h] = t
                            accumulated_hours[p]         += 1
                            accumulated_task_hours[p, t] += 1

                    for group in group_list:
                        accumulated_group_hours[group] = sum(
                            accumulated_hours[p] for p in group_people[group])

                    interrupted = model.Status == GRB.INTERRUPTED
                    if hasattr(model, '_x'):
                        del model._x
                    if active_model_ref is not None:
                        active_model_ref[0] = None

                if interrupted:
                    break

    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}") from e

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════
    assignment = partial_assignment

    # ── Counters ─────────────────────────────────────────────────────
    assigned_counts       = {p: 0 for p in people}
    task_hours_per_person = Counter()
    active_slots          = set()

    for (p, t, h, d), v in all_x_vals.items():
        if v > 0.5:
            assigned_counts[p]            += 1
            task_hours_per_person[(p, t)] += 1
            active_slots.add((p, t, h, d))

    # ── Build sol dict ───────────────────────────────────────────────
    sol = {
        "solve_time":         time.monotonic() - solve_start,
        "status":             final_status,
        "mip_gap":            final_mip_gap,
        "enforced_rest":      bool(max_consec_hours),
        "consec_relaxations": consec_relaxations,
        "assignment":         assignment,
        "workload":           assigned_counts,
        "workload_max":       float(max(assigned_counts.values())) if assigned_counts else 0.0,
        "workload_min":       float(min(assigned_counts.values())) if assigned_counts else 0.0,
        "task_workload":      {p: {t: task_hours_per_person.get((p, t), 0) for t in tasks} for p in people},
    }

    # ── Missing staff ────────────────────────────────────────────────
    sol["missing"] = [f"{t} @ {h}, {d}: {v:.0f} missing"
                      for (t, h, d), v in all_m_vals.items() if v > 0.01]

    # ── Force mandates — only violations ─────────────────────────────
    force_violations = []
    if W_FORCE > 0 and all_u_vals:
        for d in days:
            for p, t, h, _ in forced_tasks_set_per_day[d]:
                if all_u_vals.get((p, t, h, d), 0) > 0.5:
                    force_violations.append(f"{p} — '{t}' @ {h}, {d}")
    sol["force_violations"] = force_violations

    # ── Just work — only violations ──────────────────────────────────
    jw_violations = []
    if all_u_any_vals:
        for d in days:
            for p, h in must_work_set_per_day[d]:
                if all_u_any_vals.get((p, h, d), 0) > 0.5:
                    jw_violations.append(f"{p} @ {h}, {d}")
    sol["just_work_violations"] = jw_violations

    # ── Just rest — only violations ──────────────────────────────────
    jr_violations = []
    if all_u_rest_vals:
        for d in days:
            for p, h in must_rest_set_per_day[d]:
                if all_u_rest_vals.get((p, h, d), 0) > 0.5:
                    assigned_task = assignment[d].get(p, {}).get(h, "?")
                    jr_violations.append(f"{p} @ {h}, {d} → '{assigned_task}'")
    sol["just_rest_violations"] = jr_violations

    # ── Captain — only violations ────────────────────────────────────
    captain_violations = []
    if W_CAPTAIN > 0 and captain_rules and all_w_vals:
        for d in days:
            for i, (r_idx, t, h, caps, min_req) in enumerate(captain_slots_per_day[d]):
                if all_w_vals.get(i, 0) > 0.5:
                    captain_violations.append(
                        f"Rule #{r_idx+1} — '{t}' @ {h}, {d} "
                        f"(need {min_req}, captains: [{', '.join(caps)}])")
    sol["captain_violations"] = captain_violations

    # ── Quota — only violations ──────────────────────────────────────
    quota_violations = []
    for r in quota_rules_valid:
        for task, req in r["tasks"].items():
            for day in r["days"]:
                for person in r["people"]:
                    actual = sum(1 for h in hours[day]
                                 if assignment.get(day, {}).get(person, {}).get(h) == task)
                    if actual < req:
                        quota_violations.append(
                            f"Rule #{r['idx']+1} — {person} → '{task}' on {day}: {actual}/{req}h")
    sol["quota_violations"] = quota_violations

    # ── Social — only enemy violations ───────────────────────────────
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

    # ── Emergency call-ins ───────────────────────────────────────────
    sol["emerg_issues"] = [
        f"{p} @ {t}, {h}, {d}"
        for p, t, h, d in active_slots
        if (p, h, d) in emergency_set]

    # ── Group workload ───────────────────────────────────────────────
    group_target_total = total_demand / number_of_groups
    sol["group_workload"] = {}
    for g in group_list:
        total_h    = sum(assigned_counts[p] for p in group_people[g])
        diff       = total_h - group_target_total
        n_members  = max(len(group_people[g]), 1)
        sol["group_workload"][g] = {
            "total_hours":       total_h,
            "target_hours":      round(group_target_total, 2),
            "deviation":         round(diff, 2),
            "per_person_avg":    round(total_h / n_members, 2),
            "per_person_target": round(group_target_total / n_members, 2),
            "members":           group_people[g],
        }

    # ── Variety report ───────────────────────────────────────────────
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