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
    # UNPACK — extract all data from the input dict
    # ══════════════════════════════════════════════════════════════════
    people    = data["people"];     tasks        = data["tasks"]
    hours     = data["hours"];      days         = data["days"]
    demand    = data["demand"];     availability = data["availability"]
    emergency = data["emergency"]
    skills    = data["skills"];     force        = data["force"]
    social    = data["social"];     min_quota    = data["min_quota"]
    rotation  = data["rotation"];   pref_cost    = data["pref_cost"]
    X_prev    = data["X_prev"]

    max_consec_hours = data.get("max_consec_hours", {}) # dict["Persona"", tuple[max_h_trabajo, min_h_descanso]]
    hard_enemies     = data.get("hard_enemies", False)
    just_work        = data.get("just_work", {})
    solver_params    = data.get("solver_params", {})
    captain_rules    = data.get("captain_rules", [])

    # ── Disable callbacks in silent mode ──────────────────────────────
    if not data.get("live_callbacks", 1):
        ui_update_callback = None

    # ── Objective weights ─────────────────────────────────────────────
    W           = data["weights"]
    W_COVERAGE  = W["W_COVERAGE"];  W_FORCE     = W["W_FORCE"]
    W_EMERG     = W["W_EMERG"]
    W_STABILITY = W["W_STABILITY"]; W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_GLOBAL = W["W_EQ_GLOBAL"]; W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];     W_PREF      = W["W_PREF"]
    W_EQ_GROUP  = W["W_EQ_GROUP"]
    W_VARIETY   = W["W_VARIETY"]
    W_CAPTAIN   = W["W_CAPTAIN"]

    # ══════════════════════════════════════════════════════════════════
    # GROUPS — group people; if no groups, create a default one
    # ══════════════════════════════════════════════════════════════════
    raw_groups = data.get("groups", {})
    if not raw_groups:
        raw_groups = {"default": list(people)}

    group_list       = list(raw_groups.keys())
    number_of_groups = len(group_list)
    group_people     = {g: raw_groups[g] for g in group_list}
    person_to_group  = {p: g for g, members in group_people.items() for p in members}

    # ══════════════════════════════════════════════════════════════════
    # STATIC PRECOMPUTATIONS — derived structures used in the loop
    # ══════════════════════════════════════════════════════════════════

    # Qualified people per task
    people_per_task = defaultdict(list)
    for (p, t), v in skills.items():
        if v == 1:
            people_per_task[t].append(p)

    # Slots with positive demand, indexed by day
    demand_set_per_day = defaultdict(list)
    for (t, h, d), v in demand.items():
        if v > 0:
            demand_set_per_day[d].append((t, h, d))

    # Availability slots indexed by day
    available_set_per_day = defaultdict(set)
    for (p, h, d), v in availability.items():
        if v == 1:
            available_set_per_day[d].add((p, h, d))

    # Force mandates indexed by day (only active ones, demand filter applied in loop)
    forced_tasks_set_per_day = defaultdict(list)
    for (p, t, h, d), v in force.items():
        if v == 1:
            forced_tasks_set_per_day[d].append((p, t, h, d))

    # Work Any mandates (must work on something) indexed by day
    must_work_set_per_day = defaultdict(list)
    for (p, h, d), v in just_work.items():
        if v == 1:
            must_work_set_per_day[d].append((p, h))

    # Minimum quotas indexed by day
    quota_set_per_day = defaultdict(list)
    for (p, t, d), v in min_quota.items():
        if v > 0:
            quota_set_per_day[d].append((p, t, d))

    # Next hours in all cases
    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)}

    # Rotation tasks, social relationships, and emergency sets
    rotation_tasks   = {t for t in tasks if rotation.get(t, 0) == 1}
    social_friends   = [(p1, p2) for (p1, p2), sv in social.items() if sv == 1]
    social_enemies   = [(p1, p2) for (p1, p2), sv in social.items() if sv == -1]
    has_enemies      = bool(social_enemies)
    emergency_set    = {(p, h, d) for (p, h, d), v in emergency.items() if v == 1}
    preferences_values = {(p, t): v for (p, t), v in pref_cost.items() if v != 0}

    # Captain rules: expand to (rule_idx, task, hour, day, captains)
    captain_scope_per_day = defaultdict(list)
    if captain_rules and W_CAPTAIN > 0:
        for r_idx, rule in enumerate(captain_rules):
            rule_captains = tuple(rule["captains"])  # tuple for hashability safety
            for d in days:
                for t in rule["tasks"]:
                    for h in rule.get("hours", {}).get(d, []):
                        if h in hours.get(d, []) and demand.get((t, h, d), 0) > 0:
                            captain_scope_per_day[d].append(
                                (r_idx, t, h, rule_captains))

    # ══════════════════════════════════════════════════════════════════
    # PACING — assignment pace per group and per person
    # ══════════════════════════════════════════════════════════════════
    # PACING — cumulative demand and per-group / per-person targets
    # ══════════════════════════════════════════════════════════════════
    demand_per_day = {d: sum(demand[k] for k in demand_set_per_day[d]) for d in days}
    total_demand   = sum(demand_per_day.values())

    # Cumulative demand up to and including each day (in processing order)
    cumulative_demand = {}
    running_demand    = 0
    for d in days:
        running_demand       += demand_per_day[d]
        cumulative_demand[d]  = running_demand

    # Accumulators for assigned hours (updated day by day)
    accumulated_hours      = {p: 0 for p in people}
    accumulated_task_hours = defaultdict(int)
    accumulated_group_hours = {g: 0 for g in group_list}

    # ══════════════════════════════════════════════════════════════════
    # RESULT ACCUMULATORS
    # ══════════════════════════════════════════════════════════════════
    all_x_vals = {}
    all_m_vals = {}
    all_u_vals = {}
    all_u_any_vals = {}
    all_w_vals = {}
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
    # MAIN LOOP — one model per day
    # ══════════════════════════════════════════════════════════════════
    try:
        # ── Initialize Gurobi environment using 'with' ────────────────────
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 1)
            env.start()

            for loop_idx, today in enumerate(days):

                available_set_today    = available_set_per_day[today]
                demand_set_today       = demand_set_per_day[today]
                forced_tasks_set_today = forced_tasks_set_per_day[today]
                must_work_set_today    = must_work_set_per_day[today]
                people_available_today = {p for p, h, _ in available_set_today}
                hours_today            = hours[today]

                # ── Initialize Gurobi model using 'with' ──────────────────
                with gp.Model("StaffScheduler", env=env) as model:

                    # Register active model to allow external interruption
                    if active_model_ref is not None:
                        if active_model_ref[0] is not None:
                            active_model_ref[0].terminate()
                        active_model_ref[0] = model

                    interrupted = False

                    # ══════════════════════════════════════════════════════
                    # DECISION VARIABLES
                    # ══════════════════════════════════════════════════════

                    # x[p,t,h,today] = 1 if person p performs task t at hour h on day 'today'
                    x_set = {(p, t, h, today)
                        for t, h, _ in demand_set_today
                        for p in people_per_task[t]
                        if (p, h, today) in available_set_today}

                    x = model.addVars(x_set, vtype=GRB.BINARY)

                    # m[t,h,today] = uncovered demand (coverage slack)
                    m = model.addVars(demand_set_today, lb=0, ub=demand, vtype=GRB.CONTINUOUS)

                    # u[p,t,h,today] = penalty for violating force mandate
                    force_keys = [
                        (p, t, h, today)
                        for p, t, h, _ in forced_tasks_set_today
                        if demand.get((t, h, today), 0) > 0]

                    force_possible, force_impossible = [], []
                    for key in force_keys:
                        (force_possible if key in x_set else force_impossible).append(key)

                    u = model.addVars(force_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    # Filter possible vs impossible depending on if there is any x for that person+hour
                    must_work_possible = [
                        (p, h, today) for p, h in must_work_set_today
                        if (p, h, today) in available_set_today
                        and any((p, t, h, today) in x_set for t in tasks)]

                    must_work_impossible = [
                        (p, h, today) for p, h in must_work_set_today
                        if (p, h, today) not in available_set_today
                        or not any((p, t, h, today) in x_set for t in tasks)]

                    u_any_keys = [(p, h, today) for p, h in must_work_set_today]
                    u_any = model.addVars(u_any_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    # z[p,t,h,today] = difference with respect to previous planning (stability)
                    if W_STABILITY > 0 and X_prev:
                        z = model.addVars(x_set, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    # q[p,t,today] = minimum quota deficit
                    quota_keys = quota_set_per_day[today]
                    q = model.addVars(quota_keys, lb=0, ub=min_quota, vtype=GRB.CONTINUOUS)

                    # g[p,h,today] = block start indicator; r[p,today] = number of extra blocks
                    g_keys           = set()
                    people_with_gaps = set()

                    for p, t, h, _ in x_set:
                        g_keys.add((p, h, today))
                        people_with_gaps.add(p)

                    r_keys = [(p, today) for p in people_with_gaps]

                    g = model.addVars(g_keys, lb=0, vtype=GRB.CONTINUOUS)
                    r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS)

                    # j_max_g, j_min_g = max and min hours within group g today
                    # Only groups with >= 2 active people (with x vars) matter
                    people_with_x_today = set()
                    for p, t, h, _ in x_set:
                        people_with_x_today.add(p)

                    groups_with_active = {}  # {group: [active_people]}
                    for grp in group_list:
                        active = [p for p in group_people[grp] if p in people_with_x_today]
                        if len(active) >= 2:
                            groups_with_active[grp] = active

                    j_max_g = model.addVars(groups_with_active.keys(), lb=0,
                                            ub=len(hours_today), vtype=GRB.CONTINUOUS)
                    j_min_g = model.addVars(groups_with_active.keys(), lb=0,
                                            ub=len(hours_today), vtype=GRB.CONTINUOUS)

                    # delta = deviation from intra-group global pace per person
                    # Target: cumulative_demand / |G| / |P_g| (accumulated)
                    cum_demand_today = cumulative_demand[today]
                    person_pace_target = {}
                    for grp in group_list:
                        t_val = cum_demand_today / number_of_groups / max(len(group_people[grp]), 1)
                        for p in group_people[grp]:
                            person_pace_target[p] = t_val

                    delta_plus  = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)
                    delta_minus = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)

                    # gamma = deviation from inter-group pace (total hours per group)
                    # Target: cumulative_demand / |G|
                    if number_of_groups > 1:
                        group_target_total = cum_demand_today / number_of_groups
                        gamma_plus  = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                        gamma_minus = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)

                    # c[p,t,h,today] = consecutive hours indicator in rotation task
                    consec_keys = [
                        (p, t, h, today)
                        for t in rotation_tasks
                        for p in people_per_task[t]
                        for h in hours_today[:-1]
                        if (p, t, h, today)                        in x_set
                        and (p, t, h_next_all[(h, today)], today)  in x_set]

                    c = model.addVars(consec_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    # f = proximity of friends; e = penalty for enemies (soft)
                    active_tasks_hours_per_person = defaultdict(set)
                    active_people_tasks_today     = set()
                    candidates_per_task           = defaultdict(list)

                    for p, t, h, _ in x_set:
                        active_tasks_hours_per_person[p].add((t, h))
                        if (p, t) not in active_people_tasks_today:
                            active_people_tasks_today.add((p, t))
                            candidates_per_task[t].append(p)

                    # Enemy x task x hour pairs relevant for the current day
                    friends_keys = [
                        (p1, p2, t, h, today)
                        for p1, p2 in social_friends
                        for t, h in active_tasks_hours_per_person[p1]
                        if (t, h) in active_tasks_hours_per_person[p2]]   # ✅ 2-tupla

                    enemies_scope = [
                        (p1, p2, t, h, today)
                        for p1, p2 in social_enemies
                        for t, h in active_tasks_hours_per_person[p1]
                        if (t, h) in active_tasks_hours_per_person[p2]]   # ✅ 2-tupla

                    f = model.addVars(friends_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    enemies_keys = [] if hard_enemies else enemies_scope
                    e = model.addVars(enemies_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                    tasks_with_multiple_candidates = [
                        t for t, candidates in candidates_per_task.items()
                        if len(candidates) >= 2]

                    v_max = model.addVars(tasks_with_multiple_candidates, lb=0, vtype=GRB.CONTINUOUS)
                    v_min = model.addVars(tasks_with_multiple_candidates, lb=0, vtype=GRB.CONTINUOUS)

                    # w[r_idx, t, h, today] = penalty if captain rule r_idx is violated
                    captain_scope_today = captain_scope_per_day[today]
                    captain_keys = []
                    captain_possible, captain_impossible = [], []
                    sole_captain_hours_per_person: dict[str, frozenset] = {}

                    if captain_scope_today:
                        captain_keys = list(dict.fromkeys(
                            (rule_idx, task, hour, today)
                            for rule_idx, task, hour, rule_caps
                            in captain_scope_today))

                        for rule_idx, task, hour, rule_caps in captain_scope_today:
                            eligible = tuple(
                                cap for cap in rule_caps
                                if (cap, task, hour, today) in x_set)
                            (captain_possible if eligible else captain_impossible).append(
                                (rule_idx, task, hour, today, eligible))

                        w = model.addVars(captain_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                        # ══════════════════════════════════════════════════
                        # CAPTAIN-AWARE CONSECUTIVE RELAXATION
                        # If a captain is the ONLY eligible person at a
                        # given hour, forcing them to rest in that window
                        # would violate coverage. Identify those critical
                        # hours and exclude windows that contain them.
                        # ══════════════════════════════════════════════════
                        sole_captain_hours_per_person = {}
                        if max_consec_hours and W_CAPTAIN > 0 and captain_possible:
                            sole_hours_accumulator: dict[str, set] = defaultdict(set)
                            for rule_idx, cap_task, cap_hour, cap_day, eligible_caps in captain_possible:
                                if len(eligible_caps) == 1:
                                    sole_hours_accumulator[eligible_caps[0]].add(cap_hour)
                            sole_captain_hours_per_person = {
                                person_name: frozenset(hours_set)
                                for person_name, hours_set in sole_hours_accumulator.items()}

                            # Record relaxations for the final report
                            for relaxed_person, relaxed_hours in sole_captain_hours_per_person.items():
                                consec_relaxations.setdefault(
                                    relaxed_person, {}).setdefault(
                                    today, []).extend(sorted(relaxed_hours))

                    # ══════════════════════════════════════════════════════
                    # OBJECTIVE FUNCTION — minimize weighted penalties
                    # ══════════════════════════════════════════════════════
                    obj = gp.LinExpr()

                    obj += W_COVERAGE  * m.sum()
                    obj += W_FORCE     * (u.sum() + u_any.sum())

                    obj += W_EMERG     * gp.quicksum(
                        x[p, t, h, today]
                        for p, t, h, _ in x_set
                        if (p, h, today) in emergency_set)

                    obj += W_EQ_DAY    * gp.quicksum(
                        j_max_g[grp] - j_min_g[grp] for grp in groups_with_active)
                    obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

                    if number_of_groups > 1:
                        obj += W_EQ_GROUP * (gamma_plus.sum() + gamma_minus.sum())

                    obj += W_ROTATION  * c.sum()
                    obj += W_SOCIAL    * (f.sum() + e.sum())
                    obj += W_GAP       * r.sum()
                    obj += W_QUOTA     * q.sum()

                    obj += W_PREF      * gp.quicksum(
                        preferences_values[(p, t)] * x[p, t, h, today]
                        for p, t, h, _ in x_set
                        if (p, t) in preferences_values)

                    obj += W_VARIETY   * gp.quicksum(
                        v_max[t] - v_min[t] for t in tasks_with_multiple_candidates)

                    if W_CAPTAIN > 0 and captain_keys:
                        obj += W_CAPTAIN * w.sum()

                    if W_STABILITY > 0 and X_prev:
                        obj += W_STABILITY * z.sum()

                    model.setObjective(obj, GRB.MINIMIZE)

                    # ══════════════════════════════════════════════════════
                    # CONSTRAINTS
                    # ══════════════════════════════════════════════════════

                    # — Demand coverage: assigned + slack = demand —
                    model.addConstrs(
                        x.sum('*', t, h, today) + m[t, h, today] == demand[(t, h, today)]
                        for t, h, _ in demand_set_today)

                    # — One task per person per hour —
                    model.addConstrs(
                        x.sum(p, '*', h, today) <= 1
                        for p, h, _ in available_set_today)

                    # — Force mandates: if not met, penalize with u —
                    model.addConstrs(
                        1 - x[p, t, h, today] <= u[p, t, h, today]
                        for p, t, h, _ in force_possible)

                    # Impossible mandates (no x variable): maximum penalty
                    model.addConstrs(
                        u[p, t, h, today] >= 1
                        for p, t, h, _ in force_impossible)

                    # — Just Work: force to work on something (task free for the solver) —
                    model.addConstrs(
                        x.sum(p, '*', h, today) + u_any[p, h, today] >= 1
                        for p, h, _ in must_work_possible)

                    model.addConstrs(
                        u_any[p, h, today] >= 1
                        for p, h, _ in must_work_impossible)

                    # — Intra-day equity (per group): min range of hours within each group —
                    for grp, active_people in groups_with_active.items():
                        model.addConstrs(
                            x.sum(p, '*', '*', today) <= j_max_g[grp]
                            for p in active_people)
                        model.addConstrs(
                            x.sum(p, '*', '*', today) >= j_min_g[grp]
                            for p in active_people)

                    # — Intra-group global equity: follow accumulated pace per person —
                    model.addConstrs(
                        accumulated_hours[p] + x.sum(p, '*', '*', today)
                        - delta_plus[p] + delta_minus[p]
                        == person_pace_target[p]
                        for p in people_with_x_today)

                    # — Inter-group equity: balance total hours between groups —
                    if number_of_groups > 1:
                        x_set_per_group = defaultdict(list)
                        for p, t, h, _ in x_set:
                            x_set_per_group[person_to_group[p]].append((p, t, h, today))

                        model.addConstrs(
                            accumulated_group_hours[grp]
                            + gp.quicksum(x[p, t, h, today] for p, t, h, _ in x_set_per_group[grp])
                            - gamma_plus[grp] + gamma_minus[grp] == group_target_total
                            for grp in group_list)

                    # — Rotation: avoid consecutive hours in rotation tasks —
                    model.addConstrs(
                        x[p, t, h, today] + x[p, t, h_next_all[(h, today)], today] - c[p, t, h, today] <= 1
                        for p, t, h, _ in consec_keys)

                    # — Social friends: penalize working on different tasks —
                    model.addConstrs(
                        x[p1, t, h, today] - x[p2, t, h, today] <= f[p1, p2, t, h, today]
                        for p1, p2, t, h, _ in friends_keys)

                    model.addConstrs(
                        x[p2, t, h, today] - x[p1, t, h, today] <= f[p1, p2, t, h, today]
                        for p1, p2, t, h, _ in friends_keys)

                    # — Social enemies (soft): penalize coincidence —
                    if not hard_enemies:
                        model.addConstrs(
                            x[p1, t, h, today] + x[p2, t, h, today] - e[p1, p2, t, h, today] <= 1
                            for p1, p2, t, h, _ in enemies_keys)

                    # — Social enemies (hard): prohibit coincidence —
                    if hard_enemies:
                        model.addConstrs(
                            x[p1, t, h, today] + x[p2, t, h, today] <= 1
                            for p1, p2, t, h, _ in enemies_scope)

                    # — Minimum quota: each person must cover at least X hours per task —
                    model.addConstrs(
                        x.sum(p, t, '*', today) + q[p, t, today] >= min(min_quota[(p, t, today)], len(hours_today))
                        for p, t, _ in quota_keys)

                    # — Stability: penalize changes with respect to previous planning —
                    if W_STABILITY > 0 and X_prev:
                        model.addConstrs(
                            z[p, t, h, today] >= X_prev.get((p, t, h, today), 0) - x[p, t, h, today]
                            for p, t, h, _ in x_set)

                        model.addConstrs(
                            z[p, t, h, today] >= x[p, t, h, today] - X_prev.get((p, t, h, today), 0)
                            for p, t, h, _ in x_set)

                    # — Gaps: detect block starts and penalize multiple blocks —
                    gap_first_keys = [p for p in people_with_gaps
                                      if (p, hours_today[0], today) in g_keys]

                    model.addConstrs(
                        g[p, hours_today[0], today] == x.sum(p, '*', hours_today[0], today)
                        for p in gap_first_keys)

                    gap_rest_keys = [
                        (p, hour_now, hour_before)
                        for p in people_with_gaps
                        for hour_before, hour_now in zip(hours_today[:-1], hours_today[1:])
                        if (p, hour_now, today) in g_keys]

                    model.addConstrs(
                        g[p, hour_now, today] >= x.sum(p, '*', hour_now, today) - x.sum(p, '*', hour_before, today)
                        for p, hour_now, hour_before in gap_rest_keys)

                    model.addConstrs(
                        r[p, today] >= gp.quicksum(
                            g[p, h, today] for h in hours_today if (p, h, today) in g_keys) - 1
                        for p, _ in r_keys)

                    # — Task variety (proportional): equalize task_hours / target_total_hours —
                    # v_max[t] * target_p >= accum + today_hours  (captures max ratio)
                    # v_min[t] * target_p <= accum + today_hours  (captures min ratio)
                    for t in tasks_with_multiple_candidates:
                        model.addConstrs(
                            v_max[t] * person_pace_target[p]
                            >= accumulated_task_hours[p, t] + x.sum(p, t, '*', today)
                            for p in candidates_per_task[t]
                            if person_pace_target.get(p, 0) > 0)

                        model.addConstrs(
                            v_min[t] * person_pace_target[p]
                            <= accumulated_task_hours[p, t] + x.sum(p, t, '*', today)
                            for p in candidates_per_task[t]
                            if person_pace_target.get(p, 0) > 0)

                    # — Captains: at least one captain per task×hour if rule exists —
                    if W_CAPTAIN > 0 and captain_scope_today:
                        model.addConstrs(
                            gp.quicksum(
                                x[cap, cap_task, cap_hour, today]
                                for cap in eligible_caps)
                            + w[rule_idx, cap_task, cap_hour, today] >= 1
                            for rule_idx, cap_task, cap_hour, cap_day, eligible_caps
                            in captain_possible)

                        model.addConstrs(
                            w[rule_idx, cap_task, cap_hour, today] >= 1
                            for rule_idx, cap_task, cap_hour, cap_day, no_eligible
                            in captain_impossible)

                    # — Maximum consecutive hours per person ———————————————
                    # For each person with a consec limit, slide a window of
                    # size (max_work_hours + min_rest_hours) across the day.
                    # Within each window, at most max_work_hours can be worked.
                    # Windows overlapping sole-captain hours are EXCLUDED
                    # (relaxed) to avoid making the model infeasible when the
                    # captain is the only one who can cover a slot.
                    # —————————————————————————————————————————————————————————
                    for person, consec_config in max_consec_hours.items():
                        if isinstance(consec_config, tuple):
                            max_work_hours, min_rest_hours = consec_config
                        else:
                            max_work_hours, min_rest_hours = int(consec_config), 1

                        if person not in people:
                            continue
                        if max_work_hours <= 0 or min_rest_hours <= 0:
                            logging.warning(
                                "Skipping consec constraint for %s: "
                                "max_work=%d, min_rest=%d",
                                person, max_work_hours, min_rest_hours)
                            continue

                        window_size = max_work_hours + min_rest_hours
                        if len(hours_today) < window_size:
                            continue
                        if not any((person, hour, today) in available_set_today
                                   for hour in hours_today):
                            continue

                        sole_captain_hours = sole_captain_hours_per_person.get(
                            person, frozenset())

                        valid_start_positions = range(
                            len(hours_today) - window_size + 1)

                        if sole_captain_hours:
                            # Build prefix sum to quickly check whether a
                            # sliding window contains any sole-captain hour.
                            # sole_captain_prefix[k] = count of sole hours
                            # in hours_today[0:k].
                            sole_captain_prefix = [0] * (len(hours_today) + 1)
                            for hour_idx, hour in enumerate(hours_today):
                                sole_captain_prefix[hour_idx + 1] = (
                                    sole_captain_prefix[hour_idx]
                                    + (1 if hour in sole_captain_hours else 0))

                            # Only constrain windows that do NOT overlap
                            # with sole-captain hours (those are relaxed).
                            model.addConstrs(
                                x.sum(person, '*',
                                      hours_today[start:start + window_size],
                                      today)
                                <= max_work_hours
                                for start in valid_start_positions
                                if (sole_captain_prefix[start + window_size]
                                    - sole_captain_prefix[start] == 0))
                        else:
                            model.addConstrs(
                                x.sum(person, '*',
                                      hours_today[start:start + window_size],
                                      today)
                                <= max_work_hours
                                for start in valid_start_positions)

                    # ══════════════════════════════════════════════════════
                    # WARM START — initial greedy solution to speed up solver
                    # ══════════════════════════════════════════════════════
                    for key in x_set:
                        x[key].Start = 0

                    busy  = set()
                    cover = defaultdict(int)

                    for t, h, _ in demand_set_today:
                        needed = demand[(t, h, today)]
                        for p in people_per_task[t]:
                            if cover[(t, h)] >= needed:
                                break
                            if (p, t, h, today) in x_set and (p, h) not in busy:
                                x[p, t, h, today].Start = 1
                                busy.add((p, h))
                                cover[(t, h)] += 1

                    # ── Apply solver parameters ───────────────────────────
                    model._x = x
                    for param_name, param_value in solver_params.items():
                        try:
                            model.setParam(param_name, param_value)
                        except Exception as err:
                            print(f"Warning: Could not set param {param_name}={param_value}: {err}")

                    # ── Solve with or without UI callback ─────────────────
                    if ui_update_callback is not None:

                        def intermediate_solution_callback(mdl, where):
                            if where != GRB.Callback.MIPSOL:
                                return
                            now = time.monotonic()
                            if now - last_ui_update[0] < 0.5:
                                return
                            last_ui_update[0] = now
                            try:
                                x_vals = mdl.cbGetSolution(mdl._x)
                                today_assignment = {p: {h: None for h in hours_today} for p in people}
                                for (p, t, h, d), val in x_vals.items():
                                    if val > 0.5:
                                        today_assignment[p][h] = t
                                temp_assignment = {**partial_assignment, today: today_assignment}
                                ui_update_callback({"status": "Solving (New Best Found)...", "assignment": temp_assignment})
                            except Exception as exc:
                                logging.warning(f"Solver callback error (non-fatal): {exc}")

                        model.optimize(intermediate_solution_callback)

                    else:
                        model.optimize()

                    # ── Evaluate status and accumulate results ────────────
                    iter_status = status_map.get(model.Status, f"Status Code: {model.Status}")

                    if model.SolCount == 0:
                        if all_x_vals:
                            final_status = iter_status
                            break
                        else:
                            raise Exception("No feasible solution was found before stopping/timeout.")

                    if iter_status != "Optimal":
                        final_status = iter_status

                    final_mip_gap = model.MIPGap if model.NumIntVars > 0 else 0.0

                    all_x_vals.update(model.getAttr('X', x))
                    all_m_vals.update(model.getAttr('X', m))
                    if force_keys:
                        all_u_vals.update(model.getAttr('X', u))
                    if u_any_keys:
                        all_u_any_vals.update(model.getAttr('X', u_any))
                    if W_CAPTAIN > 0 and captain_keys:
                        all_w_vals.update(model.getAttr('X', w))
                    

                    # Update partial assignment and hours accumulators
                    partial_assignment[today] = {p: {h: None for h in hours_today} for p in people}

                    for p, t, h, _ in x_set:
                        if x[p, t, h, today].X > 0.5:
                            partial_assignment[today][p][h] = t
                            accumulated_hours[p]             += 1
                            accumulated_task_hours[p, t]     += 1

                    # Update group accumulators from person accumulators
                    for grp in group_list:
                        accumulated_group_hours[grp] = sum(
                            accumulated_hours[p] for p in group_people[grp])

                    interrupted = model.Status == GRB.INTERRUPTED

                    # ══════════════════════════════════════════════════════
                    # CRITICAL CLEANUP — Break the circular reference
                    # ══════════════════════════════════════════════════════
                    if hasattr(model, '_x'):
                        del model._x
                        
                    if active_model_ref is not None:
                        active_model_ref[0] = None
                        
                # End of "with gp.Model()". Memory is freed safely here.
                
                if interrupted:
                    break
                    
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi error: {e}") from e

    # ══════════════════════════════════════════════════════════════════
    # RESULTS EXTRACTION
    # ══════════════════════════════════════════════════════════════════
    solve_elapsed = time.monotonic() - solve_start
    assignment    = partial_assignment

    sol = {
        "solve_time":    solve_elapsed,
        "status":        final_status,
        "mip_gap":       final_mip_gap,
        "enforced_rest": bool(max_consec_hours),
        "consec_relaxations": consec_relaxations,
        "assignment":         assignment,
    }

    # — Uncovered demand —
    sol["missing"] = [
        f"{t} @ {h}, {d}: {v:.0f} missing"
        for (t, h, d), v in all_m_vals.items() if v > 0.01]

    # — Classic force mandates report —
    if W_FORCE > 0 and all_u_vals:
        force_issues = []
        for d in days:
            for p, t, h, _ in forced_tasks_set_per_day[d]:
                u_val = all_u_vals.get((p, t, h, d))
                if u_val is None:
                    continue
                if u_val > 0.5:
                    reason = "skill/availability mismatch" if (p, t, h, d) not in all_x_vals else "not assigned"
                    force_issues.append(f"UNFULFILLED: {p} — task '{t}' @ {h}, {d} ({reason})")
                else:
                    force_issues.append(f"FULFILLED: {p} — task '{t}' @ {h}, {d}")
        sol["force_issues"] = force_issues or ["No force mandates defined."]
    else:
        sol["force_issues"] = ["No force mandates defined."]

    # — Captain rules report —
    if W_CAPTAIN > 0 and captain_rules and all_w_vals:
        captain_issues = []
        for d in days:
            for r_idx, t, h, rule_captains in captain_scope_per_day[d]:
                w_val = all_w_vals.get((r_idx, t, h, d))
                if w_val is None:
                    continue
                if w_val > 0.5:
                    captain_issues.append(
                        f"VIOLATION: Rule #{r_idx+1} — task '{t}' @ {h}, {d} "
                        f"(no captain from [{', '.join(rule_captains)}])")
                else:
                    assigned_caps = [cap for cap in rule_captains
                                     if all_x_vals.get((cap, t, h, d), 0) > 0.5]
                    captain_issues.append(
                        f"OK: Rule #{r_idx+1} — task '{t}' @ {h}, {d} "
                        f"(captain: {', '.join(assigned_caps)})")
        sol["captain_issues"] = captain_issues or ["All captain rules satisfied."]
    else:
        sol["captain_issues"] = ["No captain rules defined."]

    # — Single pass over all_x_vals for all counters —
    assigned_counts       = {p: 0 for p in people}
    assigned_per_ptd      = Counter()
    task_hours_per_person = Counter()
    active_slots          = set()

    for (p, t, h, d), v in all_x_vals.items():
        if v > 0.5:
            assigned_counts[p]            += 1
            assigned_per_ptd[(p, t, d)]   += 1
            task_hours_per_person[(p, t)] += 1
            active_slots.add((p, t, h, d))

    # — Workload per person —
    sol["workload"]     = assigned_counts
    workload            = list(assigned_counts.values())
    sol["workload_max"] = float(max(workload)) if workload else 0.0
    sol["workload_min"] = float(min(workload)) if workload else 0.0

    # — Group workload and deviation from target —
    group_target_total = total_demand / number_of_groups
    sol["group_workload"] = {}
    group_eq_issues = []
    for g in group_list:
        total_h      = sum(assigned_counts[p] for p in group_people[g])
        diff         = total_h - group_target_total
        per_person_h = total_h / max(len(group_people[g]), 1)
        target_pp    = group_target_total / max(len(group_people[g]), 1)
        sol["group_workload"][g] = {
            "total_hours":       total_h,
            "target_hours":      round(group_target_total, 2),
            "deviation":         round(diff, 2),
            "per_person_avg":    round(per_person_h, 2),
            "per_person_target": round(target_pp, 2),
            "members":           group_people[g]}
        status_str = "OK" if abs(diff) < 0.5 else ("OVER" if diff > 0 else "SHORT")
        group_eq_issues.append(
            f"{status_str}: Group '{g}' — {total_h}h assigned "
            f"(target {group_target_total:.1f}h, deviation {diff:+.1f}h, "
            f"{per_person_h:.1f}h/person vs target {target_pp:.1f}h/person)")
    sol["group_eq_issues"] = group_eq_issues

    # — Minimum quotas report —
    if W_QUOTA > 0:
        sol["quota_issues"] = []
        for d in days:
            for p, t, _ in quota_set_per_day[d]:
                demanded = min_quota[p, t, d]
                assigned = assigned_per_ptd.get((p, t, d), 0)
                status   = "OK" if assigned >= demanded else "SHORTFALL"
                sol["quota_issues"].append(
                    f"{status}: {p} — task '{t}' on {d}: {assigned}/{demanded} h assigned")
    else:
        sol["quota_issues"] = []

    # — Split shifts (gaps) —
    gaps = []
    for d in days:
        day_gaps = []
        for p in people:
            starts, start_hours, prev_working = 0, [], False
            for h in hours[d]:
                working = assignment[d][p][h] is not None
                if working and not prev_working:
                    starts += 1
                    start_hours.append(h)
                prev_working = working
            if starts > 1:
                day_gaps.append(f"  • {p}: {starts} blocks (Starts: {', '.join(start_hours)})")
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # — Social report (friends and enemies) —
    slots_per_person = defaultdict(set)
    for p, t, h, d in active_slots:
        slots_per_person[p].add((t, h, d))

    soc_issues = []
    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for t, h, d in slots_per_person.get(p1, set()):
            if (p2, t, h, d) in active_slots:
                soc_issues.append(f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    if has_enemies:
        violations = (
            [] if hard_enemies else [
                f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                for p1, p2 in social_enemies
                for t, h, d in slots_per_person.get(p1, set())
                if (p2, t, h, d) in active_slots])
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
    sol["social_issues"] = soc_issues

    # — Emergency calls —
    sol["emerg_issues"] = [
        f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {d}"
        for p, t, h, d in active_slots
        if (p, h, d) in emergency_set]

    # — Rotation violations —
    rot_violations = [
        f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next_all[(h, d)]}, {d}"
        for p, t, h, d in active_slots
        if t in rotation_tasks
        and (h, d) in h_next_all
        and (p, t, h_next_all[(h, d)], d) in active_slots]
    sol["rotation_issues"] = rot_violations or ["SUCCESS: No consecutive hours on rotation tasks."]

    # — Hours per person per task —
    sol["task_workload"] = {
        p: {t: task_hours_per_person.get((p, t), 0) for t in tasks}
        for p in people}

    # — Task variety report (proportional) —
    variety_report = []
    for t in tasks:
        qualified = people_per_task[t]
        if not qualified:
            continue
        total_t = sum(task_hours_per_person.get((p, t), 0) for p in qualified)
        # Show proportional: task_hours / total_hours for each person
        proportions = {}
        for p in qualified:
            hrs_t = task_hours_per_person.get((p, t), 0)
            hrs_total = assigned_counts.get(p, 0)
            proportions[p] = hrs_t / hrs_total if hrs_total > 0 else 0.0
        if proportions:
            p_max = max(proportions.values())
            p_min = min(proportions.values())
            t_max = max(task_hours_per_person.get((p, t), 0) for p in qualified)
            t_min = min(task_hours_per_person.get((p, t), 0) for p in qualified)
            variety_report.append(
                f"Task '{t}': abs_spread={t_max - t_min}h, "
                f"prop_spread={p_max - p_min:.2%} "
                f"(max_prop={p_max:.1%}, min_prop={p_min:.1%})")
            for p in qualified:
                hrs  = task_hours_per_person.get((p, t), 0)
                prop = proportions[p]
                avg_prop = sum(proportions.values()) / len(proportions)
                diff = prop - avg_prop
                flag = "OK" if abs(diff) < 0.05 else ("OVER" if diff > 0 else "SHORT")
                variety_report.append(
                    f"  {flag}: {p} — {hrs}h ({prop:.1%} of {assigned_counts.get(p, 0)}h total, "
                    f"dev {diff:+.1%})")
    sol["variety_issues"] = variety_report

    # — Just Work report —
    if all_u_any_vals:
        must_work_issues = []
        for d in days:
            for p, h in must_work_set_per_day[d]:
                u_val = all_u_any_vals.get((p, h, d))
                if u_val is None:
                    continue
                assigned = assignment[d].get(p, {}).get(h)
                if u_val > 0.5:
                    avail_v = availability.get((p, h, d), 0)
                    if avail_v == 0 or not any((p, t, h, d) in all_x_vals for t in tasks):
                        must_work_issues.append(f"IMPOSSIBLE: {p} @ {h}, {d} — not available or no skill")
                    else:
                        must_work_issues.append(f"UNFULFILLED: {p} @ {h}, {d} — not assigned")
                else:
                    must_work_issues.append(f"FULFILLED: {p} @ {h}, {d} → '{assigned}'")
        sol["just_work_issues"] = must_work_issues or ["No Just Work mandates defined."]
    else:
        sol["just_work_issues"] = ["No Just Work mandates defined."]

    # ══════════════════════════════════════════════════════════════════
    # FINAL GARBAGE COLLECTION
    # ══════════════════════════════════════════════════════════════════
    gc.collect()

    return sol