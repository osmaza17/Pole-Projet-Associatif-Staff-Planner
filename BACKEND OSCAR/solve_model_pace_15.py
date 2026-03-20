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
    social    = data["social"];     quota    = data["min_quota"]
    rotation  = data["rotation"];   pref_cost    = data["pref_cost"]
    X_prev    = data["X_prev"]

    max_consec_hours = data.get("max_consec_hours", {})  # {person: (max_work, min_rest)}
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
    W_STABILITY = W["W_STABILITY"]; W_EQ_GLOBAL = W["W_EQ_GLOBAL"]
    W_ROTATION  = W["W_ROTATION"]
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
    # STATIC PRECOMPUTATIONS — derived structures used across all days
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

    # Force mandates indexed by day
    forced_tasks_set_per_day = defaultdict(list)
    for (p, t, h, d), v in force.items():
        if v == 1:
            forced_tasks_set_per_day[d].append((p, t, h, d))

    # Just-work mandates indexed by day
    must_work_set_per_day = defaultdict(list)
    for (p, h, d), v in just_work.items():
        if v == 1:
            must_work_set_per_day[d].append((p, h))

    # Minimum quotas indexed by day
    quota_set_per_day = defaultdict(list)
    for (p, t, d), v in quota.items():
        if v > 0:
            quota_set_per_day[d].append((p, t, d))

    # Next-hour mapping for all days
    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)}

    # Derived sets used in multiple blocks
    rotation_tasks     = {t for t in tasks if rotation.get(t, 0) == 1}
    social_friends     = [(p1, p2) for (p1, p2), sv in social.items() if sv ==  1]
    social_enemies     = [(p1, p2) for (p1, p2), sv in social.items() if sv == -1]
    has_enemies        = bool(social_enemies)
    emergency_set      = {(p, h, d) for (p, h, d), v in emergency.items() if v == 1}
    preferences_values = {(p, t): v for (p, t), v in pref_cost.items() if v != 0}

    # Captain rules expanded to (rule_idx, task, hour, captains) per day
    captain_scope_per_day = defaultdict(list)
    if captain_rules and W_CAPTAIN > 0:
        for r_idx, rule in enumerate(captain_rules):
            rule_captains = tuple(rule["captains"])
            for d in days:
                for t in rule["tasks"]:
                    for h in rule.get("hours", {}).get(d, []):
                        if h in hours.get(d, []) and demand.get((t, h, d), 0) > 0:
                            captain_scope_per_day[d].append((r_idx, t, h, rule_captains))

    # ══════════════════════════════════════════════════════════════════
    # PACING — cumulative demand targets for equity tracking
    # ══════════════════════════════════════════════════════════════════
    demand_per_day = {d: sum(demand[k] for k in demand_set_per_day[d]) for d in days}
    total_demand   = sum(demand_per_day.values())

    cumulative_demand = {}
    running_demand    = 0
    for d in days:
        running_demand       += demand_per_day[d]
        cumulative_demand[d]  = running_demand

    # Accumulators updated day-by-day to track cross-day equity
    accumulated_hours       = {p: 0 for p in people}
    accumulated_task_hours  = defaultdict(int)
    accumulated_group_hours = {g: 0 for g in group_list}

    # ══════════════════════════════════════════════════════════════════
    # RESULT ACCUMULATORS
    # ══════════════════════════════════════════════════════════════════
    all_x_vals     = {}
    all_m_vals     = {}
    all_u_vals     = {}
    all_u_any_vals = {}
    all_w_vals     = {}
    consec_relaxations: dict[str, dict] = {}

    partial_assignment = {d: {p: {h: None for h in hours[d]} for p in people} for d in days}

    final_status   = "Optimal"
    final_mip_gap  = 0.0
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
                    # ║  BLOCK 0 — CORE: assignment x + coverage m      ║
                    # ║  Always active. Defines the fundamental model.  ║
                    # ╚══════════════════════════════════════════════════╝

                    # x[p,t,h,d] = 1 iff person p does task t at hour h on today
                    x_set = {(p, t, h, today)
                        for t, h, _ in demand_set_today
                        for p in people_per_task[t]
                        if (p, h, today) in available_set_today}

                    x = model.addVars(x_set, vtype=GRB.BINARY)

                    # m[t,h,d] = uncovered demand (coverage slack); demand integer → m integer
                    m = model.addVars(demand_set_today, lb=0, ub=demand, vtype=GRB.INTEGER)

                    obj += W_COVERAGE * m.sum()

                    # Demand coverage: Σx + m = demand
                    model.addConstrs(
                        x.sum('*', t, h, today) + m[t, h, today] == demand[(t, h, today)]
                        for t, h, _ in demand_set_today)

                    # One task per person per hour
                    model.addConstrs(
                        x.sum(p, '*', h, today) <= 1
                        for p, h, _ in available_set_today)

                    # people_with_x_today: shared by block 3 (equity) and block 9 (gaps)
                    people_with_x_today = {p for p, t, h, _ in x_set}

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 1 — FORCE MANDATES          (W_FORCE)   ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: u[p,t,h,d], u_any[p,h,d]          ║
                    # ╚══════════════════════════════════════════════════╝
                    force_keys = []
                    u_any_keys = []

                    if W_FORCE > 0:

                        force_keys = [
                            (p, t, h, today)
                            for p, t, h, _ in forced_tasks_set_today
                            if demand.get((t, h, today), 0) > 0]

                        # u[p,t,h,d] = 1 iff force mandate violated (binary: 1 - x ∈ {0,1})
                        u = model.addVars(force_keys, vtype=GRB.BINARY)

                        u_any_keys = [(p, h, today) for p, h in must_work_set_today]

                        # u_any[p,h,d] = 1 iff just-work mandate violated (Σx ∈ {0,1} → u_any ∈ {0,1})
                        u_any = model.addVars(u_any_keys, vtype=GRB.BINARY)

                        obj += W_FORCE * (u.sum() + u_any.sum())

                        # Force: penalize if mandate not met
                        model.addConstrs(
                            1 - x[p, t, h, today] <= u[p, t, h, today]
                            for p, t, h, _ in force_keys)

                        # Just Work: person must work on something
                        model.addConstrs(
                            x.sum(p, '*', h, today) + u_any[p, h, today] >= 1
                            for p, h in must_work_set_today)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 2 — EMERGENCY CALL-INS      (W_EMERG)   ║
                    # ║  Depends on: x                                  ║
                    # ║  No new variables — objective term only         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EMERG > 0 and emergency_set:
                        obj += W_EMERG * gp.quicksum(
                            x[p, t, h, today]
                            for p, t, h, _ in x_set
                            if (p, h, today) in emergency_set)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 3 — GLOBAL EQUITY         (W_EQ_GLOBAL) ║
                    # ║  Depends on: x, people_with_x_today            ║
                    # ║  Variables: delta_plus[p], delta_minus[p]      ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EQ_GLOBAL > 0:

                        cumulative_demand_today   = cumulative_demand[today]
                        person_pace_target = {}
                        for group in group_list:
                            t_val = cumulative_demand_today / number_of_groups / max(len(group_people[group]), 1)
                            for p in group_people[group]:
                                person_pace_target[p] = t_val

                        delta_plus  = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)
                        delta_minus = model.addVars(people_with_x_today, lb=0, vtype=GRB.CONTINUOUS)

                        obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

                        # Intra-group pace: accumulated + today's hours ≈ target
                        model.addConstrs(
                            accumulated_hours[p] + x.sum(p, '*', '*', today) - delta_plus[p] + delta_minus[p] == person_pace_target[p]
                            for p in people_with_x_today)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 4 — GROUP EQUITY          (W_EQ_GROUP)  ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: gamma_plus[g], gamma_minus[g]      ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_EQ_GROUP > 0 and number_of_groups > 1:

                        cumulative_demand_today   = cumulative_demand[today]
                        group_target_today = cumulative_demand_today / number_of_groups

                        gamma_plus  = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                        gamma_minus = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)

                        obj += W_EQ_GROUP * (gamma_plus.sum() + gamma_minus.sum())

                        x_set_per_group = defaultdict(list)
                        for p, t, h, _ in x_set:
                            x_set_per_group[person_to_group[p]].append((p, t, h, today))

                        # Inter-group pace: accumulated group hours + today ≈ group target
                        model.addConstrs(
                            accumulated_group_hours[group] + gp.quicksum(x[p, t, h, today] for p, t, h, _ in x_set_per_group[group]) - gamma_plus[group] + gamma_minus[group] == group_target_today
                            for group in group_list)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 5 — ROTATION              (W_ROTATION)  ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: c[p,t,h,d]                         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_ROTATION > 0 and rotation_tasks:

                        consec_keys = [
                            (p, t, h, today)
                            for t in rotation_tasks
                            for p in people_per_task[t]
                            for h in hours_today[:-1]
                            if (p, t, h, today)                       in x_set
                            and (p, t, h_next_all[(h, today)], today) in x_set]

                        # c[p,t,h,d] = 1 iff consecutive rotation hours at h and h+1
                        # x[h] + x[h+1] - c ≤ 1, x binary → c ∈ {0,1}
                        c = model.addVars(consec_keys, vtype=GRB.BINARY)

                        obj += W_ROTATION * c.sum()

                        model.addConstrs(
                            x[p, t, h, today] + x[p, t, h_next_all[(h, today)], today] - c[p, t, h, today] <= 1
                            for p, t, h, _ in consec_keys)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 6 — SOCIAL         (W_SOCIAL / enemies) ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: f[p1,p2,t,h,d], e[p1,p2,t,h,d]   ║
                    # ║  Note: hard enemies enforced even if W_SOCIAL=0 ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_SOCIAL > 0 or hard_enemies:

                        active_tasks_hours_per_person = defaultdict(set)

                        for p, t, h, _ in x_set:
                            active_tasks_hours_per_person[p].add((t, h))

                        friends_keys = [
                            (p1, p2, t, h, today)
                            for p1, p2 in social_friends
                            for t, h in active_tasks_hours_per_person[p1]
                            if (t, h) in active_tasks_hours_per_person[p2]]

                        enemies_scope = [
                            (p1, p2, t, h, today)
                            for p1, p2 in social_enemies
                            for t, h in active_tasks_hours_per_person[p1]
                            if (t, h) in active_tasks_hours_per_person[p2]]

                        if W_SOCIAL > 0:

                            enemies_keys = [] if hard_enemies else enemies_scope

                            # f[p1,p2,t,h,d] = |x[p1]-x[p2]|; both binary → f ∈ {0,1}
                            f = model.addVars(friends_keys, vtype=GRB.BINARY)
                            # e[p1,p2,t,h,d] = x[p1]+x[p2]-1 (soft enemy overlap)
                            e = model.addVars(enemies_keys, vtype=GRB.BINARY)

                            obj += W_SOCIAL * (f.sum() + e.sum())

                            # Friends: penalize separation on same task/hour
                            model.addConstrs(
                                x[p1, t, h, today] - x[p2, t, h, today] <= f[p1, p2, t, h, today]
                                for p1, p2, t, h, _ in friends_keys)

                            model.addConstrs(
                                x[p2, t, h, today] - x[p1, t, h, today] <= f[p1, p2, t, h, today]
                                for p1, p2, t, h, _ in friends_keys)

                            # Enemies (soft): penalize co-assignment
                            if not hard_enemies:
                                model.addConstrs(
                                    x[p1, t, h, today] + x[p2, t, h, today] - e[p1, p2, t, h, today] <= 1
                                    for p1, p2, t, h, _ in enemies_keys)

                        # Enemies (hard): prohibit co-assignment regardless of W_SOCIAL
                        if hard_enemies:
                            model.addConstrs(
                                x[p1, t, h, today] + x[p2, t, h, today] <= 1
                                for p1, p2, t, h, _ in enemies_scope)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 7 — MINIMUM QUOTA           (W_QUOTA)   ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: q[p,t,d]                           ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_QUOTA > 0:

                        quota_keys = quota_set_per_day[today]

                        # q[p,t,d] = quota deficit; quota integer, Σx integer → q integer
                        q = model.addVars(quota_keys, lb=0, ub=quota, vtype=GRB.INTEGER)

                        obj += W_QUOTA * q.sum()

                        model.addConstrs(
                            x.sum(p, t, '*', today) + q[p, t, today] >= min(quota[(p, t, today)], len(hours_today))
                            for p, t, _ in quota_keys)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 8 — STABILITY             (W_STABILITY) ║
                    # ║  Depends on: x, X_prev                          ║
                    # ║  Variables: z[p,t,h,d]                         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_STABILITY > 0 and X_prev:

                        # z[p,t,h,d] = |X_prev - x|; both binary → z ∈ {0,1}
                        z = model.addVars(x_set, vtype=GRB.BINARY)

                        obj += W_STABILITY * z.sum()

                        model.addConstrs(
                            z[p, t, h, today] >= X_prev.get((p, t, h, today), 0) - x[p, t, h, today]
                            for p, t, h, _ in x_set)

                        model.addConstrs(
                            z[p, t, h, today] >= x[p, t, h, today] - X_prev.get((p, t, h, today), 0)
                            for p, t, h, _ in x_set)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 9 — SPLIT SHIFTS / GAPS       (W_GAP)   ║
                    # ║  Depends on: x, people_with_x_today            ║
                    # ║  Variables: g[p,h,d], r[p,d]                   ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_GAP > 0:

                        g_keys = {(p, h, today) for p, t, h, _ in x_set}
                        r_keys = [(p, today) for p in people_with_x_today]

                        # g[p,h,d] = block-start indicator (= Σx(p,*,h) ∈ {0,1}) → BINARY
                        # r[p,d]   = number of extra shift blocks (Σg - 1) → INTEGER
                        g = model.addVars(g_keys, vtype=GRB.BINARY)
                        r = model.addVars(r_keys, lb=0, vtype=GRB.INTEGER)

                        obj += W_GAP * r.sum()

                        gap_first_keys = [p for p in people_with_x_today
                                          if (p, hours_today[0], today) in g_keys]

                        model.addConstrs(
                            g[p, hours_today[0], today] == x.sum(p, '*', hours_today[0], today)
                            for p in gap_first_keys)

                        gap_rest_keys = [
                            (p, hour_now, hour_before)
                            for p in people_with_x_today
                            for hour_before, hour_now in zip(hours_today[:-1], hours_today[1:])
                            if (p, hour_now, today) in g_keys]

                        model.addConstrs(
                            g[p, hour_now, today] >= x.sum(p, '*', hour_now, today) - x.sum(p, '*', hour_before, today)
                            for p, hour_now, hour_before in gap_rest_keys)

                        model.addConstrs(
                            r[p, today] >= gp.quicksum(g[p, h, today] for h in hours_today if (p, h, today) in g_keys) - 1
                            for p, _ in r_keys)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 10 — PREFERENCES             (W_PREF)   ║
                    # ║  Depends on: x                                  ║
                    # ║  No new variables — objective term only         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_PREF > 0 and preferences_values:
                        obj += W_PREF * gp.quicksum(
                            preferences_values[(p, t)] * x[p, t, h, today]
                            for p, t, h, _ in x_set
                            if (p, t) in preferences_values)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 11 — TASK VARIETY           (W_VARIETY) ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: v_max[p,t]                         ║
                    # ╚══════════════════════════════════════════════════╝
                    if W_VARIETY > 0:

                        v_max_keys = [
                            (p, t)
                            for p in people
                            for t in tasks
                            if skills.get((p, t), 0) == 1
                            and any((p, t, h, today) in x_set for h in hours_today)]

                        # v_max[p,t] = hours beyond the first occurrence (repetition cost)
                        # Σx integer, free_pass ∈ {0,1} → v_max always integer
                        v_max = model.addVars(v_max_keys, lb=0, vtype=GRB.INTEGER)

                        obj += W_VARIETY * v_max.sum()

                        # First ever occurrence of (p,t) across all days is free
                        for p, t in v_max_keys:
                            free_pass = 1 if accumulated_task_hours[p, t] == 0 else 0
                            model.addConstr(
                                v_max[p, t] >= x.sum(p, t, '*', today) - free_pass)

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 12 — CAPTAIN RULES         (W_CAPTAIN)  ║
                    # ║  Depends on: x                                  ║
                    # ║  Variables: w[r_idx,t,h,d]                     ║
                    # ║  Also computes sole_captain_hours_per_person    ║
                    # ║  → consumed by block 13 (max consecutive hours) ║
                    # ╚══════════════════════════════════════════════════╝

                    # Initialized here so block 13 always has a valid reference
                    sole_captain_hours_per_person: dict[str, frozenset] = {}
                    captain_keys = []

                    if W_CAPTAIN > 0 and captain_scope_per_day[today]:

                        captain_scope_today = captain_scope_per_day[today]
                        captain_keys = list(dict.fromkeys(
                            (rule_idx, task, hour, today)
                            for rule_idx, task, hour, rule_caps in captain_scope_today))

                        captain_possible, captain_impossible = [], []
                        for rule_idx, task, hour, rule_caps in captain_scope_today:
                            eligible = tuple(
                                cap for cap in rule_caps
                                if (cap, task, hour, today) in x_set)
                            (captain_possible if eligible else captain_impossible).append(
                                (rule_idx, task, hour, today, eligible))

                        # w[r,t,h,d] = 1 iff captain rule r violated; Σx_cap + w ≥ 1 → w ∈ {0,1}
                        w = model.addVars(captain_keys, vtype=GRB.BINARY)

                        obj += W_CAPTAIN * w.sum()

                        # At least one captain must be assigned, or pay penalty
                        model.addConstrs(
                            gp.quicksum(x[cap, cap_task, cap_hour, today] for cap in eligible_caps) + w[rule_idx, cap_task, cap_hour, today] >= 1
                            for rule_idx, cap_task, cap_hour, cap_day, eligible_caps in captain_possible)

                        # No eligible captain at all → maximum penalty
                        model.addConstrs(
                            w[rule_idx, cap_task, cap_hour, today] >= 1
                            for rule_idx, cap_task, cap_hour, cap_day, no_eligible in captain_impossible)

                        # Identify sole-captain hours to relax consecutive constraints
                        # (block 13 will skip windows that contain these hours)
                        if max_consec_hours and captain_possible:
                            sole_hours_accumulator: dict[str, set] = defaultdict(set)
                            for rule_idx, cap_task, cap_hour, cap_day, eligible_caps in captain_possible:
                                if len(eligible_caps) == 1:
                                    sole_hours_accumulator[eligible_caps[0]].add(cap_hour)
                            sole_captain_hours_per_person = {
                                person_name: frozenset(hours_set)
                                for person_name, hours_set in sole_hours_accumulator.items()}

                            for relaxed_person, relaxed_hours in sole_captain_hours_per_person.items():
                                consec_relaxations.setdefault(
                                    relaxed_person, {}).setdefault(
                                    today, []).extend(sorted(relaxed_hours))

                    # ╔══════════════════════════════════════════════════╗
                    # ║  BLOCK 13 — MAX CONSECUTIVE HOURS               ║
                    # ║  Depends on: x                                  ║
                    # ║  Depends on: sole_captain_hours_per_person (B12)║
                    # ║  No weight — active if max_consec_hours is set  ║
                    # ╚══════════════════════════════════════════════════╝
                    if max_consec_hours:

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

                            sole_captain_hours    = sole_captain_hours_per_person.get(person, frozenset())
                            valid_start_positions = range(len(hours_today) - window_size + 1)

                            if sole_captain_hours:
                                # Build prefix sum to quickly detect sole-captain overlap
                                sole_captain_prefix = [0] * (len(hours_today) + 1)
                                for hour_idx, hour in enumerate(hours_today):
                                    sole_captain_prefix[hour_idx + 1] = (
                                        sole_captain_prefix[hour_idx]
                                        + (1 if hour in sole_captain_hours else 0))

                                # Only constrain windows with no sole-captain hours inside
                                model.addConstrs(
                                    x.sum(person, '*', hours_today[start:start + window_size], today) <= max_work_hours
                                    for start in valid_start_positions
                                    if (sole_captain_prefix[start + window_size] - sole_captain_prefix[start] == 0))
                            else:
                                model.addConstrs(
                                    x.sum(person, '*', hours_today[start:start + window_size], today)
                                    <= max_work_hours
                                    for start in valid_start_positions)

                    # ══════════════════════════════════════════════════
                    # SET OBJECTIVE & APPLY SOLVER PARAMETERS
                    # ══════════════════════════════════════════════════
                    model.setObjective(obj, GRB.MINIMIZE)

                    model._x = x
                    for param_name, param_value in solver_params.items():
                        try:
                            model.setParam(param_name, param_value)
                        except Exception as err:
                            print(f"Warning: Could not set param {param_name}={param_value}: {err}")

                    # ══════════════════════════════════════════════════
                    # OPTIMIZE
                    # ══════════════════════════════════════════════════
                    if ui_update_callback is None:
                        model.optimize()

                    else:
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

                    # ══════════════════════════════════════════════════
                    # ACCUMULATE RESULTS
                    # ══════════════════════════════════════════════════
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

                    if W_FORCE > 0:
                        if force_keys:
                            all_u_vals.update(model.getAttr('X', u))
                        if u_any_keys:
                            all_u_any_vals.update(model.getAttr('X', u_any))

                    if W_CAPTAIN > 0 and captain_keys:
                        all_w_vals.update(model.getAttr('X', w))

                    # Update assignment and cross-day accumulators
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

                    # ══════════════════════════════════════════════════
                    # CRITICAL CLEANUP — break circular reference
                    # ══════════════════════════════════════════════════
                    if hasattr(model, '_x'):
                        del model._x

                    if active_model_ref is not None:
                        active_model_ref[0] = None

                # End of "with gp.Model()". Memory freed here.

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
        "solve_time":         solve_elapsed,
        "status":             final_status,
        "mip_gap":            final_mip_gap,
        "enforced_rest":      bool(max_consec_hours),
        "consec_relaxations": consec_relaxations,
        "assignment":         assignment,
    }

    # — Uncovered demand —
    sol["missing"] = [
        f"{t} @ {h}, {d}: {v:.0f} missing"
        for (t, h, d), v in all_m_vals.items() if v > 0.01]

    # — Force mandates report —
    if W_FORCE > 0 and all_u_vals:
        force_issues = []
        for d in days:
            for p, t, h, _ in forced_tasks_set_per_day[d]:
                u_val = all_u_vals.get((p, t, h, d))
                if u_val is None:
                    continue
                if u_val > 0.5:
                    force_issues.append(f"UNFULFILLED: {p} — task '{t}' @ {h}, {d}")
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
                demanded = quota[p, t, d]
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

    # — Task variety report —
    variety_report = []
    for p in people:
        reps = []
        for t in tasks:
            hrs = task_hours_per_person.get((p, t), 0)
            if hrs > 1:
                reps.append(f"'{t}'×{hrs}")
        if reps:
            variety_report.append(f"REPEAT: {p} — " + ", ".join(reps))
    if not variety_report:
        variety_report.append("SUCCESS: No person repeated a task more than once.")

    for t in tasks:
        qualified = people_per_task[t]
        if not qualified:
            continue
        touched   = sum(1 for p in qualified if task_hours_per_person.get((p, t), 0) > 0)
        repeated  = sum(1 for p in qualified if task_hours_per_person.get((p, t), 0) > 1)
        total_hrs = sum(task_hours_per_person.get((p, t), 0) for p in qualified)
        variety_report.append(
            f"Task '{t}': {touched}/{len(qualified)} qualified did it, "
            f"{repeated} repeated, {total_hrs} total hours")
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