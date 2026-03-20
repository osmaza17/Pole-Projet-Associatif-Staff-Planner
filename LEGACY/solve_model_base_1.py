# ============================================================
# BACKEND: GUROBIPY MODEL
# ============================================================

import gurobipy as gp
from gurobipy import GRB


def solve_model(data, ui_update_callback=None, active_model_ref=None):
    people = data["people"];    tasks = data["tasks"]
    hours  = data["hours"];     days  = data["days"]
    demand   = data["demand"];  availability = data["availability"]
    emergency = data["emergency"]
    skills   = data["skills"];  force        = data["force"]
    social   = data["social"];  min_quota    = data["min_quota"]
    rotation = data["rotation"];pref_cost    = data["pref_cost"]
    X_prev   = data["X_prev"]
    W = data["weights"]
    enforced_rest = data.get("enforced_rest", False)
    max_consec_hours = data.get("max_consec_hours", None)
    captains = data.get("captains", [])
    hard_enemies = data.get("hard_enemies", False)

    W_COVERAGE  = W["W_COVERAGE"];  W_FORCE     = W["W_FORCE"]
    W_CAPTAIN   = W["W_CAPTAIN"];   W_EMERG     = W["W_EMERG"]
    W_STABILITY = W["W_STABILITY"]; W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_GLOBAL = W["W_EQ_GLOBAL"]; W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];     W_PREF      = W["W_PREF"]

    # =================================================================
    # PRE-COMPUTED SETS & LOOKUPS
    # =================================================================
    avail_set = {(p,h,j) for (p,h,j), v in availability.items() if v == 1}
    skill_set = {(p,t)   for (p,t),   v in skills.items()       if v == 1}

    x_set = {(p,t,h,j) for p in people for t in tasks
                   for j in days for h in hours[j]
                   if (p,t) in skill_set and (p,h,j) in avail_set}

    # h_next maps (hour, day) → next hour on that same day
    h_next = {}
    for j in days:
        for i in range(len(hours[j]) - 1):
            h_next[(hours[j][i], j)] = hours[j][i+1]

    friend_pairs = [(p1,p2) for (p1,p2), val in social.items() if val == 1]
    enemy_pairs  = [(p1,p2) for (p1,p2), val in social.items() if val == -1]

    max_hours_any_day = max(len(hours[j]) for j in days) if days else 0
    total_hours_all_days = sum(len(hours[j]) for j in days)

    # =================================================================
    # MODEL
    # =================================================================
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model("StaffScheduler", env=env)

    if active_model_ref is not None:
        active_model_ref[0] = model

    # =================================================================
    # VARIABLES — x is the ONLY binary variable.
    #             All auxiliaries are CONTINUOUS (integrality induced by x).
    #             Each auxiliary is created ONLY over the keys where it matters.
    # =================================================================

    # --- Core assignment: person p does task t at hour h on day j ----------
    x = model.addVars(x_set, vtype=GRB.BINARY, name="x")

    # --- Missing staff: how many slots of (t,h,j) are unfilled -------------
    m_keys = [(t,h,j) for t in tasks for j in days for h in hours[j]]
    m = model.addVars(m_keys, lb=0, vtype=GRB.CONTINUOUS, name="m")

    # --- Unfulfilled mandates: created for EVERY force=1 entry -------------
    force_keys = [(p,t,h,j) for (p,t,h,j), val in force.items() if val == 1]
    u = model.addVars(force_keys, lb=0, vtype=GRB.CONTINUOUS, name="u")

    # --- Deviation from previous plan: only where x can exist ---------------
    d = model.addVars(x_set, lb=0, vtype=GRB.CONTINUOUS, name="d")

    # --- Quota miss: only for (p,t) pairs with a positive quota wish --------
    quota_keys = [(p,t) for p in people for t in tasks
                  if min_quota.get((p,t), 0) > 0]
    q = model.addVars(quota_keys, lb=0, vtype=GRB.CONTINUOUS, name="q")

    # --- Work restarts (gap detection): only where person is available ------
    r_keys = [(p,h,j) for p in people for j in days for h in hours[j]
              if (p,h,j) in avail_set]
    r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS, name="r")

    # --- Equity bounds ------------------------------------------------------
    n_max = model.addVars(days, lb=0, ub=max_hours_any_day, vtype=GRB.CONTINUOUS, name="n_max")
    n_min = model.addVars(days, lb=0, ub=max_hours_any_day, vtype=GRB.CONTINUOUS, name="n_min")
    z_max = model.addVar(lb=0, ub=total_hours_all_days, vtype=GRB.CONTINUOUS, name="z_max")
    z_min = model.addVar(lb=0, ub=total_hours_all_days, vtype=GRB.CONTINUOUS, name="z_min")

    # --- Consecutive penalty: rotation=1 AND both hours have valid x --------
    consec_keys = [(p,t,h,j) for p in people for t in tasks
                   for j in days for h in hours[j][:-1]
                   if rotation.get(t, 0) == 1
                   and (p,t,h,j) in x_set
                   and (h,j) in h_next
                   and (p,t,h_next[(h,j)],j) in x_set]
    c = model.addVars(consec_keys, lb=0, vtype=GRB.CONTINUOUS, name="c")

    # --- Friends separation: only where BOTH friends can work ---------------
    v_keys = [(p1,p2,t,h,j) for p1,p2 in friend_pairs for t in tasks
              for j in days for h in hours[j]
              if (p1,t,h,j) in x_set and (p2,t,h,j) in x_set]
    v = model.addVars(v_keys, lb=0, vtype=GRB.CONTINUOUS, name="v")

    # --- Enemies together: only where BOTH enemies can work -----------------
    enemy_scope = [(p1,p2,t,h,j) for p1,p2 in enemy_pairs for t in tasks
                   for j in days for h in hours[j]
                   if (p1,t,h,j) in x_set and (p2,t,h,j) in x_set]
    if hard_enemies:
        w_keys = []
    else:
        w_keys = enemy_scope
        w = model.addVars(w_keys, lb=0, vtype=GRB.CONTINUOUS, name="w")

    # --- Captain slack: only if captains exist, only active hours -----------
    active_hj = []
    if captains:
        active_hj = [(h,j) for j in days for h in hours[j]
                     if sum(demand.get((t,h,j), 0) for t in tasks) > 0]
        k = model.addVars(active_hj, lb=0, vtype=GRB.CONTINUOUS, name="k")

    # =================================================================
    # OBJECTIVE
    # =================================================================
    obj = gp.LinExpr()

    obj += W_COVERAGE * m.sum()
    obj += W_FORCE  * u.sum()

    if captains:
        obj += W_CAPTAIN * k.sum()

    obj += W_EMERG * gp.quicksum(
        emergency.get((p,h,j), 0) * x[p,t,h,j]
        for p,t,h,j in x_set)

    obj += W_STABILITY * d.sum()

    for j in days:
        obj += W_EQ_DAY * (n_max[j] - n_min[j])

    obj += W_EQ_GLOBAL * (z_max - z_min)

    if consec_keys:
        obj += W_ROTATION * c.sum()

    if v_keys:
        obj += W_SOCIAL * v.sum()

    if w_keys:
        obj += W_SOCIAL * w.sum()

    obj += W_GAP   * r.sum()
    obj += W_QUOTA * q.sum()

    obj += W_PREF * gp.quicksum(
        pref_cost.get((p,t), 0) * x[p,t,h,j]
        for p,t,h,j in x_set)

    model.setObjective(obj, GRB.MINIMIZE)

    # =================================================================
    # CONSTRAINTS
    # =================================================================

    # A. DEMAND COVERAGE
    for t in tasks:
        for j in days:
            for h in hours[j]:
                model.addConstr(x.sum('*', t, h, j) + m[t,h,j] == demand.get((t,h,j), 0))

    # B.1. MANDATED TASKS (soft)
    model.addConstrs(
        (1 - x[p,t,h,j] <= u[p,t,h,j]
         for p,t,h,j in force_keys if (p,t,h,j) in x_set),
        name="mandate_valid")

    model.addConstrs(
        (u[p,t,h,j] >= 1
         for p,t,h,j in force_keys if (p,t,h,j) not in x_set),
        name="mandate_impossible")

    # B.2. CAPTAINS (soft)
    if captains:
        for h,j in active_hj:
            model.addConstr(
                gp.quicksum(x.sum(p, '*', h, j) for p in captains) + k[h,j] >= 1)

    # C. ANTI-UBIQUITY (max 1 task per available person-hour)
    for p in people:
        for j in days:
            for h in hours[j]:
                if (p,h,j) in avail_set:
                    model.addConstr(x.sum(p, '*', h, j) <= 1)

    # E. EQUITY — Daily bounds (only people available that day)
    for j in days:
        people_avail_j = [p for p in people
                          if any((p,h,j) in avail_set for h in hours[j])]
        for p in people_avail_j:
            model.addConstr(x.sum(p, '*', '*', j) <= n_max[j])
            model.addConstr(x.sum(p, '*', '*', j) >= n_min[j])

    # E. EQUITY — Global bounds (only people with any availability)
    people_with_avail = [p for p in people
                         if any((p,h,j) in avail_set for j in days for h in hours[j])]
    for p in people_with_avail:
        model.addConstr(x.sum(p, '*', '*', '*') <= z_max)
        model.addConstr(x.sum(p, '*', '*', '*') >= z_min)

    # F. ROTATION FATIGUE — batch
    model.addConstrs(
        (x[p,t,h,j] + x[p,t,h_next[(h,j)],j] - c[p,t,h,j] <= 1
         for p,t,h,j in consec_keys),
        name="rotation")

    # G. SOCIAL — Friends (both can work → penalize separation)
    if v_keys:
        model.addConstrs(
            (x[p1,t,h,j] - x[p2,t,h,j] <= v[p1,p2,t,h,j]
             for p1,p2,t,h,j in v_keys),
            name="friends_a")
        model.addConstrs(
            (x[p2,t,h,j] - x[p1,t,h,j] <= v[p1,p2,t,h,j]
             for p1,p2,t,h,j in v_keys),
            name="friends_b")

    # G. SOCIAL — Enemies (both can work → penalize togetherness)
    if hard_enemies:
        model.addConstrs(
            (x[p1,t,h,j] + x[p2,t,h,j] <= 1
             for p1,p2,t,h,j in enemy_scope),
            name="enemies_hard")
    elif w_keys:
        model.addConstrs(
            (x[p1,t,h,j] + x[p2,t,h,j] - w[p1,p2,t,h,j] <= 1
             for p1,p2,t,h,j in w_keys),
            name="enemies")

    # H. MINIMUM QUOTA (per day)
    for p,t in quota_keys:
        for j in days:
            target = min(min_quota.get((p,t), 0), len(hours[j]))
            if target > 0:
                model.addConstr(x.sum(p, t, '*', j) + q[p,t] >= target)

    # I. STABILITY / DEVIATION — batch
    model.addConstrs(
        (d[p,t,h,j] >= X_prev.get((p,t,h,j), 0) - x[p,t,h,j]
         for p,t,h,j in x_set),
        name="dev_a")
    model.addConstrs(
        (d[p,t,h,j] >= x[p,t,h,j] - X_prev.get((p,t,h,j), 0)
         for p,t,h,j in x_set),
        name="dev_b")

    # J.1. GAP DETECTION (work restarts)
    for j in days:
        for p in people:
            if (p, hours[j][0], j) in avail_set:
                model.addConstr(r[p, hours[j][0], j] == x.sum(p, '*', hours[j][0], j))
            for i in range(1, len(hours[j])):
                h_curr = hours[j][i]; h_prev = hours[j][i-1]
                if (p, h_curr, j) in avail_set:
                    model.addConstr(
                        r[p, h_curr, j] >= x.sum(p, '*', h_curr, j) - x.sum(p, '*', h_prev, j))

    # J.2. ENFORCED REST (sliding window)
    if enforced_rest and max_consec_hours is not None:
        Y = max_consec_hours
        for j in days:
            for p in people:
                for i in range(len(hours[j]) - Y):
                    window = hours[j][i:i+Y+1]
                    model.addConstr(
                        gp.quicksum(x.sum(p, '*', tau, j) for tau in window) <= Y)

    # =================================================================
    # SOLVER CONFIG & OPTIMIZE
    # =================================================================
    model._x = x
    model._x_set = x_set

    solver_params = data.get("solver_params", {})
    for param_name, param_value in solver_params.items():
        try:
            model.setParam(param_name, param_value)
        except Exception as e:
            print(f"Warning: Could not set param {param_name}={param_value}: {e}")

    def intermediate_solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if ui_update_callback:
                try:
                    x_vals = model.cbGetSolution(model._x)
                    temp_assignment = {j: {p: {} for p in people} for j in days}
                    for j in days:
                        for p in people:
                            for h in hours[j]:
                                assigned_task = None
                                for t in tasks:
                                    if (p,t,h,j) in model._x_set and x_vals[p,t,h,j] > 0.5:
                                        assigned_task = t; break
                                temp_assignment[j][p][h] = assigned_task
                    partial_sol = {
                        "status": "Solving (New Best Found)...",
                        "assignment": temp_assignment}
                    ui_update_callback(partial_sol)
                except Exception:
                    pass

    model.optimize(intermediate_solution_callback)

    # =================================================================
    # EXTRACT FINAL RESULTS
    # =================================================================
    sol = {}
    status_map = {
        GRB.OPTIMAL:     "Optimal",
        GRB.TIME_LIMIT:  "Time Limit Reached",
        GRB.INFEASIBLE:  "Infeasible",
        GRB.INTERRUPTED: "Interrupted by User"}
    sol["status"] = status_map.get(model.Status, f"Status Code: {model.Status}")
    sol["enforced_rest"] = enforced_rest

    if model.SolCount == 0:
        raise Exception("No feasible solution was found before stopping/timeout.")

    assignment = {}
    for j in days:
        assignment[j] = {}
        for p in people:
            assignment[j][p] = {}
            for h in hours[j]:
                assigned_task = None
                for t in tasks:
                    if (p,t,h,j) in x_set and x[p,t,h,j].X > 0.5:
                        assigned_task = t; break
                assignment[j][p][h] = assigned_task
    sol["assignment"] = assignment

    missing = []
    for t in tasks:
        for j in days:
            for h in hours[j]:
                val = m[t,h,j].X
                if val > 0.01:
                    missing.append(f"{t} @ {h}, {j}: {val:.0f} missing")
    sol["missing"] = missing

    workload = {}
    for p in people:
        total_hrs = sum(1 for t in tasks for j in days for h in hours[j]
                        if (p,t,h,j) in x_set and x[p,t,h,j].X > 0.5)
        workload[p] = total_hrs
    sol["workload"] = workload
    sol["z_max"] = z_max.X; sol["z_min"] = z_min.X

    gaps = []
    for j in days:
        day_gaps_found = []
        for p in people:
            starts_count = 0; start_times = []
            for h in hours[j]:
                if (p,h,j) in avail_set and r[p,h,j].X > 0.5:
                    starts_count += 1; start_times.append(h)
            if starts_count > 1:
                day_gaps_found.append(
                    f"  • {p}: {starts_count} blocks (Starts: {', '.join(start_times)})")
        if day_gaps_found:
            gaps.append(f"--- {j} ---")
            gaps.extend(day_gaps_found)
    sol["gaps"] = gaps

    soc_issues = []
    if friend_pairs:
        for (p1,p2) in friend_pairs:
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        p1_w = (p1,t,h,j) in x_set and x[p1,t,h,j].X > 0.5
                        p2_w = (p2,t,h,j) in x_set and x[p2,t,h,j].X > 0.5
                        if p1_w and p2_w:
                            soc_issues.append(
                                f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {j}")
    if enemy_pairs:
        enemy_violations = 0
        for p1,p2,t,h,j in w_keys:
            if w[p1,p2,t,h,j].X > 0.5:
                soc_issues.append(
                    f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {j}")
                enemy_violations += 1
        if enemy_violations == 0:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
    sol["social_issues"] = soc_issues

    captain_issues = []
    if captains:
        for h,j in active_hj:
            if k[h,j].X > 0.5:
                captain_issues.append(f"MISSING CAPTAIN @ {h}, {j}")
        if not captain_issues:
            captain_issues.append("SUCCESS: All active hours have at least one captain on duty.")
    sol["captain_issues"] = captain_issues

    emerg_issues = []
    for p in people:
        for j in days:
            for h in hours[j]:
                if emergency.get((p,h,j), 0) == 1:
                    for t in tasks:
                        if (p,t,h,j) in x_set and x[p,t,h,j].X > 0.5:
                            emerg_issues.append(f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {j}")
    sol["emerg_issues"] = emerg_issues

    rotation_issues = []
    for p in people:
        for t in tasks:
            if rotation.get(t, 0) == 1:
                for j in days:
                    for i in range(len(hours[j]) - 1):
                        h_curr = hours[j][i]; h_nxt = hours[j][i+1]
                        if ((p,t,h_curr,j) in x_set and x[p,t,h_curr,j].X > 0.5 and
                            (p,t,h_nxt,j)  in x_set and x[p,t,h_nxt,j].X  > 0.5):
                            rotation_issues.append(
                                f"CONSECUTIVE: {p} doing '{t}' at {h_curr} & {h_nxt}, {j}")
    if not rotation_issues:
        rotation_issues.append("SUCCESS: No consecutive hours on rotation tasks.")
    sol["rotation_issues"] = rotation_issues

    return sol