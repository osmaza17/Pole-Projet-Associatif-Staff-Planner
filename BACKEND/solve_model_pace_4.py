import time
import gurobipy as gp
from gurobipy import GRB


def solve_model(data, ui_update_callback=None, active_model_ref=None):

    # ── Unpack ────────────────────────────────────────────────────────
    people    = data["people"];     tasks        = data["tasks"]
    hours     = data["hours"];      days         = data["days"]
    demand    = data["demand"];     availability = data["availability"]
    emergency = data["emergency"]
    skills    = data["skills"];     force        = data["force"]
    social    = data["social"];     min_quota    = data["min_quota"]
    rotation  = data["rotation"];   pref_cost    = data["pref_cost"]
    X_prev    = data["X_prev"]

    enforced_rest    = data.get("enforced_rest", False)
    max_consec_hours = data.get("max_consec_hours", None)
    captains         = data.get("captains", [])
    hard_enemies     = data.get("hard_enemies", False)
    day_heuristics   = data.get("day_heuristics", 0)   # 0=coupled, 1=pacing

    W = data["weights"]
    W_COVERAGE  = W["W_COVERAGE"];  W_FORCE     = W["W_FORCE"]
    W_CAPTAIN   = W["W_CAPTAIN"];   W_EMERG     = W["W_EMERG"]
    W_STABILITY = W["W_STABILITY"]; W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_GLOBAL = W["W_EQ_GLOBAL"]; W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];     W_PREF      = W["W_PREF"]

    # ── Global lookups (computed once) ───────────────────────────────
    avail_set = {(p, h, d) for (p, h, d), v in availability.items() if v == 1}
    skill_set = {(p, t)    for (p, t),    v in skills.items()        if v == 1}

    # ── Pacing heuristic setup ────────────────────────────────────────
    if day_heuristics == 1:
        total_demand  = sum(demand.get((t, h, d), 0)
                            for t in tasks for d in days for h in hours[d])
        number_people = len([p for p in people
                             if any((p, h, d) in avail_set
                                    for d in days for h in hours[d])]) or 1
        number_days   = len(days) or 1
        pace          = total_demand / (number_people * number_days)
        accumulated_hours = {p: 0 for p in people}
        days_iterator     = [[d] for d in days]
    else:
        days_iterator = [days]

    # ── Result accumulators (shared across loop iterations) ──────────
    all_x_vals = {}
    all_m_vals = {}
    all_g_vals = {}

    partial_assignment = {d: {p: {h: None for h in hours[d]}
                               for p in people} for d in days}

    final_status  = "Optimal"
    final_mip_gap = 0.0
    last_ui_update = [0.0]

    status_map = {GRB.OPTIMAL:     "Optimal",
                  GRB.TIME_LIMIT:  "Time Limit Reached",
                  GRB.INFEASIBLE:  "Infeasible",
                  GRB.INTERRUPTED: "Interrupted by User"}

    solve_start = time.monotonic()

    # ═════════════════════════════════════════════════════════════════
    # MAIN LOOP — one iteration (coupled) or one per day (pacing)
    # ═════════════════════════════════════════════════════════════════
    for loop_idx, current_days in enumerate(days_iterator):

        current_days_set = set(current_days)
        avail_set_day    = {(p, h, d) for (p, h, d) in avail_set
                            if d in current_days_set}
        people_available = [p for p in people
                            if any((p, h, d) in avail_set_day
                                   for d in current_days for h in hours[d])]

        # (hour, day) → next hour, restricted to current iteration's days
        h_next = {(hours[d][i], d): hours[d][i + 1]
                  for d in current_days for i in range(len(hours[d]) - 1)}

        # ── Model ─────────────────────────────────────────────────────
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1)
        env.start()
        model = gp.Model("StaffScheduler", env=env)

        if active_model_ref is not None:
            if active_model_ref[0] is not None:
                active_model_ref[0].terminate()
            active_model_ref[0] = model

        # ── Decision variables ────────────────────────────────────────

        # x[p,t,h,d]: 1 if person p does task t at hour h on day d
        x_set = {(p, t, h, d)
                 for p in people for t in tasks
                 for d in current_days for h in hours[d]
                 if (p, t) in skill_set and (p, h, d) in avail_set}
        x = model.addVars(x_set, vtype=GRB.BINARY, name="x")

        # m[t,h,d]: uncovered demand slots
        m_keys = [(t, h, d)
                  for t in tasks for d in current_days for h in hours[d]
                  if demand.get((t, h, d), 0) > 0]
        m = model.addVars(m_keys, lb=0, vtype=GRB.CONTINUOUS, name="m")

        # u[p,t,h,d]: unfulfilled force mandate
        force_keys = [(p, t, h, d) for (p, t, h, d), v in force.items()
                      if v == 1 and d in current_days_set]
        u = model.addVars(force_keys, lb=0, vtype=GRB.CONTINUOUS, name="u")

        # z[p,t,h,d]: |x - X_prev| stability deviation
        z = model.addVars(x_set, lb=0, vtype=GRB.CONTINUOUS, name="z")

        # q[p,t,d]: quota shortfall
        quota_keys = [(p, t, d)
                      for p in people for t in tasks for d in current_days
                      if min_quota.get((p, t, d), 0) > 0]
        q = model.addVars(quota_keys, lb=0, vtype=GRB.CONTINUOUS, name="q")

        # g[p,h,d]: 1 if person starts a new work block at hour h on day d
        g = model.addVars(avail_set_day, lb=0, vtype=GRB.CONTINUOUS, name="g")

        # r[p,d]: excess work blocks (max(0, blocks-1))
        r_keys = [(p, d) for p in people for d in current_days
                  if any((p, h, d) in avail_set_day for h in hours[d])]
        r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS, name="r")

        # j_max[d], j_min[d]: daily equity bounds
        max_hours_any_day = max((len(hours[d]) for d in current_days), default=0)
        j_max = model.addVars(current_days, lb=0, ub=max_hours_any_day,
                               vtype=GRB.CONTINUOUS, name="j_max")
        j_min = model.addVars(current_days, lb=0, ub=max_hours_any_day,
                               vtype=GRB.CONTINUOUS, name="j_min")

        # Global workload equity:
        #   Coupled mode → w_max / w_min: range of total hours across people
        #   Pacing mode  → delta_plus / delta_minus: deviation from pace target
        if day_heuristics == 0:
            w_max = model.addVar(lb=0, ub=max_hours_any_day * len(current_days),
                                 vtype=GRB.CONTINUOUS, name="w_max")
            w_min = model.addVar(lb=0, ub=max_hours_any_day * len(current_days),
                                 vtype=GRB.CONTINUOUS, name="w_min")
        else:
            delta_plus  = model.addVars(people_available, lb=0,
                                        vtype=GRB.CONTINUOUS, name="delta_plus")
            delta_minus = model.addVars(people_available, lb=0,
                                        vtype=GRB.CONTINUOUS, name="delta_minus")

        # c[p,t,h,d]: consecutive rotation penalty
        # h_next already restricted to current_days, so (h,d) keys always exist
        consec_keys = [(p, t, h, d)
                       for p in people for t in tasks
                       for d in current_days for h in hours[d][:-1]
                       if rotation.get(t, 0) == 1
                       and (p, t, h,              d) in x_set
                       and (p, t, h_next[(h, d)], d) in x_set]
        c = model.addVars(consec_keys, lb=0, vtype=GRB.CONTINUOUS, name="c")

        # f[p1,p2,t,h,d]: friends separated
        friends_keys = [(p1, p2, t, h, d)
                        for (p1, p2), sv in social.items() if sv == 1
                        for t in tasks for d in current_days for h in hours[d]
                        if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]
        f = model.addVars(friends_keys, lb=0, vtype=GRB.CONTINUOUS, name="f")

        # e[p1,p2,t,h,d]: enemies together (soft mode only)
        enemies_scope = [(p1, p2, t, h, d)
                         for (p1, p2), sv in social.items() if sv == -1
                         for t in tasks for d in current_days for h in hours[d]
                         if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]
        enemies_keys = [] if hard_enemies else enemies_scope
        e = model.addVars(enemies_keys, lb=0, vtype=GRB.CONTINUOUS, name="e")

        # k[h,d]: missing captain at active hour
        active_hd = ([(h, d) for d in current_days for h in hours[d]
                      if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
                     if captains else [])
        k = model.addVars(active_hd, lb=0, vtype=GRB.CONTINUOUS, name="k")

        # ── Objective ─────────────────────────────────────────────────
        obj = gp.LinExpr()

        obj += W_COVERAGE  * m.sum()
        obj += W_FORCE     * u.sum()
        obj += W_CAPTAIN   * k.sum()
        obj += W_EMERG     * gp.quicksum(emergency.get((p, h, d), 0) * x[p, t, h, d]
                                         for p, t, h, d in x_set)
        obj += W_STABILITY * z.sum()
        obj += W_EQ_DAY    * gp.quicksum(j_max[d] - j_min[d] for d in current_days)

        # Coupled → penalize range of total hours across people
        # Pacing  → penalize sum of per-person deviations from pace target
        if day_heuristics == 0:
            obj += W_EQ_GLOBAL * (w_max - w_min)
        else:
            obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

        obj += W_ROTATION  * c.sum()
        obj += W_SOCIAL    * f.sum()
        obj += W_SOCIAL    * e.sum()
        obj += W_GAP       * r.sum()
        obj += W_QUOTA     * q.sum()
        obj += W_PREF      * gp.quicksum(pref_cost.get((p, t), 0) * x[p, t, h, d]
                                         for p, t, h, d in x_set)

        model.setObjective(obj, GRB.MINIMIZE)

        # ── Constraints ───────────────────────────────────────────────

        # A. Demand coverage: Σ_p x[p,t,h,d] + m[t,h,d] = D[t,h,d]
        model.addConstrs(
            (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
             for t, h, d in m_keys), name="coverage")

        zero_demand_keys = [(t, h, d)
                            for t in tasks for d in current_days for h in hours[d]
                            if demand.get((t, h, d), 0) == 0]
        model.addConstrs(
            (x.sum('*', t, h, d) == 0 for t, h, d in zero_demand_keys),
            name="zero_demand")

        # B. Force mandates (soft)
        model.addConstrs(
            (1 - x[p, t, h, d] <= u[p, t, h, d]
             for p, t, h, d in force_keys if (p, t, h, d) in x_set),
            name="mandate_valid")
        model.addConstrs(
            (u[p, t, h, d] >= 1
             for p, t, h, d in force_keys if (p, t, h, d) not in x_set),
            name="mandate_impossible")

        # C. Captain presence
        model.addConstrs(
            (gp.quicksum(x.sum(p, '*', h, d) for p in captains) + k[h, d] >= 1
             for h, d in active_hd), name="captain_presence")

        # D. Anti-ubiquity: person can only do one task per slot
        model.addConstrs(
            (x.sum(p, '*', h, d) <= 1 for p, h, d in avail_set_day),
            name="anti_ubiquity")

        # F.1. Daily equity bounds
        eq_day_keys = [(p, d) for d in current_days for p in people
                       if any((p, h, d) in avail_set_day for h in hours[d])]
        model.addConstrs(
            (x.sum(p, '*', '*', d) <= j_max[d] for p, d in eq_day_keys),
            name="eq_day_max")
        model.addConstrs(
            (x.sum(p, '*', '*', d) >= j_min[d] for p, d in eq_day_keys),
            name="eq_day_min")

        # F.2. Global equity
        if day_heuristics == 0:
            # w_max ≥ total_hours(p)  and  w_min ≤ total_hours(p)  ∀p available
            model.addConstrs(
                (x.sum(p, '*', '*', '*') <= w_max for p in people_available),
                name="eq_global_max")
            model.addConstrs(
                (x.sum(p, '*', '*', '*') >= w_min for p in people_available),
                name="eq_global_min")
        else:
            pace_target = pace * (loop_idx + 1)
            model.addConstrs(
                (accumulated_hours[p] + x.sum(p, '*', '*', current_days[0])
                 - delta_plus[p] + delta_minus[p] == pace_target
                 for p in people_available), name="pacing")

        # G. Rotation fatigue: no two consecutive hours on rotation tasks
        model.addConstrs(
            (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
             for p, t, h, d in consec_keys), name="rotation_fatigue")

        # H. Social — friends (Soc=+1): penalize separation
        model.addConstrs(
            (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
             for p1, p2, t, h, d in friends_keys), name="friends_right")
        model.addConstrs(
            (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
             for p1, p2, t, h, d in friends_keys), name="friends_left")

        # H. Social — enemies (Soc=-1): hard or soft
        if hard_enemies:
            model.addConstrs(
                (x[p1, t, h, d] + x[p2, t, h, d] <= 1
                 for p1, p2, t, h, d in enemies_scope), name="enemies_hard")
        else:
            model.addConstrs(
                (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
                 for p1, p2, t, h, d in enemies_keys), name="enemies_soft")

        # I. Minimum quota L[p,t,d]
        quota_constr_keys = [(p, t, d, target)
                             for p, t, d in quota_keys
                             if (target := min(min_quota.get((p, t, d), 0),
                                               len(hours[d]))) > 0]
        model.addConstrs(
            (x.sum(p, t, '*', d) + q[p, t, d] >= target
             for p, t, d, target in quota_constr_keys), name="min_quota")

        # J. Stability: z[p,t,h,d] ≥ |x - X_prev|
        model.addConstrs(
            (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
             for p, t, h, d in x_set), name="stability_right")
        model.addConstrs(
            (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
             for p, t, h, d in x_set), name="stability_left")

        # K.1. Work block restart detection
        gap_first_keys = [(p, d) for p in people for d in current_days
                          if (p, hours[d][0], d) in avail_set_day]
        model.addConstrs(
            (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
             for p, d in gap_first_keys), name="restart_first_hour")

        gap_rest_keys = [(p, hours[d][i], hours[d][i - 1], d)
                         for p in people for d in current_days
                         for i in range(1, len(hours[d]))
                         if (p, hours[d][i], d) in avail_set_day]
        model.addConstrs(
            (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
             for p, h_curr, h_prev, d in gap_rest_keys),
            name="restart_subsequent_hours")

        # K.2. r[p,d] = max(0, blocks[p,d] - 1)
        model.addConstrs(
            (r[p, d] >= gp.quicksum(g[p, h, d] for h in hours[d]
                                    if (p, h, d) in avail_set_day) - 1
             for p, d in r_keys), name="restarts_excess")

        # L. Enforced rest: sliding window of max_consec_hours
        if enforced_rest and max_consec_hours is not None:
            rest_keys = [(p, d, i)
                         for d in current_days for p in people
                         if any((p, h, d) in avail_set_day for h in hours[d])
                         for i in range(len(hours[d]) - max_consec_hours)]
            model.addConstrs(
                (gp.quicksum(x.sum(p, '*', tau, d)
                             for tau in hours[d][i:i + max_consec_hours + 1])
                 <= max_consec_hours
                 for p, d, i in rest_keys), name="enforced_rest")

        # ── Solve ─────────────────────────────────────────────────────
        model._x     = x
        model._x_set = x_set

        for param_name, param_value in data.get("solver_params", {}).items():
            try:
                model.setParam(param_name, param_value)
            except Exception as err:
                print(f"Warning: Could not set param {param_name}={param_value}: {err}")

        def intermediate_solution_callback(mdl, where):
            if where == GRB.Callback.MIPSOL and ui_update_callback:
                now = time.monotonic()
                if now - last_ui_update[0] < 0.5:
                    return
                last_ui_update[0] = now
                try:
                    x_vals = mdl.cbGetSolution(mdl._x)
                    temp_assignment = {d: {p: dict(partial_assignment[d][p])
                                           for p in people} for d in days}
                    for d in current_days:
                        for p in people:
                            for h in hours[d]:
                                temp_assignment[d][p][h] = next(
                                    (t for t in tasks
                                     if (p, t, h, d) in mdl._x_set
                                     and x_vals[p, t, h, d] > 0.5), None)
                    ui_update_callback({"status":     "Solving (New Best Found)...",
                                        "assignment": temp_assignment})
                except Exception:
                    pass

        model.optimize(intermediate_solution_callback)

        # ── Accumulate results ────────────────────────────────────────
        iter_status = status_map.get(model.Status, f"Status Code: {model.Status}")

        if model.SolCount == 0:
            if all_x_vals:
                final_status = iter_status
                break
            else:
                raise Exception("No feasible solution was found before stopping/timeout.")

        if iter_status != "Optimal":
            final_status = iter_status

        final_mip_gap = model.MIPGap

        all_x_vals.update({key: x[key].X for key in x_set})
        all_m_vals.update({key: m[key].X for key in m_keys})
        all_g_vals.update({key: g[key].X for key in avail_set_day})

        for d in current_days:
            for p in people:
                for h in hours[d]:
                    partial_assignment[d][p][h] = next(
                        (t for t in tasks
                         if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5),
                        None)

        if day_heuristics == 1:
            d_cur = current_days[0]
            for p in people:
                accumulated_hours[p] += sum(1 for h in hours[d_cur] for t in tasks
                                            if (p, t, h, d_cur) in x_set
                                            and x[p, t, h, d_cur].X > 0.5)

        if model.Status == GRB.INTERRUPTED:
            break

    # ═════════════════════════════════════════════════════════════════
    # RESULT EXTRACTION
    # ═════════════════════════════════════════════════════════════════
    solve_elapsed = time.monotonic() - solve_start

    sol = {"solve_time":    solve_elapsed,
           "status":        final_status,
           "mip_gap":       final_mip_gap,
           "enforced_rest": enforced_rest}

    # Assignment grid
    sol["assignment"] = {
        d: {p: {h: next((t for t in tasks
                         if all_x_vals.get((p, t, h, d), 0) > 0.5), None)
                for h in hours[d]}
            for p in people}
        for d in days}

    # Uncovered demand
    sol["missing"] = [f"{t} @ {h}, {d}: {v:.0f} missing"
                      for (t, h, d), v in all_m_vals.items() if v > 0.01]

    # Workload per person
    assigned_counts = {p: 0 for p in people}
    for p, t, h, d in all_x_vals:
        if all_x_vals[p, t, h, d] > 0.5:
            assigned_counts[p] += 1
    sol["workload"] = assigned_counts
    wl = list(assigned_counts.values())
    sol["w_max"] = float(max(wl)) if wl else 0.0
    sol["w_min"] = float(min(wl)) if wl else 0.0

    # Quota fulfilment
    quota_keys_all = [(p, t, d)
                      for p in people for t in tasks for d in days
                      if min_quota.get((p, t, d), 0) > 0]
    sol["quota_issues"] = []
    for p, t, d in quota_keys_all:
        demanded = min_quota[p, t, d]
        assigned = sum(1 for p2, t2, h2, d2 in all_x_vals
                       if p2 == p and t2 == t and d2 == d
                       and all_x_vals[p2, t2, h2, d2] > 0.5)
        status = "OK" if assigned >= demanded else "SHORTFALL"
        sol["quota_issues"].append(
            f"{status}: {p} — task '{t}' on {d}: {assigned}/{demanded} h assigned")

    # Work block fragmentation
    gaps = []
    for d in days:
        day_gaps = [f"  • {p}: {starts} blocks (Starts: {', '.join(hs)})"
                    for p in people
                    for starts, hs in [(
                        sum(1 for h in hours[d]
                            if (p, h, d) in avail_set and all_g_vals.get((p, h, d), 0) > 0.5),
                        [h for h in hours[d]
                         if (p, h, d) in avail_set and all_g_vals.get((p, h, d), 0) > 0.5])]
                    if starts > 1]
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # Social issues
    soc_issues = []
    active_slots = {(p, t, h, d) for (p, t, h, d), v in all_x_vals.items() if v > 0.5}

    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for p1b, t, h, d in active_slots:
            if p1b == p1 and (p2, t, h, d) in active_slots:
                soc_issues.append(f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    if any(sv == -1 for sv in social.values()):
        if not hard_enemies:
            violations = [f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                          for (p1, p2), sv in social.items() if sv == -1
                          for p1b, t, h, d in active_slots
                          if p1b == p1 and (p2, t, h, d) in active_slots]
        else:
            violations = []
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
    sol["social_issues"] = soc_issues

    # Captain coverage
    all_active_hd = ([(h, d) for d in days for h in hours[d]
                      if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
                     if captains else [])
    missing_caps = [f"MISSING CAPTAIN @ {h}, {d}"
                    for h, d in all_active_hd
                    if not any((cap, t, h, d) in active_slots
                               for cap in captains for t in tasks)]
    sol["captain_issues"] = (missing_caps or
                             ["SUCCESS: All active hours have at least one captain on duty."]
                             if all_active_hd else [])

    # Emergency call-ins
    sol["emerg_issues"] = [f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {d}"
                           for p, t, h, d in active_slots
                           if emergency.get((p, h, d), 0) == 1]

    # Rotation violations
    h_next_all = {(hours[d][i], d): hours[d][i + 1]
                  for d in days for i in range(len(hours[d]) - 1)}
    rot_violations = [f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next_all[(h, d)]}, {d}"
                      for p, t, h, d in active_slots
                      if rotation.get(t, 0) == 1
                      and (h, d) in h_next_all
                      and (p, t, h_next_all[(h, d)], d) in active_slots]
    sol["rotation_issues"] = (rot_violations or
                              ["SUCCESS: No consecutive hours on rotation tasks."])

    return sol