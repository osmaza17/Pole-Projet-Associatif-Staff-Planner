import time
import logging
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict, Counter

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

    max_consec_hours = data.get("max_consec_hours", {})
    captains         = data.get("captains", [])
    hard_enemies     = data.get("hard_enemies", False)
    day_heuristics   = data.get("day_heuristics", 0)
    solver_params    = data.get("solver_params", {})

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
        total_demand            = sum(demand.get((t, h, d), 0)
                                      for t in tasks for d in days for h in hours[d])
        total_available_people  = [p for p in people
                                   if any((p, h, d) in avail_set
                                          for d in days for h in hours[d])]
        pace                    = total_demand / (len(total_available_people) or 1) / (len(days) or 1)
        accumulated_hours       = {p: 0 for p in people}
        days_iterator           = [[d] for d in days]
    else:
        days_iterator = [days]

    # ── Result accumulators ──────────────────────────────────────────
    all_x_vals = {}
    all_m_vals = {}
    all_u_vals = {}

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
        GRB.INF_OR_UNBD: "Infeasible or Unbounded",
    }

    solve_start = time.monotonic()

    # ═════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════════════════════════════
    for loop_idx, current_days in enumerate(days_iterator):

        current_days_set = set(current_days)
        avail_set_in_day = {(p, h, d) for (p, h, d) in avail_set
                            if d in current_days_set}
        people_available_in_day = [p for p in people
                                   if any((p, h, d) in avail_set_in_day
                                   for d in current_days for h in hours[d])]

        h_next = {(hours[d][i], d): hours[d][i + 1]
                  for d in current_days for i in range(len(hours[d]) - 1)}

        # ── Model ─────────────────────────────────────────────────────
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 1)
            env.start()
        except gp.GurobiError as e:
            raise RuntimeError(f"Gurobi license error: {e}") from e

        model = gp.Model("StaffScheduler", env=env)  # ← esta línea faltaba

        if active_model_ref is not None:
            if active_model_ref[0] is not None:
                active_model_ref[0].terminate()
            active_model_ref[0] = model

        interrupted = False

        try:

            # ══════════════════════════════════════════════════════════
            # DECISION VARIABLES
            # ══════════════════════════════════════════════════════════

            # x: main assignment
            x_set = {(p, t, h, d)
                     for p in people for t in tasks
                     for d in current_days for h in hours[d]
                     if (p, t) in skill_set
                     and (p, h, d) in avail_set
                     and demand.get((t, h, d), 0) > 0}
            x = model.addVars(x_set, vtype=GRB.BINARY)

            # m: missing demand (structural slack)
            m_keys = [(t, h, d)
                      for t in tasks for d in current_days for h in hours[d]
                      if demand.get((t, h, d), 0) > 0]
            m_ub = {(t, h, d): demand.get((t, h, d), 0) for t, h, d in m_keys}
            m = model.addVars(m_keys, lb=0, ub=m_ub, vtype=GRB.CONTINUOUS)

            # u: unfulfilled force mandates
            # ① u se crea para TODOS los mandatos (posibles e imposibles)
            if W_FORCE > 0:
                force_keys = [(p, t, h, d) for (p, t, h, d), v in force.items()
                              if v == 1
                              and d in current_days_set
                              and demand.get((t, h, d), 0) > 0]
                force_possible   = [(p, t, h, d) for p, t, h, d in force_keys
                                    if (p, t, h, d) in x_set]
                force_impossible = [(p, t, h, d) for p, t, h, d in force_keys
                                    if (p, t, h, d) not in x_set]
                u = model.addVars(force_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

            # z: stability
            if W_STABILITY > 0 and X_prev:
                z = model.addVars(x_set, lb=0, ub=1, vtype=GRB.CONTINUOUS)

            # q: quota shortfall — solo si W_QUOTA > 0
            if W_QUOTA > 0:
                quota_keys = []
                for p in people:
                    for t in tasks:
                        for d in current_days:
                            if min_quota.get((p, t, d), 0) > 0:
                                quota_keys.append((p, t, d))
                q_ub = {(p, t, d): min_quota.get((p, t, d), 0) for p, t, d in quota_keys}
                q = model.addVars(quota_keys, lb=0, ub=q_ub, vtype=GRB.CONTINUOUS)

            # g, r: gap detection / fragmentation penalty
            if W_GAP > 0:
                g_keys = {(p, h, d) for p, _, h, d in x_set}
                g = model.addVars(g_keys, lb=0, vtype=GRB.CONTINUOUS)

                r_keys = [(p, d) for p in people for d in current_days
                          if any((p, h, d) in g_keys for h in hours[d])]
                r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS)

            # j_max, j_min: daily equity
            max_hours_any_day = max((len(hours[d]) for d in current_days), default=0)
           
            if W_EQ_DAY > 0:
                j_max = model.addVars(current_days, lb=0, ub=max_hours_any_day,
                                      vtype=GRB.CONTINUOUS)
                spread = model.addVars(current_days, lb=0, vtype=GRB.CONTINUOUS)

            # w_max/w_min or delta: global equity
            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
                    w_max = model.addVar(lb=0, ub=max_hours_any_day * len(current_days),
                                         vtype=GRB.CONTINUOUS)
                    w_spread = model.addVar(lb=0, vtype=GRB.CONTINUOUS)
                else:
                    delta_plus  = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)
                    delta_minus = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)

            # c: rotation fatigue
            if W_ROTATION > 0:
                consec_keys = [(p, t, h, d)
                               for p in people for t in tasks
                               for d in current_days for h in hours[d][:-1]
                               if rotation.get(t, 0) == 1
                               and (p, t, h,              d) in x_set
                               and (p, t, h_next[(h, d)], d) in x_set]
                c = model.addVars(consec_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

            # enemies_scope: siempre computado (es constraint de factibilidad)
            enemies_scope = [(p1, p2, t, h, d)
                             for (p1, p2), sv in social.items() if sv == -1
                             for t in tasks for d in current_days for h in hours[d]
                             if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]

            # f, e: social preference variables
            if W_SOCIAL > 0:
                friends_keys = [(p1, p2, t, h, d)
                                for (p1, p2), sv in social.items() if sv == 1
                                for t in tasks for d in current_days for h in hours[d]
                                if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]
                f = model.addVars(friends_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                enemies_keys = [] if hard_enemies else enemies_scope
                e = model.addVars(enemies_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

            # k: captain presence slack
            if W_CAPTAIN > 0:
                active_hd = ([(h, d) for d in current_days for h in hours[d]
                              if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
                             if captains else [])
                k = model.addVars(active_hd, lb=0, ub=1, vtype=GRB.CONTINUOUS)

            # ══════════════════════════════════════════════════════════
            # OBJECTIVE FUNCTION
            # ══════════════════════════════════════════════════════════

            obj = gp.LinExpr()

            if W_COVERAGE > 0:
                obj += W_COVERAGE * m.sum()

            if W_FORCE > 0:
                obj += W_FORCE * u.sum()

            if W_CAPTAIN > 0:
                obj += W_CAPTAIN * k.sum()

            if W_EMERG > 0:
                obj += W_EMERG * gp.quicksum(
                    emergency.get((p, h, d), 0) * x[p, t, h, d]
                    for p, t, h, d in x_set)

            if W_STABILITY > 0 and X_prev:
                obj += W_STABILITY * z.sum()

            if W_EQ_DAY > 0:
                obj += W_EQ_DAY * (spread.sum() + j_max.sum())

            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
                    obj += W_EQ_GLOBAL * (w_spread + w_max)
                else:
                    obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

            if W_ROTATION > 0:
                obj += W_ROTATION * c.sum()

            if W_SOCIAL > 0:
                obj += W_SOCIAL * f.sum()
                obj += W_SOCIAL * e.sum()

            if W_GAP > 0:
                obj += W_GAP * r.sum()

            if W_QUOTA > 0:
                obj += W_QUOTA * q.sum()

            if W_PREF > 0:
                obj += W_PREF * gp.quicksum(
                    pref_cost.get((p, t), 0) * x[p, t, h, d]
                    for p, t, h, d in x_set)

            model.setObjective(obj, GRB.MINIMIZE)

            # ══════════════════════════════════════════════════════════
            # CONSTRAINTS
            # ══════════════════════════════════════════════════════════

            # A. Demand coverage
            model.addConstrs(
                (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
                 for t, h, d in m_keys))

            # B. Force mandates
            if W_FORCE > 0:
                # Mandatos posibles: penaliza si no se asigna
                model.addConstrs(
                    (1 - x[p, t, h, d] <= u[p, t, h, d]
                     for p, t, h, d in force_possible))
                # ① Mandatos imposibles: forzar u=1 (penalización fija)
                model.addConstrs(
                    (u[p, t, h, d] >= 1
                     for p, t, h, d in force_impossible))

            # C. Captain presence
            if W_CAPTAIN > 0:
                model.addConstrs(
                    (x.sum(captains, '*', h, d) + k[h, d] >= 1
                     for h, d in active_hd))

            # D. Anti-ubiquity
            model.addConstrs(
                (x.sum(p, '*', h, d) <= 1 for p, h, d in avail_set_in_day))

            # E. Daily equity bounds
            if W_EQ_DAY > 0:
                eq_day_keys = [(p, d) for d in current_days for p in people
                               if any((p, h, d) in avail_set_in_day for h in hours[d])]
                model.addConstrs(
                    (x.sum(p, '*', '*', d) <= j_max[d] for p, d in eq_day_keys))
                model.addConstrs(
                    (j_max[d] - x.sum(p, '*', '*', d) <= spread[d] for p, d in eq_day_keys))

            # F. Global equity
            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
                    model.addConstrs(
                        (x.sum(p, '*', '*', '*') <= w_max for p in people_available_in_day))
                    model.addConstrs(
                        (w_max - x.sum(p, '*', '*', '*') <= w_spread for p in people_available_in_day))
                else:
                    pace_target = pace * (loop_idx + 1)
                    model.addConstrs(
                        (accumulated_hours[p] + x.sum(p, '*', '*', current_days[0])
                         - delta_plus[p] + delta_minus[p] == pace_target
                         for p in people_available_in_day))

            # G. Rotation fatigue
            if W_ROTATION > 0:
                model.addConstrs(
                    (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
                     for p, t, h, d in consec_keys))

            # H. Social — friends (soft)
            if W_SOCIAL > 0:
                model.addConstrs(
                    (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
                     for p1, p2, t, h, d in friends_keys))
                model.addConstrs(
                    (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
                     for p1, p2, t, h, d in friends_keys))

                # Soft enemies
                if not hard_enemies:
                    model.addConstrs(
                        (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
                         for p1, p2, t, h, d in enemies_keys))

            # Hard enemies: siempre aplicado independientemente de W_SOCIAL
            if hard_enemies:
                model.addConstrs(
                    (x[p1, t, h, d] + x[p2, t, h, d] <= 1
                     for p1, p2, t, h, d in enemies_scope))

            # I. Minimum quota — solo si W_QUOTA > 0              ⑧
            if W_QUOTA > 0:
                quota_constr_keys = []
                for p, t, d in quota_keys:
                    target = min(min_quota.get((p, t, d), 0), len(hours[d]))
                    if target > 0:
                        quota_constr_keys.append((p, t, d, target))
                model.addConstrs(
                    (x.sum(p, t, '*', d) + q[p, t, d] >= target
                     for p, t, d, target in quota_constr_keys))

            # J. Stability
            if W_STABILITY > 0 and X_prev:
                forced_slots = {
                    (p, h, d)
                    for (p, t, h, d), v in force.items()
                    if v == 1 and d in current_days_set
                }
                model.addConstrs(
                    (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
                     for p, t, h, d in x_set
                     if (p, h, d) not in forced_slots))
                model.addConstrs(
                    (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
                     for p, t, h, d in x_set
                     if (p, h, d) not in forced_slots))

            # K. Gap detection and fragmentation penalty
            if W_GAP > 0:
                gap_first_keys = [(p, d) for p in people for d in current_days
                                  if (p, hours[d][0], d) in g_keys]
                model.addConstrs(
                    (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
                     for p, d in gap_first_keys))

                gap_rest_keys = [(p, h_curr, h_prev, d)
                                 for p in people for d in current_days
                                 for h_prev, h_curr in zip(hours[d][:-1], hours[d][1:])
                                 if (p, h_curr, d) in g_keys]
                model.addConstrs(
                    (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
                     for p, h_curr, h_prev, d in gap_rest_keys))

                model.addConstrs(
                    (r[p, d] >= gp.quicksum(g[p, h, d] for h in hours[d] if (p, h, d) in g_keys) - 1
                     for p, d in r_keys))

            # L. Per-person enforced rest: sliding window (sin cambios)
            for p_rest, limit in max_consec_hours.items():
                if p_rest not in people:
                    continue
                for d in current_days:
                    if len(hours[d]) <= limit:
                        continue
                    if not any((p_rest, h, d) in avail_set_in_day for h in hours[d]):
                        continue

                    model.addConstrs(
                        (x.sum(p_rest, '*', hours[d][i:i + limit + 1], d) <= limit
                         for i in range(len(hours[d]) - limit)))        # nombre mantenido: diagnóstico

            # ── Solve ─────────────────────────────────────────────────
            model._x     = x

            # Parámetros del solver ya extraídos al inicio
            for param_name, param_value in solver_params.items():
                try:
                    model.setParam(param_name, param_value)
                except Exception as err:
                    print(f"Warning: Could not set param {param_name}={param_value}: {err}")

            _hours_map = {d: list(hours[d]) for d in current_days}
            last_sent_assignment = [None]

            def intermediate_solution_callback(mdl, where):
                if where != GRB.Callback.MIPSOL or not ui_update_callback:
                    return
                now = time.monotonic()
                if now - last_ui_update[0] < 0.5:
                    return
                last_ui_update[0] = now
                try:
                    x_vals = mdl.cbGetSolution(mdl._x)
                    temp_assignment = {d: {p: {h: None for h in _hours_map.get(d, hours[d])}
                                           for p in people} for d in days}
                    for d in days:
                        if d not in _hours_map:
                            for p in people:
                                for h in hours[d]:
                                    temp_assignment[d][p][h] = partial_assignment[d][p][h]

                    for (p, t, h, d), val in x_vals.items():
                        if val > 0.5:
                            temp_assignment[d][p][h] = t

                    if last_sent_assignment[0] is not None:
                        total_changes = sum(
                            1 for d in days
                            for p in people
                            for h in _hours_map.get(d, hours[d])
                            if temp_assignment[d][p][h] != last_sent_assignment[0][d][p][h])
                        if total_changes == 0:
                            return

                    last_sent_assignment[0] = temp_assignment
                    ui_update_callback({"status":     "Solving (New Best Found)...",
                                        "assignment": temp_assignment})
                except Exception as exc:
                    logging.warning(f"Solver callback error (non-fatal): {exc}")

            model.optimize(intermediate_solution_callback)

            # ── Accumulate results ────────────────────────────────────
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

            all_x_vals.update({key: x[key].X for key in x_set})
            all_m_vals.update({key: m[key].X for key in m_keys})

            if W_FORCE > 0:                                    # ① acumular u
                all_u_vals.update({key: u[key].X for key in force_keys})

            for d in current_days:
                for p in people:
                    for h in hours[d]:
                        partial_assignment[d][p][h] = next(
                            (t for t in tasks
                             if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5), None)

            if day_heuristics == 1:
                d_cur = current_days[0]
                for p in people:
                    accumulated_hours[p] += sum(1 for h in hours[d_cur] for t in tasks
                                                if (p, t, h, d_cur) in x_set
                                                and x[p, t, h, d_cur].X > 0.5)

            interrupted = model.Status == GRB.INTERRUPTED

        finally:                                               # ⑤ dispose con guard
            try:
                model.dispose()
                env.dispose()
            except Exception:
                pass
            if active_model_ref is not None:
                active_model_ref[0] = None

        if interrupted:
            break

    # ═════════════════════════════════════════════════════════════════
    # RESULT EXTRACTION
    # ═════════════════════════════════════════════════════════════════
    solve_elapsed = time.monotonic() - solve_start

    sol = {"solve_time":    solve_elapsed,
           "status":        final_status,
           "mip_gap":       final_mip_gap,
           "enforced_rest": bool(max_consec_hours)}

    # Assignment grid
    assignment = partial_assignment
    sol["assignment"] = assignment

    # Uncovered demand
    sol["missing"] = [f"{t} @ {h}, {d}: {v:.0f} missing"
                      for (t, h, d), v in all_m_vals.items() if v > 0.01]

    # ① Force mandate issues
    if W_FORCE > 0 and all_u_vals:

        force_issues = []
        all_force_keys = [(p, t, h, d) for (p, t, h, d), v in force.items() if v == 1]

        for p, t, h, d in all_force_keys:
            u_val = all_u_vals.get((p, t, h, d), None)
            if u_val is None:
                continue

            if u_val > 0.5:
                feasible = (p, t, h, d) in all_x_vals
                reason   = "skill/availability mismatch" if not feasible else "not assigned"
                force_issues.append(
                    f"UNFULFILLED: {p} — task '{t}' @ {h}, {d} ({reason})")
            else:
                force_issues.append(f"FULFILLED: {p} — task '{t}' @ {h}, {d}")
                
        sol["force_issues"] = force_issues or ["No force mandates defined."]
    else:
        sol["force_issues"] = ["No force mandates defined."]

    # Workload per person
    assigned_counts = {p: 0 for p in people}
    for p, t, h, d in all_x_vals:
        if all_x_vals[p, t, h, d] > 0.5:
            assigned_counts[p] += 1
    sol["workload"] = assigned_counts
    wl = list(assigned_counts.values())
    sol["workload_max"] = float(max(wl)) if wl else 0.0
    sol["workload_min"] = float(min(wl)) if wl else 0.0

    # Quota fulfilment — solo si W_QUOTA > 0                  ⑧
    if W_QUOTA > 0:
        assigned_by_ptd = Counter()
        for (p, t, h, d), v in all_x_vals.items():
            if v > 0.5:
                assigned_by_ptd[(p, t, d)] += 1

        quota_keys_all = [(p, t, d)
                          for p in people for t in tasks for d in days
                          if min_quota.get((p, t, d), 0) > 0]
        sol["quota_issues"] = []
        for p, t, d in quota_keys_all:
            demanded = min_quota[p, t, d]
            assigned = assigned_by_ptd.get((p, t, d), 0)
            status   = "OK" if assigned >= demanded else "SHORTFALL"
            sol["quota_issues"].append(
                f"{status}: {p} — task '{t}' on {d}: {assigned}/{demanded} h assigned")
    else:
        sol["quota_issues"] = []

    # Work block fragmentation
    gaps = []
    for d in days:
        day_gaps = []
        for p in people:
            starts = 0
            start_hours = []
            prev_working = False
            for h in hours[d]:
                working = assignment[d][p][h] is not None
                if working and not prev_working:
                    starts += 1
                    start_hours.append(h)
                prev_working = working
            if starts > 1:
                day_gaps.append(
                    f"  • {p}: {starts} blocks (Starts: {', '.join(start_hours)})")
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # Social issues
    active_slots = {(p, t, h, d) for (p, t, h, d), v in all_x_vals.items() if v > 0.5}

    slots_by_person = defaultdict(set)
    for p, t, h, d in active_slots:
        slots_by_person[p].add((t, h, d))

    soc_issues = []
    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for t, h, d in slots_by_person.get(p1, set()):
            if (p2, t, h, d) in active_slots:
                soc_issues.append(f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    if any(sv == -1 for sv in social.values()):
        if not hard_enemies:
            violations = []
            for (p1, p2), sv in social.items():
                if sv != -1:
                    continue
                for t, h, d in slots_by_person.get(p1, set()):
                    if (p2, t, h, d) in active_slots:
                        violations.append(
                            f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}")
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