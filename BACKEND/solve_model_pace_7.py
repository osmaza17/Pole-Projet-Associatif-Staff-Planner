import time
import logging
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Any, Optional
from collections import defaultdict

def solve_model(data: Dict[str, Any], ui_update_callback: Optional[callable] = None, 
                active_model_ref: Optional[List] = None) -> Dict[str, Any]:



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
        total_demand             = sum(demand.get((t, h, d), 0) 
                                       for t in tasks for d in days for h in hours[d])
        people_available_global  = [p for p in people
                                    if any((p, h, d) in avail_set
                                           for d in days for h in hours[d])]
        pace                     = total_demand / (len(people_available_global) or 1) / (len(days) or 1)
        accumulated_hours        = {p: 0 for p in people}
        days_iterator            = [[d] for d in days]
    else:
        days_iterator = [days]

    # ── Result accumulators ──────────────────────────────────────────
    all_x_vals = {}
    all_m_vals = {}
    all_g_vals = {}

    partial_assignment = {d: {p: {h: None for h in hours[d]} for p in people} for d in days}

    final_status  = "Optimal"
    final_mip_gap = 0.0
    last_ui_update = [0.0]

    status_map = {GRB.OPTIMAL:     "Optimal",
                  GRB.TIME_LIMIT:  "Time Limit Reached",
                  GRB.INFEASIBLE:  "Infeasible",
                  GRB.INTERRUPTED: "Interrupted by User"}

    solve_start = time.monotonic()

    # ═════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════════════════════════════
    for loop_idx, current_days in enumerate(days_iterator):

        current_days_set = set(current_days)
        avail_set_day    = {(p, h, d) for (p, h, d) in avail_set
                            if d in current_days_set}
        people_available = [p for p in people
                            if any((p, h, d) in avail_set_day
                                   for d in current_days for h in hours[d])]

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

        # [FIX 2] Initialize interrupted before try so it's always defined
        #         even if an exception occurs before model.optimize().
        interrupted = False

        # [FIX 1] Wrap everything in try/finally so model & env are always freed.
        try:

            # ── Decision variables ────────────────────────────────────

            # [FIX 6] Exclude zero-demand slots directly in x_set — eliminates
            #         the zero_demand_keys variables AND their associated constraints.
            # [FIX 5] No name= argument → avoids millions of string allocations.
            x_set = {(p, t, h, d)
                     for p in people for t in tasks
                     for d in current_days for h in hours[d]
                     if (p, t) in skill_set
                     and (p, h, d) in avail_set
                     and demand.get((t, h, d), 0) > 0}          # ← [FIX 6]
            x = model.addVars(x_set, vtype=GRB.BINARY)          # ← [FIX 5]

            # [FIX 4] Explicit UB for m: missing demand ≤ slot demand
            m_keys = [(t, h, d)
                      for t in tasks for d in current_days for h in hours[d]
                      if demand.get((t, h, d), 0) > 0]
            m_ub   = {(t, h, d): demand.get((t, h, d), 0) for t, h, d in m_keys}
            m = model.addVars(m_keys, lb=0, ub=m_ub, vtype=GRB.CONTINUOUS)   # ← [FIX 4,5]

            # [FIX 4-FORCE] Only create force keys where demand > 0.
            #               If demand is 0, the force is incoherent with the data
            #               and should not penalise the solver.
            force_keys = [(p, t, h, d) for (p, t, h, d), v in force.items()
                          if v == 1
                          and d in current_days_set
                          and demand.get((t, h, d), 0) > 0]     # ← [FIX 4]
            # [FIX 4] Explicit UB for u: unfulfilled force ∈ [0, 1]
            u = model.addVars(force_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)  # ← [FIX 4,5]

            if W_STABILITY > 0 and X_prev:
                z = model.addVars(x_set, lb=0, ub=1, vtype=GRB.CONTINUOUS)             # ← [FIX 5]

            # [FIX 1] Replace walrus operator with explicit loop for clarity
            #         and to avoid scope-leaking issues across Python versions.
            quota_keys = []
            for p in people:
                for t in tasks:
                    for d in current_days:
                        raw_quota = min_quota.get((p, t, d), 0)
                        if raw_quota > 0:
                            quota_keys.append((p, t, d))

            # [FIX 4] Explicit UB for q: shortfall ≤ the quota target itself
            q_ub = {(p, t, d): min_quota.get((p, t, d), 0) for p, t, d in quota_keys}
            q = model.addVars(quota_keys, lb=0, ub=q_ub, vtype=GRB.CONTINUOUS)  # ← [FIX 4,5]

            # [FIX 8] g_keys: projection of x_set dropping the task dimension.
            #         Only create gap-detection variables where the person can
            #         actually be assigned to at least one task in that slot.
            #         This avoids thousands of phantom variables and constraints
            #         in slots where the person has no viable assignment.
            g_keys = {(p, h, d) for p, _, h, d in x_set}
            g = model.addVars(g_keys, lb=0, vtype=GRB.CONTINUOUS)

            r_keys = [(p, d) for p in people for d in current_days
                      if any((p, h, d) in g_keys for h in hours[d])]   # ← [FIX 8] use g_keys
            r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS)            # ← [FIX 5]

            max_hours_any_day = max((len(hours[d]) for d in current_days), default=0)

            # [FIX 9] Only create equity variables/constraints when their weight > 0
            if W_EQ_DAY > 0:
                j_max = model.addVars(current_days, lb=0, ub=max_hours_any_day,
                                      vtype=GRB.CONTINUOUS)
                j_min = model.addVars(current_days, lb=0, ub=max_hours_any_day,
                                      vtype=GRB.CONTINUOUS)

            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
                    w_max = model.addVar(lb=0, ub=max_hours_any_day * len(current_days),
                                         vtype=GRB.CONTINUOUS)
                    w_min = model.addVar(lb=0, ub=max_hours_any_day * len(current_days),
                                         vtype=GRB.CONTINUOUS)
                else:
                    delta_plus  = model.addVars(people_available, lb=0, vtype=GRB.CONTINUOUS)
                    delta_minus = model.addVars(people_available, lb=0, vtype=GRB.CONTINUOUS)

            # [FIX 9] Only create rotation variables when weight > 0
            if W_ROTATION > 0:
                consec_keys = [(p, t, h, d)
                               for p in people for t in tasks
                               for d in current_days for h in hours[d][:-1]
                               if rotation.get(t, 0) == 1
                               and (p, t, h,              d) in x_set
                               and (p, t, h_next[(h, d)], d) in x_set]
                c = model.addVars(consec_keys, lb=0, vtype=GRB.CONTINUOUS)
            else:
                consec_keys = []

            # [FIX 9] Only create social variables when weight > 0
            if W_SOCIAL > 0:
                friends_keys = [(p1, p2, t, h, d)
                                for (p1, p2), sv in social.items() if sv == 1
                                for t in tasks for d in current_days for h in hours[d]
                                if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]
                f = model.addVars(friends_keys, lb=0, vtype=GRB.CONTINUOUS)

                enemies_scope = [(p1, p2, t, h, d)
                                 for (p1, p2), sv in social.items() if sv == -1
                                 for t in tasks for d in current_days for h in hours[d]
                                 if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]
                enemies_keys = [] if hard_enemies else enemies_scope
                e = model.addVars(enemies_keys, lb=0, vtype=GRB.CONTINUOUS)
            else:
                friends_keys = []
                enemies_scope = []
                enemies_keys = []

            # [FIX 9] Only create captain variables when weight > 0
            if W_CAPTAIN > 0:
                active_hd = ([(h, d) for d in current_days for h in hours[d]
                              if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
                             if captains else [])
                k = model.addVars(active_hd, lb=0, vtype=GRB.CONTINUOUS)
            else:
                active_hd = []

            # ── Objective ─────────────────────────────────────────────
            obj = gp.LinExpr()
            obj += W_COVERAGE  * m.sum()
            obj += W_FORCE     * u.sum()

            if W_CAPTAIN > 0:
                obj += W_CAPTAIN * k.sum()

            obj += W_EMERG * gp.quicksum(
                emergency.get((p, h, d), 0) * x[p, t, h, d]
                for p, t, h, d in x_set)

            if W_STABILITY > 0 and X_prev:
                obj += W_STABILITY * z.sum()

            if W_EQ_DAY > 0:
                obj += W_EQ_DAY * (j_max.sum() - j_min.sum())

            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
                    obj += W_EQ_GLOBAL * (w_max - w_min)
                else:
                    obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

            if W_ROTATION > 0:
                obj += W_ROTATION * c.sum()

            if W_SOCIAL > 0:
                obj += W_SOCIAL * f.sum()
                obj += W_SOCIAL * e.sum()

            if W_GAP > 0:
                obj += W_GAP * r.sum()

            obj += W_QUOTA * q.sum()
            obj += W_PREF * gp.quicksum(
                pref_cost.get((p, t), 0) * x[p, t, h, d]
                for p, t, h, d in x_set)

            model.setObjective(obj, GRB.MINIMIZE)

            # ── Constraints ───────────────────────────────────────────
            # [FIX 11] All constraints are named so that IIS diagnostics
            #          can report human-readable conflict sources.

            # A. Demand coverage
            model.addConstrs(
                (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
                 for t, h, d in m_keys),
                name="demand")

            # B. Force mandates (soft) — only where demand > 0 and assignment is possible
            model.addConstrs(
                (1 - x[p, t, h, d] <= u[p, t, h, d]
                 for p, t, h, d in force_keys if (p, t, h, d) in x_set),
                name="force_possible")
            # [FIX 4] Removed the penalty for force keys outside x_set.
            #         If (p,t,h,d) is not in x_set it means the person lacks skill
            #         or availability — this is a data inconsistency, not a solver failure.

            # C. Captain presence
            if W_CAPTAIN > 0:
                model.addConstrs(
                    (x.sum(captains, '*', h, d) + k[h, d] >= 1
                     for h, d in active_hd),
                    name="captain")

            # D. Anti-ubiquity
            model.addConstrs(
                (x.sum(p, '*', h, d) <= 1 for p, h, d in avail_set_day),
                name="anti_ubiquity")

            # F.1. Daily equity bounds
            if W_EQ_DAY > 0:
                eq_day_keys = [(p, d) for d in current_days for p in people
                               if any((p, h, d) in avail_set_day for h in hours[d])]
                model.addConstrs(
                    (x.sum(p, '*', '*', d) <= j_max[d] for p, d in eq_day_keys),
                    name="eq_day_max")
                model.addConstrs(
                    (x.sum(p, '*', '*', d) >= j_min[d] for p, d in eq_day_keys),
                    name="eq_day_min")

            # F.2. Global equity
            if W_EQ_GLOBAL > 0:
                if day_heuristics == 0:
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
                         for p in people_available),
                        name="eq_pace")

            # G. Rotation fatigue
            if W_ROTATION > 0:
                model.addConstrs(
                    (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
                     for p, t, h, d in consec_keys),
                    name="rotation")

            # H. Social — friends
            if W_SOCIAL > 0:
                model.addConstrs(
                    (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
                     for p1, p2, t, h, d in friends_keys),
                    name="friends_a")
                model.addConstrs(
                    (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
                     for p1, p2, t, h, d in friends_keys),
                    name="friends_b")

                # H. Social — enemies
                if hard_enemies:
                    model.addConstrs(
                        (x[p1, t, h, d] + x[p2, t, h, d] <= 1
                         for p1, p2, t, h, d in enemies_scope),
                        name="enemies_hard")
                else:
                    model.addConstrs(
                        (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
                         for p1, p2, t, h, d in enemies_keys),
                        name="enemies_soft")

            # I. Minimum quota
            # [FIX 1] Explicit loop instead of walrus operator — clearer scope,
            #         no risk of leaking 'target' into enclosing scope.
            quota_constr_keys = []
            for p, t, d in quota_keys:
                target = min(min_quota.get((p, t, d), 0), len(hours[d]))
                if target > 0:
                    quota_constr_keys.append((p, t, d, target))

            model.addConstrs(
                (x.sum(p, t, '*', d) + q[p, t, d] >= target
                 for p, t, d, target in quota_constr_keys),
                name="quota")

            # J. Stability
            if W_STABILITY > 0 and X_prev:
                model.addConstrs(
                    (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
                    for p, t, h, d in x_set),
                    name="stability_a")
                model.addConstrs(
                    (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
                    for p, t, h, d in x_set),
                    name="stability_b")

            # K.1. Work block restart detection
            # [FIX 8] Iterate over g_keys instead of avail_set_day
            gap_first_keys = [(p, d) for p in people for d in current_days
                              if (p, hours[d][0], d) in g_keys]
            model.addConstrs(
                (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
                 for p, d in gap_first_keys),
                name="gap_first")

            # [FIX 8] Filter gap_rest_keys against g_keys
            gap_rest_keys = [(p, h_curr, h_prev, d)
                             for p in people for d in current_days
                             for h_prev, h_curr in zip(hours[d][:-1], hours[d][1:])
                             if (p, h_curr, d) in g_keys]
            model.addConstrs(
                (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
                 for p, h_curr, h_prev, d in gap_rest_keys),
                name="gap_rest")

            # K.2. Excess work blocks
            # [FIX 8] Filter the g.sum() list against g_keys
            if W_GAP > 0:
                model.addConstrs(
                    (r[p, d] >= g.sum(p, [h for h in hours[d]
                                          if (p, h, d) in g_keys], d) - 1
                     for p, d in r_keys),
                    name="gap_excess")

            # L. Per-person enforced rest: sliding window
            for p_rest, limit in max_consec_hours.items():
                if p_rest not in people:
                    continue
                for d in current_days:
                    if len(hours[d]) <= limit:
                        continue
                    if not any((p_rest, h, d) in avail_set_day for h in hours[d]):
                        continue
                    
                    model.addConstrs(
                        (x.sum(p_rest, '*', hours[d][i:i + limit + 1], d) <= limit
                         for i in range(len(hours[d]) - limit)),
                        name=f"rest_{p_rest}_{d}")

            # ── Solve ─────────────────────────────────────────────────
            model._x     = x
            model._x_set = x_set

            for param_name, param_value in data.get("solver_params", {}).items():
                try:
                    model.setParam(param_name, param_value)
                except Exception as err:
                    print(f"Warning: Could not set param {param_name}={param_value}: {err}")

            # [FIX 12] Capture loop-local values explicitly so the callback
            #          doesn't depend on closure references that could change
            #          between loop iterations. Also log errors instead of
            #          silencing them with bare except.
            # [FIX 12b] Performance: iterate over the solution (x_vals) once
            #           instead of looping over all dimensions × tasks.
            #           Old approach: O(days × people × hours × tasks) per callback.
            #           New approach: O(|x_set|) single pass, writing only active vars.
            _hours_map = {d: list(hours[d]) for d in current_days}

            def intermediate_solution_callback(mdl, where):
                if where != GRB.Callback.MIPSOL or not ui_update_callback:
                    return
                now = time.monotonic()
                if now - last_ui_update[0] < 0.5:
                    return
                last_ui_update[0] = now
                try:
                    x_vals = mdl.cbGetSolution(mdl._x)
                    # Start from the partial_assignment snapshot (previous days)
                    # and fill current days with None, then overwrite from solution.
                    temp_assignment = {d: {p: {h: None for h in _hours_map.get(d, hours[d])}
                                           for p in people} for d in days}
                    # Copy already-solved days from partial_assignment
                    for d in days:
                        if d not in _hours_map:
                            for p in people:
                                for h in hours[d]:
                                    temp_assignment[d][p][h] = partial_assignment[d][p][h]

                    # Single pass over the solution — only write where val > 0.5
                    for (p, t, h, d), val in x_vals.items():
                        if val > 0.5:
                            temp_assignment[d][p][h] = t

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

            final_mip_gap = model.MIPGap

            all_x_vals.update({key: x[key].X for key in x_set})
            all_m_vals.update({key: m[key].X for key in m_keys})
            all_g_vals.update({key: g[key].X for key in g_keys})   # ← [FIX 8] iterate g_keys

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

            # [FIX 2] This assignment is now safe: if an exception occurred above,
            #         interrupted stays False (its initialised value) and the
            #         original exception propagates cleanly.
            interrupted = model.Status == GRB.INTERRUPTED

        finally:
            # [FIX 1] Always release Gurobi model and environment to prevent
            #         memory leaks and license token exhaustion in long-running apps.
            model.dispose()
            env.dispose()

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
    sol["workload_max"] = float(max(wl)) if wl else 0.0
    sol["workload_min"] = float(min(wl)) if wl else 0.0

    # Quota fulfilment
    # [FIX 6-style] Pre-aggregate assigned counts by (p,t,d) to avoid O(n²)
    from collections import Counter
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
                            if (p, h, d) in g_keys and all_g_vals.get((p, h, d), 0) > 0.5),
                        [h for h in hours[d]
                         if (p, h, d) in g_keys and all_g_vals.get((p, h, d), 0) > 0.5])]
                    if starts > 1]
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # [FIX 6/7] Social issues — index active_slots by person to avoid O(n²)
    active_slots = {(p, t, h, d) for (p, t, h, d), v in all_x_vals.items() if v > 0.5}

    slots_by_person = defaultdict(set)
    for p, t, h, d in active_slots:
        slots_by_person[p].add((t, h, d))

    soc_issues = []

    # [FIX 6] Friends: iterate the smaller set (person's slots) instead of all active_slots
    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for t, h, d in slots_by_person.get(p1, set()):
            if (p2, t, h, d) in active_slots:
                soc_issues.append(f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    # [FIX 7] Enemies: same indexed approach
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