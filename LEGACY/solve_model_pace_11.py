import time
import logging
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict, Counter

import os
from pathlib import Path
os.environ["GRB_LICENSE_FILE"] = str(Path(__file__).parent / "gurobi.lic")


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
    solver_params    = data.get("solver_params", {})

    # Si live_callbacks == 0 desactivamos completamente el callback de UI
    if not data.get("live_callbacks", 1):
        ui_update_callback = None

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

    # ── Pacing setup (siempre activo) ────────────────────────────────
    total_demand = sum(demand.get((t, h, d), 0) for t in tasks for d in days for h in hours[d])

    total_available_people = [p for p in people if any((p, h, d) in avail_set for d in days for h in hours[d])]

    pace = total_demand / (len(total_available_people) or 1) / (len(days) or 1)

    accumulated_hours = {p: 0 for p in people}

    # Cada iteración resuelve exactamente un día
    days_iterator = [[d] for d in days]

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

    # ── Crear env UNA SOLA VEZ antes del bucle ────────────────────────
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1)
        env.start()
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi license error: {e}") from e

    # ═════════════════════════════════════════════════════════════════
    # MAIN LOOP — un modelo por día
    # ═════════════════════════════════════════════════════════════════
    try:
        for loop_idx, current_days in enumerate(days_iterator):

            avail_set_in_day = {(p, h, d) for (p, h, d) in avail_set if d in set(current_days)}

            people_available_in_day = [
                p for p in people
                if any((p, h, d) in avail_set_in_day for d in current_days for h in hours[d])
            ]

            h_next = {
                (hours[d][i], d): hours[d][i + 1]
                for d in current_days for i in range(len(hours[d]) - 1)
            }

            # ── Model (env ya existe, no se recrea) ───────────────────
            model = gp.Model("StaffScheduler", env=env)

            if active_model_ref is not None:
                if active_model_ref[0] is not None:
                    active_model_ref[0].terminate()
                active_model_ref[0] = model

            interrupted = False

            try:

                # ══════════════════════════════════════════════════════════
                # DECISION VARIABLES
                # ══════════════════════════════════════════════════════════

                x_set = {
                    (p, t, h, d)
                    for p in people for t in tasks
                    for d in current_days for h in hours[d]
                    if (p, t) in skill_set
                    and (p, h, d) in avail_set
                    and demand.get((t, h, d), 0) > 0
                }
                x = model.addVars(x_set, vtype=GRB.BINARY)

                m_keys = [
                    (t, h, d)
                    for t in tasks for d in current_days for h in hours[d]
                    if demand.get((t, h, d), 0) > 0
                ]
                m_ub = {(t, h, d): demand.get((t, h, d), 0) for t, h, d in m_keys}
                m = model.addVars(m_keys, lb=0, ub=m_ub, vtype=GRB.CONTINUOUS)

                if W_FORCE > 0:
                    force_keys = [
                        (p, t, h, d) for (p, t, h, d), v in force.items()
                        if v == 1 and d in set(current_days) and demand.get((t, h, d), 0) > 0
                    ]
                    force_possible   = [(p, t, h, d) for p, t, h, d in force_keys if (p, t, h, d) in x_set]
                    force_impossible = [(p, t, h, d) for p, t, h, d in force_keys if (p, t, h, d) not in x_set]
                    u = model.addVars(force_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_STABILITY > 0 and X_prev:
                    z = model.addVars(x_set, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_QUOTA > 0:
                    quota_keys = [
                        (p, t, d)
                        for p in people for t in tasks for d in current_days
                        if min_quota.get((p, t, d), 0) > 0
                    ]
                    q_ub = {(p, t, d): min_quota.get((p, t, d), 0) for p, t, d in quota_keys}
                    q = model.addVars(quota_keys, lb=0, ub=q_ub, vtype=GRB.CONTINUOUS)

                if W_GAP > 0:
                    g_keys = {(p, h, d) for p, _, h, d in x_set}
                    g = model.addVars(g_keys, lb=0, vtype=GRB.CONTINUOUS)
                    r_keys = [
                        (p, d) for p in people for d in current_days
                        if any((p, h, d) in g_keys for h in hours[d])
                    ]
                    r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS)

                max_hours_any_day = max((len(hours[d]) for d in current_days), default=0)

                if W_EQ_DAY > 0:
                    j_max  = model.addVars(current_days, lb=0, ub=max_hours_any_day, vtype=GRB.CONTINUOUS)
                    spread = model.addVars(current_days, lb=0, vtype=GRB.CONTINUOUS)

                # Equidad global: siempre por desviación respecto al pace
                if W_EQ_GLOBAL > 0:
                    delta_plus  = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)
                    delta_minus = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)

                if W_ROTATION > 0:
                    consec_keys = [
                        (p, t, h, d)
                        for p in people for t in tasks
                        for d in current_days for h in hours[d][:-1]
                        if rotation.get(t, 0) == 1
                        and (p, t, h,              d) in x_set
                        and (p, t, h_next[(h, d)], d) in x_set
                    ]
                    c = model.addVars(consec_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                enemies_scope = [
                    (p1, p2, t, h, d)
                    for (p1, p2), sv in social.items() if sv == -1
                    for t in tasks for d in current_days for h in hours[d]
                    if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set
                ]

                if W_SOCIAL > 0:
                    friends_keys = [
                        (p1, p2, t, h, d)
                        for (p1, p2), sv in social.items() if sv == 1
                        for t in tasks for d in current_days for h in hours[d]
                        if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set
                    ]
                    f = model.addVars(friends_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)
                    enemies_keys = [] if hard_enemies else enemies_scope
                    e = model.addVars(enemies_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_CAPTAIN > 0:
                    active_hd = (
                        [(h, d) for d in current_days for h in hours[d]
                         if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
                        if captains else []
                    )
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

                # Cobertura de demanda
                model.addConstrs(
                    (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
                     for t, h, d in m_keys))

                # Force mandates
                if W_FORCE > 0:
                    model.addConstrs(
                        (1 - x[p, t, h, d] <= u[p, t, h, d]
                         for p, t, h, d in force_possible))
                    model.addConstrs(
                        (u[p, t, h, d] >= 1
                         for p, t, h, d in force_impossible))

                # Captain presence
                if W_CAPTAIN > 0:
                    model.addConstrs(
                        (x.sum(captains, '*', h, d) + k[h, d] >= 1
                         for h, d in active_hd))

                # Una tarea por persona por hora
                model.addConstrs(
                    (x.sum(p, '*', h, d) <= 1 for p, h, d in avail_set_in_day))

                # Equidad intra-día
                if W_EQ_DAY > 0:
                    eq_day_keys = [
                        (p, d) for d in current_days for p in people
                        if any((p, h, d) in avail_set_in_day for h in hours[d])
                    ]
                    model.addConstrs(
                        (x.sum(p, '*', '*', d) <= j_max[d] for p, d in eq_day_keys))
                    model.addConstrs(
                        (j_max[d] - x.sum(p, '*', '*', d) <= spread[d] for p, d in eq_day_keys))

                # Equidad global via pace acumulado
                if W_EQ_GLOBAL > 0:
                    pace_target = pace * (loop_idx + 1)
                    model.addConstrs(
                        (accumulated_hours[p] + x.sum(p, '*', '*', current_days[0])
                         - delta_plus[p] + delta_minus[p] == pace_target
                         for p in people_available_in_day))

                # Rotación (no consecutivo)
                if W_ROTATION > 0:
                    model.addConstrs(
                        (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
                         for p, t, h, d in consec_keys))

                # Social: amigos y enemigos
                if W_SOCIAL > 0:
                    model.addConstrs(
                        (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
                         for p1, p2, t, h, d in friends_keys))
                    model.addConstrs(
                        (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
                         for p1, p2, t, h, d in friends_keys))
                    if not hard_enemies:
                        model.addConstrs(
                            (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
                             for p1, p2, t, h, d in enemies_keys))

                if hard_enemies:
                    model.addConstrs(
                        (x[p1, t, h, d] + x[p2, t, h, d] <= 1
                         for p1, p2, t, h, d in enemies_scope))

                # Cuota mínima
                if W_QUOTA > 0:
                    quota_constr_keys = []
                    for p, t, d in quota_keys:
                        target = min(min_quota.get((p, t, d), 0), len(hours[d]))
                        if target > 0:
                            quota_constr_keys.append((p, t, d, target))
                    model.addConstrs(
                        (x.sum(p, t, '*', d) + q[p, t, d] >= target
                         for p, t, d, target in quota_constr_keys))

                # Estabilidad respecto a solución anterior
                if W_STABILITY > 0 and X_prev:
                    forced_slots = {
                        (p, h, d)
                        for (p, t, h, d), v in force.items()
                        if v == 1 and d in set(current_days)
                    }
                    model.addConstrs(
                        (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
                         for p, t, h, d in x_set if (p, h, d) not in forced_slots))
                    model.addConstrs(
                        (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
                         for p, t, h, d in x_set if (p, h, d) not in forced_slots))

                # Fragmentación de turno (gap)
                if W_GAP > 0:
                    gap_first_keys = [
                        (p, d) for p in people for d in current_days
                        if (p, hours[d][0], d) in g_keys
                    ]
                    model.addConstrs(
                        (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
                         for p, d in gap_first_keys))

                    gap_rest_keys = [
                        (p, h_curr, h_prev, d)
                        for p in people for d in current_days
                        for h_prev, h_curr in zip(hours[d][:-1], hours[d][1:])
                        if (p, h_curr, d) in g_keys
                    ]
                    model.addConstrs(
                        (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
                         for p, h_curr, h_prev, d in gap_rest_keys))
                    model.addConstrs(
                        (r[p, d] >= gp.quicksum(
                            g[p, h, d] for h in hours[d] if (p, h, d) in g_keys) - 1
                         for p, d in r_keys))

                # Horas consecutivas máximas
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
                             for i in range(len(hours[d]) - limit)))

                # ── Solve ─────────────────────────────────────────────────
                model._x = x

                for param_name, param_value in solver_params.items():
                    try:
                        model.setParam(param_name, param_value)
                    except Exception as err:
                        print(f"Warning: Could not set param {param_name}={param_value}: {err}")

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
                            # Clonamos días ya resueltos y sobreescribimos solo el día actual
                            temp_assignment = {
                                d: {p: dict(partial_assignment[d][p]) for p in people}
                                for d in days
                            }
                            for p in people:
                                for h in hours[current_days[0]]:
                                    temp_assignment[current_days[0]][p][h] = None
                            for (p, t, h, d), val in x_vals.items():
                                if val > 0.5:
                                    temp_assignment[d][p][h] = t
                            ui_update_callback({"status": "Solving (New Best Found)...",
                                                "assignment": temp_assignment})
                        except Exception as exc:
                            logging.warning(f"Solver callback error (non-fatal): {exc}")
                    model.optimize(intermediate_solution_callback)
                else:
                    model.optimize()

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

                if W_FORCE > 0:
                    all_u_vals.update({key: u[key].X for key in force_keys})

                for d in current_days:
                    for p in people:
                        for h in hours[d]:
                            partial_assignment[d][p][h] = next(
                                (t for t in tasks if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5),
                                None)

                # Actualizar acumulador de horas para la siguiente iteración
                d_cur = current_days[0]
                for p in people:
                    accumulated_hours[p] += sum(
                        1 for h in hours[d_cur] for t in tasks
                        if (p, t, h, d_cur) in x_set and x[p, t, h, d_cur].X > 0.5)

                interrupted = model.Status == GRB.INTERRUPTED

            finally:
                try:
                    model.dispose()   # solo el modelo; el env se reutiliza
                except Exception:
                    pass
                if active_model_ref is not None:
                    active_model_ref[0] = None

            if interrupted:
                break

    finally:
        # El env se destruye una sola vez al terminar el bucle completo
        try:
            env.dispose()
        except Exception:
            pass

    # ═════════════════════════════════════════════════════════════════
    # RESULT EXTRACTION
    # ═════════════════════════════════════════════════════════════════
    solve_elapsed = time.monotonic() - solve_start

    sol = {
        "solve_time":    solve_elapsed,
        "status":        final_status,
        "mip_gap":       final_mip_gap,
        "enforced_rest": bool(max_consec_hours),
    }

    assignment = partial_assignment
    sol["assignment"] = assignment

    sol["missing"] = [
        f"{t} @ {h}, {d}: {v:.0f} missing"
        for (t, h, d), v in all_m_vals.items() if v > 0.01
    ]

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
                force_issues.append(f"UNFULFILLED: {p} — task '{t}' @ {h}, {d} ({reason})")
            else:
                force_issues.append(f"FULFILLED: {p} — task '{t}' @ {h}, {d}")
        sol["force_issues"] = force_issues or ["No force mandates defined."]
    else:
        sol["force_issues"] = ["No force mandates defined."]

    assigned_counts = {p: 0 for p in people}
    for p, t, h, d in all_x_vals:
        if all_x_vals[p, t, h, d] > 0.5:
            assigned_counts[p] += 1
    sol["workload"] = assigned_counts
    wl = list(assigned_counts.values())
    sol["workload_max"] = float(max(wl)) if wl else 0.0
    sol["workload_min"] = float(min(wl)) if wl else 0.0

    if W_QUOTA > 0:
        assigned_by_ptd = Counter()
        for (p, t, h, d), v in all_x_vals.items():
            if v > 0.5:
                assigned_by_ptd[(p, t, d)] += 1
        quota_keys_all = [
            (p, t, d)
            for p in people for t in tasks for d in days
            if min_quota.get((p, t, d), 0) > 0
        ]
        sol["quota_issues"] = []
        for p, t, d in quota_keys_all:
            demanded = min_quota[p, t, d]
            assigned = assigned_by_ptd.get((p, t, d), 0)
            status   = "OK" if assigned >= demanded else "SHORTFALL"
            sol["quota_issues"].append(
                f"{status}: {p} — task '{t}' on {d}: {assigned}/{demanded} h assigned")
    else:
        sol["quota_issues"] = []

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
            violations = [
                f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                for (p1, p2), sv in social.items() if sv == -1
                for t, h, d in slots_by_person.get(p1, set())
                if (p2, t, h, d) in active_slots
            ]
        else:
            violations = []
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
    sol["social_issues"] = soc_issues

    all_active_hd = (
        [(h, d) for d in days for h in hours[d]
         if sum(demand.get((t, h, d), 0) for t in tasks) > 0]
        if captains else []
    )
    missing_caps = [
        f"MISSING CAPTAIN @ {h}, {d}"
        for h, d in all_active_hd
        if not any((cap, t, h, d) in active_slots for cap in captains for t in tasks)
    ]
    sol["captain_issues"] = (
        missing_caps or ["SUCCESS: All active hours have at least one captain on duty."]
        if all_active_hd else []
    )

    sol["emerg_issues"] = [
        f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {d}"
        for p, t, h, d in active_slots
        if emergency.get((p, h, d), 0) == 1
    ]

    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)
    }
    rot_violations = [
        f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next_all[(h, d)]}, {d}"
        for p, t, h, d in active_slots
        if rotation.get(t, 0) == 1
        and (h, d) in h_next_all
        and (p, t, h_next_all[(h, d)], d) in active_slots
    ]
    sol["rotation_issues"] = (rot_violations or ["SUCCESS: No consecutive hours on rotation tasks."])

    return sol