import os
from pathlib import Path
os.environ["GRB_LICENSE_FILE"] = str(Path(__file__).parent / "gurobi.lic")

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
    solver_params    = data.get("solver_params", {})

    # ── Groups ────────────────────────────────────────────────────────
    # data["groups"]: {group_id: [list of people]}
    # Every person must belong to exactly one group.
    raw_groups = data.get("groups", {})
    if not raw_groups:
        # Fallback: everyone in one implicit group
        raw_groups = {"default": list(people)}

    group_list       = list(raw_groups.keys())
    n_groups         = len(group_list)
    group_people     = {g: raw_groups[g] for g in group_list}          # g -> [people]
    group_people_set = {g: set(raw_groups[g]) for g in group_list}     # g -> {people}  O(1) lookup
    person_to_group  = {p: g for g, members in group_people.items() for p in members}

    if not data.get("live_callbacks", 1):
        ui_update_callback = None

    W = data["weights"]
    W_COVERAGE  = W["W_COVERAGE"];  W_FORCE     = W["W_FORCE"]
    W_CAPTAIN   = W["W_CAPTAIN"];   W_EMERG     = W["W_EMERG"]
    W_STABILITY = W["W_STABILITY"]; W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_GLOBAL = W["W_EQ_GLOBAL"]; W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];     W_PREF      = W["W_PREF"]
    # New weight: inter-group equity
    W_EQ_GROUP  = W.get("W_EQ_GROUP", 0)

    # ── Global lookups ────────────────────────────────────────────────
    avail_set = {(p, h, d) for (p, h, d), v in availability.items() if v == 1}
    skill_set = {(p, t)    for (p, t),    v in skills.items()        if v == 1}

    # ── Static precomputations ────────────────────────────────────────

    people_per_task = defaultdict(list)
    for p, t in skill_set:
        people_per_task[t].append(p)

    demand_pos_by_day = defaultdict(list)
    for (t, h, d), v in demand.items():
        if v > 0:
            demand_pos_by_day[d].append((t, h, d))

    avail_by_day = defaultdict(set)
    for p, h, d in avail_set:
        avail_by_day[d].add((p, h, d))

    force_by_day = defaultdict(list)
    for (p, t, h, d), v in force.items():
        if v == 1:
            force_by_day[d].append((p, t, h, d))

    rotation_tasks = {t for t in tasks if rotation.get(t, 0) == 1}
    social_friends = [(p1, p2) for (p1, p2), sv in social.items() if sv == 1]
    social_enemies = [(p1, p2) for (p1, p2), sv in social.items() if sv == -1]
    has_enemies    = bool(social_enemies)

    emerg_set    = {(p, h, d) for (p, h, d), v in emergency.items() if v == 1}
    pref_nonzero = {(p, t): v for (p, t), v in pref_cost.items() if v != 0}

    active_hd_by_day = {
        d: [(h, d) for h in hours[d]
            if any(demand.get((t, h, d), 0) > 0 for t in tasks)]
        for d in days
    }

    # ── Pacing ────────────────────────────────────────────────────────
    total_demand = sum(demand[k] for d in days for k in demand_pos_by_day[d])
    n_days       = max(len(days), 1)

    # Per-person pace within each group (drives W_EQ_GLOBAL, now intra-group).
    # Each group gets total_demand / n_groups hours total.
    # Within the group those hours are split equally among its members.
    group_person_pace = {
        g: total_demand / n_groups / max(len(group_people[g]), 1) / n_days
        for g in group_list
    }

    # Total pace per group (drives W_EQ_GROUP, inter-group).
    inter_group_pace = total_demand / n_groups / n_days

    accumulated_hours = {p: 0 for p in people}   # updated after each day

    days_iterator = [[d] for d in days]

    # ── Result accumulators ───────────────────────────────────────────
    all_x_vals = {}
    all_m_vals = {}
    all_u_vals = {}

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
        GRB.INF_OR_UNBD: "Infeasible or Unbounded",
    }

    solve_start = time.monotonic()

    # ── Gurobi env (reused across day-models) ─────────────────────────
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1)
        env.start()
    except gp.GurobiError as e:
        raise RuntimeError(f"Gurobi license error: {e}") from e

    # ═════════════════════════════════════════════════════════════════
    # MAIN LOOP — one model per day
    # ═════════════════════════════════════════════════════════════════
    try:
        for loop_idx, current_days in enumerate(days_iterator):

            current_day      = current_days[0]
            current_days_set = set(current_days)
            avail_set_in_day = avail_by_day[current_day]

            people_available_in_day = list({p for p, h, d in avail_set_in_day})

            h_next = {
                (hours[d][i], d): hours[d][i + 1]
                for d in current_days for i in range(len(hours[d]) - 1)
            }

            model = gp.Model("StaffScheduler", env=env)

            if active_model_ref is not None:
                if active_model_ref[0] is not None:
                    active_model_ref[0].terminate()
                active_model_ref[0] = model

            interrupted = False

            try:

                # ══════════════════════════════════════════════════════
                # DECISION VARIABLES
                # ══════════════════════════════════════════════════════

                x_set = {
                    (p, t, h, current_day)
                    for t, h, _ in demand_pos_by_day[current_day]
                    for p in people_per_task[t]
                    if (p, h, current_day) in avail_set
                }
                x = model.addVars(x_set, vtype=GRB.BINARY)

                m_keys = demand_pos_by_day[current_day]
                m_ub   = {(t, h, d): demand[(t, h, d)] for t, h, d in m_keys}
                m      = model.addVars(m_keys, lb=0, ub=m_ub, vtype=GRB.CONTINUOUS)

                if W_FORCE > 0:
                    force_keys = [
                        (p, t, h, d) for p, t, h, d in force_by_day[current_day]
                        if demand.get((t, h, d), 0) > 0
                    ]
                    force_possible, force_impossible = [], []
                    for key in force_keys:
                        (force_possible if key in x_set else force_impossible).append(key)
                    u = model.addVars(force_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_STABILITY > 0 and X_prev:
                    z = model.addVars(x_set, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_QUOTA > 0:
                    quota_keys = [
                        (p, t, d)
                        for p in people for t in tasks for d in current_days
                        if min_quota.get((p, t, d), 0) > 0
                    ]
                    q_ub = {(p, t, d): min_quota[(p, t, d)] for p, t, d in quota_keys}
                    q    = model.addVars(quota_keys, lb=0, ub=q_ub, vtype=GRB.CONTINUOUS)

                if W_GAP > 0:
                    g_keys = {(p, h, d) for p, _, h, d in x_set}
                    g      = model.addVars(g_keys, lb=0, vtype=GRB.CONTINUOUS)
                    r_keys = [
                        (p, d) for p in people for d in current_days
                        if any((p, h, d) in g_keys for h in hours[d])
                    ]
                    r = model.addVars(r_keys, lb=0, vtype=GRB.CONTINUOUS)

                if W_EQ_DAY > 0:
                    j_max  = model.addVars(current_days, lb=0, ub=len(hours[current_day]),
                                           vtype=GRB.CONTINUOUS)
                    spread = model.addVars(current_days, lb=0, vtype=GRB.CONTINUOUS)

                if W_EQ_GLOBAL > 0:
                    # delta variables per person (intra-group equity)
                    delta_plus  = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)
                    delta_minus = model.addVars(people_available_in_day, lb=0, vtype=GRB.CONTINUOUS)

                if W_EQ_GROUP > 0:
                    # gamma variables per group (inter-group equity)
                    gamma_plus  = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                    gamma_minus = model.addVars(group_list, lb=0, vtype=GRB.CONTINUOUS)
                    # Group accumulated hours up to (but not including) current day
                    group_accum = {
                        g: sum(accumulated_hours[p] for p in group_people[g])
                        for g in group_list
                    }

                if W_ROTATION > 0:
                    consec_keys = [
                        (p, t, h, current_day)
                        for t in rotation_tasks
                        for p in people_per_task[t]
                        for h in hours[current_day][:-1]
                        if (p, t, h,                         current_day) in x_set
                        and (p, t, h_next[(h, current_day)], current_day) in x_set
                    ]
                    c = model.addVars(consec_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                enemies_scope = (
                    [
                        (p1, p2, t, h, current_day)
                        for p1, p2 in social_enemies
                        for t in tasks for h in hours[current_day]
                        if (p1, t, h, current_day) in x_set and (p2, t, h, current_day) in x_set
                    ]
                    if hard_enemies or W_SOCIAL > 0
                    else []
                )

                if W_SOCIAL > 0:
                    friends_keys = [
                        (p1, p2, t, h, current_day)
                        for p1, p2 in social_friends
                        for t in tasks for h in hours[current_day]
                        if (p1, t, h, current_day) in x_set and (p2, t, h, current_day) in x_set
                    ]
                    f = model.addVars(friends_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)
                    enemies_keys = [] if hard_enemies else enemies_scope
                    e = model.addVars(enemies_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                if W_CAPTAIN > 0:
                    active_hd = active_hd_by_day[current_day] if captains else []
                    k = model.addVars(active_hd, lb=0, ub=1, vtype=GRB.CONTINUOUS)

                # ══════════════════════════════════════════════════════
                # OBJECTIVE FUNCTION
                # ══════════════════════════════════════════════════════

                obj = gp.LinExpr()

                if W_COVERAGE > 0:
                    obj += W_COVERAGE * m.sum()
                if W_FORCE > 0:
                    obj += W_FORCE * u.sum()
                if W_CAPTAIN > 0:
                    obj += W_CAPTAIN * k.sum()
                if W_EMERG > 0:
                    obj += W_EMERG * gp.quicksum(
                        x[p, t, h, d]
                        for p, t, h, d in x_set if (p, h, d) in emerg_set)
                if W_STABILITY > 0 and X_prev:
                    obj += W_STABILITY * z.sum()
                if W_EQ_DAY > 0:
                    obj += W_EQ_DAY * (spread.sum() + j_max.sum())
                if W_EQ_GLOBAL > 0:
                    obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())
                if W_EQ_GROUP > 0:
                    obj += W_EQ_GROUP * (gamma_plus.sum() + gamma_minus.sum())
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
                        pref_nonzero[(p, t)] * x[p, t, h, d]
                        for p, t, h, d in x_set if (p, t) in pref_nonzero)

                model.setObjective(obj, GRB.MINIMIZE)

                # ══════════════════════════════════════════════════════
                # CONSTRAINTS
                # ══════════════════════════════════════════════════════

                # — Demand coverage —
                model.addConstrs(
                    (x.sum('*', t, h, d) + m[t, h, d] == demand[(t, h, d)]
                     for t, h, d in m_keys))

                # — One task per person per hour —
                model.addConstrs(
                    (x.sum(p, '*', h, d) <= 1 for p, h, d in avail_set_in_day))

                # — Force mandates —
                if W_FORCE > 0:
                    model.addConstrs(
                        (1 - x[p, t, h, d] <= u[p, t, h, d]
                         for p, t, h, d in force_possible))
                    model.addConstrs(
                        (u[p, t, h, d] >= 1
                         for p, t, h, d in force_impossible))

                # — Captain coverage —
                if W_CAPTAIN > 0:
                    model.addConstrs(
                        (x.sum(captains, '*', h, d) + k[h, d] >= 1
                         for h, d in active_hd))

                # — Intra-day equity —
                if W_EQ_DAY > 0:
                    eq_day_keys = [(p, current_day) for p in {p for p, h, d in avail_set_in_day}]
                    model.addConstrs(
                        (x.sum(p, '*', '*', d) <= j_max[d] for p, d in eq_day_keys))
                    model.addConstrs(
                        (j_max[d] - x.sum(p, '*', '*', d) <= spread[d] for p, d in eq_day_keys))

                # — Intra-group equity (W_EQ_GLOBAL, now group-aware) —
                # Each person's target is their group's fair share per person.
                if W_EQ_GLOBAL > 0:
                    pace_target_by_person = {
                        p: group_person_pace[person_to_group[p]] * (loop_idx + 1)
                        for p in people_available_in_day
                    }
                    model.addConstrs(
                        (accumulated_hours[p] + x.sum(p, '*', '*', current_day)
                         - delta_plus[p] + delta_minus[p] == pace_target_by_person[p]
                         for p in people_available_in_day))

                # — Inter-group equity (W_EQ_GROUP, new) —
                # Each group's total accumulated hours should track its pace target.
                if W_EQ_GROUP > 0:
                    group_pace_target = inter_group_pace * (loop_idx + 1)
                    # x_set_by_group: for each group, variables of its members in today's x_set
                    x_set_by_group = {
                        g: [(p, t, h, d) for p, t, h, d in x_set if p in group_people_set[g]]
                        for g in group_list
                    }
                    model.addConstrs((
                        group_accum[g]
                        + gp.quicksum(x[p, t, h, d] for p, t, h, d in x_set_by_group[g])
                        - gamma_plus[g] + gamma_minus[g] == group_pace_target
                        for g in group_list
                    ))

                # — Rotation (no consecutive hours on rotation tasks) —
                if W_ROTATION > 0:
                    model.addConstrs(
                        (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
                         for p, t, h, d in consec_keys))

                # — Social: friends & enemies —
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

                # — Minimum quota —
                if W_QUOTA > 0:
                    quota_constr_keys = [
                        (p, t, d, tgt)
                        for p, t, d in quota_keys
                        if (tgt := min(min_quota[(p, t, d)], len(hours[d]))) > 0
                    ]
                    model.addConstrs(
                        (x.sum(p, t, '*', d) + q[p, t, d] >= tgt
                         for p, t, d, tgt in quota_constr_keys))

                # — Stability vs previous planning —
                if W_STABILITY > 0 and X_prev:
                    forced_slots = {
                        (p, h, d)
                        for (p, t, h, d), v in force.items()
                        if v == 1 and d in current_days_set
                    }
                    model.addConstrs(
                        (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
                         for p, t, h, d in x_set if (p, h, d) not in forced_slots))
                    model.addConstrs(
                        (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
                         for p, t, h, d in x_set if (p, h, d) not in forced_slots))

                # — Gaps (penalize split shifts) —
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

                # — Max consecutive hours —
                for p_rest, limit in max_consec_hours.items():
                    if p_rest not in people:
                        continue
                    if len(hours[current_day]) <= limit:
                        continue
                    if not any((p_rest, h, current_day) in avail_set_in_day for h in hours[current_day]):
                        continue
                    model.addConstrs(
                        (x.sum(p_rest, '*', hours[current_day][i:i + limit + 1], current_day) <= limit
                         for i in range(len(hours[current_day]) - limit)))

                # ══════════════════════════════════════════════════════
                # WARM START — greedy feasible solution
                # ══════════════════════════════════════════════════════

                for key in x_set:
                    x[key].Start = 0

                busy  = set()
                cover = defaultdict(int)

                for t, h, _ in demand_pos_by_day[current_day]:
                    needed = demand[(t, h, current_day)]
                    for p in people_per_task[t]:
                        if cover[(t, h)] >= needed:
                            break
                        if (p, t, h, current_day) in x_set and (p, h) not in busy:
                            x[p, t, h, current_day].Start = 1
                            busy.add((p, h))
                            cover[(t, h)] += 1

                # ── Solve ─────────────────────────────────────────────
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
                            temp_assignment = {
                                d: {p: dict(partial_assignment[d][p]) for p in people}
                                for d in days
                            }
                            for p in people:
                                for h in hours[current_day]:
                                    temp_assignment[current_day][p][h] = None
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

                # ── Accumulate results ────────────────────────────────
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

                for p in people:
                    for h in hours[current_day]:
                        partial_assignment[current_day][p][h] = None
                for (p, t, h, d) in x_set:
                    if x[p, t, h, d].X > 0.5:
                        partial_assignment[d][p][h] = t
                        accumulated_hours[p] += 1

                interrupted = model.Status == GRB.INTERRUPTED

            finally:
                try:
                    model.dispose()
                except Exception:
                    pass
                if active_model_ref is not None:
                    active_model_ref[0] = None

            if interrupted:
                break

    finally:
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
        for p, t, h, d in (k for k, v in force.items() if v == 1):
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

    # — Per-person workload —
    assigned_counts = {p: 0 for p in people}
    for p, t, h, d in all_x_vals:
        if all_x_vals[p, t, h, d] > 0.5:
            assigned_counts[p] += 1
    sol["workload"] = assigned_counts
    wl = list(assigned_counts.values())
    sol["workload_max"] = float(max(wl)) if wl else 0.0
    sol["workload_min"] = float(min(wl)) if wl else 0.0

    # — Per-group workload summary —
    group_target_total = total_demand / n_groups
    sol["group_workload"] = {}
    group_eq_issues = []
    for g in group_list:
        total_h  = sum(assigned_counts[p] for p in group_people[g])
        target_h = group_target_total
        diff     = total_h - target_h
        per_person_h   = total_h / max(len(group_people[g]), 1)
        target_pp      = target_h / max(len(group_people[g]), 1)
        sol["group_workload"][g] = {
            "total_hours":      total_h,
            "target_hours":     round(target_h, 2),
            "deviation":        round(diff, 2),
            "per_person_avg":   round(per_person_h, 2),
            "per_person_target": round(target_pp, 2),
            "members":          group_people[g],
        }
        status_str = "OK" if abs(diff) < 0.5 else ("OVER" if diff > 0 else "SHORT")
        group_eq_issues.append(
            f"{status_str}: Group '{g}' — {total_h}h assigned "
            f"(target {target_h:.1f}h, deviation {diff:+.1f}h, "
            f"{per_person_h:.1f}h/person vs target {target_pp:.1f}h/person)"
        )
    sol["group_eq_issues"] = group_eq_issues

    if W_QUOTA > 0:
        assigned_by_ptd = Counter()
        for (p, t, h, d), v in all_x_vals.items():
            if v > 0.5:
                assigned_by_ptd[(p, t, d)] += 1
        sol["quota_issues"] = []
        for p in people:
            for t in tasks:
                for d in days:
                    if min_quota.get((p, t, d), 0) == 0:
                        continue
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
            starts       = 0
            start_hours  = []
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

    if has_enemies:
        if not hard_enemies:
            violations = [
                f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                for p1, p2 in social_enemies
                for t, h, d in slots_by_person.get(p1, set())
                if (p2, t, h, d) in active_slots
            ]
        else:
            violations = []
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
    sol["social_issues"] = soc_issues

    all_active_hd = [
        (h, d) for d in days for h, _ in active_hd_by_day[d]
    ] if captains else []
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
        if (p, h, d) in emerg_set
    ]

    h_next_all = {
        (hours[d][i], d): hours[d][i + 1]
        for d in days for i in range(len(hours[d]) - 1)
    }
    rot_violations = [
        f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next_all[(h, d)]}, {d}"
        for p, t, h, d in active_slots
        if t in rotation_tasks
        and (h, d) in h_next_all
        and (p, t, h_next_all[(h, d)], d) in active_slots
    ]
    sol["rotation_issues"] = (
        rot_violations or ["SUCCESS: No consecutive hours on rotation tasks."]
    )

    return sol