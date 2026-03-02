import flet as ft
import random
import threading
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# CONSTANTS 
# ============================================================
DEFAULT_WEIGHTS = {
    "W_COVERAGE": 100000,
    "W_MANDATE": 50000,
    "W_EMERG": 10000,
    "W_STABILITY": 5000,
    "W_EQ_DAY": 1000,
    "W_EQ_TOTAL": 500,
    "W_GAP": 100,
    "W_ROTATION": 50,
    "W_SOCIAL": 10,
    "W_QUOTA": 5,
    "W_PREF": 1
}

SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

CAPTAIN_BG = "#E65100"
CAPTAIN_FG = "#FFFFFF"

DEFAULT_HOURS_TEXT = "08:00\n09:00\n10:00\n11:00\n12:00\n13:00\n14:00\n15:00\n16:00\n17:00"

# 30 distinct task colors (bg, fg) — enough to avoid repeats
TASK_COLORS = [
    ("#CE93D8", "#000000"), ("#80DEEA", "#000000"), ("#FFF59D", "#000000"),
    ("#A5D6A7", "#000000"), ("#FFAB91", "#000000"), ("#90CAF9", "#000000"),
    ("#F48FB1", "#000000"), ("#E6EE9C", "#000000"), ("#B0BEC5", "#000000"),
    ("#FFCC80", "#000000"), ("#80CBC4", "#000000"), ("#B39DDB", "#000000"),
    ("#EF9A9A", "#000000"), ("#C5E1A5", "#000000"), ("#81D4FA", "#000000"),
    ("#FFE082", "#000000"), ("#F8BBD0", "#000000"), ("#BCAAA4", "#000000"),
    ("#A1887F", "#FFFFFF"), ("#7986CB", "#FFFFFF"), ("#4DB6AC", "#FFFFFF"),
    ("#FF8A65", "#000000"), ("#AED581", "#000000"), ("#4FC3F7", "#000000"),
    ("#DCE775", "#000000"), ("#BA68C8", "#FFFFFF"), ("#4DD0E1", "#000000"),
    ("#E57373", "#000000"), ("#9575CD", "#FFFFFF"), ("#FFD54F", "#000000"),
]

UNAVAIL_COLOR = "#D32F2F"
EMERG_COLOR   = "#F57C00"
AVAIL_COLOR   = "#388E3C"

# ============================================================
# BACKEND: GUROBIPY MODEL
# ============================================================

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

    W_COVERAGE  = W["W_COVERAGE"];      W_MANDATE   = W["W_MANDATE"]
    W_EMERG     = W["W_EMERG"];         W_STABILITY = W["W_STABILITY"]
    W_EQ_DAY    = W["W_EQ_DAY"];        W_EQ_TOTAL  = W["W_EQ_TOTAL"]
    W_ROTATION  = W["W_ROTATION"];      W_SOCIAL    = W["W_SOCIAL"]
    W_GAP       = W["W_GAP"];           W_QUOTA     = W["W_QUOTA"]
    W_PREF      = W["W_PREF"]

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
    obj += W_MANDATE  * u.sum()

    if captains:
        obj += W_MANDATE * k.sum()

    obj += W_EMERG * gp.quicksum(
        emergency.get((p,h,j), 0) * x[p,t,h,j]
        for p,t,h,j in x_set)

    obj += W_PREF * gp.quicksum(
        pref_cost.get((p,t), 0) * x[p,t,h,j]
        for p,t,h,j in x_set)

    obj += W_STABILITY * d.sum()

    for j in days:
        obj += W_EQ_DAY * (n_max[j] - n_min[j])
        
    obj += W_EQ_TOTAL * (z_max - z_min)

    if consec_keys:
        obj += W_ROTATION * c.sum()

    if v_keys:
        obj += W_SOCIAL * v.sum()

    if w_keys:
        obj += W_SOCIAL * w.sum()

    obj += W_GAP   * r.sum()
    obj += W_QUOTA * q.sum()

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


# ============================================================
# FLET UI
# ============================================================

def main(page: ft.Page):
    page.title = "Staff Scheduler"
    page.scroll = ft.ScrollMode.AUTO
    page.window.width = 1200
    page.window.height = 800
    page.theme_mode = ft.ThemeMode.LIGHT

    # ── UI state stores ──────────────────────────────────────
    avail_st, demand_st, skills_st = {}, {}, {}
    force_st, social_st, quota_st, rotation_st = {}, {}, {}, {}
    captains_st = {}
    hard_enemies_st = [False]
    hours_per_day_st = {}

    running_model_ref = [None]
    ui_lock = threading.Lock()

    # ── Build cache: skip rebuild if dims unchanged ──────────
    _build_cache = {}

    def _dims_hash():
        return hash((tf_people.value, tf_tasks.value, tf_days.value,
                      tuple(sorted(hours_per_day_st.items()))))

    def _needs_rebuild(tab_name):
        key = _dims_hash()
        if _build_cache.get(tab_name) == key:
            return False
        _build_cache[tab_name] = key
        return True

    def _invalidate_cache():
        _build_cache.clear()

    solver_params_st = {
        "TimeLimit": 1200,
        "MIPGap": 0.01,
        "MIPFocus": 2,
        "Threads": 0,
        "Presolve": 2,
        "Symmetry": 2,
        "Disconnected": 2,
        "IntegralityFocus": 1,
        "Method": 3
    }

    weights_st = DEFAULT_WEIGHTS.copy()
    weights_order = list(DEFAULT_WEIGHTS.keys())
    weights_enabled = {k: True for k in DEFAULT_WEIGHTS.keys()}

    # ── Dimension TextFields ─────────────────────────────────
    tf_people = ft.TextField(
        value="Arnaud\nNina\nJoseph\nChloé\nNiels\nZiad\nTristan\nJasmin\nMarine\nNoé\nJules R\nGuillaume\nNoémie\nStanislas\nTérence\nManon In\nManon L\nPauline\nLucie\nDarius\nMattia\nPierre\nBaptiste B\nVincent\nMadeleine\nIlhan\nMatteo\nAlexandre L\nPablo\nJenaya\nLiv\nFaustin\nKenza\nJuliette B\nSarah\nAlexandre B\nRémi\nGabi\nJeanne B\nMatthieu A\nInès\nMaxime N\nAriane\nMatthias",
        multiline=True, min_lines=8, max_lines=200, label="People (one per line)", width=180)
    tf_tasks = ft.TextField(
        value="Pénélope\nAugustin B\nAgathe\nRafael\nLuna\nLéna\nCamille\nJuliette M\nPaul\nManon P\nMatthieu G\nAlix\nNadim\nJean-Louis\nArthur",
        multiline=True, min_lines=8, max_lines=200, label="Tasks (one per line)", expand=True)
    tf_days = ft.TextField(
        value="Mon\nTue\nWed",
        multiline=True, min_lines=8, max_lines=200, label="Days (one per line)", width=120)

    # Invalidate cache when dimension fields change
    def _on_dims_change(e):
        _invalidate_cache()

    tf_people.on_change = lambda e: (_on_dims_change(e), build_captains_list(e))
    tf_tasks.on_change = _on_dims_change
    tf_days.on_change = lambda e: (_on_dims_change(e), build_hours_per_day(e))

    # ── Captains column ──────────────────────────────────────
    captains_col = ft.ListView(expand=True, spacing=4)

    def build_captains_list(e=None):
        people = list(dict.fromkeys(x.strip() for x in tf_people.value.split("\n") if x.strip()))
        buf = [ft.Text("Captains", weight=ft.FontWeight.BOLD, size=12)]
        for p in people:
            if p not in captains_st:
                captains_st[p] = 0
            val = captains_st[p]
            btn = ft.Container(
                content=ft.Text("Cap" if val else "—", color=ft.Colors.BLACK, size=12, weight=ft.FontWeight.BOLD),
                width=55, height=28, data=p,
                bgcolor=ft.Colors.AMBER_400 if val else ft.Colors.GREY_400,
                alignment=ft.alignment.center, border_radius=4)
            def _click(e, _p=p):
                captains_st[_p] = 1 - captains_st[_p]
                e.control.content.value = "Cap" if captains_st[_p] else "—"
                e.control.bgcolor = ft.Colors.AMBER_400 if captains_st[_p] else ft.Colors.GREY_400
                e.control.update()
            btn.on_click = _click
            buf.append(ft.Row([ft.Text(p, size=11, width=90), btn], spacing=4))
        captains_col.controls = buf
        page.update()

    # ── Hours-per-day column (auto-generated from tf_days) ───
    hours_col = ft.ListView(expand=True, spacing=4)

    def build_hours_per_day(e=None):
        day_list = list(dict.fromkeys(x.strip() for x in tf_days.value.split("\n") if x.strip()))
        buf = [ft.Text("Hours per Day", weight=ft.FontWeight.BOLD, size=12)]
        for j in day_list:
            if j not in hours_per_day_st:
                hours_per_day_st[j] = DEFAULT_HOURS_TEXT
            tf = ft.TextField(
                value=hours_per_day_st[j], multiline=True, min_lines=4, max_lines=24,
                label=j, width=140, data=j)
            def _ch(e, _j=j):
                hours_per_day_st[_j] = e.control.value
                _invalidate_cache()
            tf.on_change = _ch
            buf.append(tf)
        hours_col.controls = buf
        page.update()

    # ── Enforced rest widgets ────────────────────────────────
    err_max_consec = ft.Text("", color=ft.Colors.RED_400, size=11, visible=False)
    tf_max_consec = ft.TextField(value="4", width=100, height=35, text_size=12,
                                  label="Max hours (Y)", visible=False,
                                  content_padding=ft.padding.all(4))

    def _toggle_enforced_rest(e):
        tf_max_consec.visible = e.control.value
        err_max_consec.visible = False
        page.update()

    sw_enforced_rest = ft.Switch(label="Enforced Rest", value=False, on_change=_toggle_enforced_rest)
    enforced_rest_col = ft.Column([sw_enforced_rest, tf_max_consec, err_max_consec], spacing=5)

    # ── dims() helper — returns (people, tasks, hours_dict, days) ─
    def dims():
        people = list(dict.fromkeys(x.strip() for x in tf_people.value.split("\n") if x.strip()))
        tasks  = list(dict.fromkeys(x.strip() for x in tf_tasks.value.split("\n") if x.strip()))
        days   = list(dict.fromkeys(x.strip() for x in tf_days.value.split("\n") if x.strip()))
        default = list(dict.fromkeys(x.strip() for x in DEFAULT_HOURS_TEXT.split("\n") if x.strip()))
        hours = {}
        for j in days:
            raw = hours_per_day_st.get(j, DEFAULT_HOURS_TEXT)
            parsed = list(dict.fromkeys(x.strip() for x in raw.split("\n") if x.strip()))
            hours[j] = parsed if parsed else default
        return people, tasks, hours, days

    # ── Tab containers ───────────────────────────────────────
    avail_ct        = ft.ListView(expand=True, spacing=5)
    demand_ct       = ft.ListView(expand=True, spacing=5)
    skills_quota_ct = ft.ListView(expand=True, spacing=5) # Replaces skills_ct and quota_ct
    force_ct        = ft.ListView(expand=True, spacing=5)
    social_ct       = ft.ListView(expand=True, spacing=5)
    rotation_ct     = ft.ListView(expand=True, spacing=5)
    weights_ct      = ft.ListView(expand=True, spacing=5)
    params_ct       = ft.ListView(expand=True, spacing=5)
    output_ct       = ft.ListView(expand=True, spacing=5)

    W_LBL = 90; W_BTN = 58; W_CELL = 65

    # ── Reusable widget helpers ──────────────────────────────

    def make_toggle(sd, key, default):
        if key not in sd: sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = 1 - _sd[_k]
            e.control.content.value = str(_sd[_k])
            e.control.bgcolor = ft.Colors.GREEN_700 if _sd[_k] else ft.Colors.RED_700
            e.control.update()
        return ft.Container(
            content=ft.Text(str(sd[key]), color=ft.Colors.WHITE, size=12, weight=ft.FontWeight.BOLD),
            width=W_BTN, height=30, data=key,
            bgcolor=ft.Colors.GREEN_700 if sd[key] else ft.Colors.RED_700,
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    _avail_lbl  = {1: "1", 0: "0", 2: "!"}
    _avail_clr  = {1: ft.Colors.GREEN_700, 0: ft.Colors.RED_700, 2: ft.Colors.ORANGE_700}
    _avail_next = {1: 0, 0: 2, 2: 1}

    def make_avail_toggle(sd, key, default=1):
        if key not in sd: sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = _avail_next[_sd[_k]]
            nv = _sd[_k]
            e.control.content.value = _avail_lbl[nv]
            e.control.bgcolor = _avail_clr[nv]
            e.control.update()
        return ft.Container(
            content=ft.Text(_avail_lbl[sd[key]], color=ft.Colors.WHITE, size=12, weight=ft.FontWeight.BOLD),
            width=W_BTN, height=30, data=key, bgcolor=_avail_clr[sd[key]],
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    def lbl(text, w=W_LBL):
        return ft.Container(ft.Text(text, size=11, no_wrap=True), width=w)

    def plbl(name, w=W_LBL):
        is_cap = captains_st.get(name, 0) == 1
        return ft.Container(
            ft.Text(name, size=11, no_wrap=True,
                    weight=ft.FontWeight.BOLD if is_cap else None,
                    color=CAPTAIN_BG if is_cap else None),
            width=w)

    def hdr_row(labels, w=W_BTN):
        return ft.Row([ft.Container(width=W_LBL)] +
                      [ft.Container(ft.Text(l, size=10), width=w) for l in labels], spacing=2)

    # ── Tab builders (buffer pattern: build list, assign, single update) ──

    def build_avail():
        if not _needs_rebuild("avail"): return
        people, tasks, hours, days = dims()
        buf = []

        def _rand_avail(e):
            for p in people:
                for j in days:
                    for h in hours[j]:
                        avail_st[(p,h,j)] = random.choice([0, 1, 2])
            _invalidate_cache()
            build_avail()

        buf.append(ft.Container(
            content=ft.Text("Random Avail", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
            on_click=_rand_avail, width=150, alignment=ft.alignment.center))

        for j in days:
            day_hours = hours[j]
            buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            buf.append(ft.Row(
                [ft.Container(width=W_LBL + W_BTN + 4)] +
                [ft.Container(ft.Text(l, size=10), width=W_BTN) for l in day_hours], spacing=2))
            for p in people:
                def _make_row_toggle(_p=p, _j=j):
                    def _click(e):
                        first_val = avail_st.get((_p, hours[_j][0], _j), 1)
                        new_val = _avail_next[first_val]
                        for _h in hours[_j]: avail_st[(_p, _h, _j)] = new_val
                        _invalidate_cache()
                        build_avail()
                    return ft.Container(
                        content=ft.Text("row", color=ft.Colors.WHITE, size=12),
                        width=W_BTN, height=30, bgcolor=ft.Colors.BLUE_GREY_400,
                        alignment=ft.alignment.center, border_radius=4, on_click=_click)
                row_btn = _make_row_toggle(p, j)
                buf.append(ft.Row(
                    [plbl(p), row_btn] +
                    [make_avail_toggle(avail_st, (p,h,j), 1) for h in day_hours], spacing=2))
            buf.append(ft.Divider())

        avail_ct.controls = buf
        page.update()

    def build_demand():
        if not _needs_rebuild("demand"): return
        people, tasks, hours, days = dims()
        buf = []

        def _rand_demand(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        demand_st[(t,h,j)] = str(random.randint(0, 4))
            _invalidate_cache()
            build_demand()

        buf.append(ft.Container(
            content=ft.Text("Random Demand", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
            on_click=_rand_demand, width=150, alignment=ft.alignment.center))

        for j in days:
            day_hours = hours[j]
            buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            buf.append(hdr_row(day_hours, W_CELL))
            for t in tasks:
                cells = []
                for h in day_hours:
                    k = (t,h,j)
                    if k not in demand_st: demand_st[k] = "1"
                    tf = ft.TextField(value=demand_st[k], width=W_CELL, height=35, text_size=12,
                                      data=k, content_padding=ft.padding.all(4))
                    def _ch(e, _k=k): demand_st[_k] = e.control.value
                    tf.on_change = _ch; cells.append(tf)
                buf.append(ft.Row([lbl(t)] + cells, spacing=2))
            buf.append(ft.Divider())

        demand_ct.controls = buf
        page.update()

    def build_skills_quota():
        if not _needs_rebuild("skills_quota"): return
        people, tasks, hours, days = dims()

        def _rand_skills(e):
            for p in people:
                for t in tasks: skills_st[(p,t)] = random.choice([0, 1])
            _invalidate_cache()
            build_skills_quota()

        def _rand_quota(e):
            for p in people:
                for t in tasks: quota_st[(p,t)] = str(random.randint(0, 2))
            _invalidate_cache()
            build_skills_quota()

        # Build Skills side
        skills_buf = []
        skills_buf.append(ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=16))
        skills_buf.append(ft.Container(
            content=ft.Text("Random Skills", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
            on_click=_rand_skills, width=150, alignment=ft.alignment.center))
        skills_buf.append(hdr_row(tasks, 70))
        for p in people:
            skills_buf.append(ft.Row([plbl(p)] + [make_toggle(skills_st, (p,t), 1) for t in tasks], spacing=2))

        # Build Quota side
        quota_buf = []
        quota_buf.append(ft.Text("Minimum Quota Matrix", weight=ft.FontWeight.BOLD, size=16))
        quota_buf.append(ft.Container(
            content=ft.Text("Random Quota", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
            on_click=_rand_quota, width=150, alignment=ft.alignment.center))
        quota_buf.append(hdr_row(tasks, 70))
        for p in people:
            cells = []
            for t in tasks:
                k = (p, t)
                if k not in quota_st: quota_st[k] = "0"
                tf = ft.TextField(value=quota_st[k], width=70, height=35, text_size=12,
                                  data=k, content_padding=ft.padding.all(4))
                def _ch(e, _k=k): quota_st[_k] = e.control.value
                tf.on_change = _ch; cells.append(tf)
            quota_buf.append(ft.Row([plbl(p)] + cells, spacing=2))

        # Combine side-by-side
        col_skills = ft.Column(skills_buf, spacing=5, scroll=ft.ScrollMode.ADAPTIVE)
        col_quota = ft.Column(quota_buf, spacing=5, scroll=ft.ScrollMode.ADAPTIVE)

        row_content = ft.Row([
            ft.Container(content=col_skills, border=ft.border.only(right=ft.border.BorderSide(1, "#CFD8DC")), padding=10, expand=True),
            ft.Container(content=col_quota, padding=10, expand=True)
        ], vertical_alignment=ft.CrossAxisAlignment.START, scroll=ft.ScrollMode.ADAPTIVE, expand=True)

        skills_quota_ct.controls = [row_content]
        page.update()

    def build_force():
        if not _needs_rebuild("force"): return
        people, tasks, hours, days = dims()
        buf = []
        for t in tasks:
            for j in days:
                day_hours = hours[j]
                buf.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=14))
                buf.append(hdr_row(day_hours))
                for p in people:
                    buf.append(ft.Row(
                        [plbl(p)] + [make_toggle(force_st, (p,t,h,j), 0) for h in day_hours], spacing=2))
                buf.append(ft.Divider())

        force_ct.controls = buf
        page.update()

    def build_social():
        if not _needs_rebuild("social"): return
        people, tasks, hours, days = dims()
        buf = []
        if len(people) < 2:
            social_ct.controls = buf; page.update(); return

        def _rand_social(e):
            for i, p1 in enumerate(people):
                for p2 in people[i+1:]: social_st[(p1,p2)] = random.choice([-1, 0, 1])
            _invalidate_cache()
            build_social()

        buf.append(ft.Container(
            content=ft.Text("Random Social", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
            on_click=_rand_social, width=150, alignment=ft.alignment.center))

        sw_hard_enemies = ft.Switch(label="Enemies: Hard Constraint", value=hard_enemies_st[0])
        def _toggle_hard_enemies(e): hard_enemies_st[0] = e.control.value
        sw_hard_enemies.on_change = _toggle_hard_enemies
        buf.append(ft.Row([sw_hard_enemies], spacing=20))

        buf.append(hdr_row(people[1:], 70))
        _map_lbl = {0: "~", 1: "+", -1: "-"}
        _map_clr = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_700, -1: ft.Colors.RED_700}
        _next_val = {0: 1, 1: -1, -1: 0}
        for i, p1 in enumerate(people):
            cells = []
            for p2 in people[1:]:
                j2 = people.index(p2)
                if j2 > i:
                    k = (p1, p2)
                    if k not in social_st: social_st[k] = 0
                    def _click(e, _k=k):
                        social_st[_k] = _next_val[social_st[_k]]
                        nv = social_st[_k]
                        e.control.content.value = _map_lbl[nv]
                        e.control.bgcolor = _map_clr[nv]
                        e.control.update()
                    btn = ft.Container(
                        content=ft.Text(_map_lbl[social_st[k]], color=ft.Colors.WHITE, size=14,
                                        weight=ft.FontWeight.BOLD),
                        width=70, height=30, data=k, bgcolor=_map_clr[social_st[k]],
                        alignment=ft.alignment.center, border_radius=4, on_click=_click)
                    cells.append(btn)
                else:
                    cells.append(ft.Container(width=70))
            if cells:
                buf.append(ft.Row([plbl(p1)] + cells, spacing=2))

        social_ct.controls = buf
        page.update()

    def build_rotation():
        if not _needs_rebuild("rotation"): return
        people, tasks, hours, days = dims()
        buf = []
        for t in tasks:
            if t not in rotation_st: rotation_st[t] = 1
            sw = ft.Switch(label=t, value=rotation_st[t] == 1, data=t)
            def _ch(e, _t=t): rotation_st[_t] = 1 if e.control.value else 0
            sw.on_change = _ch; buf.append(sw)

        rotation_ct.controls = buf
        page.update()

    def build_weights():
        items_controls = []
        for i, key in enumerate(weights_order):
            weights_st[key] = SORTED_VALUES[i] if weights_enabled[key] else 0

        header = ft.Column([
            ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=16),
            ft.Text("Drag items to reorder. Top items = Higher Cost.", italic=True, size=12),
            ft.Divider(),
        ], spacing=5)

        def handle_reorder(e):
            item = weights_order.pop(e.old_index)
            weights_order.insert(e.new_index, item)
            build_weights()

        for i, key in enumerate(weights_order):
            val = SORTED_VALUES[i] if weights_enabled[key] else 0
            sw = ft.Switch(value=weights_enabled[key], data=key)
            def _toggle(e, _k=key): weights_enabled[_k] = e.control.value; build_weights()
            sw.on_change = _toggle
            card = ft.Container(
                content=ft.Row([
                    ft.Text(f"#{i+1}", width=30, weight=ft.FontWeight.BOLD),
                    ft.Text(key, expand=True, weight=ft.FontWeight.W_800),
                    ft.Text(f"{val}            ",
                            color=ft.Colors.BLACK if weights_enabled[key] else ft.Colors.GREY_500, size=16),
                    sw,
                ], alignment=ft.MainAxisAlignment.START),
                padding=10,
                bgcolor=ft.Colors.LIGHT_BLUE_100 if weights_enabled[key] else ft.Colors.GREY_300,
                border=ft.border.all(1, ft.Colors.GREY_400), border_radius=8,
                margin=ft.margin.only(bottom=5))
            items_controls.append(card)

        r_list = ft.ReorderableListView(controls=items_controls, on_reorder=handle_reorder)
        layout = ft.Column(
            controls=[header, ft.Container(content=r_list, width=420, height=500)],
            width=420, alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        weights_ct.controls = [ft.Row([layout], alignment=ft.MainAxisAlignment.CENTER)]
        page.update()

    def build_params():
        buf = [
            ft.Text("Gurobi Solver Parameters", weight=ft.FontWeight.BOLD, size=16),
            ft.Divider()
        ]
        for key, val in solver_params_st.items():
            tf = ft.TextField(label=key, value=str(val), width=250, height=45, text_size=13)
            def _ch(e, _k=key):
                try:
                    new_val = float(e.control.value) if "." in e.control.value else int(e.control.value)
                    solver_params_st[_k] = new_val
                except ValueError: pass
            tf.on_change = _ch
            buf.append(tf)

        params_ct.controls = buf
        page.update()

    # ── Tab navigation ───────────────────────────────────────
    builders = {
        1: build_avail, 2: build_demand, 3: build_skills_quota, # Replaced old 3 & 6 mappings
        4: build_force, 5: build_social, 6: build_rotation,
        7: build_weights, 8: build_params}

    def on_tab_change(e):
        idx = e.control.selected_index
        if idx in builders:
            with ui_lock:
                builders[idx]()

    # ── Output grid ──────────────────────────────────────────

    def build_output_grid(sol, people, tasks, hours, days, availability, emergency):
        buf = []

        status     = sol.get("status", "Solving...")
        assignment = sol.get("assignment", {})
        missing_issues = sol.get("missing", [])
        workload   = sol.get("workload", {})
        gaps       = sol.get("gaps", [])
        soc_issues = sol.get("social_issues", [])
        cap_issues = sol.get("captain_issues", [])
        emerg_issues = sol.get("emerg_issues", [])
        rot_issues = sol.get("rotation_issues", [])

        buf.append(ft.Text(f"Status: {status}", weight=ft.FontWeight.BOLD, size=16))
        buf.append(ft.Divider())

        tc = {}
        for i, t in enumerate(tasks):
            bg, fg = TASK_COLORS[i % len(TASK_COLORS)]; tc[t] = (bg, fg)
        CW = 75; Ch = 36; NW = 110; TW = 50

        for j in days:
            if j not in assignment: continue
            day_hours = hours[j]
            buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=16))

            # Header row
            hdr = [ft.Container(
                ft.Text("Person", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                width=NW, height=Ch, bgcolor="#546E7A", alignment=ft.alignment.center,
                border=ft.border.all(1, "#455A64"))]
            for h in day_hours:
                hdr.append(ft.Container(
                    ft.Text(h, size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    width=CW, height=Ch, bgcolor="#546E7A", alignment=ft.alignment.center,
                    border=ft.border.all(1, "#455A64")))
            hdr.append(ft.Container(
                ft.Text("Total", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                width=TW, height=Ch, bgcolor="#546E7A", alignment=ft.alignment.center,
                border=ft.border.all(1, "#455A64")))
            buf.append(ft.Row(hdr, spacing=0))

            # Person rows
            asgn = assignment[j]
            for idx_p, p in enumerate(people):
                is_cap = captains_st.get(p, 0) == 1
                row_bg = "#ECEFF1" if idx_p % 2 == 0 else "#FFFFFF"
                name_color = CAPTAIN_BG if is_cap else ft.Colors.BLACK
                cells = [ft.Container(
                    ft.Text(p, size=12, weight=ft.FontWeight.BOLD, color=name_color),
                    width=NW, height=Ch, bgcolor=row_bg, alignment=ft.alignment.center_left,
                    padding=ft.padding.only(left=8), border=ft.border.all(1, "#CFD8DC"))]
                total = 0
                for h in day_hours:
                    task = asgn.get(p, {}).get(h)
                    avail = availability.get((p,h,j), 1)
                    griev = emergency.get((p,h,j), 0)
                    if avail == 0:   brd = ft.border.all(1.5, UNAVAIL_COLOR)
                    elif griev == 1: brd = ft.border.all(1.5, EMERG_COLOR)
                    else:            brd = ft.border.all(0.5, AVAIL_COLOR)
                    if task:
                        bg, fg = tc[task]; total += 1
                        cell = ft.Container(
                            ft.Text(task, size=11, weight=ft.FontWeight.BOLD, color=fg,
                                    text_align=ft.TextAlign.CENTER),
                            width=CW, height=Ch, bgcolor=bg, alignment=ft.alignment.center,
                            border=brd, border_radius=4)
                    else:
                        cell = ft.Container(width=CW, height=Ch, bgcolor=row_bg, border=brd)
                    cells.append(cell)
                cells.append(ft.Container(
                    ft.Text(str(int(total)), size=12, weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLACK, text_align=ft.TextAlign.CENTER),
                    width=TW, height=Ch, bgcolor=row_bg, alignment=ft.alignment.center,
                    border=ft.border.all(1, "#CFD8DC")))
                buf.append(ft.Row(cells, spacing=0))

            # Legend
            legend_items = [
                ft.Container(ft.Text(t, size=10, weight=ft.FontWeight.BOLD, color=tc[t][1]),
                             bgcolor=tc[t][0], padding=ft.padding.symmetric(6, 10), border_radius=4)
                for t in tasks]
            legend_items.append(ft.Container(ft.Text("Available", size=10, color=ft.Colors.WHITE),
                                bgcolor=AVAIL_COLOR, padding=ft.padding.symmetric(6, 10), border_radius=4))
            legend_items.append(ft.Container(ft.Text("Emergency", size=10, color=ft.Colors.WHITE),
                                bgcolor=EMERG_COLOR, padding=ft.padding.symmetric(6, 10), border_radius=4))
            legend_items.append(ft.Container(ft.Text("Unavailable", size=10, color=ft.Colors.WHITE),
                                bgcolor=UNAVAIL_COLOR, padding=ft.padding.symmetric(6, 10), border_radius=4))
            buf.append(ft.Row(legend_items, spacing=8))
            buf.append(ft.Divider())

        if "Solving" in status:
            output_ct.controls = buf
            return

        buf.append(ft.Text("MISSING STAFF", weight=ft.FontWeight.BOLD, size=14))
        if missing_issues:
            for line in missing_issues: buf.append(ft.Text(f"  {line}", size=12))
        else:
            buf.append(ft.Text("  None -- all demand covered!", size=12, italic=True))

        buf.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=14))
        for p in people:
            buf.append(ft.Text(f"  {p}: {workload.get(p, 0):.0f} hours", size=12))
        buf.append(ft.Text(
            f"  Global range: max={sol.get('z_max', 0):.0f}, min={sol.get('z_min', 0):.0f}",
            size=12, italic=True))

        if not sol.get("enforced_rest", False):
            buf.append(ft.Text("SHIFT SEGMENTS / FRAGMENTATION", weight=ft.FontWeight.BOLD, size=14))
            if gaps:
                for line in gaps: buf.append(ft.Text(f"  {line}", size=12))
            else:
                buf.append(ft.Text("  Single block shifts! (Perfect)", size=12, italic=True))

        buf.append(ft.Text("ROTATION FATIGUE", weight=ft.FontWeight.BOLD, size=14))
        if rot_issues:
            for line in rot_issues: buf.append(ft.Text(f"  {line}", size=12))
        else:
            buf.append(ft.Text("  No rotation tasks defined.", size=12, italic=True))

        buf.append(ft.Text("SOCIAL", weight=ft.FontWeight.BOLD, size=14))
        if soc_issues:
            for line in soc_issues: buf.append(ft.Text(f"  {line}", size=12))
        else:
            buf.append(ft.Text("  None -- all respected!", size=12, italic=True))

        buf.append(ft.Text("CAPTAIN PRESENCE", weight=ft.FontWeight.BOLD, size=14))
        if cap_issues:
            for line in cap_issues: buf.append(ft.Text(f"  {line}", size=12))
        else:
            buf.append(ft.Text("  No captains designated or all hours covered.", size=12, italic=True))

        buf.append(ft.Text("EMERGENCY CALL-INS", weight=ft.FontWeight.BOLD, size=14))
        if emerg_issues:
            for line in emerg_issues: buf.append(ft.Text(f"  {line}", size=12))
        else:
            buf.append(ft.Text("  None -- no emergency hours used!", size=12, italic=True))

        output_ct.controls = buf

    # ── Solve / Stop ─────────────────────────────────────────

    def do_solve(e):
        people, tasks, hours, days = dims()

        # Validate enforced rest
        enforced_rest = sw_enforced_rest.value
        max_consec_hours = None
        if enforced_rest:
            raw_y = tf_max_consec.value.strip()
            try:
                max_consec_hours = int(raw_y)
                if str(max_consec_hours) != raw_y or max_consec_hours < 1: raise ValueError
            except (ValueError, TypeError):
                err_max_consec.value = "Error: Y must be a positive integer."
                err_max_consec.visible = True; page.update(); return
            err_max_consec.visible = False; page.update()

        # Build parameter dicts from UI state
        availability = {}
        emergency = {}
        for p in people:
            for j in days:
                for h in hours[j]:
                    val = avail_st.get((p,h,j), 1)
                    availability[(p,h,j)] = 1 if val in (1, 2) else 0
                    emergency[(p,h,j)]    = 1 if val == 2 else 0

        demand = {}
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    raw = demand_st.get((t,h,j), "1")
                    try:    demand[(t,h,j)] = int(raw)
                    except: demand[(t,h,j)] = 0

        skills = {}
        for p in people:
            for t in tasks: skills[(p,t)] = skills_st.get((p,t), 1)

        force = {}
        for p in people:
            for t in tasks:
                for j in days:
                    for h in hours[j]: force[(p,t,h,j)] = force_st.get((p,t,h,j), 0)

        social = {}
        for i, p1 in enumerate(people):
            for p2 in people[i+1:]: social[(p1,p2)] = social_st.get((p1,p2), 0)

        mq = {}
        for p in people:
            for t in tasks:
                raw = quota_st.get((p,t), "0")
                try:    mq[(p,t)] = int(raw)
                except: mq[(p,t)] = 0

        rotation  = {t: rotation_st.get(t, 1) for t in tasks}
        pref_cost = {(p,t): 1 for p in people for t in tasks}
        X_prev    = {(p,t,h,j): 0 for p in people for t in tasks for j in days for h in hours[j]}
        weights   = weights_st.copy()
        captains  = [p for p in people if captains_st.get(p, 0) == 1]

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, emergency=emergency, demand=demand, skills=skills,
            force=force, social=social, min_quota=mq,
            pref_cost=pref_cost, rotation=rotation, X_prev=X_prev, weights=weights,
            enforced_rest=enforced_rest, max_consec_hours=max_consec_hours,
            captains=captains, solver_params=solver_params_st,
            hard_enemies=hard_enemies_st[0])

        output_ct.controls = [ft.ProgressRing(), ft.Text("Solving in Background...", italic=True)]
        tabs.selected_index = 9; page.update() # Updated index since we merged a tab

        def update_ui_with_temp_solution(partial_sol):
            if not ui_lock.acquire(blocking=False): return
            try:
                build_output_grid(partial_sol, people, tasks, hours, days, availability, emergency)
                page.update()
            except Exception: pass
            finally: ui_lock.release()

        def run_solver():
            try:
                final_sol = solve_model(
                    data, ui_update_callback=update_ui_with_temp_solution,
                    active_model_ref=running_model_ref)
                build_output_grid(final_sol, people, tasks, hours, days, availability, emergency)
            except Exception as ex:
                output_ct.controls = [ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=14)]
            running_model_ref[0] = None
            if len(output_ct.controls) > 0 and isinstance(output_ct.controls[0], ft.ProgressRing):
                output_ct.controls.pop(0)
            page.update()

        threading.Thread(target=run_solver, daemon=True).start()

    def do_stop(e):
        if running_model_ref[0]:
            output_ct.controls.insert(1, ft.Text(
                "Interruption requested... Halting at current node.",
                color=ft.Colors.ORANGE_700, italic=True))
            page.update()
            running_model_ref[0].terminate()

    # ── Action buttons ───────────────────────────────────────
    solve_btn = ft.Container(
        content=ft.Text("SOLVE", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.BLUE_700, padding=10, border_radius=8,
        on_click=do_solve, width=200, alignment=ft.alignment.center)
    stop_btn = ft.Container(
        content=ft.Text("STOP", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.RED_700, padding=10, border_radius=8,
        on_click=do_stop, width=120, alignment=ft.alignment.center)

    # ── Dimensions tab layout ────────────────────────────────
    dim_tab_content = ft.Container(
        content=ft.Row(
            controls=[tf_people, captains_col, tf_tasks, tf_days, hours_col, enforced_rest_col],
            spacing=20,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START),
        padding=20, expand=True)

    # ── Tabs ─────────────────────────────────────────────────
    tabs = ft.Tabs(
        selected_index=0, on_change=on_tab_change, expand=True,
        tabs=[
            ft.Tab(text="Dimensions",   content=dim_tab_content),
            ft.Tab(text="Availability", content=ft.Container(avail_ct, padding=10, expand=True)),
            ft.Tab(text="Demand",       content=ft.Container(demand_ct, padding=10, expand=True)),
            ft.Tab(text="Skills & Quota", content=ft.Container(skills_quota_ct, padding=10, expand=True)),
            ft.Tab(text="Force",        content=ft.Container(force_ct, padding=10, expand=True)),
            ft.Tab(text="Social",       content=ft.Container(social_ct, padding=10, expand=True)),
            ft.Tab(text="Rotation",     content=ft.Container(rotation_ct, padding=10, expand=True)),
            ft.Tab(text="Weights",      content=ft.Container(weights_ct, padding=10, expand=True)),
            ft.Tab(text="Parameters",   content=ft.Container(params_ct, padding=10, expand=True)),
            ft.Tab(text="Output",       content=ft.Container(output_ct, padding=10, expand=True)),
        ])

    page.add(ft.Row([solve_btn, stop_btn], alignment=ft.MainAxisAlignment.CENTER), tabs)
    build_captains_list()
    build_hours_per_day()

ft.app(target=main, view=ft.AppView.WEB_BROWSER)