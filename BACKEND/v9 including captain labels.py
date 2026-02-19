import flet as ft
from pulp import *

# ============================================================
# CONSTANTS 
# ============================================================
DEFAULT_WEIGHTS = {
    "W_COVERAGE": 100000,
    "W_MANDATE": 50000,
    "W_STABILITY": 10000,
    "W_EQ_DAY": 5000,
    "W_EQ_TOTAL": 1000,
    "W_GAP": 500,
    "W_ROTATION": 100,
    "W_SOCIAL": 50,
    "W_QUOTA": 10,
    "W_PREF": 5
}

# Pre-calculate sorted values (descending) to assign based on rank
SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

CAPTAIN_BG = "#FFAB00"   # Amber highlight for captain names
CAPTAIN_FG = "#000000"

def solve_model(data):

    
    # ============================================================
    # 1. UNPACK INDICES AND SETS
    # ============================================================
    people = data["people"];    tasks = data["tasks"]
    hours  = data["hours"];     days  = data["days"]
    demand   = data["demand"];      availability = data["availability"]
    skills   = data["skills"];      force        = data["force"]
    social   = data["social"];      min_quota    = data["min_quota"]
    rotation = data["rotation"];    pref_cost    = data["pref_cost"]
    X_prev   = data["X_prev"]
    W = data["weights"] # <--- Esto recibirá los pesos actualizados
    enforced_rest = data.get("enforced_rest", False)
    max_consec_hours = data.get("max_consec_hours", None)
    captains = data.get("captains", [])

    W_COVERAGE  = W["W_COVERAGE"];      W_MANDATE   = W["W_MANDATE"]
    W_STABILITY = W["W_STABILITY"];     W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_TOTAL  = W["W_EQ_TOTAL"];      W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];        W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];         W_PREF      = W["W_PREF"]

    model = LpProblem("StaffScheduler", LpMinimize)

    
    # ============================================================
    # 2. DECISION VARIABLES
    # ============================================================
    x = LpVariable.dicts("x", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), cat=LpBinary)
    m = LpVariable.dicts("m", ((t,h,j) for t in tasks for h in hours for j in days), lowBound=0, cat=LpInteger)
    u = LpVariable.dicts("u", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), lowBound=0, cat=LpInteger)
    d = LpVariable.dicts("d", ((p,t,h,j) for p in people for t in tasks for h in hours for j in days), cat=LpBinary)
    q = LpVariable.dicts("q", ((p,t) for p in people for t in tasks), cat=LpBinary)
    r = LpVariable.dicts("r", ((p,h,j) for p in people for h in hours for j in days), cat=LpBinary)

    n_max = LpVariable.dicts("n_max", days, lowBound=0, cat=LpInteger)
    n_min = LpVariable.dicts("n_min", days, lowBound=0, cat=LpInteger)
    z_max = LpVariable("z_max", lowBound=0, cat=LpInteger)
    z_min = LpVariable("z_min", lowBound=0, cat=LpInteger)

    consec_keys = [(p,t,h,j) for p in people for t in tasks for h in hours[:-1] for j in days if rotation.get(t, 0) == 1]
    c = LpVariable.dicts("c", consec_keys, cat=LpBinary)

    friend_pairs = [(p1,p2) for (p1,p2), val in social.items() if val == 1]
    enemy_pairs  = [(p1,p2) for (p1,p2), val in social.items() if val == -1]
    v_keys = ((p1,p2,t,h,j) for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days)
    v = LpVariable.dicts("v", v_keys, cat=LpBinary)
    w_keys = ((p1,p2,t,h,j) for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days)
    w = LpVariable.dicts("w", w_keys, cat=LpBinary)

    # B.2. Captain slack variable k
    k = LpVariable.dicts("k", ((h,j) for h in hours for j in days), cat=LpBinary)

    # ============================================================
    # 3. OBJECTIVE FUNCTION
    # ============================================================
    obj = []
    obj += [W_COVERAGE * m[(t,h,j)] for t in tasks for h in hours for j in days]
    obj += [W_MANDATE * u[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]
    # Captain penalty (W_MANDATE) — only active hours matter; inactive ones stay 0 naturally
    obj += [W_MANDATE * k[(h,j)] for h in hours for j in days]
    obj += [W_STABILITY * d[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]
    obj += [W_EQ_DAY * (n_max[j] - n_min[j]) for j in days]
    obj.append(W_EQ_TOTAL * (z_max - z_min))
    if consec_keys: obj += [W_ROTATION * c[k_] for k_ in consec_keys]
    if friend_pairs: obj += [W_SOCIAL * v[(p1,p2,t,h,j)] for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days]
    if enemy_pairs: obj += [W_SOCIAL * w[(p1,p2,t,h,j)] for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days]
    obj += [W_GAP * r[(p,h,j)] for p in people for h in hours for j in days]
    obj += [W_QUOTA * q[(p,t)] for p in people for t in tasks]
    obj += [W_PREF * pref_cost.get((p,t), 0) * x[(p,t,h,j)] for p in people for t in tasks for h in hours for j in days]
    model += lpSum(obj), "Total_Penalty"

    # ============================================================
    # 4. CONSTRAINTS
    # ============================================================
    for t in tasks:
        for h in hours:
            for j in days:
                model += (lpSum(x[(p,t,h,j)] for p in people) + m[(t,h,j)] == demand.get((t,h,j),0), f"Coverage_{t}_{h}_{j}")
    for p in people:
        for t in tasks:
            for h in hours:
                for j in days:
                    if force.get((p,t,h,j), 0) == 1:
                        model += (1 - x[(p,t,h,j)] <= u[(p,t,h,j)], f"Mandate_{p}_{t}_{h}_{j}")

    # B.2. Captain Presence Mandate (Soft Constraint)
    if captains:
        for j in days:
            for h in hours:
                total_demand = sum(demand.get((t,h,j), 0) for t in tasks)
                if total_demand > 0:
                    model += (lpSum(x[(p,t,h,j)] for p in captains for t in tasks) + k[(h,j)] >= 1, f"Captain_{h}_{j}")

    for p in people:
        for h in hours:
            for j in days:
                if availability.get((p,h,j), 0) == 0:
                    model += (lpSum(x[(p,t,h,j)] for t in tasks) == 0, f"Avail_Zero_{p}_{h}_{j}")
                else:
                    model += (lpSum(x[(p,t,h,j)] for t in tasks) <= 1, f"Avail_Max1_{p}_{h}_{j}")
    for p in people:
        for t in tasks:
            if skills.get((p,t), 0) == 0:
                for h in hours:
                    for j in days:
                        model += (x[(p,t,h,j)] == 0, f"Skill_{p}_{t}_{h}_{j}")
    for j in days:
        for p in people:
            daily_hours = lpSum(x[(p,t,h,j)] for t in tasks for h in hours)
            model += (daily_hours <= n_max[j], f"DayMax_{p}_{j}")
            model += (daily_hours >= n_min[j], f"DayMin_{p}_{j}")
    for p in people:
        total_hours = lpSum(x[(p,t,h,j)] for t in tasks for h in hours for j in days)
        model += (total_hours <= z_max, f"GlobalMax_{p}")
        model += (total_hours >= z_min, f"GlobalMin_{p}")
    for key in consec_keys:
        p,t,h,j = key
        idx = hours.index(h)
        next_h = hours[idx + 1]
        model += (x[(p,t,h,j)] + x[(p,t,next_h,j)] - c[key] <= 1, f"Rotation_{p}_{t}_{h}_{j}")
    for (p1, p2) in friend_pairs:
        for t in tasks:
            for h in hours:
                for j in days:
                    diff_a = x[(p1,t,h,j)] - x[(p2,t,h,j)]
                    diff_b = x[(p2,t,h,j)] - x[(p1,t,h,j)]
                    model += (diff_a <= v[(p1,p2,t,h,j)], f"TogetherA_{p1}_{p2}_{t}_{h}_{j}")
                    model += (diff_b <= v[(p1,p2,t,h,j)], f"TogetherB_{p1}_{p2}_{t}_{h}_{j}")
    for (p1, p2) in enemy_pairs:
        for t in tasks:
            for h in hours:
                for j in days:
                    model += (x[(p1,t,h,j)] + x[(p2,t,h,j)] - w[(p1,p2,t,h,j)] <= 1, f"Separate_{p1}_{p2}_{t}_{h}_{j}")
    for p in people:
        for t in tasks:
            for j in days:
                target = min(min_quota.get((p,t), 0), len(hours))
                if target > 0:
                    model += (lpSum(x[(p,t,h,j)] for h in hours) + q[(p,t)] >= target, f"Quota_{p}_{t}_{j}")
    for p in people:
        for t in tasks:
            for h in hours:
                for j in days:
                    prev = X_prev.get((p,t,h,j), 0)
                    model += (d[(p,t,h,j)] >= prev - x[(p,t,h,j)], f"DevA_{p}_{t}_{h}_{j}")
                    model += (d[(p,t,h,j)] >= x[(p,t,h,j)] - prev, f"DevB_{p}_{t}_{h}_{j}")
    for j in days:
        for p in people:
            work_h0 = lpSum(x[(p,t,hours[0],j)] for t in tasks)
            model += (r[(p, hours[0], j)] == work_h0, f"Start_0_{p}_{j}")
            for i in range(1, len(hours)):
                h_curr = hours[i]; h_prev = hours[i-1]
                curr_work = lpSum(x[(p,t,h_curr,j)] for t in tasks)
                prev_work = lpSum(x[(p,t,h_prev,j)] for t in tasks)
                model += (r[(p, h_curr, j)] >= curr_work - prev_work, f"Start_{h_curr}_{p}_{j}")

    # J.2. Sliding Window (Hard Constraint) - Enforced Rest
    if enforced_rest and max_consec_hours is not None:
        Y = max_consec_hours
        for j in days:
            for p in people:
                for i in range(len(hours) - Y):
                    window_hours = hours[i:i+Y+1]
                    model += (lpSum(x[(p,t,tau,j)] for t in tasks for tau in window_hours) <= Y, f"SlidingWindow_{p}_{j}_{hours[i]}")

    # ============================================================
    # 5. SOLVER
    # ============================================================

    solver = GUROBI(
        msg=True,          # Shows solver progress in console (nodes, gap,etc.)
        timeLimit=1200,    # Max execution seconds. If reached, returns the best solution found
        gapRel=0.01,      # Max % difference between solution and theoretical bound to stop (0.01 = 1%)
        mip=True,          # True=solves integer (binary/integer). False=ignores integrality, solves continuous LP

        # --- MIP Performance ---
        Threads=1,              # Parallel CPU threads. More is not always better due to coordination overhead
        MIPFocus=2,             # Strategy=0=balanced, 1=find feasible quickly, 2=prove optimality, 3=improve bound
        Presolve=2,             # Prior simplification=0=off, 1=conservative, 2=aggressive (reduces model but costs time)
        PrePasses=-1,           # Presolve passes. -1=no limit,Gurobi decides when to stop
        # PreSparsify=1,        # Reduces constraint matrix density to accelerate linear algebra
        # PreDual=-1,           # Dualizes the model in presolve. Sometimes the dual solves faster

        # --- Structure ---
        Symmetry=2,             # Detects interchangeable solutions to prune redundant branches. 0=off, 1=conservative, 2=aggressive
        Disconnected=2,       # Detects independent subproblems to solve them separately
        IntegralityFocus=1,   # Strives harder for integers to be exactly integers (not almost integers)

        # --- Solver Method ---
        Method=1,            # For LP=-1=auto, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent
        # NodeMethod=1,           # LP at MIP nodes=-1=auto, 0=primal, 1=dual (reuses parent basis), 2=barrier
        # ConcurrentMIP=1,      # Launches N MIP solves with different strategies. The first to finish wins

        # --- Solution Pool (stores multiple feasible solutions) ---
        # PoolSolutions=10,     # How many solutions to store besides the optimal one
        # PoolGap=0.1,          # Only stores solutions within this % of the optimal
        # PoolSearchMode=0,     # 0=stores those found along the way, 1=searches more actively, 2=searches for the N best
    )

    # --- Solve the model ---
    model.solve(solver)


    # ============================================================
    # 6. OUTPUT EXTRACTION 
    # ============================================================    
    sol = {}
    sol["status"] = LpStatus[model.status]
    sol["enforced_rest"] = enforced_rest
    assignment = {}
    for j in days:
        assignment[j] = {}
        for p in people:
            assignment[j][p] = {}
            for h in hours:
                assigned_task = None
                for t in tasks:
                    val = x[(p,t,h,j)].varValue
                    if val and val > 0.5:
                        assigned_task = t; break
                assignment[j][p][h] = assigned_task
    sol["assignment"] = assignment

    missing = []
    for t in tasks:
        for h in hours:
            for j in days:
                val = m[(t,h,j)].varValue
                if val and val > 0.01: missing.append(f"{t} @ {h}, {j}: {val:.0f} missing")
    sol["missing"] = missing

    workload = {}
    for p in people:
        total_hrs = sum(x[(p,t,h,j)].varValue for t in tasks for h in hours for j in days if x[(p,t,h,j)].varValue)
        workload[p] = total_hrs
    sol["workload"] = workload
    sol["z_max"] = z_max.varValue; sol["z_min"] = z_min.varValue

    # --- MODIFICACIÓN: GAPS AGRUPADOS POR DÍA ---
    gaps = []
    for j in days:
        day_gaps_found = []
        for p in people:
            starts_count = 0; start_times = []
            for h in hours:
                val = r[(p,h,j)].varValue
                if val and val > 0.5: starts_count += 1; start_times.append(h)
            if starts_count > 1: 
                day_gaps_found.append(f"  • {p}: {starts_count} blocks (Starts: {', '.join(start_times)})")
        
        # Si hubo gaps este día, añadimos cabecera y lista
        if day_gaps_found:
            gaps.append(f"--- {j} ---")
            gaps.extend(day_gaps_found)
    sol["gaps"] = gaps

    soc_issues = []
    # 1. FRIENDS (Coincidencias positivas)
    if friend_pairs:
        for (p1,p2) in friend_pairs:
            for t in tasks:
                for h in hours:
                    for j in days:
                        val1 = x[(p1,t,h,j)].varValue
                        val2 = x[(p2,t,h,j)].varValue
                        if val1 and val1 > 0.5 and val2 and val2 > 0.5:
                            soc_issues.append(f"MATCH: Friends {p1}&{p2} together @ {t}, {h}, {j}")

    # 2. ENEMIES (Violaciones negativas + Aviso de Éxito)
    if enemy_pairs:
        enemy_violations = 0
        for (p1,p2) in enemy_pairs:
            for t in tasks:
                for h in hours:
                    for j in days:
                        val = w[(p1,p2,t,h,j)].varValue
                        if val and val > 0.5: 
                            soc_issues.append(f"VIOLATION: Enemies {p1}&{p2} together @ {t}, {h}, {j}")
                            enemy_violations += 1
        
        # --- MODIFICACIÓN: AVISO SI NO HAY INCUMPLIMIENTOS ---
        if enemy_violations == 0:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")
                            
    sol["social_issues"] = soc_issues

    # Captain presence extraction
    captain_issues = []
    if captains:
        for j in days:
            for h in hours:
                total_demand = sum(demand.get((t,h,j), 0) for t in tasks)
                if total_demand > 0:
                    val = k[(h,j)].varValue
                    if val and val > 0.5:
                        captain_issues.append(f"MISSING CAPTAIN @ {h}, {j}")
        if not captain_issues:
            captain_issues.append("SUCCESS: All active hours have at least one captain on duty.")
    sol["captain_issues"] = captain_issues

    return sol

# ============================================================
# FLET UI
# ============================================================

TASK_COLORS = [
    ("#CE93D8", "#000000"), ("#80DEEA", "#000000"), ("#FFF59D", "#000000"),
    ("#A5D6A7", "#000000"), ("#FFAB91", "#000000"), ("#90CAF9", "#000000"),
    ("#F48FB1", "#000000"), ("#E6EE9C", "#000000"), ("#B0BEC5", "#000000"),
]
UNAVAIL_COLOR = "#ED97B5"

def main(page: ft.Page):
    page.title = "Staff Scheduler"
    page.scroll = ft.ScrollMode.AUTO
    page.window.width = 1200
    page.window.height = 800

    avail_st,demand_st,skills_st = {}, {}, {}
    force_st,social_st,quota_st,rotation_st = {}, {}, {}, {}
    captains_st = {}
    
    # --- CAMBIO: Inicializamos el orden inicial basado en el diccionario por defecto ---
    # Convertimos a lista para poder manipular índices
    weights_st = DEFAULT_WEIGHTS.copy()
    weights_order = list(DEFAULT_WEIGHTS.keys()) 
    weights_enabled = {k: True for k in DEFAULT_WEIGHTS.keys()}

    tf_people = ft.TextField(
        value="Christopher\nBrooklyn\nEzekiel\nBella\nMiles\nClaire\nJaxon\nSkylar", multiline=True, min_lines=8, max_lines=200, label="People (one per line)", expand=True)
    tf_tasks = ft.TextField(
        value="Buy Milk\nMeditate\nPlan Week", multiline=True, min_lines=8, max_lines=200, label="Tasks (one per line)", expand=True)
    tf_hours = ft.TextField(
        value="08:00\n09:00\n10:00\n11:00\n12:00\n13:00\n14:00\n15:00\n16:00\n17:00", multiline=True, min_lines=8, max_lines=200, label="Hours (one per line)", expand=True)
    tf_days = ft.TextField(
        value="Mon\nTue\nWed", multiline=True, min_lines=8, max_lines=200, label="Days (one per line)", expand=True)

    # --- Captain toggles column ---
    captains_col = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=4)

    def build_captains_list(e=None):
        people = [x.strip() for x in tf_people.value.split("\n") if x.strip()]
        captains_col.controls.clear()
        captains_col.controls.append(ft.Text("Captains", weight=ft.FontWeight.BOLD, size=12))
        for p in people:
            if p not in captains_st:
                captains_st[p] = 0
            val = captains_st[p]
            btn = ft.ElevatedButton(
                "Cap" if val else "—", width=55, height=28,
                bgcolor=ft.Colors.AMBER_400 if val else ft.Colors.GREY_400,
                color=ft.Colors.BLACK, style=ft.ButtonStyle(padding=0), data=p)
            def _click(e, _p=p):
                captains_st[_p] = 1 - captains_st[_p]
                e.control.text = "Cap" if captains_st[_p] else "—"
                e.control.bgcolor = ft.Colors.AMBER_400 if captains_st[_p] else ft.Colors.GREY_400
                e.control.update()
            btn.on_click = _click
            captains_col.controls.append(ft.Row([ft.Text(p, size=11, width=90), btn], spacing=4))
        page.update()

    tf_people.on_change = build_captains_list

    # --- Enforced Rest UI controls ---
    err_max_consec = ft.Text("", color=ft.Colors.RED_400, size=11, visible=False)
    tf_max_consec = ft.TextField(value="4", width=100, height=35, text_size=12, label="Max hours (Y)", visible=False, content_padding=ft.padding.all(4))

    def _toggle_enforced_rest(e):
        tf_max_consec.visible = e.control.value
        err_max_consec.visible = False
        page.update()

    sw_enforced_rest = ft.Switch(label="Enforced Rest", value=False, on_change=_toggle_enforced_rest)

    enforced_rest_col = ft.Column([
        sw_enforced_rest,
        tf_max_consec,
        err_max_consec,
    ], spacing=5)

    def dims():
        return (
            [x.strip() for x in tf_people.value.split("\n") if x.strip()],
            [x.strip() for x in tf_tasks.value.split("\n") if x.strip()],
            [x.strip() for x in tf_hours.value.split("\n") if x.strip()],
            [x.strip() for x in tf_days.value.split("\n") if x.strip()],
        )

    avail_ct    = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    demand_ct   = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    skills_ct   = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    force_ct    = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    social_ct   = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    quota_ct    = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    rotation_ct = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    weights_ct  = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
    output_ct   = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)

    W_LBL = 90; W_BTN = 58; W_CELL = 65

    def make_toggle(sd, key, default):
        if key not in sd: sd[key] = default
        val = sd[key]
        btn = ft.ElevatedButton(
            str(val), width=W_BTN, height=30, data=key,
            bgcolor=ft.Colors.GREEN_400 if val else ft.Colors.RED_400,
            color=ft.Colors.WHITE, style=ft.ButtonStyle(padding=0))
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = 1 - _sd[_k]
            e.control.text = str(_sd[_k])
            e.control.bgcolor = ft.Colors.GREEN_400 if _sd[_k] else ft.Colors.RED_400
            e.control.update()
        btn.on_click = _click
        return btn

    def lbl(text,w=W_LBL): return ft.Container(ft.Text(text,size=11, no_wrap=True), width=w)

    def plbl(name, w=W_LBL):
        """Person label — captain names shown in yellow."""
        is_cap = captains_st.get(name, 0) == 1
        return ft.Container(
            ft.Text(name, size=11, no_wrap=True,
                    weight=ft.FontWeight.BOLD if is_cap else None,
                    color=CAPTAIN_BG if is_cap else None),
            width=w,
        )

    def hdr_row(labels, w=W_BTN):
        return ft.Row([ft.Container(width=W_LBL)] + [ft.Container(ft.Text(l, size=10), width=w) for l in labels], spacing=2)

    def build_avail():
        people,tasks,hours,days = dims(); avail_ct.controls.clear()
        for j in days:
            avail_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            avail_ct.controls.append(hdr_row(hours))
            for p in people:
                avail_ct.controls.append(ft.Row([plbl(p)] + [make_toggle(avail_st,(p,h,j), 1) for h in hours], spacing=2))
            avail_ct.controls.append(ft.Divider())
        page.update()

    def build_demand():
        people,tasks,hours,days = dims(); demand_ct.controls.clear()
        for j in days:
            demand_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            demand_ct.controls.append(hdr_row(hours,W_CELL))
            for t in tasks:
                cells = []
                for h in hours:
                    k = (t,h,j); 
                    if k not in demand_st: demand_st[k] = "1"
                    tf = ft.TextField(value=demand_st[k], width=W_CELL, height=35, text_size=12, data=k, content_padding=ft.padding.all(4))
                    def _ch(e, _k=k): demand_st[_k] = e.control.value
                    tf.on_change = _ch; cells.append(tf)
                demand_ct.controls.append(ft.Row([lbl(t)] + cells, spacing=2))
            demand_ct.controls.append(ft.Divider())
        page.update()

    def build_skills():
        people,tasks,hours,days = dims(); skills_ct.controls.clear()
        skills_ct.controls.append(hdr_row(tasks,70))
        for p in people:
            skills_ct.controls.append(ft.Row([plbl(p)] + [make_toggle(skills_st,(p,t), 1) for t in tasks], spacing=2))
        page.update()

    def build_force():
        people,tasks,hours,days = dims(); force_ct.controls.clear()
        for t in tasks:
            for j in days:
                force_ct.controls.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=14))
                force_ct.controls.append(hdr_row(hours))
                for p in people:
                    force_ct.controls.append(ft.Row([plbl(p)] + [make_toggle(force_st,(p,t,h,j), 0) for h in hours], spacing=2))
                force_ct.controls.append(ft.Divider())
        page.update()

    def build_social():
        people,tasks,hours,days = dims(); social_ct.controls.clear()
        if len(people) < 2: page.update(); return
        social_ct.controls.append(hdr_row(people[1:], 70))
        _map_lbl = {0: "~", 1: "+", -1: "-"}; _map_clr = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_400, -1: ft.Colors.RED_400}
        _next_val = {0: 1, 1: -1, -1: 0}
        for i, p1 in enumerate(people):
            cells = []
            for p2 in people[1:]:
                j2 = people.index(p2)
                if j2 > i:
                    k = (p1,p2); 
                    if k not in social_st: social_st[k] = 0
                    sv = social_st[k]
                    btn = ft.ElevatedButton(_map_lbl[sv], width=70, height=30, data=k, bgcolor=_map_clr[sv], color=ft.Colors.WHITE, style=ft.ButtonStyle(padding=0))
                    def _click(e, _k=k):
                        social_st[_k] = _next_val[social_st[_k]]
                        nv = social_st[_k]
                        e.control.text = _map_lbl[nv]; e.control.bgcolor = _map_clr[nv]; e.control.update()
                    btn.on_click = _click; cells.append(btn)
                else: cells.append(ft.Container(width=70))
            if cells: social_ct.controls.append(ft.Row([plbl(p1)] + cells, spacing=2))
        page.update()

    def build_quota():
        people,tasks,hours,days = dims(); quota_ct.controls.clear()
        quota_ct.controls.append(hdr_row(tasks,70))
        for p in people:
            cells = []
            for t in tasks:
                k = (p,t); 
                if k not in quota_st: quota_st[k] = "0"
                tf = ft.TextField(value=quota_st[k], width=70, height=35, text_size=12, data=k, content_padding=ft.padding.all(4))
                def _ch(e, _k=k): quota_st[_k] = e.control.value
                tf.on_change = _ch; cells.append(tf)
            quota_ct.controls.append(ft.Row([plbl(p)] + cells, spacing=2))
        page.update()

    def build_rotation():
        people,tasks,hours,days = dims(); rotation_ct.controls.clear()
        for t in tasks:
            if t not in rotation_st: rotation_st[t] = 1
            sw = ft.Switch(label=t,value=rotation_st[t]==1, data=t)
            def _ch(e, _t=t): rotation_st[_t] = 1 if e.control.value else 0
            sw.on_change = _ch; rotation_ct.controls.append(sw)
        page.update()
        
    def build_weights():
 
        items_controls = []
        for i, key in enumerate(weights_order):
            weights_st[key] = SORTED_VALUES[i] if weights_enabled[key] else 0

        weights_ct.controls.clear()
        
        # Encabezado
        header = ft.Column([
            ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=16),
            ft.Text("Drag items to reorder. Top items = Higher Cost.", italic=True, size=12),
            ft.Divider(),
        ], spacing=5)

        # 2. Función que maneja el reordenamiento automático
        def handle_reorder(e):
            # Flet nos da el índice antiguo y el nuevo automáticamente
            item = weights_order.pop(e.old_index)
            weights_order.insert(e.new_index, item)
            # Reconstruimos para actualizar los números de Costo (Valores)
            build_weights()

        # 3. Construir los items de la lista
        for i, key in enumerate(weights_order):
            val = SORTED_VALUES[i] if weights_enabled[key] else 0
            
            # Switch de activación
            sw = ft.Switch(value=weights_enabled[key], data=key)
            def _toggle(e, _k=key): weights_enabled[_k] = e.control.value; build_weights()
            sw.on_change = _toggle
            
            # Tarjeta visual
            card = ft.Container(
                content=ft.Row([
                    ft.Text(f"#{i+1}", width=30, weight=ft.FontWeight.BOLD),
                    ft.Text(key, expand=True, weight=ft.FontWeight.W_800),
                    ft.Text(f"{val}            ", color=ft.Colors.BLACK if weights_enabled[key] else ft.Colors.GREY_500, size=16),
                    sw,
                ], alignment=ft.MainAxisAlignment.START),
                padding=10,
                bgcolor=ft.Colors.BLUE if weights_enabled[key] else ft.Colors.GREY_700,
                border=ft.border.all(1, ft.Colors.GREY_500),
                border_radius=8,
                margin=ft.margin.only(bottom=5) # <--- AQUÍ AÑADIMOS EL ESPACIADO
            )
            items_controls.append(card)

        # 4. Componente nativo de lista reordenable
        # Eliminamos 'spacing' y 'divider_color' que causaban el error
        r_list = ft.ReorderableListView(
            controls=items_controls,
            on_reorder=handle_reorder
        )

        # 5. Contenedor para limitar el ancho ("pequeñita") y centrar
        layout = ft.Column(
            controls=[
                header,
                ft.Container(
                    content=r_list,
                    width=420,  # Ancho restringido
                    height=500, # Altura fija para hacer scroll si es necesario
                )
            ],
            width=420,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        weights_ct.controls.append(
            ft.Row([layout], alignment=ft.MainAxisAlignment.CENTER)
        )
        
        page.update()

    builders = {1: build_avail, 2: build_demand, 3: build_skills,
                4: build_force, 5: build_social, 6: build_quota, 7: build_rotation, 8: build_weights}

    def on_tab_change(e):
        idx = e.control.selected_index
        if idx in builders: builders[idx]()

    def build_output_grid(sol, people,tasks,hours,days, availability):
        output_ct.controls.clear()
        output_ct.controls.append(ft.Text(f"Status: {sol['status']}", weight=ft.FontWeight.BOLD, size=16))
        output_ct.controls.append(ft.Divider())
        tc = {}
        for i, t in enumerate(tasks): bg, fg = TASK_COLORS[i % len(TASK_COLORS)]; tc[t] = (bg, fg)
        CW = 75; Ch = 36; NW = 110; TW = 50
        for j in days:
            output_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=16))
            hdr = [ft.Container(ft.Text("Person", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE), width=NW, height=Ch,bgcolor="#455A64", alignment=ft.alignment.center, border=ft.border.all(1, "#37474F"))]
            for h in hours: hdr.append(ft.Container(ft.Text(h,size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE), width=CW, height=Ch,bgcolor="#455A64", alignment=ft.alignment.center, border=ft.border.all(1, "#37474F")))
            hdr.append(ft.Container(ft.Text("Total", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE), width=TW, height=Ch,bgcolor="#455A64", alignment=ft.alignment.center, border=ft.border.all(1, "#37474F")))
            output_ct.controls.append(ft.Row(hdr, spacing=0))
            asgn = sol["assignment"][j]
            for idx_p,p in enumerate(people):
                is_cap = captains_st.get(p, 0) == 1
                row_bg = "#263238" if idx_p % 2 == 0 else "#37474F"
                name_color = CAPTAIN_BG if is_cap else ft.Colors.WHITE
                cells = [ft.Container(ft.Text(p,size=12, weight=ft.FontWeight.BOLD, color=name_color), width=NW, height=Ch,bgcolor=row_bg, alignment=ft.alignment.center_left, padding=ft.padding.only(left=8), border=ft.border.all(1, "#455A64"))]
                total = 0
                for h in hours:
                    task = asgn[p][h]; avail = availability.get((p,h,j), 1)
                    if task: bg, fg = tc[task]; total += 1; cell = ft.Container(ft.Text(task, size=11, weight=ft.FontWeight.BOLD, color=fg, text_align=ft.TextAlign.CENTER), width=CW, height=Ch,bgcolor=bg, alignment=ft.alignment.center, border=ft.border.all(1, "#455A64"), border_radius=4)
                    elif avail == 0: cell = ft.Container(width=CW, height=Ch,bgcolor=UNAVAIL_COLOR, border=ft.border.all(1, "#455A64"), border_radius=4)
                    else: cell = ft.Container(width=CW, height=Ch,bgcolor=row_bg, border=ft.border.all(1, "#455A64"))
                    cells.append(cell)
                cells.append(ft.Container(ft.Text(str(int(total)), size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, text_align=ft.TextAlign.CENTER), width=TW, height=Ch,bgcolor=row_bg, alignment=ft.alignment.center, border=ft.border.all(1, "#455A64")))
                output_ct.controls.append(ft.Row(cells, spacing=0))
            legend_items = [ft.Container(ft.Text(t,size=10, weight=ft.FontWeight.BOLD, color=tc[t][1]), bgcolor=tc[t][0], padding=ft.padding.symmetric(6, 10), border_radius=4) for t in tasks]
            legend_items.append(ft.Container(ft.Text("Unavailable", size=10), bgcolor=UNAVAIL_COLOR, padding=ft.padding.symmetric(6, 10), border_radius=4))
            output_ct.controls.append(ft.Row(legend_items, spacing=8))
            output_ct.controls.append(ft.Divider())
        output_ct.controls.append(ft.Text("MISSING STAFF", weight=ft.FontWeight.BOLD, size=14))
        if sol["missing"]:
            for line in sol["missing"]: output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else: output_ct.controls.append(ft.Text("  None -- all demand covered!", size=12, italic=True))
        output_ct.controls.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=14))
        for p in people: output_ct.controls.append(ft.Text(f"  {p}: {sol['workload'][p]:.0f} hours", size=12))
        output_ct.controls.append(ft.Text(f"  Global range: max={sol['z_max']:.0f}, min={sol['z_min']:.0f}", size=12, italic=True))
        if not sol.get("enforced_rest", False):
            output_ct.controls.append(ft.Text("SHIFT SEGMENTS / FRAGMENTATION", weight=ft.FontWeight.BOLD, size=14))
            if sol["gaps"]:
                for line in sol["gaps"]: output_ct.controls.append(ft.Text(f"  {line}", size=12))
            else: output_ct.controls.append(ft.Text("  Single block shifts! (Perfect)", size=12, italic=True))
        output_ct.controls.append(ft.Text("SOCIAL", weight=ft.FontWeight.BOLD, size=14))
        if sol["social_issues"]:
            for line in sol["social_issues"]: output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else: output_ct.controls.append(ft.Text("  None -- all respected!", size=12, italic=True))
        # Captain presence output
        output_ct.controls.append(ft.Text("CAPTAIN PRESENCE", weight=ft.FontWeight.BOLD, size=14))
        if sol["captain_issues"]:
            for line in sol["captain_issues"]: output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else: output_ct.controls.append(ft.Text("  No captains designated.", size=12, italic=True))

    def do_solve(e):
        people, tasks, hours, days = dims()

        # --- Validación Enforced Rest ---
        enforced_rest = sw_enforced_rest.value
        max_consec_hours = None
        if enforced_rest:
            raw_y = tf_max_consec.value.strip()
            try:
                max_consec_hours = int(raw_y)
                if str(max_consec_hours) != raw_y or max_consec_hours < 1:
                    raise ValueError
            except (ValueError, TypeError):
                err_max_consec.value = "Error: Y must be a positive integer (no decimals or text)."
                err_max_consec.visible = True
                page.update()
                return
            err_max_consec.visible = False
            page.update()

        availability = {}
        for p in people:
            for h in hours:
                for j in days: availability[(p,h,j)] = avail_st.get((p,h,j), 1)
        demand = {}
        for t in tasks:
            for h in hours:
                for j in days:
                    raw = demand_st.get((t,h,j), "1")
                    try: demand[(t,h,j)] = int(raw)
                    except: demand[(t,h,j)] = 0
        skills = {}
        for p in people:
            for t in tasks: skills[(p,t)] = skills_st.get((p,t), 1)
        force = {}
        for p in people:
            for t in tasks:
                for h in hours:
                    for j in days: force[(p,t,h,j)] = force_st.get((p,t,h,j), 0)
        social = {}
        for i, p1 in enumerate(people):
            for p2 in people[i+1:]: social[(p1,p2)] = social_st.get((p1,p2), 0)
        mq = {}
        for p in people:
            for t in tasks:
                raw = quota_st.get((p,t), "0")
                try: mq[(p,t)] = int(raw)
                except: mq[(p,t)] = 0
        rotation = {t: rotation_st.get(t,1) for t in tasks}
        pref_cost = {(p,t): 1 for p in people for t in tasks}
        X_prev = {(p,t,h,j): 0 for p in people for t in tasks for h in hours for j in days}
        
        # --- CAMBIO: Los pesos ya están actualizados en weights_st gracias a build_weights ---
        weights = weights_st.copy()

        # Extract captains list from captains_st
        captains = [p for p in people if captains_st.get(p, 0) == 1]

        data = dict(people=people, tasks=tasks, hours=hours, days=days,
                    availability=availability, demand=demand, skills=skills,
                    force=force, social=social, min_quota=mq,
                    pref_cost=pref_cost, rotation=rotation, X_prev=X_prev, weights=weights,
                    enforced_rest=enforced_rest, max_consec_hours=max_consec_hours,
                    captains=captains)

        output_ct.controls.clear()
        output_ct.controls.append(ft.ProgressRing())
        output_ct.controls.append(ft.Text("Solving...", italic=True))
        tabs.selected_index = 9
        page.update()

        try:
            sol = solve_model(data) 
            build_output_grid(sol, people, tasks, hours, days, availability)
        except Exception as ex:
            output_ct.controls.clear()
            output_ct.controls.append(ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=14))
            import traceback
            traceback.print_exc()
        page.update()

    solve_btn = ft.ElevatedButton("SOLVE", on_click=do_solve, bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE, height=45, width=200)

    # People TextField needs a fixed width now that it shares space with captains_col
    tf_people.expand = False
    tf_people.width = 180

    dim_tab_content = ft.Container(content=ft.Row(controls=[tf_people, captains_col, tf_tasks, tf_hours, tf_days, enforced_rest_col], spacing=20, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START), padding=20, expand=True)

    tabs = ft.Tabs(
        selected_index=0, on_change=on_tab_change, expand=True,
        tabs=[
            ft.Tab(text="Dimensions",   content=dim_tab_content),
            ft.Tab(text="Availability", content=ft.Container(avail_ct,padding=10, expand=True)),
            ft.Tab(text="Demand",       content=ft.Container(demand_ct,padding=10, expand=True)),
            ft.Tab(text="Skills",       content=ft.Container(skills_ct,padding=10, expand=True)),
            ft.Tab(text="Force",        content=ft.Container(force_ct,padding=10, expand=True)),
            ft.Tab(text="Social",       content=ft.Container(social_ct,padding=10, expand=True)),
            ft.Tab(text="Min Quota",    content=ft.Container(quota_ct,padding=10, expand=True)),
            ft.Tab(text="Rotation",     content=ft.Container(rotation_ct,padding=10, expand=True)),
            ft.Tab(text="Weights",      content=ft.Container(weights_ct,padding=10, expand=True)),
            ft.Tab(text="Output",       content=ft.Container(output_ct,padding=10, expand=True)),
        ])

    page.add(ft.Row([solve_btn], alignment=ft.MainAxisAlignment.CENTER), tabs)

    # Build captain list on startup so it matches initial people
    build_captains_list()

ft.app(target=main)