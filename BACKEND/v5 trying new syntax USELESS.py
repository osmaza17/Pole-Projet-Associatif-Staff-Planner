import flet as ft
from pulp import *


def solve_model(data: dict):

    # ============================================================
    # 0. DEFINE PARAMETERS
    # ============================================================

    # --- Dimensions ---
    people = data["people"]
    tasks = data["tasks"]
    hours = data["hours"]  
    days = data["days"]


    # --- INPUT Parameters ---
    # A[p,h,j] : Availability (1 = present)
    availability   = data["availability"]

    # D[t,h,j] : Demand (how many people needed)
    demand         = data["demand"]

    # S[p,t] : Skill matrix (1 = qualified)
    skills         = data["skills"]

    # F[p,t,h,j] : Force / Mandate matrix (1 = manager requires it)
    force          = data["force"]

    # E[p1,p2] : Social affinity  (1 = friends/together, -1 = enemies/separate, 0 = neutral)
    social         = data["social"]

    # L[p,t] : Min-quota wish (1 = person wants to try task at least once)
    min_quota      = data["min_quota"]

    # B[p,t] : Preference cost (higher = more dislike)
    pref_cost      = data["pref_cost"]

    # R[t] : Rotation switch (1 = no consecutive hours on that task)
    rotation       = data["rotation"]

    # X_prev[p,t,h,j] : Anchor plan from previous schedule
    X_prev         = data["X_prev"]


    # --- Priority Weights ---
    W = data["weights"]

    W_COVERAGE  = W["W_COVERAGE"]
    W_MANDATE   = W["W_MANDATE"]
    W_STABILITY = W["W_STABILITY"]
    W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_TOTAL  = W["W_EQ_TOTAL"]
    W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"]
    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"]
    W_PREF      = W["W_PREF"]


    # --- Auxiliary Functions ---
    next_hour = {hours[i]: hours[i+1] for i in range(len(hours)-1)}
    friend_pairs = [(p1,p2) for (p1,p2),v in social.items() if v==1]
    enemy_pairs  = [(p1,p2) for (p1,p2),v in social.items() if v==-1]


    # ============================================================
    # 1.  CREATE THE MODEL
    # ============================================================

    model = LpProblem("Staff_Scheduling", LpMinimize)


    # ============================================================
    # 2.  DECISION VARIABLES
    # ============================================================

    x = LpVariable.dicts("x", (people, tasks, hours, days), cat=LpBinary)

    m = LpVariable.dicts("m", (tasks, hours, days), lowBound=0, cat=LpInteger)

    u = LpVariable.dicts("u", (people, tasks, hours, days), lowBound=0, cat=LpInteger)

    d = LpVariable.dicts("d", (people, tasks, hours, days), cat=LpBinary)

    n_max = LpVariable.dicts("n_max", days, lowBound=0, cat=LpInteger)
    n_min = LpVariable.dicts("n_min", days, lowBound=0, cat=LpInteger)

    z_max = LpVariable("z_max", lowBound=0, cat=LpInteger)
    z_min = LpVariable("z_min", lowBound=0, cat=LpInteger)


    consec_keys = [
        (p,t,h,j)
        for p in people for t in tasks for h in hours[:-1] for j in days
        if rotation.get(t, 0) == 1
    ]
    c = LpVariable.dicts("c", consec_keys, cat=LpBinary)


    friend_pairs = [(p1,p2) for (p1,p2), val in social.items() if val == 1]
    enemy_pairs  = [(p1,p2) for (p1,p2), val in social.items() if val == -1]

    v = LpVariable.dicts(
        "v",
        ((p1,p2,t,h,j) for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days),
        lowBound=0, cat=LpContinuous
    )

 
    w = LpVariable.dicts(
        "w",
        ((p1,p2,t,h,j) for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days),
        lowBound=0, cat=LpContinuous
    )


    q = LpVariable.dicts("q", (people, tasks), cat=LpBinary)


    r = LpVariable.dicts("r", (people, hours, days), cat=LpBinary)


    # ============================================================
    # 3.  OBJECTIVE FUNCTION
    # ============================================================

    obj = []

    # 1. Coverage
    obj += [W_COVERAGE * m[t][h][j] for t in tasks for h in hours for j in days]

    # 2. Mandate
    obj += [W_MANDATE * u[p][t][h][j] for p in people for t in tasks for h in hours for j in days]

    # 3. Stability
    obj += [W_STABILITY * d[p][t][h][j] for p in people for t in tasks for h in hours for j in days]

    # 4. Daily Equity
    obj += [W_EQ_DAY * (n_max[j] - n_min[j]) for j in days]

    # 5. Global Equity
    obj.append(W_EQ_TOTAL * (z_max - z_min))

    # 6. Rotation Fatigue
    obj += [W_ROTATION * c[k] for k in consec_keys]

    # 7. Social
    if friend_pairs:
        obj += [W_SOCIAL * v[(p1,p2,t,h,j)] for (p1,p2) in friend_pairs for t in tasks for h in hours for j in days]
    if enemy_pairs:
        obj += [W_SOCIAL * w[(p1,p2,t,h,j)] for (p1,p2) in enemy_pairs for t in tasks for h in hours for j in days]

    # 8. Gap Penalty
    obj += [W_GAP * r[p][h][j] for p in people for h in hours for j in days]

    # 9. Quota
    obj += [W_QUOTA * q[p][t] for p in people for t in tasks]

    # 10. Preferences
    obj += [W_PREF * pref_cost.get((p,t), 0) * x[p][t][h][j] for p in people for t in tasks for h in hours for j in days]

    model += lpSum(obj), "Total_Penalty"


    # ============================================================
    # 4.  CONSTRAINTS
    # ============================================================

    # --- A. Task Coverage ---
    for t in tasks:
        for h in hours:
            for j in days:
                # Usamos x[p][t][h][j] y m[t][h][j]
                model += (lpSum(x[p][t][h][j] for p in people) + m[t][h][j] == demand.get((t,h,j),0), f"Coverage_{t}_{h}_{j}")

    # --- B. Manual Mandates ---
    for p in people:
        for t in tasks:
            for h in hours:
                for j in days:
                    if force.get((p,t,h,j), 0) == 1:
                        model += (1 - x[p][t][h][j] <= u[p][t][h][j], f"Mandate_{p}_{t}_{h}_{j}")

    # --- C. Physical Availability ---
    for p in people:
        for h in hours:
            for j in days:
                if availability.get((p,h,j), 0) == 0:
                    # Optimización: Fijar a 0 directamente si no está disponible
                    model += (lpSum(x[p][t][h][j] for t in tasks) == 0, f"Avail_Zero_{p}_{h}_{j}")
                else:
                    model += (lpSum(x[p][t][h][j] for t in tasks) <= 1, f"Avail_Max1_{p}_{h}_{j}")

    # --- D. Skill Filter ---
    for p in people:
        for t in tasks:
            if skills.get((p,t), 0) == 0:
                for h in hours:
                    for j in days:
                        model += (x[p][t][h][j] == 0, f"Skill_{p}_{t}_{h}_{j}")

    # --- E. Double-Squeeze Equity ---
    for j in days:
        for p in people:
            daily_hours = lpSum(x[p][t][h][j] for t in tasks for h in hours)
            model += (daily_hours <= n_max[j], f"DayMax_{p}_{j}")
            model += (daily_hours >= n_min[j], f"DayMin_{p}_{j}")

    for p in people:
        total_hours = lpSum(x[p][t][h][j] for t in tasks for h in hours for j in days)
        model += (total_hours <= z_max, f"GlobalMax_{p}")
        model += (total_hours >= z_min, f"GlobalMin_{p}")

    # --- F. Rotation Fatigue ---
    for key in consec_keys:
        p, t, h, j = key
        # Accedemos a next_hour (asumiendo que es una lista o dict de h -> h+1)
        # x[p][t][h][j] + x[p][t][h+1][j] - c...
        next_h = hours[hours.index(h) + 1] # O tu dict next_hour
        model += (x[p][t][h][j] + x[p][t][next_h][j] - c[key] <= 1, f"Rotation_{p}_{t}_{h}_{j}")

    # --- G. Social Constraints ---
    # Friends
    for (p1, p2) in friend_pairs:
        for t in tasks:
            for h in hours:
                for j in days:
                    diff_a = x[p1][t][h][j] - x[p2][t][h][j]
                    diff_b = x[p2][t][h][j] - x[p1][t][h][j]
                    model += (diff_a <= v[(p1,p2,t,h,j)], f"TogetherA_{p1}_{p2}_{t}_{h}_{j}")
                    model += (diff_b <= v[(p1,p2,t,h,j)], f"TogetherB_{p1}_{p2}_{t}_{h}_{j}")

    # Enemies
    for (p1, p2) in enemy_pairs:
        for t in tasks:
            for h in hours:
                for j in days:
                    model += (x[p1][t][h][j] + x[p2][t][h][j] - w[(p1,p2,t,h,j)] <= 1, f"Separate_{p1}_{p2}_{t}_{h}_{j}")

    # --- H. Minimum Quota ---
    for p in people:
        for t in tasks:
            for j in days:
                # Sumamos todas las horas para esa tarea
                total_task_hours = lpSum(x[p][t][h][j] for h in hours)

                # Nos aseguramos de que la cuota mínima no puede ser mayor que el número de horas. Si no, toma el menor entre los dos
                wanted = min(min_quota.get((p,t), 0), len(hours))
                print(wanted)
                if wanted > 0:
                    model += (total_task_hours + q[p][t] >= wanted, f"Quota_{p}_{t}_{j}")

    # --- I. Stability / Deviation ---
    for p in people:
        for t in tasks:
            for h in hours:
                for j in days:
                    prev = X_prev.get((p,t,h,j), 0)
                    # d >= prev - new
                    model += (d[p][t][h][j] >= prev - x[p][t][h][j], f"DevA_{p}_{t}_{h}_{j}")
                    # d >= new - prev
                    model += (d[p][t][h][j] >= x[p][t][h][j] - prev, f"DevB_{p}_{t}_{h}_{j}")

    # --- J. Shift Continuity / Block Counting ---
    for j in days:
        for p in people:
            
            # Hora 0: Si trabaja, es un inicio de bloque
            work_h0 = lpSum(x[p][t][hours[0]][j] for t in tasks)
            model += (r[p][hours[0]][j] == work_h0, f"Start_0_{p}_{j}")
            
            # Horas siguientes
            for i in range(1, len(hours)):
                h_curr = hours[i]
                h_prev = hours[i-1]
                
                curr_work = lpSum(x[p][t][h_curr][j] for t in tasks)
                prev_work = lpSum(x[p][t][h_prev][j] for t in tasks)
                
                # Rising Edge Detector: r >= curr - prev
                model += (r[p][h_curr][j] >= curr_work - prev_work, f"Start_{h_curr}_{p}_{j}")



    # ============================================================
    # 5.  SOLVE
    # ============================================================

    solver = GUROBI(
        msg=True,          # Shows solver progress in console (nodes, gap,etc.)
        timeLimit=1200,    # Max execution seconds. If reached, returns the best solution found
        gapRel=0.001,      # Max % difference between solution and theoretical bound to stop (0.01 = 1%)
        mip=True,          # True=solves integer (binary/integer). False=ignores integrality, solves continuous LP

        # --- MIP Performance ---
        Threads=12,              # Parallel CPU threads. More is not always better due to coordination overhead
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
    # 6.  OUTPUT
    # ============================================================
    sol = {}
    sol["status"] = LpStatus[model.status]

    # --- 1. Assignments ---
    # Access: x[p][t][h][j]
    assignment = {}
    for j in days:
        assignment[j] = {}
        for p in people:
            assignment[j][p] = {}
            for h in hours:
                assigned_task = None
                for t in tasks:
                    val = x[p][t][h][j].varValue
                    if val and val > 0.5:
                        assigned_task = t
                        break
                assignment[j][p][h] = assigned_task
    sol["assignment"] = assignment

    # --- 2. Missing Staff ---
    # Access: m[t][h][j]
    missing = []
    for t in tasks:
        for h in hours:
            for j in days:
                val = m[t][h][j].varValue
                if val and val > 0.01:
                    missing.append(f"{t} @ {h}, {j}: {val:.0f} missing")
    sol["missing"] = missing

    # --- 3. Workload ---
    workload = {}
    for p in people:
        total_hrs = sum(x[p][t][h][j].varValue 
                        for t in tasks 
                        for h in hours 
                        for j in days 
                        if x[p][t][h][j].varValue)
        workload[p] = total_hrs
    sol["workload"] = workload
    
    # Global bounds are simple variables
    sol["z_max"] = z_max.varValue
    sol["z_min"] = z_min.varValue

    # --- 4. Gaps / Fragmentation (Using 'r') ---
    # r[p][h][j] == 1 means a block started. 
    # If sum(r) > 1 per day, there is a gap/break.
    gaps = []
    for p in people:
        for j in days:
            starts_count = 0
            start_times = []
            for h in hours:
                val = r[p][h][j].varValue
                if val and val > 0.5:
                    starts_count += 1
                    start_times.append(h)
            
            # If more than 1 start, strictly speaking, the schedule is fragmented
            if starts_count > 1:
                gaps.append(f"{p} on {j}: {starts_count} blocks (Starts at: {', '.join(start_times)})")
            elif starts_count == 1:
                # Optional: log single starts just to know when they start
                pass 
                
    sol["gaps"] = gaps

    # --- 5. Social Issues ---
    # These variables (v, w) were kept as TUPLE keys in the definition for efficiency
    # So we keep the tuple access logic here: v[(...)]
    soc_issues = []
    if friend_pairs:
        for (p1,p2) in friend_pairs:
            for t in tasks:
                for h in hours:
                    for j in days:
                        # Tuple access is correct here because we defined v with tuples
                        val = v[(p1,p2,t,h,j)].varValue
                        if val and val > 0.5:
                            soc_issues.append(f"Friends {p1}&{p2} separated @ {t}, {h}, {j}")
    
    if enemy_pairs:
        for (p1,p2) in enemy_pairs:
            for t in tasks:
                for h in hours:
                    for j in days:
                        # Tuple access is correct here
                        val = w[(p1,p2,t,h,j)].varValue
                        if val and val > 0.5:
                            soc_issues.append(f"Enemies {p1}&{p2} together @ {t}, {h}, {j}")
    sol["social_issues"] = soc_issues

    return sol


# ============================================================
# FLET UI
# ============================================================

# Palette for task colors (cycles if more tasks than colors)
TASK_COLORS = [
    ("#CE93D8", "#000000"),  # purple
    ("#80DEEA", "#000000"),  # cyan
    ("#FFF59D", "#000000"),  # yellow
    ("#A5D6A7", "#000000"),  # green
    ("#FFAB91", "#000000"),  # orange
    ("#90CAF9", "#000000"),  # blue
    ("#F48FB1", "#000000"),  # pink
    ("#E6EE9C", "#000000"),  # lime
    ("#B0BEC5", "#000000"),  # grey
]
UNAVAIL_COLOR = "#ED97B5"   # light pink for unavailable slots


def main(page: ft.Page):
    page.title = "Staff Scheduler"
    page.scroll = ft.ScrollMode.AUTO
    page.window.width = 1200
    page.window.height = 800

    avail_st,demand_st,skills_st = {}, {}, {}
    force_st,social_st,quota_st,rotation_st = {}, {}, {}, {}

    tf_people = ft.TextField(
        value="Christopher\nBrooklyn\nEzekiel\nBella\nMiles\nClaire\nJaxon\nSkylar", multiline=True, min_lines=8, max_lines=200,
        label="People (one per line)", expand=True)
    tf_tasks = ft.TextField(
        value="Comprar leche\nMeditar\nPlanificar semana", multiline=True, min_lines=8, max_lines=200,
        label="Tasks (one per line)", expand=True)
    tf_hours = ft.TextField(
        value="08:00\n09:00\n10:00\n11:00\n12:00\n13:00\n14:00\n15:00\n16:00\n17:00",
        multiline=True, min_lines=8, max_lines=200,
        label="Hours (one per line)", expand=True)
    tf_days = ft.TextField(
        value="Mon\nTue\nWed", multiline=True, min_lines=8, max_lines=200,
        label="Days (one per line)", expand=True)

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

    def lbl(text,w=W_LBL):
        return ft.Container(ft.Text(text,size=11, no_wrap=True), width=w)

    def hdr_row(labels, w=W_BTN):
        return ft.Row(
            [ft.Container(width=W_LBL)] + [ft.Container(ft.Text(l, size=10), width=w) for l in labels],
            spacing=2)

    def build_avail():
        P,T,H,D = dims(); avail_ct.controls.clear()
        for j in D:
            avail_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            avail_ct.controls.append(hdr_row(H))
            for p in P:
                avail_ct.controls.append(ft.Row(
                    [lbl(p)] + [make_toggle(avail_st,(p,h,j), 1) for h in H], spacing=2))
            avail_ct.controls.append(ft.Divider())
        page.update()

    def build_demand():
        P,T,H,D = dims(); demand_ct.controls.clear()
        for j in D:
            demand_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
            demand_ct.controls.append(hdr_row(H,W_CELL))
            for t in T:
                cells = []
                for h in H:
                    k = (t,h,j)
                    if k not in demand_st: demand_st[k] = "1"
                    tf = ft.TextField(value=demand_st[k], width=W_CELL, height=35,
                                      text_size=12, data=k, content_padding=ft.padding.all(4))
                    def _ch(e, _k=k): demand_st[_k] = e.control.value
                    tf.on_change = _ch; cells.append(tf)
                demand_ct.controls.append(ft.Row([lbl(t)] + cells, spacing=2))
            demand_ct.controls.append(ft.Divider())
        page.update()

    def build_skills():
        P,T,H,D = dims(); skills_ct.controls.clear()
        skills_ct.controls.append(hdr_row(T,70))
        for p in P:
            skills_ct.controls.append(ft.Row(
                [lbl(p)] + [make_toggle(skills_st,(p,t), 1) for t in T], spacing=2))
        page.update()

    def build_force():
        P,T,H,D = dims(); force_ct.controls.clear()
        for t in T:
            for j in D:
                force_ct.controls.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=14))
                force_ct.controls.append(hdr_row(H))
                for p in P:
                    force_ct.controls.append(ft.Row(
                        [lbl(p)] + [make_toggle(force_st,(p,t,h,j), 0) for h in H], spacing=2))
                force_ct.controls.append(ft.Divider())
        page.update()

    def build_social():
        P,T,H,D = dims(); social_ct.controls.clear()
        if len(P) < 2: page.update(); return
        social_ct.controls.append(hdr_row(P[1:], 70))
        _map_lbl  = {0: "~", 1: "+", -1: "-"}
        _map_clr  = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_400, -1: ft.Colors.RED_400}
        _next_val = {0: 1, 1: -1, -1: 0}
        for i, p1 in enumerate(P):
            cells = []
            for p2 in P[1:]:
                j2 = P.index(p2)
                if j2 > i:
                    k = (p1,p2)
                    if k not in social_st: social_st[k] = 0
                    sv = social_st[k]
                    btn = ft.ElevatedButton(_map_lbl[sv], width=70, height=30, data=k,
                                            bgcolor=_map_clr[sv], color=ft.Colors.WHITE,
                                            style=ft.ButtonStyle(padding=0))
                    def _click(e, _k=k):
                        social_st[_k] = _next_val[social_st[_k]]
                        nv = social_st[_k]
                        e.control.text = _map_lbl[nv]
                        e.control.bgcolor = _map_clr[nv]
                        e.control.update()
                    btn.on_click = _click; cells.append(btn)
                else:
                    cells.append(ft.Container(width=70))
            if cells:
                social_ct.controls.append(ft.Row([lbl(p1)] + cells, spacing=2))
        page.update()

    def build_quota():
        P,T,H,D = dims(); quota_ct.controls.clear()
        quota_ct.controls.append(hdr_row(T,70))
        for p in P:
            cells = []
            for t in T:
                k = (p,t)
                if k not in quota_st: quota_st[k] = "0"
                tf = ft.TextField(value=quota_st[k], width=70, height=35,
                                  text_size=12, data=k, content_padding=ft.padding.all(4))
                def _ch(e, _k=k): quota_st[_k] = e.control.value
                tf.on_change = _ch; cells.append(tf)
            quota_ct.controls.append(ft.Row([lbl(p)] + cells, spacing=2))
        page.update()

    def build_rotation():
        P,T,H,D = dims(); rotation_ct.controls.clear()
        for t in T:
            if t not in rotation_st: rotation_st[t] = 1
            sw = ft.Switch(label=t,value=rotation_st[t]==1, data=t)
            def _ch(e, _t=t): rotation_st[_t] = 1 if e.control.value else 0
            sw.on_change = _ch; rotation_ct.controls.append(sw)
        page.update()

    builders = {1: build_avail, 2: build_demand, 3: build_skills,
                4: build_force, 5: build_social, 6: build_quota, 7: build_rotation}

    def on_tab_change(e):
        idx = e.control.selected_index
        if idx in builders: builders[idx]()

    # ---- Build visual schedule grid ----
    def build_output_grid(sol, P,T,H,D, availability):
        output_ct.controls.clear()
        output_ct.controls.append(ft.Text(f"Status: {sol['status']}", weight=ft.FontWeight.BOLD, size=16))
        output_ct.controls.append(ft.Divider())

        # Build task -> color map
        tc = {}
        for i, t in enumerate(T):
            bg, fg = TASK_COLORS[i % len(TASK_COLORS)]
            tc[t] = (bg, fg)

        CW = 75  # cell width
        Ch = 36  # cell height
        NW = 110 # name column width
        TW = 50  # total column width

        for j in D:
            output_ct.controls.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=16))

            # Header row
            hdr = [ft.Container(
                ft.Text("Person", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                width=NW, height=Ch,bgcolor="#455A64",
                alignment=ft.alignment.center,
                border=ft.border.all(1, "#37474F"))]
            for h in H:
                hdr.append(ft.Container(
                    ft.Text(h,size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    width=CW, height=Ch,bgcolor="#455A64",
                    alignment=ft.alignment.center,
                    border=ft.border.all(1, "#37474F")))
            hdr.append(ft.Container(
                ft.Text("Total", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                width=TW, height=Ch,bgcolor="#455A64",
                alignment=ft.alignment.center,
                border=ft.border.all(1, "#37474F")))
            output_ct.controls.append(ft.Row(hdr, spacing=0))

            # Data rows
            asgn = sol["assignment"][j]
            for idx_p,p in enumerate(P):
                row_bg = "#263238" if idx_p % 2 == 0 else "#37474F"
                cells = [ft.Container(
                    ft.Text(p,size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    width=NW, height=Ch,bgcolor=row_bg,
                    alignment=ft.alignment.center_left,
                    padding=ft.padding.only(left=8),
                    border=ft.border.all(1, "#455A64"))]

                total = 0
                for h in H:
                    task = asgn[p][h]
                    avail = availability.get((p,h,j), 1)
                    if task:
                        bg, fg = tc[task]
                        total += 1
                        cell = ft.Container(
                            ft.Text(task, size=11, weight=ft.FontWeight.BOLD, color=fg, text_align=ft.TextAlign.CENTER),
                            width=CW, height=Ch,bgcolor=bg,
                            alignment=ft.alignment.center,
                            border=ft.border.all(1, "#455A64"),
                            border_radius=4)
                    elif avail == 0:
                        cell = ft.Container(
                            width=CW, height=Ch,bgcolor=UNAVAIL_COLOR,
                            border=ft.border.all(1, "#455A64"),
                            border_radius=4)
                    else:
                        cell = ft.Container(
                            width=CW, height=Ch,bgcolor=row_bg,
                            border=ft.border.all(1, "#455A64"))
                    cells.append(cell)

                cells.append(ft.Container(
                    ft.Text(str(int(total)), size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE,
                            text_align=ft.TextAlign.CENTER),
                    width=TW, height=Ch,bgcolor=row_bg,
                    alignment=ft.alignment.center,
                    border=ft.border.all(1, "#455A64")))
                output_ct.controls.append(ft.Row(cells, spacing=0))

            # Legend
            legend_items = [ft.Container(
                ft.Text(t,size=10, weight=ft.FontWeight.BOLD, color=tc[t][1]),
                bgcolor=tc[t][0], padding=ft.padding.symmetric(6, 10),
                border_radius=4) for t in T]
            legend_items.append(ft.Container(
                ft.Text("Unavailable", size=10), bgcolor=UNAVAIL_COLOR,
                padding=ft.padding.symmetric(6, 10), border_radius=4))
            output_ct.controls.append(ft.Row(legend_items, spacing=8))
            output_ct.controls.append(ft.Divider())

        # --- Summary sections ---
        output_ct.controls.append(ft.Text("MISSING STAFF", weight=ft.FontWeight.BOLD, size=14))
        if sol["missing"]:
            for line in sol["missing"]:
                output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else:
            output_ct.controls.append(ft.Text("  None -- all demand covered!", size=12, italic=True))

        output_ct.controls.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=14))
        for p in P:
            output_ct.controls.append(ft.Text(f"  {p}: {sol['workload'][p]:.0f} hours", size=12))
        output_ct.controls.append(ft.Text(
            f"  Global range: max={sol['z_max']:.0f}, min={sol['z_min']:.0f}", size=12, italic=True))

        # UPDATED LABEL
        output_ct.controls.append(ft.Text("SHIFT SEGMENTS / FRAGMENTATION", weight=ft.FontWeight.BOLD, size=14))
        if sol["gaps"]:
            for line in sol["gaps"]:
                output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else:
            output_ct.controls.append(ft.Text("  Single block shifts! (Perfect)", size=12, italic=True))

        output_ct.controls.append(ft.Text("SOCIAL", weight=ft.FontWeight.BOLD, size=14))
        if sol["social_issues"]:
            for line in sol["social_issues"]:
                output_ct.controls.append(ft.Text(f"  {line}", size=12))
        else:
            output_ct.controls.append(ft.Text("  None -- all respected!", size=12, italic=True))

    # --- solve ---
    def do_solve(e):
        P,T,H,D = dims()

        availability = {}
        for p in P:
            for h in H:
                for j in D:
                    availability[(p,h,j)] = avail_st.get((p,h,j), 1)
        demand = {}
        for t in T:
            for h in H:
                for j in D:
                    raw = demand_st.get((t,h,j), "1")
                    try: demand[(t,h,j)] = int(raw)
                    except: demand[(t,h,j)] = 0
        skills = {}
        for p in P:
            for t in T:
                skills[(p,t)] = skills_st.get((p,t), 1)
        force = {}
        for p in P:
            for t in T:
                for h in H:
                    for j in D:
                        force[(p,t,h,j)] = force_st.get((p,t,h,j), 0)
        social = {}
        for i, p1 in enumerate(P):
            for p2 in P[i+1:]:
                social[(p1,p2)] = social_st.get((p1,p2), 0)
        mq = {}
        for p in P:
            for t in T:
                raw = quota_st.get((p,t), "0")
                try: mq[(p,t)] = int(raw)
                except: mq[(p,t)] = 0
        rotation = {t: rotation_st.get(t,1) for t in T}
        pref_cost = {(p,t): 1 for p in P for t in T}
        X_prev = {(p,t,h,j): 0 for p in P for t in T for h in H for j in D}
        weights = dict(W_COVERAGE=100000, W_MANDATE=50000, W_STABILITY=10000,
                       W_EQ_DAY=5000, W_EQ_TOTAL=1000, W_ROTATION=500,
                       W_SOCIAL=100, W_GAP=50, W_QUOTA=10, W_PREF=5)

        data = dict(people=P,tasks=T,hours=H,days=D,
                    availability=availability, demand=demand, skills=skills,
                    force=force, social=social, min_quota=mq,
                    pref_cost=pref_cost,rotation=rotation, X_prev=X_prev, weights=weights)

        output_ct.controls.clear()
        output_ct.controls.append(ft.ProgressRing())
        output_ct.controls.append(ft.Text("Solving...", italic=True))
        tabs.selected_index = 8
        page.update()

        try:
            # Assuming solve_model is defined globally or imported
            sol = solve_model(data) 
            build_output_grid(sol, P,T,H,D, availability)
        except Exception as ex:
            output_ct.controls.clear()
            output_ct.controls.append(ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=14))
            import traceback
            traceback.print_exc()
        page.update()

    solve_btn = ft.ElevatedButton("SOLVE", on_click=do_solve,
                                   bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE,
                                   height=45, width=200)

    dim_tab_content = ft.Container(
        content=ft.Row(
            controls=[tf_people, tf_tasks, tf_hours, tf_days],
            spacing=20,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
        padding=20, expand=True)

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
            ft.Tab(text="Output",       content=ft.Container(output_ct,padding=10, expand=True)),
        ])

    page.add(ft.Row([solve_btn], alignment=ft.MainAxisAlignment.CENTER), tabs)

ft.app(target=main)