"""
╔══════════════════════════════════════════════════════════════════╗
║       STAFFING OPTIMIZER  ·  Flet UI + PuLP / HiGHS   v4       ║
║                                                                  ║
║  Reactivo: las matrices se actualizan automáticamente           ║
║  al cambiar personas, tareas, días u horarios.                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import flet as ft
import pulp


def main(page: ft.Page):

    page.title = "Staffing Optimizer"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.scroll = ft.ScrollMode.ADAPTIVE

    # ═════════════════════════════════════════════
    #  ESTADO GLOBAL
    # ═════════════════════════════════════════════

    people: list[str] = []
    tasks:  list[str] = []
    day_configs: list[dict] = []
    day_labels:  list[list[str]] = []

    avail_data:    dict[tuple, int]   = {}
    demand_data:   dict[tuple, int]   = {}
    skill_data:    dict[tuple, int]   = {}
    force_data:    dict[tuple, int]   = {}
    rotation_data: dict[str, int]     = {}
    social_data:   dict[tuple, int]   = {}
    pref_data:     dict[tuple, float] = {}
    quota_data:    dict[tuple, int]   = {}
    weight_data:   dict[str, float]   = {
        "W_COVERAGE": 1000, "W_STABILITY": 50,  "W_MANDATE": 200,
        "W_EQ_DAY":   10,   "W_EQ_TOTAL":  20,  "W_ROTATION": 5,
        "W_SOCIAL":   8,    "W_GAP":       15,  "W_QUOTA":    12,
        "W_PREF":     2,
    }

    # Snapshot para saber si realmente cambió algo
    _prev_snapshot = {"people": [], "tasks": [], "days": []}

    # ═════════════════════════════════════════════
    #  UTILIDADES
    # ═════════════════════════════════════════════

    def time_to_min(t: str) -> int:
        parts = t.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def min_to_time(m: int) -> str:
        return f"{m // 60:02d}:{m % 60:02d}"

    def compute_labels(start: str, end: str, div: int) -> list[str]:
        try:
            s, e = time_to_min(start), time_to_min(end)
        except (ValueError, IndexError):
            return []
        if e <= s or div <= 0:
            return []
        return [min_to_time(s + i * div) for i in range((e - s) // div)]

    CELL    = 56
    LABEL_W = 100

    def heading(text):
        return ft.Text(text, size=16, weight=ft.FontWeight.BOLD)

    def caption(text):
        return ft.Text(text, size=12, italic=True, color=ft.Colors.GREY_700)

    def label_cell(text, width=LABEL_W):
        return ft.Container(width=width,
                            content=ft.Text(text, size=12, no_wrap=True))

    def header_cell(text, width=CELL):
        return ft.Container(width=width, alignment=ft.alignment.center,
                            content=ft.Text(text, size=10,
                                            weight=ft.FontWeight.BOLD))

    def toggle_cell(store, key, *,
                    on_color=ft.Colors.GREEN_400,
                    off_color=ft.Colors.GREY_300, default=1):
        store.setdefault(key, default)

        def flip(e):
            store[key] = 1 - store[key]
            e.control.bgcolor = on_color if store[key] else off_color
            e.control.update()

        return ft.Container(width=CELL, height=34, border_radius=6,
                            bgcolor=on_color if store[key] else off_color,
                            on_click=flip, alignment=ft.alignment.center)

    def int_cell(store, key, default=1, width=CELL):
        store.setdefault(key, default)

        def changed(e):
            try:    store[key] = max(0, int(e.control.value))
            except: store[key] = 0

        return ft.TextField(value=str(store[key]), width=width, height=36,
                            text_align=ft.TextAlign.CENTER,
                            content_padding=ft.padding.all(4),
                            on_change=changed)

    def float_cell(store, key, default=0.0, width=CELL):
        store.setdefault(key, default)

        def changed(e):
            try:    store[key] = max(0.0, float(e.control.value))
            except: store[key] = 0.0

        return ft.TextField(value=str(store[key]), width=width, height=36,
                            text_align=ft.TextAlign.CENTER,
                            content_padding=ft.padding.all(4),
                            on_change=changed)

    def shift_header_row(d):
        return ft.Row(
            [ft.Container(width=LABEL_W)]
            + [header_cell(lbl) for lbl in day_labels[d]], spacing=2)

    # ═════════════════════════════════════════════
    #  CONSTRUCTORES DE MATRICES
    # ═════════════════════════════════════════════

    def build_skills_tab():
        hdr = ft.Row([ft.Container(width=LABEL_W)]
                     + [header_cell(t, 70) for t in tasks], spacing=2)
        rows = [heading("Matriz de Habilidades"),
                caption("Verde = puede hacer la tarea."),
                ft.Container(height=6), hdr]
        for p in people:
            rows.append(ft.Row(
                [label_cell(p)]
                + [ft.Container(width=70, alignment=ft.alignment.center,
                                content=toggle_cell(skill_data, (p, t)))
                   for t in tasks], spacing=2))

        rows += [ft.Divider(height=20),
                 heading("Rotación por Tarea"),
                 caption("Verde = requiere rotación.")]
        rot = ft.Row(spacing=12)
        for t in tasks:
            rotation_data.setdefault(t, 0)
            rot.controls.append(ft.Row([
                ft.Text(t, size=12),
                toggle_cell(rotation_data, t,
                            on_color=ft.Colors.TEAL_400, default=0),
            ], spacing=4))
        rows.append(rot)
        return ft.Column(rows, spacing=4)

    def build_prefs_tab():
        rows = [heading("Costo de Preferencia (B)"),
                caption("Mayor = más disgusto.")]
        hdr = ft.Row([ft.Container(width=LABEL_W)]
                     + [header_cell(t, 60) for t in tasks], spacing=2)
        rows.append(hdr)
        for p in people:
            rows.append(ft.Row(
                [label_cell(p)]
                + [float_cell(pref_data, (p, t), 0.0, 60) for t in tasks],
                spacing=2))
        rows += [ft.Divider(height=20),
                 heading("Cuota Mínima (L)"),
                 caption("Verde = desea al menos 1 turno.")]
        hdr2 = ft.Row([ft.Container(width=LABEL_W)]
                      + [header_cell(t, 70) for t in tasks], spacing=2)
        rows.append(hdr2)
        for p in people:
            rows.append(ft.Row(
                [label_cell(p)]
                + [ft.Container(width=70, alignment=ft.alignment.center,
                                content=toggle_cell(quota_data, (p, t),
                                                    on_color=ft.Colors.AMBER_400,
                                                    default=0))
                   for t in tasks], spacing=2))
        return ft.Column(rows, spacing=4)

    def build_social_tab():
        rows = [heading("Afinidad Social"),
                caption("Click: Gris=Neutral · Verde=Juntos · Rojo=Separados")]
        colors = {0: ft.Colors.GREY_300, 1: ft.Colors.GREEN_400, -1: ft.Colors.RED_400}
        sym    = {0: "—", 1: "✓", -1: "✗"}
        hdr = ft.Row([ft.Container(width=LABEL_W)]
                     + [header_cell(pp, 60) for pp in people], spacing=2)
        rows.append(hdr)
        for p in people:
            cells = [label_cell(p)]
            for pp in people:
                if p == pp:
                    cells.append(ft.Container(width=60, height=34,
                                              bgcolor=ft.Colors.GREY_200,
                                              border_radius=6))
                    continue
                social_data.setdefault((p, pp), 0)
                txt = ft.Text(sym[social_data[(p, pp)]], size=11,
                              weight=ft.FontWeight.BOLD)
                c = ft.Container(
                    width=60, height=34, border_radius=6,
                    bgcolor=colors[social_data[(p, pp)]],
                    alignment=ft.alignment.center, content=txt)

                def cycle(e, _p=p, _pp=pp, _c=c, _t=txt):
                    v = social_data[(_p, _pp)]
                    nv = {0: 1, 1: -1, -1: 0}[v]
                    social_data[(_p, _pp)] = nv
                    social_data[(_pp, _p)] = nv
                    _c.bgcolor = colors[nv]
                    _t.value = sym[nv]
                    _c.update()

                c.on_click = cycle
                cells.append(c)
            rows.append(ft.Row(cells, spacing=2))
        return ft.Column(rows, spacing=4)

    def build_availability(d):
        ns = len(day_labels[d])
        rows = [shift_header_row(d)]
        for p in people:
            rows.append(ft.Row(
                [label_cell(p)]
                + [toggle_cell(avail_data, (d, p, h)) for h in range(ns)],
                spacing=2))
        return ft.Column(rows, spacing=4)

    def build_demand(d):
        ns = len(day_labels[d])
        rows = [shift_header_row(d)]
        for t in tasks:
            rows.append(ft.Row(
                [label_cell(t)]
                + [int_cell(demand_data, (d, t, h)) for h in range(ns)],
                spacing=2))
        return ft.Column(rows, spacing=4)

    def build_force(d):
        ns = len(day_labels[d])
        grid = ft.Column()

        def on_task(e):
            st = e.control.value
            grid.controls.clear()
            if not st:
                page.update()
                return
            grid.controls.append(shift_header_row(d))
            for p in people:
                grid.controls.append(ft.Row(
                    [label_cell(p)]
                    + [toggle_cell(force_data, (d, p, st, h),
                                   on_color=ft.Colors.ORANGE_400, default=0)
                       for h in range(ns)],
                    spacing=2))
            page.update()

        dd = ft.Dropdown(label="Seleccionar tarea", width=220,
                         options=[ft.dropdown.Option(t) for t in tasks],
                         on_change=on_task)
        return ft.Column([dd, grid], spacing=10)

    def build_weights_tab():
        nice = {
            "W_COVERAGE": "Cobertura",    "W_STABILITY": "Estabilidad",
            "W_MANDATE":  "Mandato",      "W_EQ_DAY":    "Equidad Diaria",
            "W_EQ_TOTAL": "Equidad Global","W_ROTATION":  "Rotación",
            "W_SOCIAL":   "Social",       "W_GAP":       "Huecos",
            "W_QUOTA":    "Cuota",        "W_PREF":      "Preferencia",
        }
        rows = [heading("Pesos de Prioridad"),
                caption("Mayor peso = más importancia.")]
        for k, lbl in nice.items():
            def cb(e, key=k):
                try:    weight_data[key] = float(e.control.value)
                except: pass
            rows.append(ft.Row([
                ft.Text(lbl, size=13, width=140),
                ft.TextField(value=str(weight_data[k]), width=100, height=36,
                             text_align=ft.TextAlign.CENTER,
                             content_padding=ft.padding.all(4),
                             on_change=cb),
            ], spacing=8))
        return ft.Column(rows, spacing=6)

    # ═════════════════════════════════════════════
    #  ENTRADA: PERSONAS Y TAREAS
    # ═════════════════════════════════════════════

    people_field = ft.TextField(
        label="Personas (una por línea)",
        multiline=True, min_lines=4, max_lines=12, width=300,
        value="Ana\nBob\nCho\nDan\nEva")

    tasks_field = ft.TextField(
        label="Tareas (una por línea)",
        multiline=True, min_lines=4, max_lines=12, width=300,
        value="Sound\nLights\nDoor")

    # ═════════════════════════════════════════════
    #  ENTRADA: DÍAS
    # ═════════════════════════════════════════════

    num_days_field = ft.TextField(
        label="Cantidad de días", width=120, value="2",
        keyboard_type=ft.KeyboardType.NUMBER)

    day_config_area = ft.Column(spacing=10)
    day_rows_controls: list[dict] = []

    # ═════════════════════════════════════════════
    #  ÁREA DE MATRICES Y RESULTADOS
    # ═════════════════════════════════════════════

    matrix_area  = ft.Column()
    results_area = ft.Column()

    # ═════════════════════════════════════════════
    #  REBUILD CENTRAL  (se llama en cada cambio)
    # ═════════════════════════════════════════════

    def _parse_inputs():
        """Lee personas, tareas y configs de día desde los campos."""
        people.clear()
        for line in people_field.value.strip().splitlines():
            n = line.strip()
            if n and n not in people:
                people.append(n)

        tasks.clear()
        for line in tasks_field.value.strip().splitlines():
            n = line.strip()
            if n and n not in tasks:
                tasks.append(n)

        day_configs.clear()
        day_labels.clear()
        for row in day_rows_controls:
            s = row["start"].value.strip()
            e = row["end"].value.strip()
            try:    d = int(row["div"].value)
            except: d = 60
            day_configs.append({"start": s, "end": e, "div": d})
            day_labels.append(compute_labels(s, e, d))

    def _snapshot():
        return {
            "people": list(people),
            "tasks":  list(tasks),
            "days":   [(dc["start"], dc["end"], dc["div"])
                       for dc in day_configs],
        }

    def rebuild_all(_=None):
        """Reconstruye las pestañas de matrices si los inputs cambiaron."""
        _parse_inputs()

        snap = _snapshot()
        if snap == _prev_snapshot.get("__last"):
            return
        _prev_snapshot["__last"] = snap

        # No construir si faltan datos fundamentales
        if not people or not tasks or not day_labels:
            matrix_area.controls.clear()
            page.update()
            return

        # Si algún día tiene 0 turnos, no construir
        if any(len(lbl) == 0 for lbl in day_labels):
            matrix_area.controls.clear()
            page.update()
            return

        results_area.controls.clear()

        nd = len(day_configs)

        tabs = [
            ft.Tab(text="Skills", icon=ft.Icons.BUILD_OUTLINED,
                   content=ft.Container(padding=20,
                                        content=build_skills_tab())),
            ft.Tab(text="Preferencias", icon=ft.Icons.STAR_OUTLINE,
                   content=ft.Container(padding=20,
                                        content=build_prefs_tab())),
            ft.Tab(text="Social", icon=ft.Icons.PEOPLE_OUTLINE,
                   content=ft.Container(padding=20,
                                        content=build_social_tab())),
        ]

        for d in range(nd):
            dc = day_configs[d]
            lbl = f"Día {d+1} ({dc['start']}–{dc['end']}, {dc['div']}′)"
            tabs.append(ft.Tab(
                text=lbl, icon=ft.Icons.CALENDAR_TODAY,
                content=ft.Container(padding=20, content=ft.Column([
                    heading(f"Disponibilidad — Día {d+1}"),
                    caption("Verde = presente."),
                    build_availability(d), ft.Divider(height=20),
                    heading(f"Demanda — Día {d+1}"),
                    caption("Personas necesarias por tarea y turno."),
                    build_demand(d), ft.Divider(height=20),
                    heading(f"Forzar Asignación — Día {d+1}"),
                    caption("Naranja = forzar."),
                    build_force(d),
                ], spacing=10, scroll=ft.ScrollMode.ADAPTIVE))))

        tabs.append(ft.Tab(text="Pesos", icon=ft.Icons.TUNE,
                           content=ft.Container(padding=20,
                                                content=build_weights_tab())))

        matrix_area.controls.clear()
        matrix_area.controls += [
            ft.Container(
                content=ft.Tabs(tabs=tabs, animation_duration=200,
                                scrollable=True),
                height=620,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10),
            ft.Container(height=12),
            optimize_btn,
            results_area,
        ]
        page.update()

    def rebuild_day_rows(_=None):
        """Reconstruye las filas de config por día y luego rebuild_all."""
        try:    nd = max(1, int(num_days_field.value))
        except: nd = 1

        day_config_area.controls.clear()
        day_rows_controls.clear()

        for d in range(nd):
            start_f = ft.TextField(label="Inicio", width=100, value="09:00",
                                   text_align=ft.TextAlign.CENTER,
                                   on_change=rebuild_all)
            end_f   = ft.TextField(label="Fin", width=100, value="18:00",
                                   text_align=ft.TextAlign.CENTER,
                                   on_change=rebuild_all)
            div_dd  = ft.Dropdown(
                label="División", width=130, value="60",
                options=[ft.dropdown.Option(str(m), f"{m} min")
                         for m in [15, 30, 60, 120]],
                on_change=rebuild_all)

            day_rows_controls.append(
                {"start": start_f, "end": end_f, "div": div_dd})

            day_config_area.controls.append(
                ft.Row([
                    ft.Text(f"Día {d+1}", weight=ft.FontWeight.BOLD, width=60),
                    start_f, end_f, div_dd,
                ], spacing=10))

        rebuild_all()

    # Conectar on_change
    people_field.on_change   = rebuild_all
    tasks_field.on_change    = rebuild_all
    num_days_field.on_change = rebuild_day_rows

    # ═════════════════════════════════════════════
    #  OPTIMIZAR  (PuLP + HiGHS)
    # ═════════════════════════════════════════════

    def run_optimization(_):
        _parse_inputs()

        if not people or not tasks or not day_labels:
            return
        if any(len(l) == 0 for l in day_labels):
            return

        results_area.controls.clear()
        results_area.controls.append(
            ft.ProgressRing(width=28, height=28, stroke_width=3))
        page.update()

        nd = len(day_configs)
        ds = list(range(nd))
        W  = weight_data
        hs_d = {d: list(range(len(day_labels[d]))) for d in ds}

        # ── Leer UI ──────────────────────────────
        A = {p: {h: {d: avail_data.get((d, p, h), 1)
                      for d in ds}
                 for h in range(max(len(day_labels[d]) for d in ds))}
             for p in people}

        D_dem = {t: {h: {d: demand_data.get((d, t, h), 1)
                          for d in ds}
                     for h in range(max(len(day_labels[d]) for d in ds))}
                 for t in tasks}

        S_sk  = {p: {t: skill_data.get((p, t), 1) for t in tasks}
                 for p in people}

        F_f = {p: {t: {h: {d: force_data.get((d, p, t, h), 0)
                            for d in ds}
                       for h in range(max(len(day_labels[d]) for d in ds))}
                   for t in tasks}
               for p in people}

        R_rot = {t: rotation_data.get(t, 0) for t in tasks}
        B_pr  = {p: {t: pref_data.get((p, t), 0.0) for t in tasks}
                 for p in people}
        L_q   = {p: {t: quota_data.get((p, t), 0) for t in tasks}
                 for p in people}
        E_soc = {p: {pp: social_data.get((p, pp), 0) for pp in people}
                 for p in people}
        X_prev = {p: {t: {h: {d: 0 for d in ds}
                          for h in range(max(len(day_labels[d]) for d in ds))}
                      for t in tasks}
                  for p in people}

        # Combos
        all_c = [(p,t,h,d) for d in ds for h in hs_d[d]
                 for p in people for t in tasks]
        rot_c = [(p,t,h,d) for d in ds for h in hs_d[d][:-1]
                 for p in people for t in tasks]
        env_i = [(p,h,d) for d in ds for h in hs_d[d] for p in people]

        tog = [(p,pp) for p in people for pp in people
               if p < pp and E_soc[p][pp] == 1]
        sep = [(p,pp) for p in people for pp in people
               if p < pp and E_soc[p][pp] == -1]

        # ── Modelo ───────────────────────────────
        mdl = pulp.LpProblem("Staffing", pulp.LpMinimize)

        x  = pulp.LpVariable.dicts("x",  (c for c in all_c), cat="Binary")
        mi = pulp.LpVariable.dicts("m",  ((t,h,d) for d in ds for h in hs_d[d] for t in tasks), lowBound=0)
        U  = pulp.LpVariable.dicts("U",  (c for c in all_c), lowBound=0)

        n_mx = pulp.LpVariable.dicts("nMax", ds, lowBound=0)
        n_mn = pulp.LpVariable.dicts("nMin", ds, lowBound=0)
        N_mx = pulp.LpVariable("Nmax", lowBound=0)
        N_mn = pulp.LpVariable("Nmin", lowBound=0)

        C  = pulp.LpVariable.dicts("C",  (c for c in rot_c), cat="Binary")
        Ts = pulp.LpVariable.dicts("Ts", ((pr,t,h,d) for pr in tog for d in ds for h in hs_d[d] for t in tasks), lowBound=0)
        Ss = pulp.LpVariable.dicts("Ss", ((pr,t,h,d) for pr in sep for d in ds for h in hs_d[d] for t in tasks), lowBound=0)
        Q  = pulp.LpVariable.dicts("Q",  (people, tasks), cat="Binary")
        Es = pulp.LpVariable.dicts("Es", (c for c in all_c), cat="Binary")

        Str = pulp.LpVariable.dicts("Str", (i for i in env_i), cat="Binary")
        AL  = pulp.LpVariable.dicts("AL",  (i for i in env_i), cat="Binary")
        Zg  = pulp.LpVariable.dicts("Zg",  (i for i in env_i), cat="Binary")

        # ── Restricciones ────────────────────────

        for d in ds:
            for h in hs_d[d]:
                for t in tasks:
                    mdl += (pulp.lpSum(x[p,t,h,d] for p in people)
                            + mi[t,h,d] == D_dem[t][h][d])

        for p,t,h,d in all_c:
            mdl += F_f[p][t][h][d] - x[p,t,h,d] <= U[p,t,h,d]

        for d in ds:
            for h in hs_d[d]:
                for p in people:
                    mdl += pulp.lpSum(x[p,t,h,d] for t in tasks) <= A[p][h][d]

        for p in people:
            for t in tasks:
                if S_sk[p][t] == 0:
                    for d in ds:
                        for h in hs_d[d]:
                            mdl += x[p,t,h,d] == 0

        for p in people:
            for d in ds:
                s = pulp.lpSum(x[p,t,h,d] for t in tasks for h in hs_d[d])
                mdl += s <= n_mx[d]
                mdl += s >= n_mn[d]

        for p in people:
            s = pulp.lpSum(x[p,t,h,d] for t in tasks for d in ds for h in hs_d[d])
            mdl += s <= N_mx
            mdl += s >= N_mn

        for p,t,h,d in rot_c:
            mdl += x[p,t,h,d] + x[p,t,h+1,d] - C[p,t,h,d] <= 2 - R_rot[t]

        for pr in tog:
            pa, pb = pr
            for d in ds:
                for h in hs_d[d]:
                    for t in tasks:
                        mdl += x[pa,t,h,d] - x[pb,t,h,d] <= Ts[pr,t,h,d]
                        mdl += x[pb,t,h,d] - x[pa,t,h,d] <= Ts[pr,t,h,d]

        for pr in sep:
            pa, pb = pr
            for d in ds:
                for h in hs_d[d]:
                    for t in tasks:
                        mdl += x[pa,t,h,d] + x[pb,t,h,d] - Ss[pr,t,h,d] <= 1

        for p in people:
            for t in tasks:
                mdl += (pulp.lpSum(x[p,t,h,d] for d in ds for h in hs_d[d])
                        + Q[p][t] >= L_q[p][t])

        for p,t,h,d in all_c:
            mdl += Es[p,t,h,d] >= X_prev[p][t][h][d] - x[p,t,h,d]
            mdl += Es[p,t,h,d] >= x[p,t,h,d] - X_prev[p][t][h][d]

        for d in ds:
            for p in people:
                for h in hs_d[d]:
                    mdl += Str[p,h,d] >= pulp.lpSum(x[p,t,h,d] for t in tasks)
                    if h > hs_d[d][0]:
                        mdl += Str[p,h,d] >= Str[p,h-1,d]

        for d in ds:
            for p in people:
                for h in hs_d[d]:
                    mdl += AL[p,h,d] >= pulp.lpSum(x[p,t,h,d] for t in tasks)
                    if h < hs_d[d][-1]:
                        mdl += AL[p,h,d] >= AL[p,h+1,d]

        for p,h,d in env_i:
            mdl += (Zg[p,h,d] >= Str[p,h,d] + AL[p,h,d] - 1
                    - pulp.lpSum(x[p,t,h,d] for t in tasks))

        # ── Objetivo ─────────────────────────────
        mdl += (
            W["W_COVERAGE"]  * pulp.lpSum(mi[t,h,d] for d in ds for h in hs_d[d] for t in tasks)
          + W["W_STABILITY"] * pulp.lpSum(Es[c] for c in all_c)
          + W["W_MANDATE"]   * pulp.lpSum(U[c] for c in all_c)
          + W["W_EQ_DAY"]    * pulp.lpSum(n_mx[d] - n_mn[d] for d in ds)
          + W["W_EQ_TOTAL"]  * (N_mx - N_mn)
          + W["W_ROTATION"]  * pulp.lpSum(C[c] for c in rot_c)
          + W["W_SOCIAL"]    * (pulp.lpSum(Ts[pr,t,h,d] for pr in tog for d in ds for h in hs_d[d] for t in tasks)
                               + pulp.lpSum(Ss[pr,t,h,d] for pr in sep for d in ds for h in hs_d[d] for t in tasks))
          + W["W_GAP"]       * pulp.lpSum(Zg[i] for i in env_i)
          + W["W_QUOTA"]     * pulp.lpSum(Q[p][t] for p in people for t in tasks)
          + W["W_PREF"]      * pulp.lpSum(B_pr[p][t] * x[p,t,h,d] for p,t,h,d in all_c)
        )

        # ── Resolver ─────────────────────────────
        solver = pulp.HiGHS(msg=0)
        mdl.solve(solver)

        status = pulp.LpStatus[mdl.status]
        results_area.controls.clear()

        if status != "Optimal":
            results_area.controls.append(
                ft.Text(f"Sin solución óptima. Estado: {status}",
                        color=ft.Colors.RED_700, size=16,
                        weight=ft.FontWeight.BOLD))
            page.update()
            return

        # ── Resultados ───────────────────────────
        results_area.controls += [
            ft.Divider(height=20),
            ft.Text(f"✓  Óptimo — Costo: {pulp.value(mdl.objective):.2f}",
                    size=18, weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN_800)]

        for d in ds:
            labels = day_labels[d]
            dc = day_configs[d]

            cols = [ft.DataColumn(ft.Text("Turno", weight=ft.FontWeight.BOLD))]
            for p in people:
                cols.append(ft.DataColumn(ft.Text(p, weight=ft.FontWeight.BOLD)))

            rows_dt = []
            for h in hs_d[d]:
                s_str = labels[h]
                e_str = min_to_time(time_to_min(labels[h]) + dc["div"])
                cells = [ft.DataCell(ft.Text(f"{s_str}–{e_str}", size=11))]
                for p in people:
                    val = [t for t in tasks
                           if pulp.value(x[p,t,h,d]) is not None
                           and pulp.value(x[p,t,h,d]) > 0.5]
                    txt  = val[0] if val else "—"
                    bg   = ft.Colors.GREEN_100 if val else None
                    cells.append(ft.DataCell(
                        ft.Container(content=ft.Text(txt, size=12),
                                     bgcolor=bg, padding=4,
                                     border_radius=4)))
                rows_dt.append(ft.DataRow(cells=cells))

            results_area.controls += [
                ft.Container(height=16),
                ft.Text(f"Día {d+1}  ({dc['start']}–{dc['end']}, "
                        f"turnos de {dc['div']}′)",
                        size=16, weight=ft.FontWeight.BOLD),
                ft.DataTable(columns=cols, rows=rows_dt,
                             border=ft.border.all(1, ft.Colors.GREY_300),
                             border_radius=8,
                             horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_200),
                             column_spacing=16)]

        # Faltantes
        miss = []
        for d in ds:
            for h in hs_d[d]:
                for t in tasks:
                    v = pulp.value(mi[t,h,d])
                    if v and v > 0.01:
                        miss.append(f"   {t} | {day_labels[d][h]} | Día {d+1} → faltan {v:.0f}")
        if miss:
            results_area.controls.append(ft.Container(height=12))
            results_area.controls.append(
                ft.Text("⚠  Personal faltante:",
                        color=ft.Colors.ORANGE_800, weight=ft.FontWeight.BOLD))
            for m in miss:
                results_area.controls.append(ft.Text(m, size=13))

        # Resumen
        results_area.controls += [
            ft.Container(height=12),
            ft.Text("Resumen de turnos por persona",
                    size=16, weight=ft.FontWeight.BOLD)]

        s_cols = ([ft.DataColumn(ft.Text("Persona", weight=ft.FontWeight.BOLD))]
                 + [ft.DataColumn(ft.Text(f"Día {d+1}", weight=ft.FontWeight.BOLD)) for d in ds]
                 + [ft.DataColumn(ft.Text("Total", weight=ft.FontWeight.BOLD))])
        s_rows = []
        for p in people:
            cells = [ft.DataCell(ft.Text(p))]
            total = 0
            for d in ds:
                cnt = sum(1 for t in tasks for h in hs_d[d]
                          if pulp.value(x[p,t,h,d]) is not None
                          and pulp.value(x[p,t,h,d]) > 0.5)
                total += cnt
                cells.append(ft.DataCell(ft.Text(str(cnt))))
            cells.append(ft.DataCell(ft.Text(str(total), weight=ft.FontWeight.BOLD)))
            s_rows.append(ft.DataRow(cells=cells))

        results_area.controls.append(
            ft.DataTable(columns=s_cols, rows=s_rows,
                         border=ft.border.all(1, ft.Colors.GREY_300),
                         border_radius=8))
        page.update()

    optimize_btn = ft.ElevatedButton(
        "Optimizar", icon=ft.Icons.ROCKET_LAUNCH,
        on_click=run_optimization,
        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700,
                             color=ft.Colors.WHITE,
                             padding=ft.padding.symmetric(28, 16)))

    # ═════════════════════════════════════════════
    #  LAYOUT PRINCIPAL
    # ═════════════════════════════════════════════

    page.add(
        ft.Text("Staffing Optimizer", size=28, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        ft.Row([
            ft.Column([heading("Personas"), people_field], spacing=6),
            ft.Column([heading("Tareas"),   tasks_field],  spacing=6),
        ], spacing=40),
        ft.Divider(height=16),
        heading("Configuración de Días"),
        num_days_field,
        ft.Container(height=6),
        day_config_area,
        ft.Container(height=12),
        matrix_area,
    )

    # Construir todo al inicio
    rebuild_day_rows()


ft.app(target=main)