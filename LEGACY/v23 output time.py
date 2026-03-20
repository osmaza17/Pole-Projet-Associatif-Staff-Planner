import flet as ft
import random
import threading

from solve_model_pace_2 import solve_model

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_WEIGHTS = {
    "W_COVERAGE": 30000,
    "W_FORCE": 10000,
    "W_CAPTAIN": 8000,
    "W_STABILITY": 7000,
    "W_EQ_DAY": 5000,
    "W_GAP": 1000,
    "W_EMERG": 750,
    "W_EQ_GLOBAL": 500,
    "W_ROTATION": 100,
    "W_SOCIAL": 50,
    "W_QUOTA": 10,
    "W_PREF": 1
}

SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

CAPTAIN_BG = "#E65100"
CAPTAIN_FG = "#FFFFFF"

DEFAULT_HOURS_TEXT = "08:00\n09:00\n10:00\n11:00\n12:00\n13:00\n14:00\n15:00\n16:00\n17:00"

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
# FLET UI
# ============================================================

def main(page: ft.Page):
    page.title = "Staff Scheduler"
    page.scroll = None
    page.window.width = 1200
    page.window.height = 800
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0

    # ── UI state stores ──────────────────────────────────────
    avail_st, demand_st, skills_st = {}, {}, {}
    force_st, social_st, quota_st, rotation_st = {}, {}, {}, {}
    captains_st = {}
    hard_enemies_st = [False]
    hours_per_day_st = {}

    # Pagination / Dropdown States
    avail_filter_st  = [None]
    demand_filter_st = [None]
    force_filter_st  = [None, None]   # [day, task]

    running_model_ref = [None]
    ui_lock = threading.Lock()
    validation_errors = {"demand": set(), "quota": set()}
    solve_blocked = [False]

    # ── Solution history ─────────────────────────────────────
    # Each entry: {sol, people, tasks, hours, days, availability, emergency}
    solution_history = []

    # ── Build cache ──────────────────────────────────────────
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

    # ── Solve-blocked helper ─────────────────────────────────
    def _update_solve_blocked():
        has_errors = bool(validation_errors["demand"] or validation_errors["quota"])
        solve_blocked[0] = has_errors
        solve_btn_ref[0].bgcolor = ft.Colors.GREY_500 if has_errors else "#1565C0"
        solve_btn_ref[0].disabled = has_errors
        try:
            solve_btn_ref[0].update()
        except Exception:
            pass

    solve_btn_ref = [None]

    def _validate_nonneg_int(value):
        v = value.strip()
        if v == "":
            return True
        try:
            n = int(v)
            return n >= 0 and str(n) == v
        except (ValueError, TypeError):
            return False

    def make_reset_btn(text, on_click_func):
        return ft.Container(
            content=ft.Text(text, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=12),
            bgcolor=ft.Colors.RED_500, padding=10, border_radius=4,
            on_click=on_click_func, width=150, alignment=ft.alignment.center)

    DEFAULT_SOLVER_PARAMS = {
        "TimeLimit": 1200, "MIPGap": 0.001, "MIPFocus": 2,
        "Threads": 0, "Presolve": 2, "Symmetry": 2,
        "Disconnected": 2, "IntegralityFocus": 1, "Method": 3
    }
    solver_params_st = DEFAULT_SOLVER_PARAMS.copy()
    weights_st       = DEFAULT_WEIGHTS.copy()
    weights_order    = list(DEFAULT_WEIGHTS.keys())
    weights_enabled  = {k: True for k in DEFAULT_WEIGHTS.keys()}

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

    def _on_dims_change(e):
        _invalidate_cache()

    tf_people.on_change = lambda e: (_on_dims_change(e), build_captains_list(e))
    tf_tasks.on_change  = lambda e: (_on_dims_change(e), build_rotation_list(e))
    tf_days.on_change   = lambda e: (_on_dims_change(e), build_hours_per_day(e))

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
                content=ft.Text("Cap" if val else "—", color=ft.Colors.BLACK, size=12,
                                weight=ft.FontWeight.BOLD),
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

    # ── Rotation column ──────────────────────────────────────
    rotation_col = ft.ListView(expand=True, spacing=4)

    def build_rotation_list(e=None):
        tasks = list(dict.fromkeys(x.strip() for x in tf_tasks.value.split("\n") if x.strip()))
        buf = [ft.Text("Rotation", weight=ft.FontWeight.BOLD, size=12)]
        for t in tasks:
            if t not in rotation_st:
                rotation_st[t] = 1
            val = rotation_st[t]
            btn = ft.Container(
                content=ft.Text("Rot" if val else "—", color=ft.Colors.BLACK, size=12,
                                weight=ft.FontWeight.BOLD),
                width=55, height=28, data=t,
                bgcolor=ft.Colors.GREEN_400 if val else ft.Colors.GREY_400,
                alignment=ft.alignment.center, border_radius=4)
            def _click(e, _t=t):
                rotation_st[_t] = 1 - rotation_st[_t]
                e.control.content.value = "Rot" if rotation_st[_t] else "—"
                e.control.bgcolor = ft.Colors.GREEN_400 if rotation_st[_t] else ft.Colors.GREY_400
                e.control.update()
            btn.on_click = _click
            buf.append(ft.Row([ft.Text(t, size=11, width=90), btn], spacing=4))
        rotation_col.controls = buf
        page.update()

    # ── Hours-per-day column ─────────────────────────────────
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

    # ── Enforced Rest & Solver Parameters Column ─────────────
    err_max_consec = ft.Text("", color=ft.Colors.RED_400, size=11, visible=False)
    tf_max_consec  = ft.TextField(value="4", width=100, height=35, text_size=12,
                                   label="Max hours (Y)", visible=False,
                                   content_padding=ft.padding.all(4))

    def _toggle_enforced_rest(e):
        tf_max_consec.visible = e.control.value
        err_max_consec.visible = False
        page.update()

    sw_enforced_rest = ft.Switch(label="Enforced Rest", value=False, on_change=_toggle_enforced_rest)
    enforced_rest_col = ft.Column([sw_enforced_rest, tf_max_consec, err_max_consec], spacing=5)

    param_tfs = {}
    for key, val in solver_params_st.items():
        tf = ft.TextField(label=key, value=str(val), width=150, height=45, text_size=12)
        def _ch_param(e, _k=key):
            try:
                new_val = float(e.control.value) if "." in e.control.value else int(e.control.value)
                solver_params_st[_k] = new_val
            except ValueError:
                pass
        tf.on_change = _ch_param
        param_tfs[key] = tf

    def _reset_params(e):
        solver_params_st.clear()
        solver_params_st.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS[k])
            tf.update()

    params_section = ft.Column([
        ft.Text("Gurobi Parameters", weight=ft.FontWeight.BOLD, size=14),
        make_reset_btn("Reset Params", _reset_params),
        ft.Container(
            content=ft.Column(list(param_tfs.values()), spacing=8, scroll=ft.ScrollMode.ADAPTIVE),
            height=350)
    ], spacing=10)

    right_controls_col = ft.Column([enforced_rest_col, ft.Divider(), params_section], width=180)

    # ── dims() helper ────────────────────────────────────────
    def dims():
        people = list(dict.fromkeys(x.strip() for x in tf_people.value.split("\n") if x.strip()))
        tasks  = list(dict.fromkeys(x.strip() for x in tf_tasks.value.split("\n") if x.strip()))
        days   = list(dict.fromkeys(x.strip() for x in tf_days.value.split("\n") if x.strip()))
        default = list(dict.fromkeys(x.strip() for x in DEFAULT_HOURS_TEXT.split("\n") if x.strip()))
        hours = {}
        for j in days:
            raw    = hours_per_day_st.get(j, DEFAULT_HOURS_TEXT)
            parsed = list(dict.fromkeys(x.strip() for x in raw.split("\n") if x.strip()))
            hours[j] = parsed if parsed else default
        return people, tasks, hours, days

    # ── Tab containers ───────────────────────────────────────
    avail_ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    demand_ct  = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    skills_ct  = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    quota_ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    force_ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    social_ct  = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    weights_ct = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)
    output_ct  = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)

    def _scrollable_tab(ct):
        return ft.Container(
            content=ft.Row(
                controls=[ct],
                scroll=ft.ScrollMode.ALWAYS,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START),
            padding=10, expand=True)

    W_LBL = 80; W_CELL = 50; H_BTN = 26; H_TF = 30

    # ── Reusable toggle helpers ──────────────────────────────
    def make_toggle(sd, key, default):
        if key not in sd:
            sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = 1 - _sd[_k]
            e.control.content.value = str(_sd[_k])
            e.control.bgcolor = ft.Colors.GREEN_700 if _sd[_k] else ft.Colors.RED_700
            e.control.update()
        return ft.Container(
            content=ft.Text(str(sd[key]), color=ft.Colors.WHITE, size=11,
                            weight=ft.FontWeight.BOLD),
            width=W_CELL, height=H_BTN, data=key,
            bgcolor=ft.Colors.GREEN_700 if sd[key] else ft.Colors.RED_700,
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    _avail_lbl  = {1: "1", 0: "0", 2: "!"}
    _avail_clr  = {1: ft.Colors.GREEN_700, 0: ft.Colors.RED_700, 2: ft.Colors.ORANGE_700}
    _avail_next = {1: 0, 0: 2, 2: 1}

    def make_avail_toggle(sd, key, default=1):
        if key not in sd:
            sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = _avail_next[_sd[_k]]
            nv = _sd[_k]
            e.control.content.value = _avail_lbl[nv]
            e.control.bgcolor = _avail_clr[nv]
            e.control.update()
        return ft.Container(
            content=ft.Text(_avail_lbl[sd[key]], color=ft.Colors.WHITE, size=11,
                            weight=ft.FontWeight.BOLD),
            width=W_CELL, height=H_BTN, data=key, bgcolor=_avail_clr[sd[key]],
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    def make_force_toggle(sd, key, default, task_bg, task_fg):
        if key not in sd:
            sd[key] = default
        def _click(e, _sd=sd, _k=key, _tbg=task_bg, _tfg=task_fg):
            _sd[_k] = 1 - _sd[_k]
            e.control.content.value = str(_sd[_k])
            if _sd[_k]:
                e.control.bgcolor = _tbg
                e.control.content.color = _tfg
            else:
                e.control.bgcolor = ft.Colors.GREY_300
                e.control.content.color = ft.Colors.GREY_600
            e.control.update()
        return ft.Container(
            content=ft.Text(str(sd[key]),
                            color=task_fg if sd[key] else ft.Colors.GREY_600,
                            size=11, weight=ft.FontWeight.BOLD),
            width=W_CELL, height=H_BTN, data=key,
            bgcolor=task_bg if sd[key] else ft.Colors.GREY_300,
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

    def hdr_row(labels, w=None):
        if w is None:
            w = W_CELL
        return ft.Row(
            [ft.Container(width=W_LBL)] +
            [ft.Container(
                ft.Text(l, size=9, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=w, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for l in labels],
            spacing=2, wrap=False)

    # ── Navigation Dropdown helper ───────────────────────────
    def make_nav_dropdown(label, value, options, on_change, width=200):
        dd = ft.Dropdown(
            label=label,
            value=value,
            options=[ft.dropdown.Option(o) for o in options],
            width=width,
        )

        def _nav(direction):
            current = dd.value
            if current in options:
                idx = options.index(current)
            else:
                idx = 0
            new_idx = (idx + direction) % len(options)
            dd.value = options[new_idx]
            dd.update()
            on_change(options[new_idx])

        def _dd_change(e):
            on_change(e.control.value)

        dd.on_change = _dd_change

        def _nav_btn(icon, direction):
            return ft.IconButton(
                icon=icon,
                icon_size=18,
                tooltip="Previous" if direction == -1 else "Next",
                style=ft.ButtonStyle(
                    padding=ft.padding.all(4),
                    shape=ft.RoundedRectangleBorder(radius=6),
                ),
                on_click=lambda e, d=direction: _nav(d),
            )

        return ft.Row(
            [_nav_btn(ft.Icons.CHEVRON_LEFT, -1), dd, _nav_btn(ft.Icons.CHEVRON_RIGHT, 1)],
            spacing=2,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # ── Tab builders ─────────────────────────────────────────

    def build_avail():
        people, tasks, hours, days = dims()
        if not days:
            return

        if avail_filter_st[0] not in days:
            avail_filter_st[0] = days[0]

        def _on_day_change(new_val):
            avail_filter_st[0] = new_val
            build_avail()

        def _reset_avail(e):
            avail_st.clear()
            build_avail()

        def _rand_avail(e):
            for p in people:
                for j in days:
                    for h in hours[j]:
                        avail_st[(p, h, j)] = random.choice([0, 1, 2])
            build_avail()

        nav_day = make_nav_dropdown(
            label="Select Day",
            value=avail_filter_st[0],
            options=days,
            on_change=_on_day_change,
            width=200,
        )

        buf = [ft.Row([
            nav_day,
            make_reset_btn("Reset to Default", _reset_avail),
            ft.Container(
                content=ft.Text("Random Avail (All Days)", color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD, size=12),
                bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                on_click=_rand_avail, width=180, alignment=ft.alignment.center),
        ], spacing=20)]

        j = avail_filter_st[0]
        day_hours = hours[j]
        buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
        buf.append(ft.Row(
            [ft.Container(width=W_LBL + W_CELL + 4)] +
            [ft.Container(
                ft.Text(l, size=9, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for l in day_hours],
            spacing=2, wrap=False))

        for p in people:
            def _make_row_toggle(_p=p, _j=j):
                def _click(e):
                    first_val = avail_st.get((_p, hours[_j][0], _j), 1)
                    new_val = _avail_next[first_val]
                    for _h in hours[_j]:
                        avail_st[(_p, _h, _j)] = new_val
                    build_avail()
                return ft.Container(
                    content=ft.Text("row", color=ft.Colors.WHITE, size=10),
                    width=W_CELL, height=H_BTN, bgcolor=ft.Colors.BLUE_GREY_400,
                    alignment=ft.alignment.center, border_radius=4, on_click=_click)

            row_btn = _make_row_toggle(p, j)
            buf.append(ft.Row(
                [plbl(p), row_btn] +
                [make_avail_toggle(avail_st, (p, h, j), 1) for h in day_hours],
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        avail_ct.controls = buf
        page.update()

    def build_demand():
        people, tasks, hours, days = dims()
        if not days:
            return

        if demand_filter_st[0] not in days:
            demand_filter_st[0] = days[0]

        def _on_day_change(new_val):
            demand_filter_st[0] = new_val
            build_demand()

        def _reset_demand(e):
            demand_st.clear()
            validation_errors["demand"].clear()
            _update_solve_blocked()
            build_demand()

        def _rand_demand(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        demand_st[(t, h, j)] = str(random.randint(0, 4))
            validation_errors["demand"].clear()
            _update_solve_blocked()
            build_demand()

        def _zero_demand(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        demand_st[(t, h, j)] = ""
            validation_errors["demand"].clear()
            _update_solve_blocked()
            build_demand()

        nav_day = make_nav_dropdown(
            label="Select Day",
            value=demand_filter_st[0],
            options=days,
            on_change=_on_day_change,
            width=200,
        )

        demand_error_text = ft.Text("", color=ft.Colors.RED_400, size=12, visible=False)

        buf = [ft.Row([
            nav_day,
            make_reset_btn("Reset to Default", _reset_demand),
            ft.Container(
                content=ft.Text("Set All to 0", color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD, size=12),
                bgcolor=ft.Colors.ORANGE_700, padding=10, border_radius=4,
                on_click=_zero_demand, width=150, alignment=ft.alignment.center),
            ft.Container(
                content=ft.Text("Random Demand (All Days)", color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD, size=12),
                bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                on_click=_rand_demand, width=190, alignment=ft.alignment.center),
        ], spacing=20)]

        j = demand_filter_st[0]
        day_hours = hours[j]
        buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
        buf.append(demand_error_text)
        buf.append(hdr_row(day_hours))

        for t in tasks:
            cells = []
            for h in day_hours:
                k = (t, h, j)
                if k not in demand_st:
                    demand_st[k] = "1"
                is_valid = _validate_nonneg_int(demand_st[k])
                tf = ft.TextField(
                    value=demand_st[k], width=W_CELL, height=H_TF, text_size=11,
                    data=k, content_padding=ft.padding.all(2),
                    border_color=ft.Colors.RED_400 if not is_valid else None)
                def _ch(e, _k=k, _err=demand_error_text):
                    demand_st[_k] = e.control.value
                    if _validate_nonneg_int(e.control.value):
                        validation_errors["demand"].discard(_k)
                        e.control.border_color = None
                    else:
                        validation_errors["demand"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    has_err = bool(validation_errors["demand"])
                    _err.value = (
                        f"⚠ {len(validation_errors['demand'])} cell(s) have invalid values. "
                        "Only non-negative integers allowed.") if has_err else ""
                    _err.visible = has_err
                    _update_solve_blocked()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row([lbl(t)] + cells, spacing=2, wrap=False))

        buf.append(ft.Divider())
        demand_ct.controls = buf
        page.update()

    def build_skills():
        if not _needs_rebuild("skills"):
            return
        people, tasks, hours, days = dims()

        def _reset_skills(e):
            skills_st.clear()
            _invalidate_cache()
            build_skills()

        def _rand_skills(e):
            for p in people:
                for t in tasks:
                    skills_st[(p, t)] = random.choice([0, 1])
            _invalidate_cache()
            build_skills()

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=16),
            ft.Row([
                make_reset_btn("Reset Skills", _reset_skills),
                ft.Container(
                    content=ft.Text("Random Skills", color=ft.Colors.WHITE,
                                    weight=ft.FontWeight.BOLD, size=12),
                    bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                    on_click=_rand_skills, width=150, alignment=ft.alignment.center),
            ], spacing=10),
            hdr_row(tasks, 70),
        ]
        for p in people:
            buf.append(ft.Row(
                [plbl(p)] + [make_toggle(skills_st, (p, t), 1) for t in tasks],
                spacing=2))

        skills_ct.controls = buf
        page.update()

    def build_quota():
        if not _needs_rebuild("quota"):
            return
        people, tasks, hours, days = dims()

        def _reset_quota(e):
            quota_st.clear()
            validation_errors["quota"].clear()
            _update_solve_blocked()
            _invalidate_cache()
            build_quota()

        def _rand_quota(e):
            for p in people:
                for t in tasks:
                    quota_st[(p, t)] = str(random.randint(0, 2))
            validation_errors["quota"].clear()
            _update_solve_blocked()
            _invalidate_cache()
            build_quota()

        quota_error_text = ft.Text("", color=ft.Colors.RED_400, size=12, visible=False)

        buf = [
            ft.Text("Minimum Quota Matrix", weight=ft.FontWeight.BOLD, size=16),
            ft.Row([
                make_reset_btn("Reset Quota", _reset_quota),
                ft.Container(
                    content=ft.Text("Random Quota", color=ft.Colors.WHITE,
                                    weight=ft.FontWeight.BOLD, size=12),
                    bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                    on_click=_rand_quota, width=150, alignment=ft.alignment.center),
            ], spacing=10),
            quota_error_text,
            hdr_row(tasks),
        ]
        for p in people:
            cells = []
            for t in tasks:
                k = (p, t)
                if k not in quota_st:
                    quota_st[k] = ""
                is_valid = _validate_nonneg_int(quota_st[k])
                tf = ft.TextField(
                    value=quota_st[k], width=W_CELL, height=H_TF, text_size=11,
                    data=k, content_padding=ft.padding.all(2),
                    border_color=ft.Colors.RED_400 if not is_valid else None)
                def _ch(e, _k=k, _err=quota_error_text):
                    quota_st[_k] = e.control.value
                    if _validate_nonneg_int(e.control.value):
                        validation_errors["quota"].discard(_k)
                        e.control.border_color = None
                    else:
                        validation_errors["quota"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    has_err = bool(validation_errors["quota"])
                    _err.value = (
                        f"⚠ {len(validation_errors['quota'])} cell(s) have invalid values. "
                        "Only non-negative integers allowed.") if has_err else ""
                    _err.visible = has_err
                    _update_solve_blocked()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row([plbl(p)] + cells, spacing=2, wrap=False))

        quota_ct.controls = buf
        page.update()

    def build_force():
        people, tasks, hours, days = dims()
        if not days or not tasks:
            return

        if force_filter_st[0] not in days:
            force_filter_st[0] = days[0]
        if force_filter_st[1] not in tasks:
            force_filter_st[1] = tasks[0]

        def _on_day_change(new_val):
            force_filter_st[0] = new_val
            build_force()

        def _on_task_change(new_val):
            force_filter_st[1] = new_val
            build_force()

        def _reset_force_all(e):
            force_st.clear()
            build_force()

        def _reset_force_current(e):
            j = force_filter_st[0]
            t = force_filter_st[1]
            for p in people:
                for h in hours[j]:
                    force_st.pop((p, t, h, j), None)
            build_force()

        nav_day  = make_nav_dropdown(
            label="Select Day",
            value=force_filter_st[0],
            options=days,
            on_change=_on_day_change,
            width=150,
        )
        nav_task = make_nav_dropdown(
            label="Select Task",
            value=force_filter_st[1],
            options=tasks,
            on_change=_on_task_change,
            width=200,
        )

        buf = [ft.Row([
            nav_day,
            nav_task,
            make_reset_btn("Reset All", _reset_force_all),
            make_reset_btn("Reset Current", _reset_force_current),
        ], spacing=20)]

        j = force_filter_st[0]
        t = force_filter_st[1]
        day_hours = hours[j]

        task_idx = tasks.index(t) if t in tasks else 0
        task_bg, task_fg = TASK_COLORS[task_idx % len(TASK_COLORS)]

        buf.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=14))
        buf.append(hdr_row(day_hours))

        for p in people:
            buf.append(ft.Row(
                [plbl(p)] +
                [make_force_toggle(force_st, (p, t, h, j), 0, task_bg, task_fg)
                 for h in day_hours],
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        force_ct.controls = buf
        page.update()

    def build_social():
        if not _needs_rebuild("social"):
            return
        people, tasks, hours, days = dims()
        buf = []
        if len(people) < 2:
            social_ct.controls = buf; page.update(); return

        def _reset_social(e):
            social_st.clear()
            hard_enemies_st[0] = False
            _invalidate_cache()
            build_social()

        def _rand_social(e):
            for i, p1 in enumerate(people):
                for p2 in people[i + 1:]:
                    social_st[(p1, p2)] = random.choice([-1, 0, 1])
            _invalidate_cache()
            build_social()

        buf.append(ft.Row([
            make_reset_btn("Reset to Default", _reset_social),
            ft.Container(
                content=ft.Text("Random Social", color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD, size=12),
                bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                on_click=_rand_social, width=150, alignment=ft.alignment.center),
        ], spacing=10))

        sw_hard_enemies = ft.Switch(label="Enemies: Hard Constraint", value=hard_enemies_st[0])
        def _toggle_hard_enemies(e):
            hard_enemies_st[0] = e.control.value
        sw_hard_enemies.on_change = _toggle_hard_enemies
        buf.append(ft.Row([sw_hard_enemies], spacing=20))

        buf.append(hdr_row(people[1:]))
        _map_lbl  = {0: "~", 1: "+", -1: "-"}
        _map_clr  = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_700, -1: ft.Colors.RED_700}
        _next_val = {0: 1, 1: -1, -1: 0}
        for i, p1 in enumerate(people):
            cells = []
            for p2 in people[1:]:
                j2 = people.index(p2)
                if j2 > i:
                    k = (p1, p2)
                    if k not in social_st:
                        social_st[k] = 0
                    def _click(e, _k=k):
                        social_st[_k] = _next_val[social_st[_k]]
                        nv = social_st[_k]
                        e.control.content.value = _map_lbl[nv]
                        e.control.bgcolor = _map_clr[nv]
                        e.control.update()
                    btn = ft.Container(
                        content=ft.Text(_map_lbl[social_st[k]], color=ft.Colors.WHITE,
                                        size=12, weight=ft.FontWeight.BOLD),
                        width=W_CELL, height=H_BTN, data=k, bgcolor=_map_clr[social_st[k]],
                        alignment=ft.alignment.center, border_radius=4, on_click=_click)
                    cells.append(btn)
                else:
                    cells.append(ft.Container(width=W_CELL))
            if cells:
                buf.append(ft.Row([plbl(p1)] + cells, spacing=2, wrap=False))

        social_ct.controls = buf
        page.update()

    def build_weights():
        def _reset_weights(e):
            weights_order.clear()
            weights_order.extend(list(DEFAULT_WEIGHTS.keys()))
            for k in DEFAULT_WEIGHTS.keys():
                weights_enabled[k] = True
                weights_st[k] = DEFAULT_WEIGHTS[k]
            build_weights()

        for i, key in enumerate(weights_order):
            weights_st[key] = SORTED_VALUES[i] if weights_enabled[key] else 0

        header = ft.Column([
            ft.Row([
                ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=16),
                make_reset_btn("Reset to Default", _reset_weights),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=420),
            ft.Text("Drag items to reorder. Top items = Higher Cost.", italic=True, size=12),
            ft.Divider(),
        ], spacing=5)

        items_controls = []

        def handle_reorder(e):
            item = weights_order.pop(e.old_index)
            weights_order.insert(e.new_index, item)
            build_weights()

        for i, key in enumerate(weights_order):
            val = SORTED_VALUES[i] if weights_enabled[key] else 0
            sw = ft.Switch(value=weights_enabled[key], data=key)
            def _toggle(e, _k=key):
                weights_enabled[_k] = e.control.value
                build_weights()
            sw.on_change = _toggle
            card = ft.Container(
                content=ft.Row([
                    ft.Text(f"#{i + 1}", width=30, weight=ft.FontWeight.BOLD),
                    ft.Text(key, expand=True, weight=ft.FontWeight.W_800),
                    ft.Text(f"{val}            ",
                            color=ft.Colors.BLACK if weights_enabled[key] else ft.Colors.GREY_500,
                            size=16),
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

    # ── Output: single-run column builder ────────────────────
    #
    # Returns a ft.Container(Column) for one solver run.
    # is_live=True → the column represents the currently-running solve;
    # the header is orange and issue sections are omitted while solving.

    def _build_run_column(run_idx, sol, people, tasks, hours, days,
                          availability, emergency, is_live=False):
        status     = sol.get("status", "Solving...")
        is_solving = "Solving" in status
        assignment = sol.get("assignment", {})

        buf = []

        # ── Column header ─────────────────────────────────────
        if is_live and is_solving:
            hdr_label = f"⟳  Run #{run_idx + 1}  ·  Solving…"
            hdr_bg    = "#E65100"
        else:
            hdr_label = f"Run #{run_idx + 1}  ·  {status}"
            hdr_bg    = "#1565C0"

        buf.append(ft.Container(
            content=ft.Text(hdr_label, weight=ft.FontWeight.BOLD, size=13,
                            color=ft.Colors.WHITE, no_wrap=True),
            bgcolor=hdr_bg,
            padding=ft.padding.symmetric(8, 12),
            border_radius=6,
            margin=ft.margin.only(bottom=6)))
        
        # ── Solve time + MIP gap (solo runs finalizados) ──────────
        if not (is_live and is_solving):
            solve_time = sol.get("solve_time", 0.0)
            mip_gap    = sol.get("mip_gap", 0.0)
            buf.append(ft.Text(
                f"⏱  Solve time: {solve_time:.1f} s",
                size=12, italic=True, color=ft.Colors.GREY_700))
            buf.append(ft.Text(
                f"◎  MIP Gap: {mip_gap * 100:.3f} %",
                size=12, italic=True, color=ft.Colors.GREY_700))

        # Grid sizing constants — defined here so col_width can always use them
        CW = 40; Ch = 12; NW = 50; TW = 25 ###########################################################################

        # ── Spinner (before first intermediate solution arrives) ──
        if is_live and is_solving and not any(assignment.get(d) for d in days):
            buf.append(ft.Row([
                ft.ProgressRing(width=22, height=22, stroke_width=3),
                ft.Text("Solving in background…", italic=True, size=12),
            ], spacing=10))
        else:
            # ── Live progress indicator (intermediate solutions) ──
            if is_live and is_solving:
                buf.append(ft.Row([
                    ft.ProgressRing(width=16, height=16, stroke_width=2),
                    ft.Text("Updating…", italic=True, size=11,
                            color=ft.Colors.ORANGE_700),
                ], spacing=6))

            # ── Assignment grid ───────────────────────────────────
            tc = {}
            for i, t in enumerate(tasks):
                bg, fg = TASK_COLORS[i % len(TASK_COLORS)]
                tc[t] = (bg, fg)

            for j in days:
                if j not in assignment:
                    continue
                day_hours = hours[j]
                buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=16))

                hdr = (
                    [ft.Container(
                        ft.Text("Person", size=11, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                        width=NW, height=Ch, bgcolor="#546E7A",
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#455A64"))] +
                    [ft.Container(
                        ft.Text(h, size=11, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                        width=CW, height=Ch, bgcolor="#546E7A",
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#455A64"))
                     for h in day_hours] +
                    [ft.Container(
                        ft.Text("Total", size=11, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                        width=TW, height=Ch, bgcolor="#546E7A",
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#455A64"))])
                buf.append(ft.Row(hdr, spacing=0, wrap=False))

                asgn = assignment[j]
                for idx_p, p in enumerate(people):
                    is_cap     = captains_st.get(p, 0) == 1
                    row_bg     = "#ECEFF1" if idx_p % 2 == 0 else "#FFFFFF"
                    name_color = CAPTAIN_BG if is_cap else ft.Colors.BLACK
                    cells = [ft.Container(
                        ft.Text(p, size=12, weight=ft.FontWeight.BOLD, color=name_color),
                        width=NW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center_left,
                        padding=ft.padding.only(left=8),
                        border=ft.border.all(1, "#CFD8DC"))]
                    total = 0
                    for h in day_hours:
                        task  = asgn.get(p, {}).get(h)
                        avail = availability.get((p, h, j), 1)
                        griev = emergency.get((p, h, j), 0)
                        if avail == 0:    brd = ft.border.all(1.5, UNAVAIL_COLOR)
                        elif griev == 1:  brd = ft.border.all(1.5, EMERG_COLOR)
                        else:             brd = ft.border.all(0.5, AVAIL_COLOR)
                        if task:
                            bg, fg = tc[task]; total += 1
                            cell = ft.Container(
                                ft.Text(task, size=11, weight=ft.FontWeight.BOLD, color=fg,
                                        text_align=ft.TextAlign.CENTER),
                                width=CW, height=Ch, bgcolor=bg,
                                alignment=ft.alignment.center, border=brd, border_radius=4)
                        else:
                            cell = ft.Container(width=CW, height=Ch, bgcolor=row_bg, border=brd)
                        cells.append(cell)
                    cells.append(ft.Container(
                        ft.Text(str(int(total)), size=12, weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER),
                        width=TW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#CFD8DC")))
                    buf.append(ft.Row(cells, spacing=0, wrap=False))

                legend_items = [
                    ft.Container(
                        ft.Text(t, size=10, weight=ft.FontWeight.BOLD, color=tc[t][1]),
                        bgcolor=tc[t][0], padding=ft.padding.symmetric(6, 10),
                        border_radius=4)
                    for t in tasks]
                for lbl_txt, clr in [("Available",   AVAIL_COLOR),
                                     ("Emergency",   EMERG_COLOR),
                                     ("Unavailable", UNAVAIL_COLOR)]:
                    legend_items.append(ft.Container(
                        ft.Text(lbl_txt, size=10, color=ft.Colors.WHITE),
                        bgcolor=clr, padding=ft.padding.symmetric(6, 10), border_radius=4))
                buf.append(ft.Row(legend_items, spacing=8, wrap=True))
                buf.append(ft.Divider())

            # ── Issue sections (only for finalised runs) ──────────
            if not (is_live and is_solving):
                missing_issues = sol.get("missing", [])
                workload       = sol.get("workload", {})
                gaps           = sol.get("gaps", [])
                soc_issues     = sol.get("social_issues", [])
                cap_issues     = sol.get("captain_issues", [])
                emerg_issues   = sol.get("emerg_issues", [])
                rot_issues     = sol.get("rotation_issues", [])
                quota_issues   = sol.get("quota_issues", [])

                for section_title, issues, empty_msg in [
                    ("MISSING STAFF",                  missing_issues, "None -- all demand covered!"),
                    ("SHIFT SEGMENTS / FRAGMENTATION", gaps,           "Single block shifts! (Perfect)"),
                    ("ROTATION FATIGUE",               rot_issues,     "No rotation tasks defined."),
                    ("SOCIAL",                         soc_issues,     "None -- all respected!"),
                    ("CAPTAIN PRESENCE",               cap_issues,     "No captains designated or all hours covered."),
                    ("EMERGENCY CALL-INS",             emerg_issues,   "None -- no emergency hours used!"),
                    ("QUOTA FULFILMENT",               quota_issues,   "No quotas defined."),
                ]:
                    if section_title == "SHIFT SEGMENTS / FRAGMENTATION" and sol.get("enforced_rest", False):
                        continue
                    buf.append(ft.Text(section_title, weight=ft.FontWeight.BOLD, size=14))
                    if issues:
                        for line in issues:
                            buf.append(ft.Text(f"  {line}", size=12))
                    else:
                        buf.append(ft.Text(f"  {empty_msg}", size=12, italic=True))

                buf.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=14))
                for p in people:
                    buf.append(ft.Text(f"  {p}: {workload.get(p, 0):.0f} hours", size=12))
                buf.append(ft.Text(
                    f"  Global range: max={sol.get('w_max', 0):.0f}, min={sol.get('w_min', 0):.0f}",
                    size=12, italic=True))

        # Compute column width from the widest day's grid
        max_hours  = max((len(hours[d]) for d in days), default=0)
        col_width  = max(NW + CW * max_hours + TW + 30, 460)

        return ft.Container(
            content=ft.Column(controls=buf, spacing=5, tight=True),
            width=col_width,
            padding=ft.padding.symmetric(horizontal=16),
            # Vertical separator between columns
            border=ft.border.only(right=ft.border.BorderSide(1, "#B0BEC5")))

    # ── Output: full rebuild from history + optional live col ─
    #
    # Calling with no arguments finalises the output (no live column).
    # Calling with live_* arguments appends a live column at the right.

    def _rebuild_output(live_sol=None, live_people=None, live_tasks=None,
                        live_hours=None, live_days=None,
                        live_avail=None, live_emerg=None):
        cols = []
        for i, entry in enumerate(solution_history):
            cols.append(_build_run_column(
                i,
                entry["sol"],   entry["people"], entry["tasks"],
                entry["hours"], entry["days"],
                entry["availability"], entry["emergency"]))

        if live_sol is not None:
            cols.append(_build_run_column(
                len(solution_history), live_sol,
                live_people, live_tasks, live_hours, live_days,
                live_avail, live_emerg,
                is_live=True))

        # All run columns share a single horizontal Row (horizontal overflow
        # is handled by the _scrollable_tab wrapper's Row scroll).
        # Vertical scroll is provided by output_ct itself.
        output_ct.controls = (
            [ft.Row(controls=cols, spacing=0, wrap=False,
                    vertical_alignment=ft.CrossAxisAlignment.START)]
            if cols else [])

    # ── Solve / Stop ─────────────────────────────────────────

    def do_solve(e):
        # Cancela la ejecución previa si existe. No ponemos None aquí:
        # solve_model (P1) sobreescribirá running_model_ref[0] con el nuevo
        # modelo justo al arrancar, por lo que Stop siempre apunta al activo.
        if running_model_ref[0] is not None:
            running_model_ref[0].terminate()

        if solve_blocked[0]:
            return
        people, tasks, hours, days = dims()

        enforced_rest     = sw_enforced_rest.value
        max_consec_hours  = None
        if enforced_rest:
            raw_y = tf_max_consec.value.strip()
            try:
                max_consec_hours = int(raw_y)
                if str(max_consec_hours) != raw_y or max_consec_hours < 1:
                    raise ValueError
            except (ValueError, TypeError):
                err_max_consec.value   = "Error: Y must be a positive integer."
                err_max_consec.visible = True
                page.update()
                return
            err_max_consec.visible = False
            page.update()

        availability, emergency = {}, {}
        for p in people:
            for j in days:
                for h in hours[j]:
                    val = avail_st.get((p, h, j), 1)
                    availability[(p, h, j)] = 1 if val in (1, 2) else 0
                    emergency[(p, h, j)]    = 1 if val == 2 else 0

        demand = {}
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    raw = demand_st.get((t, h, j), "1").strip()
                    try:    demand[(t, h, j)] = int(raw) if raw else 0
                    except: demand[(t, h, j)] = 0

        skills = {(p, t): skills_st.get((p, t), 1) for p in people for t in tasks}
        force  = {(p, t, h, j): force_st.get((p, t, h, j), 0)
                  for p in people for t in tasks for j in days for h in hours[j]}
        social = {(p1, p2): social_st.get((p1, p2), 0)
                  for i, p1 in enumerate(people) for p2 in people[i + 1:]}
        mq     = {}
        for p in people:
            for t in tasks:
                raw = quota_st.get((p, t), "0").strip()
                try:    mq[(p, t)] = int(raw) if raw else 0
                except: mq[(p, t)] = 0

        rotation  = {t: rotation_st.get(t, 1) for t in tasks}
        pref_cost = {(p, t): 1 for p in people for t in tasks}
        X_prev    = {(p, t, h, j): 0
                     for p in people for t in tasks for j in days for h in hours[j]}
        weights   = weights_st.copy()
        captains  = [p for p in people if captains_st.get(p, 0) == 1]

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, emergency=emergency, demand=demand, skills=skills,
            force=force, social=social, min_quota=mq,
            pref_cost=pref_cost, rotation=rotation, X_prev=X_prev, weights=weights,
            enforced_rest=enforced_rest, max_consec_hours=max_consec_hours,
            captains=captains, solver_params=solver_params_st,
            hard_enemies=hard_enemies_st[0],
            day_heuristics=1 if day_heuristics_sw.value else 0,
        )

        # Show a live column (spinner) alongside any previous runs
        _rebuild_output(
            live_sol={"status": "Solving...", "assignment": {}},
            live_people=people, live_tasks=tasks, live_hours=hours,
            live_days=days, live_avail=availability, live_emerg=emergency)
        switch_page(8)

        # Throttle del UI: fuera del hilo para que la cadencia sea global.
        last_ui_update = [0.0]

        def update_ui_with_temp_solution(partial_sol):
            if not ui_lock.acquire(blocking=False):
                return
            try:
                _rebuild_output(
                    live_sol=partial_sol,
                    live_people=people, live_tasks=tasks, live_hours=hours,
                    live_days=days, live_avail=availability, live_emerg=emergency)
                page.update()
            except Exception:
                pass
            finally:
                ui_lock.release()

        def run_solver():
            try:
                final_sol = solve_model(
                    data, ui_update_callback=update_ui_with_temp_solution,
                    active_model_ref=running_model_ref)
                # Persist the completed run into history
                solution_history.append({
                    "sol":          final_sol,
                    "people":       people,
                    "tasks":        tasks,
                    "hours":        hours,
                    "days":         days,
                    "availability": availability,
                    "emergency":    emergency,
                })
                # Redraw without the live column (it is now in history)
                _rebuild_output()
            except Exception as ex:
                output_ct.controls = [ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=14)]
            # No ponemos running_model_ref[0] = None: dejar la referencia al
            # modelo terminado es inocuo (terminate() sobre un modelo ya
            # finalizado es un no-op en Gurobi) y evita la race condition en
            # la que este hilo borraría la referencia al modelo de una
            # ejecución posterior que haya arrancado entre tanto.
            page.update()

        threading.Thread(target=run_solver, daemon=True).start()

    def do_stop(e):
        # Termina el solver; run_solver() detectará el estado INTERRUPTED,
        # añadirá la solución parcial al historial y eliminará la live column.
        if running_model_ref[0] is not None:
            running_model_ref[0].terminate()

    # ==================================================================
    # SIDEBAR NAVIGATION LAYOUT
    # ==================================================================

    SIDEBAR_WIDTH        = 200
    SIDEBAR_BG           = "#263238"
    SIDEBAR_SELECTED_BG  = "#37474F"
    SIDEBAR_TEXT_COLOR   = "#ECEFF1"
    SIDEBAR_SELECTED_TEXT = "#4FC3F7"

    menu_items_def = [
        ("Dimensions",    ft.Icons.GRID_VIEW,           0),
        ("Availability",  ft.Icons.EVENT_AVAILABLE,      1),
        ("Demand",        ft.Icons.TRENDING_UP,          2),
        ("Skills",        ft.Icons.STAR_BORDER,          3),
        ("Quota",         ft.Icons.FORMAT_LIST_NUMBERED, 4),
        ("Force",         ft.Icons.LOCK_OUTLINE,         5),
        ("Social",        ft.Icons.PEOPLE_OUTLINE,       6),
        ("Weights",       ft.Icons.TUNE,                 7),
        ("Output",        ft.Icons.ASSESSMENT,           8),
    ]

    dim_tab_content = ft.Container(
        content=ft.Row(
            controls=[tf_people, captains_col, tf_tasks, rotation_col,
                      tf_days, hours_col, right_controls_col],
            spacing=20, alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START),
        padding=20, expand=True)

    page_contents = {
        0: dim_tab_content,
        1: _scrollable_tab(avail_ct),
        2: _scrollable_tab(demand_ct),
        3: _scrollable_tab(skills_ct),
        4: _scrollable_tab(quota_ct),
        5: _scrollable_tab(force_ct),
        6: _scrollable_tab(social_ct),
        7: ft.Container(weights_ct, padding=10, expand=True),
        8: _scrollable_tab(output_ct),
    }

    builders = {
        1: build_avail, 2: build_demand, 3: build_skills,
        4: build_quota, 5: build_force,  6: build_social, 7: build_weights}

    content_area = ft.Container(expand=True, padding=0)
    content_area.content = page_contents[0]

    selected_index_ref = [0]
    menu_btn_refs = []

    def switch_page(idx):
        selected_index_ref[0] = idx
        content_area.content = page_contents[idx]
        for i, btn_container in enumerate(menu_btn_refs):
            selected = (i == idx)
            btn_container.bgcolor = SIDEBAR_SELECTED_BG if selected else None
            btn_container.content.controls[0].color = SIDEBAR_SELECTED_TEXT if selected else SIDEBAR_TEXT_COLOR
            btn_container.content.controls[1].color = SIDEBAR_SELECTED_TEXT if selected else SIDEBAR_TEXT_COLOR
            btn_container.content.controls[1].weight = ft.FontWeight.BOLD if selected else None
        if idx in builders:
            with ui_lock:
                builders[idx]()
        page.update()

    def _on_menu_click(e):
        switch_page(e.control.data)

    for label, icon, idx in menu_items_def:
        is_selected = (idx == 0)
        btn = ft.Container(
            content=ft.Row([
                ft.Icon(icon,
                        color=SIDEBAR_SELECTED_TEXT if is_selected else SIDEBAR_TEXT_COLOR,
                        size=20),
                ft.Text(label, size=13,
                        color=SIDEBAR_SELECTED_TEXT if is_selected else SIDEBAR_TEXT_COLOR,
                        weight=ft.FontWeight.BOLD if is_selected else None),
            ], spacing=10),
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            border_radius=8,
            bgcolor=SIDEBAR_SELECTED_BG if is_selected else None,
            on_click=_on_menu_click,
            data=idx,
            ink=True,
        )
        menu_btn_refs.append(btn)

    day_heuristics_sw = ft.Switch(
        label="Day Heuristics",
        value=False,
        tooltip=(
            "Desacopla los días: resuelve un submodelo por día usando "
            "una heurística de ritmo (pacing). "
            "Reduce drásticamente el tiempo de resolución con muchos días, "
            "personas y tareas, a costa de perder la equidad global exacta."
        ),
        label_style=ft.TextStyle(color=ft.Colors.WHITE, size=11),
    )

    solve_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE, size=18),
            ft.Text("SOLVE", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor="#1565C0", padding=ft.padding.symmetric(horizontal=12, vertical=8),
        border_radius=8, on_click=do_solve,
        width=SIDEBAR_WIDTH - 32, alignment=ft.alignment.center)
    solve_btn_ref[0] = solve_btn

    stop_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE, size=18),
            ft.Text("STOP", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor="#C62828", padding=ft.padding.symmetric(horizontal=12, vertical=8),
        border_radius=8, on_click=do_stop,
        width=SIDEBAR_WIDTH - 32, alignment=ft.alignment.center)

    sidebar = ft.Container(
        width=SIDEBAR_WIDTH, bgcolor=SIDEBAR_BG,
        padding=ft.padding.only(top=12, bottom=12, left=8, right=8),
        content=ft.Column(
            controls=[
                ft.Container(
                    content=ft.Text("Staff Scheduler", size=16, weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.WHITE),
                    padding=ft.padding.only(left=8, bottom=4)),
                ft.Container(
                    content=ft.Column(
                        [solve_btn, day_heuristics_sw, stop_btn],
                        spacing=6,
                    ),
                    padding=ft.padding.only(bottom=8),
                ),
                ft.Divider(color="#455A64", height=1),
                ft.Column(
                    controls=menu_btn_refs, spacing=2,
                    scroll=ft.ScrollMode.AUTO, expand=True),
            ],
            spacing=4, expand=True),
        border=ft.border.only(right=ft.border.BorderSide(1, "#455A64")))

    main_layout = ft.Row(
        controls=[sidebar, content_area],
        spacing=0, expand=True,
        vertical_alignment=ft.CrossAxisAlignment.START)

    page.add(main_layout)

    # Initial builds
    build_captains_list()
    build_rotation_list()
    build_hours_per_day()

ft.app(target=main, view=ft.AppView.WEB_BROWSER)