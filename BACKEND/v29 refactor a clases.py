import flet as ft
import random
import threading

from solve_model_pace_10 import solve_model


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "W_COVERAGE": 30000, "W_FORCE": 10000, "W_CAPTAIN": 8000,
    "W_STABILITY": 7000, "W_EQ_DAY": 5000, "W_GAP": 1000,
    "W_EMERG": 750,      "W_EQ_GLOBAL": 500, "W_ROTATION": 100,
    "W_SOCIAL": 50,      "W_QUOTA": 10,      "W_PREF": 1,
}
SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

DEFAULT_SOLVER_PARAMS = {
    "TimeLimit": 1200, "MIPGap": 0.01, "MIPFocus": 2,
    "Threads": 0, "Presolve": 2, "Symmetry": 2,
    "Disconnected": 2, "IntegralityFocus": 1, "Method": 3,
}

DEFAULT_HOURS_TEXT = (
    "08:00\n09:00\n10:00\n11:00\n12:00\n"
    "13:00\n14:00\n15:00\n16:00\n17:00"
)

TASK_COLORS = [
    ("#CE93D8","#000000"), ("#80DEEA","#000000"), ("#FFF59D","#000000"),
    ("#A5D6A7","#000000"), ("#FFAB91","#000000"), ("#90CAF9","#000000"),
    ("#F48FB1","#000000"), ("#E6EE9C","#000000"), ("#B0BEC5","#000000"),
    ("#FFCC80","#000000"), ("#80CBC4","#000000"), ("#B39DDB","#000000"),
    ("#EF9A9A","#000000"), ("#C5E1A5","#000000"), ("#81D4FA","#000000"),
    ("#FFE082","#000000"), ("#F8BBD0","#000000"), ("#BCAAA4","#000000"),
    ("#A1887F","#FFFFFF"), ("#7986CB","#FFFFFF"), ("#4DB6AC","#FFFFFF"),
    ("#FF8A65","#000000"), ("#AED581","#000000"), ("#4FC3F7","#000000"),
    ("#DCE775","#000000"), ("#BA68C8","#FFFFFF"), ("#4DD0E1","#000000"),
    ("#E57373","#000000"), ("#9575CD","#FFFFFF"), ("#FFD54F","#000000"),
]

CAPTAIN_BG        = "#E65100"
CAPTAIN_FG        = "#FFFFFF"   # defined in original (unused but kept for fidelity)
UNAVAIL_COLOR     = "#D32F2F"
EMERG_COLOR       = "#F57C00"
AVAIL_COLOR       = "#388E3C"
DIFF_ADD_COLOR    = "#2E7D32"
DIFF_REMOVE_COLOR = "#C62828"
DIFF_CHANGE_COLOR = "#E65100"

SIDEBAR_WIDTH         = 200
SIDEBAR_BG            = "#263238"
SIDEBAR_SELECTED_BG   = "#37474F"
SIDEBAR_TEXT_COLOR    = "#ECEFF1"
SIDEBAR_SELECTED_TEXT = "#4FC3F7"


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE  — single source of truth for all application data
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    _DEFAULT_PEOPLE = (
        "Arnaud\nNina\nJoseph\nChloé\nNiels\nZiad\nTristan\nJasmin\nMarine\nNoé\n"
        "Jules R\nGuillaume\nNoémie\nStanislas\nTérence\nManon In\nManon L\nPauline\n"
        "Lucie\nDarius\nMattia\nPierre\nBaptiste B\nVincent\nMadeleine\nIlhan\nMatteo\n"
        "Alexandre L\nPablo\nJenaya\nLiv\nFaustin\nKenza\nJuliette B\nSarah\n"
        "Alexandre B\nRémi\nGabi\nJeanne B\nMatthieu A\nInès\nMaxime N\nAriane\nMatthias"
    )
    _DEFAULT_TASKS = (
        "Pénélope\nAugustin B\nAgathe\nRafael\nLuna\nLéna\nCamille\nJuliette M\n"
        "Paul\nManon P\nMatthieu G\nAlix\nNadim\nJean-Louis\nArthur"
    )
    _DEFAULT_DAYS = "Mon\nTue\nWed"

    def __init__(self):
        self.people_text = self._DEFAULT_PEOPLE
        self.tasks_text  = self._DEFAULT_TASKS
        self.days_text   = self._DEFAULT_DAYS

        self.avail_st    : dict = {}
        self.demand_st   : dict = {}
        self.skills_st   : dict = {}
        self.force_st    : dict = {}
        self.social_st   : dict = {}
        self.quota_st    : dict = {}
        self.rotation_st : dict = {}
        self.captains_st : dict = {}
        self.hard_enemies: bool = False
        self.hours_per_day: dict = {}

        self.consec_global_val   : str  = ""
        self.consec_personalized : bool = False
        self.consec_per_person   : dict = {}

        self.avail_filter  = None
        self.demand_filter = None
        self.force_filter  = [None, None]
        self.quota_filter  = None

        self.validation_errors: dict = {
            "demand": set(), "quota": set(), "consec": set()
        }

        self.solve_blocked     : bool = False
        self.solver_running    : bool = False
        self.running_model_ref : list = [None]

        self.solution_history: list = []
        self.diff_state: dict = {"ref": None, "cmp": None}

        self.weights_st      = DEFAULT_WEIGHTS.copy()
        self.weights_order   = list(DEFAULT_WEIGHTS.keys())
        self.weights_enabled = {k: True for k in DEFAULT_WEIGHTS}
        self.solver_params   = DEFAULT_SOLVER_PARAMS.copy()

        self._build_cache: dict = {}

    # ── Dimension parsing ──────────────────────────────────────────────────

    def dims(self):
        def _parse(txt):
            return list(dict.fromkeys(x.strip() for x in txt.split("\n") if x.strip()))
        people = _parse(self.people_text)
        tasks  = _parse(self.tasks_text)
        days   = _parse(self.days_text)
        default_hrs = _parse(DEFAULT_HOURS_TEXT)
        hours = {}
        for j in days:
            raw    = self.hours_per_day.get(j, DEFAULT_HOURS_TEXT)
            parsed = _parse(raw)
            hours[j] = parsed if parsed else default_hrs
        return people, tasks, hours, days

    # ── Rebuild-cache helpers ──────────────────────────────────────────────

    def _dims_hash(self) -> int:
        # Hash raw strings — cheap, avoids parsing full lists
        return hash((
            self.people_text, self.tasks_text, self.days_text,
            tuple(sorted(self.hours_per_day.items())),
        ))

    def needs_rebuild(self, tab_name: str) -> bool:
        key = self._dims_hash()
        if self._build_cache.get(tab_name) == key:
            return False
        self._build_cache[tab_name] = key
        return True

    def invalidate_cache(self):
        self._build_cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS  — pure static widget factories & validators
# [EFFICIENCY LVL 1] avail toggle dicts promoted to class-level constants
# ══════════════════════════════════════════════════════════════════════════════

class UIHelpers:
    W_LBL  = 80
    W_CELL = 50
    H_BTN  = 26
    H_TF   = 30

    # [LVL 1] — defined once at class level, never recreated per call
    _AVAIL_LBL  = {1: "1",  0: "0",  2: "!"}
    _AVAIL_CLR  = {1: ft.Colors.GREEN_700, 0: ft.Colors.RED_700, 2: ft.Colors.ORANGE_700}
    _AVAIL_NEXT = {1: 0, 0: 2, 2: 1}

    # ── Validators ────────────────────────────────────────────────────────

    @staticmethod
    def validate_nonneg_int(value: str) -> bool:
        v = value.strip()
        if v == "":
            return True
        try:
            n = int(v)
            return n >= 0 and str(n) == v
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_positive_int(value: str) -> bool:
        v = value.strip()
        if v == "":
            return True
        try:
            n = int(v)
            return n >= 1 and str(n) == v
        except (ValueError, TypeError):
            return False

    # ── Generic widget factories ──────────────────────────────────────────

    @staticmethod
    def make_reset_btn(text: str, on_click) -> ft.Container:
        return ft.Container(
            content=ft.Text(text, color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=12),
            bgcolor=ft.Colors.RED_500, padding=10, border_radius=4,
            on_click=on_click, width=150, alignment=ft.alignment.center)

    @staticmethod
    def lbl(text: str, w: int = None) -> ft.Container:
        return ft.Container(
            ft.Text(text, size=11, no_wrap=True),
            width=w or UIHelpers.W_LBL)

    @staticmethod
    def plbl(name: str, captains_st: dict, w: int = None) -> ft.Container:
        is_cap = captains_st.get(name, 0) == 1
        return ft.Container(
            ft.Text(name, size=11, no_wrap=True,
                    weight=ft.FontWeight.BOLD if is_cap else None,
                    color=CAPTAIN_BG if is_cap else None),
            width=w or UIHelpers.W_LBL)

    @staticmethod
    def hdr_row(labels: list, w: int = None) -> ft.Row:
        w = w or UIHelpers.W_CELL
        return ft.Row(
            [ft.Container(width=UIHelpers.W_LBL)] +
            [ft.Container(
                ft.Text(lbl, size=9, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=w, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for lbl in labels],
            spacing=2, wrap=False)

    @staticmethod
    def make_nav_dropdown(label: str, value, options: list,
                          on_change, width: int = 200) -> ft.Row:
        dd = ft.Dropdown(
            label=label, value=value,
            options=[ft.dropdown.Option(o) for o in options],
            width=width)

        def _nav(direction):
            idx = options.index(dd.value) if dd.value in options else 0
            dd.value = options[(idx + direction) % len(options)]
            dd.update()
            on_change(dd.value)

        dd.on_change = lambda e: on_change(e.control.value)

        def _btn(icon, direction):
            return ft.IconButton(
                icon=icon, icon_size=18,
                tooltip="Previous" if direction == -1 else "Next",
                style=ft.ButtonStyle(padding=ft.padding.all(4),
                                     shape=ft.RoundedRectangleBorder(radius=6)),
                on_click=lambda e, d=direction: _nav(d))

        return ft.Row(
            [_btn(ft.Icons.CHEVRON_LEFT, -1), dd, _btn(ft.Icons.CHEVRON_RIGHT, 1)],
            spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # ── Toggle factories ──────────────────────────────────────────────────

    @staticmethod
    def make_toggle(sd: dict, key, default: int) -> ft.Container:
        if key not in sd:
            sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = 1 - _sd[_k]
            e.control.content.value = str(_sd[_k])
            e.control.bgcolor = ft.Colors.GREEN_700 if _sd[_k] else ft.Colors.RED_700
            e.control.update()
        return ft.Container(
            content=ft.Text(str(sd[key]), color=ft.Colors.WHITE,
                            size=11, weight=ft.FontWeight.BOLD),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=key,
            bgcolor=ft.Colors.GREEN_700 if sd[key] else ft.Colors.RED_700,
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    @staticmethod
    def make_avail_toggle(sd: dict, key, default: int = 1) -> ft.Container:
        # [LVL 1] uses class-level constants instead of recreating dicts each call
        _lbl  = UIHelpers._AVAIL_LBL
        _clr  = UIHelpers._AVAIL_CLR
        _next = UIHelpers._AVAIL_NEXT
        if key not in sd:
            sd[key] = default
        def _click(e, _sd=sd, _k=key):
            _sd[_k] = _next[_sd[_k]]
            nv = _sd[_k]
            e.control.content.value = _lbl[nv]
            e.control.bgcolor = _clr[nv]
            e.control.update()
        return ft.Container(
            content=ft.Text(_lbl[sd[key]], color=ft.Colors.WHITE,
                            size=11, weight=ft.FontWeight.BOLD),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=key,
            bgcolor=_clr[sd[key]],
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    @staticmethod
    def make_force_toggle(sd: dict, key, default: int,
                          task_bg: str, task_fg: str) -> ft.Container:
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
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=key,
            bgcolor=task_bg if sd[key] else ft.Colors.GREY_300,
            alignment=ft.alignment.center, border_radius=4, on_click=_click)


# ══════════════════════════════════════════════════════════════════════════════
# BASE TAB
# ══════════════════════════════════════════════════════════════════════════════

class BaseTab:
    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)

    def build(self):
        raise NotImplementedError

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[self._ct],
                scroll=ft.ScrollMode.ALWAYS,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START),
            padding=10, expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSIONS TAB
# [EFFICIENCY LVL 3] debounce on all three dimension TextFields
# ══════════════════════════════════════════════════════════════════════════════

class DimensionsTab:

    def __init__(self, state: AppState, page: ft.Page, on_solve_blocked_update):
        self.state = state
        self.page  = page
        self._on_solve_blocked_update = on_solve_blocked_update

        # [LVL 3] debounce timer references — one per dimension field
        self._debounce_people: threading.Timer | None = None
        self._debounce_tasks : threading.Timer | None = None
        self._debounce_days  : threading.Timer | None = None

        self._build_ui()

    # ── [LVL 3] Debounce helper ────────────────────────────────────────────

    def _debounce(self, attr: str, fn, delay: float = 0.3):
        """Cancel any pending timer stored in `attr` and schedule `fn` after `delay` s."""
        old: threading.Timer | None = getattr(self, attr, None)
        if old is not None:
            old.cancel()
        t = threading.Timer(delay, fn)
        t.daemon = True   # won't block process shutdown
        t.start()
        setattr(self, attr, t)

    # ── Initial widget construction ────────────────────────────────────────

    def _build_ui(self):
        s = self.state

        self.tf_people = ft.TextField(
            value=s.people_text, multiline=True, min_lines=1, max_lines=200,
            label="People (one per line)", width=140,
            on_change=self._on_people_change)
        self.tf_tasks = ft.TextField(
            value=s.tasks_text, multiline=True, min_lines=8, max_lines=200,
            label="Tasks (one per line)", width=180, # <--- ADD A FIXED WIDTH
            on_change=self._on_tasks_change)
        self.tf_days = ft.TextField(
            value=s.days_text, multiline=True, min_lines=1, max_lines=200,
            label="Days (one per line)", width=120,
            on_change=self._on_days_change)

        self._err_consec = ft.Text("", color=ft.Colors.RED_400, size=11, visible=False)

        self._tf_consec_global = ft.TextField(
            value="", width=110, height=38, text_size=12,
            hint_text="Max consec", content_padding=ft.padding.all(4),
            on_change=self._on_consec_global_change)

        self._sw_personalize = ft.Switch(
            label="", value=False,
            on_change=self._on_personalize_toggle)

        enforced_rest_col = ft.Column([
            ft.Row([self._sw_personalize, self._tf_consec_global],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self._err_consec,
        ], spacing=5)

        self._param_tfs: dict = {}
        for key, val in s.solver_params.items():
            tf = ft.TextField(label=key, value=str(val), width=120, height=45, text_size=15)
            def _ch(e, _k=key):
                try:
                    s.solver_params[_k] = (
                        float(e.control.value) if "." in e.control.value
                        else int(e.control.value))
                except ValueError:
                    pass
            tf.on_change = _ch
            self._param_tfs[key] = tf

        tf_list = list(self._param_tfs.values())
        params_section = ft.Column([
            ft.Text("Gurobi Parameters", weight=ft.FontWeight.BOLD, size=14),
            UIHelpers.make_reset_btn("Reset Params", self._reset_params),
            ft.Row([ft.Column(tf_list[::2],  spacing=8),
                    ft.Column(tf_list[1::2], spacing=8)],
                   spacing=8, vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=10)

        self._right_col = ft.Column(
            [enforced_rest_col, ft.Divider(), params_section], width=250)

        self._captains_col = ft.ListView(width=230, spacing=4) # You can adjust 200 to your liking
        self._rotation_col = ft.ListView(width=150, spacing=4)
        self._hours_col    = ft.ListView(width=150, spacing=4)

    # ── TextField on_change handlers (debounced) ───────────────────────────

    def _on_people_change(self, e):
        self.state.people_text = e.control.value
        self.state.invalidate_cache()
        # [LVL 3] debounce: only rebuild after 300 ms of inactivity
        self._debounce('_debounce_people', self._build_captains_list)

    def _on_tasks_change(self, e):
        self.state.tasks_text = e.control.value
        self.state.invalidate_cache()
        # [LVL 3]
        self._debounce('_debounce_tasks', self._build_rotation_list)

    def _on_days_change(self, e):
        self.state.days_text = e.control.value
        self.state.invalidate_cache()
        # [LVL 3]
        self._debounce('_debounce_days', self._build_hours_per_day)

    # ── Enforced-rest handlers ─────────────────────────────────────────────

    def _on_consec_global_change(self, e):
        val = e.control.value
        self.state.consec_global_val = val
        ok = UIHelpers.validate_positive_int(val)
        if ok:
            self.state.validation_errors["consec"].discard("_global")
            e.control.border_color = None
        else:
            self.state.validation_errors["consec"].add("_global")
            e.control.border_color = ft.Colors.RED_400
        if not self.state.consec_personalized:
            people, _, _, _ = self.state.dims()
            for p in people:
                self.state.consec_per_person[p] = val
            self._build_captains_list()
        self._update_err_consec()
        self._on_solve_blocked_update()
        e.control.update()

    def _on_personalize_toggle(self, e):
        self.state.consec_personalized = e.control.value
        if not e.control.value:
            people, _, _, _ = self.state.dims()
            gv = self.state.consec_global_val
            for p in people:
                self.state.consec_per_person[p] = gv
            for k in list(self.state.validation_errors["consec"]):
                if k != "_global":
                    self.state.validation_errors["consec"].discard(k)
            self._update_err_consec()
            self._on_solve_blocked_update()
        self._build_captains_list()

    def _update_err_consec(self):
        has_err = bool(self.state.validation_errors["consec"])
        self._err_consec.value   = "⚠ Max consec: only positive integers (≥ 1) allowed." if has_err else ""
        self._err_consec.visible = has_err
        try:
            self._err_consec.update()
        except Exception:
            pass

    # ── Params reset ───────────────────────────────────────────────────────

    def _reset_params(self, e):
        self.state.solver_params.clear()
        self.state.solver_params.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in self._param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS[k])
            tf.update()

    # ── Sub-list builders ──────────────────────────────────────────────────

    def _build_captains_list(self):
        s = self.state
        people, _, _, _ = s.dims()

        for p_old in list(s.consec_per_person.keys()):
            if p_old not in people:
                s.consec_per_person.pop(p_old, None)
                s.validation_errors["consec"].discard(f"person_{p_old}")
        self._update_err_consec()
        self._on_solve_blocked_update()

        is_pers = s.consec_personalized
        buf = [ft.Text("Cap / Rest", weight=ft.FontWeight.BOLD, size=12)]

        for p in people:
            s.captains_st.setdefault(p, 0)
            s.consec_per_person.setdefault(p, s.consec_global_val)

            cap_btn = ft.Container(
                content=ft.Text("Cap" if s.captains_st[p] else "—",
                                color=ft.Colors.BLACK, size=12, weight=ft.FontWeight.BOLD),
                width=48, height=28, data=p,
                bgcolor=ft.Colors.AMBER_400 if s.captains_st[p] else ft.Colors.GREY_400,
                alignment=ft.alignment.center, border_radius=4)
            def _cap_click(e, _p=p):
                s.captains_st[_p] = 1 - s.captains_st[_p]
                e.control.content.value = "Cap" if s.captains_st[_p] else "—"
                e.control.bgcolor = ft.Colors.AMBER_400 if s.captains_st[_p] else ft.Colors.GREY_400
                e.control.update()
            cap_btn.on_click = _cap_click

            cv       = s.consec_per_person.get(p, "")
            is_valid = UIHelpers.validate_positive_int(cv)
            tf_p = ft.TextField(
                value=cv, width=52, height=28, text_size=11,
                content_padding=ft.padding.all(2), disabled=not is_pers,
                border_color=ft.Colors.RED_400 if (not is_valid and is_pers) else None)
            def _ch_c(e, _p=p):
                s.consec_per_person[_p] = e.control.value
                if UIHelpers.validate_positive_int(e.control.value):
                    s.validation_errors["consec"].discard(f"person_{_p}")
                    e.control.border_color = None
                else:
                    s.validation_errors["consec"].add(f"person_{_p}")
                    e.control.border_color = ft.Colors.RED_400
                self._update_err_consec()
                self._on_solve_blocked_update()
                e.control.update()
            tf_p.on_change = _ch_c

            buf.append(ft.Row([ft.Text(p, size=11, width=100), cap_btn, tf_p], spacing=4))

        self._captains_col.controls = buf
        self.page.update()

    def _build_rotation_list(self):
        s = self.state
        _, tasks, _, _ = s.dims()
        buf = [ft.Text("Rotation", weight=ft.FontWeight.BOLD, size=12)]
        for t in tasks:
            s.rotation_st.setdefault(t, 1)
            btn = ft.Container(
                content=ft.Text("Rot" if s.rotation_st[t] else "—",
                                color=ft.Colors.BLACK, size=12, weight=ft.FontWeight.BOLD),
                width=55, height=28, data=t,
                bgcolor=ft.Colors.GREEN_400 if s.rotation_st[t] else ft.Colors.GREY_400,
                alignment=ft.alignment.center, border_radius=4)
            def _click(e, _t=t):
                s.rotation_st[_t] = 1 - s.rotation_st[_t]
                e.control.content.value = "Rot" if s.rotation_st[_t] else "—"
                e.control.bgcolor = ft.Colors.GREEN_400 if s.rotation_st[_t] else ft.Colors.GREY_400
                e.control.update()
            btn.on_click = _click
            buf.append(ft.Row([UIHelpers.lbl(t, 90), btn], spacing=4))
        self._rotation_col.controls = buf
        self.page.update()

    def _build_hours_per_day(self):
        s = self.state
        _, _, _, days = s.dims()
        buf = [ft.Text("Hours per Day", weight=ft.FontWeight.BOLD, size=12)]
        for j in days:
            s.hours_per_day.setdefault(j, DEFAULT_HOURS_TEXT)
            tf = ft.TextField(
                value=s.hours_per_day[j], multiline=True,
                min_lines=4, max_lines=24, label=j, width=120, data=j)
            def _ch(e, _j=j):
                s.hours_per_day[_j] = e.control.value
                s.invalidate_cache()
            tf.on_change = _ch
            buf.append(tf)
        self._hours_col.controls = buf
        self.page.update()

    # ── Public API ─────────────────────────────────────────────────────────

    def initial_build(self):
        self._build_captains_list()
        self._build_rotation_list()
        self._build_hours_per_day()

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[
                    self._right_col, self.tf_people, self._captains_col,
                    self.tf_tasks,   self._rotation_col,
                    self.tf_days,    self._hours_col,
                ],
                spacing=20,
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.ADAPTIVE), # <--- ADD THIS LINE
            padding=20, expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# AVAILABILITY TAB
# [EFFICIENCY LVL 2] cell_map for partial row updates — avoids full rebuild
#                    on row-toggle interactions
# ══════════════════════════════════════════════════════════════════════════════

class AvailabilityTab(BaseTab):

    def __init__(self, state: AppState, page: ft.Page):
        super().__init__(state, page)
        # [LVL 2] maps (person, hour, day) → ft.Container toggle widget
        # populated during build(), cleared on every full rebuild so refs stay fresh
        self._cell_map: dict = {}

    def build(self):
        s = self.state
        people, _, hours, days = s.dims()
        if not days:
            return
        if s.avail_filter not in days:
            s.avail_filter = days[0]

        # [LVL 2] Clear stale widget references before this rebuild
        self._cell_map.clear()

        def _on_day(v):
            s.avail_filter = v
            self.build()

        def _reset(e):
            s.avail_st.clear()
            self.build()

        def _rand(e):
            for p in people:
                for j in days:
                    for h in hours[j]:
                        s.avail_st[(p, h, j)] = random.choice([0, 1, 2])
            self.build()

        j       = s.avail_filter
        day_hrs = hours[j]
        nav     = UIHelpers.make_nav_dropdown("Select Day", j, days, _on_day, 200)

        buf = [ft.Row([
            nav,
            UIHelpers.make_reset_btn("Reset to Default", _reset),
            ft.Container(
                ft.Text("Random Avail (All Days)", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=12),
                bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                on_click=_rand, width=180, alignment=ft.alignment.center),
        ], spacing=20)]

        buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14))
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 4)] +
            [ft.Container(ft.Text(h, size=9, no_wrap=True, overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

        for p in people:
            # [LVL 2] Row toggle: update only the cells of this row via cell_map
            #         instead of calling self.build() (which would rebuild all 440+ widgets)
            def _row_toggle(_p=p, _j=j):
                def _click(e):
                    fv = s.avail_st.get((_p, hours[_j][0], _j), 1)
                    nv = UIHelpers._AVAIL_NEXT[fv]
                    for h in hours[_j]:
                        s.avail_st[(_p, h, _j)] = nv
                        cell = self._cell_map.get((_p, h, _j))
                        if cell is not None:
                            cell.content.value = UIHelpers._AVAIL_LBL[nv]
                            cell.bgcolor       = UIHelpers._AVAIL_CLR[nv]
                            cell.update()
                return ft.Container(
                    ft.Text("row", color=ft.Colors.WHITE, size=10),
                    width=UIHelpers.W_CELL, height=UIHelpers.H_BTN,
                    bgcolor=ft.Colors.BLUE_GREY_400,
                    alignment=ft.alignment.center, border_radius=4, on_click=_click)

            # Build cell toggles and register them in cell_map
            cell_widgets = []
            for h in day_hrs:
                toggle = UIHelpers.make_avail_toggle(s.avail_st, (p, h, j))
                self._cell_map[(p, h, j)] = toggle   # [LVL 2] store ref
                cell_widgets.append(toggle)

            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st), _row_toggle(p, j)] + cell_widgets,
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND TAB
# ══════════════════════════════════════════════════════════════════════════════

class DemandTab(BaseTab):

    def __init__(self, state: AppState, page: ft.Page, on_solve_blocked_update):
        super().__init__(state, page)
        self._on_solve_blocked_update = on_solve_blocked_update

    def build(self):
        s = self.state
        _, tasks, hours, days = s.dims()
        if not days:
            return
        if s.demand_filter not in days:
            s.demand_filter = days[0]

        def _on_day(v):
            s.demand_filter = v
            self.build()

        def _reset(e):
            s.demand_st.clear()
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _rand(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        s.demand_st[(t, h, j)] = str(random.randint(0, 4))
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _zero(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        s.demand_st[(t, h, j)] = ""
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        j       = s.demand_filter
        day_hrs = hours[j]
        nav     = UIHelpers.make_nav_dropdown("Select Day", j, days, _on_day, 200)
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=12, visible=False)

        buf = [ft.Row([
            nav,
            UIHelpers.make_reset_btn("Reset to Default", _reset),
            ft.Container(ft.Text("Set All to 0", color=ft.Colors.WHITE,
                                 weight=ft.FontWeight.BOLD, size=12),
                         bgcolor=ft.Colors.ORANGE_700, padding=10, border_radius=4,
                         on_click=_zero, width=150, alignment=ft.alignment.center),
            ft.Container(ft.Text("Random Demand (All Days)", color=ft.Colors.WHITE,
                                 weight=ft.FontWeight.BOLD, size=12),
                         bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                         on_click=_rand, width=190, alignment=ft.alignment.center),
        ], spacing=20)]

        buf += [ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14),
                err_txt, UIHelpers.hdr_row(day_hrs)]

        for t in tasks:
            cells = []
            for h in day_hrs:
                k = (t, h, j)
                s.demand_st.setdefault(k, "1")
                is_ok = UIHelpers.validate_nonneg_int(s.demand_st[k])
                tf = ft.TextField(
                    value=s.demand_st[k], width=UIHelpers.W_CELL, height=UIHelpers.H_TF,
                    text_size=11, data=k, content_padding=ft.padding.all(2),
                    border_color=ft.Colors.RED_400 if not is_ok else None)
                def _ch(e, _k=k, _err=err_txt):
                    s.demand_st[_k] = e.control.value
                    if UIHelpers.validate_nonneg_int(e.control.value):
                        s.validation_errors["demand"].discard(_k)
                        e.control.border_color = None
                    else:
                        s.validation_errors["demand"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    n = len(s.validation_errors["demand"])
                    _err.value   = f"⚠ {n} cell(s) have invalid values. Only non-negative integers allowed." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row([UIHelpers.lbl(t)] + cells, spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS TAB
# ══════════════════════════════════════════════════════════════════════════════

class SkillsTab(BaseTab):

    def build(self):
        s = self.state
        if not s.needs_rebuild("skills"):
            return
        people, tasks, _, _ = s.dims()

        def _reset(e):
            s.skills_st.clear()
            s.invalidate_cache()
            self.build()

        def _rand(e):
            for p in people:
                for t in tasks:
                    s.skills_st[(p, t)] = random.choice([0, 1])
            s.invalidate_cache()
            self.build()

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=16),
            ft.Row([
                UIHelpers.make_reset_btn("Reset Skills", _reset),
                ft.Container(ft.Text("Random Skills", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=12),
                             bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                             on_click=_rand, width=150, alignment=ft.alignment.center),
            ], spacing=10),
            UIHelpers.hdr_row(tasks),
        ]
        for p in people:
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st)] +
                [UIHelpers.make_toggle(s.skills_st, (p, t), 1) for t in tasks],
                spacing=2))

        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# QUOTA TAB
# ══════════════════════════════════════════════════════════════════════════════

class QuotaTab(BaseTab):

    def __init__(self, state: AppState, page: ft.Page, on_solve_blocked_update):
        super().__init__(state, page)
        self._on_solve_blocked_update = on_solve_blocked_update

    def build(self):
        s = self.state
        people, tasks, hours, days = s.dims()
        if not days:
            return
        if s.quota_filter not in days:
            s.quota_filter = days[0]

        def _on_day(v):
            s.quota_filter = v
            self.build()

        def _reset(e):
            s.quota_st.clear()
            s.validation_errors["quota"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _rand(e):
            for p in people:
                for t in tasks:
                    for j in days:
                        s.quota_st[(p, t, j)] = str(random.randint(0, 2))
            s.validation_errors["quota"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _zero(e):
            for p in people:
                for t in tasks:
                    for j in days:
                        s.quota_st[(p, t, j)] = ""
            s.validation_errors["quota"].clear()
            self._on_solve_blocked_update()
            self.build()

        j       = s.quota_filter
        nav     = UIHelpers.make_nav_dropdown("Select Day", j, days, _on_day, 200)
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=12, visible=False)

        buf = [
            ft.Text("Minimum Quota Matrix (per Day)", weight=ft.FontWeight.BOLD, size=16),
            ft.Row([
                nav,
                UIHelpers.make_reset_btn("Reset All Days", _reset),
                ft.Container(ft.Text("Set All to 0", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=12),
                             bgcolor=ft.Colors.ORANGE_700, padding=10, border_radius=4,
                             on_click=_zero, width=150, alignment=ft.alignment.center),
                ft.Container(ft.Text("Random Quota (All Days)", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=12),
                             bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                             on_click=_rand, width=180, alignment=ft.alignment.center),
            ], spacing=20),
            err_txt,
            ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=14),
            UIHelpers.hdr_row(tasks),
        ]

        for p in people:
            cells = []
            for t in tasks:
                k = (p, t, j)
                s.quota_st.setdefault(k, "")
                is_ok = UIHelpers.validate_nonneg_int(s.quota_st[k])
                tf = ft.TextField(
                    value=s.quota_st[k], width=UIHelpers.W_CELL, height=UIHelpers.H_TF,
                    text_size=11, data=k, content_padding=ft.padding.all(2),
                    border_color=ft.Colors.RED_400 if not is_ok else None)
                def _ch(e, _k=k, _err=err_txt):
                    s.quota_st[_k] = e.control.value
                    if UIHelpers.validate_nonneg_int(e.control.value):
                        s.validation_errors["quota"].discard(_k)
                        e.control.border_color = None
                    else:
                        s.validation_errors["quota"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    n = len(s.validation_errors["quota"])
                    _err.value   = f"⚠ {n} cell(s) have invalid values. Only non-negative integers allowed." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st)] + cells, spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# FORCE TAB
# ══════════════════════════════════════════════════════════════════════════════

class ForceTab(BaseTab):

    def build(self):
        s = self.state
        people, tasks, hours, days = s.dims()
        if not days or not tasks:
            return
        if s.force_filter[0] not in days:
            s.force_filter[0] = days[0]
        if s.force_filter[1] not in tasks:
            s.force_filter[1] = tasks[0]

        def _on_day(v):
            s.force_filter[0] = v
            self.build()

        def _on_task(v):
            s.force_filter[1] = v
            self.build()

        def _reset_all(e):
            s.force_st.clear()
            self.build()

        def _reset_cur(e):
            jj, tt = s.force_filter
            for p in people:
                for h in hours[jj]:
                    s.force_st.pop((p, tt, h, jj), None)
            self.build()

        j, t    = s.force_filter
        day_hrs = hours[j]
        ti      = tasks.index(t) if t in tasks else 0
        tbg, tfg = TASK_COLORS[ti % len(TASK_COLORS)]

        buf = [ft.Row([
            UIHelpers.make_nav_dropdown("Select Day",  j, days,  _on_day,  150),
            UIHelpers.make_nav_dropdown("Select Task", t, tasks, _on_task, 200),
            UIHelpers.make_reset_btn("Reset All",     _reset_all),
            UIHelpers.make_reset_btn("Reset Current", _reset_cur),
        ], spacing=20)]

        buf.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=14))
        buf.append(UIHelpers.hdr_row(day_hrs))
        for p in people:
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st)] +
                [UIHelpers.make_force_toggle(s.force_st, (p, t, h, j), 0, tbg, tfg)
                 for h in day_hrs],
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL TAB
# ══════════════════════════════════════════════════════════════════════════════

class SocialTab(BaseTab):

    def build(self):
        s = self.state
        if not s.needs_rebuild("social"):
            return
        people, _, _, _ = s.dims()

        buf = []
        if len(people) < 2:
            self._ct.controls = buf
            self.page.update()
            return

        def _reset(e):
            s.social_st.clear()
            s.hard_enemies = False
            s.invalidate_cache()
            self.build()

        def _rand(e):
            for i, p1 in enumerate(people):
                for p2 in people[i + 1:]:
                    s.social_st[(p1, p2)] = random.choice([-1, 0, 1])
            s.invalidate_cache()
            self.build()

        buf.append(ft.Row([
            UIHelpers.make_reset_btn("Reset to Default", _reset),
            ft.Container(ft.Text("Random Social", color=ft.Colors.WHITE,
                                 weight=ft.FontWeight.BOLD, size=12),
                         bgcolor=ft.Colors.PURPLE_400, padding=10, border_radius=4,
                         on_click=_rand, width=150, alignment=ft.alignment.center),
        ], spacing=10))

        sw_hard = ft.Switch(label="Enemies: Hard Constraint", value=s.hard_enemies)
        sw_hard.on_change = lambda e: setattr(s, "hard_enemies", e.control.value)
        buf.append(ft.Row([sw_hard]))
        buf.append(UIHelpers.hdr_row(people[1:]))

        _lbl  = {0: "~", 1: "+", -1: "-"}
        _clr  = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_700, -1: ft.Colors.RED_700}
        _next = {0: 1, 1: -1, -1: 0}

        for i, p1 in enumerate(people):
            cells = []
            for p2 in people[1:]:
                j2 = people.index(p2)
                if j2 > i:
                    k = (p1, p2)
                    s.social_st.setdefault(k, 0)
                    def _click(e, _k=k):
                        s.social_st[_k] = _next[s.social_st[_k]]
                        nv = s.social_st[_k]
                        e.control.content.value = _lbl[nv]
                        e.control.bgcolor = _clr[nv]
                        e.control.update()
                    btn = ft.Container(
                        content=ft.Text(_lbl[s.social_st[k]], color=ft.Colors.WHITE,
                                        size=12, weight=ft.FontWeight.BOLD),
                        width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=k,
                        bgcolor=_clr[s.social_st[k]],
                        alignment=ft.alignment.center, border_radius=4, on_click=_click)
                    cells.append(btn)
                else:
                    cells.append(ft.Container(width=UIHelpers.W_CELL))
            if cells:
                buf.append(ft.Row(
                    [UIHelpers.plbl(p1, s.captains_st)] + cells, spacing=2, wrap=False))

        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTS TAB
# ══════════════════════════════════════════════════════════════════════════════

class WeightsTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)

    def build(self):
        s = self.state

        def _reset(e):
            s.weights_order.clear()
            s.weights_order.extend(list(DEFAULT_WEIGHTS.keys()))
            for k in DEFAULT_WEIGHTS:
                s.weights_enabled[k] = True
                s.weights_st[k] = DEFAULT_WEIGHTS[k]
            self.build()

        for i, key in enumerate(s.weights_order):
            s.weights_st[key] = SORTED_VALUES[i] if s.weights_enabled[key] else 0

        header = ft.Column([
            ft.Row([
                ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=16),
                UIHelpers.make_reset_btn("Reset to Default", _reset),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=420),
            ft.Text("Drag items to reorder. Top items = Higher Cost.", italic=True, size=12),
            ft.Divider(),
        ], spacing=5)

        items = []

        def _handle_reorder(e):
            item = s.weights_order.pop(e.old_index)
            s.weights_order.insert(e.new_index, item)
            self.build()

        for i, key in enumerate(s.weights_order):
            val = SORTED_VALUES[i] if s.weights_enabled[key] else 0
            sw  = ft.Switch(value=s.weights_enabled[key], data=key)
            def _toggle(e, _k=key):
                s.weights_enabled[_k] = e.control.value
                self.build()
            sw.on_change = _toggle
            items.append(ft.Container(
                content=ft.Row([
                    ft.Text(f"#{i + 1}", width=30, weight=ft.FontWeight.BOLD),
                    ft.Text(key, expand=True, weight=ft.FontWeight.W_800),
                    ft.Text(f"{val}            ",
                            color=ft.Colors.BLACK if s.weights_enabled[key] else ft.Colors.GREY_500,
                            size=16),
                    sw,
                ], alignment=ft.MainAxisAlignment.START),
                padding=10,
                bgcolor=ft.Colors.LIGHT_BLUE_100 if s.weights_enabled[key] else ft.Colors.GREY_300,
                border=ft.border.all(1, ft.Colors.GREY_400), border_radius=8,
                margin=ft.margin.only(bottom=5)))

        layout = ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.ReorderableListView(controls=items, on_reorder=_handle_reorder),
                    width=420, height=500),
            ],
            width=420, alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        self._ct.controls = [ft.Row([layout], alignment=ft.MainAxisAlignment.CENTER)]
        self.page.update()

    def get_container(self) -> ft.Container:
        return ft.Container(self._ct, padding=10, expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT TAB
# ══════════════════════════════════════════════════════════════════════════════

class OutputTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Column(expand=True, spacing=5, scroll=ft.ScrollMode.ALWAYS)

    # ── Diff helpers ───────────────────────────────────────────────────────

    def _handle_diff_click(self, run_idx: int):
        ds = self.state.diff_state
        if run_idx == ds["ref"]:
            ds["ref"] = ds["cmp"]
            ds["cmp"] = None
        elif run_idx == ds["cmp"]:
            ds["cmp"] = None
        elif ds["ref"] is None:
            ds["ref"] = run_idx
        elif ds["cmp"] is None:
            ds["cmp"] = run_idx
        self.rebuild()
        self.page.update()

    @staticmethod
    def _diff_icon(task, ref_task):
        if task == ref_task:
            return None
        if ref_task is None:
            char, color = "+", DIFF_ADD_COLOR
        elif task is None:
            char, color = "−", DIFF_REMOVE_COLOR
        else:
            char, color = "⇄", DIFF_CHANGE_COLOR
        return ft.Container(
            content=ft.Text(char, size=8, color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
            width=13, height=13, bgcolor=color, border_radius=7,
            alignment=ft.alignment.center, right=1, top=1)

    # ── Single-run column ──────────────────────────────────────────────────

    def _build_run_column(self, run_idx, sol, people, tasks, hours, days,
                          availability, emergency, is_live=False, on_delete=None):
        s          = self.state
        status     = sol.get("status", "Solving...")
        is_solving = "Solving" in status
        assignment = sol.get("assignment", {})

        diff_mode = None
        if not is_live:
            if run_idx == s.diff_state["ref"]:
                diff_mode = "ref"
            elif run_idx == s.diff_state["cmp"]:
                diff_mode = "cmp"

        ref_sol = None
        if diff_mode == "cmp" and s.diff_state["ref"] is not None:
            ri = s.diff_state["ref"]
            if 0 <= ri < len(s.solution_history):
                ref_sol = s.solution_history[ri]["sol"]

        both_sel = s.diff_state["ref"] is not None and s.diff_state["cmp"] is not None
        buf      = []

        if is_live:
            d_lbl, d_bg, d_fg, d_click = "Comparar", ft.Colors.GREY_400, ft.Colors.GREY_600, None
        elif diff_mode == "ref":
            d_lbl, d_bg, d_fg = "Referencia", ft.Colors.LIGHT_BLUE_300, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)
        elif diff_mode == "cmp":
            d_lbl, d_bg, d_fg = "Comparado", ft.Colors.AMBER_300, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)
        elif both_sel:
            d_lbl, d_bg, d_fg, d_click = "Comparar", ft.Colors.GREY_400, ft.Colors.GREY_600, None
        else:
            d_lbl, d_bg, d_fg = "Comparar", ft.Colors.GREEN_200, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)

        diff_btn = ft.Container(
            content=ft.Text(d_lbl, size=11, color=d_fg,
                            weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=d_bg, padding=ft.padding.symmetric(4, 8),
            border_radius=4, on_click=d_click)

        hdr_lbl = (f"⟳  Run #{run_idx+1}  ·  Solving…"
                   if (is_live and is_solving)
                   else f"Run #{run_idx+1}  ·  {status}")
        hdr_bg  = "#E65100" if (is_live and is_solving) else "#1565C0"

        close_btn = (
            [ft.IconButton(
                icon=ft.Icons.CLOSE, icon_color=ft.Colors.WHITE, icon_size=16,
                tooltip="Remove this run",
                style=ft.ButtonStyle(padding=ft.padding.all(2)),
                on_click=lambda e, _cb=on_delete: _cb())]
            if on_delete is not None else [])

        buf.append(ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(hdr_lbl, weight=ft.FontWeight.BOLD, size=13,
                            color=ft.Colors.WHITE, no_wrap=True, expand=True),
                    diff_btn, *close_btn,
                ],
                spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=hdr_bg,
            padding=ft.padding.symmetric(8, 12), border_radius=6,
            margin=ft.margin.only(bottom=6)))

        if not (is_live and is_solving):
            buf.append(ft.Text(f"⏱  Solve time: {sol.get('solve_time', 0.0):.1f} s",
                               size=12, italic=True, color=ft.Colors.GREY_700))
            buf.append(ft.Text(f"◎  MIP Gap: {sol.get('mip_gap', 0.0) * 100:.3f} %",
                               size=12, italic=True, color=ft.Colors.GREY_700))

        CW = 40; NW = 50; TW = 25; Ch = 26

        if is_live and is_solving and not any(assignment.get(d) for d in days):
            buf.append(ft.Row([
                ft.ProgressRing(width=22, height=22, stroke_width=3),
                ft.Text("Solving in background…", italic=True, size=12),
            ], spacing=10))
        else:
            if is_live and is_solving:
                buf.append(ft.Row([
                    ft.ProgressRing(width=16, height=16, stroke_width=2),
                    ft.Text("Updating…", italic=True, size=11, color=ft.Colors.ORANGE_700),
                ], spacing=6))

            tc       = {t: TASK_COLORS[i % len(TASK_COLORS)] for i, t in enumerate(tasks)}
            ref_asgn = ref_sol.get("assignment", {}) if ref_sol else {}

            for j in days:
                if j not in assignment:
                    continue
                day_hrs = hours[j]
                buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=16))

                def _hc(txt, w):
                    return ft.Container(
                        ft.Text(txt, size=11, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                        width=w, height=Ch, bgcolor="#546E7A",
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#455A64"))

                buf.append(ft.Row(
                    [_hc("Person", NW)] + [_hc(h, CW) for h in day_hrs] + [_hc("Total", TW)],
                    spacing=0, wrap=False))

                asgn = assignment[j]
                for idx_p, p in enumerate(people):
                    is_cap     = s.captains_st.get(p, 0) == 1
                    row_bg     = "#ECEFF1" if idx_p % 2 == 0 else "#FFFFFF"
                    name_color = CAPTAIN_BG if is_cap else ft.Colors.BLACK
                    cells      = [ft.Container(
                        ft.Text(p, size=12, weight=ft.FontWeight.BOLD, color=name_color),
                        width=NW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center_left, padding=ft.padding.only(left=8),
                        border=ft.border.all(1, "#CFD8DC"))]
                    total = 0
                    for h in day_hrs:
                        task  = asgn.get(p, {}).get(h)
                        avail = availability.get((p, h, j), 1)
                        griev = emergency.get((p, h, j), 0)
                        if avail == 0:   brd = ft.border.all(1.5, UNAVAIL_COLOR)
                        elif griev == 1: brd = ft.border.all(1.5, EMERG_COLOR)
                        else:            brd = ft.border.all(0.5, AVAIL_COLOR)
                        if task:
                            bg, fg = tc[task]
                            total += 1
                            cell = ft.Container(
                                ft.Text(task, size=11, weight=ft.FontWeight.BOLD, color=fg,
                                        text_align=ft.TextAlign.CENTER),
                                width=CW, height=Ch, bgcolor=bg,
                                alignment=ft.alignment.center, border=brd, border_radius=4)
                        else:
                            cell = ft.Container(
                                width=CW, height=Ch, bgcolor=row_bg, border=brd)
                        if diff_mode == "cmp" and ref_sol is not None:
                            ref_task = ref_asgn.get(j, {}).get(p, {}).get(h)
                            icon = self._diff_icon(task, ref_task)
                            if icon is not None:
                                cell = ft.Stack(controls=[cell, icon], width=CW, height=Ch)
                        cells.append(cell)
                    cells.append(ft.Container(
                        ft.Text(str(int(total)), size=12, weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER),
                        width=TW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center, border=ft.border.all(1, "#CFD8DC")))
                    buf.append(ft.Row(cells, spacing=0, wrap=False))

                legend = [
                    ft.Container(
                        ft.Text(t, size=10, weight=ft.FontWeight.BOLD, color=tc[t][1]),
                        bgcolor=tc[t][0], padding=ft.padding.symmetric(6, 10), border_radius=4)
                    for t in tasks]
                for lbl_txt, clr in [("Available", AVAIL_COLOR),
                                     ("Emergency", EMERG_COLOR),
                                     ("Unavailable", UNAVAIL_COLOR)]:
                    legend.append(ft.Container(
                        ft.Text(lbl_txt, size=10, color=ft.Colors.WHITE),
                        bgcolor=clr, padding=ft.padding.symmetric(6, 10), border_radius=4))
                if diff_mode == "cmp":
                    for char, color, desc in [
                        ("+", DIFF_ADD_COLOR,    "Nueva asignación"),
                        ("−", DIFF_REMOVE_COLOR, "Eliminada"),
                        ("⇄", DIFF_CHANGE_COLOR, "Cambiada"),
                    ]:
                        legend.append(ft.Row([
                            ft.Container(
                                ft.Text(char, size=8, color=ft.Colors.WHITE,
                                        weight=ft.FontWeight.BOLD),
                                width=13, height=13, bgcolor=color,
                                border_radius=7, alignment=ft.alignment.center),
                            ft.Text(desc, size=10),
                        ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER))
                buf.append(ft.Row(legend, spacing=8, wrap=True))
                buf.append(ft.Divider())

            if not (is_live and is_solving):
                for section_title, issues, empty_msg in [
                    ("MISSING STAFF",
                     sol.get("missing", []),         "None -- all demand covered!"),
                    ("SHIFT SEGMENTS / FRAGMENTATION",
                     sol.get("gaps", []),             "Single block shifts! (Perfect)"),
                    ("ROTATION FATIGUE",
                     sol.get("rotation_issues", []),  "No rotation tasks defined."),
                    ("SOCIAL",
                     sol.get("social_issues", []),    "None -- all respected!"),
                    ("CAPTAIN PRESENCE",
                     sol.get("captain_issues", []),   "No captains designated or all hours covered."),
                    ("EMERGENCY CALL-INS",
                     sol.get("emerg_issues", []),     "None -- no emergency hours used!"),
                    ("QUOTA FULFILMENT",
                     sol.get("quota_issues", []),     "No quotas defined."),
                    ("FORCE MANDATES",
                     sol.get("force_issues", []),     "No force mandates defined."),
                ]:
                    
                    if section_title == "SHIFT SEGMENTS / FRAGMENTATION" and sol.get("enforced_rest", False):
                        continue
                    buf.append(ft.Text(section_title, weight=ft.FontWeight.BOLD, size=14))
                    if issues:
                        for line in issues:
                            buf.append(ft.Text(f"  {line}", size=12))
                    else:
                        buf.append(ft.Text(f"  {empty_msg}", size=12, italic=True))

                workload     = sol.get("workload", {})
                ref_workload = ref_sol.get("workload", {}) if ref_sol else {}
                buf.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=14))
                for p in people:
                    ch = workload.get(p, 0)
                    if diff_mode == "cmp" and ref_sol is not None:
                        ph = ref_workload.get(p, 0)
                        if ch != ph:
                            buf.append(ft.Text(
                                f"  {p}: {ch:.0f} h  ({ph:.0f} prev)",
                                size=12, color=ft.Colors.ORANGE_700,
                                weight=ft.FontWeight.BOLD))
                        else:
                            buf.append(ft.Text(f"  {p}: {ch:.0f} hours", size=12))
                    else:
                        buf.append(ft.Text(f"  {p}: {ch:.0f} hours", size=12))
                buf.append(ft.Text(
                    f"  Global range: "
                    f"max={sol.get('workload_max', 0):.0f}, "
                    f"min={sol.get('workload_min', 0):.0f}",
                    size=12, italic=True))

        max_hrs   = max((len(hours[d]) for d in days), default=0)
        col_width = max(NW + CW * max_hrs + TW + 30, 460)
        return ft.Container(
            content=ft.Column(controls=buf, spacing=5, tight=True),
            width=col_width,
            padding=ft.padding.symmetric(horizontal=16),
            border=ft.border.only(right=ft.border.BorderSide(1, "#B0BEC5")))

    # ── Full rebuild ───────────────────────────────────────────────────────

    def rebuild(self, live_sol=None, live_people=None, live_tasks=None,
                live_hours=None, live_days=None,
                live_avail=None, live_emerg=None):
        s    = self.state
        cols = []

        for i, entry in enumerate(s.solution_history):
            def _make_delete(idx):
                def _delete():
                    ds = s.diff_state
                    if ds["ref"] == idx:
                        ds["ref"] = ds["cmp"]
                        ds["cmp"] = None
                    elif ds["cmp"] == idx:
                        ds["cmp"] = None
                    for key in ("ref", "cmp"):
                        if ds[key] is not None and ds[key] > idx:
                            ds[key] -= 1
                    s.solution_history.pop(idx)
                    self.rebuild()
                    self.page.update()
                return _delete

            cols.append(self._build_run_column(
                i, entry["sol"], entry["people"], entry["tasks"],
                entry["hours"], entry["days"],
                entry["availability"], entry["emergency"],
                on_delete=_make_delete(i)))

        cols.reverse()

        if live_sol is not None:
            cols.insert(0, self._build_run_column(
                len(s.solution_history), live_sol,
                live_people, live_tasks, live_hours, live_days,
                live_avail, live_emerg, is_live=True))

        self._ct.controls = (
            [ft.Row(controls=cols, spacing=0, wrap=False,
                    vertical_alignment=ft.CrossAxisAlignment.START)]
            if cols else [])

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[self._ct],
                scroll=ft.ScrollMode.ALWAYS,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START),
            padding=10, expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class SolverController:

    def __init__(self, state: AppState, page: ft.Page,
                 output_tab: OutputTab,
                 on_solve_blocked_update,
                 switch_page_cb,
                 ui_lock: threading.Lock,
                 day_heuristics_sw: ft.Switch):
        self.state    = state
        self.page     = page
        self._out     = output_tab
        self._upd     = on_solve_blocked_update
        self._switch  = switch_page_cb
        self._lock    = ui_lock
        self._sw_heur = day_heuristics_sw

    def do_solve(self, e):
        s = self.state
        if s.running_model_ref[0] is not None:
            s.running_model_ref[0].terminate()
        if s.solve_blocked:
            return

        people, tasks, hours, days = s.dims()

        max_consec: dict = {}
        if s.consec_personalized:
            for p in people:
                raw = s.consec_per_person.get(p, "").strip()
                if raw:
                    try:
                        max_consec[p] = int(raw)
                    except ValueError:
                        pass
        else:
            raw = s.consec_global_val.strip()
            if raw:
                try:
                    n = int(raw)
                    for p in people:
                        max_consec[p] = n
                except ValueError:
                    pass

        availability: dict = {}
        emergency   : dict = {}
        for p in people:
            for j in days:
                for h in hours[j]:
                    v = s.avail_st.get((p, h, j), 1)
                    availability[(p, h, j)] = 1 if v in (1, 2) else 0
                    emergency   [(p, h, j)] = 1 if v == 2      else 0

        demand: dict = {}
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    raw = s.demand_st.get((t, h, j), "1").strip()
                    try:    demand[(t, h, j)] = int(raw) if raw else 0
                    except: demand[(t, h, j)] = 0

        mq: dict = {}
        for p in people:
            for t in tasks:
                for j in days:
                    raw = s.quota_st.get((p, t, j), "").strip()
                    try:    mq[(p, t, j)] = int(raw) if raw else 0
                    except: mq[(p, t, j)] = 0

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, emergency=emergency,
            demand=demand,
            skills   = {(p, t): s.skills_st.get((p, t), 1) for p in people for t in tasks},
            force    = {(p, t, h, j): s.force_st.get((p, t, h, j), 0)
                        for p in people for t in tasks for j in days for h in hours[j]},
            social   = {(p1, p2): s.social_st.get((p1, p2), 0)
                        for i, p1 in enumerate(people) for p2 in people[i + 1:]},
            min_quota=mq,
            pref_cost={(p, t): 1 for p in people for t in tasks},
            rotation ={t: s.rotation_st.get(t, 1) for t in tasks},
            X_prev   ={(p, t, h, j): 0
                       for p in people for t in tasks for j in days for h in hours[j]},
            weights          = s.weights_st.copy(),
            max_consec_hours = max_consec,
            captains         = [p for p in people if s.captains_st.get(p, 0) == 1],
            solver_params    = s.solver_params,
            hard_enemies     = s.hard_enemies,
            day_heuristics   = 1 if self._sw_heur.value else 0,
        )

        self._out.rebuild(
            live_sol={"status": "Solving...", "assignment": {}},
            live_people=people, live_tasks=tasks, live_hours=hours,
            live_days=days, live_avail=availability, live_emerg=emergency)
        self._switch(8)

        def _update_ui(partial_sol):
            if not self._lock.acquire(blocking=False):
                return
            try:
                self._out.rebuild(
                    live_sol=partial_sol,
                    live_people=people, live_tasks=tasks, live_hours=hours,
                    live_days=days, live_avail=availability, live_emerg=emergency)
                self.page.update()
            except Exception:
                pass
            finally:
                self._lock.release()

        def _run():
            try:
                final = solve_model(
                    data,
                    ui_update_callback=_update_ui,
                    active_model_ref=s.running_model_ref)
                s.solution_history.append({
                    "sol":          final,
                    "people":       people,
                    "tasks":        tasks,
                    "hours":        hours,
                    "days":         days,
                    "availability": availability,
                    "emergency":    emergency,
                })
                self._out.rebuild()
            except Exception as ex:
                self._out._ct.controls = [
                    ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=14)]
            finally:
                s.solver_running = False
                self._upd()
            self.page.update()

        s.solver_running = True
        self._upd()
        threading.Thread(target=_run, daemon=True).start()

    def do_stop(self, e):
        if self.state.running_model_ref[0] is not None:
            self.state.running_model_ref[0].terminate()


# ══════════════════════════════════════════════════════════════════════════════
# STAFF SCHEDULER APP
# ══════════════════════════════════════════════════════════════════════════════

class StaffSchedulerApp:

    def __init__(self, page: ft.Page):
        self.page       = page
        self.state      = AppState()
        self._ui_lock   = threading.Lock()
        self._solve_btn = None   # guard: created later; early callbacks check for None
        self._configure_page()
        self._build_ui()

    def _configure_page(self):
        p = self.page
        p.title         = "Staff Scheduler"
        p.scroll        = None
        p.window.maximized = True
        p.theme_mode    = ft.ThemeMode.LIGHT
        p.padding       = 0

    def _update_solve_blocked(self):
        s       = self.state
        has_err = bool(s.validation_errors["demand"] or
                       s.validation_errors["quota"]  or
                       s.validation_errors["consec"])
        s.solve_blocked = has_err
        if self._solve_btn is None:   # guard: not yet created
            return
        blocked = has_err or s.solver_running
        self._solve_btn.bgcolor  = ft.Colors.GREY_500 if blocked else "#1565C0"
        self._solve_btn.disabled = blocked
        try:
            self._solve_btn.update()
        except Exception:
            pass

    def _build_ui(self):
        s = self.state

        # Tabs — DimensionsTab does NOT call initial_build in __init__,
        # so the callback fires only after self._solve_btn exists below.
        self._dims_tab    = DimensionsTab(s, self.page, self._update_solve_blocked)
        self._avail_tab   = AvailabilityTab(s, self.page)
        self._demand_tab  = DemandTab(s, self.page, self._update_solve_blocked)
        self._skills_tab  = SkillsTab(s, self.page)
        self._quota_tab   = QuotaTab(s, self.page, self._update_solve_blocked)
        self._force_tab   = ForceTab(s, self.page)
        self._social_tab  = SocialTab(s, self.page)
        self._weights_tab = WeightsTab(s, self.page)
        self._output_tab  = OutputTab(s, self.page)

        # Sidebar action widgets
        self._solve_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE, size=18),
                ft.Text("SOLVE", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=13),
            ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#1565C0",
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            border_radius=8, on_click=self._do_solve,
            width=SIDEBAR_WIDTH - 32, alignment=ft.alignment.center)

        self._stop_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE, size=18),
                ft.Text("STOP", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=13),
            ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#C62828",
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            border_radius=8, on_click=self._do_stop,
            width=SIDEBAR_WIDTH - 32, alignment=ft.alignment.center)

        self._day_heuristics_sw = ft.Switch(
            label="Day Heuristics", value=False,
            tooltip=(
                "Desacopla los días: resuelve un submodelo por día usando "
                "una heurística de ritmo (pacing). "
                "Reduce drásticamente el tiempo de resolución con muchos días, "
                "personas y tareas, a costa de perder la equidad global exacta."
            ),
            label_style=ft.TextStyle(color=ft.Colors.WHITE, size=11))

        self._solver = SolverController(
            state=s, page=self.page,
            output_tab=self._output_tab,
            on_solve_blocked_update=self._update_solve_blocked,
            switch_page_cb=self._switch_page,
            ui_lock=self._ui_lock,
            day_heuristics_sw=self._day_heuristics_sw)

        self._page_contents = {
            0: self._dims_tab.get_container(),
            1: self._avail_tab.get_container(),
            2: self._demand_tab.get_container(),
            3: self._skills_tab.get_container(),
            4: self._quota_tab.get_container(),
            5: self._force_tab.get_container(),
            6: self._social_tab.get_container(),
            7: self._weights_tab.get_container(),
            8: self._output_tab.get_container(),
        }
        self._builders = {
            1: self._avail_tab.build,
            2: self._demand_tab.build,
            3: self._skills_tab.build,
            4: self._quota_tab.build,
            5: self._force_tab.build,
            6: self._social_tab.build,
            7: self._weights_tab.build,
        }

        # Sidebar menu buttons
        _menu_def = [
            ("Dimensions",   ft.Icons.GRID_VIEW,           0),
            ("Availability", ft.Icons.EVENT_AVAILABLE,      1),
            ("Demand",       ft.Icons.TRENDING_UP,          2),
            ("Skills",       ft.Icons.STAR_BORDER,          3),
            ("Quota",        ft.Icons.FORMAT_LIST_NUMBERED, 4),
            ("Force",        ft.Icons.LOCK_OUTLINE,         5),
            ("Social",       ft.Icons.PEOPLE_OUTLINE,       6),
            ("Weights",      ft.Icons.TUNE,                 7),
            ("Output",       ft.Icons.ASSESSMENT,           8),
        ]
        self._selected_idx  = 0
        self._menu_btn_refs = []
        self._content_area  = ft.Container(
            expand=True, padding=0, content=self._page_contents[0])

        for label, icon, idx in _menu_def:
            sel = (idx == 0)
            btn = ft.Container(
                content=ft.Row([
                    ft.Icon(icon,
                            color=SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR,
                            size=20),
                    ft.Text(label, size=13,
                            color=SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR,
                            weight=ft.FontWeight.BOLD if sel else None),
                ], spacing=10),
                padding=ft.padding.symmetric(horizontal=16, vertical=12),
                border_radius=8,
                bgcolor=SIDEBAR_SELECTED_BG if sel else None,
                on_click=self._on_menu_click, data=idx, ink=True)
            self._menu_btn_refs.append(btn)

        sidebar = ft.Container(
            width=SIDEBAR_WIDTH, bgcolor=SIDEBAR_BG,
            padding=ft.padding.only(top=12, bottom=12, left=8, right=8),
            content=ft.Column(
                controls=[
                    ft.Container(
                        ft.Text("Staff Scheduler", size=16,
                                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        padding=ft.padding.only(left=8, bottom=4)),
                    ft.Container(
                        ft.Column([
                            self._solve_btn, self._day_heuristics_sw, self._stop_btn,
                        ], spacing=6),
                        padding=ft.padding.only(bottom=8)),
                    ft.Divider(color="#455A64", height=1),
                    ft.Column(controls=self._menu_btn_refs, spacing=2,
                              scroll=ft.ScrollMode.AUTO, expand=True),
                ],
                spacing=4, expand=True),
            border=ft.border.only(right=ft.border.BorderSide(1, "#455A64")))

        self.page.add(ft.Row(
            controls=[sidebar, self._content_area],
            spacing=0, expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START))

        # Initial population — called AFTER self._solve_btn exists
        self._dims_tab.initial_build()

    def _on_menu_click(self, e):
        self._switch_page(e.control.data)

    def _switch_page(self, idx: int):
        self._selected_idx = idx
        self._content_area.content = self._page_contents[idx]
        for i, btn in enumerate(self._menu_btn_refs):
            sel = (i == idx)
            btn.bgcolor                    = SIDEBAR_SELECTED_BG if sel else None
            btn.content.controls[0].color  = SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR
            btn.content.controls[1].color  = SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR
            btn.content.controls[1].weight = ft.FontWeight.BOLD if sel else None
        if idx in self._builders:
            with self._ui_lock:
                self._builders[idx]()
        self.page.update()

    def _do_solve(self, e):
        self._solver.do_solve(e)

    def _do_stop(self, e):
        self._solver.do_stop(e)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(page: ft.Page):
    StaffSchedulerApp(page)


ft.app(target=main)