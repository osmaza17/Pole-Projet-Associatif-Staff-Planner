


import flet as ft
import random
import threading

from solve_model_pace_10 import solve_model


# ══════════════════════════════════════════════════════════════════════════════
# SCALE FACTOR
# ══════════════════════════════════════════════════════════════════════════════

SCALE = 0.75

def _s(v: float) -> int:
    return max(1, round(v * SCALE))


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "W_COVERAGE": 300000, "W_FORCE": 100000, "W_CAPTAIN": 10000,
    "W_STABILITY": 7000, "W_EQ_DAY": 5000, "W_GAP": 1000,
    "W_EMERG": 750,      "W_EQ_GLOBAL": 500, "W_ROTATION": 100,
    "W_SOCIAL": 50,      "W_QUOTA": 10,      "W_PREF": 1,
}
SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

DEFAULT_SOLVER_PARAMS = {
    "TimeLimit": 1200, "MIPGap": 0.01, "MIPFocus": 2,
    "Threads": 0, "Presolve": 2, "Symmetry": 2,
    "Disconnected": 2, "IntegralityFocus": 1, "Method": 3,
    "Cuts": -1,
    "Heuristics": 0.05,
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
CAPTAIN_FG        = "#FFFFFF"
UNAVAIL_COLOR     = "#D32F2F"
EMERG_COLOR       = "#F57C00"
AVAIL_COLOR       = "#388E3C"
DIFF_ADD_COLOR    = "#2E7D32"
DIFF_REMOVE_COLOR = "#C62828"
DIFF_CHANGE_COLOR = "#E65100"

SIDEBAR_WIDTH         = _s(200)
SIDEBAR_BG            = "#263238"
SIDEBAR_SELECTED_BG   = "#37474F"
SIDEBAR_TEXT_COLOR    = "#ECEFF1"
SIDEBAR_SELECTED_TEXT = "#4FC3F7"

BASE_ACTIVE_BG = "#2E7D32"
BASE_ACTIVE_FG = "#FFFFFF"
BASE_IDLE_BG   = "#A5D6A7"
BASE_IDLE_FG   = "#000000"


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE
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
        self.base_run_idx: int | None = None

        self.weights_st      = DEFAULT_WEIGHTS.copy()
        self.weights_order   = list(DEFAULT_WEIGHTS.keys())
        self.weights_enabled = {k: True for k in DEFAULT_WEIGHTS}
        self.solver_params   = DEFAULT_SOLVER_PARAMS.copy()

        self._build_cache: dict = {}

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

    def _dims_hash(self) -> int:
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

    def build_x_prev(self, new_people, new_tasks, new_hours, new_days) -> dict:
        X_prev = {
            (p, t, h, j): 0
            for p in new_people for t in new_tasks
            for j in new_days for h in new_hours[j]
        }
        if self.base_run_idx is None:
            return X_prev
        if not (0 <= self.base_run_idx < len(self.solution_history)):
            return X_prev
        entry         = self.solution_history[self.base_run_idx]
        base_asgn     = entry["sol"]["assignment"]
        base_people_s = set(entry["people"])
        base_tasks_s  = set(entry["tasks"])
        for p in new_people:
            if p not in base_people_s:
                continue
            for j in new_days:
                if j not in base_asgn:
                    continue
                day_asgn = base_asgn[j]
                for h in new_hours[j]:
                    assigned_task = day_asgn.get(p, {}).get(h)
                    for t in new_tasks:
                        if t not in base_tasks_s:
                            continue
                        X_prev[(p, t, h, j)] = 1 if assigned_task == t else 0
        return X_prev


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class UIHelpers:
    W_LBL  = _s(80)
    W_CELL = _s(50)
    H_BTN  = _s(26)
    H_TF   = _s(30)

    _AVAIL_LBL  = {1: "1",  0: "0",  2: "!"}
    _AVAIL_CLR  = {1: ft.Colors.GREEN_700, 0: ft.Colors.RED_700, 2: ft.Colors.ORANGE_700}
    _AVAIL_NEXT = {1: 0, 0: 2, 2: 1}

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

    @staticmethod
    def make_reset_btn(text: str, on_click) -> ft.Container:
        return ft.Container(
            content=ft.Text(text, color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
            bgcolor=ft.Colors.RED_500, padding=_s(8), border_radius=4,
            on_click=on_click, width=_s(150), alignment=ft.alignment.center)

    @staticmethod
    def lbl(text: str, w: int = None) -> ft.Container:
        return ft.Container(
            ft.Text(text, size=_s(11), no_wrap=True),
            width=w or UIHelpers.W_LBL)

    @staticmethod
    def plbl(name: str, captains_st: dict, w: int = None) -> ft.Container:
        is_cap = captains_st.get(name, 0) == 1
        return ft.Container(
            ft.Text(name, size=_s(11), no_wrap=True,
                    weight=ft.FontWeight.BOLD if is_cap else None,
                    color=CAPTAIN_BG if is_cap else None),
            width=w or UIHelpers.W_LBL)

    @staticmethod
    def hdr_row(labels: list, w: int = None) -> ft.Row:
        w = w or UIHelpers.W_CELL
        return ft.Row(
            [ft.Container(width=UIHelpers.W_LBL)] +
            [ft.Container(
                ft.Text(lbl, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=w, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for lbl in labels],
            spacing=2, wrap=False)

    # ── Generic row/col toggle button ─────────────────────────────────────

    @staticmethod
    def make_rc_btn(label: str = "·") -> ft.Container:
        """Bare blue-grey button shell; caller sets on_click."""
        return ft.Container(
            ft.Text(label, color=ft.Colors.WHITE, size=_s(10),
                    text_align=ft.TextAlign.CENTER),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN,
            bgcolor=ft.Colors.BLUE_GREY_400,
            alignment=ft.alignment.center, border_radius=4)

    @staticmethod
    def make_inc_btn(delta: int) -> ft.Container:
        """Green (+1) or red (-1) increment button shell; caller sets on_click."""
        label = "+1" if delta > 0 else "-1"
        color = ft.Colors.GREEN_700 if delta > 0 else ft.Colors.RED_700
        return ft.Container(
            ft.Text(label, color=ft.Colors.WHITE, size=_s(10),
                    text_align=ft.TextAlign.CENTER),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN,
            bgcolor=color, alignment=ft.alignment.center, border_radius=4)

    @staticmethod
    def make_nav_dropdown(label: str, value, options: list,
                          on_change, width: int = 200) -> ft.Row:
        dd = ft.Dropdown(
            label=label, value=value,
            options=[ft.dropdown.Option(o, text_style=ft.TextStyle(size=_s(11))) for o in options],
            width=_s(width),
            text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            dense=True,
            content_padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)))

        def _nav(direction):
            idx = options.index(dd.value) if dd.value in options else 0
            dd.value = options[(idx + direction) % len(options)]
            dd.update()
            on_change(dd.value)

        dd.on_change = lambda e: on_change(e.control.value)

        def _btn(icon, direction):
            return ft.IconButton(
                icon=icon, icon_size=_s(18),
                tooltip="Previous" if direction == -1 else "Next",
                style=ft.ButtonStyle(padding=ft.padding.all(_s(4)),
                                     shape=ft.RoundedRectangleBorder(radius=6)),
                on_click=lambda e, d=direction: _nav(d))

        return ft.Row(
            [_btn(ft.Icons.CHEVRON_LEFT, -1), dd, _btn(ft.Icons.CHEVRON_RIGHT, 1)],
            spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)

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
                            size=_s(11), weight=ft.FontWeight.BOLD),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=key,
            bgcolor=ft.Colors.GREEN_700 if sd[key] else ft.Colors.RED_700,
            alignment=ft.alignment.center, border_radius=4, on_click=_click)

    @staticmethod
    def make_avail_toggle(sd: dict, key, default: int = 1) -> ft.Container:
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
                            size=_s(11), weight=ft.FontWeight.BOLD),
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
                            size=_s(11), weight=ft.FontWeight.BOLD),
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
        self._ct   = ft.Column(expand=True, spacing=_s(5), scroll=ft.ScrollMode.ALWAYS)

    def build(self):
        raise NotImplementedError

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[self._ct],
                scroll=ft.ScrollMode.ALWAYS,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START),
            padding=_s(10), expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSIONS TAB
# ══════════════════════════════════════════════════════════════════════════════

class DimensionsTab:

    def __init__(self, state: AppState, page: ft.Page, on_solve_blocked_update):
        self.state = state
        self.page  = page
        self._on_solve_blocked_update = on_solve_blocked_update

        self._debounce_people: threading.Timer | None = None
        self._debounce_tasks : threading.Timer | None = None
        self._debounce_days  : threading.Timer | None = None

        self._build_ui()

    def _debounce(self, attr: str, fn, delay: float = 0.3):
        old: threading.Timer | None = getattr(self, attr, None)
        if old is not None:
            old.cancel()
        t = threading.Timer(delay, fn)
        t.daemon = True
        t.start()
        setattr(self, attr, t)

    def _build_ui(self):
        s = self.state

        self.tf_people = ft.TextField(
            value=s.people_text, multiline=True, min_lines=1, max_lines=200,
            label="People (one per line)", width=_s(140), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_people_change)
        self.tf_tasks = ft.TextField(
            value=s.tasks_text, multiline=True, min_lines=8, max_lines=200,
            label="Tasks (one per line)", width=_s(180), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_tasks_change)
        self.tf_days = ft.TextField(
            value=s.days_text, multiline=True, min_lines=1, max_lines=200,
            label="Days (one per line)", width=_s(120), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_days_change)

        self._err_consec = ft.Text("", color=ft.Colors.RED_400, size=_s(11), visible=False)

        self._tf_consec_global = ft.TextField(
            value="", width=_s(110), height=_s(38), text_size=_s(12),
            hint_text="Max consec", content_padding=ft.padding.all(_s(4)),
            on_change=self._on_consec_global_change)

        self._sw_personalize = ft.Switch(
            label="", value=False,
            on_change=self._on_personalize_toggle)

        enforced_rest_col = ft.Column([
            ft.Row([self._sw_personalize, self._tf_consec_global],
                   spacing=_s(8), vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self._err_consec,
        ], spacing=_s(5))

        self._param_tfs: dict = {}
        for key, val in s.solver_params.items():
            tf = ft.TextField(label=key, value=str(val),
                              width=_s(120), height=_s(45), text_size=_s(12),
                              label_style=ft.TextStyle(size=_s(11)))
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
            ft.Text("Gurobi Parameters", weight=ft.FontWeight.BOLD, size=_s(14)),
            UIHelpers.make_reset_btn("Reset Params", self._reset_params),
            ft.Row([ft.Column(tf_list[::2],  spacing=_s(8)),
                    ft.Column(tf_list[1::2], spacing=_s(8))],
                   spacing=_s(8), vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=_s(10))

        self._right_col = ft.Column(
            [enforced_rest_col, ft.Divider(), params_section], width=_s(250))

        self._captains_col = ft.ListView(width=_s(230), spacing=_s(4))
        self._rotation_col = ft.ListView(width=_s(150), spacing=_s(4))
        self._hours_col    = ft.ListView(width=_s(150), spacing=_s(4))

    def _on_people_change(self, e):
        self.state.people_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_people', self._build_captains_list)

    def _on_tasks_change(self, e):
        self.state.tasks_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_tasks', self._build_rotation_list)

    def _on_days_change(self, e):
        self.state.days_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_days', self._build_hours_per_day)

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

    def _reset_params(self, e):
        self.state.solver_params.clear()
        self.state.solver_params.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in self._param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS[k])
            tf.update()

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
        buf = [ft.Text("Cap / Rest", weight=ft.FontWeight.BOLD, size=_s(12))]

        for p in people:
            s.captains_st.setdefault(p, 0)
            s.consec_per_person.setdefault(p, s.consec_global_val)

            cap_btn = ft.Container(
                content=ft.Text("Cap" if s.captains_st[p] else "—",
                                color=ft.Colors.BLACK, size=_s(12), weight=ft.FontWeight.BOLD),
                width=_s(48), height=_s(28), data=p,
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
                value=cv, width=_s(52), height=_s(28), text_size=_s(11),
                content_padding=ft.padding.all(_s(2)), disabled=not is_pers,
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

            buf.append(ft.Row([ft.Text(p, size=_s(11), width=_s(100)), cap_btn, tf_p],
                              spacing=_s(4)))

        self._captains_col.controls = buf
        self.page.update()

    def _build_rotation_list(self):
        s = self.state
        _, tasks, _, _ = s.dims()
        buf = [ft.Text("Rotation", weight=ft.FontWeight.BOLD, size=_s(12))]
        for t in tasks:
            s.rotation_st.setdefault(t, 1)
            btn = ft.Container(
                content=ft.Text("Rot" if s.rotation_st[t] else "—",
                                color=ft.Colors.BLACK, size=_s(12), weight=ft.FontWeight.BOLD),
                width=_s(55), height=_s(28), data=t,
                bgcolor=ft.Colors.GREEN_400 if s.rotation_st[t] else ft.Colors.GREY_400,
                alignment=ft.alignment.center, border_radius=4)
            def _click(e, _t=t):
                s.rotation_st[_t] = 1 - s.rotation_st[_t]
                e.control.content.value = "Rot" if s.rotation_st[_t] else "—"
                e.control.bgcolor = ft.Colors.GREEN_400 if s.rotation_st[_t] else ft.Colors.GREY_400
                e.control.update()
            btn.on_click = _click
            buf.append(ft.Row([UIHelpers.lbl(t, _s(90)), btn], spacing=_s(4)))
        self._rotation_col.controls = buf
        self.page.update()

    def _build_hours_per_day(self):
        s = self.state
        _, _, _, days = s.dims()
        buf = [ft.Text("Hours per Day", weight=ft.FontWeight.BOLD, size=_s(12))]
        for j in days:
            s.hours_per_day.setdefault(j, DEFAULT_HOURS_TEXT)
            tf = ft.TextField(
                value=s.hours_per_day[j], multiline=True,
                min_lines=4, max_lines=24, label=j,
                width=_s(120), text_size=_s(11),
                label_style=ft.TextStyle(size=_s(11)), data=j)
            def _ch(e, _j=j):
                s.hours_per_day[_j] = e.control.value
                s.invalidate_cache()
            tf.on_change = _ch
            buf.append(tf)
        self._hours_col.controls = buf
        self.page.update()

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
                spacing=_s(20),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.ADAPTIVE),
            padding=_s(20), expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# AVAILABILITY TAB
# Row buttons already existed; col buttons added.
# Both use cell_map for in-place updates (no full rebuild).
# ══════════════════════════════════════════════════════════════════════════════

class AvailabilityTab(BaseTab):

    def __init__(self, state: AppState, page: ft.Page):
        super().__init__(state, page)
        self._cell_map: dict = {}

    def build(self):
        s = self.state
        people, _, hours, days = s.dims()
        if not days:
            return
        if s.avail_filter not in days:
            s.avail_filter = days[0]

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
                        weight=ft.FontWeight.BOLD, size=_s(12)),
                bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                on_click=_rand, width=_s(180), alignment=ft.alignment.center),
        ], spacing=_s(20))]

        buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)))

        # ── Hour-label header row ──────────────────────────────────────────
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 4)] +
            [ft.Container(ft.Text(h, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

        # ── Col-toggle buttons row ─────────────────────────────────────────
        def _make_col_toggle(_h):
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __h=_h):
                fv = s.avail_st.get((people[0], __h, j), 1)
                nv = UIHelpers._AVAIL_NEXT[fv]
                for pp in people:
                    s.avail_st[(pp, __h, j)] = nv
                    cell = self._cell_map.get((pp, __h, j))
                    if cell is not None:
                        cell.content.value = UIHelpers._AVAIL_LBL[nv]
                        cell.bgcolor       = UIHelpers._AVAIL_CLR[nv]
                        cell.update()
            btn.on_click = _click
            return btn

        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 4)] +
            [_make_col_toggle(h) for h in day_hrs],
            spacing=2, wrap=False))

        # ── Person rows with row-toggle buttons ────────────────────────────
        for p in people:
            def _row_toggle(_p=p, _j=j):
                btn = UIHelpers.make_rc_btn("row")
                def _click(e, __p=_p, __j=_j):
                    fv = s.avail_st.get((__p, hours[__j][0], __j), 1)
                    nv = UIHelpers._AVAIL_NEXT[fv]
                    for h in hours[__j]:
                        s.avail_st[(__p, h, __j)] = nv
                        cell = self._cell_map.get((__p, h, __j))
                        if cell is not None:
                            cell.content.value = UIHelpers._AVAIL_LBL[nv]
                            cell.bgcolor       = UIHelpers._AVAIL_CLR[nv]
                            cell.update()
                btn.on_click = _click
                return btn

            cell_widgets = []
            for h in day_hrs:
                toggle = UIHelpers.make_avail_toggle(s.avail_st, (p, h, j))
                self._cell_map[(p, h, j)] = toggle
                cell_widgets.append(toggle)

            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st), _row_toggle(p, j)] + cell_widgets,
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# DEMAND TAB  (unchanged from original)
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
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=_s(12), visible=False)

        buf = [ft.Row([
            nav,
            UIHelpers.make_reset_btn("Reset to Default", _reset),
            ft.Container(ft.Text("Set All to 0", color=ft.Colors.WHITE,
                                 weight=ft.FontWeight.BOLD, size=_s(12)),
                         bgcolor=ft.Colors.ORANGE_700, padding=_s(8), border_radius=4,
                         on_click=_zero, width=_s(150), alignment=ft.alignment.center),
            ft.Container(ft.Text("Random Demand (All Days)", color=ft.Colors.WHITE,
                                 weight=ft.FontWeight.BOLD, size=_s(12)),
                         bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                         on_click=_rand, width=_s(190), alignment=ft.alignment.center),
        ], spacing=_s(20))]

        buf += [ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)), err_txt]

        # ── helpers ────────────────────────────────────────────────────────
        _spacer = ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 2)

        def _adj(k, delta):
            cur = s.demand_st.get(k, "1")
            try:    v = int(cur) if cur.strip() else 0
            except: v = 0
            nv = max(0, v + delta)
            s.demand_st[k] = "" if nv == 0 else str(nv)
            s.validation_errors["demand"].discard(k)

        def _col_btn(h, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _h=h, _d=delta):
                for t in tasks:
                    _adj((t, _h, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        def _row_btn(t, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _t=t, _d=delta):
                for h in day_hrs:
                    _adj((_t, h, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        # top +1 col buttons
        buf.append(ft.Row([_spacer] + [_col_btn(h, +1) for h in day_hrs], spacing=2, wrap=False))
        # hour header labels
        buf.append(ft.Row(
            [_spacer] +
            [ft.Container(ft.Text(h, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

        for t in tasks:
            cells = []
            for h in day_hrs:
                k = (t, h, j)
                s.demand_st.setdefault(k, "1")
                is_ok = UIHelpers.validate_nonneg_int(s.demand_st[k])
                tf = ft.TextField(
                    value=s.demand_st[k], width=UIHelpers.W_CELL, height=UIHelpers.H_TF,
                    text_size=_s(11), data=k, content_padding=ft.padding.all(_s(2)),
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
                    _err.value   = f"⚠ {n} cell(s) invalid. Only non-negative integers." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row(
                [UIHelpers.lbl(t), _row_btn(t, +1)] + cells + [_row_btn(t, -1)],
                spacing=2, wrap=False))

        # bottom -1 col buttons
        buf.append(ft.Row([_spacer] + [_col_btn(h, -1) for h in day_hrs], spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS TAB  — row & col toggles added
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

        # ── Col toggle: flip all people for one task ───────────────────────
        def _make_col_toggle(_t):
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __t=_t):
                fv = s.skills_st.get((people[0], __t), 1)
                nv = 1 - fv
                for pp in people:
                    s.skills_st[(pp, __t)] = nv
                s.invalidate_cache()
                self.build()
            btn.on_click = _click
            return btn

        # ── Row toggle: flip all tasks for one person ──────────────────────
        def _make_row_toggle(_p):
            btn = UIHelpers.make_rc_btn("row")
            def _click(e, __p=_p):
                fv = s.skills_st.get((__p, tasks[0]), 1)
                nv = 1 - fv
                for tt in tasks:
                    s.skills_st[(__p, tt)] = nv
                s.invalidate_cache()
                self.build()
            btn.on_click = _click
            return btn

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=_s(16)),
            ft.Row([
                UIHelpers.make_reset_btn("Reset Skills", _reset),
                ft.Container(ft.Text("Random Skills", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=_s(12)),
                             bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                             on_click=_rand, width=_s(150), alignment=ft.alignment.center),
            ], spacing=_s(10)),
            # header labels row
            ft.Row(
                [ft.Container(width=UIHelpers.W_LBL),
                 ft.Container(width=UIHelpers.W_CELL)] +   # spacer above row-btn col
                [ft.Container(
                    ft.Text(t, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                    width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
                 for t in tasks],
                spacing=2, wrap=False),
            # col-toggle row
            ft.Row(
                [ft.Container(width=UIHelpers.W_LBL),
                 ft.Container(width=UIHelpers.W_CELL)] +
                [_make_col_toggle(t) for t in tasks],
                spacing=2, wrap=False),
        ]

        for p in people:
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st),
                 _make_row_toggle(p)] +
                [UIHelpers.make_toggle(s.skills_st, (p, t), 1) for t in tasks],
                spacing=2))

        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# QUOTA TAB  (unchanged)
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
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=_s(12), visible=False)

        buf = [
            ft.Text("Minimum Quota Matrix (per Day)", weight=ft.FontWeight.BOLD, size=_s(16)),
            ft.Row([
                nav,
                UIHelpers.make_reset_btn("Reset All Days", _reset),
                ft.Container(ft.Text("Set All to 0", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=_s(12)),
                             bgcolor=ft.Colors.ORANGE_700, padding=_s(8), border_radius=4,
                             on_click=_zero, width=_s(150), alignment=ft.alignment.center),
                ft.Container(ft.Text("Random Quota (All Days)", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=_s(12)),
                             bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                             on_click=_rand, width=_s(180), alignment=ft.alignment.center),
            ], spacing=_s(20)),
            err_txt,
            ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)),
        ]

        # ── helpers ────────────────────────────────────────────────────────
        _spacer = ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 2)

        def _adj(k, delta):
            cur = s.quota_st.get(k, "")
            try:    v = int(cur) if cur.strip() else 0
            except: v = 0
            nv = max(0, v + delta)
            s.quota_st[k] = "" if nv == 0 else str(nv)
            s.validation_errors["quota"].discard(k)

        def _col_btn(t, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _t=t, _d=delta):
                for p in people:
                    _adj((p, _t, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        def _row_btn(p, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _p=p, _d=delta):
                for t in tasks:
                    _adj((_p, t, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        # top +1 col buttons
        buf.append(ft.Row([_spacer] + [_col_btn(t, +1) for t in tasks], spacing=2, wrap=False))
        # task header labels
        buf.append(ft.Row(
            [_spacer] +
            [ft.Container(ft.Text(t, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for t in tasks],
            spacing=2, wrap=False))

        for p in people:
            cells = []
            for t in tasks:
                k = (p, t, j)
                s.quota_st.setdefault(k, "")
                is_ok = UIHelpers.validate_nonneg_int(s.quota_st[k])
                tf = ft.TextField(
                    value=s.quota_st[k], width=UIHelpers.W_CELL, height=UIHelpers.H_TF,
                    text_size=_s(11), data=k, content_padding=ft.padding.all(_s(2)),
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
                    _err.value   = f"⚠ {n} cell(s) invalid. Only non-negative integers." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()
                tf.on_change = _ch
                cells.append(tf)
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st), _row_btn(p, +1)] + cells + [_row_btn(p, -1)],
                spacing=2, wrap=False))

        # bottom -1 col buttons
        buf.append(ft.Row([_spacer] + [_col_btn(t, -1) for t in tasks], spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# FORCE TAB  — row & col toggles added
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

        # ── Col toggle: flip all people for one hour ───────────────────────
        def _make_col_toggle(_h):
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __h=_h):
                fv = s.force_st.get((people[0], t, __h, j), 0)
                nv = 1 - fv
                for pp in people:
                    s.force_st[(pp, t, __h, j)] = nv
                self.build()
            btn.on_click = _click
            return btn

        # ── Row toggle: flip all hours for one person ──────────────────────
        def _make_row_toggle(_p):
            btn = UIHelpers.make_rc_btn("row")
            def _click(e, __p=_p):
                fv = s.force_st.get((__p, t, day_hrs[0], j), 0)
                nv = 1 - fv
                for h in day_hrs:
                    s.force_st[(__p, t, h, j)] = nv
                self.build()
            btn.on_click = _click
            return btn

        buf = [ft.Row([
            UIHelpers.make_nav_dropdown("Select Day",  j, days,  _on_day,  150),
            UIHelpers.make_nav_dropdown("Select Task", t, tasks, _on_task, 200),
            UIHelpers.make_reset_btn("Reset All",     _reset_all),
            UIHelpers.make_reset_btn("Reset Current", _reset_cur),
        ], spacing=_s(20))]

        buf.append(ft.Text(f"-- {t} / {j} --", weight=ft.FontWeight.BOLD, size=_s(14)))

        # header labels
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [ft.Container(
                ft.Text(h, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

        # col toggles
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [_make_col_toggle(h) for h in day_hrs],
            spacing=2, wrap=False))

        for p in people:
            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.captains_st),
                 _make_row_toggle(p)] +
                [UIHelpers.make_force_toggle(s.force_st, (p, t, h, j), 0, tbg, tfg)
                 for h in day_hrs],
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL TAB  — row & col toggles added
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
                                 weight=ft.FontWeight.BOLD, size=_s(12)),
                         bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                         on_click=_rand, width=_s(150), alignment=ft.alignment.center),
        ], spacing=_s(10)))

        sw_hard = ft.Switch(label="Enemies: Hard Constraint", value=s.hard_enemies)
        sw_hard.on_change = lambda e: setattr(s, "hard_enemies", e.control.value)
        buf.append(ft.Row([sw_hard]))

        _lbl  = {0: "~", 1: "+", -1: "-"}
        _clr  = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_700, -1: ft.Colors.RED_700}
        _next = {0: 1, 1: -1, -1: 0}

        col_people = people[1:]   # columns shown in header

        # ── Col toggle: flip all (p1, p2) pairs where p1 comes before p2 ──
        def _make_col_toggle(_p2):
            p2_idx = people.index(_p2)
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __p2=_p2, __p2_idx=p2_idx):
                pairs = [(people[i], __p2) for i in range(__p2_idx)]
                if not pairs:
                    return
                fv = s.social_st.get(pairs[0], 0)
                nv = _next[fv]
                for p1_, p2_ in pairs:
                    s.social_st[(p1_, p2_)] = nv
                s.invalidate_cache()
                self.build()
            btn.on_click = _click
            return btn

        # ── Row toggle: flip all (p1, p2) pairs where p2 comes after p1 ───
        def _make_row_toggle(_p1):
            p1_idx = people.index(_p1)
            btn = UIHelpers.make_rc_btn("row")
            def _click(e, __p1=_p1, __p1_idx=p1_idx):
                pairs = [(__p1, people[j]) for j in range(__p1_idx + 1, len(people))]
                if not pairs:
                    return
                fv = s.social_st.get(pairs[0], 0)
                nv = _next[fv]
                for p1_, p2_ in pairs:
                    s.social_st[(p1_, p2_)] = nv
                s.invalidate_cache()
                self.build()
            btn.on_click = _click
            return btn

        # header labels row  [spacer_name | spacer_rowbtn | label_p2 ...]
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [ft.Container(
                ft.Text(p, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for p in col_people],
            spacing=2, wrap=False))

        # col-toggle row  [spacer_name | spacer_rowbtn | col_btn_p2 ...]
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [_make_col_toggle(p2) for p2 in col_people],
            spacing=2, wrap=False))

        # person rows
        for i, p1 in enumerate(people):
            cells = []
            for p2 in col_people:
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
                                        size=_s(12), weight=ft.FontWeight.BOLD),
                        width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=k,
                        bgcolor=_clr[s.social_st[k]],
                        alignment=ft.alignment.center, border_radius=4, on_click=_click)
                    cells.append(btn)
                else:
                    cells.append(ft.Container(width=UIHelpers.W_CELL))

            if cells:
                buf.append(ft.Row(
                    [UIHelpers.plbl(p1, s.captains_st),
                     _make_row_toggle(p1)] + cells,
                    spacing=2, wrap=False))

        self._ct.controls = buf
        self.page.update()


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTS TAB
# ══════════════════════════════════════════════════════════════════════════════

class WeightsTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Column(expand=True, spacing=_s(5), scroll=ft.ScrollMode.ALWAYS)

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
                ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=_s(16)),
                UIHelpers.make_reset_btn("Reset to Default", _reset),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=_s(420)),
            ft.Text("Drag items to reorder. Top items = Higher Cost.",
                    italic=True, size=_s(12)),
            ft.Divider(),
        ], spacing=_s(5))

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
                    ft.Text(f"#{i + 1}", width=_s(30), weight=ft.FontWeight.BOLD,
                            size=_s(12)),
                    ft.Text(key, expand=True, weight=ft.FontWeight.W_800, size=_s(12)),
                    ft.Text(f"{val}",
                            color=ft.Colors.BLACK if s.weights_enabled[key] else ft.Colors.GREY_500,
                            size=_s(13)),
                    sw,
                ], alignment=ft.MainAxisAlignment.START),
                padding=_s(8),
                bgcolor=ft.Colors.LIGHT_BLUE_100 if s.weights_enabled[key] else ft.Colors.GREY_300,
                border=ft.border.all(1, ft.Colors.GREY_400), border_radius=6,
                margin=ft.margin.only(bottom=_s(4))))

        layout = ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.ReorderableListView(controls=items, on_reorder=_handle_reorder),
                    width=_s(420), height=_s(500)),
            ],
            width=_s(420), alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        self._ct.controls = [ft.Row([layout], alignment=ft.MainAxisAlignment.CENTER)]
        self.page.update()

    def get_container(self) -> ft.Container:
        return ft.Container(self._ct, padding=_s(10), expand=True)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT TAB
# ══════════════════════════════════════════════════════════════════════════════

class OutputTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Column(expand=True, spacing=_s(5), scroll=ft.ScrollMode.ALWAYS)

    def _handle_base_click(self, run_idx: int):
        s = self.state
        s.base_run_idx = None if s.base_run_idx == run_idx else run_idx
        self.rebuild()
        self.page.update()

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
            content=ft.Text(char, size=_s(8), color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
            width=_s(13), height=_s(13), bgcolor=color, border_radius=7,
            alignment=ft.alignment.center, right=1, top=1)

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
            content=ft.Text(d_lbl, size=_s(11), color=d_fg,
                            weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=d_bg, padding=ft.padding.symmetric(_s(4), _s(8)),
            border_radius=4, on_click=d_click)

        hdr_lbl = (f"⟳  Run #{run_idx+1}  ·  Solving…"
                   if (is_live and is_solving)
                   else f"Run #{run_idx+1}  ·  {status}")
        hdr_bg  = "#E65100" if (is_live and is_solving) else "#1565C0"

        close_btn = (
            [ft.IconButton(
                icon=ft.Icons.CLOSE, icon_color=ft.Colors.WHITE, icon_size=_s(16),
                tooltip="Remove this run",
                style=ft.ButtonStyle(padding=ft.padding.all(_s(2))),
                on_click=lambda e, _cb=on_delete: _cb())]
            if on_delete is not None else [])

        title_row = ft.Row(
            controls=[
                ft.Text(hdr_lbl, weight=ft.FontWeight.BOLD, size=_s(13),
                        color=ft.Colors.WHITE, no_wrap=True, expand=True),
                *close_btn,
            ],
            spacing=_s(4), vertical_alignment=ft.CrossAxisAlignment.CENTER)

        if not is_live:
            is_base    = (run_idx == s.base_run_idx)
            base_label = "✓ Base activo" if is_base else "📌 Usar como Base"
            base_btn   = ft.Container(
                content=ft.Text(base_label, size=_s(11),
                                color=BASE_ACTIVE_FG if is_base else BASE_IDLE_FG,
                                weight=ft.FontWeight.BOLD, no_wrap=True),
                bgcolor=BASE_ACTIVE_BG if is_base else BASE_IDLE_BG,
                padding=ft.padding.symmetric(_s(4), _s(8)),
                border_radius=4,
                tooltip=(
                    "Base activo. Pulsa de nuevo para desactivarlo."
                    if is_base else
                    "Marcar como Base: el próximo Solve minimizará cambios respecto a este planning."
                ),
                on_click=lambda e, _i=run_idx: self._handle_base_click(_i))
            action_row = ft.Row(controls=[base_btn, diff_btn],
                                spacing=_s(6),
                                vertical_alignment=ft.CrossAxisAlignment.CENTER)
            header_content = ft.Column(controls=[title_row, action_row],
                                       spacing=_s(4), tight=True)
        else:
            header_content = ft.Column(
                controls=[title_row, ft.Row([diff_btn], spacing=_s(6))],
                spacing=_s(4), tight=True)

        buf.append(ft.Container(
            content=header_content,
            bgcolor=hdr_bg,
            padding=ft.padding.symmetric(_s(8), _s(12)), border_radius=6,
            margin=ft.margin.only(bottom=_s(6))))

        if not (is_live and is_solving):
            buf.append(ft.Text(f"⏱  Solve time: {sol.get('solve_time', 0.0):.1f} s",
                               size=_s(12), italic=True, color=ft.Colors.GREY_700))
            buf.append(ft.Text(f"◎  MIP Gap: {sol.get('mip_gap', 0.0) * 100:.3f} %",
                               size=_s(12), italic=True, color=ft.Colors.GREY_700))

        CW = _s(40); NW = _s(50); TW = _s(25); Ch = _s(26)

        if is_live and is_solving and not any(assignment.get(d) for d in days):
            buf.append(ft.Row([
                ft.ProgressRing(width=_s(22), height=_s(22), stroke_width=3),
                ft.Text("Solving in background…", italic=True, size=_s(12)),
            ], spacing=_s(10)))
        else:
            if is_live and is_solving:
                buf.append(ft.Row([
                    ft.ProgressRing(width=_s(16), height=_s(16), stroke_width=2),
                    ft.Text("Updating…", italic=True, size=_s(11), color=ft.Colors.ORANGE_700),
                ], spacing=_s(6)))

            tc       = {t: TASK_COLORS[i % len(TASK_COLORS)] for i, t in enumerate(tasks)}
            ref_asgn = ref_sol.get("assignment", {}) if ref_sol else {}

            for j in days:
                if j not in assignment:
                    continue
                day_hrs = hours[j]
                buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(16)))

                def _hc(txt, w):
                    return ft.Container(
                        ft.Text(txt, size=_s(11), weight=ft.FontWeight.BOLD,
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
                        ft.Text(p, size=_s(12), weight=ft.FontWeight.BOLD, color=name_color),
                        width=NW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center_left,
                        padding=ft.padding.only(left=_s(8)),
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
                                ft.Text(task, size=_s(11), weight=ft.FontWeight.BOLD,
                                        color=fg, text_align=ft.TextAlign.CENTER),
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
                        ft.Text(str(int(total)), size=_s(12), weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER),
                        width=TW, height=Ch, bgcolor=row_bg,
                        alignment=ft.alignment.center,
                        border=ft.border.all(1, "#CFD8DC")))
                    buf.append(ft.Row(cells, spacing=0, wrap=False))

                legend = [
                    ft.Container(
                        ft.Text(t, size=_s(10), weight=ft.FontWeight.BOLD, color=tc[t][1]),
                        bgcolor=tc[t][0],
                        padding=ft.padding.symmetric(_s(4), _s(8)), border_radius=4)
                    for t in tasks]
                for lbl_txt, clr in [("Available", AVAIL_COLOR),
                                     ("Emergency", EMERG_COLOR),
                                     ("Unavailable", UNAVAIL_COLOR)]:
                    legend.append(ft.Container(
                        ft.Text(lbl_txt, size=_s(10), color=ft.Colors.WHITE),
                        bgcolor=clr,
                        padding=ft.padding.symmetric(_s(4), _s(8)), border_radius=4))
                if diff_mode == "cmp":
                    for char, color, desc in [
                        ("+", DIFF_ADD_COLOR,    "Nueva asignación"),
                        ("−", DIFF_REMOVE_COLOR, "Eliminada"),
                        ("⇄", DIFF_CHANGE_COLOR, "Cambiada"),
                    ]:
                        legend.append(ft.Row([
                            ft.Container(
                                ft.Text(char, size=_s(8), color=ft.Colors.WHITE,
                                        weight=ft.FontWeight.BOLD),
                                width=_s(13), height=_s(13), bgcolor=color,
                                border_radius=7, alignment=ft.alignment.center),
                            ft.Text(desc, size=_s(10)),
                        ], spacing=_s(4), vertical_alignment=ft.CrossAxisAlignment.CENTER))
                buf.append(ft.Row(legend, spacing=_s(8), wrap=True))
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
                    buf.append(ft.Text(section_title, weight=ft.FontWeight.BOLD, size=_s(14)))
                    if issues:
                        for line in issues:
                            buf.append(ft.Text(f"  {line}", size=_s(12)))
                    else:
                        buf.append(ft.Text(f"  {empty_msg}", size=_s(12), italic=True))

                workload     = sol.get("workload", {})
                ref_workload = ref_sol.get("workload", {}) if ref_sol else {}
                buf.append(ft.Text("WORKLOAD EQUITY", weight=ft.FontWeight.BOLD, size=_s(14)))
                for p in people:
                    ch = workload.get(p, 0)
                    if diff_mode == "cmp" and ref_sol is not None:
                        ph = ref_workload.get(p, 0)
                        if ch != ph:
                            buf.append(ft.Text(
                                f"  {p}: {ch:.0f} h  ({ph:.0f} prev)",
                                size=_s(12), color=ft.Colors.ORANGE_700,
                                weight=ft.FontWeight.BOLD))
                        else:
                            buf.append(ft.Text(f"  {p}: {ch:.0f} hours", size=_s(12)))
                    else:
                        buf.append(ft.Text(f"  {p}: {ch:.0f} hours", size=_s(12)))
                buf.append(ft.Text(
                    f"  Global range: "
                    f"max={sol.get('workload_max', 0):.0f}, "
                    f"min={sol.get('workload_min', 0):.0f}",
                    size=_s(12), italic=True))

        max_hrs   = max((len(hours[d]) for d in days), default=0)
        col_width = max(NW + CW * max_hrs + TW + _s(30), _s(460))
        return ft.Container(
            content=ft.Column(controls=buf, spacing=_s(5), tight=True),
            width=col_width,
            padding=ft.padding.symmetric(horizontal=_s(16)),
            border=ft.border.only(right=ft.border.BorderSide(1, "#B0BEC5")))

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
                    if s.base_run_idx == idx:
                        s.base_run_idx = None
                    elif s.base_run_idx is not None and s.base_run_idx > idx:
                        s.base_run_idx -= 1
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
            padding=_s(10), expand=True)


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

        X_prev      = s.build_x_prev(people, tasks, hours, days)
        using_base  = s.base_run_idx is not None
        live_status = (f"Solving… (Base: Run #{s.base_run_idx + 1})"
                       if using_base else "Solving...")

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
            X_prev   = X_prev,
            weights          = s.weights_st.copy(),
            max_consec_hours = max_consec,
            captains         = [p for p in people if s.captains_st.get(p, 0) == 1],
            solver_params    = s.solver_params,
            hard_enemies     = s.hard_enemies,
            day_heuristics   = 1 if self._sw_heur.value else 0,
        )

        self._out.rebuild(
            live_sol={"status": live_status, "assignment": {}},
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
                s.base_run_idx = None
                self._out.rebuild()
            except Exception as ex:
                self._out._ct.controls = [
                    ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=_s(14))]
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
        self._solve_btn = None
        self._configure_page()
        self._build_ui()

    def _configure_page(self):
        p = self.page
        p.title            = "Staff Scheduler"
        p.scroll           = None
        p.window.maximized = True
        p.theme_mode       = ft.ThemeMode.LIGHT
        p.padding          = 0

    def _update_solve_blocked(self):
        s       = self.state
        has_err = bool(s.validation_errors["demand"] or
                       s.validation_errors["quota"]  or
                       s.validation_errors["consec"])
        s.solve_blocked = has_err
        if self._solve_btn is None:
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

        self._dims_tab    = DimensionsTab(s, self.page, self._update_solve_blocked)
        self._avail_tab   = AvailabilityTab(s, self.page)
        self._demand_tab  = DemandTab(s, self.page, self._update_solve_blocked)
        self._skills_tab  = SkillsTab(s, self.page)
        self._quota_tab   = QuotaTab(s, self.page, self._update_solve_blocked)
        self._force_tab   = ForceTab(s, self.page)
        self._social_tab  = SocialTab(s, self.page)
        self._weights_tab = WeightsTab(s, self.page)
        self._output_tab  = OutputTab(s, self.page)

        self._solve_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE, size=_s(18)),
                ft.Text("SOLVE", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(13)),
            ], spacing=_s(6), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#1565C0",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=8, on_click=self._do_solve,
            width=SIDEBAR_WIDTH - _s(32), alignment=ft.alignment.center)

        self._stop_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE, size=_s(18)),
                ft.Text("STOP", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(13)),
            ], spacing=_s(6), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#C62828",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=8, on_click=self._do_stop,
            width=SIDEBAR_WIDTH - _s(32), alignment=ft.alignment.center)

        self._day_heuristics_sw = ft.Switch(
            label="Day Heuristics", value=False,
            label_style=ft.TextStyle(color=ft.Colors.WHITE, size=_s(11)))

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
                            size=_s(20)),
                    ft.Text(label, size=_s(13),
                            color=SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR,
                            weight=ft.FontWeight.BOLD if sel else None),
                ], spacing=_s(10)),
                padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(12)),
                border_radius=8,
                bgcolor=SIDEBAR_SELECTED_BG if sel else None,
                on_click=self._on_menu_click, data=idx, ink=True)
            self._menu_btn_refs.append(btn)

        sidebar = ft.Container(
            width=SIDEBAR_WIDTH, bgcolor=SIDEBAR_BG,
            padding=ft.padding.only(top=_s(12), bottom=_s(12),
                                    left=_s(8), right=_s(8)),
            content=ft.Column(
                controls=[
                    ft.Container(
                        ft.Text("Staff Scheduler", size=_s(16),
                                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        padding=ft.padding.only(left=_s(8), bottom=_s(4))),
                    ft.Container(
                        ft.Column([
                            self._solve_btn, self._day_heuristics_sw, self._stop_btn,
                        ], spacing=_s(6)),
                        padding=ft.padding.only(bottom=_s(8))),
                    ft.Divider(color="#455A64", height=1),
                    ft.Column(controls=self._menu_btn_refs, spacing=_s(2),
                              scroll=ft.ScrollMode.AUTO, expand=True),
                ],
                spacing=_s(4), expand=True),
            border=ft.border.only(right=ft.border.BorderSide(1, "#455A64")))

        self.page.add(ft.Row(
            controls=[sidebar, self._content_area],
            spacing=0, expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START))

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