"""
FinderTab — Personal schedule finder with optional compare mode.

Auto-jumps to the latest run whenever a new one is added to history.
Manual navigation within the tab is preserved as long as history length
stays constant between builds.
"""

import flet as ft
from constants import _s, TASK_COLORS, TRAVEL_LABEL, TRAVEL_COLOR, TRAVEL_FG_COLOR

_DAY_COL_WIDTH     = _s(260)
_DAY_COL_WIDTH_CMP = _s(230)
_SLOT_HEIGHT       = _s(44)
_HEADER_HEIGHT     = _s(42)
_SUBHEADER_HEIGHT  = _s(26)
_SIDE_PANEL_WIDTH  = _s(360)

_SOCIAL_LBL = {1: ("friend", "#2E7D32", "#C8E6C9"),
               -1: ("enemy",  "#C62828", "#FFCDD2"),
               0:  ("neutral", "#546E7A", "#ECEFF1")}


class FinderTab:
    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page = page
        self._selected_run_idx: int | None = None
        self._last_seen_history_len: int = 0
        self._sel_person: dict[int, str | None] = {1: None, 2: None}
        self._compare_mode: bool = False

        self._run_dd = ft.Dropdown(
            label="Run", width=_s(220), text_size=_s(13),
            label_style=ft.TextStyle(size=_s(12)), dense=True,
            content_padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            on_change=self._on_run_change)

        self._sel = {1: self._make_selector(1, "Person A"),
                     2: self._make_selector(2, "Person B")}

        self._compare_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.COMPARE_ARROWS, color=ft.Colors.WHITE, size=_s(16)),
                ft.Text("Compare", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(12)),
            ], spacing=_s(5), tight=True),
            bgcolor="#546E7A",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=6, on_click=self._toggle_compare, ink=True,
            tooltip="Toggle side-by-side comparison")

        self._affinity_badge = ft.Container(visible=False)

        self._days_row = ft.Row(
            spacing=_s(22), run_spacing=_s(22),
            vertical_alignment=ft.CrossAxisAlignment.START, wrap=True)
        self._days_scroller = ft.Column(
            controls=[self._days_row], spacing=0,
            scroll=ft.ScrollMode.AUTO, expand=True)

        self._prefs_box  = ft.Container()
        self._social_box = ft.Container()
        self._side_panel = ft.Container(
            content=ft.Column(
                controls=[self._prefs_box, self._social_box],
                spacing=_s(20), scroll=ft.ScrollMode.AUTO, expand=True),
            width=_SIDE_PANEL_WIDTH,
            padding=ft.padding.only(right=_s(28)))

        main_body = ft.Row(
            controls=[
                self._side_panel,
                ft.VerticalDivider(width=1),
                ft.Container(content=self._days_scroller, expand=True,
                             padding=ft.padding.only(left=_s(28))),
            ],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.START, expand=True)

        s1, s2 = self._sel[1], self._sel[2]
        top_bar = ft.Column(
            controls=[
                ft.Row([self._run_dd], spacing=_s(4)),
                ft.Row([
                    s1["prev"], s1["dd"], s1["next"], s1["counter"],
                    ft.Container(width=_s(18)),
                    self._compare_btn,
                    ft.Container(width=_s(18)),
                    s2["prev"], s2["dd"], s2["next"], s2["counter"],
                    ft.Container(width=_s(22)),
                    self._affinity_badge,
                ], spacing=_s(6),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True),
            ],
            spacing=_s(18))

        self._ct = ft.Column(
            controls=[
                ft.Container(content=top_bar,
                             padding=ft.padding.only(
                                 left=_s(30), right=_s(30),
                                 top=_s(18), bottom=_s(16))),
                ft.Divider(height=1),
                ft.Container(content=main_body, expand=True,
                             padding=ft.padding.symmetric(
                                 horizontal=_s(30), vertical=_s(20))),
            ],
            spacing=0, expand=True)

    # ---------- helpers ----------

    def _make_selector(self, which: int, label: str) -> dict:
        btn_style = ft.ButtonStyle(padding=ft.padding.all(_s(4)),
                                   shape=ft.RoundedRectangleBorder(radius=6))
        disabled = (which == 2)
        dd = ft.Dropdown(
            label=label, width=_s(220), text_size=_s(13),
            label_style=ft.TextStyle(size=_s(12)), dense=True,
            content_padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            on_change=lambda e, w=which: self._on_person_change(w, e.control.value),
            disabled=disabled)
        prev_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT, icon_size=_s(22), tooltip="Previous",
            style=btn_style, disabled=disabled,
            on_click=lambda e, w=which: self._step_person(-1, w))
        next_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT, icon_size=_s(22), tooltip="Next",
            style=btn_style, disabled=disabled,
            on_click=lambda e, w=which: self._step_person(1, w))
        counter = ft.Text("", size=_s(11), color=ft.Colors.GREY_600)
        return {"dd": dd, "prev": prev_btn, "next": next_btn, "counter": counter}

    def _current_entry(self):
        history = self.state.solution_history
        if not history:
            return None, None
        idx = self._resolve_run_idx(history)
        if 0 <= idx < len(history):
            return idx, history[idx]
        return None, None

    def _resolve_run_idx(self, history: list) -> int:
        n = len(history)
        if n == 0:
            return 0
        if (self._selected_run_idx is not None
                and 0 <= self._selected_run_idx < n):
            return self._selected_run_idx
        return n - 1

    # ---------- lifecycle ----------

    def get_container(self) -> ft.Container:
        return ft.Container(content=self._ct, padding=0, expand=True)

    def build(self):
        history = self.state.solution_history
        self._run_dd.options = [
            ft.dropdown.Option(key=str(i), text=f"Run #{i + 1}",
                               text_style=ft.TextStyle(size=_s(12)))
            for i in range(len(history))]

        if not history:
            self._run_dd.value = None
            self._last_seen_history_len = 0
            for w in (1, 2):
                self._sel[w]["dd"].options = []
                self._sel[w]["dd"].value = None
                self._sel[w]["counter"].value = ""
            self._prefs_box.content = None
            self._social_box.content = None
            self._affinity_badge.visible = False
            self._days_row.controls = [
                ft.Text("No runs available yet. Solve at least once.",
                        size=_s(14), italic=True, color=ft.Colors.GREY_600)]
            return

        # Auto-jump to the latest run whenever a new one has been added
        # since the last time this tab was built. Manual navigation within
        # the tab is preserved as long as history length stays constant.
        if len(history) > self._last_seen_history_len:
            self._selected_run_idx = len(history) - 1
            self._sel_person = {1: None, 2: None}
        self._last_seen_history_len = len(history)

        self._selected_run_idx = self._resolve_run_idx(history)
        self._run_dd.value = str(self._selected_run_idx)
        people = history[self._selected_run_idx]["people"]

        opts = [ft.dropdown.Option(key=p, text=p,
                                   text_style=ft.TextStyle(size=_s(12)))
                for p in people]
        for w in (1, 2):
            self._sel[w]["dd"].options = list(opts)

        if self._sel_person[1] not in people:
            self._sel_person[1] = people[0] if people else None
        if (self._sel_person[2] not in people
                or self._sel_person[2] == self._sel_person[1]):
            self._sel_person[2] = next(
                (p for p in people if p != self._sel_person[1]), None)

        self._sel[1]["dd"].value = self._sel_person[1]
        self._sel[2]["dd"].value = self._sel_person[2]

        self._apply_compare_state()
        self._update_person_counter()
        self._rebuild_content()

    # ---------- state sync ----------

    def _apply_compare_state(self):
        on = self._compare_mode
        for key in ("dd", "prev", "next"):
            self._sel[2][key].disabled = not on
        self._compare_btn.bgcolor = "#1565C0" if on else "#546E7A"

        p1, p2 = self._sel_person[1], self._sel_person[2]
        self._affinity_badge.visible = on and bool(p1 and p2)
        if not self._affinity_badge.visible:
            return

        val = self._get_social(p1, p2)
        lbl, fg, bg = _SOCIAL_LBL[val]
        hard = self.state.hard_enemies and val == -1
        extra = "  (HARD)" if hard else ""
        self._affinity_badge.content = ft.Row([
            ft.Icon(ft.Icons.PEOPLE, color=fg, size=_s(16)),
            ft.Text(f"{p1} ↔ {p2}: {lbl}{extra}",
                    size=_s(12), weight=ft.FontWeight.BOLD, color=fg),
        ], spacing=_s(6), tight=True)
        self._affinity_badge.bgcolor = bg
        self._affinity_badge.padding = ft.padding.symmetric(
            horizontal=_s(12), vertical=_s(8))
        self._affinity_badge.border_radius = 6
        self._affinity_badge.border = ft.border.all(2, fg)

    def _update_person_counter(self):
        _, entry = self._current_entry()
        people = entry["people"] if entry else []
        n = len(people)
        for w in (1, 2):
            p = self._sel_person[w]
            self._sel[w]["counter"].value = (
                f"{people.index(p) + 1}/{n}" if p in people else "")

    # ---------- event handlers ----------

    def _on_run_change(self, e):
        try:
            self._selected_run_idx = int(e.control.value)
        except (TypeError, ValueError):
            self._selected_run_idx = None
        self.build()
        self.page.update()

    def _on_person_change(self, which: int, value: str | None):
        self._sel_person[which] = value
        self._apply_compare_state()
        self._update_person_counter()
        self._rebuild_content()
        self.page.update()

    def _toggle_compare(self, e):
        self._compare_mode = not self._compare_mode
        if self._compare_mode:
            _, entry = self._current_entry()
            if entry is not None:
                people = entry["people"]
                if (self._sel_person[2] not in people
                        or self._sel_person[2] == self._sel_person[1]):
                    self._sel_person[2] = next(
                        (p for p in people if p != self._sel_person[1]), None)
                    self._sel[2]["dd"].value = self._sel_person[2]
        self._apply_compare_state()
        self._update_person_counter()
        self._rebuild_content()
        self.page.update()

    def _step_person(self, direction: int, which: int):
        _, entry = self._current_entry()
        if entry is None:
            return
        people = entry["people"]
        if not people:
            return
        cur_name = self._sel_person[which]
        cur = people.index(cur_name) if cur_name in people else 0
        new_name = people[(cur + direction) % len(people)]
        self._sel_person[which] = new_name
        self._sel[which]["dd"].value = new_name
        self._apply_compare_state()
        self._update_person_counter()
        self._rebuild_content()
        self.page.update()

    # ---------- content build ----------

    def _rebuild_content(self):
        idx, entry = self._current_entry()
        if idx is None or self._sel_person[1] is None:
            self._prefs_box.content = None
            self._social_box.content = None
            self._days_row.controls = [
                ft.Text("Select a run and a person.",
                        size=_s(14), italic=True, color=ft.Colors.GREY_600)]
            return

        tasks = entry["tasks"]
        tc = {t: TASK_COLORS[i % len(TASK_COLORS)] for i, t in enumerate(tasks)}
        tc[TRAVEL_LABEL] = (TRAVEL_COLOR, TRAVEL_FG_COLOR)

        persons = [self._sel_person[1]]
        if self._compare_mode and self._sel_person[2]:
            persons.append(self._sel_person[2])

        self._days_row.controls = self._build_day_columns(persons, entry, tc)
        self._prefs_box.content  = self._build_prefs_box(
            self._sel_person[1], tasks, tc)
        self._social_box.content = self._build_social_box(
            self._sel_person[1], entry)

    def _build_prefs_box(self, person: str, tasks: list, tc: dict):
        s = self.state
        enabled = s.pref_enabled_st.get(person, True)
        order = [t for t in s.pref_order_st.get(person, list(tasks)) if t in tasks]

        chips = []
        for rank, t in enumerate(order):
            bg, fg = tc.get(t, ("#ECEFF1", ft.Colors.BLACK))
            op = 1.0 if enabled else 0.4
            chips.append(ft.Container(
                ft.Row([
                    ft.Container(
                        ft.Text(str(rank), size=_s(10), color=fg,
                                weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER),
                        bgcolor="#FFFFFF", border_radius=9,
                        width=_s(18), height=_s(18),
                        alignment=ft.alignment.center),
                    ft.Text(t, size=_s(12), color=fg,
                            weight=ft.FontWeight.BOLD, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                ], spacing=_s(8),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=bg, opacity=op,
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
                border_radius=5,
            ))

        status_lbl = "ranked" if enabled else "disabled"
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SORT, size=_s(15), color="#37474F"),
                    ft.Text(f"{person} — task preferences",
                            size=_s(12), weight=ft.FontWeight.BOLD,
                            color="#37474F", expand=True),
                    ft.Container(
                        ft.Text(status_lbl, size=_s(9), color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD),
                        bgcolor="#2E7D32" if enabled else "#9E9E9E",
                        padding=ft.padding.symmetric(
                            horizontal=_s(6), vertical=_s(2)),
                        border_radius=3),
                ], spacing=_s(6),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Container(height=_s(4)),
                ft.Column(chips, spacing=_s(4), tight=True),
            ], spacing=_s(4), tight=True),
            padding=_s(10), bgcolor="#FAFAFA",
            border=ft.border.all(1, "#CFD8DC"), border_radius=6,
        )

    def _get_social(self, p1: str, p2: str) -> int:
        s = self.state
        if (p1, p2) in s.social_st:
            return s.social_st[(p1, p2)]
        if (p2, p1) in s.social_st:
            return s.social_st[(p2, p1)]
        return 0

    def _build_social_box(self, person: str, entry: dict):
        friends, enemies = [], []
        for q in entry["people"]:
            if q == person:
                continue
            v = self._get_social(person, q)
            if v == 1:
                friends.append(q)
            elif v == -1:
                enemies.append(q)

        def _chip(name: str, kind: int):
            _, fg, bg = _SOCIAL_LBL[kind]
            return ft.Container(
                ft.Text(name, size=_s(11), color=fg,
                        weight=ft.FontWeight.BOLD, no_wrap=True),
                bgcolor=bg,
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                border_radius=12)

        def _col(title: str, kind: int, names: list):
            _, head_fg, _ = _SOCIAL_LBL[kind]
            items = ([_chip(n, kind) for n in names] if names
                     else [ft.Text("—", size=_s(11),
                                   italic=True, color="#9E9E9E")])
            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(
                            ft.Icons.SENTIMENT_SATISFIED if kind == 1
                            else ft.Icons.SENTIMENT_VERY_DISSATISFIED,
                            size=_s(14), color=head_fg),
                        ft.Text(title, size=_s(12),
                                weight=ft.FontWeight.BOLD, color=head_fg),
                    ], spacing=_s(4)),
                    ft.Container(height=_s(4)),
                    ft.Column(items, spacing=_s(4), tight=True),
                ], spacing=_s(2), tight=True),
                padding=_s(8), expand=True)

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.PEOPLE_OUTLINE, size=_s(15), color="#37474F"),
                    ft.Text(f"{person} — social", size=_s(12),
                            weight=ft.FontWeight.BOLD, color="#37474F"),
                ], spacing=_s(6)),
                ft.Row([
                    _col("Friends", 1, friends),
                    ft.VerticalDivider(width=1),
                    _col("Enemies", -1, enemies),
                ], spacing=_s(6),
                   vertical_alignment=ft.CrossAxisAlignment.START),
            ], spacing=_s(4), tight=True),
            padding=_s(10), bgcolor="#FAFAFA",
            border=ft.border.all(1, "#CFD8DC"), border_radius=6)

    def _build_day_columns(self, persons: list, entry: dict, tc: dict) -> list:
        hours = entry["hours"]; days = entry["days"]
        assignment = entry["sol"].get("assignment", {})
        availability = entry["availability"]; emergency = entry["emergency"]

        compare = self._compare_mode and len(persons) == 2
        col_width = _DAY_COL_WIDTH_CMP if compare else _DAY_COL_WIDTH

        day_columns = []
        for j in days:
            day_hours = hours.get(j, [])
            asgn_day = assignment.get(j, {})
            n_slots = len(day_hours)

            day_header = ft.Container(
                content=ft.Text(j, size=_s(13), weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                bgcolor="#1565C0",
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
                border_radius=ft.border_radius.only(top_left=8, top_right=8),
                alignment=ft.alignment.center, height=_HEADER_HEIGHT)

            sub_cells = []
            for p in persons:
                p_color = self.state.person_colors.get(p, "#37474F")
                sub_cells.append(ft.Container(
                    content=ft.Text(p, size=_s(10), weight=ft.FontWeight.BOLD,
                                    color=p_color, no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS),
                    bgcolor="#ECEFF1", alignment=ft.alignment.center,
                    expand=True, height=_SUBHEADER_HEIGHT,
                    padding=ft.padding.symmetric(horizontal=_s(4))))
            sub_header = ft.Row(sub_cells, spacing=0, tight=True)

            person_columns = []
            for p in persons:
                slot_controls = []
                for idx, h in enumerate(day_hours):
                    task = asgn_day.get(p, {}).get(h)
                    avail = availability.get((p, h, j), 1)
                    emerg = emergency.get((p, h, j), 0)
                    slot_controls.append(
                        _build_slot_row(h, task, avail, emerg, tc,
                                        is_last=(idx == n_slots - 1),
                                        compact=compare))
                person_columns.append(ft.Container(
                    content=ft.Column(slot_controls, spacing=0, tight=True),
                    expand=True))

            slots_container = ft.Container(
                content=ft.Row(person_columns, spacing=0, tight=True,
                               vertical_alignment=ft.CrossAxisAlignment.START),
                height=n_slots * _SLOT_HEIGHT,
                border=ft.border.only(
                    left=ft.border.BorderSide(1, "#CFD8DC"),
                    right=ft.border.BorderSide(1, "#CFD8DC"),
                    bottom=ft.border.BorderSide(1, "#CFD8DC")),
                border_radius=ft.border_radius.only(
                    bottom_left=8, bottom_right=8),
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS)

            day_columns.append(ft.Container(
                content=ft.Column([day_header, sub_header, slots_container],
                                  spacing=0),
                width=col_width * (2 if compare else 1)))

        return day_columns


def _build_slot_row(hour: str, task: str | None,
                    avail: int, emerg: int, tc: dict,
                    is_last: bool = False,
                    compact: bool = False) -> ft.Container:
    dot_color = ("#EF5350" if avail == 0
                 else "#FF9800" if emerg == 1
                 else "#66BB6A")
    avail_dot = ft.Container(width=_s(8), height=_s(8),
                             bgcolor=dot_color, border_radius=_s(4))
    hour_label = ft.Text(hour, size=_s(11 if compact else 13),
                         weight=ft.FontWeight.BOLD, color="#546E7A",
                         no_wrap=True, width=_s(42 if compact else 50))

    if task:
        bg, fg = tc.get(task, ("#ECEFF1", ft.Colors.BLACK))
        display = (("🚗" if compact else "🚗  Travel")
                   if task == TRAVEL_LABEL else task)
        task_cell = ft.Container(
            ft.Text(display, size=_s(11 if compact else 13),
                    weight=ft.FontWeight.BOLD, color=fg, no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS),
            bgcolor=bg,
            padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
            border_radius=5, expand=True)
    else:
        unavail = avail == 0
        lbl = ("✗" if compact else "✗  unavailable") if unavail else "—"
        clr = "#BDBDBD" if unavail else "#D5D5D5"
        task_cell = ft.Container(
            ft.Text(lbl, size=_s(10 if compact else 11),
                    italic=True, color=clr),
            expand=True, padding=ft.padding.only(left=_s(8)))

    bottom_border = (ft.border.only() if is_last
                     else ft.border.only(
                         bottom=ft.border.BorderSide(1, "#ECEFF1")))

    return ft.Container(
        content=ft.Row([hour_label, avail_dot, task_cell],
                       spacing=_s(5),
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
        height=_SLOT_HEIGHT,
        padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(3)),
        border=bottom_border)