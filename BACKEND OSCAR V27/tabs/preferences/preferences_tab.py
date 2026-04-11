"""
preferences_tab.py
==================
Per-person drag-and-drop task preference ranking using ReorderableListView.
Top = most preferred (cost 0), bottom = least preferred (cost N-1).
Cyclic button per person: ON → ranked costs; OFF → all costs = 0.
Live cost matrix displayed below.
"""

import flet as ft
from constants import _s, TASK_COLORS
from state import AppState


class PreferencesTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page = page
        self._ct = ft.Column(spacing=_s(8), expand=True, scroll=ft.ScrollMode.AUTO)
        self._container = ft.Container(content=self._ct, expand=True, padding=_s(8))

    def get_container(self) -> ft.Container:
        return self._container

    # ── Helpers ───────────────────────────────────────────────────────

    def _safe_update(self):
        try:
            self._ct.update()
        except Exception:
            pass

    def _ensure_state(self, people, tasks):
        """Initialise / sync pref_order and pref_enabled on state."""
        s = self.state
        if not hasattr(s, "pref_order_st"):
            s.pref_order_st = {}
        if not hasattr(s, "pref_enabled_st"):
            s.pref_enabled_st = {}

        tasks_set = set(tasks)
        for p in people:
            kept = [t for t in s.pref_order_st.get(p, []) if t in tasks_set]
            kept_set = set(kept)
            s.pref_order_st[p] = kept + [t for t in tasks if t not in kept_set]
            s.pref_enabled_st.setdefault(p, True)

    @staticmethod
    def _toggle_btn(enabled: bool, label_on="ON", label_off="OFF",
                    icon_size=16, font_size=11, width=80, **kw) -> ft.Container:
        on = enabled
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE if on else ft.Icons.CANCEL,
                        color=ft.Colors.WHITE, size=_s(icon_size)),
                ft.Text(label_on if on else label_off,
                        color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD,
                        size=_s(font_size)),
            ], spacing=_s(4), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#2E7D32" if on else "#C62828",
            padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(5)),
            border_radius=5,
            width=_s(width) if width else None,
            alignment=ft.alignment.center,
            **kw,
        )

    @staticmethod
    def _cost_style(val: int, enabled: bool, n_tasks: int):
        """Return (bgcolor, fgcolor) for a cost cell."""
        if not enabled:
            return "#E0E0E0", "#9E9E9E"
        if val == 0:
            return "#C8E6C9", "#1B5E20"
        if val <= n_tasks // 3:
            return "#FFF9C4", "#F57F17"
        return "#FFCDD2", "#B71C1C"

    # ── Build / rebuild ──────────────────────────────────────────────

    def build(self):
        s = self.state
        people, tasks, _, _ = s.dims()

        if not people or not tasks:
            self._ct.controls = [
                ft.Text("Define people and tasks first.", size=_s(13), italic=True)]
            self._safe_update()
            return

        self._ensure_state(people, tasks)

        all_on = all(s.pref_enabled_st[p] for p in people)
        toggle_all = self._toggle_btn(
            all_on,
            label_on="ALL ON — click to disable all",
            label_off="ALL OFF — click to enable all",
            icon_size=22, font_size=13, width=None,
            on_click=lambda _: self._toggle_all(people),
            tooltip="Toggle all preferences on/off",
        )
        # Extra padding for the big button
        toggle_all.padding = ft.padding.symmetric(horizontal=_s(20), vertical=_s(10))
        toggle_all.border_radius = 8

        self._ct.controls = [
            ft.Row([
                toggle_all,
                ft.Text("Drag tasks to reorder (top = most preferred, cost 0).",
                        size=_s(12), italic=True, color="#546E7A", expand=True),
            ], spacing=_s(12), vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row(
                [self._build_person_column(p, tasks) for p in people],
                spacing=_s(12), scroll=ft.ScrollMode.AUTO,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.Divider(height=_s(1)),
            ft.Text("Cost Matrix (pref_cost)", size=_s(12),
                    weight=ft.FontWeight.BOLD, color="#37474F"),
            self._build_cost_matrix(people, tasks),
        ]
        self._safe_update()

    # ── Person column ────────────────────────────────────────────────

    def _build_person_column(self, person: str, tasks: list) -> ft.Container:
        s = self.state
        enabled = s.pref_enabled_st.get(person, True)
        order = s.pref_order_st.get(person, list(tasks))

        color_map = {t: TASK_COLORS[i % len(TASK_COLORS)]
                     for i, t in enumerate(tasks)}

        items = []
        for rank, t in enumerate(order):
            bg, fg = color_map.get(t, ("#B0BEC5", "#000000"))
            op = 1.0 if enabled else 0.4
            items.append(ft.Container(
                key=f"{person}_{rank}",
                content=ft.Row([
                    ft.Icon(ft.Icons.DRAG_INDICATOR, size=_s(14), color="#B0BEC5"),
                    ft.Container(
                        content=ft.Text(str(rank), size=_s(9), color=fg,
                                        weight=ft.FontWeight.BOLD,
                                        text_align=ft.TextAlign.CENTER),
                        width=_s(20), height=_s(18), border_radius=3,
                        bgcolor=bg, opacity=op, alignment=ft.alignment.center),
                    ft.Text(t, size=_s(10),
                            color=fg if enabled else "#9E9E9E",
                            weight=ft.FontWeight.BOLD, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, width=_s(90)),
                ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=bg if enabled else "#E0E0E0", opacity=op,
                padding=ft.padding.symmetric(horizontal=_s(4), vertical=_s(3)),
                border_radius=4,
                border=ft.border.only(bottom=ft.border.BorderSide(1, "#CFD8DC")),
            ))

        p_color = s.person_colors.get(person, "#37474F")

        return ft.Container(
            content=ft.Column([
                ft.Text(person, size=_s(12), weight=ft.FontWeight.BOLD,
                        color=p_color, no_wrap=True),
                self._toggle_btn(enabled,
                                 on_click=lambda _, _p=person: self._on_toggle(_p)),
                ft.Container(
                    content=ft.ReorderableListView(
                        controls=items,
                        on_reorder=lambda e, _p=person: self._on_reorder(e, _p)),
                    height=_s(max(120, len(order) * 30)), width=_s(170)),
            ], spacing=_s(4), horizontal_alignment=ft.CrossAxisAlignment.START),
            width=_s(180), padding=_s(6),
            border=ft.border.all(1, "#CFD8DC"), border_radius=6, bgcolor="#FAFAFA",
        )

    # ── Cost matrix table ────────────────────────────────────────────

    def _build_cost_matrix(self, people: list, tasks: list) -> ft.Container:
        pref_cost = self.get_pref_cost(people, tasks)
        n = len(tasks)

        header = [ft.DataColumn(ft.Text("Person", size=_s(10),
                                        weight=ft.FontWeight.BOLD))]
        header += [ft.DataColumn(ft.Text(t, size=_s(9), weight=ft.FontWeight.BOLD))
                   for t in tasks]

        rows = []
        for p in people:
            enabled = self.state.pref_enabled_st.get(p, True)
            p_color = self.state.person_colors.get(p, "#37474F")
            cells = [ft.DataCell(ft.Text(p, size=_s(9), color=p_color,
                                         weight=ft.FontWeight.BOLD))]
            for t in tasks:
                val = pref_cost[(p, t)]
                bg, fg = self._cost_style(val, enabled, n)
                cells.append(ft.DataCell(ft.Container(
                    content=ft.Text(str(val), size=_s(9), color=fg,
                                    weight=ft.FontWeight.BOLD,
                                    text_align=ft.TextAlign.CENTER),
                    bgcolor=bg, border_radius=3,
                    width=_s(30), height=_s(20), alignment=ft.alignment.center)))
            rows.append(ft.DataRow(cells=cells))

        return ft.Container(
            content=ft.Row([ft.DataTable(
                columns=header, rows=rows, column_spacing=_s(6),
                data_row_max_height=_s(28), heading_row_height=_s(30),
                horizontal_lines=ft.border.BorderSide(1, "#E0E0E0"),
            )], scroll=ft.ScrollMode.AUTO),
            padding=_s(4))

    # ── Callbacks ────────────────────────────────────────────────────

    def _toggle_all(self, people: list):
        new_val = not all(self.state.pref_enabled_st.get(p, True) for p in people)
        for p in people:
            self.state.pref_enabled_st[p] = new_val
        self.build()

    def _on_toggle(self, person: str):
        self.state.pref_enabled_st[person] = not self.state.pref_enabled_st.get(person, True)
        self.build()

    def _on_reorder(self, e: ft.OnReorderEvent, person: str):
        order = self.state.pref_order_st.get(person, [])
        if not order or e.old_index == e.new_index or not (0 <= e.old_index < len(order)):
            return
        item = order.pop(e.old_index)
        order.insert(max(0, min(e.new_index, len(order))), item)
        self.build()

    # ── Public: pref_cost dict for solver ────────────────────────────

    def get_pref_cost(self, people: list, tasks: list) -> dict:
        s = self.state
        tasks_set = set(tasks)
        result = {}
        for p in people:
            enabled = s.pref_enabled_st.get(p, True)
            order = s.pref_order_st.get(p, list(tasks))
            rank_map = {t: rank for rank, t in enumerate(order) if t in tasks_set}
            for t in tasks:
                result[(p, t)] = rank_map.get(t, len(tasks)) if enabled else 0
        return result