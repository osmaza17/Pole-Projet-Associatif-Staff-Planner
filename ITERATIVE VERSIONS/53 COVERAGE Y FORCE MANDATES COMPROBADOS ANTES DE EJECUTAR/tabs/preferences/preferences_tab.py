"""
preferences_tab.py
==================
Per-person drag-and-drop task preference ranking using ReorderableListView.
Top = most preferred (cost 0), bottom = least preferred (cost N-1).
Features: search/filter people, reset order, pagination (max PAGE_SIZE people
          visible at once).
"""

import flet as ft
from constants import _s, TASK_COLORS
from state import AppState

PAGE_SIZE = 30


class PreferencesTab:

    def __init__(self, state: AppState, page: ft.Page):
        self.state = state
        self.page = page
        self._filter_text = ""
        self._page_idx = 0
        self._ct = ft.Column(spacing=_s(8), expand=True, scroll=ft.ScrollMode.AUTO)
        self._container = ft.Container(content=self._ct, expand=True, padding=_s(8))

        self._search_field = ft.TextField(
            value="",
            hint_text="Filter people…",
            prefix_icon=ft.Icons.SEARCH,
            on_change=self._on_filter_change,
            width=_s(220), height=_s(38),
            text_size=_s(12), hint_style=ft.TextStyle(size=_s(11)),
            border_radius=6, dense=True,
            content_padding=ft.padding.only(left=_s(8)),
        )
        self._match_label = ft.Text(size=_s(10), italic=True, color="#78909C")
        self._people_row = ft.Row(
            spacing=_s(12), run_spacing=_s(12), wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

        self._page_label = ft.Text(size=_s(11), weight=ft.FontWeight.BOLD,
                                   color="#37474F")
        self._btn_prev = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT, icon_size=_s(20),
            tooltip="Previous page",
            on_click=lambda _: self._change_page(-1),
            style=ft.ButtonStyle(padding=ft.padding.all(_s(4)),
                                 shape=ft.RoundedRectangleBorder(radius=6)),
        )
        self._btn_next = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT, icon_size=_s(20),
            tooltip="Next page",
            on_click=lambda _: self._change_page(1),
            style=ft.ButtonStyle(padding=ft.padding.all(_s(4)),
                                 shape=ft.RoundedRectangleBorder(radius=6)),
        )
        self._pagination_row = ft.Row(
            [self._btn_prev, self._page_label, self._btn_next],
            spacing=_s(4),
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def get_container(self) -> ft.Container:
        return self._container

    # ── Helpers ───────────────────────────────────────────────────────

    def _safe_update(self):
        try:
            self._ct.update()
        except Exception:
            pass

    def _ensure_state(self, people, tasks):
        """Initialise / sync pref_order on state."""
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
    def _icon_btn(icon, tooltip, on_click, color="#546E7A", size=14):
        return ft.IconButton(
            icon=icon, icon_size=_s(size), icon_color=color,
            tooltip=tooltip, on_click=on_click,
            style=ft.ButtonStyle(padding=ft.padding.all(2)),
        )

    # ── Pagination helpers ───────────────────────────────────────────

    def _total_pages(self, visible_count: int) -> int:
        if visible_count == 0:
            return 1
        return (visible_count + PAGE_SIZE - 1) // PAGE_SIZE

    def _clamp_page(self, visible_count: int):
        max_page = self._total_pages(visible_count) - 1
        self._page_idx = max(0, min(self._page_idx, max_page))

    def _change_page(self, delta: int):
        people, tasks, _, _ = self.state.dims()
        if not people or not tasks:
            return
        self._ensure_state(people, tasks)
        visible = self._visible_people(people)
        self._page_idx += delta
        self._clamp_page(len(visible))
        self._refresh_people_row(people, tasks)
        try:
            self._people_row.update()
            self._pagination_row.update()
            self._match_label.update()
        except Exception:
            pass

    # ── Filtered people list ─────────────────────────────────────────

    def _visible_people(self, people: list) -> list:
        filt = self._filter_text.strip().lower()
        return [p for p in people if filt in p.lower()] if filt else list(people)

    def _refresh_people_row(self, people: list, tasks: list):
        visible = self._visible_people(people)
        self._clamp_page(len(visible))

        start = self._page_idx * PAGE_SIZE
        end = start + PAGE_SIZE
        page_people = visible[start:end]

        self._people_row.controls = [
            self._build_person_column(p, tasks) for p in page_people
        ]

        total_pages = self._total_pages(len(visible))
        self._page_label.value = f"Page {self._page_idx + 1} / {total_pages}"
        self._btn_prev.disabled = (self._page_idx <= 0)
        self._btn_next.disabled = (self._page_idx >= total_pages - 1)
        self._pagination_row.visible = (total_pages > 1)

        filt = self._filter_text.strip()
        if filt:
            self._match_label.value = (
                f"Showing {len(page_people)} of {len(visible)} matches "
                f"({len(people)} total)"
            )
            self._match_label.visible = True
        elif total_pages > 1:
            self._match_label.value = (
                f"Showing {start + 1}\u2013{start + len(page_people)} of {len(visible)}"
            )
            self._match_label.visible = True
        else:
            self._match_label.value = ""
            self._match_label.visible = False

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
        self._refresh_people_row(people, tasks)

        self._ct.controls = [
            ft.Row(
                [self._search_field, self._match_label],
                spacing=_s(12),
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=True, run_spacing=_s(8),
            ),
            ft.Row([
                ft.Text(
                    "Drag tasks to reorder (top = most preferred, cost 0).",
                    size=_s(12), italic=True, color="#546E7A",
                ),
                self._pagination_row,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self._people_row,
        ]
        self._safe_update()

    # ── Person column ────────────────────────────────────────────────

    def _build_person_column(self, person: str, tasks: list) -> ft.Container:
        s = self.state
        order = s.pref_order_st.get(person, list(tasks))

        color_map = {t: TASK_COLORS[i % len(TASK_COLORS)]
                     for i, t in enumerate(tasks)}

        items = []
        for rank, t in enumerate(order):
            bg, fg = color_map.get(t, ("#B0BEC5", "#000000"))
            items.append(ft.Container(
                key=f"{person}_{rank}",
                content=ft.Row([
                    ft.Icon(ft.Icons.DRAG_INDICATOR, size=_s(14), color="#B0BEC5"),
                    ft.Container(
                        content=ft.Text(str(rank), size=_s(9), color=fg,
                                        weight=ft.FontWeight.BOLD,
                                        text_align=ft.TextAlign.CENTER),
                        width=_s(20), height=_s(18), border_radius=3,
                        bgcolor=bg, alignment=ft.alignment.center),
                    ft.Text(t, size=_s(11), color=fg,
                            weight=ft.FontWeight.BOLD, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS, width=_s(130)),
                ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=bg,
                padding=ft.padding.symmetric(horizontal=_s(4), vertical=_s(3)),
                border_radius=4,
                border=ft.border.only(bottom=ft.border.BorderSide(1, "#CFD8DC")),
            ))

        p_color = s.person_colors.get(person, "#37474F")

        actions = ft.Row([
            self._icon_btn(ft.Icons.RESTART_ALT, "Reset order",
                           lambda _, _p=person: self._on_reset(_p, tasks),
                           color="#EF6C00", size=16),
        ], spacing=_s(2), vertical_alignment=ft.CrossAxisAlignment.CENTER)

        return ft.Container(
            content=ft.Column([
                ft.Text(person, size=_s(12), weight=ft.FontWeight.BOLD,
                        color=p_color, no_wrap=True),
                actions,
                ft.Container(
                    content=ft.ReorderableListView(
                        controls=items,
                        on_reorder=lambda e, _p=person: self._on_reorder(e, _p)),
                    height=_s(max(120, len(order) * 30)), width=_s(210)),
            ], spacing=_s(4), horizontal_alignment=ft.CrossAxisAlignment.START),
            width=_s(220), padding=_s(6),
            border=ft.border.all(1, "#CFD8DC"), border_radius=6, bgcolor="#FAFAFA",
        )

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_reorder(self, e, person: str):
        order = self.state.pref_order_st.get(person, [])
        if (not order or e.old_index == e.new_index
                or not (0 <= e.old_index < len(order))):
            return
        item = order.pop(e.old_index)
        order.insert(max(0, min(e.new_index, len(order))), item)
        self.build()

    def _on_reset(self, person: str, tasks: list):
        self.state.pref_order_st[person] = list(tasks)
        self.build()

    def _on_filter_change(self, e):
        self._filter_text = e.control.value or ""
        self._page_idx = 0
        people, tasks, _, _ = self.state.dims()
        if people and tasks:
            self._ensure_state(people, tasks)
            self._refresh_people_row(people, tasks)
            try:
                self._people_row.update()
                self._pagination_row.update()
                self._match_label.update()
            except Exception:
                pass

    def _on_filter_clear(self, _):
        self._filter_text = ""
        self._page_idx = 0
        self._search_field.value = ""
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