import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


class SkillsTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._cell_map: dict = {}
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)
        self._gesture = {"origin": None, "visited": set()}
        self._grid_people: list = []
        self._grid_tasks: list = []

    # ── Bajo nivel: set + repintado ─────────────────────────────────
    def _set_cell(self, p, t, nv: int):
        self.state.skills_st[(p, t)] = nv
        cell = self._cell_map.get((p, t))
        if cell is not None:
            cell.content.value = str(nv)
            cell.bgcolor = UIHelpers._AVAIL_CLR[nv]
            cell.update()

    # ── Gesto: binario 0↔1 ──────────────────────────────────────────
    def _advance(self, p, t):
        cur = self.state.skills_st.get((p, t), 1)
        nv = 1 - cur
        self._set_cell(p, t, nv)
        self._on_solve_blocked_update()

    def _begin_gesture(self, p, t):
        self._gesture["origin"] = self.state.skills_st.get((p, t), 1)
        self._gesture["visited"] = {(p, t)}
        self._advance(p, t)

    def _continue_gesture(self, p, t):
        g = self._gesture
        if g["origin"] is None:
            return
        if (p, t) in g["visited"]:
            return
        if self.state.skills_st.get((p, t), 1) != g["origin"]:
            return
        g["visited"].add((p, t))
        self._advance(p, t)

    def _end_gesture(self):
        self._gesture["origin"] = None
        self._gesture["visited"] = set()

    # ── Coordenadas locales -> (p, t) ────────────────────────────────
    def _coords_to_cell(self, x: float, y: float):
        people = self._grid_people
        tasks = self._grid_tasks
        if not people or not tasks:
            return None

        W_CELL = UIHelpers.W_CELL
        H_BTN = UIHelpers.H_BTN

        if x < 0:
            return None
        stride_x = W_CELL
        c, c_off = divmod(int(x), stride_x)
        if c_off >= W_CELL or c < 0 or c >= len(tasks):
            return None

        if y < 0:
            return None
        stride_y = H_BTN
        r, r_off = divmod(int(y), stride_y)
        if r_off >= H_BTN or r < 0 or r >= len(people):
            return None

        return (people[r], tasks[c])

    def _notify_and_rebuild(self):
        self.state.invalidate_cache()
        self._on_solve_blocked_update()
        self.build()

    def _bulk_set(self, keys, value):
        for k in keys:
            self.state.skills_st[k] = value

    def build(self):
        s = self.state
        if not s.needs_rebuild("skills"):
            return
        people, tasks, _, _ = s.dims()

        self._cell_map.clear()
        self._end_gesture()
        self._grid_people = list(people)
        self._grid_tasks = list(tasks)

        def on_reset(_):
            s.skills_st.clear()
            self._notify_and_rebuild()

        def make_col_toggle(task):
            btn = UIHelpers.make_rc_btn("col")
            def click(_, _t=task):
                nv = 1 - s.skills_st.get((people[0], _t), 1)
                self._bulk_set(((p, _t) for p in people), nv)
                self._notify_and_rebuild()
            btn.on_click = click
            return btn

        def make_row_toggle(person):
            btn = UIHelpers.make_rc_btn("row")
            def click(_, _p=person):
                nv = 1 - s.skills_st.get((_p, tasks[0]), 1)
                self._bulk_set(((_p, t) for t in tasks), nv)
                self._notify_and_rebuild()
            btn.on_click = click
            return btn

        W = UIHelpers.W_CELL
        header_offset = UIHelpers.W_LBL + W

        task_headers = [
            ft.Container(
                ft.Text(t, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                width=W, clip_behavior=ft.ClipBehavior.HARD_EDGE)
            for t in tasks
        ]

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=_s(16)),
            UIHelpers.make_reset_btn("Reset Skills", on_reset),
            ft.Row([ft.Container(width=header_offset)] + task_headers, spacing=2, wrap=False),
            ft.Row([ft.Container(width=header_offset)] + [make_col_toggle(t) for t in tasks], spacing=2, wrap=False),
        ]

        person_rows = []
        name_widgets = []
        row_toggle_widgets = []
        for p in people:
            cell_widgets = []
            for t in tasks:
                toggle = UIHelpers.make_toggle(s.skills_st, (p, t), 1)
                toggle.on_click = None
                self._cell_map[(p, t)] = toggle
                cell_widgets.append(toggle)

            person_rows.append(ft.Row(cell_widgets, spacing=2, wrap=False))
            name_widgets.append(
                ft.Container(
                    content=UIHelpers.plbl(p, s.person_colors.get(p)),
                    height=UIHelpers.H_BTN,
                    alignment=ft.alignment.center_left,
                )
            )
            row_toggle_widgets.append(make_row_toggle(p))

        names_col = ft.Column(name_widgets, spacing=2, tight=True)
        row_toggles_col = ft.Column(row_toggle_widgets, spacing=2, tight=True)
        grid_column = ft.Column(person_rows, spacing=2, tight=True)

        def _on_pan_start(e):
            rc = self._coords_to_cell(e.local_x, e.local_y)
            if rc is None:
                return
            if self._gesture["origin"] is None:
                self._begin_gesture(*rc)
            else:
                self._continue_gesture(*rc)

        def _on_pan_update(e):
            rc = self._coords_to_cell(e.local_x, e.local_y)
            if rc is not None:
                self._continue_gesture(*rc)

        def _on_tap_down(e):
            rc = self._coords_to_cell(e.local_x, e.local_y)
            if rc is None:
                return
            self._begin_gesture(*rc)

        grid_area = ft.GestureDetector(
            content=grid_column,
            drag_interval=10,
            on_tap_down=_on_tap_down,
            on_pan_start=_on_pan_start,
            on_pan_update=_on_pan_update,
            on_pan_end=lambda e: self._end_gesture(),
        )

        buf.append(ft.Row(
            [names_col, row_toggles_col, grid_area],
            spacing=2, wrap=False))

        self.set_matrix_columns(len(tasks))
        self._ct.controls = buf
        self.page.update()