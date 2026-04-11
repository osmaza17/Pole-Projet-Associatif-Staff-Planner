import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


class AvailabilityTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._cell_map: dict = {}
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)
        # Estado del gesto activo (patrón mockup.py)
        self._gesture = {"origin": None, "visited": set()}
        # Layout del grid (rellenado en build)
        self._grid_people: list = []
        self._grid_hours: list = []
        self._grid_day: str = ""

    # ── Primitiva de bajo nivel: set + repintado, sin notificar ─────
    def _set_cell(self, p, h, j, nv: int):
        """Asigna nv a la celda y refresca el widget. NO llama a _on_solve_blocked_update."""
        self.state.avail_st[(p, h, j)] = nv
        cell = self._cell_map.get((p, h, j))
        if cell is not None:
            cell.content.value = UIHelpers._AVAIL_LBL[nv]
            cell.bgcolor = UIHelpers._AVAIL_CLR[nv]
            cell.update()

    # ── Ciclo de vida del gesto (idéntico a mockup.py) ──────────────
    def _advance(self, p, h, j):
        cur = self.state.avail_st.get((p, h, j), 1)
        nv = UIHelpers._AVAIL_NEXT[cur]
        self._set_cell(p, h, j, nv)
        self._on_solve_blocked_update()

    def _begin_gesture(self, p, h, j):
        s = self.state
        self._gesture["origin"] = s.avail_st.get((p, h, j), 1)
        self._gesture["visited"] = {(p, h, j)}
        self._advance(p, h, j)

    def _continue_gesture(self, p, h, j):
        s = self.state
        g = self._gesture
        if g["origin"] is None:
            return
        if (p, h, j) in g["visited"]:
            return
        if s.avail_st.get((p, h, j), 1) != g["origin"]:
            return
        g["visited"].add((p, h, j))
        self._advance(p, h, j)

    def _end_gesture(self):
        self._gesture["origin"] = None
        self._gesture["visited"] = set()

    # ── Traductor de coordenadas locales -> (p, h, j) ───────────────
    def _coords_to_cell(self, x: float, y: float):
        people = self._grid_people
        hours  = self._grid_hours
        j      = self._grid_day
        if not people or not hours:
            return None

        W_CELL = UIHelpers.W_CELL
        H_BTN  = UIHelpers.H_BTN
        HGAP   = 0
        VGAP   = 0

        if x < 0:
            return None
        stride_x = W_CELL + HGAP
        c, c_off = divmod(int(x), stride_x)
        if c_off >= W_CELL or c < 0 or c >= len(hours):
            return None

        if y < 0:
            return None
        stride_y = H_BTN + VGAP
        r, r_off = divmod(int(y), stride_y)
        if r_off >= H_BTN or r < 0 or r >= len(people):
            return None

        return (people[r], hours[c], j)

    def build(self):
        s = self.state
        people, _, hours, days = s.dims()
        if not days:
            return
        if s.avail_filter not in days:
            s.avail_filter = days[0]

        self._cell_map.clear()
        self._end_gesture()

        def _on_day(v):
            s.avail_filter = v
            self.build()

        def _reset(e):
            s.avail_st.clear()
            self._on_solve_blocked_update()
            self.build()

        j       = s.avail_filter
        day_hrs = hours[j]

        self._grid_people = list(people)
        self._grid_hours  = list(day_hrs)
        self._grid_day    = j

        def _next_from(ref_key):
            return UIHelpers._AVAIL_NEXT[s.avail_st.get(ref_key, 1)]

        def _make_col_toggle(_h):
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __h=_h):
                nv = _next_from((people[0], __h, j))
                for pp in people:
                    self._set_cell(pp, __h, j, nv)
                self._on_solve_blocked_update()
            btn.on_click = _click
            return btn

        def _make_row_toggle(_p):
            btn = UIHelpers.make_rc_btn("row")
            def _click(e, __p=_p):
                nv = _next_from((__p, day_hrs[0], j))
                for h in day_hrs:
                    self._set_cell(__p, h, j, nv)
                self._on_solve_blocked_update()
            btn.on_click = _click
            return btn

        # ── Offset de cabecera: W_LBL (nombres) + W_CELL (row toggle) ──
        header_offset = UIHelpers.W_LBL + UIHelpers.W_CELL

        buf = [
            UIHelpers.make_tab_bar(days, j, _on_day),
            UIHelpers.make_reset_btn("Reset to Default", _reset),
            ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)),
            # Cabecera de horas
            ft.Row(
                [ft.Container(width=header_offset)] +
                [ft.Container(ft.Text(h, size=_s(9), no_wrap=True,
                                      overflow=ft.TextOverflow.CLIP),
                              width=UIHelpers.W_CELL,
                              clip_behavior=ft.ClipBehavior.HARD_EDGE)
                 for h in day_hrs],
                spacing=2, wrap=False),
            # Fila de toggles de columna
            ft.Row(
                [ft.Container(width=header_offset)] +
                [_make_col_toggle(h) for h in day_hrs],
                spacing=2, wrap=False),
        ]

        # Filas de personas
        person_rows = []
        name_widgets = []
        row_toggle_widgets = []
        for p in people:
            cell_widgets = []
            for h in day_hrs:
                toggle = UIHelpers.make_avail_toggle(s.avail_st, (p, h, j))
                toggle.on_click = None
                self._cell_map[(p, h, j)] = toggle
                cell_widgets.append(toggle)

            person_rows.append(ft.Row(cell_widgets, spacing=2, wrap=False))
            name_widgets.append(
                ft.Container(
                    content=UIHelpers.plbl(p, s.person_colors.get(p)),
                    height=UIHelpers.H_BTN,
                    alignment=ft.alignment.center_left,
                )
            )
            row_toggle_widgets.append(_make_row_toggle(p))

        # ── ORDEN: nombres | row toggles | grid ──
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
        buf.append(ft.Divider())

        self.set_matrix_columns(len(day_hrs))
        self._ct.controls = buf
        self.page.update()