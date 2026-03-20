import random
import flet as ft
from constants import _s, AVAIL_COLOR, UNAVAIL_COLOR, EMERG_COLOR
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


class AvailabilityTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._cell_map: dict = {}
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)

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
            self._on_solve_blocked_update()
            self.build()

        def _rand(e):
            for p in people:
                for j in days:
                    for h in hours[j]:
                        s.avail_st[(p, h, j)] = random.choice([0, 1, 2])
            self._on_solve_blocked_update()
            self.build()

        j       = s.avail_filter
        day_hrs = hours[j]

        buf = [
            UIHelpers.make_tab_bar(days, j, _on_day),
            ft.Row([
                UIHelpers.make_reset_btn("Reset to Default", _reset),
                ft.Container(
                    ft.Text("Random Avail (All Days)", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                    on_click=_rand, width=_s(180), alignment=ft.alignment.center),
            ], spacing=_s(20)),
        ]

        buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)))
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 4)] +
            [ft.Container(ft.Text(h, size=_s(9), no_wrap=True,
                                  overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL,
                          clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

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
                self._on_solve_blocked_update()
            btn.on_click = _click
            return btn

        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 4)] +
            [_make_col_toggle(h) for h in day_hrs],
            spacing=2, wrap=False))

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
                    self._on_solve_blocked_update()
                btn.on_click = _click
                return btn

            cell_widgets = []
            for h in day_hrs:
                toggle = UIHelpers.make_avail_toggle(s.avail_st, (p, h, j))
                # Wrap the toggle's on_click to also fire the revalidation callback
                _orig_click = toggle.on_click
                def _wrapped(e, _orig=_orig_click):
                    _orig(e)
                    self._on_solve_blocked_update()
                toggle.on_click = _wrapped
                self._cell_map[(p, h, j)] = toggle
                cell_widgets.append(toggle)

            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.person_colors.get(p)), _row_toggle(p, j)] +
                cell_widgets,
                spacing=2, wrap=False))

        buf.append(ft.Divider())
        self.set_matrix_columns(len(day_hrs))
        self._ct.controls = buf
        self.page.update()