import random
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


class SkillsTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)

    def build(self):
        s = self.state
        if not s.needs_rebuild("skills"):
            return
        people, tasks, _, _ = s.dims()

        def _reset(e):
            s.skills_st.clear()
            s.invalidate_cache()
            self._on_solve_blocked_update()
            self.build()

        def _rand(e):
            for p in people:
                for t in tasks:
                    s.skills_st[(p, t)] = random.choice([0, 1])
            s.invalidate_cache()
            self._on_solve_blocked_update()
            self.build()

        def _make_col_toggle(_t):
            btn = UIHelpers.make_rc_btn("col")
            def _click(e, __t=_t):
                fv = s.skills_st.get((people[0], __t), 1)
                nv = 1 - fv
                for pp in people:
                    s.skills_st[(pp, __t)] = nv
                s.invalidate_cache()
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        def _make_row_toggle(_p):
            btn = UIHelpers.make_rc_btn("row")
            def _click(e, __p=_p):
                fv = s.skills_st.get((__p, tasks[0]), 1)
                nv = 1 - fv
                for tt in tasks:
                    s.skills_st[(__p, tt)] = nv
                s.invalidate_cache()
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=_s(16)),
            ft.Row([
                UIHelpers.make_reset_btn("Reset Skills", _reset),
                ft.Container(ft.Text("Random Skills", color=ft.Colors.WHITE,
                                     weight=ft.FontWeight.BOLD, size=_s(12)),
                             bgcolor=ft.Colors.PURPLE_400, padding=_s(8),
                             border_radius=4, on_click=_rand, width=_s(150),
                             alignment=ft.alignment.center),
            ], spacing=_s(10)),
            ft.Row(
                [ft.Container(width=UIHelpers.W_LBL),
                 ft.Container(width=UIHelpers.W_CELL)] +
                [ft.Container(ft.Text(t, size=_s(9), no_wrap=True,
                                      overflow=ft.TextOverflow.CLIP),
                              width=UIHelpers.W_CELL,
                              clip_behavior=ft.ClipBehavior.HARD_EDGE)
                 for t in tasks],
                spacing=2, wrap=False),
            ft.Row(
                [ft.Container(width=UIHelpers.W_LBL),
                 ft.Container(width=UIHelpers.W_CELL)] +
                [_make_col_toggle(t) for t in tasks],
                spacing=2, wrap=False),
        ]

        for p in people:
            row_toggles = []
            for t in tasks:
                toggle = UIHelpers.make_toggle(s.skills_st, (p, t), 1)
                # Wrap on_click to also fire the revalidation callback
                _orig_click = toggle.on_click
                def _wrapped(e, _orig=_orig_click):
                    _orig(e)
                    self._on_solve_blocked_update()
                toggle.on_click = _wrapped
                row_toggles.append(toggle)

            buf.append(ft.Row(
                [UIHelpers.plbl(p, s.person_colors.get(p)), _make_row_toggle(p)] +
                row_toggles,
                spacing=2))

        self.set_matrix_columns(len(tasks))
        self._ct.controls = buf
        self.page.update()