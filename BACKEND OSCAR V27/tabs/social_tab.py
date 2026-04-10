import random
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


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
        col_people = people[1:]

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

        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [ft.Container(ft.Text(p, size=_s(9), no_wrap=True,
                                  overflow=ft.TextOverflow.CLIP),
                          width=UIHelpers.W_CELL,
                          clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for p in col_people],
            spacing=2, wrap=False))
        buf.append(ft.Row(
            [ft.Container(width=UIHelpers.W_LBL),
             ft.Container(width=UIHelpers.W_CELL)] +
            [_make_col_toggle(p2) for p2 in col_people],
            spacing=2, wrap=False))

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
                        alignment=ft.alignment.center, border_radius=4,
                        on_click=_click)
                    cells.append(btn)
                else:
                    cells.append(ft.Container(width=UIHelpers.W_CELL))

            if cells:
                buf.append(ft.Row(
                    [UIHelpers.plbl(p1, s.person_colors.get(p1)), _make_row_toggle(p1)] + cells,
                    spacing=2, wrap=False))

        self.set_matrix_columns(len(col_people))
        self._ct.controls = buf
        self.page.update()