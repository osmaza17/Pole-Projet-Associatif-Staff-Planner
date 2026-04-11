import random
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab

_LBL  = {0: "~", 1: "+", -1: "-"}
_CLR  = {0: ft.Colors.GREY_400, 1: ft.Colors.GREEN_700, -1: ft.Colors.RED_700}
_NEXT = {0: 1, 1: -1, -1: 0}


class SocialTab(BaseTab):

    def _rebuild(self):
        self.state.invalidate_cache()
        self.build()

    def _cycle_pairs(self, pairs):
        """Cycle all pairs to the next value (based on first pair's current)."""
        if not pairs:
            return
        nv = _NEXT[self.state.social_st.get(pairs[0], 0)]
        for k in pairs:
            self.state.social_st[k] = nv
        self._rebuild()

    def build(self):
        s = self.state
        if not s.needs_rebuild("social"):
            return
        people, _, _, _ = s.dims()

        if len(people) < 2:
            self._ct.controls = []
            self.page.update()
            return

        def on_reset(_):
            s.social_st.clear()
            s.hard_enemies = False
            self._rebuild()

        def on_random(_):
            for i, p1 in enumerate(people):
                for p2 in people[i + 1:]:
                    s.social_st[(p1, p2)] = random.choice([-1, 0, 1])
            self._rebuild()

        sw_hard = ft.Switch(label="Enemies: Hard Constraint", value=s.hard_enemies,
                            on_change=lambda e: setattr(s, "hard_enemies", e.control.value))

        col_people = people[1:]
        pidx = {p: i for i, p in enumerate(people)}
        W = UIHelpers.W_CELL
        spacer = [ft.Container(width=UIHelpers.W_LBL), ft.Container(width=W)]

        col_headers = [
            ft.Container(ft.Text(p, size=_s(9), no_wrap=True,
                                 overflow=ft.TextOverflow.CLIP),
                         width=W, clip_behavior=ft.ClipBehavior.HARD_EDGE)
            for p in col_people
        ]

        def col_toggle(p2):
            btn = UIHelpers.make_rc_btn("col")
            j = pidx[p2]
            btn.on_click = lambda _, _p2=p2, _j=j: self._cycle_pairs(
                [(people[i], _p2) for i in range(_j)])
            return btn

        def row_toggle(p1):
            btn = UIHelpers.make_rc_btn("row")
            i = pidx[p1]
            btn.on_click = lambda _, _p1=p1, _i=i: self._cycle_pairs(
                [(_p1, people[j]) for j in range(_i + 1, len(people))])
            return btn

        def make_cell(p1, p2):
            k = (p1, p2)
            s.social_st.setdefault(k, 0)
            v = s.social_st[k]

            def on_click(e, _k=k):
                nv = _NEXT[s.social_st[_k]]
                s.social_st[_k] = nv
                e.control.content.value = _LBL[nv]
                e.control.bgcolor = _CLR[nv]
                e.control.update()

            return ft.Container(
                content=ft.Text(_LBL[v], color=ft.Colors.WHITE,
                                size=_s(12), weight=ft.FontWeight.BOLD),
                width=W, height=UIHelpers.H_BTN, bgcolor=_CLR[v],
                alignment=ft.alignment.center, border_radius=4,
                on_click=on_click)

        buf = [
            ft.Row([
                UIHelpers.make_reset_btn("Reset to Default", on_reset),
                ft.Container(
                    ft.Text("Random Social", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                    on_click=on_random, width=_s(150), alignment=ft.alignment.center),
            ], spacing=_s(10)),
            ft.Row([sw_hard]),
            ft.Row(spacer + col_headers, spacing=2, wrap=False),
            ft.Row(spacer + [col_toggle(p) for p in col_people], spacing=2, wrap=False),
        ]

        for i, p1 in enumerate(people[:-1]):  # last person has no columns
            cells = [make_cell(p1, p2) if pidx[p2] > i else ft.Container(width=W)
                     for p2 in col_people]
            buf.append(ft.Row(
                [UIHelpers.plbl(p1, s.person_colors.get(p1)), row_toggle(p1)] + cells,
                spacing=2, wrap=False))

        self.set_matrix_columns(len(col_people))
        self._ct.controls = buf
        self.page.update()