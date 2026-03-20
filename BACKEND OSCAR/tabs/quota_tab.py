import random
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab

_EMPTY_BG  = ft.Colors.with_opacity(0.18, ft.Colors.BLACK)
_FILLED_BG = ft.Colors.with_opacity(0.08, ft.Colors.BLUE)


class QuotaTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        super().__init__(state, page)
        self._on_solve_blocked_update = on_solve_blocked_update

        self._top_ct = ft.Column(spacing=_s(6), tight=True)

        self._matrix_col = ft.Column(
            spacing=2, tight=True, scroll=ft.ScrollMode.AUTO)
        self._matrix_row = ft.Row(
            controls=[self._matrix_col],
            scroll=ft.ScrollMode.AUTO, expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START)

        self._ct.controls = [
            self._top_ct,
            ft.Divider(height=_s(6)),
            self._matrix_row,
        ]
        self._ct.expand = True

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
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=_s(12), visible=False)

        # ── Fixed top section ─────────────────────────────────────────
        self._top_ct.controls = [
            ft.Text("Minimum Quota Matrix (per Day)",
                    weight=ft.FontWeight.BOLD, size=_s(16)),
            # ── Tab bar de días ──────────────────────────────────────
            UIHelpers.make_tab_bar(days, j, _on_day),
            # ── Botones de acción ────────────────────────────────────
            ft.Row([
                UIHelpers.make_reset_btn("Reset All Days", _reset),
                ft.Container(
                    ft.Text("Set All to 0", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.ORANGE_700, padding=_s(8),
                    border_radius=4, on_click=_zero, width=_s(150),
                    alignment=ft.alignment.center),
                ft.Container(
                    ft.Text("Random Quota (All Days)", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.PURPLE_400, padding=_s(8),
                    border_radius=4, on_click=_rand, width=_s(180),
                    alignment=ft.alignment.center),
            ], spacing=_s(20)),
            err_txt,
            ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)),
        ]

        # ── Matrix rows ───────────────────────────────────────────────
        matrix_buf = []
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

        matrix_buf.append(ft.Row(
            [_spacer] + [_col_btn(t, +1) for t in tasks],
            spacing=2, wrap=False))
        matrix_buf.append(ft.Row(
            [_spacer] +
            [ft.Container(
                ft.Text(t, size=_s(9), no_wrap=True,
                        overflow=ft.TextOverflow.CLIP),
                width=UIHelpers.W_CELL,
                clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for t in tasks],
            spacing=2, wrap=False))

        for p in people:
            cells = []
            for t in tasks:
                k = (p, t, j)
                s.quota_st.setdefault(k, "")
                is_ok = UIHelpers.validate_nonneg_int(s.quota_st[k])
                val = s.quota_st[k]
                is_empty = not val or not val.strip()
                tf = ft.TextField(
                    value=val, width=UIHelpers.W_CELL,
                    height=UIHelpers.H_TF, text_size=_s(11), data=k,
                    content_padding=ft.padding.all(_s(2)),
                    bgcolor=_EMPTY_BG if is_empty else _FILLED_BG,
                    border_color=ft.Colors.RED_400 if not is_ok else None)

                def _ch(e, _k=k, _err=err_txt):
                    s.quota_st[_k] = e.control.value
                    is_empty_now = not e.control.value or not e.control.value.strip()
                    if UIHelpers.validate_nonneg_int(e.control.value):
                        s.validation_errors["quota"].discard(_k)
                        e.control.border_color = None
                    else:
                        s.validation_errors["quota"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    e.control.bgcolor = _EMPTY_BG if is_empty_now else _FILLED_BG
                    n = len(s.validation_errors["quota"])
                    _err.value   = f"⚠ {n} cell(s) invalid." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()

                tf.on_change = _ch
                cells.append(tf)

            matrix_buf.append(ft.Row(
                [UIHelpers.plbl(p, s.person_colors.get(p)), _row_btn(p, +1)] +
                cells + [_row_btn(p, -1)],
                spacing=2, wrap=False))

        matrix_buf.append(ft.Row(
            [_spacer] + [_col_btn(t, -1) for t in tasks],
            spacing=2, wrap=False))

        self._matrix_col.controls = matrix_buf
        self.set_matrix_columns(len(tasks))
        self.page.update()