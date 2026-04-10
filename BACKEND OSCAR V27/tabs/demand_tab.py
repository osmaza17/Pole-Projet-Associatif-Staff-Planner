import random
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab

_EMPTY_BG  = ft.Colors.with_opacity(0.18, ft.Colors.BLACK)
_FILLED_BG = ft.Colors.with_opacity(0.08, ft.Colors.BLUE)


class DemandTab(BaseTab):

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
        _, tasks, hours, days = s.dims()
        if not days:
            return
        if s.demand_filter not in days:
            s.demand_filter = days[0]

        def _on_day(v):
            s.demand_filter = v
            self.build()

        def _reset(e):
            s.demand_st.clear()
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _rand(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        s.demand_st[(t, h, j)] = str(random.randint(0, 4))
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        def _zero(e):
            for t in tasks:
                for j in days:
                    for h in hours[j]:
                        s.demand_st[(t, h, j)] = ""
            s.validation_errors["demand"].clear()
            self._on_solve_blocked_update()
            self.build()

        j       = s.demand_filter
        day_hrs = hours[j]
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=_s(12), visible=False)

        # ── Fixed top section ─────────────────────────────────────────
        self._top_ct.controls = [
            # ── Tab bar de días ──────────────────────────────────────
            UIHelpers.make_tab_bar(days, j, _on_day),
            # ── Botones de acción ────────────────────────────────────
            ft.Row([
                UIHelpers.make_reset_btn("Reset to Default", _reset),
                ft.Container(
                    ft.Text("Set All to 0", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.ORANGE_700, padding=_s(8), border_radius=4,
                    on_click=_zero, width=_s(150), alignment=ft.alignment.center),
                ft.Container(
                    ft.Text("Random Demand (All Days)", color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
                    bgcolor=ft.Colors.PURPLE_400, padding=_s(8), border_radius=4,
                    on_click=_rand, width=_s(190), alignment=ft.alignment.center),
            ], spacing=_s(20)),
            ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(14)),
            err_txt,
        ]

        # ── Matrix rows ───────────────────────────────────────────────
        matrix_buf = []
        _spacer = ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 2)

        def _adj(k, delta):
            cur = s.demand_st.get(k, "1")
            try:    v = int(cur) if cur.strip() else 0
            except: v = 0
            nv = max(0, v + delta)
            s.demand_st[k] = "" if nv == 0 else str(nv)
            s.validation_errors["demand"].discard(k)

        def _col_btn(h, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _h=h, _d=delta):
                for t in tasks:
                    _adj((t, _h, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        def _row_btn(t, delta):
            btn = UIHelpers.make_inc_btn(delta)
            def _click(e, _t=t, _d=delta):
                for h in day_hrs:
                    _adj((_t, h, j), _d)
                self._on_solve_blocked_update()
                self.build()
            btn.on_click = _click
            return btn

        matrix_buf.append(ft.Row(
            [_spacer] + [_col_btn(h, +1) for h in day_hrs],
            spacing=2, wrap=False))
        matrix_buf.append(ft.Row(
            [_spacer] +
            [ft.Container(
                ft.Text(h, size=_s(9), no_wrap=True,
                        overflow=ft.TextOverflow.CLIP),
                width=UIHelpers.W_CELL,
                clip_behavior=ft.ClipBehavior.HARD_EDGE)
             for h in day_hrs],
            spacing=2, wrap=False))

        for idx_t, t in enumerate(tasks):
            cells = []
            for idx_h, h in enumerate(day_hrs):
                k = (t, h, j)
                s.demand_st.setdefault(k, "1")
                is_ok = UIHelpers.validate_nonneg_int(s.demand_st[k])
                val = s.demand_st[k]
                is_empty = not val or not val.strip()
                tf = ft.TextField(
                    value=val, width=UIHelpers.W_CELL,
                    height=UIHelpers.H_TF, text_size=_s(11), data=k,
                    content_padding=ft.padding.all(_s(2)),
                    bgcolor=_EMPTY_BG if is_empty else _FILLED_BG,
                    border_color=ft.Colors.RED_400 if not is_ok else None)

                def _ch(e, _k=k, _t_idx=idx_t, _h_idx=idx_h, _err=err_txt):
                    pasted = e.control.value
                    if "\t" in pasted or "\n" in pasted:
                        for r_off, row_str in enumerate(pasted.strip().split("\n")):
                            ti2 = _t_idx + r_off
                            if ti2 >= len(tasks): break
                            for c_off, val_str in enumerate(row_str.split("\t")):
                                hi2 = _h_idx + c_off
                                if hi2 >= len(day_hrs): break
                                tkey = (tasks[ti2], day_hrs[hi2], j)
                                clean = val_str.strip()
                                s.demand_st[tkey] = clean
                                if UIHelpers.validate_nonneg_int(clean):
                                    s.validation_errors["demand"].discard(tkey)
                                else:
                                    s.validation_errors["demand"].add(tkey)
                        self._on_solve_blocked_update()
                        self.build()
                        return
                    s.demand_st[_k] = pasted
                    is_empty_now = not pasted or not pasted.strip()
                    if UIHelpers.validate_nonneg_int(pasted):
                        s.validation_errors["demand"].discard(_k)
                        e.control.border_color = None
                    else:
                        s.validation_errors["demand"].add(_k)
                        e.control.border_color = ft.Colors.RED_400
                    e.control.bgcolor = _EMPTY_BG if is_empty_now else _FILLED_BG
                    n = len(s.validation_errors["demand"])
                    _err.value   = f"⚠ {n} cell(s) invalid." if n else ""
                    _err.visible = bool(n)
                    self._on_solve_blocked_update()
                    e.control.update()
                    _err.update()

                tf.on_change = _ch
                cells.append(tf)

            matrix_buf.append(ft.Row(
                [UIHelpers.lbl(t), _row_btn(t, +1)] + cells + [_row_btn(t, -1)],
                spacing=2, wrap=False))

        matrix_buf.append(ft.Row(
            [_spacer] + [_col_btn(h, -1) for h in day_hrs],
            spacing=2, wrap=False))

        self._matrix_col.controls = matrix_buf
        self.set_matrix_columns(len(day_hrs))
        self.page.update()