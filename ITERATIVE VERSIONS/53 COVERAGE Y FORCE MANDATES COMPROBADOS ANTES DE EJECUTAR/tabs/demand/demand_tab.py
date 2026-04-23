import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab

_EMPTY_BG  = ft.Colors.with_opacity(0.18, ft.Colors.BLACK)
_FILLED_BG = ft.Colors.with_opacity(0.08, ft.Colors.BLUE)


class DemandTab(BaseTab):

    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        super().__init__(state, page)
        self._on_solve_blocked = on_solve_blocked_update

        self._top_ct = ft.Column(spacing=_s(6), tight=True)
        self._matrix_col = ft.Column(spacing=2, tight=True, scroll=ft.ScrollMode.AUTO)
        self._matrix_row = ft.Row(
            controls=[self._matrix_col], scroll=ft.ScrollMode.AUTO,
            expand=True, vertical_alignment=ft.CrossAxisAlignment.START)

        self._ct.controls = [self._top_ct, ft.Divider(height=_s(6)), self._matrix_row]
        self._ct.expand = True

    # ── Helpers ───────────────────────────────────────────────────────

    def _all_keys(self):
        """Yield all (task, hour, day) keys."""
        _, tasks, hours, days = self.state.dims()
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    yield (t, h, j)

    def _fill_all(self, value_fn):
        """Set every demand cell via value_fn(task, hour, day), clear errors, rebuild."""
        s = self.state
        _, tasks, hours, days = s.dims()
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    s.demand_st[(t, h, j)] = value_fn(t, h, j)
        s.validation_errors["demand"].clear()
        self._on_solve_blocked()
        self.build()

    def _adjust(self, key, delta):
        """Increment/decrement a single cell, floor at 0."""
        s = self.state
        cur = s.demand_st.get(key, "1")
        try:
            v = int(cur) if cur.strip() else 0
        except ValueError:
            v = 0
        nv = max(0, v + delta)
        s.demand_st[key] = "" if nv == 0 else str(nv)
        s.validation_errors["demand"].discard(key)

    def _validate_and_track(self, key, value):
        """Validate a cell value, update error set. Returns is_valid."""
        errs = self.state.validation_errors["demand"]
        if UIHelpers.validate_nonneg_int(value):
            errs.discard(key)
            return True
        errs.add(key)
        return False

    def _update_error_text(self, err_txt):
        n = len(self.state.validation_errors["demand"])
        err_txt.value = f"⚠ {n} cell(s) invalid." if n else ""
        err_txt.visible = bool(n)

    # ── Build ─────────────────────────────────────────────────────────

    def build(self):
        s = self.state
        _, tasks, hours, days = s.dims()
        if not days:
            return
        if s.demand_filter not in days:
            s.demand_filter = days[0]

        j = s.demand_filter
        day_hrs = hours[j]
        err_txt = ft.Text("", color=ft.Colors.RED_400, size=_s(12), visible=False)

        self._build_top_bar(days, j, err_txt)
        self._build_matrix(tasks, day_hrs, j, err_txt)
        self.set_matrix_columns(len(day_hrs))
        self.page.update()

    def _build_top_bar(self, days, current_day, err_txt):
        self._top_ct.controls = [
            UIHelpers.make_tab_bar(days, current_day, self._on_day_select),
            ft.Row([
                UIHelpers.make_reset_btn("Reset to Default", self._on_reset),
                self._action_btn("Set All to 0", ft.Colors.ORANGE_700, 150, self._on_zero),

            ], spacing=_s(20)),
            ft.Text(f"-- {current_day} --", weight=ft.FontWeight.BOLD, size=_s(14)),
            err_txt,
        ]

    @staticmethod
    def _action_btn(label, color, width, on_click):
        return ft.Container(
            ft.Text(label, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=_s(12)),
            bgcolor=color, padding=_s(8), border_radius=4,
            on_click=on_click, width=_s(width), alignment=ft.alignment.center)

    def _build_matrix(self, tasks, day_hrs, j, err_txt):
        spacer = ft.Container(width=UIHelpers.W_LBL + UIHelpers.W_CELL + 2)
        rows = []

        # Column +1 buttons
        rows.append(ft.Row(
            [spacer] + [self._col_btn(h, j, tasks, day_hrs, +1) for h in day_hrs],
            spacing=2, wrap=False))

        # Column headers
        rows.append(ft.Row(
            [spacer] + [
                ft.Container(
                    ft.Text(h, size=_s(9), no_wrap=True, overflow=ft.TextOverflow.CLIP),
                    width=UIHelpers.W_CELL, clip_behavior=ft.ClipBehavior.HARD_EDGE)
                for h in day_hrs],
            spacing=2, wrap=False))

        # Data rows
        for idx_t, t in enumerate(tasks):
            cells = [
                self._make_cell(t, h, j, idx_t, idx_h, tasks, day_hrs, err_txt)
                for idx_h, h in enumerate(day_hrs)
            ]
            rows.append(ft.Row(
                [UIHelpers.lbl(t), self._row_btn(t, j, day_hrs, +1)]
                + cells
                + [self._row_btn(t, j, day_hrs, -1)],
                spacing=2, wrap=False))

        # Column -1 buttons
        rows.append(ft.Row(
            [spacer] + [self._col_btn(h, j, tasks, day_hrs, -1) for h in day_hrs],
            spacing=2, wrap=False))

        self._matrix_col.controls = rows

    # ── Cell factory ──────────────────────────────────────────────────

    def _make_cell(self, t, h, j, idx_t, idx_h, tasks, day_hrs, err_txt):
        k = (t, h, j)
        s = self.state
        s.demand_st.setdefault(k, "1")
        val = s.demand_st[k]
        is_ok = UIHelpers.validate_nonneg_int(val)
        is_empty = not val or not val.strip()

        tf = ft.TextField(
            value=val, width=UIHelpers.W_CELL, height=UIHelpers.H_TF,
            text_size=_s(11), data=k,
            content_padding=ft.padding.all(_s(2)),
            bgcolor=_EMPTY_BG if is_empty else _FILLED_BG,
            border_color=ft.Colors.RED_400 if not is_ok else None,
            on_change=lambda e, _k=k, _ti=idx_t, _hi=idx_h:
                self._on_cell_change(e, _k, _ti, _hi, tasks, day_hrs, j, err_txt),
        )
        return tf

    def _on_cell_change(self, e, key, t_idx, h_idx, tasks, day_hrs, j, err_txt):
        s = self.state
        pasted = e.control.value

        # ── Multi-cell paste (tab/newline separated) ──────────────
        if "\t" in pasted or "\n" in pasted:
            for r_off, row_str in enumerate(pasted.strip().split("\n")):
                ti = t_idx + r_off
                if ti >= len(tasks):
                    break
                for c_off, val_str in enumerate(row_str.split("\t")):
                    hi = h_idx + c_off
                    if hi >= len(day_hrs):
                        break
                    cell_key = (tasks[ti], day_hrs[hi], j)
                    clean = val_str.strip()
                    s.demand_st[cell_key] = clean
                    self._validate_and_track(cell_key, clean)
            self._on_solve_blocked()
            self.build()
            return

        # ── Single cell edit ──────────────────────────────────────
        s.demand_st[key] = pasted
        is_ok = self._validate_and_track(key, pasted)
        is_empty = not pasted or not pasted.strip()

        e.control.border_color = None if is_ok else ft.Colors.RED_400
        e.control.bgcolor = _EMPTY_BG if is_empty else _FILLED_BG
        self._update_error_text(err_txt)
        self._on_solve_blocked()
        e.control.update()
        err_txt.update()

    # ── Increment/decrement buttons ───────────────────────────────────

    def _col_btn(self, h, j, tasks, day_hrs, delta):
        btn = UIHelpers.make_inc_btn(delta)
        def _click(_e, _h=h):
            for t in tasks:
                self._adjust((t, _h, j), delta)
            self._on_solve_blocked()
            self.build()
        btn.on_click = _click
        return btn

    def _row_btn(self, t, j, day_hrs, delta):
        btn = UIHelpers.make_inc_btn(delta)
        def _click(_e, _t=t):
            for h in day_hrs:
                self._adjust((_t, h, j), delta)
            self._on_solve_blocked()
            self.build()
        btn.on_click = _click
        return btn

    # ── Top-bar callbacks ─────────────────────────────────────────────

    def _on_day_select(self, v):
        self.state.demand_filter = v
        self.build()

    def _on_reset(self, _e):
        self.state.demand_st.clear()
        self.state.validation_errors["demand"].clear()
        self._on_solve_blocked()
        self.build()

    def _on_zero(self, _e):
        self._fill_all(lambda t, h, j: "")