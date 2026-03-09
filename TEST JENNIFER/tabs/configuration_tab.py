import flet as ft
from constants import (
    _s, DEFAULT_SOLVER_PARAMS, DEFAULT_WEIGHTS
)
from ui_helpers import UIHelpers


class ConfigurationTab:
    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page  = page
        self._build_ui()

    # ── Static layout construction (called once) ──────────────────────

    def _build_ui(self):
        s = self.state

        # ── Gurobi Parameters ─────────────────────────────────────────
        self._param_tfs: dict = {}
        for key, val in s.solver_params.items():
            tf = ft.TextField(
                label=key, value=str(val),
                width=_s(120), height=_s(38), text_size=_s(11),
                content_padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
                label_style=ft.TextStyle(size=_s(10)))
            def _ch(e, _k=key):
                try:
                    s.solver_params[_k] = (
                        float(e.control.value) if "." in e.control.value
                        else int(e.control.value))
                except ValueError:
                    pass
            tf.on_change = _ch
            self._param_tfs[key] = tf

        tf_list = list(self._param_tfs.values())
        self._params_section = ft.Column([
            ft.Text("Gurobi Parameters",
                    weight=ft.FontWeight.BOLD, size=_s(13)),
            UIHelpers.make_reset_btn("Reset Params", self._reset_params),
            ft.Row([
                ft.Column(tf_list[::2],  spacing=_s(6)),
                ft.Column(tf_list[1::2], spacing=_s(6)),
            ], spacing=_s(6), vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=_s(8))

        # ── Weights list (rebuilt on every reorder/toggle/value change) ─
        self._reorderable = ft.ReorderableListView(
            controls=[], on_reorder=self._handle_reorder)

        self._weights_section = ft.Column([
            ft.Text("Priority Ranking",
                    weight=ft.FontWeight.BOLD, size=_s(13)),
            ft.Row([
                ft.Text(
                    "Drag to reorder  ·  edit values  ·  toggle to enable / disable",
                    size=_s(9), italic=True,
                    color=ft.Colors.GREY_500, expand=True),
                UIHelpers.make_reset_btn("Reset", self._reset_weights),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(height=_s(1)),
            ft.Container(
                content=self._reorderable,
                expand=True),
        ], width=_s(100), spacing=_s(6), expand=True)

    # ── Sort by descending value ──────────────────────────────────────

    def _sort_weights_order(self):
        """Sort weights_order: enabled descending by value, disabled last.
        Disabled items sort among themselves by their last saved value
        (descending) so the list stays intuitive."""
        s = self.state
        s.weights_order.sort(
            key=lambda k: (
                0 if s.weights_enabled.get(k, True) else 1,
                -(s.weights_st.get(k, 0) if s.weights_enabled.get(k, True)
                  else s.weights_last_value.get(k, 0)),
            )
        )

    # ── Reorder, value edit, toggle, reset ────────────────────────────

    def _handle_reorder(self, e):
        """Drag-and-drop: move item, then redistribute the current pool
        of enabled values by position (highest value → position 1, etc.).
        This keeps the list always sorted descending."""
        s = self.state
        item = s.weights_order.pop(e.old_index)
        s.weights_order.insert(e.new_index, item)

        # Collect current enabled values and sort descending
        enabled_keys = [k for k in s.weights_order
                        if s.weights_enabled.get(k, True)]
        enabled_vals = sorted(
            [s.weights_st[k] for k in enabled_keys], reverse=True)

        # Assign by position: first enabled key gets highest value, etc.
        for k, v in zip(enabled_keys, enabled_vals):
            s.weights_st[k] = v
            s.weights_last_value[k] = v

        self.build()

    def _handle_value_commit(self, key: str, raw: str):
        """Called on Enter (on_submit) or click-outside (on_blur).
        Validates input, syncs enabled↔value=0, re-sorts, rebuilds."""
        s = self.state
        raw = raw.strip().replace(",", "")

        # ── Validate: non-negative integer only ───────────────────────
        if not raw:
            v = 0
        else:
            try:
                v = int(raw)
            except ValueError:
                self.build()      # revert display to current state
                return
            if v < 0:
                self.build()      # revert display
                return

        s.weights_st[key] = v

        # ── Sync: value=0 ↔ disabled, value>0 → save as last_value ──
        if v == 0:
            # Don't overwrite last_value — we want to keep the old one
            s.weights_enabled[key] = False
        else:
            s.weights_last_value[key] = v
            s.weights_enabled[key] = True

        self._sort_weights_order()
        self.build()

    def _handle_toggle(self, key: str, enabled: bool):
        """Toggle switch on/off: save/restore value, rebuild."""
        s = self.state
        s.weights_enabled[key] = enabled

        if not enabled:
            # Save current value before zeroing
            if s.weights_st[key] > 0:
                s.weights_last_value[key] = s.weights_st[key]
            s.weights_st[key] = 0
        else:
            # Restore the last saved value
            restored = s.weights_last_value.get(key, DEFAULT_WEIGHTS.get(key, 1))
            s.weights_st[key] = max(1, restored)
            s.weights_last_value[key] = s.weights_st[key]

        self._sort_weights_order()
        self.build()

    def _reset_params(self, e):
        self.state.solver_params.clear()
        self.state.solver_params.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in self._param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS[k])
            tf.update()

    def _reset_weights(self, e):
        s = self.state
        s.weights_order.clear()
        s.weights_order.extend(list(DEFAULT_WEIGHTS.keys()))
        for k in DEFAULT_WEIGHTS:
            s.weights_enabled[k] = True
            s.weights_st[k] = DEFAULT_WEIGHTS[k]
            s.weights_last_value[k] = DEFAULT_WEIGHTS[k]
        self._sort_weights_order()
        self.build()

    # ── Build (weights list only) ─────────────────────────────────────

    def build(self):
        s = self.state

        items = []
        for i, key in enumerate(s.weights_order):
            enabled = s.weights_enabled.get(key, True)

            # ── Switch ────────────────────────────────────────────────
            sw = ft.Switch(value=enabled, scale=0.45)
            def _toggle(e, _k=key):
                self._handle_toggle(_k, e.control.value)
            sw.on_change = _toggle

            # ── Position badge ────────────────────────────────────────
            badge = ft.Container(
                content=ft.Text(
                    str(i + 1), size=_s(7), color=ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER),
                width=_s(14), height=_s(14), border_radius=7,
                bgcolor="#546E7A" if enabled else "#BDBDBD",
                alignment=ft.alignment.center)

            # ── Weight name label ─────────────────────────────────────
            label = ft.Text(
                key, size=_s(9),
                weight=ft.FontWeight.W_600 if enabled else ft.FontWeight.NORMAL,
                color="#37474F" if enabled else "#9E9E9E",
                no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                width=_s(90))

            # ── Editable value TextField ──────────────────────────────
            # Enabled: show current value. Disabled: show last value
            # (greyed out) so the user sees what will be restored.
            if enabled:
                display_val = f"{s.weights_st[key]:,}"
            else:
                last = s.weights_last_value.get(key, 0)
                display_val = f"{last:,}" if last > 0 else "0"

            val_tf = ft.TextField(
                value=display_val,
                width=_s(80), height=_s(28),
                text_size=_s(10),
                text_align=ft.TextAlign.RIGHT,
                content_padding=ft.padding.symmetric(
                    horizontal=_s(6), vertical=_s(2)),
                border_color="#90A4AE" if enabled else "#E0E0E0",
                color="#263238" if enabled else "#BDBDBD",
                disabled=not enabled,
                input_filter=ft.InputFilter(
                    regex_string=r"[0-9,]",
                    allow=True),
                on_submit=lambda e, _k=key: self._handle_value_commit(
                    _k, e.control.value),
                on_blur=lambda e, _k=key: self._handle_value_commit(
                    _k, e.control.value),
            )

            # ── Drag handle ───────────────────────────────────────────
            handle = ft.Icon(ft.Icons.DRAG_INDICATOR,
                             size=_s(11), color="#B0BEC5")

            row = ft.Row([
                sw, badge, label, val_tf, handle,
            ], spacing=_s(2),
               vertical_alignment=ft.CrossAxisAlignment.CENTER)

            items.append(ft.Container(
                key=str(i),
                content=row,
                bgcolor=ft.Colors.WHITE if enabled else "#FAFAFA",
                border=ft.border.all(1, "#CFD8DC" if enabled else "#EEEEEE"),
                border_radius=4,
                padding=ft.padding.symmetric(horizontal=_s(4), vertical=_s(2)),
                margin=ft.margin.only(bottom=_s(1))))

        self._reorderable.controls = items
        self.page.update()

    # ── Layout ────────────────────────────────────────────────────────

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[
                    self._weights_section,
                    ft.VerticalDivider(
                        width=1, color=ft.Colors.BLUE_GREY_200),
                    ft.ListView(
                        controls=[self._params_section],
                        width=_s(280), expand=True),
                ],
                spacing=_s(20),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True),
            padding=_s(16), expand=True)