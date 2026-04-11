import flet as ft
from constants import _s, DEFAULT_SOLVER_PARAMS, DEFAULT_WEIGHTS
from ui_helpers import UIHelpers


class ConfigurationTab:
    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page = page
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self):
        self._build_params_section()
        self._build_weights_section()

    def _build_params_section(self):
        s = self.state
        self._param_tfs: dict = {}

        for key, val in s.solver_params.items():
            tf = ft.TextField(
                label=key, value=str(val),
                width=_s(120), height=_s(38), text_size=_s(11),
                content_padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
                label_style=ft.TextStyle(size=_s(10)),
                on_change=lambda e, k=key: self._on_param_change(k, e.control.value),
            )
            self._param_tfs[key] = tf

        tf_list = list(self._param_tfs.values())
        self._params_section = ft.Column([
            ft.Text("Gurobi Parameters", weight=ft.FontWeight.BOLD, size=_s(13)),
            UIHelpers.make_reset_btn("Reset Params", self._reset_params),
            ft.Row([
                ft.Column(tf_list[::2], spacing=_s(6)),
                ft.Column(tf_list[1::2], spacing=_s(6)),
            ], spacing=_s(6), vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=_s(8))

    def _build_weights_section(self):
        self._reorderable = ft.ReorderableListView(
            controls=[], on_reorder=self._handle_reorder)

        self._weights_section = ft.Column([
            ft.Text("Priority Ranking", weight=ft.FontWeight.BOLD, size=_s(13)),
            ft.Row([
                ft.Text(
                    "Drag to reorder  ·  edit values  ·  toggle to enable / disable",
                    size=_s(9), italic=True, color=ft.Colors.GREY_500, expand=True),
                UIHelpers.make_reset_btn("Reset", self._reset_weights),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(height=_s(1)),
            ft.Container(content=self._reorderable, expand=True),
        ], width=_s(100), spacing=_s(6), expand=True)

    # ── Helpers ───────────────────────────────────────────────────────

    def _sort_and_rebuild(self):
        """Sort weights by enabled-first + descending value, then rebuild UI."""
        s = self.state
        s.weights_order.sort(key=lambda k: (
            not s.weights_enabled.get(k, True),
            -(s.weights_st.get(k, 0) if s.weights_enabled.get(k, True)
              else s.weights_last_value.get(k, 0)),
        ))
        self.build()

    def _set_weight(self, key: str, value: int):
        """Central setter: updates value, syncs enabled/last_value."""
        s = self.state
        s.weights_st[key] = value
        if value == 0:
            s.weights_enabled[key] = False
        else:
            s.weights_enabled[key] = True
            s.weights_last_value[key] = value

    # ── Handlers ──────────────────────────────────────────────────────

    def _on_param_change(self, key: str, raw: str):
        try:
            self.state.solver_params[key] = (
                float(raw) if "." in raw else int(raw))
        except ValueError:
            pass

    def _handle_reorder(self, e):
        s = self.state
        item = s.weights_order.pop(e.old_index)
        s.weights_order.insert(e.new_index, item)

        enabled = [k for k in s.weights_order if s.weights_enabled.get(k, True)]
        vals = sorted((s.weights_st[k] for k in enabled), reverse=True)
        for k, v in zip(enabled, vals):
            s.weights_st[k] = v
            s.weights_last_value[k] = v

        self.build()

    def _handle_value_commit(self, key: str, raw: str):
        raw = raw.strip().replace(",", "")
        try:
            v = int(raw) if raw else 0
        except ValueError:
            self.build()
            return
        if v < 0:
            self.build()
            return

        self._set_weight(key, v)
        self._sort_and_rebuild()

    def _handle_toggle(self, key: str, enabled: bool):
        s = self.state
        if enabled:
            restored = s.weights_last_value.get(key, DEFAULT_WEIGHTS.get(key, 1))
            self._set_weight(key, max(1, restored))
        else:
            if s.weights_st[key] > 0:
                s.weights_last_value[key] = s.weights_st[key]
            s.weights_st[key] = 0
            s.weights_enabled[key] = False

        self._sort_and_rebuild()

    def _reset_params(self, _e):
        self.state.solver_params.clear()
        self.state.solver_params.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in self._param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS[k])
            tf.update()

    def _reset_weights(self, _e):
        s = self.state
        s.weights_order[:] = list(DEFAULT_WEIGHTS.keys())
        for k, v in DEFAULT_WEIGHTS.items():
            s.weights_enabled[k] = True
            s.weights_st[k] = v
            s.weights_last_value[k] = v
        self._sort_and_rebuild()

    # ── Build weight rows ─────────────────────────────────────────────

    def _make_weight_row(self, index: int, key: str) -> ft.Container:
        s = self.state
        enabled = s.weights_enabled.get(key, True)

        if enabled:
            display_val = f"{s.weights_st[key]:,}"
        else:
            last = s.weights_last_value.get(key, 0)
            display_val = f"{last:,}" if last else "0"

        row = ft.Row([
            ft.Switch(
                value=enabled, scale=0.45,
                on_change=lambda e, k=key: self._handle_toggle(k, e.control.value)),
            ft.Container(
                content=ft.Text(
                    str(index + 1), size=_s(7), color=ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                width=_s(14), height=_s(14), border_radius=7,
                bgcolor="#546E7A" if enabled else "#BDBDBD",
                alignment=ft.alignment.center),
            ft.Text(
                key, size=_s(9),
                weight=ft.FontWeight.W_600 if enabled else ft.FontWeight.NORMAL,
                color="#37474F" if enabled else "#9E9E9E",
                no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS, width=_s(90)),
            ft.TextField(
                value=display_val, width=_s(80), height=_s(28),
                text_size=_s(10), text_align=ft.TextAlign.RIGHT,
                content_padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2)),
                border_color="#90A4AE" if enabled else "#E0E0E0",
                color="#263238" if enabled else "#BDBDBD",
                disabled=not enabled,
                input_filter=ft.InputFilter(regex_string=r"[0-9,]", allow=True),
                on_submit=lambda e, k=key: self._handle_value_commit(k, e.control.value),
                on_blur=lambda e, k=key: self._handle_value_commit(k, e.control.value)),
            ft.Icon(ft.Icons.DRAG_INDICATOR, size=_s(11), color="#B0BEC5"),
        ], spacing=_s(2), vertical_alignment=ft.CrossAxisAlignment.CENTER)

        return ft.Container(
            key=str(index), content=row,
            bgcolor=ft.Colors.WHITE if enabled else "#FAFAFA",
            border=ft.border.all(1, "#CFD8DC" if enabled else "#EEEEEE"),
            border_radius=4,
            padding=ft.padding.symmetric(horizontal=_s(4), vertical=_s(2)),
            margin=ft.margin.only(bottom=_s(1)))

    def build(self):
        self._reorderable.controls = [
            self._make_weight_row(i, key)
            for i, key in enumerate(self.state.weights_order)
        ]
        self.page.update()

    # ── Layout ────────────────────────────────────────────────────────

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row([
                self._weights_section,
                ft.VerticalDivider(width=1, color=ft.Colors.BLUE_GREY_200),
                ft.ListView(controls=[self._params_section], width=_s(280), expand=True),
            ], spacing=_s(20),
               alignment=ft.MainAxisAlignment.START,
               vertical_alignment=ft.CrossAxisAlignment.START,
               expand=True),
            padding=_s(16), expand=True)