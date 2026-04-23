import flet as ft
from constants import _s, DEFAULT_SOLVER_PARAMS, DEFAULT_WEIGHTS, SOLVER_PARAM_DEFS
from ui_helpers import UIHelpers


# ── Font sizes (absolute px — not scaled, so they stay readable) ─────
_FT  = 15   # section / category titles
_FL  = 13   # param key name and weight key
_FH  = 11   # hint text under each param
_FV  = 13   # TextField value text
_FRI = 10   # rank-index digit inside the circle


class ConfigurationTab:
    """Solver parameters + weight-priority ranking tab.

    Public API (stable):
        - build()                Rebuild weight rows from state.
        - refresh_from_state()   Re-sync TextFields and weights from state
                                 (used after profile load).
        - get_container()        Return the Flet container for this tab.
    """

    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page  = page
        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self):
        self._build_params_section()
        self._build_weights_section()

    # ─────────────────────────────────────────────────────────────────
    # PARAMS SECTION
    # ─────────────────────────────────────────────────────────────────

    def _build_params_section(self):
        s = self.state
        self._param_tfs: dict[str, ft.TextField] = {}

        # Ensure state carries every param (matters when loading old profiles)
        for key, default in DEFAULT_SOLVER_PARAMS.items():
            s.solver_params.setdefault(key, default)

        category_widgets: list[ft.Control] = []

        for cat_def in SOLVER_PARAM_DEFS:
            param_rows: list[ft.Control] = []

            for p in cat_def["params"]:
                key         = p["key"]
                hint        = p["hint"]
                current_val = s.solver_params.get(key, p["default"])

                tf = ft.TextField(
                    value=str(current_val),
                    width=130,
                    height=38,
                    text_size=_FV,
                    content_padding=ft.padding.symmetric(
                        horizontal=8, vertical=4),
                    on_change=lambda e, k=key: self._on_param_change(
                        k, e.control.value),
                )
                self._param_tfs[key] = tf

                row = ft.Row(
                    controls=[
                        ft.Column(
                            controls=[
                                ft.Text(key,  size=_FL,
                                        weight=ft.FontWeight.W_600,
                                        color="#263238"),
                                ft.Text(hint, size=_FH,
                                        color="#78909C",
                                        italic=True),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                        tf,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=10,
                )
                param_rows.append(row)

                # Thin separator between params (skip after last)
                if p is not cat_def["params"][-1]:
                    param_rows.append(
                        ft.Divider(height=1, color="#EEEEEE"))

            category_widgets.append(
                ft.Column(
                    controls=[
                        # Coloured category header bar
                        ft.Container(
                            content=ft.Text(
                                cat_def["category"],
                                size=12,
                                weight=ft.FontWeight.BOLD,
                                color="#FFFFFF",
                            ),
                            bgcolor=cat_def["color"],
                            padding=ft.padding.symmetric(
                                horizontal=12, vertical=7),
                            border_radius=ft.border_radius.only(
                                top_left=6, top_right=6),
                        ),
                        # Param rows inside a card
                        ft.Container(
                            content=ft.Column(param_rows, spacing=10),
                            bgcolor="#FAFAFA",
                            border=ft.border.all(1, "#E0E0E0"),
                            border_radius=ft.border_radius.only(
                                bottom_left=6, bottom_right=6),
                            padding=ft.padding.symmetric(
                                horizontal=12, vertical=10),
                        ),
                    ],
                    spacing=0,
                )
            )

        self._params_section = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Text("Gurobi Parameters",
                                weight=ft.FontWeight.BOLD, size=_FT),
                        UIHelpers.make_reset_btn(
                            "Reset Params", self._reset_params),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Column(category_widgets, spacing=12),
            ],
            spacing=12,
        )

    # ─────────────────────────────────────────────────────────────────
    # WEIGHTS SECTION
    # ─────────────────────────────────────────────────────────────────

    def _build_weights_section(self):
        self._reorderable = ft.ReorderableListView(
            controls=[], on_reorder=self._handle_reorder)

        self._weights_section = ft.Column(
            controls=[
                ft.Text("Priority Ranking",
                        weight=ft.FontWeight.BOLD, size=_FT),
                ft.Row(
                    controls=[
                        ft.Text(
                            "Drag to reorder  ·  edit values  ·  toggle on/off",
                            size=11, italic=True,
                            color=ft.Colors.GREY_500,
                            expand=True),
                        UIHelpers.make_reset_btn("Reset", self._reset_weights),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(height=1),
                ft.Container(content=self._reorderable, expand=True),
            ],
            width=290,
            spacing=8,
            expand=True,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _sort_and_rebuild(self):
        s = self.state
        s.weights_order.sort(key=lambda k: (
            not s.weights_enabled.get(k, True),
            -(s.weights_st.get(k, 0) if s.weights_enabled.get(k, True)
              else s.weights_last_value.get(k, 0)),
        ))
        self.build()

    def _set_weight(self, key: str, value: int):
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
        vals    = sorted((s.weights_st[k] for k in enabled), reverse=True)
        for k, v in zip(enabled, vals):
            s.weights_st[k]         = v
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
            restored = s.weights_last_value.get(
                key, DEFAULT_WEIGHTS.get(key, 1))
            self._set_weight(key, max(1, restored))
        else:
            if s.weights_st[key] > 0:
                s.weights_last_value[key] = s.weights_st[key]
            s.weights_st[key]      = 0
            s.weights_enabled[key] = False
        self._sort_and_rebuild()

    def _reset_params(self, _e):
        self.state.solver_params.clear()
        self.state.solver_params.update(DEFAULT_SOLVER_PARAMS)
        for k, tf in self._param_tfs.items():
            tf.value = str(DEFAULT_SOLVER_PARAMS.get(k, 0))
            try:
                tf.update()
            except Exception:
                pass

    def _reset_weights(self, _e):
        s = self.state
        s.weights_order[:] = list(DEFAULT_WEIGHTS.keys())
        for k, v in DEFAULT_WEIGHTS.items():
            s.weights_enabled[k]    = True
            s.weights_st[k]         = v
            s.weights_last_value[k] = v
        self._sort_and_rebuild()

    # ── Build weight rows ─────────────────────────────────────────────

    def _make_weight_row(self, index: int, key: str) -> ft.Container:
        s       = self.state
        enabled = s.weights_enabled.get(key, True)

        last        = s.weights_last_value.get(key, 0)
        display_val = (f"{s.weights_st[key]:,}" if enabled
                       else (f"{last:,}" if last else "0"))

        row = ft.Row(
            controls=[
                ft.Switch(
                    value=enabled,
                    scale=0.55,
                    on_change=lambda e, k=key: self._handle_toggle(
                        k, e.control.value)),
                ft.Container(
                    content=ft.Text(
                        str(index + 1),
                        size=_FRI,
                        color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER),
                    width=18, height=18, border_radius=9,
                    bgcolor="#546E7A" if enabled else "#BDBDBD",
                    alignment=ft.alignment.center),
                ft.Text(
                    key,
                    size=_FL,
                    weight=ft.FontWeight.W_600 if enabled
                    else ft.FontWeight.NORMAL,
                    color="#37474F" if enabled else "#9E9E9E",
                    no_wrap=True,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    width=110),
                ft.TextField(
                    value=display_val,
                    width=90,
                    height=34,
                    text_size=_FV,
                    text_align=ft.TextAlign.RIGHT,
                    content_padding=ft.padding.symmetric(
                        horizontal=6, vertical=2),
                    border_color="#90A4AE" if enabled else "#E0E0E0",
                    color="#263238" if enabled else "#BDBDBD",
                    disabled=not enabled,
                    input_filter=ft.InputFilter(
                        regex_string=r"[0-9,]", allow=True),
                    on_submit=lambda e, k=key: self._handle_value_commit(
                        k, e.control.value),
                    on_blur=lambda e, k=key: self._handle_value_commit(
                        k, e.control.value)),
                ft.Icon(ft.Icons.DRAG_INDICATOR, size=14, color="#B0BEC5"),
            ],
            spacing=5,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        return ft.Container(
            key=str(index),
            content=row,
            bgcolor=ft.Colors.WHITE if enabled else "#FAFAFA",
            border=ft.border.all(1, "#CFD8DC" if enabled else "#EEEEEE"),
            border_radius=4,
            padding=ft.padding.symmetric(horizontal=6, vertical=4),
            margin=ft.margin.only(bottom=2))

    # ── Public API ────────────────────────────────────────────────────

    def build(self):
        """Rebuild the weight rows list from current state."""
        self._reorderable.controls = [
            self._make_weight_row(i, key)
            for i, key in enumerate(self.state.weights_order)
        ]
        try:
            self.page.update()
        except Exception:
            pass

    def refresh_from_state(self):
        """Re-sync all TextFields and weight rows from state."""
        s = self.state
        # Fill any keys missing from older saved profiles
        for key, default in DEFAULT_SOLVER_PARAMS.items():
            s.solver_params.setdefault(key, default)

        for k, tf in self._param_tfs.items():
            tf.value = str(
                s.solver_params.get(k, DEFAULT_SOLVER_PARAMS.get(k, 0)))
            try:
                tf.update()
            except Exception:
                pass
        self.build()

    # ── Layout ────────────────────────────────────────────────────────

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[
                    self._weights_section,
                    ft.VerticalDivider(width=1, color=ft.Colors.BLUE_GREY_200),
                    ft.ListView(
                        controls=[self._params_section],
                        expand=True,
                        padding=ft.padding.only(right=8),
                    ),
                ],
                spacing=_s(20),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True,
            ),
            padding=_s(16),
            expand=True,
        )