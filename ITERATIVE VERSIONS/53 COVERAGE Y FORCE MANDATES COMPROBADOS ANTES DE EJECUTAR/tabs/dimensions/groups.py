"""
Groups manager for the Dimensions tab.

Builds the dynamic group-member text fields, the "Add Group" button,
the "Modify rest conditions" button, and the inter-group equity mode
toggle (Off / Custom ratios) with per-group ratio fields.
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS


_MODE_HELP = {
    "off":
        "All people are treated as a single pool — hours distributed "
        "equally. Visual groups are cosmetic only.",
    "shares":
        "Define relative weight per group. Values are normalized "
        "automatically (50/30/20 ≡ 5/3/2). Equal values → equal hours.",
}


class GroupsManager:
    """Manages the people-groups row (add/remove groups, member editing)."""

    def __init__(self, state, page: ft.Page, people_groups_row: ft.Row,
                 debounce_fn,
                 on_text_change_cb=None,
                 on_structure_change_cb=None,
                 rest_btn_callback=None):
        self._state = state
        self._page = page
        self._row = people_groups_row
        self._debounce = debounce_fn
        self._on_text_change = on_text_change_cb or (lambda: None)
        self._on_structure_change = on_structure_change_cb or (lambda: None)
        self._rest_btn_callback = rest_btn_callback

    # ── Rename helper ─────────────────────────────────────────────

    def _rename_group(self, old_name: str, new_name: str) -> bool:
        """Rename a group key in groups_st, preserving insertion order.
        Returns True on success, False if the name is invalid or
        collides with an existing group."""
        s = self._state
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return False
        if new_name in s.groups_st:
            return False

        # Rebuild dict preserving order with the new key
        rebuilt = {}
        for k, v in s.groups_st.items():
            rebuilt[new_name if k == old_name else k] = v
        s.groups_st.clear()
        s.groups_st.update(rebuilt)

        # Also rename in group_shares_st so the ratio value follows
        if old_name in s.group_shares_st:
            s.group_shares_st[new_name] = s.group_shares_st.pop(old_name)

        s.invalidate_cache()
        return True

    # ── Build ─────────────────────────────────────────────────────

    def build(self):
        """Rebuild the groups UI from current state."""
        s = self._state

        if not s.groups_st:
            s.groups_st["Group 1"] = ""

        # Ensure mode attribute exists
        if not hasattr(s, "eq_group_mode"):
            s.eq_group_mode = "off"
        if s.eq_group_mode not in ("off", "shares"):
            s.eq_group_mode = "off"

        is_ratios = s.eq_group_mode == "shares"
        gnames = list(s.groups_st.keys())
        cards = []

        for idx, gname in enumerate(gnames):
            label_color = GROUP_HEADER_COLORS[idx % len(GROUP_HEADER_COLORS)]

            # ── Delete button ─────────────────────────────────────
            def _del(e, _k=gname):
                if len(s.groups_st) > 1:
                    s.groups_st.pop(_k, None)
                    s.group_shares_st.pop(_k, None)
                    s.invalidate_cache()
                    self._on_structure_change()

            del_btn = ft.Container(
                content=ft.IconButton(
                    icon=ft.Icons.CLOSE, icon_size=_s(14),
                    icon_color=ft.Colors.RED_400,
                    tooltip="Remove group", on_click=_del,
                    style=ft.ButtonStyle(
                        padding=ft.padding.all(0),
                        shape=ft.CircleBorder())),
                width=_s(28), height=_s(28),
                alignment=ft.alignment.center,
                visible=len(s.groups_st) > 1)

            # ── Editable group name ───────────────────────────────
            tf_name = ft.TextField(
                value=gname,
                width=_s(120), height=_s(30),
                text_size=_s(11),
                text_style=ft.TextStyle(
                    weight=ft.FontWeight.BOLD, color=label_color),
                content_padding=ft.padding.symmetric(
                    horizontal=_s(8), vertical=_s(4)),
                border_color=ft.Colors.TRANSPARENT,
                focused_border_color=label_color,
                cursor_color=label_color,
                border_radius=4,
                data=gname)

            def _commit_rename(e, _old=gname, _color=label_color):
                new_val = e.control.value.strip()
                if not new_val:
                    e.control.value = _old
                    e.control.data = _old
                    try:
                        e.control.update()
                    except Exception:
                        pass
                    return
                if new_val == _old:
                    return
                ok = self._rename_group(_old, new_val)
                if ok:
                    self._on_structure_change()
                else:
                    e.control.value = _old
                    e.control.data = _old
                    try:
                        e.control.update()
                    except Exception:
                        pass

            tf_name.on_submit = _commit_rename
            tf_name.on_blur = _commit_rename

            # ── Ratio text field ──────────────────────────────────
            init_ratio = s.group_shares_st.get(gname, "") or ""
            tf_ratio = ft.TextField(
                value=init_ratio,
                width=_s(80), height=_s(30),
                text_size=_s(11),
                hint_text="e.g. 50",
                label="Ratio",
                label_style=ft.TextStyle(size=_s(9)),
                content_padding=ft.padding.symmetric(
                    horizontal=_s(6), vertical=_s(4)),
                border_color="#B0BEC5" if is_ratios else ft.Colors.GREY_300,
                focused_border_color=label_color if is_ratios else ft.Colors.GREY_300,
                cursor_color=label_color,
                border_radius=4,
                read_only=not is_ratios,
                color=ft.Colors.BLACK if is_ratios else ft.Colors.GREY_400,
                bgcolor=ft.Colors.WHITE if is_ratios else "#F5F5F5",
                keyboard_type=ft.KeyboardType.NUMBER,
                data=gname)

            def _on_ratio_change(ev, _g=gname, _tf=tf_ratio):
                raw = (ev.control.value or "").strip()
                if raw == "":
                    s.group_shares_st[_g] = ""
                    _tf.border_color = None
                else:
                    try:
                        v = float(raw)
                        if v < 0:
                            _tf.border_color = ft.Colors.RED_400
                            s.group_shares_st[_g] = ""
                        else:
                            _tf.border_color = None
                            s.group_shares_st[_g] = raw
                    except ValueError:
                        _tf.border_color = ft.Colors.RED_400
                        s.group_shares_st[_g] = ""
                try:
                    _tf.update()
                except Exception:
                    pass
                s.invalidate_cache()
                self._on_equity_change()

            tf_ratio.on_change = _on_ratio_change

            header_row = ft.Row(
                [tf_name, tf_ratio, del_btn],
                spacing=_s(4),
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True)

            # ── Members text field ────────────────────────────────
            tf_members = ft.TextField(
                value=s.groups_st[gname], multiline=True,
                min_lines=20, max_lines=20, width=_s(240),
                text_size=_s(11),
                hint_text="One person per line…",
                content_padding=ft.padding.symmetric(
                    horizontal=_s(12), vertical=_s(10)))

            def _ch_members(e, _k=gname):
                s.groups_st[_k] = e.control.value
                s.invalidate_cache()
                self._debounce('_debounce_people', self._on_text_change)

            tf_members.on_change = _ch_members

            # ── Card column ───────────────────────────────────────
            card = ft.Column(
                [header_row, tf_members],
                spacing=_s(4), tight=True,
                width=_s(250))
            cards.append(card)

        # ── Action buttons ────────────────────────────────────────
        def _add_group(e):
            i = len(s.groups_st) + 1
            new_key = f"Group {i}"
            while new_key in s.groups_st:
                i += 1
                new_key = f"Group {i}"
            s.groups_st[new_key] = ""
            s.invalidate_cache()
            self._on_structure_change()

        add_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ADD, size=_s(14), color=ft.Colors.WHITE),
                ft.Text("Add Group", size=_s(11), color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True),
            bgcolor=ft.Colors.BLUE_GREY_400,
            padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            border_radius=4, on_click=_add_group)

        rest_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.BEDTIME_OUTLINED, size=_s(14),
                        color=ft.Colors.WHITE),
                ft.Text("Modify rest conditions", size=_s(11),
                        color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True),
            bgcolor="#455A64",
            padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            border_radius=4,
            on_click=self._rest_btn_callback,
            visible=self._rest_btn_callback is not None)

        # ── Equity mode toggle ────────────────────────────────────
        current_mode = s.eq_group_mode

        def _on_mode_change(ev):
            sel = ev.control.selected
            new_mode = next(iter(sel)) if sel else "off"
            if new_mode == s.eq_group_mode:
                return
            s.eq_group_mode = new_mode
            s.invalidate_cache()
            self.build()
            self._on_equity_change()

        mode_toggle = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="off",
                    label=ft.Text("Off", size=_s(10)),
                    icon=ft.Icon(ft.Icons.BLOCK, size=_s(12))),
                ft.Segment(
                    value="shares",
                    label=ft.Text("Custom ratios", size=_s(10)),
                    icon=ft.Icon(ft.Icons.TUNE, size=_s(12))),
            ],
            selected={current_mode},
            allow_multiple_selection=False,
            allow_empty_selection=False,
            on_change=_on_mode_change,
            style=ft.ButtonStyle(
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2)),
                text_style=ft.TextStyle(size=_s(10)),
            ))

        help_text = ft.Text(
            _MODE_HELP[current_mode],
            size=_s(9), italic=True, color=ft.Colors.GREY_600,
            max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)

        equity_row = ft.Row(
            [ft.Icon(ft.Icons.BALANCE, size=_s(14), color="#455A64"),
             ft.Text("Group equity:", size=_s(11),
                     weight=ft.FontWeight.BOLD, color="#455A64"),
             mode_toggle,
             ft.Container(content=help_text, expand=True)],
            spacing=_s(6),
            vertical_alignment=ft.CrossAxisAlignment.CENTER)

        # ── Assemble ──────────────────────────────────────────────
        self._row.controls = [
            ft.Row([add_btn, rest_btn], spacing=_s(8)),
            equity_row,
            ft.Row(cards, spacing=_s(10), run_spacing=_s(10),
                   vertical_alignment=ft.CrossAxisAlignment.START,
                   wrap=True),
        ]
        self._page.update()

    # ── Equity helpers ────────────────────────────────────────────

    def _on_equity_change(self):
        """Called when mode or any ratio value changes."""
        s = self._state
        s.reconcile_group_equity()
        s.validate_group_equity()
        self._on_text_change()