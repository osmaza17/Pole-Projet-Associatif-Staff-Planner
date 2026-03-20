"""
Groups manager for the Dimensions tab.

Builds the dynamic group-member text fields, the "Add Group" button,
and the "Modify rest conditions" button.
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS


class GroupsManager:
    """
    Manages the people-groups row (add/remove groups, member editing).

    Parameters
    ----------
    state                  : AppState
    page                   : ft.Page
    people_groups_row      : ft.Row  – container widget to populate
    debounce_fn            : callable(attr, fn, delay) – debounce helper
    on_text_change_cb      : callable() – called when member text changes
                             (no UI rebuild needed, just revalidation)
    on_structure_change_cb : callable() – called on add/delete group
                             (triggers full UI rebuild + revalidation)
    rest_btn_callback      : callable | None – opens the rest-conditions dialog
    """

    def __init__(self, state, page: ft.Page, people_groups_row: ft.Row,
                 debounce_fn,
                 on_text_change_cb=None,
                 on_structure_change_cb=None,
                 rest_btn_callback=None):
        self._state                = state
        self._page                 = page
        self._row                  = people_groups_row
        self._debounce             = debounce_fn
        self._on_text_change       = on_text_change_cb      or (lambda: None)
        self._on_structure_change  = on_structure_change_cb or (lambda: None)
        self._rest_btn_callback    = rest_btn_callback

    def build(self):
        """Rebuild the groups UI from current state."""
        s = self._state

        if not s.groups_st:
            s.groups_st["Group 1"] = ""

        gnames = list(s.groups_st.keys())
        cards  = []

        for idx, gname in enumerate(gnames):
            label_color = GROUP_HEADER_COLORS[
                idx % len(GROUP_HEADER_COLORS)]

            def _del(e, _k=gname):
                if len(s.groups_st) > 1:
                    s.groups_st.pop(_k, None)
                    s.invalidate_cache()
                    # Structural change: needs full rebuild + revalidation
                    self._on_structure_change()

            tf_members = ft.TextField(
                value=s.groups_st[gname], multiline=True,
                min_lines=3, max_lines=200, width=_s(220),
                text_size=_s(11),
                label=gname,
                label_style=ft.TextStyle(
                    size=_s(11), color=label_color,
                    weight=ft.FontWeight.BOLD),
                hint_text="One person per line…",
                suffix=ft.IconButton(
                    icon=ft.Icons.CLOSE, icon_size=_s(13),
                    tooltip="Remove group", on_click=_del),
                content_padding=ft.padding.symmetric(
                    horizontal=_s(12), vertical=_s(10)))

            def _ch_members(e, _k=gname):
                s.groups_st[_k] = e.control.value
                s.invalidate_cache()
                # Text-only change: debounce revalidation, do NOT rebuild UI
                # (rebuilding would destroy active TextFields and lose focus)
                self._debounce(
                    '_debounce_people', self._on_text_change)

            tf_members.on_change = _ch_members
            cards.append(tf_members)

        # ── "Add group" button ─────────────────────────────────────────
        def _add_group(e):
            i = len(s.groups_st) + 1
            new_key = f"Group {i}"
            while new_key in s.groups_st:
                i += 1
                new_key = f"Group {i}"
            s.groups_st[new_key] = ""
            # Structural change: build() called directly here
            self.build()

        add_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ADD, size=_s(14), color=ft.Colors.WHITE),
                ft.Text("Add Group", size=_s(11), color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True),
            bgcolor=ft.Colors.BLUE_GREY_400,
            padding=ft.padding.symmetric(
                horizontal=_s(10), vertical=_s(6)),
            border_radius=4,
            on_click=_add_group)

        # ── "Modify rest conditions" button ───────────────────────────
        rest_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.BEDTIME_OUTLINED, size=_s(14),
                        color=ft.Colors.WHITE),
                ft.Text("Modify rest conditions", size=_s(11),
                        color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True),
            bgcolor="#455A64",
            padding=ft.padding.symmetric(
                horizontal=_s(10), vertical=_s(6)),
            border_radius=4,
            on_click=self._rest_btn_callback,
            visible=self._rest_btn_callback is not None)

        self._row.controls = [
            ft.Column([
                ft.Row([add_btn, rest_btn], spacing=_s(8)),
                ft.Row(cards, spacing=_s(10),
                       vertical_alignment=ft.CrossAxisAlignment.START,
                       scroll=ft.ScrollMode.AUTO),
            ], spacing=_s(8)),
        ]
        self._page.update()