"""
displacement_tab.py
===================
Tab for assigning tasks to locations and configuring travel times
between locations.

Layout:
  LEFT  — upper-triangular matrix of travel-time TextFields
  RIGHT — grid of location zones with drag-and-drop task chips
"""

import flet as ft
from constants import _s, TASK_COLORS, loc_color

_ZONE_WIDTH  = _s(260)
_CHIP_H      = _s(28)
_MATRIX_CELL = _s(58)
_MATRIX_LBL  = _s(90)


class DisplacementTab:

    def __init__(self, state, page, on_change=None):
        self.state = state
        self.page = page
        self._on_change = on_change
        self._ct = ft.Container(expand=True, padding=_s(10))

    def get_container(self) -> ft.Container:
        return self._ct

    # ── Build ─────────────────────────────────────────────────────────

    def build(self):
        s = self.state
        _, tasks, _, _ = s.dims()
        s.sync_displacement_tasks(tasks)
        locs = s.location_names_st

        left_col = ft.Column([
            ft.Text("Travel time (hours)", size=_s(12),
                    weight=ft.FontWeight.BOLD, color="#37474F"),
            self._build_matrix(locs),
        ], spacing=_s(8), scroll=ft.ScrollMode.AUTO,
           width=max(_s(200), _MATRIX_LBL + _MATRIX_CELL * len(locs) + _s(20)))

        right_panel = ft.Column([
            ft.ElevatedButton(
                "Add location", icon=ft.Icons.ADD_LOCATION_ALT,
                on_click=self._add_location,
                style=ft.ButtonStyle(
                    bgcolor="#1565C0", color=ft.Colors.WHITE,
                    padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(8)),
                    shape=ft.RoundedRectangleBorder(radius=6))),
            ft.Row(self._build_zones(tasks, locs),
                   wrap=True, spacing=_s(10), run_spacing=_s(10),
                   vertical_alignment=ft.CrossAxisAlignment.START),
        ], spacing=_s(12), expand=True, scroll=ft.ScrollMode.AUTO)

        self._ct.content = ft.Row(
            [left_col, ft.VerticalDivider(width=1), right_panel],
            spacing=_s(12), expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START)
        try:
            self._ct.update()
        except Exception:
            pass

    # ── Travel-time matrix ────────────────────────────────────────────

    def _build_matrix(self, locs):
        n = len(locs)
        if n < 2:
            return ft.Container(
                ft.Text("Add a second location to set travel times.",
                        size=_s(11), italic=True, color="#607D8B"),
                padding=_s(8))

        def _lbl(text):
            return ft.Container(
                ft.Text(text, size=_s(9), no_wrap=True,
                        text_align=ft.TextAlign.CENTER,
                        overflow=ft.TextOverflow.ELLIPSIS),
                width=_MATRIX_CELL, alignment=ft.alignment.center)

        header = ft.Row(
            [ft.Container(width=_MATRIX_LBL)] + [_lbl(loc) for loc in locs],
            spacing=_s(2))

        rows = [header]
        for i, loc_i in enumerate(locs):
            cells = [ft.Container(
                ft.Text(loc_i, size=_s(9), no_wrap=True,
                        overflow=ft.TextOverflow.ELLIPSIS),
                width=_MATRIX_LBL)]
            for j, loc_j in enumerate(locs):
                if j <= i:
                    cells.append(ft.Container(
                        ft.Text("0" if i == j else "", size=_s(9),
                                text_align=ft.TextAlign.CENTER, color="#BDBDBD"),
                        width=_MATRIX_CELL, alignment=ft.alignment.center))
                else:
                    cells.append(ft.TextField(
                        value=str(self.state.travel_time_st.get((i, j), "")),
                        width=_MATRIX_CELL, height=_s(38), text_size=_s(10),
                        text_align=ft.TextAlign.CENTER, dense=True, hint_text="0",
                        content_padding=ft.padding.symmetric(
                            horizontal=_s(4), vertical=_s(2)),
                        on_change=lambda e, k=(i, j): self._on_matrix_change(e, k)))
            rows.append(ft.Row(cells, spacing=_s(2)))

        return ft.Column(rows, spacing=_s(4))

    def _on_matrix_change(self, e, key):
        val = e.control.value.strip()
        if val and not val.isdigit():
            e.control.error_text = "int"
            e.control.update()
            return
        e.control.error_text = None
        e.control.update()
        self.state.travel_time_st[key] = val

    # ── Location zones ────────────────────────────────────────────────

    def _build_zones(self, tasks, locs):
        by_loc: dict[int, list[str]] = {i: [] for i in range(len(locs))}
        for t in tasks:
            by_loc.setdefault(self.state.task_location_idx_st.get(t, 0), []).append(t)

        task_idx = {t: i for i, t in enumerate(self.state.dims()[1])}
        return [
            self._build_zone(name, by_loc.get(i, []), i, removable=(i > 0), task_idx=task_idx)
            for i, name in enumerate(locs)
        ]

    def _build_zone(self, loc_name, zone_tasks, loc_idx, removable, task_idx):
        loc_bg, loc_fg = loc_color(loc_idx)

        # Header
        title_tf = ft.TextField(
            value=loc_name, border=ft.InputBorder.NONE, dense=True, expand=True,
            text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=loc_fg, size=_s(13)),
            content_padding=ft.padding.symmetric(horizontal=_s(4)),
            on_blur=lambda e, i=loc_idx: self._rename_location(e, i),
            on_submit=lambda e, i=loc_idx: self._rename_location(e, i))

        hdr_controls = [ft.Icon(ft.Icons.PLACE, color=loc_fg, size=_s(16)), title_tf]
        if removable:
            hdr_controls.append(ft.IconButton(
                ft.Icons.CLOSE, icon_size=_s(14), icon_color="#B71C1C",
                tooltip="Remove location",
                on_click=lambda _e, i=loc_idx: self._remove_location(i),
                style=ft.ButtonStyle(padding=ft.padding.all(_s(2)),
                                     shape=ft.RoundedRectangleBorder(radius=4))))

        header_bar = ft.Container(
            content=ft.Row(hdr_controls, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=loc_bg,
            padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(6)),
            border_radius=ft.border_radius.only(top_left=8, top_right=8))

        # Chips
        chips = [self._make_chip(t, task_idx) for t in zone_tasks]
        count = len(zone_tasks)

        body = ft.Container(
            content=ft.Column([
                header_bar,
                ft.Container(
                    content=ft.Column([
                        ft.Row(chips, wrap=True, spacing=_s(4), run_spacing=_s(4)),
                        ft.Text(f"{count} task{'s' if count != 1 else ''}",
                                size=_s(9), color="#90A4AE", italic=True),
                    ], spacing=_s(6)),
                    padding=_s(10)),
            ], spacing=0),
            border_radius=8, bgcolor="#FAFAFA",
            border=ft.border.all(2, loc_bg), width=_ZONE_WIDTH,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS)

        return ft.DragTarget(
            group="disp_tasks", content=body,
            on_accept=lambda e, li=loc_idx: self._on_drop(e, li))

    def _make_chip(self, task, task_idx):
        ci = task_idx.get(task, 0) % len(TASK_COLORS)
        bg, fg = TASK_COLORS[ci]
        chip = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.DRAG_INDICATOR, size=_s(12), color=fg),
                ft.Text(task, size=_s(11), color=fg, no_wrap=True),
            ], spacing=_s(2), tight=True),
            padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
            border_radius=16, bgcolor=bg, height=_CHIP_H)
        return ft.Draggable(
            group="disp_tasks", content=chip,
            content_feedback=ft.Container(
                ft.Text(task, size=_s(11)),
                padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
                border_radius=16, bgcolor="#E0E0E0"),
            data=task)

    # ── Actions ───────────────────────────────────────────────────────

    def _on_drop(self, e, target_idx):
        src = self.page.get_control(e.src_id)
        task = src.data
        if self.state.task_location_idx_st.get(task, 0) == target_idx:
            return
        self.state.task_location_idx_st[task] = target_idx
        self.build()

    def _add_location(self, _e):
        s = self.state
        existing = set(s.location_names_st)
        n = len(s.location_names_st) + 1
        name = f"Location {n}"
        while name in existing:
            n += 1
            name = f"Location {n}"
        s.location_names_st.append(name)
        self.build()

    def _remove_location(self, idx):
        s = self.state
        if idx <= 0 or idx >= len(s.location_names_st):
            return

        for t in s.task_location_idx_st:
            cur = s.task_location_idx_st[t]
            if cur == idx:
                s.task_location_idx_st[t] = idx - 1
            elif cur > idx:
                s.task_location_idx_st[t] -= 1

        old_tt = dict(s.travel_time_st)
        s.travel_time_st.clear()
        for (a, b), v in old_tt.items():
            if a == idx or b == idx:
                continue
            na = a - (a > idx)
            nb = b - (b > idx)
            s.travel_time_st[(min(na, nb), max(na, nb))] = v

        s.location_names_st.pop(idx)
        self.build()

    def _rename_location(self, e, idx):
        new_name = e.control.value.strip()
        if not new_name:
            e.control.value = self.state.location_names_st[idx]
            e.control.update()
            return
        self.state.location_names_st[idx] = new_name
        self.build()


