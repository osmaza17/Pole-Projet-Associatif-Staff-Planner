"""
builders.py — stateless UI builders for the Dimensions tab.

Contains the task priority/rotation/duration list and the
hours-per-day column builder.
"""

import flet as ft
from constants import _s, DEFAULT_HOURS_TEXT
from ui_helpers import UIHelpers

# ── Constants for rotation mode ───────────────────────────────────

_MODE_CYCLE = ["R", "N", "S"]
_MODE_INDEX = {"R": 0, "N": 1, "S": 2}

# ── Notion palette ────────────────────────────────────────────────

_NOTION_TEXT        = "#37352F"
_NOTION_TEXT_MUTED  = "#9B9A97"
_NOTION_DIVIDER     = "#EDEDEC"
_NOTION_PILL_BG     = "#F1F1EF"


# ── Task priority / rotation / duration list ──────────────────────

def build_rotation_list(state, rotation_col: ft.ReorderableListView,
                        page: ft.Page, on_solve_blocked_update):
    """Populate *rotation_col* with one row per task."""
    s = state
    _, tasks, _, _ = s.dims()

    if not hasattr(s, 'task_duration_st'):
        s.task_duration_st = {}

    buf = []
    N = len(tasks)

    for rank, t in enumerate(tasks):
        priority_weight = 1 + (N - rank)

        drag_handle = ft.Icon(
            ft.Icons.DRAG_INDICATOR, size=_s(14),
            color=_NOTION_TEXT_MUTED, tooltip="Drag to reorder priority")

        rank_badge = ft.Container(
            content=ft.Text(f"{rank + 1}", size=_s(10),
                            weight=ft.FontWeight.W_500,
                            color=_NOTION_TEXT_MUTED,
                            text_align=ft.TextAlign.CENTER),
            width=_s(22), height=_s(22),
            alignment=ft.alignment.center,
            bgcolor=_NOTION_PILL_BG,
            border_radius=4,
            tooltip=f"Priority weight: {priority_weight}×")

        mode = s.rotation_st.get(t, "R")
        seg = ft.SegmentedButton(
            selected={str(_MODE_INDEX[mode])},
            allow_multiple_selection=False,
            allow_empty_selection=False,
            show_selected_icon=False,
            style=ft.ButtonStyle(
                padding=ft.padding.symmetric(horizontal=_s(4), vertical=_s(0)),
                text_style=ft.TextStyle(size=_s(9),
                                        weight=ft.FontWeight.W_500),
                side=ft.BorderSide(1, _NOTION_DIVIDER),
                bgcolor={
                    ft.ControlState.SELECTED: _NOTION_PILL_BG,
                    ft.ControlState.DEFAULT:  ft.Colors.TRANSPARENT,
                },
                color={
                    ft.ControlState.SELECTED: _NOTION_TEXT,
                    ft.ControlState.DEFAULT:  _NOTION_TEXT_MUTED,
                },
            ),
            segments=[
                ft.Segment(value="0", label=ft.Text("ROT",   size=_s(9))),
                ft.Segment(value="1", label=ft.Text("OFF",   size=_s(9))),
                ft.Segment(value="2", label=ft.Text("STICK", size=_s(9))),
            ],
            data=t)

        def _on_seg(e):
            idx = int(next(iter(e.control.selected)))
            s.rotation_st[e.control.data] = _MODE_CYCLE[idx]

        seg.on_change = _on_seg

        s.task_duration_st.setdefault(t, "1")
        dur_tf = ft.TextField(
            value=s.task_duration_st[t],
            width=_s(62), height=_s(30), text_size=_s(11),
            text_align=ft.TextAlign.CENTER,
            suffix_text="h",
            content_padding=ft.padding.symmetric(
                horizontal=_s(6), vertical=_s(2)),
            border_color=_NOTION_DIVIDER,
            focused_border_color=_NOTION_TEXT_MUTED,
            cursor_color=_NOTION_TEXT,
            border_radius=4,
            input_filter=ft.NumbersOnlyInputFilter(),
            data=t)

        def _on_dur(e):
            s.task_duration_st[e.control.data] = e.control.value
            s.invalidate_cache()
            on_solve_blocked_update()

        dur_tf.on_change = _on_dur

        row = ft.Container(
            key=f"task_row_{t}",
            content=ft.Row(
                [drag_handle, rank_badge,
                 UIHelpers.lbl(t, _s(140)), seg, dur_tf],
                spacing=_s(10),
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.symmetric(
                horizontal=_s(8), vertical=_s(4)),
            border=ft.border.only(
                bottom=ft.border.BorderSide(1, _NOTION_DIVIDER)))
        buf.append(row)

    rotation_col.controls = buf
    page.update()


def handle_reorder(e: ft.OnReorderEvent, state, tf_tasks: ft.TextField,
                   rotation_col: ft.ReorderableListView,
                   page: ft.Page, on_solve_blocked_update):
    """Process a drag-and-drop reorder event on the task list."""
    _, tasks, _, _ = state.dims()

    if not tasks:
        return
    old_idx, new_idx = e.old_index, e.new_index
    if old_idx == new_idx or not (0 <= old_idx < len(tasks)):
        return

    item = tasks.pop(old_idx)
    new_idx = max(0, min(new_idx, len(tasks)))
    tasks.insert(new_idx, item)

    new_text = "\n".join(tasks)
    state.tasks_text = new_text
    tf_tasks.value = new_text
    state.invalidate_cache()

    build_rotation_list(state, rotation_col, page, on_solve_blocked_update)
    on_solve_blocked_update()


# ── Hours-per-day column ──────────────────────────────────────────

def build_hours_per_day(state, hours_col: ft.ListView,
                        page: ft.Page, on_solve_blocked_update):
    """Populate *hours_col* with one TextField per day."""
    s = state
    _, _, _, days = s.dims()

    buf = [ft.Text("Hours per Day", weight=ft.FontWeight.BOLD, size=_s(12))]
    for j in days:
        s.hours_per_day.setdefault(j, DEFAULT_HOURS_TEXT)
        tf = ft.TextField(
            value=s.hours_per_day[j], multiline=True,
            min_lines=4, max_lines=24, label=j,
            width=_s(120), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)), data=j)

        def _ch(e, _j=j):
            s.hours_per_day[_j] = e.control.value
            s.invalidate_cache()
            on_solve_blocked_update()

        tf.on_change = _ch
        buf.append(tf)

    hours_col.controls = buf
    page.update()