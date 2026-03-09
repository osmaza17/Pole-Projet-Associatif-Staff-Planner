"""
Grid builders for the Output tab.

Two alternative visualisations of the assignment matrix:
  • build_task_grid   — tasks on Y-axis, hours on X-axis (cells = list of names)
  • build_person_grid — people on Y-axis, hours on X-axis (cells = task label)

Both are pure functions (no class / no self) so they are easy to test in isolation.
"""

import flet as ft
from constants import (
    _s,
    UNAVAIL_COLOR, EMERG_COLOR, AVAIL_COLOR,
    DIFF_ADD_COLOR, DIFF_REMOVE_COLOR, DIFF_CHANGE_COLOR,
)


# ── Shared helpers ──────────────────────────────────────────────────────

def _diff_icon(task, ref_task):
    """Small coloured badge indicating add / remove / change."""
    if task == ref_task:
        return None
    if ref_task is None:
        char, color = "+", DIFF_ADD_COLOR
    elif task is None:
        char, color = "−", DIFF_REMOVE_COLOR
    else:
        char, color = "⇄", DIFF_CHANGE_COLOR
    return ft.Container(
        content=ft.Text(char, size=_s(8), color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
        width=_s(13), height=_s(13), bgcolor=color, border_radius=7,
        alignment=ft.alignment.center, right=1, top=1)


def _header_cell(txt, w, h):
    """Column-header cell reused by both grid variants."""
    return ft.Container(
        ft.Text(txt, size=_s(11), weight=ft.FontWeight.BOLD,
                color=ft.Colors.WHITE),
        width=w, height=h, bgcolor="#546E7A",
        alignment=ft.alignment.center,
        border=ft.border.all(1, "#455A64"))


# ── Task-view grid ─────────────────────────────────────────────────────

def build_task_grid(day, hours_list, tasks, people, asgn_day,
                    availability, emergency, tc,
                    NW, CW, TW, Ch,
                    diff_mode=None, ref_asgn_day=None):
    """
    Tasks on Y-axis, hours on X-axis.
    Each cell lists the names of people assigned to that task at that hour.
    """
    buf = []

    # Header row
    buf.append(ft.Row(
        [_header_cell("Task", NW, Ch)] +
        [_header_cell(h, CW, Ch) for h in hours_list] +
        [_header_cell("Total", TW, Ch)],
        spacing=0, wrap=False))

    for t_idx, t in enumerate(tasks):
        row_bg     = "#ECEFF1" if t_idx % 2 == 0 else "#FFFFFF"
        t_bg, t_fg = tc[t]

        assigned_per_hour = {
            h: [p for p in people if asgn_day.get(p, {}).get(h) == t]
            for h in hours_list
        }
        ref_per_hour = {}
        if diff_mode == "cmp" and ref_asgn_day is not None:
            ref_per_hour = {
                h: [p for p in people if ref_asgn_day.get(p, {}).get(h) == t]
                for h in hours_list
            }
        max_names = max(
            (len(set(assigned_per_hour.get(h, [])) | set(ref_per_hour.get(h, [])))
             for h in hours_list),
            default=0)
        row_h = max(Ch, _s(4) + _s(18) * max(max_names, 1))

        # Task-name cell
        name_cell = ft.Container(
            ft.Text(t, size=_s(12), weight=ft.FontWeight.BOLD, color=t_fg,
                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
            width=NW, height=row_h, bgcolor=t_bg,
            alignment=ft.alignment.top_left,
            padding=ft.padding.only(left=_s(8), top=_s(4)),
            border=ft.border.all(1, "#CFD8DC"))

        cells = [name_cell]
        total = 0

        for h in hours_list:
            assigned = assigned_per_hour[h]
            total   += len(assigned)

            # Outline colour based on availability / emergency
            if assigned:
                avail = availability.get((assigned[0], h, day), 1)
                griev = emergency.get((assigned[0], h, day), 0)
            else:
                avail, griev = 1, 0

            if diff_mode == "cmp":
                outline = None
            elif avail == 0:   outline = UNAVAIL_COLOR
            elif griev == 1:   outline = EMERG_COLOR
            else:              outline = None

            if assigned or (diff_mode == "cmp" and ref_per_hour.get(h)):
                name_rows = _build_diff_name_rows(
                    assigned, diff_mode, ref_per_hour.get(h, []),
                    ref_asgn_day, t_fg)

                inner = ft.Column(name_rows, spacing=_s(1), tight=True)
                inner_cell = ft.Container(
                    inner, width=CW - _s(4), height=row_h - _s(4),
                    bgcolor=t_bg,
                    alignment=ft.alignment.top_left,
                    padding=ft.padding.only(left=_s(8), top=_s(6)),
                    border_radius=6,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS)
                cell = ft.Container(
                    inner_cell,
                    width=CW, height=row_h,
                    bgcolor=outline if outline else row_bg,
                    border_radius=7,
                    alignment=ft.alignment.center,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS)
            else:
                cell = ft.Container(
                    width=CW, height=row_h,
                    bgcolor=outline if outline else row_bg,
                    border_radius=6,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS)

            cells.append(cell)

        # Total cell
        cells.append(ft.Container(
            ft.Text(str(int(total)), size=_s(12), weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER),
            width=TW, height=row_h, bgcolor=row_bg,
            alignment=ft.alignment.center,
            border=ft.border.all(1, "#CFD8DC")))

        buf.append(ft.Row(cells, spacing=0, wrap=False,
                          vertical_alignment=ft.CrossAxisAlignment.START))

    return buf


# ── Person-view grid ───────────────────────────────────────────────────

def build_person_grid(day, hours_list, people, asgn_day, person_colors,
                      availability, emergency, tc,
                      NW, CW, TW, Ch,
                      diff_mode=None, ref_asgn_day=None):
    """
    People on Y-axis, hours on X-axis.
    Each cell shows the task label (colour-coded).
    """
    buf = []

    # Header row
    buf.append(ft.Row(
        [_header_cell("Person", NW, Ch)] +
        [_header_cell(h, CW, Ch) for h in hours_list] +
        [_header_cell("Total", TW, Ch)],
        spacing=0, wrap=False))

    for idx_p, p in enumerate(people):
        row_bg  = "#ECEFF1" if idx_p % 2 == 0 else "#FFFFFF"
        p_color = person_colors.get(p, ft.Colors.BLACK)
        cells   = [ft.Container(
            ft.Text(p, size=_s(12), weight=ft.FontWeight.BOLD, color=p_color),
            width=NW, height=Ch, bgcolor=row_bg,
            alignment=ft.alignment.center_left,
            padding=ft.padding.only(left=_s(8)),
            border=ft.border.all(1, "#CFD8DC"))]
        total = 0

        for h in hours_list:
            task  = asgn_day.get(p, {}).get(h)
            avail = availability.get((p, h, day), 1)
            griev = emergency.get((p, h, day), 0)

            if avail == 0:     brd = ft.border.all(1.5, UNAVAIL_COLOR)
            elif griev == 1:   brd = ft.border.all(1.5, EMERG_COLOR)
            else:              brd = ft.border.all(0.5, AVAIL_COLOR)

            if task:
                bg, fg = tc[task]
                total += 1
                cell = ft.Container(
                    ft.Text(task, size=_s(11), weight=ft.FontWeight.BOLD,
                            color=fg, text_align=ft.TextAlign.CENTER),
                    width=CW, height=Ch, bgcolor=bg,
                    alignment=ft.alignment.center, border=brd, border_radius=4)
            else:
                cell = ft.Container(
                    width=CW, height=Ch, bgcolor=row_bg, border=brd)

            if diff_mode == "cmp" and ref_asgn_day is not None:
                ref_task = ref_asgn_day.get(p, {}).get(h)
                icon = _diff_icon(task, ref_task)
                if icon is not None:
                    cell = ft.Stack(controls=[cell, icon], width=CW, height=Ch)

            cells.append(cell)

        # Total cell
        cells.append(ft.Container(
            ft.Text(str(int(total)), size=_s(12), weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER),
            width=TW, height=Ch, bgcolor=row_bg,
            alignment=ft.alignment.center,
            border=ft.border.all(1, "#CFD8DC")))
        buf.append(ft.Row(cells, spacing=0, wrap=False))

    return buf


# ── Internal: diff name rows for task-view cells ──────────────────────

def _build_diff_name_rows(assigned, diff_mode, ref_list, ref_asgn_day, t_fg):
    """Build the list of ft.Text / ft.Row controls inside a task-view cell."""
    name_rows = []

    if diff_mode == "cmp" and ref_asgn_day is not None:
        ref_here  = set(ref_list)
        cur_here  = set(assigned)
        added     = cur_here - ref_here
        removed   = ref_here - cur_here
        unchanged = cur_here & ref_here

        for p in [x for x in assigned if x in unchanged]:
            name_rows.append(
                ft.Text(p, size=_s(10), color=t_fg,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS))
        for p in [x for x in assigned if x in added]:
            name_rows.append(ft.Row([
                ft.Container(
                    ft.Text("+", size=_s(7), color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
                    width=_s(12), height=_s(12), bgcolor=DIFF_ADD_COLOR,
                    border_radius=6, alignment=ft.alignment.center),
                ft.Text(p, size=_s(10), color=t_fg, weight=ft.FontWeight.BOLD,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
            ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER,
               tight=True))
        for p in sorted(removed):
            name_rows.append(ft.Row([
                ft.Container(
                    ft.Text("−", size=_s(7), color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
                    width=_s(12), height=_s(12), bgcolor=DIFF_REMOVE_COLOR,
                    border_radius=6, alignment=ft.alignment.center),
                ft.Text(p, size=_s(10), color=ft.Colors.GREY_500,
                        italic=True,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
            ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER,
               tight=True))
    else:
        for p in assigned:
            name_rows.append(
                ft.Text(p, size=_s(10), color=t_fg,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS))

    return name_rows