"""
Run column builder for the Output tab.

Pure functions that build a single run column (header + grids + legends +
violations + workload + task hours). Receives data and callbacks; never
touches OutputTab state directly.

Scale conventions:
    _s        — container margins / paddings / list spacing.
    _s_stats  — top banner buttons, run-level stats strip, "Solving…"
                indicators, and every non-grid content piece.
    _s_grid   — output grid cell dimensions, day headers that sit
                above the grid, and the grid legend.
"""

import flet as ft
from constants import (
    _s, _s_stats, _s_grid,
    TASK_COLORS, loc_color,
    UNAVAIL_COLOR, EMERG_COLOR, AVAIL_COLOR,
    DIFF_ADD_COLOR, DIFF_REMOVE_COLOR, DIFF_CHANGE_COLOR,
    BASE_ACTIVE_BG, BASE_ACTIVE_FG, BASE_IDLE_BG, BASE_IDLE_FG,
    TRAVEL_COLOR, TRAVEL_FG_COLOR, TRAVEL_LABEL,
    GROUP_HEADER_COLORS,
)
from .grids import build_task_grid, build_person_grid
from .workload import build_workload_section
from .output_sections import (
    build_violations_section, build_variety_section,
    build_emergency_section, build_task_hours_section,
    build_rest_enforcement_section,
    build_coverage_gaps_section,
)


# ── Status colour mapping for per-day chips ───────────────────────────

_DAY_STATUS_STYLE: dict[str, tuple[str, str]] = {
    # status_text → (bgcolor, text_color)
    "Optimal":                 ("#4CAF50", ft.Colors.WHITE),
    "Suboptimal":              ("#FFC107", ft.Colors.BLACK),
    "Time Limit Reached":      ("#FF9800", ft.Colors.BLACK),
    "Infeasible":              ("#F44336", ft.Colors.WHITE),
    "Infeasible or Unbounded": ("#D32F2F", ft.Colors.WHITE),
    "Interrupted by User":     ("#9E9E9E", ft.Colors.WHITE),
    "Not attempted":           ("#616161", ft.Colors.WHITE),
}
_DEFAULT_STATUS_STYLE = ("#795548", ft.Colors.WHITE)


# ── Header sub-builders ───────────────────────────────────────────────

def _diff_button(is_live, diff_mode, both_selected, on_diff_click):
    """Builds the comparison toggle button."""
    if is_live:
        lbl, bg, fg, click = "Compare", ft.Colors.GREY_400, ft.Colors.GREY_600, None
    elif diff_mode == "ref":
        lbl, bg, fg, click = "Reference", ft.Colors.LIGHT_BLUE_300, ft.Colors.BLACK, on_diff_click
    elif diff_mode == "cmp":
        lbl, bg, fg, click = "Compared", ft.Colors.AMBER_300, ft.Colors.BLACK, on_diff_click
    elif both_selected:
        lbl, bg, fg, click = "Compare", ft.Colors.GREY_400, ft.Colors.GREY_600, None
    else:
        lbl, bg, fg, click = "Compare", ft.Colors.GREEN_200, ft.Colors.BLACK, on_diff_click
        
    return ft.Container(
        content=ft.Text(lbl, size=_s_stats(11), color=fg,
                        weight=ft.FontWeight.BOLD, no_wrap=True),
        bgcolor=bg, padding=ft.padding.symmetric(_s_stats(4), _s_stats(8)),
        border_radius=4, on_click=click)


def _mode_toggle(view_mode, on_view_mode_click):
    """Builds the view mode toggle buttons (Person, Task, Location)."""
    def btn(lbl, mode_val):
        selected = (view_mode == mode_val)
        return ft.Container(
            content=ft.Text(lbl, size=_s_stats(11),
                            color=ft.Colors.WHITE if selected else ft.Colors.DEEP_PURPLE_900,
                            weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=ft.Colors.DEEP_PURPLE_700 if selected else ft.Colors.DEEP_PURPLE_100,
            padding=ft.padding.symmetric(_s_stats(4), _s_stats(8)),
            border_radius=4,
            on_click=lambda e, _m=mode_val: on_view_mode_click(_m))
            
    return ft.Row([
        btn("👤 Person", "person"),
        btn("📋 Task", "task"),
        btn("🗺 Location", "location"),
    ], spacing=_s_stats(2))


def _build_header(run_idx, sol, is_live, is_solving, view_mode, diff_mode,
                  both_selected, base_selected,
                  on_base_click, on_diff_click, on_view_mode_click, on_delete):
    """Builds the top header of a run column including status, controls, and active base."""
    status_str = str(sol.get('status') or 'Solving...')
    hdr_lbl = (f"⟳  Run #{run_idx+1}  ·  Solving…"
               if is_live and is_solving
               else f"Run #{run_idx+1}  ·  {status_str}")
    hdr_bg = "#E65100" if is_live and is_solving else "#1565C0"

    close_btn = (
        [ft.IconButton(icon=ft.Icons.CLOSE, icon_color=ft.Colors.WHITE,
                       icon_size=_s_stats(16), tooltip="Remove this run",
                       style=ft.ButtonStyle(padding=ft.padding.all(_s_stats(2))),
                       on_click=lambda e: on_delete())]
        if on_delete else [])

    title_row = ft.Row(
        controls=[ft.Text(hdr_lbl, weight=ft.FontWeight.BOLD, size=_s_stats(13),
                          color=ft.Colors.WHITE, no_wrap=True, expand=True),
                  *close_btn],
        spacing=_s_stats(4),
        vertical_alignment=ft.CrossAxisAlignment.CENTER)

    diff_btn = _diff_button(is_live, diff_mode, both_selected, on_diff_click)
    mode_tog = _mode_toggle(view_mode, on_view_mode_click)

    if is_live:
        body = ft.Column([title_row,
                          ft.Row([diff_btn, mode_tog], spacing=_s_stats(6))],
                         spacing=_s_stats(4), tight=True)
    else:
        base_btn = ft.Container(
            content=ft.Text(
                "✓ Active Base" if base_selected else "📌 Set as Base",
                size=_s_stats(11),
                color=BASE_ACTIVE_FG if base_selected else BASE_IDLE_FG,
                weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=BASE_ACTIVE_BG if base_selected else BASE_IDLE_BG,
            padding=ft.padding.symmetric(_s_stats(4), _s_stats(8)),
            border_radius=4,
            on_click=lambda e: on_base_click())
            
        body = ft.Column([
            title_row,
            ft.Row([base_btn, diff_btn, mode_tog], spacing=_s_stats(6),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ], spacing=_s_stats(4), tight=True)

    return ft.Container(
        content=body, bgcolor=hdr_bg,
        padding=ft.padding.symmetric(_s_stats(8), _s_stats(12)),
        border_radius=6, margin=ft.margin.only(bottom=_s(6)))


# ── Per-day status chips ──────────────────────────────────────────────

def _build_day_status_chips(day_statuses: dict, day_mip_gaps: dict) -> ft.Control:
    """
    Build a compact row of coloured chips, one per day, showing
    the solver status (and MIP gap where relevant).
    """
    chips = []
    for day, status in day_statuses.items():
        bg, fg = _DAY_STATUS_STYLE.get(status, _DEFAULT_STATUS_STYLE)
        gap = day_mip_gaps.get(day, 0.0)
        
        # Short label: day name + abbreviated status
        _abbrev = {
            "Optimal": "✓",
            "Suboptimal": "Sub",
            "Time Limit Reached": "TL",
            "Infeasible": "✗",
            "Infeasible or Unbounded": "✗/U",
            "Interrupted by User": "Stop",
            "Not attempted": "—",
        }
        short = _abbrev.get(status, "?")
        
        # Tooltip with full detail
        tip = f"{day}: {status}"
        if gap > 0 and status not in ("Infeasible", "Infeasible or Unbounded",
                                      "Not attempted"):
            tip += f"  (gap {gap*100:.3f}%)"

        # Robust way to apply tooltips without an explicit wrapping component.
        chips.append(
            ft.Container(
                tooltip=tip,
                content=ft.Text(f"{day} {short}", size=_s_stats(10),
                                color=fg, weight=ft.FontWeight.BOLD,
                                no_wrap=True),
                bgcolor=bg,
                padding=ft.padding.symmetric(_s_stats(3), _s_stats(6)),
                border_radius=4
            )
        )

    return ft.Container(
        content=ft.Row(chips, spacing=_s_stats(4), wrap=True),
        padding=ft.padding.only(bottom=_s(4)))


# ── Legend builder (sits inside the grid section → _s_grid) ───────────

def _build_legend(loc_view, tasks, tc, location_names, task_location_idx,
                  has_travel, diff_mode):
    """Builds the color legend for tasks, locations, travel, and availability states."""
    # Explicit typing to prevent Pylance "list[Row] is not assignable to Iterable[Container]"
    legend: list[ft.Control]
    
    if loc_view:
        locs_in_run = sorted({task_location_idx.get(t, 0) for t in tasks})
        legend = [
            ft.Container(
                ft.Text(location_names[li] if li < len(location_names) else f"Loc {li}",
                        size=_s_grid(10), weight=ft.FontWeight.BOLD,
                        color=loc_color(li)[1]),
                bgcolor=loc_color(li)[0],
                padding=ft.padding.symmetric(_s_grid(4), _s_grid(8)),
                border_radius=4)
            for li in locs_in_run]
    else:
        legend = [
            ft.Container(
                ft.Text(t, size=_s_grid(10), weight=ft.FontWeight.BOLD,
                        color=tc[t][1]),
                bgcolor=tc[t][0],
                padding=ft.padding.symmetric(_s_grid(4), _s_grid(8)),
                border_radius=4)
            for t in tasks]

    if has_travel:
        legend.append(ft.Container(
            ft.Text("🚗 Travel", size=_s_grid(10), weight=ft.FontWeight.BOLD,
                    color=TRAVEL_FG_COLOR),
            bgcolor=TRAVEL_COLOR,
            padding=ft.padding.symmetric(_s_grid(4), _s_grid(8)),
            border_radius=4))

    for lbl, clr in [("Available", AVAIL_COLOR),
                     ("Emergency", EMERG_COLOR),
                     ("Unavailable", UNAVAIL_COLOR)]:
        legend.append(ft.Container(
            ft.Text(lbl, size=_s_grid(10), color=ft.Colors.WHITE),
            bgcolor=clr,
            padding=ft.padding.symmetric(_s_grid(4), _s_grid(8)),
            border_radius=4))

    if diff_mode == "cmp":
        for char, color, desc in [
            ("+", DIFF_ADD_COLOR, "New assignment"),
            ("−", DIFF_REMOVE_COLOR, "Removed"),
            ("⇄", DIFF_CHANGE_COLOR, "Changed"),
        ]:
            legend.append(ft.Row([
                ft.Container(
                    ft.Text(char, size=_s_grid(8), color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
                    width=_s_grid(13), height=_s_grid(13), bgcolor=color,
                    border_radius=7, alignment=ft.alignment.center),
                ft.Text(desc, size=_s_grid(10)),
            ], spacing=_s_grid(4),
                vertical_alignment=ft.CrossAxisAlignment.CENTER))
                
    return legend


# ── Main entry point ──────────────────────────────────────────────────

def build_run_column(*, run_idx, sol, people, tasks, hours, days,
                     availability, emergency, groups, is_live,
                     view_mode, diff_mode, ref_sol, base_selected,
                     both_diff_selected, person_colors,
                     task_location_idx, location_names,
                     on_base_click, on_diff_click, on_view_mode_click,
                     on_delete):
    """
    Main constructor for a single output column containing all grids, 
    statistics, headers, and specific sections.
    """
    # Safe retrieval to prevent TypeErrors and KeyErrors
    status = str(sol.get("status") or "Solving...")
    is_solving = "Solving" in status
    assignment = sol.get("assignment", {})
    task_view = (view_mode == "task")
    loc_view = (view_mode == "location")

    # Explicit typing to prevent Pylance object assignment errors
    buf: list[ft.Control] = [_build_header(run_idx, sol, is_live, is_solving, view_mode,
                                           diff_mode, both_diff_selected, base_selected,
                                           on_base_click, on_diff_click, on_view_mode_click,
                                           on_delete)]

    # Run stats strip (solve time + MIP gap) — _s_stats: part of the info layer.
    if not (is_live and is_solving):
        buf.append(ft.Container(
            content=ft.Row([
                ft.Text(f"⏱ {sol.get('solve_time', 0.0):.1f}s",
                        size=_s_stats(12), color=ft.Colors.GREY_700),
                ft.Text(f"◎ Gap {sol.get('mip_gap', 0.0)*100:.3f}%",
                        size=_s_stats(12), color=ft.Colors.GREY_700),
            ], spacing=_s_stats(12)),
            padding=ft.padding.only(bottom=_s(4))))

        # Per-day status chips
        day_statuses = sol.get("day_statuses", {})
        day_mip_gaps = sol.get("day_mip_gaps", {})
        if day_statuses:
            show_chips = (
                len(day_statuses) > 1
                or any(s != "Optimal" for s in day_statuses.values()))
            if show_chips:
                buf.append(_build_day_status_chips(day_statuses, day_mip_gaps))

    # Robust dimension calculations using generators with safe defaults
    max_task_len = max((len(t) for t in tasks), default=0)
    max_p_len = max((len(p) for p in people), default=0)
    max_h_len = max((len(h) for h in hours.get(days[0], [])) if days else [], default=0)

    # Grid cell dimensions
    CW = _s_grid(min(80, max(55, (max(max_task_len, max_h_len) + 4) * 8))) + _s_grid(16)
    NW = _s_grid(max(75, (max_p_len + 4) * 8)) + _s_grid(16)
    TW = _s_grid(50) + _s_grid(16)
    Ch = _s_grid(34)

    if is_live and is_solving and not any(assignment.get(d) for d in days):
        # Full-screen loading indicator
        buf.append(ft.Row([
            ft.ProgressRing(width=_s_stats(22), height=_s_stats(22), stroke_width=3),
            ft.Text("Solving in background…", italic=True, size=_s_stats(12)),
        ], spacing=_s_stats(10)))
    else:
        if is_live and is_solving:
            # Inline loading indicator
            buf.append(ft.Row([
                ft.ProgressRing(width=_s_stats(16), height=_s_stats(16), stroke_width=2),
                ft.Text("Updating…", italic=True, size=_s_stats(11),
                        color=ft.Colors.ORANGE_700),
            ], spacing=_s_stats(6)))

        if loc_view:
            tc = {t: loc_color(task_location_idx.get(t, 0)) for t in tasks}
        else:
            tc = {t: TASK_COLORS[i % len(TASK_COLORS)] for i, t in enumerate(tasks)}
        tc[TRAVEL_LABEL] = (TRAVEL_COLOR, TRAVEL_FG_COLOR)

        ref_asgn = ref_sol.get("assignment", {}) if ref_sol else {}

        for j in days:
            if j not in assignment:
                continue
            
            # Safe access to the daily hours dictionary
            day_hrs = hours.get(j, [])
            
            # Day header — sits above the grid, scales with _s_grid
            buf.append(ft.Text(f"-- {j} --",
                               weight=ft.FontWeight.BOLD, size=_s_grid(16)))

            asgn = assignment[j]
            ref_day = ref_asgn.get(j, {}) if ref_asgn else {}

            if task_view:
                buf.extend(build_task_grid(
                    j, day_hrs, tasks, people, asgn,
                    availability, emergency, tc, NW, CW, TW, Ch,
                    diff_mode=diff_mode,
                    ref_asgn_day=ref_day if ref_asgn else None))
            else:
                buf.extend(build_person_grid(
                    j, day_hrs, people, asgn, person_colors,
                    availability, emergency, tc, NW, CW, TW, Ch,
                    diff_mode=diff_mode,
                    ref_asgn_day=ref_day if ref_asgn else None))

            buf.append(ft.Row(_build_legend(
                loc_view, tasks, tc, location_names, task_location_idx,
                bool(sol.get("travel_label")), diff_mode),
                spacing=_s_grid(8), wrap=True))
            buf.append(ft.Divider())

        if not (is_live and is_solving):
            # Hard-violation sections first

            buf.extend(build_coverage_gaps_section(sol.get("coverage_gaps_messages", [])))

            for title, key in [
                ("CAPTAIN RULES",     "captain_violations"),
                ("QUOTA FULFILMENT",  "quota_violations"),
                ("ENEMY SEPARATION",  "enemy_violations"),
            ]:
                buf.extend(build_violations_section(title, sol.get(key, [])))

            # Informational sections
            buf.extend(build_rest_enforcement_section(
                sol.get("enforced_rest", False),
                sol.get("consec_relaxations", {}),
                days=days))

            buf.extend(build_emergency_section(sol.get("emerg_issues", [])))
            buf.extend(build_workload_section(
                sol, people, groups or {}, diff_mode, ref_sol))
            buf.extend(build_variety_section(sol.get("variety_report", []), tasks))

            # Build person→group colour mapping for consistent tinting
            if groups:
                person_group_color = {
                    p: GROUP_HEADER_COLORS[i % len(GROUP_HEADER_COLORS)]
                    for i, (_g, members) in enumerate(groups.items())
                    for p in members
                }
            else:
                person_group_color = {}

            buf.extend(build_task_hours_section(
                sol.get("task_workload", {}), people, tasks,
                person_group_color=person_group_color))

    # Outer container and dynamic column width adjustment
    max_hrs = max((len(hours.get(d, [])) for d in days), default=0)
    col_width = max(NW + CW * max_hrs + TW + _s(30), _s(500))
    
    return ft.Container(
        content=ft.ListView(controls=buf, spacing=_s(5), expand=True),
        width=col_width,
        padding=ft.padding.symmetric(horizontal=_s(16)),
        border=ft.border.only(right=ft.border.BorderSide(1, "#B0BEC5")))