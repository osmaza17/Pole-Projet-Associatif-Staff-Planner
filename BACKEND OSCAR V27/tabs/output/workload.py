"""
Workload equity section for the Output tab.

Renders per-group and per-person workload bars, deviation info,
intra-/inter-group spread, and optional diff highlighting.
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS


def build_workload_section(sol, people, groups, diff_mode, ref_sol):
    """
    Build the WORKLOAD EQUITY controls list.

    Parameters
    ----------
    sol        : dict  – current solution (keys: workload, group_workload, …)
    people     : list  – ordered people names
    groups     : dict  – {group_name: [members]} or empty dict
    diff_mode  : str | None – "ref", "cmp", or None
    ref_sol    : dict | None – reference solution for comparison
    """
    buf = []
    buf.append(ft.Text("WORKLOAD EQUITY",
                       weight=ft.FontWeight.BOLD, size=_s(14)))

    workload     = sol.get("workload", {})
    ref_workload = ref_sol.get("workload", {}) if ref_sol else {}
    group_wl     = sol.get("group_workload", {})

    # ── Flat mode (no groups) ──────────────────────────────────────────
    if not groups or not group_wl:
        for p in people:
            ch = workload.get(p, 0)
            if diff_mode == "cmp" and ref_sol is not None:
                ph      = ref_workload.get(p, 0)
                changed = ch != ph
                buf.append(ft.Text(
                    f"  {p}: {ch:.0f}h" + (f"  (← {ph:.0f}h)" if changed else ""),
                    size=_s(12),
                    color=ft.Colors.ORANGE_700 if changed else None,
                    weight=ft.FontWeight.BOLD if changed else None))
            else:
                buf.append(ft.Text(f"  {p}: {ch:.0f}h", size=_s(12)))
        wl_vals = [workload.get(p, 0) for p in people]
        if wl_vals:
            buf.append(ft.Text(
                f"  Global: max={max(wl_vals):.0f}h  min={min(wl_vals):.0f}h  "
                f"spread={max(wl_vals)-min(wl_vals):.0f}h",
                size=_s(12), italic=True, color=ft.Colors.GREY_700))
        return buf

    # ── Grouped mode ───────────────────────────────────────────────────
    group_totals = []

    for g_idx, (gname, members) in enumerate(groups.items()):
        bg    = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
        ginfo = group_wl.get(gname, {})

        total_h  = ginfo.get("total_hours",
                             sum(workload.get(p, 0) for p in members))
        target_h = ginfo.get("target_hours",      0.0)
        dev      = ginfo.get("deviation",         total_h - target_h)
        pp_avg   = ginfo.get("per_person_avg",
                             total_h / max(len(members), 1))
        pp_tgt   = ginfo.get("per_person_target",
                             target_h / max(len(members), 1))

        group_totals.append((gname, total_h))

        member_hours = [workload.get(p, 0) for p in members if p in workload]
        intra_max    = max(member_hours) if member_hours else 0
        intra_min    = min(member_hours) if member_hours else 0
        intra_spread = intra_max - intra_min
        dev_clr      = ("#C62828" if dev < -0.5 else
                        "#2E7D32" if dev > 0.5 else "#546E7A")

        # Group header
        buf.append(ft.Container(
            content=ft.Row([
                ft.Text(f"▶  {gname}", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(13), expand=True),
                ft.Text(f"{total_h:.0f}h total", color=ft.Colors.WHITE,
                        size=_s(12), weight=ft.FontWeight.BOLD),
            ], spacing=_s(8)),
            bgcolor=bg,
            padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            border_radius=ft.border_radius.only(top_left=6, top_right=6),
            margin=ft.margin.only(top=_s(6))))

        # Stats bar
        buf.append(ft.Container(
            content=ft.Row([
                ft.Text(f"Target: {target_h:.1f}h", size=_s(11),
                        color=ft.Colors.GREY_800),
                ft.Container(width=1, height=_s(12),
                             bgcolor=ft.Colors.GREY_400),
                ft.Text(f"Dev: {dev:+.1f}h", size=_s(11),
                        color=dev_clr, weight=ft.FontWeight.BOLD),
                ft.Container(width=1, height=_s(12),
                             bgcolor=ft.Colors.GREY_400),
                ft.Text(f"Avg/person: {pp_avg:.1f}h  (tgt {pp_tgt:.1f}h)",
                        size=_s(11), color=ft.Colors.GREY_700),
                ft.Container(width=1, height=_s(12),
                             bgcolor=ft.Colors.GREY_400),
                ft.Text(f"Intra-spread: {intra_spread:.0f}h  "
                        f"(↑{intra_max:.0f}h  ↓{intra_min:.0f}h)",
                        size=_s(11), color=ft.Colors.GREY_700),
            ], spacing=_s(10), wrap=True),
            bgcolor="#ECEFF1",
            padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(4)),
            border=ft.border.only(
                left=ft.border.BorderSide(3, bg),
                right=ft.border.BorderSide(1, "#CFD8DC"))))

        # Per-person bars
        bar_max = max(intra_max, 1)
        for p in members:
            ch       = workload.get(p, 0)
            bar_fill = round((ch / bar_max) * 20)
            bar_str  = "█" * bar_fill + "░" * (20 - bar_fill)

            if diff_mode == "cmp" and ref_sol is not None:
                ph      = ref_workload.get(p, 0)
                changed = ch != ph
                suffix  = f"  ← was {ph:.0f}h" if changed else ""
                clr     = ft.Colors.ORANGE_700 if changed else ft.Colors.BLACK
                wgt     = ft.FontWeight.BOLD if changed else None
            else:
                suffix, clr, wgt = "", ft.Colors.BLACK, None

            buf.append(ft.Container(
                content=ft.Row([
                    ft.Text(p, size=_s(12), width=_s(110), color=clr, weight=wgt,
                            no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(f"{ch:.0f}h", size=_s(12), width=_s(32),
                            color=clr, weight=wgt,
                            text_align=ft.TextAlign.RIGHT),
                    ft.Text(bar_str, size=_s(9), color=bg,
                            font_family="monospace"),
                    ft.Text(suffix, size=_s(11), color=ft.Colors.ORANGE_700,
                            italic=True),
                ], spacing=_s(6)),
                padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(2)),
                bgcolor=ft.Colors.WHITE,
                border=ft.border.only(
                    left=ft.border.BorderSide(3, bg),
                    bottom=ft.border.BorderSide(1, "#ECEFF1"))))

    # ── Inter-group summary ────────────────────────────────────────────
    if group_totals:
        g_max_name, g_max_val = max(group_totals, key=lambda x: x[1])
        g_min_name, g_min_val = min(group_totals, key=lambda x: x[1])
        inter_spread = g_max_val - g_min_val
        all_wl = [workload.get(p, 0) for p in people]
        buf.append(ft.Container(
            content=ft.Column([
                ft.Text("◎  Inter-group summary",
                        weight=ft.FontWeight.BOLD, size=_s(12),
                        color=ft.Colors.WHITE),
                ft.Row([
                    ft.Text(f"Inter-group spread: {inter_spread:.0f}h",
                            size=_s(12), color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD),
                    ft.Text(f"(max: {g_max_name} {g_max_val:.0f}h  ·  "
                            f"min: {g_min_name} {g_min_val:.0f}h)",
                            size=_s(11), color="#B0BEC5"),
                ], spacing=_s(10), wrap=True),
                ft.Text(
                    f"Global individual: max={max(all_wl):.0f}h  "
                    f"min={min(all_wl):.0f}h  "
                    f"spread={max(all_wl)-min(all_wl):.0f}h"
                    if all_wl else "",
                    size=_s(11), color="#B0BEC5"),
            ], spacing=_s(4)),
            bgcolor="#37474F",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=6, margin=ft.margin.only(top=_s(8))))

    return buf