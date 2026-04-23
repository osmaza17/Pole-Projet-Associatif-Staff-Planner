"""
Workload equity section for the Output tab.

Three render paths, selected by the solver's eq_group_mode:

  • "shares" — user-defined per-group ratios. Strong visual emphasis:
      coloured group headers, target/dev as KPI, share% badge, and an
      inter-group spread summary at the bottom as the "headline" metric.

  • "off"    — solver balances ONE global pool by individual capacity.
      Per-group figures are shown as *reference* (muted headers,
      neutral palette, capacity-derived share% badge) and the headline
      metric is the global individual spread, placed at the top.

  • flat     — no groups defined (defensive fallback).

Diff highlighting (ref vs cmp) is supported in all three modes.
"""

import flet as ft
from constants import _s_stats, GROUP_HEADER_COLORS


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def build_workload_section(sol, people, groups, diff_mode, ref_sol):
    """
    Parameters
    ----------
    sol        : dict  – current solution (must contain eq_group_mode,
                         workload, group_workload)
    people     : list  – ordered people names
    groups     : dict  – {group_name: [members]} or empty dict
    diff_mode  : str | None – "ref", "cmp", or None
    ref_sol    : dict | None – reference solution for "cmp" comparison
    """
    mode = sol.get("eq_group_mode", "off")
    group_wl = sol.get("group_workload", {})
    has_groups = bool(groups) and bool(group_wl)

    if not has_groups:
        return [
            _section_title("WORKLOAD · Flat view", subtitle=None)
        ] + _render_flat(sol, people, diff_mode, ref_sol)

    if mode == "shares":
        return [
            _section_title(
                "WORKLOAD EQUITY · Shares-weighted",
                subtitle="Solver optimised per-group ratios "
                         "against your user-defined shares.")
        ] + _render_shares(sol, people, groups, diff_mode, ref_sol)

    # "off"
    return [
        _section_title(
            "WORKLOAD · Grouped view",
            subtitle="Solver optimises ONE global pool by individual "
                     "capacity. Per-group figures are reference only.")
    ] + _render_off(sol, people, groups, diff_mode, ref_sol)


# ══════════════════════════════════════════════════════════════════════
# SHARED ATOMS
# ══════════════════════════════════════════════════════════════════════

def _section_title(title, subtitle):
    parts = [ft.Text(title, weight=ft.FontWeight.BOLD, size=_s_stats(15))]
    if subtitle:
        parts.append(ft.Text(subtitle, size=_s_stats(12),
                             italic=True, color=ft.Colors.GREY_700))
    return ft.Container(
        content=ft.Column(parts, spacing=_s_stats(2), tight=True),
        padding=ft.padding.only(top=_s_stats(4), bottom=_s_stats(2)))


def _group_header(gname, bg, total_h, share_pct, *, prominent):
    """Coloured header row for a group block.

    prominent=True  → full colour (used in "shares")
    prominent=False → muted grey header, BUT with a coloured dot next
                      to the name so groups stay visually identifiable
                      (used in "off" — see _render_off).
    """
    header_bg = bg if prominent else "#455A64"
    fg_title  = ft.Colors.WHITE
    fg_total  = ft.Colors.WHITE if prominent else "#CFD8DC"

    children = []

    # In muted mode, prefix with a coloured dot to give the group a
    # visual identity even though the header background is neutral.
    # In prominent mode the whole header is the group colour, so the
    # dot would be redundant — the "▶" glyph stays instead.
    if prominent:
        children.append(ft.Text(
            f"▶  {gname}", color=fg_title,
            weight=ft.FontWeight.BOLD, size=_s_stats(14), expand=True))
    else:
        children.append(ft.Container(
            width=_s_stats(12), height=_s_stats(12),
            bgcolor=bg, border_radius=6))
        children.append(ft.Text(
            gname, color=fg_title,
            weight=ft.FontWeight.BOLD, size=_s_stats(14), expand=True))

    if share_pct is not None:
        # Fixed dark text on white — contrast doesn't depend on the
        # group's header colour (GROUP_HEADER_COLORS could include
        # pastel tones that'd make a coloured-text badge unreadable).
        children.append(ft.Container(
            content=ft.Text(f"{share_pct:.1f}%", size=_s_stats(12),
                            color="#1A237E", weight=ft.FontWeight.BOLD),
            bgcolor=ft.Colors.WHITE,
            padding=ft.padding.symmetric(horizontal=_s_stats(8), vertical=_s_stats(2)),
            border_radius=10))

    children.append(ft.Text(f"{total_h:.0f}h total",
                            color=fg_total, size=_s_stats(13),
                            weight=ft.FontWeight.BOLD))

    return ft.Container(
        content=ft.Row(children, spacing=_s_stats(8),
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=header_bg,
        padding=ft.padding.symmetric(horizontal=_s_stats(10), vertical=_s_stats(6)),
        border_radius=ft.border_radius.only(top_left=6, top_right=6),
        margin=ft.margin.only(top=_s_stats(6)))


def _stats_strip(items, accent_color, *, bg="#ECEFF1"):
    """Horizontal strip of pipe-separated stats.

    `items` is a list of (label_or_None, value_text, color, bold) tuples.
    """
    row_controls = []
    for i, (label, value, color, bold) in enumerate(items):
        if i > 0:
            row_controls.append(ft.Container(width=1, height=_s_stats(14),
                                             bgcolor=ft.Colors.GREY_400))
        text = f"{label}: {value}" if label else value
        row_controls.append(ft.Text(
            text,
            size=_s_stats(12), color=color or ft.Colors.GREY_800,
            weight=ft.FontWeight.BOLD if bold else None))
    return ft.Container(
        content=ft.Row(row_controls, spacing=_s_stats(10), wrap=True),
        bgcolor=bg,
        padding=ft.padding.symmetric(horizontal=_s_stats(10), vertical=_s_stats(5)),
        border=ft.border.only(
            left=ft.border.BorderSide(3, accent_color),
            right=ft.border.BorderSide(1, "#CFD8DC")))


def _member_bar_row(p, ch, bar_max, accent_color, diff_mode, ph):
    """One person's row: name · hours · ASCII bar · optional diff suffix."""
    if bar_max > 0:
        bar_fill = min(round((ch / bar_max) * 20), 20)
    else:
        bar_fill = 0
    bar_str = "█" * bar_fill + "░" * (20 - bar_fill)

    if diff_mode == "cmp" and ph is not None:
        changed = ch != ph
        suffix  = f"  ← was {ph:.0f}h" if changed else ""
        clr     = ft.Colors.ORANGE_700 if changed else ft.Colors.BLACK
        wgt     = ft.FontWeight.BOLD if changed else None
    else:
        suffix, clr, wgt = "", ft.Colors.BLACK, None

    return ft.Container(
        content=ft.Row([
            ft.Text(p, size=_s_stats(13), width=_s_stats(120), color=clr, weight=wgt,
                    no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(f"{ch:.0f}h", size=_s_stats(13), width=_s_stats(36),
                    color=clr, weight=wgt,
                    text_align=ft.TextAlign.RIGHT),
            ft.Text(bar_str, size=_s_stats(10), color=accent_color,
                    font_family="monospace"),
            ft.Text(suffix, size=_s_stats(12), color=ft.Colors.ORANGE_700,
                    italic=True),
        ], spacing=_s_stats(6)),
        padding=ft.padding.symmetric(horizontal=_s_stats(14), vertical=_s_stats(2)),
        bgcolor=ft.Colors.WHITE,
        border=ft.border.only(
            left=ft.border.BorderSide(3, accent_color),
            bottom=ft.border.BorderSide(1, "#ECEFF1")))


def _dev_color(dev: float) -> str:
    return ("#C62828" if dev < -0.5 else
            "#2E7D32" if dev > 0.5  else "#546E7A")


def _collect_intra(members, workload):
    vals = [workload.get(p, 0) for p in members if p in workload]
    if not vals:
        return 0, 0, 0
    mx, mn = max(vals), min(vals)
    return mx, mn, mx - mn


# ══════════════════════════════════════════════════════════════════════
# MODE: "shares"
# ══════════════════════════════════════════════════════════════════════

def _render_shares(sol, people, groups, diff_mode, ref_sol):
    buf = []
    workload     = sol.get("workload", {})
    ref_workload = ref_sol.get("workload", {}) if ref_sol else {}
    group_wl     = sol.get("group_workload", {})

    group_totals = []

    for g_idx, (gname, members) in enumerate(groups.items()):
        bg    = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
        ginfo = group_wl.get(gname, {})

        total_h   = ginfo.get("total_hours",
                              sum(workload.get(p, 0) for p in members))
        target_h  = ginfo.get("target_hours",      0.0)
        dev       = ginfo.get("deviation",         total_h - target_h)
        pp_avg    = ginfo.get("per_person_avg",
                              total_h / max(len(members), 1))
        pp_tgt    = ginfo.get("per_person_target",
                              target_h / max(len(members), 1))
        share_pct = ginfo.get("share_pct")

        group_totals.append((gname, total_h))

        intra_max, intra_min, intra_spread = _collect_intra(members, workload)
        dev_clr = _dev_color(dev)

        buf.append(_group_header(gname, bg, total_h, share_pct,
                                 prominent=True))
        buf.append(_stats_strip([
            ("Target", f"{target_h:.1f}h",       ft.Colors.GREY_800, False),
            ("Dev",    f"{dev:+.1f}h",           dev_clr,            True),
            (None,     f"Avg/person {pp_avg:.1f}h (tgt {pp_tgt:.1f}h)",
                                                 ft.Colors.GREY_700, False),
            ("Intra-spread",
                       f"{intra_spread:.0f}h (↑{intra_max:.0f} ↓{intra_min:.0f})",
                                                 ft.Colors.GREY_700, False),
        ], accent_color=bg))

        bar_max = max(intra_max, 1)
        for p in members:
            ch = workload.get(p, 0)
            ph = (ref_workload.get(p, 0)
                  if diff_mode == "cmp" and ref_sol else None)
            buf.append(_member_bar_row(p, ch, bar_max, bg, diff_mode, ph))

    # Headline KPI for "shares": inter-group spread.
    if group_totals:
        buf.append(_inter_group_summary(group_totals, workload, people))

    return buf


# ══════════════════════════════════════════════════════════════════════
# MODE: "off"
# ══════════════════════════════════════════════════════════════════════

def _render_off(sol, people, groups, diff_mode, ref_sol):
    buf = []
    workload     = sol.get("workload", {})
    ref_workload = ref_sol.get("workload", {}) if ref_sol else {}
    group_wl     = sol.get("group_workload", {})

    # Headline KPI for "off": global individual spread (the solver
    # optimises one pool, so this is what actually tells you if
    # equity worked).
    all_wl = [workload.get(p, 0) for p in people]
    if all_wl:
        buf.append(_global_spread_strip(all_wl))

    # Per-group sections: each group gets its GROUP_HEADER_COLORS
    # tint (same cycle as "shares") for visual identity, but the
    # overall styling stays muted via _group_header(prominent=False)
    # — header bg is neutral grey, figures marked as reference.
    for g_idx, (gname, members) in enumerate(groups.items()):
        group_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
        ginfo = group_wl.get(gname, {})
        total_h   = ginfo.get("total_hours",
                              sum(workload.get(p, 0) for p in members))
        target_h  = ginfo.get("target_hours", 0.0)
        dev       = ginfo.get("deviation",    total_h - target_h)
        share_pct = ginfo.get("share_pct")  # capacity-derived in "off"

        intra_max, intra_min, intra_spread = _collect_intra(members, workload)
        dev_clr = _dev_color(dev)

        buf.append(_group_header(gname, group_color, total_h, share_pct,
                                 prominent=False))
        buf.append(_stats_strip([
            ("Ref. target (by capacity)", f"{target_h:.1f}h",
                                          ft.Colors.GREY_700, False),
            ("Dev",                       f"{dev:+.1f}h", dev_clr, True),
            ("Intra-spread",
                f"{intra_spread:.0f}h (↑{intra_max:.0f} ↓{intra_min:.0f})",
                                          ft.Colors.GREY_700, False),
        ], accent_color=group_color, bg="#F5F5F5"))

        bar_max = max(intra_max, 1)
        for p in members:
            ch = workload.get(p, 0)
            ph = (ref_workload.get(p, 0)
                  if diff_mode == "cmp" and ref_sol else None)
            buf.append(_member_bar_row(p, ch, bar_max, group_color,
                                       diff_mode, ph))

    return buf


# ══════════════════════════════════════════════════════════════════════
# MODE: flat (no groups)
# ══════════════════════════════════════════════════════════════════════

def _render_flat(sol, people, diff_mode, ref_sol):
    buf = []
    workload     = sol.get("workload", {})
    ref_workload = ref_sol.get("workload", {}) if ref_sol else {}

    for p in people:
        ch = workload.get(p, 0)
        if diff_mode == "cmp" and ref_sol is not None:
            ph      = ref_workload.get(p, 0)
            changed = ch != ph
            buf.append(ft.Text(
                f"  {p}: {ch:.0f}h"
                + (f"  (← {ph:.0f}h)" if changed else ""),
                size=_s_stats(13),
                color=ft.Colors.ORANGE_700 if changed else None,
                weight=ft.FontWeight.BOLD if changed else None))
        else:
            buf.append(ft.Text(f"  {p}: {ch:.0f}h", size=_s_stats(13)))

    all_wl = [workload.get(p, 0) for p in people]
    if all_wl:
        buf.append(ft.Text(
            f"  Global: max={max(all_wl):.0f}h  min={min(all_wl):.0f}h  "
            f"spread={max(all_wl)-min(all_wl):.0f}h",
            size=_s_stats(13), italic=True, color=ft.Colors.GREY_700))
    return buf


# ══════════════════════════════════════════════════════════════════════
# HEADLINE STRIPS
# ══════════════════════════════════════════════════════════════════════

def _global_spread_strip(all_wl):
    """Headline KPI for 'off' mode, placed at the TOP of the section."""
    mx, mn = max(all_wl), min(all_wl)
    return ft.Container(
        content=ft.Row([
            ft.Text("◎  Global individual spread",
                    weight=ft.FontWeight.BOLD, size=_s_stats(13),
                    color=ft.Colors.WHITE),
            ft.Container(width=1, height=_s_stats(14), bgcolor="#78909C"),
            ft.Text(f"{mx - mn:.0f}h",
                    size=_s_stats(16), weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE),
            ft.Text(f"(max {mx:.0f}h · min {mn:.0f}h)",
                    size=_s_stats(12), color="#B0BEC5"),
        ], spacing=_s_stats(10),
           vertical_alignment=ft.CrossAxisAlignment.CENTER, wrap=True),
        bgcolor="#37474F",
        padding=ft.padding.symmetric(horizontal=_s_stats(12), vertical=_s_stats(8)),
        border_radius=6, margin=ft.margin.only(top=_s_stats(4), bottom=_s_stats(6)))


def _inter_group_summary(group_totals, workload, people):
    """Headline KPI for 'shares' mode, placed at the BOTTOM."""
    g_max_name, g_max_val = max(group_totals, key=lambda x: x[1])
    g_min_name, g_min_val = min(group_totals, key=lambda x: x[1])
    inter_spread = g_max_val - g_min_val
    all_wl = [workload.get(p, 0) for p in people]

    children = [
        ft.Text("◎  Inter-group summary",
                weight=ft.FontWeight.BOLD, size=_s_stats(13),
                color=ft.Colors.WHITE),
        ft.Row([
            ft.Text(f"Inter-group spread: {inter_spread:.0f}h",
                    size=_s_stats(13), color=ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD),
            ft.Text(f"(max: {g_max_name} {g_max_val:.0f}h  ·  "
                    f"min: {g_min_name} {g_min_val:.0f}h)",
                    size=_s_stats(12), color="#B0BEC5"),
        ], spacing=_s_stats(10), wrap=True),
    ]
    if all_wl:
        children.append(ft.Text(
            f"Global individual: max={max(all_wl):.0f}h  "
            f"min={min(all_wl):.0f}h  "
            f"spread={max(all_wl)-min(all_wl):.0f}h",
            size=_s_stats(12), color="#B0BEC5"))

    return ft.Container(
        content=ft.Column(children, spacing=_s_stats(4)),
        bgcolor="#37474F",
        padding=ft.padding.symmetric(horizontal=_s_stats(12), vertical=_s_stats(8)),
        border_radius=6, margin=ft.margin.only(top=_s_stats(8)))