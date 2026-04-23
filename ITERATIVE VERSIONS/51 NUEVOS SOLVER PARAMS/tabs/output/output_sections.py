"""
Section helpers for the Output tab.

Pure builders for the lower sections of a run column:
  - section headers (ok / warning / info / neutral)
  - violation, info & note chips
  - violations sections
  - rest-enforcement section
  - variety report section
  - "task hours by person" section

All functions are pure: they take data and return Flet controls.
"""

import flet as ft
from constants import _s_stats


# ══════════════════════════════════════════════════════════════════════
# ATOMIC UI ELEMENTS
# ══════════════════════════════════════════════════════════════════════

def section_header(title: str, count: int = 0,
                   is_ok: bool = False, is_info: bool = False) -> ft.Container:
    """
    Section header with semantic styling.

    Priority: is_info (with count>0) > is_ok > count>0 (warn) > neutral.
    The is_info variant is for sections that show *informational*
    items rather than violations (e.g. rest relaxations, which are
    expected side-effects of stronger rules, not errors).
    """
    if is_info and count > 0:
        icon, bg, fg = f"ℹ {count}", "#E3F2FD", "#1565C0"
    elif is_ok:
        icon, bg, fg = "✓", "#E8F5E9", "#2E7D32"
    elif count > 0:
        icon, bg, fg = f"⚠ {count}", "#FFF3E0", "#E65100"
    else:
        icon, bg, fg = "—", "#F5F5F5", "#757575"
    return ft.Container(
        content=ft.Row([
            ft.Text(icon, size=_s_stats(12), weight=ft.FontWeight.BOLD, color=fg),
            ft.Text(title, size=_s_stats(14), weight=ft.FontWeight.BOLD, color=fg),
        ], spacing=_s_stats(6), vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=bg, padding=ft.padding.symmetric(_s_stats(6), _s_stats(10)),
        border_radius=4, margin=ft.margin.only(top=_s_stats(8)))


def violation_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s_stats(12), color="#BF360C"),
        bgcolor="#FFCCBC", padding=ft.padding.symmetric(_s_stats(3), _s_stats(8)),
        border_radius=4, margin=ft.margin.only(left=_s_stats(12), top=_s_stats(2)))


def info_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s_stats(12), color="#1565C0"),
        bgcolor="#BBDEFB", padding=ft.padding.symmetric(_s_stats(3), _s_stats(8)),
        border_radius=4, margin=ft.margin.only(left=_s_stats(12), top=_s_stats(2)))


def note_chip(text: str) -> ft.Container:
    """Neutral/informational chip — softer than info_chip."""
    return ft.Container(
        content=ft.Text(text, size=_s_stats(12), color="#37474F"),
        bgcolor="#ECEFF1", padding=ft.padding.symmetric(_s_stats(3), _s_stats(8)),
        border_radius=4, margin=ft.margin.only(left=_s_stats(12), top=_s_stats(2)))


# ══════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_violations_section(title: str, violations: list) -> list:
    if not violations:
        return [section_header(title, is_ok=True)]
    controls = [section_header(title, count=len(violations))]
    controls.extend(violation_chip(v) for v in violations)
    return controls


def build_rest_enforcement_section(enforced_rest: bool,
                                   consec_relaxations: dict,
                                   days: list | None = None) -> list:
    """
    Surface rest enforcement info.

    - enforced_rest is False                    → section is hidden.
    - enforced_rest, no relaxations             → "✓ OK" header.
    - enforced_rest + relaxations               → info header + one
      compact line per person summarising every relaxed day/hour.

    Relaxations are *expected*: the solver drops the max-consecutive
    constraint on slots where a single captain is the only eligible
    worker. This is informational, not a violation.

    `days` (optional): the solver's horizon ordering. When provided,
    days within each person's line are listed in that order — not
    alphabetically (which would read "Fri, Mon, Thu, Tue, Wed").
    """
    if not enforced_rest:
        return []

    if not consec_relaxations:
        return [section_header("REST ENFORCEMENT", is_ok=True)]

    # Count total relaxed hours across everyone.
    total_relaxed = sum(
        len(hrs)
        for by_day in consec_relaxations.values()
        for hrs in by_day.values()
        if hrs)
    if total_relaxed == 0:
        return [section_header("REST ENFORCEMENT", is_ok=True)]

    controls = [section_header("REST ENFORCEMENT",
                               count=total_relaxed, is_info=True)]

    # One line per person — D2 requested compact format:
    # "Ana — Mon (14h, 15h), Tue (10h): 3 hours relaxed (sole captain)"
    for person in sorted(consec_relaxations.keys()):
        by_day = consec_relaxations[person]
        # Iterate in the caller's day order (solver horizon), falling
        # back to sorted keys only when no order is provided.
        day_iter = days if days is not None else sorted(by_day.keys())
        parts = []
        person_total = 0
        for day in day_iter:
            hrs = by_day.get(day)
            if not hrs:
                continue
            parts.append(f"{day} ({', '.join(f'{h}h' for h in hrs)})")
            person_total += len(hrs)
        if not parts:
            continue
        plural = "s" if person_total != 1 else ""
        line = (f"{person} — {', '.join(parts)}: "
                f"{person_total} hour{plural} relaxed (sole eligible captain)")
        controls.append(note_chip(line))

    return controls


def build_variety_section(variety_report: list, tasks: list) -> list:
    if not variety_report:
        return []
    controls = [section_header("TASK VARIETY")]
    for entry in variety_report:
        t = entry["task"]
        line = (f"'{t}': {entry['touched']}/{entry['qualified']} qualified worked it, "
                f"{entry['repeated']} repeated, {entry['total_hours']}h total")
        controls.append(info_chip(line))
    return controls


def build_emergency_section(emerg: list) -> list:
    if not emerg:
        return []
    controls = [section_header("EMERGENCY CALL-INS", count=len(emerg))]
    controls.extend(info_chip(line) for line in emerg)
    return controls


def build_task_hours_section(task_workload: dict, people: list, tasks: list,
                             person_group_color: dict | None = None) -> list:
    """
    Task-by-task breakdown of who worked how many hours.

    person_group_color : dict | None
        Mapping person → group colour hex. When provided, each chip
        renders the person's name in their group's colour (bold, on a
        neutral-grey background). Lets the user cross-reference with
        the WORKLOAD section at a glance. Persons without a mapped
        group fall back to neutral grey text.
    """
    if not task_workload:
        return []

    person_group_color = person_group_color or {}

    controls = [section_header("TASK HOURS BY PERSON")]
    for t in tasks:
        persons_on = [(p, task_workload.get(p, {}).get(t, 0))
                      for p in people
                      if task_workload.get(p, {}).get(t, 0) > 0]
        if not persons_on:
            continue
        max_h = max(h for _, h in persons_on)
        min_h = min(h for _, h in persons_on)
        total = sum(h for _, h in persons_on)
        controls.append(ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"■ {t}", size=_s_stats(13),
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.BLUE_GREY_800),
                    ft.Text(f"{total}h total  ·  range {min_h:.0f}–{max_h:.0f}h",
                            size=_s_stats(12), italic=True,
                            color=ft.Colors.GREY_600),
                ], spacing=_s_stats(8),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([
                    _person_task_chip(p, h, person_group_color.get(p))
                    for p, h in sorted(persons_on, key=lambda x: -x[1])
                ], spacing=_s_stats(4), wrap=True),
            ], spacing=_s_stats(3), tight=True),
            padding=ft.padding.only(left=_s_stats(12), top=_s_stats(4), bottom=_s_stats(2))))
    return controls


def _person_task_chip(p: str, h: float, group_color: str | None) -> ft.Container:
    """Per-person chip with optional group-colour tinting.

    When group_color is given, the person's name renders in that
    colour with bold weight, over a neutral-grey background. The
    background stays neutral (not the group colour) so legibility
    is preserved regardless of how saturated / pastel the group
    colour is.
    """
    if group_color:
        text_color = group_color
        weight = ft.FontWeight.BOLD
    else:
        text_color = ft.Colors.BLUE_GREY_700
        weight = None
    return ft.Container(
        content=ft.Text(f"{p} {h:.0f}h", size=_s_stats(11),
                        color=text_color, weight=weight),
        bgcolor="#ECEFF1",
        padding=ft.padding.symmetric(_s_stats(2), _s_stats(6)),
        border_radius=3)