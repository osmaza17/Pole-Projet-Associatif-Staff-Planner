"""
Section helpers for the Output tab.

Pure builders for the lower sections of a run column:
  - section headers (ok / warning / neutral)
  - violation & info chips
  - violations sections
  - variety report section
  - "task hours by person" section

All functions are pure: they take data and return Flet controls.
"""

import flet as ft
from constants import _s


# ── Atomic UI elements ────────────────────────────────────────────────

def section_header(title: str, count: int = 0, is_ok: bool = False) -> ft.Container:
    if is_ok:
        icon, bg, fg = "✓", "#E8F5E9", "#2E7D32"
    elif count > 0:
        icon, bg, fg = f"⚠ {count}", "#FFF3E0", "#E65100"
    else:
        icon, bg, fg = "—", "#F5F5F5", "#757575"
    return ft.Container(
        content=ft.Row([
            ft.Text(icon, size=_s(11), weight=ft.FontWeight.BOLD, color=fg),
            ft.Text(title, size=_s(13), weight=ft.FontWeight.BOLD, color=fg),
        ], spacing=_s(6), vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=bg, padding=ft.padding.symmetric(_s(6), _s(10)),
        border_radius=4, margin=ft.margin.only(top=_s(8)))


def violation_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s(11), color="#BF360C"),
        bgcolor="#FFCCBC", padding=ft.padding.symmetric(_s(3), _s(8)),
        border_radius=4, margin=ft.margin.only(left=_s(12), top=_s(2)))


def info_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s(11), color="#1565C0"),
        bgcolor="#BBDEFB", padding=ft.padding.symmetric(_s(3), _s(8)),
        border_radius=4, margin=ft.margin.only(left=_s(12), top=_s(2)))


# ── Section builders ──────────────────────────────────────────────────

def build_violations_section(title: str, violations: list) -> list:
    if not violations:
        return [section_header(title, is_ok=True)]
    controls = [section_header(title, count=len(violations))]
    controls.extend(violation_chip(v) for v in violations)
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


def build_task_hours_section(task_workload: dict, people: list, tasks: list) -> list:
    if not task_workload:
        return []
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
                    ft.Text(f"■ {t}", size=_s(12),
                            weight=ft.FontWeight.W_600,
                            color=ft.Colors.BLUE_GREY_800),
                    ft.Text(f"{total}h total  ·  range {min_h:.0f}–{max_h:.0f}h",
                            size=_s(11), italic=True,
                            color=ft.Colors.GREY_600),
                ], spacing=_s(8),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row([
                    ft.Container(
                        ft.Text(f"{p} {h:.0f}h", size=_s(10),
                                color=ft.Colors.BLUE_GREY_700),
                        bgcolor="#ECEFF1",
                        padding=ft.padding.symmetric(_s(2), _s(6)),
                        border_radius=3)
                    for p, h in sorted(persons_on, key=lambda x: -x[1])
                ], spacing=_s(4), wrap=True),
            ], spacing=_s(3), tight=True),
            padding=ft.padding.only(left=_s(12), top=_s(4), bottom=_s(2))))
    return controls