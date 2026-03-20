"""
Dimensions tab — defines people, groups, tasks, days, hours, consecutive
limits, rotation settings, captain rules, and mandatory rules.

Sub-components are delegated to sibling modules:
  • captain_rules.py     → CaptainRulesManager
  • mandatory_rules.py   → MandatoryRulesManager
  • groups.py            → GroupsManager
  • max_consec_hours.py  → MaxConsecHoursManager
"""

import threading
import flet as ft
from constants import _s, DEFAULT_HOURS_TEXT
from ui_helpers import UIHelpers
from .captain_rules    import CaptainRulesManager
from .mandatory_rules  import MandatoryRulesManager
from .groups           import GroupsManager
from .max_consec_hours import MaxConsecHoursManager


class DimensionsTab:
    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self.state = state
        self.page  = page
        self._on_solve_blocked_update = on_solve_blocked_update

        self._debounce_people: threading.Timer | None = None
        self._debounce_tasks : threading.Timer | None = None
        self._debounce_days  : threading.Timer | None = None

        self._build_ui()

        # Sub-managers (created after widgets exist)
        self._captain_mgr    = CaptainRulesManager(
            state, page, self._captain_rules_col)
        self._mandatory_mgr  = MandatoryRulesManager(
            state, page, self._mandatory_rules_col)
        self._max_consec_mgr = MaxConsecHoursManager(
            state, page, on_solve_blocked_update)
        self._groups_mgr     = GroupsManager(
            state, page, self._people_groups_row,
            self._debounce,
            on_text_change_cb=self._on_groups_text_change,
            on_structure_change_cb=self._on_groups_structure_change,
            rest_btn_callback=self._max_consec_mgr.open_dialog)

    # ── Debounce helper ────────────────────────────────────────────────

    def _debounce(self, attr: str, fn, delay: float = 0.3):
        old = getattr(self, attr, None)
        if old is not None:
            old.cancel()
        t = threading.Timer(delay, fn)
        t.daemon = True
        t.start()
        setattr(self, attr, t)

    # ── Initial widget construction ────────────────────────────────────

    def _build_ui(self):
        s = self.state

        self.tf_tasks = ft.TextField(
            value=s.tasks_text, multiline=True, min_lines=8, max_lines=200,
            label="Tasks (one per line)", width=_s(280), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_tasks_change)

        self.tf_days = ft.TextField(
            value=s.days_text, multiline=True, min_lines=1, max_lines=200,
            label="Days (one per line)", width=_s(120), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_days_change)

        self._rotation_col = ft.ListView(
            width=_s(280), spacing=_s(4), expand=True)
        self._hours_col = ft.ListView(
            width=_s(200), spacing=_s(4), expand=True)

        self._people_groups_row = ft.Row(
            spacing=_s(10),
            vertical_alignment=ft.CrossAxisAlignment.START,
            scroll=ft.ScrollMode.AUTO)

        self._captain_rules_col   = ft.ListView(
            width=_s(320), spacing=_s(4), expand=True)
        self._mandatory_rules_col = ft.ListView(
            width=_s(320), spacing=_s(4), expand=True)

        # ── Redundant-force-mandate warning banner ─────────────────────
        self._mandatory_conflict_text = ft.Text(
            "",
            size=_s(11),
            color=ft.Colors.RED_700,
            weight=ft.FontWeight.BOLD,
            visible=False)

    # ── onChange handlers ──────────────────────────────────────────────

    def _on_tasks_change(self, e):
        self.state.tasks_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_tasks', self._build_rotation_list)
        self._on_solve_blocked_update()

    def _on_days_change(self, e):
        self.state.days_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_days', self._build_hours_per_day)
        self._on_solve_blocked_update()

    # ── Conflict banner ────────────────────────────────────────────────

    def _refresh_mandatory_conflict_banner(self):
        """
        Scan saved mandatory_rules for pairs that are inherently
        contradictory: same person, same (hour, day), both specific
        (just_do_anything=False), but different tasks.
        """
        s      = self.state
        rules  = s.mandatory_rules
        n      = len(rules)
        issues = []

        for i in range(n):
            ri = rules[i]
            if ri.get("just_do_anything", False):
                continue
            pi = ri["person"]
            ti = ri.get("task")
            if not ti:
                continue
            slots_i = {
                (h, d)
                for d, hs in ri.get("hours", {}).items()
                for h in hs
            }

            for j in range(i + 1, n):
                rj = rules[j]
                if rj.get("just_do_anything", False):
                    continue
                if rj["person"] != pi:
                    continue
                tj = rj.get("task")
                if not tj:
                    continue
                slots_j = {
                    (h, d)
                    for d, hs in rj.get("hours", {}).items()
                    for h in hs
                }
                overlap = slots_i & slots_j
                if overlap:
                    ex_h, ex_d = next(iter(overlap))
                    issues.append(
                        f"#{i+1} ({pi} → '{ti}') ↔ "
                        f"#{j+1} ({pi} → '{tj}') @ {ex_h} {ex_d}")

        if issues:
            shown = issues[:3]
            tail  = f" … (+{len(issues)-3} more)" if len(issues) > 3 else ""
            self._mandatory_conflict_text.value = (
                "⚠ Redundant force mandates detected: "
                + " | ".join(shown) + tail)
            self._mandatory_conflict_text.visible = True
        else:
            self._mandatory_conflict_text.value   = ""
            self._mandatory_conflict_text.visible = False

        try:
            self._mandatory_conflict_text.update()
        except Exception:
            pass

    # ── Sub-list builders ──────────────────────────────────────────────

    def _build_rest_list(self):
        """
        No-op — kept for backward compatibility with app._apply_loaded_profile.
        Rest conditions are edited exclusively through MaxConsecHoursManager.
        """
        pass

    def _build_rotation_list(self):
        s = self.state
        _, tasks, _, _ = s.dims()
        buf = []
        for t in tasks:
            val = s.rotation_st.get(t, 1)
            btn = ft.ElevatedButton(
                text="ON" if val else "OFF",
                bgcolor=ft.Colors.GREEN_200 if val else ft.Colors.RED_200,
                color=ft.Colors.BLACK87,
                style=ft.ButtonStyle(
                    padding=ft.padding.symmetric(
                        horizontal=_s(8), vertical=_s(4))),
                height=_s(30), width=_s(60), data=t)

            def _click(e):
                curr = s.rotation_st.get(e.control.data, 1)
                nxt  = 0 if curr else 1
                s.rotation_st[e.control.data] = nxt
                e.control.text    = "ON" if nxt else "OFF"
                e.control.bgcolor = (ft.Colors.GREEN_200 if nxt
                                     else ft.Colors.RED_200)
                e.control.update()

            btn.on_click = _click
            buf.append(ft.Row([UIHelpers.lbl(t, _s(150)), btn],
                               spacing=_s(4)))
        self._rotation_col.controls = buf
        self.page.update()

    def _build_hours_per_day(self):
        s = self.state
        _, _, _, days = s.dims()
        buf = [ft.Text("Hours per Day", weight=ft.FontWeight.BOLD,
                       size=_s(12))]
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
                self._on_solve_blocked_update()

            tf.on_change = _ch
            buf.append(tf)

        self._hours_col.controls = buf
        self.page.update()

    # ── Public API ─────────────────────────────────────────────────────

    def validate_rules(self):
        """
        Re-validate all captain and mandatory rules against current state.
        Updates s.validation_errors["rules"] and refreshes card visuals.
        """
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._refresh_mandatory_conflict_banner()

    def initial_build(self):
        self._build_rotation_list()
        self._build_hours_per_day()
        self._groups_mgr.build()
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._refresh_mandatory_conflict_banner()

    def _on_groups_text_change(self):
        self._on_solve_blocked_update()

    def _on_groups_structure_change(self):
        self._groups_mgr.build()
        self._on_solve_blocked_update()

    # kept for backward compat
    def _on_groups_people_change(self):
        self._on_groups_structure_change()

    # ── Forwarding methods (called by app._apply_loaded_profile) ───────

    def _build_groups_list(self):
        self._groups_mgr.build()

    def _build_captain_rules_summary(self):
        self._captain_mgr.validate_and_refresh()

    def _build_mandatory_rules_summary(self):
        self._mandatory_mgr.validate_and_refresh()
        self._refresh_mandatory_conflict_banner()

    # ── Layout ─────────────────────────────────────────────────────────

    @staticmethod
    def _section(title: str, *controls) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            title, size=_s(12), weight=ft.FontWeight.BOLD,
                            color=ft.Colors.WHITE),
                        bgcolor=ft.Colors.BLUE_GREY_700,
                        padding=ft.padding.symmetric(
                            horizontal=_s(12), vertical=_s(6)),
                        border_radius=ft.border_radius.only(
                            top_left=6, top_right=6)),
                    ft.Container(
                        content=ft.Row(
                            list(controls), spacing=_s(12),
                            vertical_alignment=ft.CrossAxisAlignment.START,
                            wrap=False),
                        padding=ft.padding.symmetric(
                            horizontal=_s(12), vertical=_s(10))),
                ], spacing=0, tight=True),
            border=ft.border.all(1, ft.Colors.BLUE_GREY_200),
            border_radius=6)

    def get_container(self) -> ft.Container:
        def _header(title: str) -> ft.Text:
            return ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD)

        return ft.Container(
            content=ft.Row(
                controls=[
                    # ── Mandatory Rules ────────────────────────────────
                    ft.Column([
                        _header("⚠  Mandatory Rules"),
                        ft.Divider(height=_s(4)),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED,
                                        size=_s(14),
                                        color=ft.Colors.RED_700),
                                self._mandatory_conflict_text,
                            ], spacing=_s(6),
                               vertical_alignment=ft.CrossAxisAlignment.START,
                               wrap=True),
                            bgcolor="#FFEBEE",
                            border=ft.border.all(1, ft.Colors.RED_300),
                            border_radius=6,
                            padding=ft.padding.symmetric(
                                horizontal=_s(10), vertical=_s(6)),
                            visible=False,
                        ),
                        self._mandatory_conflict_text,
                        self._mandatory_rules_col,
                    ], spacing=_s(4), expand=True,
                       scroll=ft.ScrollMode.AUTO),

                    ft.VerticalDivider(width=_s(4)),

                    # ── Captain Rules ──────────────────────────────────
                    ft.Column([
                        _header("🎖  Captain Rules"),
                        ft.Divider(height=_s(4)),
                        self._captain_rules_col,
                    ], spacing=_s(4), expand=True,
                       scroll=ft.ScrollMode.AUTO),

                    ft.VerticalDivider(width=_s(4)),

                    # ── Days & Hours ───────────────────────────────────
                    ft.Column([
                        _header("📅  Days & Hours"),
                        ft.Divider(height=_s(4)),
                        self.tf_days,
                        self._hours_col,
                    ], spacing=_s(4), expand=True,
                       scroll=ft.ScrollMode.AUTO),

                    ft.VerticalDivider(width=_s(4)),

                    # ── Tasks & Rotation ───────────────────────────────
                    ft.Column([
                        _header("🗂  Tasks"),
                        ft.Divider(height=_s(4)),
                        self.tf_tasks,
                        ft.Text("Rotation", weight=ft.FontWeight.BOLD,
                                size=_s(11)),
                        self._rotation_col,
                    ], spacing=_s(4), expand=True,
                       scroll=ft.ScrollMode.AUTO),

                    ft.VerticalDivider(width=_s(4)),

                    # ── People & Groups ────────────────────────────────
                    ft.Column([
                        _header("👥  People & Groups"),
                        ft.Divider(height=_s(4)),
                        self._people_groups_row,
                    ], spacing=_s(4), expand=True,
                       scroll=ft.ScrollMode.AUTO),
                ],
                spacing=_s(16),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True,
                scroll=ft.ScrollMode.ALWAYS),
            padding=_s(20), expand=True)