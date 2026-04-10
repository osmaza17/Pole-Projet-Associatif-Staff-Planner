"""
Rules tab — three horizontal rows, one per rule type.

Each row contains a single ft.Row(scroll=AUTO) populated by the
corresponding manager. Managers (CaptainRulesManager, MandatoryRulesManager,
QuotaRulesManager) all expose the same contract:
    - build_summary()        → fills self._col.controls with [add_btn, *cards]
    - validate_and_refresh() → re-runs validation and rebuilds the summary

CaptainRulesManager and MandatoryRulesManager are reused unmodified from
the dimensions package — they only need a container with a `controls`
attribute, which an ft.Row provides.
"""

import flet as ft
from constants import _s
from tabs.dimensions.captain_rules   import CaptainRulesManager
from tabs.dimensions.mandatory_rules import MandatoryRulesManager
from .quota_rules                    import QuotaRulesManager


class RulesTab:
    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self.state = state
        self.page  = page
        self._on_solve_blocked_update = on_solve_blocked_update

        self._build_ui()

        # Each manager writes its [add_btn, *cards] into the row it owns.
        self._captain_mgr = CaptainRulesManager(
            state, page, self._captain_row)
        self._mandatory_mgr = MandatoryRulesManager(
            state, page, self._mandatory_row)
        self._quota_mgr = QuotaRulesManager(
            state, page, self._quota_row, on_solve_blocked_update)

    # ── UI scaffolding ────────────────────────────────────────────────

    def _build_ui(self):
        self._captain_row = ft.Row(
            spacing=_s(8),
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.START,
            wrap=False,
            expand=True)
        self._mandatory_row = ft.Row(
            spacing=_s(8),
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.START,
            wrap=False,
            expand=True)
        self._quota_row = ft.Row(
            spacing=_s(8),
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.START,
            wrap=False,
            expand=True)

    @staticmethod
    def _section(title: str, icon: str, color: str,
                 row: ft.Row) -> ft.Container:
        header = ft.Container(
            content=ft.Row([
                ft.Icon(icon, size=_s(16), color=ft.Colors.WHITE),
                ft.Text(title, size=_s(13),
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE),
            ], spacing=_s(6)),
            bgcolor=color,
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(6)),
            border_radius=ft.border_radius.only(top_left=8, top_right=8))

        body = ft.Container(
            content=row,
            padding=ft.padding.all(_s(10)),
            bgcolor="#FAFAFA",
            border=ft.border.all(1, "#CFD8DC"),
            border_radius=ft.border_radius.only(
                bottom_left=8, bottom_right=8),
            expand=True)

        return ft.Container(
            content=ft.Column([header, body],
                              spacing=0, tight=True, expand=True),
            expand=True,
            padding=ft.padding.symmetric(vertical=_s(2)))

    def get_container(self) -> ft.Container:
        captain_section = self._section(
            "🎖  Captain Rules",
            ft.Icons.MILITARY_TECH, "#E65100",
            self._captain_row)
        mandatory_section = self._section(
            "⚠  Mandatory Rules",
            ft.Icons.ASSIGNMENT_IND, "#1565C0",
            self._mandatory_row)
        quota_section = self._section(
            "📊  Quota Rules",
            ft.Icons.FORMAT_LIST_NUMBERED, "#6A1B9A",
            self._quota_row)

        return ft.Container(
            content=ft.Column(
                controls=[
                    captain_section,
                    mandatory_section,
                    quota_section,
                ],
                spacing=_s(8),
                expand=True),
            padding=_s(16),
            expand=True)

    # ── Public API ────────────────────────────────────────────────────

    def initial_build(self):
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._quota_mgr.validate_and_refresh()

    def build(self):
        """Called when the tab becomes visible — re-validates everything."""
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._quota_mgr.validate_and_refresh()

    def validate_rules(self):
        """Called from app._update_solve_blocked() to re-check all rules."""
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._quota_mgr.validate_and_refresh()

    def rebuild_all(self):
        """Called by app._apply_loaded_profile() after loading a profile."""
        self._captain_mgr.validate_and_refresh()
        self._mandatory_mgr.validate_and_refresh()
        self._quota_mgr.validate_and_refresh()