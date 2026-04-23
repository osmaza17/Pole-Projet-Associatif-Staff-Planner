"""
Dimensions tab — orchestrator for people, groups, tasks, days, hours,
rotation settings, task durations, and task priority.

Delegates list-building to builders, groups, and max_consec_hours.
"""

import threading
import flet as ft
from constants import _s
from .groups import GroupsManager
from .max_consec_hours import MaxConsecHoursManager
from .builders import build_rotation_list, handle_reorder, build_hours_per_day


class DimensionsTab:
    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self.state = state
        self.page = page
        self._on_solve_blocked_update = on_solve_blocked_update

        self._debounce_people: threading.Timer | None = None
        self._debounce_tasks: threading.Timer | None = None
        self._debounce_days: threading.Timer | None = None

        self._build_ui()

        self._max_consec_mgr = MaxConsecHoursManager(
            state, page, on_solve_blocked_update)
        self._groups_mgr = GroupsManager(
            state, page, self._people_groups_row,
            self._debounce,
            on_text_change_cb=self._on_groups_text_change,
            on_structure_change_cb=self._on_groups_structure_change,
            rest_btn_callback=self._max_consec_mgr.open_dialog)

    # ── Debounce helper ────────────────────────────────────────────

    def _debounce(self, attr: str, fn, delay: float = 0.3):
        old = getattr(self, attr, None)
        if old is not None:
            old.cancel()
        t = threading.Timer(delay, fn)
        t.daemon = True
        t.start()
        setattr(self, attr, t)

    # ── Widget construction ────────────────────────────────────────

    def _build_ui(self):
        s = self.state

        self.tf_days = ft.TextField(
            value=s.days_text, multiline=True, min_lines=1, max_lines=12,
            label="Days (one per line)", width=_s(120), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_days_change)

        self._hours_col = ft.ListView(
            width=_s(200), spacing=_s(4), expand=True)

        self.tf_tasks = ft.TextField(
            value=s.tasks_text, multiline=True, min_lines=8, max_lines=200,
            label="Tasks (one per line)", width=_s(220), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_tasks_change)

        self._rotation_col = ft.ReorderableListView(
            width=_s(560), expand=True,
            on_reorder=self._on_rotation_reorder)

        self._people_groups_row = ft.Column(
            spacing=_s(8),
            expand=True,
            scroll=ft.ScrollMode.AUTO)

    # ── onChange handlers ──────────────────────────────────────────

    def _on_tasks_change(self, e):
        self.state.tasks_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_tasks', self._rebuild_rotation)
        self._on_solve_blocked_update()

    def _on_days_change(self, e):
        self.state.days_text = e.control.value
        self.state.invalidate_cache()
        self._debounce('_debounce_days', self._rebuild_hours)
        self._on_solve_blocked_update()

    def _on_rotation_reorder(self, e: ft.OnReorderEvent):
        handle_reorder(e, self.state, self.tf_tasks,
                       self._rotation_col, self.page,
                       self._on_solve_blocked_update)

    # ── Delegated rebuilds ────────────────────────────────────────

    def _rebuild_rotation(self):
        build_rotation_list(self.state, self._rotation_col,
                            self.page, self._on_solve_blocked_update)

    def _rebuild_hours(self):
        build_hours_per_day(self.state, self._hours_col,
                            self.page, self._on_solve_blocked_update)

    # ── Public API ────────────────────────────────────────────────

    def initial_build(self):
        self._rebuild_rotation()
        self._rebuild_hours()
        self._groups_mgr.build()

    def _on_groups_text_change(self):
        self._on_solve_blocked_update()

    def _on_groups_structure_change(self):
        self._groups_mgr.build()
        # Group renames/deletions/additions can leave shares and equality
        # rules referencing stale groups. Reconcile immediately and
        # refresh the solve button's enabled state.
        self.state.reconcile_group_equity()
        self.state.validate_group_equity()
        self._on_solve_blocked_update()

    def _build_groups_list(self):
        self._groups_mgr.build()

    def _build_hours_per_day(self):
        self._rebuild_hours()

    def _build_rotation_list(self):
        self._rebuild_rotation()

    # ── Layout ────────────────────────────────────────────────────

    def get_container(self) -> ft.Container:
        def _header(title: str) -> ft.Text:
            return ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD)

        left_col = ft.Column([
            _header("📅  Days & Hours"),
            ft.Divider(height=_s(4)),
            self.tf_days,
            self._hours_col,
        ], spacing=_s(4), width=_s(200), expand=False,
            scroll=ft.ScrollMode.AUTO)

        tasks_col = ft.Column([
            _header("🗂  Tasks"),
            ft.Divider(height=_s(4)),
            ft.Container(content=self.tf_tasks, expand=True),
        ], spacing=_s(4), width=_s(240), expand=False)

        rotation_col = ft.Column([
            _header("🏆  Priority · Rotation · Duration   "
                    "(drag  ≡  to reorder — top = highest)"),
            ft.Divider(height=_s(4)),
            ft.Container(content=self._rotation_col, expand=True),
        ], spacing=_s(4), width=_s(560), expand=False)

        people_col = ft.Column([
            _header("👥  People & Groups"),
            ft.Divider(height=_s(4)),
            self._people_groups_row,
        ], spacing=_s(4), expand=True,
            scroll=ft.ScrollMode.AUTO)

        return ft.Container(
            content=ft.Row(
                controls=[
                    left_col,
                    ft.VerticalDivider(width=_s(4)),
                    tasks_col,
                    ft.VerticalDivider(width=_s(4)),
                    rotation_col,
                    ft.VerticalDivider(width=_s(4)),
                    people_col,
                ],
                spacing=_s(16),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True),
            padding=_s(20), expand=True)