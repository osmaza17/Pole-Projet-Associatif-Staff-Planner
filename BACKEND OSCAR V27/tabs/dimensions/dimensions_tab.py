"""
Dimensions tab — defines people, groups, tasks, days, hours, consecutive
limits, rotation settings, task durations, and TASK PRIORITY.

Layout:
  ┌──────────┬─────────────────────────────────────────────────┐
  │          │  Row 1: Tasks TF  |  Priority / Rotation list   │
  │ Days &   ├─────────────────────────────────────────────────┤
  │ Hours    │  Row 2: People & Groups                         │
  │ (column) │                                                 │
  └──────────┴─────────────────────────────────────────────────┘

Task priority
─────────────
The right-hand list is a drag-and-drop ReorderableListView. Dragging a
task rewrites `state.tasks_text` — the textfield on the left remains
the single source of truth, so save/load and manual editing keep
working with zero extra wiring.

  • Top of the list = highest priority (biggest penalty if uncovered).
  • Priority multipliers are computed in solver_controller as:
        priority(rank) = 1 + (N - rank)    (rank 0 = top)
  • A new task typed anywhere in the textfield appears in the same
    position in the reorderable list on the next rebuild.
"""

import threading
import flet as ft
from constants import _s, DEFAULT_HOURS_TEXT
from ui_helpers import UIHelpers
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

        # ── Days & Hours (left column) ─────────────────────────────────
        self.tf_days = ft.TextField(
            value=s.days_text, multiline=True, min_lines=1, max_lines=12,
            label="Days (one per line)", width=_s(120), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_days_change)

        self._hours_col = ft.ListView(
            width=_s(200), spacing=_s(4), expand=True)

        # ── Tasks (top-right row, left side) ───────────────────────────
        self.tf_tasks = ft.TextField(
            value=s.tasks_text, multiline=True, min_lines=8, max_lines=200,
            label="Tasks (one per line)", width=_s(220), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            on_change=self._on_tasks_change)

        # ── Priority / Rotation / Duration list (drag-and-drop) ───────
        # ReorderableListView doesn't support `spacing`; items get a
        # bottom border inside each row container instead.
        self._rotation_col = ft.ReorderableListView(
            width=_s(560),
            expand=True,
            on_reorder=self._on_rotation_reorder,
        )

        # ── People & Groups (bottom-right row) ─────────────────────────
        self._people_groups_row = ft.Row(
            spacing=_s(10),
            vertical_alignment=ft.CrossAxisAlignment.START,
            scroll=ft.ScrollMode.AUTO)

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

    # ── Drag-and-drop reorder ──────────────────────────────────────────

    def _on_rotation_reorder(self, e: ft.OnReorderEvent):
        """User dragged a task to a new position.

        We rewrite `state.tasks_text` (the single source of truth) and
        then rebuild the list so rank badges refresh.
        """
        s = self.state
        _, tasks, _, _ = s.dims()

        if not tasks:
            return

        old_idx = e.old_index
        new_idx = e.new_index

        if old_idx == new_idx:
            return
        if not (0 <= old_idx < len(tasks)):
            return

        # Reorder: pop from old, clamp, insert at new.
        # Flet's on_reorder uses the "after removal" index convention,
        # so a simple pop+insert is correct in both directions.
        item = tasks.pop(old_idx)
        new_idx = max(0, min(new_idx, len(tasks)))
        tasks.insert(new_idx, item)

        # Update the source of truth.
        new_text = "\n".join(tasks)
        s.tasks_text = new_text
        self.tf_tasks.value = new_text
        s.invalidate_cache()

        # Rebuild the list so rank badges and tooltips update.
        self._build_rotation_list()
        self._on_solve_blocked_update()

    # ── Sub-list builders ──────────────────────────────────────────────

    _TASK_MODE_CYCLE = ["R", "N", "S"]
    _TASK_MODE_LABEL = {"R": "ROT", "N": "OFF", "S": "STICK"}
    _TASK_MODE_COLOR = {
        "R": ft.Colors.GREEN_200,
        "N": ft.Colors.RED_200,
        "S": ft.Colors.AMBER_200,
    }

    @staticmethod
    def _rank_tier_color(rank: int, total: int) -> str:
        """Return a badge color based on priority tier.

        Top item is red (critical), top third is orange (high),
        middle third is blue-grey (normal), bottom third is grey (low).
        """
        if rank == 0:
            return ft.Colors.RED_600
        if total >= 3 and rank < total // 3:
            return ft.Colors.ORANGE_600
        if total >= 3 and rank < (2 * total) // 3:
            return ft.Colors.BLUE_GREY_500
        return ft.Colors.GREY_500

    def _build_rotation_list(self):
        s = self.state
        _, tasks, _, _ = s.dims()

        # Ensure task_duration_st exists on state
        if not hasattr(s, 'task_duration_st'):
            s.task_duration_st = {}

        buf = []
        _MODE_INDEX = {"R": 0, "N": 1, "S": 2}
        N = len(tasks)

        for rank, t in enumerate(tasks):
            priority_weight = 1 + (N - rank)  # matches solver_controller

            # ── Drag handle (visual affordance) ──────────────────────
            drag_handle = ft.Icon(
                ft.Icons.DRAG_INDICATOR,
                size=_s(18),
                color=ft.Colors.GREY_500,
                tooltip="Drag to reorder priority",
            )

            # ── Rank badge (#1 = highest priority) ───────────────────
            rank_badge = ft.Container(
                content=ft.Text(
                    f"#{rank + 1}",
                    size=_s(10),
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE,
                ),
                bgcolor=self._rank_tier_color(rank, N),
                padding=ft.padding.symmetric(
                    horizontal=_s(6), vertical=_s(2)),
                border_radius=4,
                width=_s(34),
                alignment=ft.alignment.center,
                tooltip=f"Priority weight: {priority_weight}×",
            )

            # ── Rotation mode segmented button ───────────────────────
            mode = s.rotation_st.get(t, "R")
            seg = ft.SegmentedButton(
                selected={str(_MODE_INDEX[mode])},
                allow_multiple_selection=False,
                allow_empty_selection=False,
                style=ft.ButtonStyle(
                    padding=ft.padding.symmetric(
                        horizontal=_s(4), vertical=_s(2)),
                    text_style=ft.TextStyle(size=_s(9)),
                ),
                segments=[
                    ft.Segment(value="0", label=ft.Text("ROT",  size=_s(9))),
                    ft.Segment(value="1", label=ft.Text("OFF",  size=_s(9))),
                    ft.Segment(value="2", label=ft.Text("STICK", size=_s(9))),
                ],
                data=t,
            )

            def _on_seg_change(e):
                idx = int(next(iter(e.control.selected)))
                s.rotation_st[e.control.data] = self._TASK_MODE_CYCLE[idx]

            seg.on_change = _on_seg_change

            # ── Duration TextField ───────────────────────────────────
            s.task_duration_st.setdefault(t, "1")
            dur_tf = ft.TextField(
                value=s.task_duration_st[t],
                width=_s(55),
                text_size=_s(11),
                text_align=ft.TextAlign.CENTER,
                label="h",
                label_style=ft.TextStyle(size=_s(9)),
                input_filter=ft.NumbersOnlyInputFilter(),
                data=t,
            )

            def _on_dur_change(e):
                s.task_duration_st[e.control.data] = e.control.value
                s.invalidate_cache()
                self._on_solve_blocked_update()

            dur_tf.on_change = _on_dur_change

            # ── Row container (stable key + bottom divider) ──────────
            # The `key` helps Flet track widgets across rebuilds; using
            # the task name is fine because dims() deduplicates.
            row = ft.Container(
                key=f"task_row_{t}",
                content=ft.Row(
                    [
                        drag_handle,
                        rank_badge,
                        UIHelpers.lbl(t, _s(140)),
                        seg,
                        dur_tf,
                    ],
                    spacing=_s(6),
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.padding.symmetric(
                    horizontal=_s(6), vertical=_s(4)),
                border=ft.border.only(
                    bottom=ft.border.BorderSide(1, ft.Colors.GREY_300)),
            )

            buf.append(row)

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

    def initial_build(self):
        self._build_rotation_list()
        self._build_hours_per_day()
        self._groups_mgr.build()

    def _on_groups_text_change(self):
        self._on_solve_blocked_update()

    def _on_groups_structure_change(self):
        self._groups_mgr.build()
        self._on_solve_blocked_update()

    def _build_groups_list(self):
        self._groups_mgr.build()

    # ── Layout ─────────────────────────────────────────────────────────

    def get_container(self) -> ft.Container:
        def _header(title: str) -> ft.Text:
            return ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD)

        # ── Left column: Days & Hours ──────────────────────────────────
        left_col = ft.Column(
            [
                _header("📅  Days & Hours"),
                ft.Divider(height=_s(4)),
                self.tf_days,
                self._hours_col,
            ],
            spacing=_s(4),
            width=_s(200),
            expand=False,
            scroll=ft.ScrollMode.AUTO,
        )

        # ── Top-right row: Tasks + Priority/Rotation/Duration list ────
        top_right = ft.Row(
            [
                ft.Column(
                    [
                        _header("🗂  Tasks"),
                        ft.Divider(height=_s(4)),
                        ft.Container(
                            content=self.tf_tasks,
                            height=_s(280),
                        ),
                    ],
                    spacing=_s(4),
                    width=_s(240),
                    expand=False,
                ),
                ft.VerticalDivider(width=_s(4)),
                ft.Column(
                    [
                        _header("🏆  Priority · Rotation · Duration   "
                                "(drag  ≡  to reorder — top = highest)"),
                        ft.Divider(height=_s(4)),
                        ft.Container(
                            content=self._rotation_col,
                            expand=True,
                        ),
                    ],
                    spacing=_s(4),
                    expand=True,
                ),
            ],
            spacing=_s(10),
            vertical_alignment=ft.CrossAxisAlignment.START,
            expand=True,
        )

        # ── Bottom-right row: People & Groups ──────────────────────────
        bottom_right = ft.Column(
            [
                _header("👥  People & Groups"),
                ft.Divider(height=_s(4)),
                self._people_groups_row,
            ],
            spacing=_s(4),
            expand=True,
        )

        # ── Right side: top + bottom stacked ───────────────────────────
        right_side = ft.Column(
            [
                ft.Container(content=top_right, height=_s(600)),
                ft.Divider(height=_s(4)),
                ft.Container(content=bottom_right, expand=True),
            ],
            spacing=_s(4),
            expand=True,
        )

        return ft.Container(
            content=ft.Row(
                controls=[
                    left_col,
                    ft.VerticalDivider(width=_s(4)),
                    right_side,
                ],
                spacing=_s(16),
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True,
            ),
            padding=_s(20),
            expand=True,
        )