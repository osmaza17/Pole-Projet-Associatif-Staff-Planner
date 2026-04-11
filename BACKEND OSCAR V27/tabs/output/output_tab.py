"""
OutputTab — top-level container for the Output tab.

Owns view-mode state and orchestrates rebuilding run columns. All UI
construction is delegated to pure builders in output_run_column.
"""

import flet as ft
from constants import _s
from .output_run_column import build_run_column


class OutputTab:

    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page = page
        self._ct = ft.Row(
            expand=True, spacing=_s(5),
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.START)
        # Per-run view mode: "person" (default), "task", or "location"
        self._view_mode_runs: dict = {}

    # ── Handlers ──────────────────────────────────────────────────────

    def _handle_base_click(self, run_idx: int):
        s = self.state
        s.base_run_idx = None if s.base_run_idx == run_idx else run_idx
        self.rebuild()
        self.page.update()

    def _handle_diff_click(self, run_idx: int):
        ds = self.state.diff_state
        if run_idx == ds["ref"]:
            ds["ref"] = ds["cmp"]; ds["cmp"] = None
        elif run_idx == ds["cmp"]:
            ds["cmp"] = None
        elif ds["ref"] is None:
            ds["ref"] = run_idx
        elif ds["cmp"] is None:
            ds["cmp"] = run_idx
        self.rebuild()
        self.page.update()

    def _handle_view_mode_click(self, key, mode):
        if mode == "person":
            self._view_mode_runs.pop(key, None)
        else:
            self._view_mode_runs[key] = mode
        self.rebuild()
        self.page.update()

    def _make_delete(self, idx):
        s = self.state
        def _delete():
            ds = s.diff_state
            if ds["ref"] == idx:
                ds["ref"] = ds["cmp"]; ds["cmp"] = None
            elif ds["cmp"] == idx:
                ds["cmp"] = None
            for key in ("ref", "cmp"):
                if ds[key] is not None and ds[key] > idx:
                    ds[key] -= 1
            if s.base_run_idx == idx:
                s.base_run_idx = None
            elif s.base_run_idx is not None and s.base_run_idx > idx:
                s.base_run_idx -= 1
            new_vm = {}
            for k, v in self._view_mode_runs.items():
                if k == "live":
                    new_vm[k] = v
                elif k < idx:
                    new_vm[k] = v
                elif k > idx:
                    new_vm[k - 1] = v
            self._view_mode_runs = new_vm
            s.solution_history.pop(idx)
            self.rebuild()
            self.page.update()
        return _delete

    # ── Column construction ───────────────────────────────────────────

    def _column_for(self, run_idx, sol, people, tasks, hours, days,
                    availability, emergency, groups, is_live, on_delete):
        s = self.state
        tv_key = "live" if is_live else run_idx
        view_mode = self._view_mode_runs.get(tv_key, "person")

        diff_mode = None
        if not is_live:
            if run_idx == s.diff_state["ref"]:   diff_mode = "ref"
            elif run_idx == s.diff_state["cmp"]: diff_mode = "cmp"

        ref_sol = None
        if diff_mode == "cmp" and s.diff_state["ref"] is not None:
            ri = s.diff_state["ref"]
            if 0 <= ri < len(s.solution_history):
                ref_sol = s.solution_history[ri]["sol"]

        both_sel = (s.diff_state["ref"] is not None
                    and s.diff_state["cmp"] is not None)
        base_sel = (not is_live) and (run_idx == s.base_run_idx)

        return build_run_column(
            run_idx=run_idx, sol=sol, people=people, tasks=tasks,
            hours=hours, days=days,
            availability=availability, emergency=emergency,
            groups=groups, is_live=is_live,
            view_mode=view_mode, diff_mode=diff_mode, ref_sol=ref_sol,
            base_selected=base_sel, both_diff_selected=both_sel,
            person_colors=s.person_colors,
            task_location_idx=s.task_location_idx_st,
            location_names=s.location_names_st,
            on_base_click=(None if is_live
                           else (lambda _i=run_idx: self._handle_base_click(_i))),
            on_diff_click=(lambda _i=run_idx: self._handle_diff_click(_i)),
            on_view_mode_click=(lambda m, _k=tv_key:
                                self._handle_view_mode_click(_k, m)),
            on_delete=on_delete)

    # ── Rebuild ───────────────────────────────────────────────────────

    def rebuild(self, live_sol=None, live_people=None, live_tasks=None,
                live_hours=None, live_days=None,
                live_avail=None, live_emerg=None, live_groups=None):
        s = self.state
        cols = []
        for i, entry in enumerate(s.solution_history):
            cols.append(self._column_for(
                i, entry["sol"], entry["people"], entry["tasks"],
                entry["hours"], entry["days"],
                entry["availability"], entry["emergency"],
                groups=entry.get("groups", {}),
                is_live=False,
                on_delete=self._make_delete(i)))
        cols.reverse()

        if live_sol is not None:
            cols.insert(0, self._column_for(
                len(s.solution_history), live_sol,
                live_people, live_tasks, live_hours, live_days,
                live_avail, live_emerg, groups=live_groups,
                is_live=True, on_delete=None))

        self._ct.controls = cols

    def get_container(self) -> ft.Container:
        return ft.Container(content=self._ct, padding=_s(10), expand=True)