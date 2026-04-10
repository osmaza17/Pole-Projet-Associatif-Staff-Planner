import flet as ft
from constants import (
    _s, TASK_COLORS,
    UNAVAIL_COLOR, EMERG_COLOR, AVAIL_COLOR,
    DIFF_ADD_COLOR, DIFF_REMOVE_COLOR, DIFF_CHANGE_COLOR,
    BASE_ACTIVE_BG, BASE_ACTIVE_FG, BASE_IDLE_BG, BASE_IDLE_FG,
)
from .grids import build_task_grid, build_person_grid
from .workload import build_workload_section


# ── Helpers for styled issue sections ────────────────────────────────

def _section_header(title: str, count: int = 0, is_ok: bool = False) -> ft.Container:
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
        bgcolor=bg,
        padding=ft.padding.symmetric(_s(6), _s(10)),
        border_radius=4,
        margin=ft.margin.only(top=_s(8)),
    )


def _violation_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s(11), color="#BF360C"),
        bgcolor="#FFCCBC",
        padding=ft.padding.symmetric(_s(3), _s(8)),
        border_radius=4,
        margin=ft.margin.only(left=_s(12), top=_s(2)),
    )


def _info_chip(text: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(text, size=_s(11), color="#1565C0"),
        bgcolor="#BBDEFB",
        padding=ft.padding.symmetric(_s(3), _s(8)),
        border_radius=4,
        margin=ft.margin.only(left=_s(12), top=_s(2)),
    )


def _build_violations_section(title: str, violations: list[str]) -> list[ft.Control]:
    if not violations:
        return [_section_header(title, is_ok=True)]
    controls: list[ft.Control] = [_section_header(title, count=len(violations))]
    for v in violations:
        controls.append(_violation_chip(v))
    return controls


def _build_variety_section(variety_report: list[dict], tasks: list[str]) -> list[ft.Control]:
    if not variety_report:
        return []
    controls: list[ft.Control] = [_section_header("TASK VARIETY")]
    for entry in variety_report:
        t = entry["task"]
        line = (f"'{t}': {entry['touched']}/{entry['qualified']} qualified worked it, "
                f"{entry['repeated']} repeated, {entry['total_hours']}h total")
        controls.append(_info_chip(line))
    return controls


class OutputTab:

    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.Row(
            expand=True, spacing=_s(5),
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.START)
        self._task_view_runs: set = set()

    # ── Diff & base button handlers ───────────────────────────────────

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

    def _handle_task_view_click(self, key):
        self._task_view_runs.symmetric_difference_update({key})
        self.rebuild()
        self.page.update()

    # ── Run column builder ────────────────────────────────────────────

    def _build_run_column(self, run_idx, sol, people, tasks, hours, days,
                          availability, emergency, groups=None,
                          is_live=False, on_delete=None):
        s          = self.state
        status     = sol.get("status", "Solving...")
        is_solving = "Solving" in status
        assignment = sol.get("assignment", {})

        tv_key    = "live" if is_live else run_idx
        task_view = tv_key in self._task_view_runs

        diff_mode = None
        if not is_live:
            if run_idx == s.diff_state["ref"]:   diff_mode = "ref"
            elif run_idx == s.diff_state["cmp"]: diff_mode = "cmp"

        ref_sol = None
        if diff_mode == "cmp" and s.diff_state["ref"] is not None:
            ri = s.diff_state["ref"]
            if 0 <= ri < len(s.solution_history):
                ref_sol = s.solution_history[ri]["sol"]

        both_sel = s.diff_state["ref"] is not None and s.diff_state["cmp"] is not None
        buf = []

        # ── Header buttons ─────────────────────────────────────────────
        if is_live:
            d_lbl, d_bg, d_fg, d_click = "Compare", ft.Colors.GREY_400, ft.Colors.GREY_600, None
        elif diff_mode == "ref":
            d_lbl, d_bg, d_fg = "Reference", ft.Colors.LIGHT_BLUE_300, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)
        elif diff_mode == "cmp":
            d_lbl, d_bg, d_fg = "Compared", ft.Colors.AMBER_300, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)
        elif both_sel:
            d_lbl, d_bg, d_fg, d_click = "Compare", ft.Colors.GREY_400, ft.Colors.GREY_600, None
        else:
            d_lbl, d_bg, d_fg = "Compare", ft.Colors.GREEN_200, ft.Colors.BLACK
            d_click = lambda e, _i=run_idx: self._handle_diff_click(_i)

        diff_btn = ft.Container(
            content=ft.Text(d_lbl, size=_s(11), color=d_fg,
                            weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=d_bg, padding=ft.padding.symmetric(_s(4), _s(8)),
            border_radius=4, on_click=d_click)

        tv_lbl = "👤 Person View" if task_view else "📋 Task View"
        tv_bg  = ft.Colors.DEEP_PURPLE_200 if task_view else ft.Colors.DEEP_PURPLE_100
        tv_btn = ft.Container(
            content=ft.Text(tv_lbl, size=_s(11), color=ft.Colors.DEEP_PURPLE_900,
                            weight=ft.FontWeight.BOLD, no_wrap=True),
            bgcolor=tv_bg, padding=ft.padding.symmetric(_s(4), _s(8)),
            border_radius=4,
            on_click=lambda e, _k=tv_key: self._handle_task_view_click(_k))

        hdr_lbl = (f"⟳  Run #{run_idx+1}  ·  Solving…"
                   if is_live and is_solving
                   else f"Run #{run_idx+1}  ·  {status}")
        hdr_bg = "#E65100" if is_live and is_solving else "#1565C0"

        close_btn = (
            [ft.IconButton(icon=ft.Icons.CLOSE, icon_color=ft.Colors.WHITE,
                           icon_size=_s(16), tooltip="Remove this run",
                           style=ft.ButtonStyle(padding=ft.padding.all(_s(2))),
                           on_click=lambda e, _cb=on_delete: _cb())]
            if on_delete else [])

        title_row = ft.Row(
            controls=[ft.Text(hdr_lbl, weight=ft.FontWeight.BOLD, size=_s(13),
                              color=ft.Colors.WHITE, no_wrap=True, expand=True),
                      *close_btn],
            spacing=_s(4), vertical_alignment=ft.CrossAxisAlignment.CENTER)

        if not is_live:
            is_base  = run_idx == s.base_run_idx
            base_btn = ft.Container(
                content=ft.Text(
                    "✓ Active Base" if is_base else "📌 Set as Base",
                    size=_s(11),
                    color=BASE_ACTIVE_FG if is_base else BASE_IDLE_FG,
                    weight=ft.FontWeight.BOLD, no_wrap=True),
                bgcolor=BASE_ACTIVE_BG if is_base else BASE_IDLE_BG,
                padding=ft.padding.symmetric(_s(4), _s(8)), border_radius=4,
                on_click=lambda e, _i=run_idx: self._handle_base_click(_i))
            header_content = ft.Column([
                title_row,
                ft.Row([base_btn, diff_btn, tv_btn], spacing=_s(6),
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=_s(4), tight=True)
        else:
            header_content = ft.Column([
                title_row,
                ft.Row([diff_btn, tv_btn], spacing=_s(6)),
            ], spacing=_s(4), tight=True)

        buf.append(ft.Container(
            content=header_content, bgcolor=hdr_bg,
            padding=ft.padding.symmetric(_s(8), _s(12)),
            border_radius=6, margin=ft.margin.only(bottom=_s(6))))

        if not (is_live and is_solving):
            buf.append(ft.Container(
                content=ft.Row([
                    ft.Text(f"⏱ {sol.get('solve_time', 0.0):.1f}s", size=_s(11),
                            color=ft.Colors.GREY_700),
                    ft.Text(f"◎ Gap {sol.get('mip_gap', 0.0)*100:.3f}%", size=_s(11),
                            color=ft.Colors.GREY_700),
                ], spacing=_s(12)),
                padding=ft.padding.only(bottom=_s(4)),
            ))

        # ── Cell sizing ────────────────────────────────────────────────
        max_task_len = max([len(t) for t in tasks] + [0])
        max_p_len    = max([len(p) for p in people] + [0])
        max_h_len    = max([len(h) for h in hours.get(days[0], [])] + [0]) if days else 0

        CW = _s(min(80, max(55, (max(max_task_len, max_h_len) + 4) * 8))) + _s(16)
        NW = _s(max(75, (max_p_len + 4) * 8)) + _s(16)
        TW = _s(50) + _s(16)
        Ch = _s(34)

        # ── Grid content ──────────────────────────────────────────────
        if is_live and is_solving and not any(assignment.get(d) for d in days):
            buf.append(ft.Row([
                ft.ProgressRing(width=_s(22), height=_s(22), stroke_width=3),
                ft.Text("Solving in background…", italic=True, size=_s(12)),
            ], spacing=_s(10)))
        else:
            if is_live and is_solving:
                buf.append(ft.Row([
                    ft.ProgressRing(width=_s(16), height=_s(16), stroke_width=2),
                    ft.Text("Updating…", italic=True, size=_s(11),
                            color=ft.Colors.ORANGE_700),
                ], spacing=_s(6)))

            tc       = {t: TASK_COLORS[i % len(TASK_COLORS)] for i, t in enumerate(tasks)}
            ref_asgn = ref_sol.get("assignment", {}) if ref_sol else {}

            for j in days:
                if j not in assignment:
                    continue
                day_hrs = hours[j]
                buf.append(ft.Text(f"-- {j} --", weight=ft.FontWeight.BOLD, size=_s(16)))

                asgn    = assignment[j]
                ref_day = ref_asgn.get(j, {}) if ref_asgn else {}

                if task_view:
                    buf.extend(build_task_grid(
                        j, day_hrs, tasks, people, asgn,
                        availability, emergency, tc,
                        NW, CW, TW, Ch,
                        diff_mode=diff_mode,
                        ref_asgn_day=ref_day if ref_asgn else None))
                else:
                    buf.extend(build_person_grid(
                        j, day_hrs, people, asgn, s.person_colors,
                        availability, emergency, tc,
                        NW, CW, TW, Ch,
                        diff_mode=diff_mode,
                        ref_asgn_day=ref_day if ref_asgn else None))

                # Legend
                legend = [
                    ft.Container(
                        ft.Text(t, size=_s(10), weight=ft.FontWeight.BOLD, color=tc[t][1]),
                        bgcolor=tc[t][0],
                        padding=ft.padding.symmetric(_s(4), _s(8)), border_radius=4)
                    for t in tasks]
                for lbl, clr in [("Available", AVAIL_COLOR),
                                 ("Emergency", EMERG_COLOR),
                                 ("Unavailable", UNAVAIL_COLOR)]:
                    legend.append(ft.Container(
                        ft.Text(lbl, size=_s(10), color=ft.Colors.WHITE),
                        bgcolor=clr,
                        padding=ft.padding.symmetric(_s(4), _s(8)), border_radius=4))
                if diff_mode == "cmp":
                    for char, color, desc in [
                        ("+", DIFF_ADD_COLOR, "New assignment"),
                        ("−", DIFF_REMOVE_COLOR, "Removed"),
                        ("⇄", DIFF_CHANGE_COLOR, "Changed"),
                    ]:
                        legend.append(ft.Row([
                            ft.Container(
                                ft.Text(char, size=_s(8), color=ft.Colors.WHITE,
                                        weight=ft.FontWeight.BOLD),
                                width=_s(13), height=_s(13), bgcolor=color,
                                border_radius=7, alignment=ft.alignment.center),
                            ft.Text(desc, size=_s(10)),
                        ], spacing=_s(4),
                           vertical_alignment=ft.CrossAxisAlignment.CENTER))
                buf.append(ft.Row(legend, spacing=_s(8), wrap=True))
                buf.append(ft.Divider())

            # ── Post-grid: violations & reports ────────────────────────
            if not (is_live and is_solving):

                # Missing staff
                missing = sol.get("missing", [])
                buf.extend(_build_violations_section("MISSING STAFF", missing))

                # Constraint violations (only shown when broken)
                buf.extend(_build_violations_section(
                    "FORCE MANDATES", sol.get("force_violations", [])))
                buf.extend(_build_violations_section(
                    "JUST WORK", sol.get("just_work_violations", [])))
                buf.extend(_build_violations_section(
                    "JUST REST", sol.get("just_rest_violations", [])))
                buf.extend(_build_violations_section(
                    "CAPTAIN RULES", sol.get("captain_violations", [])))
                buf.extend(_build_violations_section(
                    "QUOTA FULFILMENT", sol.get("quota_violations", [])))
                buf.extend(_build_violations_section(
                    "ENEMY SEPARATION", sol.get("enemy_violations", [])))

                # Emergency call-ins
                emerg = sol.get("emerg_issues", [])
                if emerg:
                    buf.append(_section_header("EMERGENCY CALL-INS", count=len(emerg)))
                    for line in emerg:
                        buf.append(_info_chip(line))

                # Workload
                buf.extend(build_workload_section(
                    sol, people, groups or {}, diff_mode, ref_sol))

                # Task variety
                buf.extend(_build_variety_section(
                    sol.get("variety_report", []), tasks))

                # Task workload detail
                task_workload = sol.get("task_workload", {})
                if task_workload:
                    buf.append(_section_header("TASK HOURS BY PERSON"))
                    for t in tasks:
                        persons_on = [(p, task_workload.get(p, {}).get(t, 0))
                                      for p in people
                                      if task_workload.get(p, {}).get(t, 0) > 0]
                        if not persons_on:
                            continue
                        max_h = max(h for _, h in persons_on)
                        min_h = min(h for _, h in persons_on)
                        total = sum(h for _, h in persons_on)

                        buf.append(ft.Container(
                            content=ft.Column([
                                ft.Row([
                                    ft.Text(f"■ {t}", size=_s(12),
                                            weight=ft.FontWeight.W_600,
                                            color=ft.Colors.BLUE_GREY_800),
                                    ft.Text(f"{total}h total  ·  "
                                            f"range {min_h:.0f}–{max_h:.0f}h",
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
                            padding=ft.padding.only(left=_s(12), top=_s(4), bottom=_s(2)),
                        ))

        # ── Wrap in scrollable container ───────────────────────────────
        max_hrs   = max((len(hours[d]) for d in days), default=0)
        col_width = max(NW + CW * max_hrs + TW + _s(30), _s(500))
        return ft.Container(
            content=ft.ListView(controls=buf, spacing=_s(5), expand=True),
            width=col_width,
            padding=ft.padding.symmetric(horizontal=_s(16)),
            border=ft.border.only(right=ft.border.BorderSide(1, "#B0BEC5")))

    # ── Public rebuild ────────────────────────────────────────────────

    def rebuild(self, live_sol=None, live_people=None, live_tasks=None,
                live_hours=None, live_days=None,
                live_avail=None, live_emerg=None, live_groups=None):
        s    = self.state
        cols = []

        for i, entry in enumerate(s.solution_history):
            def _make_delete(idx):
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
                    new_tv = set()
                    for k in self._task_view_runs:
                        if k == "live":
                            new_tv.add(k)
                        elif k < idx:
                            new_tv.add(k)
                        elif k > idx:
                            new_tv.add(k - 1)
                    self._task_view_runs = new_tv
                    s.solution_history.pop(idx)
                    self.rebuild()
                    self.page.update()
                return _delete

            cols.append(self._build_run_column(
                i, entry["sol"], entry["people"], entry["tasks"],
                entry["hours"], entry["days"],
                entry["availability"], entry["emergency"],
                groups=entry.get("groups", {}),
                on_delete=_make_delete(i)))

        cols.reverse()

        if live_sol is not None:
            cols.insert(0, self._build_run_column(
                len(s.solution_history), live_sol,
                live_people, live_tasks, live_hours, live_days,
                live_avail, live_emerg, groups=live_groups, is_live=True))

        self._ct.controls = cols

    def get_container(self) -> ft.Container:
        return ft.Container(content=self._ct, padding=_s(10), expand=True)