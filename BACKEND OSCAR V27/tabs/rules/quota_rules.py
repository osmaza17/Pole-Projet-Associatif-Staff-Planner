"""
quota_rules.py  ←  tabs/rules/quota_rules.py

Quota Rules manager — each rule defines a minimum number of hours
that each selected person must work on each selected task, on each
selected day individually.

Stored rule format in state.quota_rules:
    {
        "people": [str, ...],           # selected persons
        "tasks":  {task: int_hours},    # required hours per task per day
        "days":   [str, ...],           # selected days
    }
"""

import copy
import flet as ft
from constants import _s, GROUP_HEADER_COLORS


class QuotaRulesManager:

    _BTN_COLOR   = "#6A1B9A"
    _CARD_BG     = "#F3E5F5"
    _CARD_BORDER = "#CE93D8"

    def __init__(self, state, page: ft.Page, container,
                 on_solve_blocked_update):
        self._state       = state
        self._page        = page
        self._col         = container
        self._on_solve_blocked_update = on_solve_blocked_update
        self._rule_errors : dict = {}

    # ══════════════════════════════════════════════════════════════════
    # Summary list
    # ══════════════════════════════════════════════════════════════════

    def build_summary(self):
        s = self._state

        add_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ADD_CHART, size=_s(16), color=ft.Colors.WHITE),
                ft.Text("Add Quota Rule", size=_s(12),
                        color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            ], spacing=_s(6), tight=True),
            bgcolor=self._BTN_COLOR,
            padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(8)),
            border_radius=6,
            on_click=lambda ev: self.open_dialog(ev),
            ink=True)

        buf = [add_btn]

        if not s.quota_rules:
            buf.append(ft.Container(
                ft.Text("No quota rules defined yet.\n"
                        "Click the button on the left to add one.",
                        size=_s(11), italic=True, color=ft.Colors.GREY_500,
                        text_align=ft.TextAlign.CENTER),
                padding=ft.padding.all(_s(12)),
                alignment=ft.alignment.center))
        else:
            for idx, rule in enumerate(s.quota_rules):
                buf.append(self._build_rule_card(idx, rule))

        self._col.controls = buf
        try:
            self._col.update()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════

    def validate_and_refresh(self):
        s = self._state
        people, tasks, _, days = s.dims()
        people_set = set(people)
        tasks_set  = set(tasks)
        days_set   = set(days)

        self._rule_errors = {}
        new_keys: set     = set()

        for idx, rule in enumerate(s.quota_rules):
            errs = []

            r_people = rule.get("people", [])
            r_tasks  = rule.get("tasks",  {})
            r_days   = rule.get("days",   [])

            missing_people = [p for p in r_people if p not in people_set]
            if missing_people:
                shown = missing_people[:3]
                tail  = (f" (+{len(missing_people)-3} more)"
                         if len(missing_people) > 3 else "")
                errs.append(f"Person(s) no longer exist: {', '.join(shown)}{tail}")

            missing_tasks = [t for t in r_tasks if t not in tasks_set]
            if missing_tasks:
                shown = missing_tasks[:3]
                tail  = (f" (+{len(missing_tasks)-3} more)"
                         if len(missing_tasks) > 3 else "")
                errs.append(f"Task(s) no longer exist: {', '.join(shown)}{tail}")

            for t, h in r_tasks.items():
                try:
                    if int(h) < 1:
                        errs.append(f"Invalid hours for '{t}': {h}")
                except (ValueError, TypeError):
                    errs.append(f"Invalid hours for '{t}': {h}")

            missing_days = [d for d in r_days if d not in days_set]
            if missing_days:
                errs.append(f"Day(s) no longer exist: {', '.join(missing_days)}")

            active_people = [p for p in r_people if p in people_set]
            active_tasks  = {t: h for t, h in r_tasks.items() if t in tasks_set}
            active_days   = [d for d in r_days if d in days_set]
            if not active_people:
                errs.append("No valid people in rule")
            if not active_tasks:
                errs.append("No valid tasks in rule")
            if not active_days:
                errs.append("No valid days in rule")

            if errs:
                self._rule_errors[idx] = errs
                new_keys.add(f"quota_{idx}")

        s.validation_errors["rules"] = (
            {k for k in s.validation_errors.get("rules", set())
             if not k.startswith("quota_")}
            | new_keys)

        self.build_summary()

    # ══════════════════════════════════════════════════════════════════
    # Rule card
    # ══════════════════════════════════════════════════════════════════

    def _build_rule_card(self, idx, rule):
        s      = self._state
        errs   = self._rule_errors.get(idx, [])
        is_bad = bool(errs)

        people_list = rule.get("people", [])
        tasks_dict  = rule.get("tasks",  {})
        days_list   = rule.get("days",   [])

        # People chips
        ppl_chips = []
        for p in people_list[:5]:
            ppl_chips.append(ft.Container(
                ft.Text(p, size=_s(9), color="#4A148C"),
                bgcolor="#E1BEE7", border_radius=10,
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))))
        if len(people_list) > 5:
            ppl_chips.append(ft.Text(
                f"+{len(people_list)-5}", size=_s(9),
                color="#6A1B9A", weight=ft.FontWeight.BOLD))

        # Task chips with hours
        tsk_chips = []
        for t, h in list(tasks_dict.items())[:5]:
            tsk_chips.append(ft.Container(
                ft.Row([
                    ft.Text(t, size=_s(9), color="#1B5E20"),
                    ft.Container(
                        ft.Text(f"{h}h", size=_s(9), color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD),
                        bgcolor="#1B5E20", border_radius=6,
                        padding=ft.padding.symmetric(
                            horizontal=_s(4), vertical=_s(0))),
                ], spacing=_s(3), tight=True),
                bgcolor="#C8E6C9", border_radius=10,
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))))
        if len(tasks_dict) > 5:
            tsk_chips.append(ft.Text(
                f"+{len(tasks_dict)-5}", size=_s(9),
                color="#2E7D32", weight=ft.FontWeight.BOLD))

        # Day chips
        day_chips = []
        for d in days_list:
            day_chips.append(ft.Container(
                ft.Text(d, size=_s(9), color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
                bgcolor="#1565C0", border_radius=4,
                padding=ft.padding.symmetric(horizontal=_s(5), vertical=_s(1))))

        def _del(e, _i=idx):
            s.quota_rules.pop(_i)
            self._on_solve_blocked_update()
            self.validate_and_refresh()

        def _edit(e, _i=idx):
            self.open_dialog(e, edit_index=_i)

        def _dup(e, _i=idx):
            s.quota_rules.insert(_i + 1, copy.deepcopy(s.quota_rules[_i]))
            self._on_solve_blocked_update()
            self.validate_and_refresh()

        error_block = ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ERROR_OUTLINE, size=_s(11),
                        color=ft.Colors.RED_700),
                ft.Text(err, size=_s(9), color=ft.Colors.RED_700, expand=True),
            ], spacing=_s(4))
            for err in errs
        ], spacing=_s(2), tight=True) if is_bad else None

        rows = [
            ft.Row([
                ft.Container(
                    ft.Text(f"#{idx+1}", size=_s(10),
                            weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    bgcolor=self._BTN_COLOR, border_radius=4,
                    padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.EDIT_OUTLINED, icon_size=_s(16),
                    icon_color=self._BTN_COLOR, tooltip="Edit rule",
                    on_click=_edit,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
                ft.IconButton(
                    icon=ft.Icons.COPY_ALL, icon_size=_s(16),
                    icon_color="#1565C0", tooltip="Duplicate rule",
                    on_click=_dup,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE, icon_size=_s(16),
                    icon_color=ft.Colors.RED_400, tooltip="Delete rule",
                    on_click=_del,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row(ppl_chips, spacing=_s(3), wrap=True, run_spacing=_s(2)),
            ft.Row(tsk_chips, spacing=_s(3), wrap=True, run_spacing=_s(2)),
            ft.Row(day_chips, spacing=_s(3), wrap=True, run_spacing=_s(2)),
        ]
        if error_block:
            rows.append(ft.Divider(height=_s(4), color=ft.Colors.RED_300))
            rows.append(error_block)

        return ft.Container(
            content=ft.Column(rows, spacing=_s(3), tight=True),
            width=_s(320),
            border=ft.border.all(2 if is_bad else 1,
                                 ft.Colors.RED_400 if is_bad
                                 else self._CARD_BORDER),
            border_radius=8, padding=ft.padding.all(_s(8)),
            bgcolor="#FFF0F0" if is_bad else self._CARD_BG)

    # ══════════════════════════════════════════════════════════════════
    # Dialog (Add / Edit)
    # ══════════════════════════════════════════════════════════════════

    def open_dialog(self, e, edit_index=None):
        s = self._state
        people, tasks, _, days = s.dims()
        is_edit = edit_index is not None

        # ── Per-dialog state ──────────────────────────────────────────
        sel_people: set  = set()
        sel_tasks : dict = {}    # {task: hours_int}
        sel_days  : set  = set()
        person_filter = [""]
        task_filter   = [""]

        if is_edit:
            r = s.quota_rules[edit_index]
            sel_people.update(r.get("people", []))
            for t, h in r.get("tasks", {}).items():
                try:
                    sel_tasks[t] = max(1, int(h))
                except (ValueError, TypeError):
                    sel_tasks[t] = 1
            sel_days.update(r.get("days", []))

        # ── Widget refs ───────────────────────────────────────────────
        people_chips_col = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
        tasks_chips_col  = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
        days_chips_col   = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
        preview_text     = ft.Text("", size=_s(11), italic=True,
                                    color=ft.Colors.GREY_500)

        save_btn = ft.ElevatedButton(
            "Update rule" if is_edit else "Save rule",
            icon=ft.Icons.SAVE, disabled=True,
            bgcolor=self._BTN_COLOR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(10)),
                shape=ft.RoundedRectangleBorder(radius=6)))

        # ── Validation ────────────────────────────────────────────────

        def _can_save():
            if not sel_people or not sel_tasks or not sel_days:
                return False
            for t, h in sel_tasks.items():
                if not isinstance(h, int) or h < 1:
                    return False
            return True

        def _refresh_save_btn():
            save_btn.disabled = not _can_save()
            try: save_btn.update()
            except Exception: pass

        # ── Chip factory ──────────────────────────────────────────────

        def _chip(label, selected, on_click,
                  sel_bg, sel_fg,
                  idle_bg="#ECEFF1", idle_fg="#424242",
                  extra_right=None):
            icon = ft.Icons.CHECK_CIRCLE if selected else ft.Icons.CHECK_BOX_OUTLINE_BLANK
            row_controls = [
                ft.Icon(icon, size=_s(14),
                        color=sel_fg if selected else ft.Colors.GREY_400),
                ft.Text(label, size=_s(11),
                        color=sel_fg if selected else idle_fg,
                        weight=ft.FontWeight.BOLD if selected else None),
            ]
            if extra_right is not None:
                row_controls.append(extra_right)
            return ft.Container(
                content=ft.Row(row_controls, spacing=_s(4), tight=True),
                bgcolor=sel_bg if selected else idle_bg,
                border_radius=16,
                border=ft.border.all(1.5, sel_bg if selected else ft.Colors.GREY_300),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=on_click, ink=True)

        def _make_search_field(hint, filter_ref, rebuild_fn, accent):
            def _on_change(ev):
                filter_ref[0] = ev.control.value or ""
                rebuild_fn()
            def _clear(ev):
                filter_ref[0] = ""
                tf = ev.control.parent
                if isinstance(tf, ft.TextField):
                    tf.value = ""
                    try: tf.update()
                    except Exception: pass
                rebuild_fn()
            return ft.TextField(
                hint_text=hint, prefix_icon=ft.Icons.SEARCH,
                on_change=_on_change, text_size=_s(11), height=_s(36),
                content_padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
                border_color=ft.Colors.GREY_400, focused_border_color=accent,
                border_radius=8, bgcolor=ft.Colors.WHITE,
                suffix=ft.IconButton(
                    icon=ft.Icons.CLEAR, icon_size=_s(14),
                    icon_color=ft.Colors.GREY_400, tooltip="Clear search",
                    style=ft.ButtonStyle(padding=ft.padding.all(0)),
                    on_click=_clear))

        # ── Preview ───────────────────────────────────────────────────

        def _refresh_preview():
            n_people = len(sel_people)
            n_tasks  = len(sel_tasks)
            n_days   = len(sel_days)
            if not (n_people and n_tasks and n_days):
                preview_text.value = "Select people, tasks (with hours), and days."
                preview_text.color = ft.Colors.GREY_500
            else:
                tasks_str = ", ".join(f"{t}={h}h" for t, h in sel_tasks.items())
                preview_text.value = (
                    f"👥 {n_people} people · "
                    f"📋 {tasks_str} per person per day · "
                    f"📅 {n_days} day(s)")
                preview_text.color = self._BTN_COLOR if _can_save() else ft.Colors.GREY_600
            try: preview_text.update()
            except Exception: pass

        # ── People chips (grouped) ────────────────────────────────────

        def _rebuild_people_chips():
            query  = person_filter[0].lower().strip()
            groups = s.build_groups(people)
            sections = []

            for g_idx, (gname, members) in enumerate(groups.items()):
                visible = [p for p in members if not query or query in p.lower()]
                if not visible:
                    continue
                g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]

                group_all_selected = all(p in sel_people for p in visible)

                def _toggle_group(ev, _members=visible, _all=group_all_selected):
                    if _all:
                        for p in _members:
                            sel_people.discard(p)
                    else:
                        for p in _members:
                            sel_people.add(p)
                    _rebuild_people_chips()
                    _refresh_preview()
                    _refresh_save_btn()

                group_chip = ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            ft.Icons.GROUP if group_all_selected else ft.Icons.GROUPS_OUTLINED,
                            size=_s(14),
                            color=ft.Colors.WHITE if group_all_selected else g_color),
                        ft.Text(
                            f"{gname} ({len(visible)})",
                            size=_s(11),
                            color=ft.Colors.WHITE if group_all_selected else g_color,
                            weight=ft.FontWeight.BOLD),
                    ], spacing=_s(4), tight=True),
                    bgcolor=g_color if group_all_selected else "#FFFFFF",
                    border=ft.border.all(1.5, g_color),
                    border_radius=16,
                    padding=ft.padding.symmetric(
                        horizontal=_s(10), vertical=_s(5)),
                    on_click=_toggle_group, ink=True)

                wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
                for p in visible:
                    is_sel = p in sel_people
                    def _toggle_p(ev, _p=p):
                        if _p in sel_people:
                            sel_people.discard(_p)
                        else:
                            sel_people.add(_p)
                        _rebuild_people_chips()
                        _refresh_preview()
                        _refresh_save_btn()
                    wrap.controls.append(
                        _chip(p, is_sel, _toggle_p,
                              sel_bg="#E1BEE7", sel_fg="#4A148C"))

                sections.append(ft.Column([
                    group_chip,
                    wrap,
                ], spacing=_s(4), tight=True))

            if not sections:
                sections = [ft.Text("No people match the search.",
                                    size=_s(11), italic=True,
                                    color=ft.Colors.GREY_500)]
            people_chips_col.controls = sections
            try: people_chips_col.update()
            except Exception: pass

        # ── Task chips ────────────────────────────────────────────────

        def _rebuild_task_chips():
            query   = task_filter[0].lower().strip()
            visible = [t for t in tasks if not query or query in t.lower()]
            if not visible:
                tasks_chips_col.controls = [ft.Text(
                    "No tasks match.", size=_s(11), italic=True,
                    color=ft.Colors.GREY_500)]
                try: tasks_chips_col.update()
                except Exception: pass
                return

            wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
            for t in visible:
                is_sel = t in sel_tasks

                if is_sel:
                    hours_val = sel_tasks[t]
                    tf_hours = ft.TextField(
                        value=str(hours_val),
                        width=_s(45), height=_s(26), text_size=_s(11),
                        content_padding=ft.padding.symmetric(
                            horizontal=_s(4), vertical=_s(2)),
                        text_align=ft.TextAlign.CENTER,
                        border_color="#1B5E20")

                    def _on_hours_change(ev, _t=t, _tf=tf_hours):
                        v = ev.control.value.strip()
                        if v == "":
                            sel_tasks[_t] = 0
                            _tf.border_color = ft.Colors.RED_400
                        else:
                            try:
                                n = int(v)
                                if n < 1 or str(n) != v:
                                    raise ValueError
                                sel_tasks[_t] = n
                                _tf.border_color = "#1B5E20"
                            except ValueError:
                                sel_tasks[_t] = 0
                                _tf.border_color = ft.Colors.RED_400
                        try: _tf.update()
                        except Exception: pass
                        _refresh_preview()
                        _refresh_save_btn()

                    tf_hours.on_change = _on_hours_change

                    def _toggle_t(ev, _t=t):
                        sel_tasks.pop(_t, None)
                        _rebuild_task_chips()
                        _refresh_preview()
                        _refresh_save_btn()

                    extra = ft.Row([
                        tf_hours,
                        ft.Text("h", size=_s(10), color="#1B5E20",
                                weight=ft.FontWeight.BOLD),
                    ], spacing=_s(2), tight=True)

                    wrap.controls.append(
                        _chip(t, True, _toggle_t,
                              sel_bg="#C8E6C9", sel_fg="#1B5E20",
                              extra_right=extra))
                else:
                    def _toggle_t(ev, _t=t):
                        sel_tasks[_t] = 1
                        _rebuild_task_chips()
                        _refresh_preview()
                        _refresh_save_btn()
                    wrap.controls.append(
                        _chip(t, False, _toggle_t,
                              sel_bg="#C8E6C9", sel_fg="#1B5E20"))

            tasks_chips_col.controls = [wrap]
            try: tasks_chips_col.update()
            except Exception: pass

        # ── Day chips ─────────────────────────────────────────────────

        def _rebuild_day_chips():
            wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
            for d in days:
                is_sel = d in sel_days
                def _toggle_d(ev, _d=d):
                    if _d in sel_days:
                        sel_days.discard(_d)
                    else:
                        sel_days.add(_d)
                    _rebuild_day_chips()
                    _refresh_preview()
                    _refresh_save_btn()
                wrap.controls.append(
                    _chip(d, is_sel, _toggle_d,
                          sel_bg="#BBDEFB", sel_fg="#0D47A1"))
            days_chips_col.controls = [wrap]
            try: days_chips_col.update()
            except Exception: pass

        # ── Section action buttons ────────────────────────────────────

        def _action_btn(label, icon, color, on_click):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon, size=_s(13), color=color),
                    ft.Text(label, size=_s(11), color=color,
                            weight=ft.FontWeight.BOLD),
                ], spacing=_s(4), tight=True),
                border=ft.border.all(1, color), border_radius=16,
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=on_click, ink=True)

        def _sel_all_people(ev):
            q = person_filter[0].lower().strip()
            for p in people:
                if not q or q in p.lower():
                    sel_people.add(p)
            _rebuild_people_chips(); _refresh_preview(); _refresh_save_btn()

        def _clr_people(ev):
            sel_people.clear()
            _rebuild_people_chips(); _refresh_preview(); _refresh_save_btn()

        def _sel_all_tasks(ev):
            q = task_filter[0].lower().strip()
            for t in tasks:
                if (not q or q in t.lower()) and t not in sel_tasks:
                    sel_tasks[t] = 1
            _rebuild_task_chips(); _refresh_preview(); _refresh_save_btn()

        def _clr_tasks(ev):
            sel_tasks.clear()
            _rebuild_task_chips(); _refresh_preview(); _refresh_save_btn()

        def _sel_all_days(ev):
            for d in days:
                sel_days.add(d)
            _rebuild_day_chips(); _refresh_preview(); _refresh_save_btn()

        def _clr_days(ev):
            sel_days.clear()
            _rebuild_day_chips(); _refresh_preview(); _refresh_save_btn()

        # ── Save / Cancel ─────────────────────────────────────────────

        def _on_save(ev):
            if not _can_save():
                return
            new_rule = {
                "people": sorted(sel_people),
                "tasks":  {t: int(h) for t, h in sel_tasks.items()},
                "days":   [d for d in days if d in sel_days],
            }
            if is_edit:
                s.quota_rules[edit_index] = new_rule
            else:
                s.quota_rules.append(new_rule)
            self._page.close(dlg)
            self._on_solve_blocked_update()
            self.validate_and_refresh()

        def _on_cancel(ev):
            self._page.close(dlg)

        save_btn.on_click = _on_save

        # ── Layout ────────────────────────────────────────────────────

        person_search = _make_search_field(
            "Search people…", person_filter,
            _rebuild_people_chips, "#6A1B9A")
        task_search = _make_search_field(
            "Search tasks…", task_filter,
            _rebuild_task_chips, "#1B5E20")

        def _col_section(icon, title, color, search_tf, chips_col,
                         sel_all_fn, clear_fn, section_height=_s(280)):
            inner = [
                ft.Row([
                    ft.Icon(icon, size=_s(15), color=color),
                    ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD,
                            color=color),
                ], spacing=_s(5)),
                ft.Row([
                    _action_btn("Select all", ft.Icons.SELECT_ALL, color, sel_all_fn),
                    _action_btn("Clear all",  ft.Icons.DESELECT, ft.Colors.GREY_600, clear_fn),
                ], spacing=_s(6)),
            ]
            if search_tf is not None:
                inner.append(search_tf)
            inner.append(ft.Container(
                chips_col,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=8, padding=ft.padding.all(_s(8)),
                height=section_height, expand=True))
            return ft.Column(inner, spacing=_s(5), tight=True, expand=True)

        people_section = _col_section(
            ft.Icons.PEOPLE, "People", "#6A1B9A",
            person_search, people_chips_col,
            _sel_all_people, _clr_people)
        tasks_section = _col_section(
            ft.Icons.CHECK_BOX, "Tasks (min hours/day)", "#1B5E20",
            task_search, tasks_chips_col,
            _sel_all_tasks, _clr_tasks)
        days_section = _col_section(
            ft.Icons.CALENDAR_MONTH, "Days", "#0D47A1",
            None, days_chips_col,
            _sel_all_days, _clr_days)

        preview_bar = ft.Container(
            preview_text, bgcolor="#F3E5F5", border_radius=8,
            border=ft.border.all(1, "#CE93D8"),
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)))

        content = ft.Column([
            ft.Text(
                "Define a minimum quota: how many hours of each task must each "
                "selected person work, on each selected day.\n\n"
                "Example: if you select Alice & Bob, task 'Iron' with 2h, "
                "and days Mon & Tue → both Alice and Bob must each do at least "
                "2 hours of Iron on Monday, and 2 hours of Iron on Tuesday.",
                size=_s(11), italic=True, color=ft.Colors.GREY_600),
            ft.Divider(height=_s(2)),
            preview_bar,
            ft.Row([
                people_section,
                ft.VerticalDivider(width=_s(4)),
                tasks_section,
                ft.VerticalDivider(width=_s(4)),
                days_section,
            ], spacing=_s(8),
               vertical_alignment=ft.CrossAxisAlignment.START,
               expand=True),
        ], spacing=_s(8), expand=True, width=_s(1200), height=_s(640))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.ADD_CHART, size=_s(20), color=self._BTN_COLOR),
                ft.Text("Edit Quota Rule" if is_edit else "Add Quota Rule",
                        weight=ft.FontWeight.BOLD, size=_s(16)),
            ], spacing=_s(8)),
            content=content,
            actions=[
                ft.TextButton("Cancel", on_click=_on_cancel,
                    style=ft.ButtonStyle(padding=ft.padding.symmetric(
                        horizontal=_s(16), vertical=_s(8)))),
                save_btn,
            ],
            actions_alignment=ft.MainAxisAlignment.END)

        # Initial render
        _rebuild_people_chips()
        _rebuild_task_chips()
        _rebuild_day_chips()
        _refresh_preview()
        _refresh_save_btn()

        self._page.open(dlg)