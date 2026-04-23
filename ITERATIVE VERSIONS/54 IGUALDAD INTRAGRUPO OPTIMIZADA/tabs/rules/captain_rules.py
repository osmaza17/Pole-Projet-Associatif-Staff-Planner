"""
captain_rules.py  ←  tabs/dimensions/captain_rules.py
Captain Rules manager for the Dimensions tab.
"""

import copy
import flet as ft
from constants import _s, GROUP_HEADER_COLORS


class CaptainRulesManager:

    _BTN_COLOR   = "#E65100"
    _CARD_BG     = "#FFF3E0"
    _CARD_BORDER = "#FFAB91"

    def __init__(self, state, page: ft.Page, captain_rules_col: ft.ListView):
        self._state       = state
        self._page        = page
        self._col         = captain_rules_col
        self._rule_errors : dict = {}

    # ── Summary list ───────────────────────────────────────────────────

    def build_summary(self):
        s = self._state

        add_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.PEOPLE, size=_s(16), color=ft.Colors.WHITE),
                ft.Text("Add Captain Rule", size=_s(12), color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD),
            ], spacing=_s(6), tight=True),
            bgcolor=self._BTN_COLOR,
            padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(8)),
            border_radius=6,
            on_click=lambda ev: self.open_dialog(ev),
            ink=True)

        buf = [add_btn]

        if not s.captain_rules:
            buf.append(ft.Container(
                ft.Text(
                    "No captain rules defined yet.\n"
                    "Click the button above to add one.",
                    size=_s(11), italic=True, color=ft.Colors.GREY_500,
                    text_align=ft.TextAlign.CENTER),
                padding=ft.padding.all(_s(12)),
                alignment=ft.alignment.center))
        else:
            for idx, rule in enumerate(s.captain_rules):
                buf.append(self._build_rule_card(idx, rule))

        self._col.controls = buf
        try:
            self._col.update()
        except Exception:
            pass

    # ── Validation ─────────────────────────────────────────────────────

    def validate_and_refresh(self):
        s = self._state
        people, _, _, _ = s.dims()
        people_set = set(people)

        self._rule_errors = {}
        new_keys: set     = set()

        _, tasks_list, _, days_list = s.dims()
        tasks_set = set(tasks_list)
        days_set  = set(days_list)

        for idx, rule in enumerate(s.captain_rules):
            errs       = []
            caps       = rule.get("captains", [])
            task_list  = rule.get("tasks", [])
            t          = task_list[0] if task_list else None
            hrs        = rule.get("hours", {})
            min_req    = rule.get("min_required", 1)

            active_caps_count = len([c for c in caps if c in people_set])
            if not isinstance(min_req, int) or min_req < 1:
                errs.append(f"min_required must be ≥ 1 (got {min_req})")
            elif active_caps_count > 0 and min_req > active_caps_count:
                errs.append(
                    f"min_required ({min_req}) > available captains ({active_caps_count})")

            missing = [c for c in caps if c not in people_set]
            if missing:
                errs.append(
                    f"Captain(s) no longer in groups: {', '.join(missing)}")

            if t and t not in tasks_set:
                errs.append(f"Task '{t}' no longer exists")
                t = None

            missing_days = [d for d in hrs if d not in days_set]
            if missing_days:
                errs.append(
                    f"Day(s) no longer exist: {', '.join(missing_days)}")

            active_caps = [c for c in caps if c in people_set]
            active_hrs  = {d: hs for d, hs in hrs.items() if d in days_set}

            if active_caps and t:
                no_demand = []
                for d, hs in active_hrs.items():
                    for h in hs:
                        raw = s.demand_st.get((t, h, d), "1")
                        try:
                            dem = int(raw) if str(raw).strip() else 0
                        except (ValueError, TypeError):
                            dem = 0
                        if dem == 0:
                            no_demand.append(f"{h} {d}")
                if no_demand:
                    shown = no_demand[:3]
                    tail  = f" (+{len(no_demand)-3} more)" if len(no_demand) > 3 else ""
                    errs.append(
                        f"No demand for '{t}' at: {', '.join(shown)}{tail}")

                skilled_caps = [
                    c for c in active_caps
                    if s.skills_st.get((c, t), 1) == 1
                ]
                if not skilled_caps:
                    cap_str = (", ".join(active_caps)
                               if len(active_caps) <= 3
                               else ", ".join(active_caps[:3])
                                  + f" +{len(active_caps)-3}")
                    errs.append(
                        f"No captain ({cap_str}) has skill for '{t}'")
                else:
                    bad_slots = []
                    for d, hs in hrs.items():
                        for h in hs:
                            ok_caps = [
                                c for c in active_caps
                                if s.avail_st.get((c, h, d), 1) in (1, 2)
                                and s.skills_st.get((c, t), 1) == 1
                            ]
                            if len(ok_caps) < min_req:
                                bad_slots.append(f"{h} {d}")
                    if bad_slots:
                        shown = bad_slots[:3]
                        tail  = (f" (+{len(bad_slots)-3} more)"
                                 if len(bad_slots) > 3 else "")
                        errs.append(
                            f"Fewer than {min_req} available+skilled captain(s) for '{t}' at: "
                            f"{', '.join(shown)}{tail}")

            elif active_caps and not t:
                bad_slots = []
                for d, hs in hrs.items():
                    for h in hs:
                        avail_caps = [
                            c for c in active_caps
                            if s.avail_st.get((c, h, d), 1) in (1, 2)
                        ]
                        if len(avail_caps) < min_req:
                            bad_slots.append(f"{h} {d}")
                if bad_slots:
                    shown = bad_slots[:3]
                    tail  = (f" (+{len(bad_slots)-3} more)"
                             if len(bad_slots) > 3 else "")
                    errs.append(
                        f"Fewer than {min_req} captain(s) available at: "
                        f"{', '.join(shown)}{tail}")

            if errs:
                self._rule_errors[idx] = errs
                new_keys.add(f"captain_{idx}")

        s.validation_errors["rules"] = (
            {k for k in s.validation_errors.get("rules", set())
             if not k.startswith("captain_")}
            | new_keys)

        self.build_summary()

    # ── Single rule card ───────────────────────────────────────────────

    def _build_rule_card(self, idx, rule):
        s         = self._state
        errs      = self._rule_errors.get(idx, [])
        is_bad    = bool(errs)

        cap_names   = rule["captains"]
        tsk_names   = rule["tasks"]
        hrs         = rule.get("hours", {})
        min_req     = rule.get("min_required", 1)

        cap_chips = []
        for c in cap_names[:4]:
            cap_chips.append(ft.Container(
                ft.Text(c, size=_s(9), color="#4A148C"),
                bgcolor="#E1BEE7", border_radius=10,
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))))
        if len(cap_names) > 4:
            cap_chips.append(ft.Text(
                f"+{len(cap_names)-4}", size=_s(9),
                color="#6A1B9A", weight=ft.FontWeight.BOLD))

        tsk_chips = []
        for t in tsk_names[:4]:
            tsk_chips.append(ft.Container(
                ft.Text(t, size=_s(9), color="#1B5E20"),
                bgcolor="#C8E6C9", border_radius=10,
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))))
        if len(tsk_names) > 4:
            tsk_chips.append(ft.Text(
                f"+{len(tsk_names)-4}", size=_s(9),
                color="#2E7D32", weight=ft.FontWeight.BOLD))

        min_req_badge = ft.Container(
            ft.Text(f"Min: {min_req}", size=_s(9), color="#BF360C",
                    weight=ft.FontWeight.BOLD),
            bgcolor="#FFCCBC", border_radius=10,
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2)))

        hours_parts = []
        for d, hs in hrs.items():
            if hs:
                hour_badges = [
                    ft.Container(
                        ft.Text(h, size=_s(8), color="#0D47A1"),
                        bgcolor="#BBDEFB", border_radius=8,
                        padding=ft.padding.symmetric(
                            horizontal=_s(4), vertical=_s(1)))
                    for h in hs
                ]
                hours_parts.append(ft.Column([
                    ft.Container(
                        ft.Text(d, size=_s(9), color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD),
                        bgcolor="#1565C0", border_radius=4,
                        padding=ft.padding.symmetric(
                            horizontal=_s(5), vertical=_s(1))),
                    ft.Row(hour_badges, spacing=_s(2),
                           wrap=True, run_spacing=_s(2)),
                ], spacing=_s(2), tight=True))

        def _del(e, _i=idx):
            s.captain_rules.pop(_i)
            self.validate_and_refresh()

        def _edit(e, _i=idx):
            self.open_dialog(e, edit_index=_i)

        def _dup(e, _i=idx):
            s.captain_rules.insert(_i + 1, copy.deepcopy(s.captain_rules[_i]))
            self.validate_and_refresh()

        error_block = ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ERROR_OUTLINE, size=_s(11),
                        color=ft.Colors.RED_700),
                ft.Text(err, size=_s(9), color=ft.Colors.RED_700,
                        expand=True),
            ], spacing=_s(4))
            for err in errs
        ], spacing=_s(2), tight=True) if is_bad else None

        rows = [
            ft.Row([
                ft.Container(
                    ft.Text(f"#{idx+1}", size=_s(10),
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.WHITE),
                    bgcolor=self._BTN_COLOR, border_radius=4,
                    padding=ft.padding.symmetric(
                        horizontal=_s(6), vertical=_s(2))),
                min_req_badge,
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.EDIT_OUTLINED, icon_size=_s(16),
                    icon_color=self._BTN_COLOR, tooltip="Edit rule",
                    on_click=_edit,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
                ft.IconButton(
                    icon=ft.Icons.COPY_ALL, icon_size=_s(16),
                    icon_color="#6A1B9A", tooltip="Duplicate rule",
                    on_click=_dup,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE, icon_size=_s(16),
                    icon_color=ft.Colors.RED_400, tooltip="Delete rule",
                    on_click=_del,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row(cap_chips, spacing=_s(3), wrap=True, run_spacing=_s(2)),
            ft.Row(tsk_chips, spacing=_s(3), wrap=True, run_spacing=_s(2)),
            ft.Column(hours_parts, spacing=_s(3), tight=True)
            if hours_parts
            else ft.Text("No hours", size=_s(9), italic=True,
                         color=ft.Colors.GREY_500),
        ]
        if error_block:
            rows.append(ft.Divider(height=_s(4), color=ft.Colors.RED_300))
            rows.append(error_block)

        return ft.Container(
            content=ft.Column(rows, spacing=_s(3), tight=True),
            border=ft.border.all(2 if is_bad else 1,
                                 ft.Colors.RED_400 if is_bad
                                 else self._CARD_BORDER),
            border_radius=8,
            padding=ft.padding.all(_s(8)),
            bgcolor="#FFF0F0" if is_bad else self._CARD_BG)

    # ══════════════════════════════════════════════════════════════════
    # Add / Edit dialog
    # ══════════════════════════════════════════════════════════════════

    def open_dialog(self, e, edit_index=None):
        s = self._state
        people, tasks, hours_map, days = s.dims()
        is_edit = edit_index is not None

        def _has_demand(t, h, d):
            raw = s.demand_st.get((t, h, d), "1")
            try:
                return int(raw) > 0 if str(raw).strip() else False
            except (ValueError, TypeError):
                return False

        def _make_panel_state(prefill=None):
            ps = {
                "sel_captains":      set(),
                "sel_task":          [None],
                "sel_hours":         {d: set() for d in days},
                "min_required":      [1],
                "captain_filter":    [""],
                "task_filter":       [""],
                "captain_chips_col": None,
                "task_chips_col":    None,
                "hours_col_inner":   None,
                "preview_text":      None,
                "min_req_tf":        None,
                "min_req_err":       None,
            }
            if prefill:
                ps["sel_captains"].update(prefill.get("captains", []))
                task_list = prefill.get("tasks", [])
                ps["sel_task"][0] = task_list[0] if task_list else None
                for d_key, hs in prefill.get("hours", {}).items():
                    if d_key in ps["sel_hours"]:
                        ps["sel_hours"][d_key].update(hs)
                ps["min_required"][0] = prefill.get("min_required", 1)
            return ps

        def _chip_selectable(label, selected, on_click,
                             sel_bg, sel_fg,
                             idle_bg="#ECEFF1", idle_fg="#424242",
                             radio=False):
            icon = (ft.Icons.RADIO_BUTTON_CHECKED if radio
                    else ft.Icons.CHECK_CIRCLE) if selected else (
                   ft.Icons.RADIO_BUTTON_UNCHECKED if radio
                   else ft.Icons.CIRCLE_OUTLINED)
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon, size=_s(14),
                            color=sel_fg if selected else ft.Colors.GREY_400),
                    ft.Text(label, size=_s(11),
                            color=sel_fg if selected else idle_fg,
                            weight=ft.FontWeight.BOLD if selected else None),
                ], spacing=_s(4), tight=True),
                bgcolor=sel_bg if selected else idle_bg,
                border_radius=16,
                border=ft.border.all(1.5, sel_bg if selected else ft.Colors.GREY_300),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=on_click, ink=True)

        def _chip_no_demand(label):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.MONEY_OFF, size=_s(11),
                            color=ft.Colors.RED_400),
                    ft.Text(label, size=_s(10), color=ft.Colors.RED_400,
                            italic=True),
                ], spacing=_s(3), tight=True),
                border=ft.border.all(1.5, ft.Colors.RED_300),
                border_radius=8, bgcolor="#FFEBEE",
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(3)),
                tooltip="No demand for selected task at this hour")

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

        def _validate_min_required(ps):
            raw = ps["min_req_tf"].value.strip() if ps["min_req_tf"] else "1"
            n_caps = len(ps["sel_captains"])
            err_text = ps["min_req_err"]
            try:
                val = int(raw)
                if val < 1:
                    if err_text:
                        err_text.value = "Must be ≥ 1"
                        err_text.visible = True
                        try: err_text.update()
                        except Exception: pass
                    return False
                if n_caps > 0 and val > n_caps:
                    if err_text:
                        err_text.value = f"Must be ≤ {n_caps} (pool size)"
                        err_text.visible = True
                        try: err_text.update()
                        except Exception: pass
                    return False
                ps["min_required"][0] = val
                if err_text:
                    err_text.value = ""
                    err_text.visible = False
                    try: err_text.update()
                    except Exception: pass
                return True
            except (ValueError, TypeError):
                if err_text:
                    err_text.value = "Must be a positive integer"
                    err_text.visible = True
                    try: err_text.update()
                    except Exception: pass
                return False

        def _panel_can_save(ps):
            any_hour = any(ps["sel_hours"][d] for d in days)
            if not (bool(ps["sel_captains"]) and ps["sel_task"][0] and any_hour):
                return False
            return _validate_min_required(ps)

        save_btn = ft.ElevatedButton(
            "Update rule" if is_edit else "Save rules",
            icon=ft.Icons.SAVE, disabled=True,
            bgcolor=self._BTN_COLOR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(10)),
                shape=ft.RoundedRectangleBorder(radius=6)))

        def _can_save_all():
            return panels and all(_panel_can_save(ps) for ps in panels)

        def _refresh_save_btn():
            save_btn.disabled = not _can_save_all()
            try: save_btn.update()
            except Exception: pass

        def _rebuild_preview(ps):
            caps = ps["sel_captains"]
            task = ps["sel_task"][0]
            hrs  = ps["sel_hours"]
            min_r = ps["min_required"][0]
            if not caps and not task:
                ps["preview_text"].value = "Select captains, a task and hours to create a rule."
                ps["preview_text"].color = ft.Colors.GREY_500
            else:
                parts = []
                if caps:
                    parts.append(f"👤 {len(caps)} captain{'s' if len(caps) > 1 else ''}")
                parts.append(f"🔢 min {min_r}")
                if task:
                    parts.append(f"📋 {task}")
                h_count = sum(len(hrs[d]) for d in days)
                if h_count:
                    day_parts = [f"{d}:{len(hrs[d])}" for d in days if hrs[d]]
                    parts.append(f"🕐 {h_count}h ({', '.join(day_parts)})")
                ps["preview_text"].value = "  ·  ".join(parts)
                ps["preview_text"].color = (self._BTN_COLOR
                                            if _panel_can_save(ps)
                                            else ft.Colors.GREY_600)
            try: ps["preview_text"].update()
            except Exception: pass

        def _rebuild_captain_chips(ps):
            query  = ps["captain_filter"][0].lower().strip()
            groups = s.build_groups(people)
            sections = []
            for g_idx, (gname, members) in enumerate(groups.items()):
                visible = [p for p in members if not query or query in p.lower()]
                if not visible:
                    continue
                g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
                wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
                for p in visible:
                    is_sel = p in ps["sel_captains"]
                    def _toggle(ev, _p=p, _ps=ps):
                        if _p in _ps["sel_captains"]:
                            _ps["sel_captains"].discard(_p)
                        else:
                            _ps["sel_captains"].add(_p)
                        _rebuild_captain_chips(_ps)
                        _validate_min_required(_ps)
                        _rebuild_preview(_ps)
                        _refresh_save_btn()
                    wrap.controls.append(
                        _chip_selectable(p, is_sel, _toggle,
                                         sel_bg="#E1BEE7", sel_fg="#4A148C"))
                sections.append(ft.Column([
                    ft.Container(
                        ft.Text(gname, size=_s(9), color=g_color,
                                weight=ft.FontWeight.BOLD),
                        padding=ft.padding.only(bottom=_s(2))),
                    wrap,
                ], spacing=_s(2), tight=True))
            if not sections:
                sections = [ft.Text("No people match the search.",
                                    size=_s(11), italic=True,
                                    color=ft.Colors.GREY_500)]
            ps["captain_chips_col"].controls = sections
            try: ps["captain_chips_col"].update()
            except Exception: pass

        def _rebuild_task_chips(ps):
            query   = ps["task_filter"][0].lower().strip()
            visible = [t for t in tasks if not query or query in t.lower()]
            if not visible:
                ps["task_chips_col"].controls = [ft.Text(
                    "No tasks match.", size=_s(11), italic=True,
                    color=ft.Colors.GREY_500)]
            else:
                wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
                for t in visible:
                    is_sel = t == ps["sel_task"][0]
                    def _toggle_t(ev, _t=t, _ps=ps):
                        _ps["sel_task"][0] = (None if _ps["sel_task"][0] == _t else _t)
                        if _ps["sel_task"][0]:
                            for _d in days:
                                _ps["sel_hours"][_d] = {
                                    _h for _h in _ps["sel_hours"][_d]
                                    if _has_demand(_ps["sel_task"][0], _h, _d)}
                        _rebuild_task_chips(_ps)
                        _rebuild_hours_section(_ps)
                        _rebuild_preview(_ps)
                        _refresh_save_btn()
                    wrap.controls.append(
                        _chip_selectable(t, is_sel, _toggle_t,
                                         sel_bg="#C8E6C9", sel_fg="#1B5E20",
                                         radio=True))
                ps["task_chips_col"].controls = [wrap]
            try: ps["task_chips_col"].update()
            except Exception: pass

        def _rebuild_hours_section(ps):
            t = ps["sel_task"][0]
            if t:
                for _d in days:
                    ps["sel_hours"][_d] = {
                        _h for _h in ps["sel_hours"][_d]
                        if _has_demand(t, _h, _d)}
            controls = []
            for d in days:
                wrap = ft.Row(wrap=True, spacing=_s(3), run_spacing=_s(3))
                for h in hours_map[d]:
                    no_demand = bool(t) and not _has_demand(t, h, d)
                    is_sel    = h in ps["sel_hours"][d]
                    if no_demand:
                        wrap.controls.append(_chip_no_demand(h))
                    else:
                        def _toggle_h(ev, _d=d, _h=h, _ps=ps):
                            if _h in _ps["sel_hours"][_d]:
                                _ps["sel_hours"][_d].discard(_h)
                            else:
                                _ps["sel_hours"][_d].add(_h)
                            _rebuild_hours_section(_ps)
                            _rebuild_preview(_ps)
                            _refresh_save_btn()
                        wrap.controls.append(
                            _chip_selectable(h, is_sel, _toggle_h,
                                             sel_bg="#BBDEFB", sel_fg="#0D47A1",
                                             idle_bg="#F5F5F5", radio=False))

                def _sel_all_h(ev, _d=d, _ps=ps):
                    _t = _ps["sel_task"][0]
                    for _h in hours_map[_d]:
                        if not _t or _has_demand(_t, _h, _d):
                            _ps["sel_hours"][_d].add(_h)
                    _rebuild_hours_section(_ps)
                    _rebuild_preview(_ps)
                    _refresh_save_btn()

                def _clr_h(ev, _d=d, _ps=ps):
                    _ps["sel_hours"][_d].clear()
                    _rebuild_hours_section(_ps)
                    _rebuild_preview(_ps)
                    _refresh_save_btn()

                controls.append(ft.Column([
                    ft.Row([
                        ft.Container(
                            ft.Text(d, size=_s(11), color=ft.Colors.WHITE,
                                    weight=ft.FontWeight.BOLD),
                            bgcolor="#1565C0", border_radius=4,
                            padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(4))),
                        ft.TextButton("Select all", on_click=_sel_all_h,
                            style=ft.ButtonStyle(padding=ft.padding.symmetric(
                                horizontal=_s(8), vertical=_s(4)))),
                        ft.TextButton("Clear", on_click=_clr_h,
                            style=ft.ButtonStyle(
                                padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
                                color=ft.Colors.GREY_600)),
                    ], spacing=_s(6), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    wrap,
                ], spacing=_s(3), tight=True))
            ps["hours_col_inner"].controls = controls
            try: ps["hours_col_inner"].update()
            except Exception: pass

        def _build_panel_widget(ps, pidx, removable=False):
            ps["captain_chips_col"] = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
            ps["task_chips_col"]    = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
            ps["hours_col_inner"]   = ft.Column(spacing=_s(6), scroll=ft.ScrollMode.AUTO)
            ps["preview_text"]      = ft.Text(
                "Select captains, a task and hours to create a rule.",
                size=_s(11), italic=True, color=ft.Colors.GREY_500)

            ps["min_req_err"] = ft.Text("", size=_s(9), color=ft.Colors.RED_400, visible=False)
            ps["min_req_tf"]  = ft.TextField(
                value=str(ps["min_required"][0]),
                width=_s(80), height=_s(36), text_size=_s(11),
                label="Min captains",
                label_style=ft.TextStyle(size=_s(9)),
                content_padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)),
                border_color=ft.Colors.GREY_400,
                focused_border_color=self._BTN_COLOR,
                border_radius=8)

            def _on_min_req_change(ev, _ps=ps):
                _validate_min_required(_ps)
                _rebuild_preview(_ps)
                _refresh_save_btn()

            ps["min_req_tf"].on_change = _on_min_req_change

            cap_search = _make_search_field(
                "Search people…", ps["captain_filter"],
                lambda _ps=ps: _rebuild_captain_chips(_ps), "#6A1B9A")
            tsk_search = _make_search_field(
                "Search tasks…", ps["task_filter"],
                lambda _ps=ps: _rebuild_task_chips(_ps), "#2E7D32")

            def _remove(ev):
                panels.pop(pidx)
                _rebuild_all_panels()
                _refresh_save_btn()

            header = ft.Row([
                ft.Container(
                    ft.Text(f"Rule {pidx + 1}", size=_s(11),
                            weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    bgcolor=self._BTN_COLOR, border_radius=4,
                    padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(3))),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.CLOSE, icon_size=_s(15),
                    icon_color=ft.Colors.RED_400,
                    tooltip="Remove this sub-rule",
                    visible=removable, on_click=_remove,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

            preview_bar = ft.Container(
                ps["preview_text"], bgcolor="#FFF3E0", border_radius=8,
                border=ft.border.all(1, "#FFCC80"),
                padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)))

            def _col_section(icon, title, color, search_tf, chips_col,
                             sel_all_fn=None, clear_fn=None, section_height=_s(300),
                             extra_header_controls=None):
                action_row = []
                if sel_all_fn:
                    action_row.append(_action_btn("Select all", ft.Icons.SELECT_ALL, color, sel_all_fn))
                if clear_fn:
                    action_row.append(_action_btn("Clear", ft.Icons.DESELECT, ft.Colors.GREY_600, clear_fn))
                inner = [
                    ft.Row([ft.Icon(icon, size=_s(15), color=color),
                            ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD, color=color)],
                           spacing=_s(5)),
                ]
                if extra_header_controls:
                    inner.append(ft.Row(extra_header_controls, spacing=_s(6),
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER))
                if action_row:
                    inner.append(ft.Row(action_row, spacing=_s(6)))
                inner += [
                    search_tf,
                    ft.Container(chips_col,
                                 border=ft.border.all(1, ft.Colors.GREY_300),
                                 border_radius=8, padding=ft.padding.all(_s(8)),
                                 height=section_height, expand=True),
                ]
                return ft.Column(inner, spacing=_s(5), tight=True, expand=True)

            def _sel_all_caps(ev, _ps=ps):
                q = _ps["captain_filter"][0].lower().strip()
                for p in people:
                    if not q or q in p.lower():
                        _ps["sel_captains"].add(p)
                _rebuild_captain_chips(_ps)
                _validate_min_required(_ps)
                _rebuild_preview(_ps)
                _refresh_save_btn()

            def _clr_caps(ev, _ps=ps):
                _ps["sel_captains"].clear()
                _rebuild_captain_chips(_ps)
                _validate_min_required(_ps)
                _rebuild_preview(_ps)
                _refresh_save_btn()

            def _sel_all_hrs(ev, _ps=ps):
                _t = _ps["sel_task"][0]
                for d in days:
                    for h in hours_map[d]:
                        if not _t or _has_demand(_t, h, d):
                            _ps["sel_hours"][d].add(h)
                _rebuild_hours_section(_ps); _rebuild_preview(_ps); _refresh_save_btn()

            def _clr_all_hrs(ev, _ps=ps):
                for d in days:
                    _ps["sel_hours"][d].clear()
                _rebuild_hours_section(_ps); _rebuild_preview(_ps); _refresh_save_btn()

            min_req_row = [ps["min_req_tf"], ps["min_req_err"]]

            hours_section = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCHEDULE, size=_s(15), color="#0D47A1"),
                    ft.Text("Hours", size=_s(12), weight=ft.FontWeight.BOLD, color="#0D47A1"),
                    ft.Text("(red = no demand for selected task)",
                            size=_s(9), italic=True, color=ft.Colors.GREY_500),
                ], spacing=_s(5)),
                ft.Row([
                    _action_btn("Select all", ft.Icons.SELECT_ALL, "#0D47A1", _sel_all_hrs),
                    _action_btn("Clear all", ft.Icons.DESELECT, ft.Colors.GREY_600, _clr_all_hrs),
                ], spacing=_s(6)),
                ft.Container(ps["hours_col_inner"],
                             border=ft.border.all(1, ft.Colors.GREY_300),
                             border_radius=8, padding=ft.padding.all(_s(8)),
                             height=_s(300), expand=True),
            ], spacing=_s(5), tight=True, expand=True)

            return ft.Container(
                content=ft.Column([
                    header, preview_bar,
                    ft.Row([
                        _col_section(ft.Icons.PEOPLE, "Captains", "#6A1B9A",
                                     cap_search, ps["captain_chips_col"],
                                     sel_all_fn=_sel_all_caps, clear_fn=_clr_caps,
                                     extra_header_controls=min_req_row),
                        ft.VerticalDivider(width=_s(4)),
                        _col_section(ft.Icons.CHECK_BOX, "Task", "#2E7D32",
                                     tsk_search, ps["task_chips_col"]),
                        ft.VerticalDivider(width=_s(4)),
                        hours_section,
                    ], spacing=_s(8), vertical_alignment=ft.CrossAxisAlignment.START, expand=True),
                ], spacing=_s(8), tight=True),
                border=ft.border.all(1, self._CARD_BORDER),
                border_radius=8, padding=ft.padding.all(_s(10)),
                bgcolor="#FFFDF8")

        def _action_btn(label, icon, color, on_click):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(icon, size=_s(13), color=color),
                    ft.Text(label, size=_s(11), color=color, weight=ft.FontWeight.BOLD),
                ], spacing=_s(4), tight=True),
                border=ft.border.all(1, color), border_radius=16,
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=on_click, ink=True)

        panels = []
        if is_edit:
            panels.append(_make_panel_state(prefill=s.captain_rules[edit_index]))
        else:
            panels.append(_make_panel_state())

        panels_col = ft.Column(spacing=_s(12), scroll=ft.ScrollMode.AUTO)

        def _rebuild_all_panels():
            removable = len(panels) > 1
            widgets   = []
            for pidx, ps in enumerate(panels):
                w = _build_panel_widget(ps, pidx, removable=removable)
                widgets.append(w)
                _rebuild_captain_chips(ps)
                _rebuild_task_chips(ps)
                _rebuild_hours_section(ps)
                _rebuild_preview(ps)
            panels_col.controls = widgets
            try: panels_col.update()
            except Exception: pass

        def _add_panel(ev):
            panels.append(_make_panel_state())
            _rebuild_all_panels()
            _refresh_save_btn()

        and_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ADD, size=_s(14), color=self._BTN_COLOR),
                ft.Text("And", size=_s(12), color=self._BTN_COLOR, weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True),
            border=ft.border.all(1.5, self._BTN_COLOR), border_radius=8,
            padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(8)),
            on_click=_add_panel, ink=True, visible=not is_edit)

        def _on_save(ev):
            if not _can_save_all():
                return
            new_rules = [
                {
                    "captains":     sorted(ps["sel_captains"]),
                    "tasks":        [ps["sel_task"][0]],
                    "min_required": ps["min_required"][0],
                    "hours":        {d: sorted(ps["sel_hours"][d])
                                     for d in days if ps["sel_hours"][d]},
                }
                for ps in panels
            ]
            if is_edit:
                s.captain_rules[edit_index] = new_rules[0]
            else:
                s.captain_rules.extend(new_rules)
            self._page.close(dlg)
            self.validate_and_refresh()

        def _on_cancel(ev):
            self._page.close(dlg)

        save_btn.on_click = _on_save
        _rebuild_all_panels()
        _refresh_save_btn()

        content = ft.Column([
            ft.Text(
                "Define which people (captains) must be present for "
                "a specific task during chosen hours. "
                "Use 'Min captains' to require more than one captain per hour. "
                "Hours in red = no demand for the selected task. "
                "Use \"And\" to add more captain rules in one go.",
                size=_s(11), italic=True, color=ft.Colors.GREY_600),
            ft.Divider(height=_s(2)),
            ft.Column([panels_col, and_btn],
                      spacing=_s(10), tight=True, expand=True,
                      scroll=ft.ScrollMode.AUTO),
        ], spacing=_s(8), expand=True, width=_s(1200), height=_s(640))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.PEOPLE, size=_s(20), color=self._BTN_COLOR),
                ft.Text("Edit Captain Rule" if is_edit else "Add Captain Rules",
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
        self._page.open(dlg)