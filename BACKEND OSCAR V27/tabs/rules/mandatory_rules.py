"""
mandatory_rules.py  ←  tabs/dimensions/mandatory_rules.py

Stored rule format in state.mandatory_rules:
    {
        "person":           str,
        "task":             str | None,
        "just_do_anything": bool,
        "just_rest":        bool,
        "hours":            {day: [hour, …]}
    }
"""

import copy
import flet as ft
from constants import _s, GROUP_HEADER_COLORS


class MandatoryRulesManager:

    _BTN_COLOR   = "#1565C0"
    _CARD_BG     = "#E3F2FD"
    _CARD_BORDER = "#90CAF9"

    def __init__(self, state, page: ft.Page, col: ft.ListView):
        self._state       = state
        self._page        = page
        self._col         = col
        self._rule_errors : dict = {}

    # ══════════════════════════════════════════════════════════════════
    # Summary list
    # ══════════════════════════════════════════════════════════════════

    def build_summary(self):
        s = self._state

        add_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ASSIGNMENT_IND, size=_s(16), color=ft.Colors.WHITE),
                ft.Text("Add Mandatory Rule", size=_s(12),
                        color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
            ], spacing=_s(6), tight=True),
            bgcolor=self._BTN_COLOR,
            padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(8)),
            border_radius=6, on_click=lambda ev: self.open_dialog(ev), ink=True)

        buf = [add_btn]

        if not s.mandatory_rules:
            buf.append(ft.Container(
                ft.Text("No mandatory rules defined yet.\n"
                        "Click the button above to add one.",
                        size=_s(11), italic=True, color=ft.Colors.GREY_500,
                        text_align=ft.TextAlign.CENTER),
                padding=ft.padding.all(_s(12)),
                alignment=ft.alignment.center))
        else:
            for idx, rule in enumerate(s.mandatory_rules):
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
        people, _, _, _ = s.dims()
        people_set = set(people)

        self._rule_errors = {}
        new_keys: set     = set()

        _, tasks_list, _, days_list = s.dims()
        tasks_set = set(tasks_list)
        days_set  = set(days_list)

        # Pre-index all rule slots for cross-rule conflict detection
        # work_slots[person] = {(h,d): rule_idx, …}
        # rest_slots[person] = {(h,d): rule_idx, …}
        work_slots: dict[str, dict[tuple, int]] = {}
        rest_slots: dict[str, dict[tuple, int]] = {}
        for ri, rule in enumerate(s.mandatory_rules):
            p    = rule.get("person")
            jrest = rule.get("just_rest", False)
            hrs   = rule.get("hours", {})
            bucket = rest_slots if jrest else work_slots
            bucket.setdefault(p, {})
            for d, hs in hrs.items():
                for h in hs:
                    bucket[p][(h, d)] = ri

        for idx, rule in enumerate(s.mandatory_rules):
            errs = []
            p    = rule.get("person")
            t    = rule.get("task")
            jda  = rule.get("just_do_anything", False)
            jrest = rule.get("just_rest", False)
            hrs  = rule.get("hours", {})

            # ── 1. Person still exists? ────────────────────────────────
            if p not in people_set:
                errs.append(f"Person '{p}' no longer exists in any group")

            # ── 2. Task still exists? (specific mandates only) ─────────
            if not jda and not jrest and t and t not in tasks_set:
                errs.append(f"Task '{t}' no longer exists")
                t = None

            # ── 3. Days still exist? ────────────────────────────────────
            missing_days = [d for d in hrs if d not in days_set]
            if missing_days:
                errs.append(
                    f"Day(s) no longer exist: {', '.join(missing_days)}")

            active_hrs = {d: hs for d, hs in hrs.items() if d in days_set}

            if p in people_set:
                if not jda and not jrest and t:
                    # ── Skill check ────────────────────────────────────
                    if s.skills_st.get((p, t), 1) != 1:
                        errs.append(f"'{p}' no longer has skill for '{t}'")

                    # ── Demand check ───────────────────────────────────
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
                        tail  = (f" (+{len(no_demand)-3} more)"
                                 if len(no_demand) > 3 else "")
                        errs.append(
                            f"No demand for '{t}' at: {', '.join(shown)}{tail}")

                # ── Availability check (all mandate types) ─────────────
                unavail = []
                for d, hs in active_hrs.items():
                    for h in hs:
                        v = s.avail_st.get((p, h, d), 1)
                        if v not in (1, 2):
                            unavail.append(f"{h} {d}")
                if unavail:
                    shown = unavail[:3]
                    tail  = (f" (+{len(unavail)-3} more)"
                             if len(unavail) > 3 else "")
                    errs.append(
                        f"'{p}' unavailable at: {', '.join(shown)}{tail}")

                # ── Work ↔ Rest conflict check ─────────────────────────
                if jrest:
                    # This is a rest rule — check if any work rule hits same slots
                    conflict_bucket = work_slots.get(p, {})
                else:
                    # This is a work rule — check if any rest rule hits same slots
                    conflict_bucket = rest_slots.get(p, {})

                conflicts = []
                for d, hs in active_hrs.items():
                    for h in hs:
                        other_idx = conflict_bucket.get((h, d))
                        if other_idx is not None and other_idx != idx:
                            conflicts.append(f"{h} {d} (rule #{other_idx+1})")
                if conflicts:
                    shown = conflicts[:3]
                    tail  = (f" (+{len(conflicts)-3} more)"
                             if len(conflicts) > 3 else "")
                    kind = "work" if jrest else "rest"
                    errs.append(
                        f"Conflicts with {kind} rule at: {', '.join(shown)}{tail}")

            if errs:
                self._rule_errors[idx] = errs
                new_keys.add(f"mandatory_{idx}")

        s.validation_errors["rules"] = (
            {k for k in s.validation_errors.get("rules", set())
             if not k.startswith("mandatory_")}
            | new_keys)

        self.build_summary()

    # ══════════════════════════════════════════════════════════════════
    # Rule card
    # ══════════════════════════════════════════════════════════════════

    def _build_rule_card(self, idx, rule):
        s      = self._state
        errs   = self._rule_errors.get(idx, [])
        is_bad = bool(errs)

        person = rule["person"]
        task   = rule.get("task")
        jda    = rule.get("just_do_anything", False)
        jrest  = rule.get("just_rest", False)
        hrs    = rule.get("hours", {})

        person_chip = ft.Container(
            ft.Text(person, size=_s(9), color="#4A148C"),
            bgcolor="#E1BEE7", border_radius=10,
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2)))

        if jrest:
            task_label = "Forced Rest"
            task_fg    = "#B71C1C"
            task_bg    = "#FFCDD2"
        elif jda:
            task_label = "Any task"
            task_fg    = "#004D40"
            task_bg    = "#B2DFDB"
        else:
            task_label = task or "?"
            task_fg    = "#1B5E20"
            task_bg    = "#C8E6C9"

        task_chip = ft.Container(
            ft.Text(task_label, size=_s(9), color=task_fg),
            bgcolor=task_bg, border_radius=10,
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2)))

        hours_parts = []
        for d, hs in hrs.items():
            if hs:
                badges = [
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
                    ft.Row(badges, spacing=_s(2), wrap=True, run_spacing=_s(2)),
                ], spacing=_s(2), tight=True))

        def _del(e, _i=idx):
            s.mandatory_rules.pop(_i)
            self.validate_and_refresh()

        def _edit(e, _i=idx):
            self.open_dialog(e, edit_index=_i)

        def _dup(e, _i=idx):
            s.mandatory_rules.insert(_i + 1, copy.deepcopy(s.mandatory_rules[_i]))
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
                    icon_color="#6A1B9A", tooltip="Duplicate rule",
                    on_click=_dup,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE, icon_size=_s(16),
                    icon_color=ft.Colors.RED_400, tooltip="Delete rule",
                    on_click=_del,
                    style=ft.ButtonStyle(padding=ft.padding.all(0))),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([
                person_chip,
                ft.Icon(ft.Icons.ARROW_FORWARD, size=_s(12), color=ft.Colors.GREY_400),
                task_chip,
            ], spacing=_s(4), vertical_alignment=ft.CrossAxisAlignment.CENTER),
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
            border_radius=8, padding=ft.padding.all(_s(8)),
            bgcolor="#FFF0F0" if is_bad else self._CARD_BG)

    # ══════════════════════════════════════════════════════════════════
    # Dialog
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

        def _has_skill(p, t):
            return s.skills_st.get((p, t), 1) == 1

        def _is_avail(p, h, d):
            return s.avail_st.get((p, h, d), 1) in (1, 2)

        def _find_conflicts(ps, pidx, all_panels):
            p     = ps["person"][0]
            t     = ps["task"][0]
            jda   = ps["just_do_anything"][0]
            jrest = ps["just_rest"][0]
            if not p:
                return []
            my_hd = {(h, d) for d, hs in ps["sel_hours"].items() for h in hs}
            if not my_hd:
                return []

            is_work = not jrest   # jda or specific task → work
            conflicts = []

            # Check against saved rules
            for si, saved in enumerate(s.mandatory_rules):
                if is_edit and si == edit_index:
                    continue
                if saved["person"] != p:
                    continue
                saved_is_rest = saved.get("just_rest", False)

                # work ↔ rest conflict
                if is_work != saved_is_rest:
                    # same type — check task conflict (existing logic)
                    if not is_work and not saved_is_rest:
                        continue
                    # different types — not a rest/work conflict
                    if is_work == saved_is_rest:
                        continue

                # Detect: work↔rest OR same-person-same-slot task conflicts
                should_check = False
                if is_work and saved_is_rest:
                    should_check = True
                elif (not is_work) and (not saved_is_rest):
                    should_check = True  # rest vs work
                elif is_work and not saved_is_rest:
                    # both work — check task conflict
                    if not jda and not saved.get("just_do_anything", False):
                        st = saved.get("task")
                        if t and st and t != st:
                            should_check = True

                if should_check:
                    for d, hs in saved.get("hours", {}).items():
                        for h in hs:
                            if (h, d) in my_hd:
                                if is_work and saved_is_rest:
                                    conflicts.append(f"Rule #{si+1}: {p} forced REST @ {h} {d}")
                                elif not is_work and not saved_is_rest:
                                    saved_t = saved.get("task") or "work"
                                    conflicts.append(f"Rule #{si+1}: {p} forced WORK ('{saved_t}') @ {h} {d}")
                                else:
                                    st = saved.get("task")
                                    conflicts.append(f"Rule #{si+1}: {p} → '{st}' @ {h} {d}")

            # Check against other panels in the dialog
            for oi, op in enumerate(all_panels):
                if oi == pidx or op["person"][0] != p:
                    continue
                op_is_rest = op["just_rest"][0]
                should_check = False
                if is_work and op_is_rest:
                    should_check = True
                elif not is_work and not op_is_rest:
                    should_check = True
                elif is_work and not op_is_rest:
                    if not jda and not op["just_do_anything"][0]:
                        ot = op["task"][0]
                        if t and ot and t != ot:
                            should_check = True
                if should_check:
                    for d, hs in op["sel_hours"].items():
                        for h in hs:
                            if (h, d) in my_hd:
                                kind = "REST" if op_is_rest else f"WORK ('{op['task'][0]}')"
                                conflicts.append(f"Panel #{oi+1}: {p} forced {kind} @ {h} {d}")

            return conflicts

        def _make_panel_state(prefill=None):
            ps = {
                "person":           [None],
                "task":             [None],
                "just_do_anything": [False],
                "just_rest":        [False],
                "sel_hours":        {d: set() for d in days},
                "person_filter":    [""],
                "task_filter":      [""],
                "person_chips_col": None,
                "task_chips_col":   None,
                "hours_col_inner":  None,
                "preview_text":     None,
                "conflict_text":    None,
            }
            if prefill:
                ps["person"][0]           = prefill.get("person")
                ps["task"][0]             = prefill.get("task")
                ps["just_do_anything"][0] = prefill.get("just_do_anything", False)
                ps["just_rest"][0]        = prefill.get("just_rest", False)
                for d, hs in prefill.get("hours", {}).items():
                    if d in ps["sel_hours"]:
                        ps["sel_hours"][d] = set(hs)
            return ps

        # ── Chip factories ────────────────────────────────────────────

        def _chip_normal(label, selected, on_click, sel_bg, sel_fg,
                         idle_bg="#ECEFF1", idle_fg="#424242", multi_select=False):
            icon = (ft.Icons.CHECK_CIRCLE if multi_select
                    else ft.Icons.RADIO_BUTTON_CHECKED) if selected else (
                   ft.Icons.CHECK_BOX_OUTLINE_BLANK if multi_select
                   else ft.Icons.RADIO_BUTTON_UNCHECKED)
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

        def _chip_blocked(label):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.BLOCK, size=_s(12), color=ft.Colors.RED_400),
                    ft.Text(label, size=_s(10), color=ft.Colors.RED_400, italic=True),
                ], spacing=_s(4), tight=True),
                bgcolor="#FFEBEE", border_radius=16,
                border=ft.border.all(1.5, ft.Colors.RED_300),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                tooltip="This person lacks the required skill")

        def _chip_muted(label):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CIRCLE_OUTLINED, size=_s(14), color=ft.Colors.GREY_400),
                    ft.Text(label, size=_s(11), color=ft.Colors.GREY_400),
                ], spacing=_s(4), tight=True),
                bgcolor="#F5F5F5", border_radius=16,
                border=ft.border.all(1, ft.Colors.GREY_300),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                tooltip="Disabled while 'Just do anything' or 'Just rest' is active")

        def _chip_unavail(label):
            return ft.Container(
                content=ft.Text(label, size=_s(10), color=ft.Colors.RED_400, italic=True),
                border=ft.border.all(1.5, ft.Colors.RED_300),
                border_radius=8, bgcolor="#FFEBEE",
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(3)),
                tooltip="Person unavailable at this hour")

        def _chip_no_demand(label):
            return ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.MONEY_OFF, size=_s(11), color=ft.Colors.RED_400),
                    ft.Text(label, size=_s(10), color=ft.Colors.RED_400, italic=True),
                ], spacing=_s(3), tight=True),
                border=ft.border.all(1.5, ft.Colors.RED_300),
                border_radius=8, bgcolor="#FFEBEE",
                padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(3)),
                tooltip="No demand for the selected task at this hour")

        def _make_search_field(hint, filter_ref, rebuild_fn, accent):
            def _on_change(e):
                filter_ref[0] = e.control.value or ""
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

        # ══════════════════════════════════════════════════════════════
        # Per-panel rebuild functions
        # ══════════════════════════════════════════════════════════════

        def _rebuild_preview(ps, pidx, all_panels):
            p     = ps["person"][0]
            t     = ps["task"][0]
            jda   = ps["just_do_anything"][0]
            jrest = ps["just_rest"][0]
            hrs   = ps["sel_hours"]
            if not p:
                ps["preview_text"].value = "Select a person to begin."
                ps["preview_text"].color = ft.Colors.GREY_500
            else:
                if jrest:
                    task_str = "FORCED REST"
                elif jda:
                    task_str = "any available task"
                else:
                    task_str = f"'{t}'" if t else "…"
                hour_parts = [f"{d}: {', '.join(sorted(hrs[d]))}"
                              for d in days if hrs[d]]
                if hour_parts:
                    ps["preview_text"].value = (
                        f"{p} → {task_str} — {'; '.join(hour_parts)}")
                    ps["preview_text"].color = (
                        "#B71C1C" if jrest else self._BTN_COLOR)
                else:
                    ps["preview_text"].value = f"{p} selected — choose mode and hours."
                    ps["preview_text"].color = ft.Colors.GREY_600
            try: ps["preview_text"].update()
            except Exception: pass

            conflicts = _find_conflicts(ps, pidx, all_panels)
            if conflicts:
                ps["conflict_text"].value   = "⚠  " + " | ".join(conflicts[:4])
                ps["conflict_text"].visible = True
            else:
                ps["conflict_text"].value   = ""
                ps["conflict_text"].visible = False
            try: ps["conflict_text"].update()
            except Exception: pass

        def _rebuild_person_chips(ps, pidx, all_panels):
            query  = ps["person_filter"][0].lower().strip()
            groups = s.build_groups(people)
            sections = []
            for g_idx, (gname, members) in enumerate(groups.items()):
                visible = [p for p in members if not query or query in p.lower()]
                if not visible:
                    continue
                g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
                wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
                for p in visible:
                    is_sel = p == ps["person"][0]
                    def _toggle(e, _p=p, _ps=ps, _pidx=pidx):
                        if _ps["person"][0] == _p:
                            return
                        _ps["person"][0]           = _p
                        _ps["task"][0]             = None
                        _ps["just_do_anything"][0] = False
                        _ps["just_rest"][0]        = False
                        for _d in days:
                            _ps["sel_hours"][_d].clear()
                        _rebuild_person_chips(_ps, _pidx, panels)
                        _rebuild_task_chips(_ps, _pidx, panels)
                        _rebuild_hours_section(_ps, _pidx, panels)
                        _rebuild_preview(_ps, _pidx, panels)
                        _refresh_save_btn()
                    wrap.controls.append(
                        _chip_normal(p, is_sel, _toggle,
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
            ps["person_chips_col"].controls = sections
            try: ps["person_chips_col"].update()
            except Exception: pass

        def _rebuild_task_chips(ps, pidx, all_panels):
            p     = ps["person"][0]
            jda   = ps["just_do_anything"][0]
            jrest = ps["just_rest"][0]
            query = ps["task_filter"][0].lower().strip()
            controls = []

            # ── Just do anything toggle ───────────────────────────────
            def _toggle_jda(e, _ps=ps, _pidx=pidx):
                _ps["just_do_anything"][0] = not _ps["just_do_anything"][0]
                if _ps["just_do_anything"][0]:
                    _ps["task"][0]      = None
                    _ps["just_rest"][0] = False
                _rebuild_task_chips(_ps, _pidx, panels)
                _rebuild_hours_section(_ps, _pidx, panels)
                _rebuild_preview(_ps, _pidx, panels)
                _refresh_save_btn()

            controls.append(ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.SHUFFLE if jda else ft.Icons.SHUFFLE_OUTLINED,
                            size=_s(14),
                            color=ft.Colors.WHITE if jda else "#004D40"),
                    ft.Text("Just do anything", size=_s(11),
                            color=ft.Colors.WHITE if jda else "#004D40",
                            weight=ft.FontWeight.BOLD if jda else None),
                ], spacing=_s(4), tight=True),
                bgcolor="#00796B" if jda else "#E0F2F1",
                border_radius=16, border=ft.border.all(1.5, "#00796B"),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=_toggle_jda, ink=True))

            # ── Just rest toggle ──────────────────────────────────────
            def _toggle_jrest(e, _ps=ps, _pidx=pidx):
                _ps["just_rest"][0] = not _ps["just_rest"][0]
                if _ps["just_rest"][0]:
                    _ps["task"][0]             = None
                    _ps["just_do_anything"][0] = False
                _rebuild_task_chips(_ps, _pidx, panels)
                _rebuild_hours_section(_ps, _pidx, panels)
                _rebuild_preview(_ps, _pidx, panels)
                _refresh_save_btn()

            controls.append(ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.BEDTIME if jrest else ft.Icons.BEDTIME_OUTLINED,
                            size=_s(14),
                            color=ft.Colors.WHITE if jrest else "#B71C1C"),
                    ft.Text("Just rest", size=_s(11),
                            color=ft.Colors.WHITE if jrest else "#B71C1C",
                            weight=ft.FontWeight.BOLD if jrest else None),
                ], spacing=_s(4), tight=True),
                bgcolor="#C62828" if jrest else "#FFEBEE",
                border_radius=16, border=ft.border.all(1.5, "#C62828"),
                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                on_click=_toggle_jrest, ink=True))

            controls.append(ft.Divider(height=_s(6)))

            visible_tasks = [t for t in tasks if not query or query in t.lower()]
            if not visible_tasks:
                controls.append(ft.Text("No tasks match.", size=_s(11),
                                        italic=True, color=ft.Colors.GREY_500))
            else:
                wrap = ft.Row(wrap=True, spacing=_s(4), run_spacing=_s(4))
                for t in visible_tasks:
                    no_skill = p and not _has_skill(p, t)
                    is_sel   = t == ps["task"][0]
                    if no_skill:
                        wrap.controls.append(_chip_blocked(t))
                    elif jda or jrest:
                        wrap.controls.append(_chip_muted(t))
                    else:
                        def _toggle_t(e, _t=t, _ps=ps, _pidx=pidx):
                            _ps["task"][0] = (None if _ps["task"][0] == _t else _t)
                            if _ps["task"][0]:
                                for _d in days:
                                    _ps["sel_hours"][_d] = {
                                        _h for _h in _ps["sel_hours"][_d]
                                        if _has_demand(_ps["task"][0], _h, _d)}
                            _rebuild_task_chips(_ps, _pidx, panels)
                            _rebuild_hours_section(_ps, _pidx, panels)
                            _rebuild_preview(_ps, _pidx, panels)
                            _refresh_save_btn()
                        wrap.controls.append(
                            _chip_normal(t, is_sel, _toggle_t,
                                         sel_bg="#C8E6C9", sel_fg="#1B5E20"))
                controls.append(wrap)

            ps["task_chips_col"].controls = controls
            try: ps["task_chips_col"].update()
            except Exception: pass

        def _rebuild_hours_section(ps, pidx, all_panels):
            p     = ps["person"][0]
            t     = ps["task"][0]
            jda   = ps["just_do_anything"][0]
            jrest = ps["just_rest"][0]

            # Only filter by demand when we have a specific task (not jda, not jrest)
            if t and not jda and not jrest:
                for _d in days:
                    ps["sel_hours"][_d] = {
                        _h for _h in ps["sel_hours"][_d]
                        if _has_demand(t, _h, _d)}

            controls = []
            for d in days:
                wrap = ft.Row(wrap=True, spacing=_s(3), run_spacing=_s(3))
                for h in hours_map[d]:
                    unavail   = p and not _is_avail(p, h, d)
                    # For just_rest and jda, no demand check
                    no_demand = (not jda and not jrest) and bool(t) and not _has_demand(t, h, d)
                    is_sel    = h in ps["sel_hours"][d]
                    if unavail:
                        wrap.controls.append(_chip_unavail(h))
                    elif no_demand:
                        wrap.controls.append(_chip_no_demand(h))
                    else:
                        def _toggle_h(e, _d=d, _h=h, _ps=ps, _pidx=pidx):
                            if _h in _ps["sel_hours"][_d]:
                                _ps["sel_hours"][_d].discard(_h)
                            else:
                                _ps["sel_hours"][_d].add(_h)
                            _rebuild_hours_section(_ps, _pidx, panels)
                            _rebuild_preview(_ps, _pidx, panels)
                            _refresh_save_btn()
                        wrap.controls.append(
                            _chip_normal(h, is_sel, _toggle_h,
                                         sel_bg="#BBDEFB", sel_fg="#0D47A1",
                                         idle_bg="#F5F5F5", multi_select=True))

                def _sel_all_h(ev, _d=d, _ps=ps, _pidx=pidx):
                    _t     = _ps["task"][0]
                    _jda   = _ps["just_do_anything"][0]
                    _jrest = _ps["just_rest"][0]
                    for _h in hours_map[_d]:
                        avail_ok  = not _ps["person"][0] or _is_avail(_ps["person"][0], _h, _d)
                        demand_ok = _jda or _jrest or not _t or _has_demand(_t, _h, _d)
                        if avail_ok and demand_ok:
                            _ps["sel_hours"][_d].add(_h)
                    _rebuild_hours_section(_ps, _pidx, panels)
                    _rebuild_preview(_ps, _pidx, panels)
                    _refresh_save_btn()

                def _clr_h(ev, _d=d, _ps=ps, _pidx=pidx):
                    _ps["sel_hours"][_d].clear()
                    _rebuild_hours_section(_ps, _pidx, panels)
                    _rebuild_preview(_ps, _pidx, panels)
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

        # ══════════════════════════════════════════════════════════════
        # Panel widget builder
        # ══════════════════════════════════════════════════════════════

        def _build_panel_widget(ps, pidx, removable=False):
            ps["person_chips_col"] = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
            ps["task_chips_col"]   = ft.Column(spacing=_s(4), scroll=ft.ScrollMode.AUTO)
            ps["hours_col_inner"]  = ft.Column(spacing=_s(6), scroll=ft.ScrollMode.AUTO)
            ps["preview_text"]     = ft.Text("", size=_s(11), italic=True, color=ft.Colors.GREY_500)
            ps["conflict_text"]    = ft.Text("", size=_s(10), color=ft.Colors.RED_700, visible=False)

            p_search = _make_search_field(
                "Search people…", ps["person_filter"],
                lambda _ps=ps, _pi=pidx: _rebuild_person_chips(_ps, _pi, panels),
                "#6A1B9A")
            t_search = _make_search_field(
                "Search tasks…", ps["task_filter"],
                lambda _ps=ps, _pi=pidx: _rebuild_task_chips(_ps, _pi, panels),
                "#1B5E20")

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
                ft.Column([ps["preview_text"], ps["conflict_text"]],
                          spacing=_s(2), tight=True),
                bgcolor="#E3F2FD", border_radius=8,
                border=ft.border.all(1, "#90CAF9"),
                padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)))

            def _col_section(icon, title, color, search_tf, chips_col):
                return ft.Column([
                    ft.Row([
                        ft.Icon(icon, size=_s(15), color=color),
                        ft.Text(title, size=_s(12), weight=ft.FontWeight.BOLD, color=color),
                    ], spacing=_s(5)),
                    search_tf,
                    ft.Container(chips_col,
                                 border=ft.border.all(1, ft.Colors.GREY_300),
                                 border_radius=8, padding=ft.padding.all(_s(8)),
                                 height=_s(280), expand=True),
                ], spacing=_s(5), tight=True, expand=True)

            hours_section = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCHEDULE, size=_s(15), color="#1565C0"),
                    ft.Text("Hours", size=_s(12), weight=ft.FontWeight.BOLD, color="#1565C0"),
                    ft.Text("(red = unavailable or no demand)",
                            size=_s(9), italic=True, color=ft.Colors.GREY_500),
                ], spacing=_s(5)),
                ft.Container(ps["hours_col_inner"],
                             border=ft.border.all(1, ft.Colors.GREY_300),
                             border_radius=8, padding=ft.padding.all(_s(8)),
                             height=_s(280), expand=True),
            ], spacing=_s(5), tight=True, expand=True)

            return ft.Container(
                content=ft.Column([
                    header, preview_bar,
                    ft.Row([
                        _col_section(ft.Icons.PERSON, "Person", "#6A1B9A",
                                     p_search, ps["person_chips_col"]),
                        ft.VerticalDivider(width=_s(4)),
                        _col_section(ft.Icons.ASSIGNMENT, "Task / Mode", "#1B5E20",
                                     t_search, ps["task_chips_col"]),
                        ft.VerticalDivider(width=_s(4)),
                        hours_section,
                    ], spacing=_s(8), vertical_alignment=ft.CrossAxisAlignment.START, expand=True),
                ], spacing=_s(8), tight=True),
                border=ft.border.all(1, self._CARD_BORDER),
                border_radius=8, padding=ft.padding.all(_s(10)),
                bgcolor="#F8FBFF")

        # ══════════════════════════════════════════════════════════════
        # Panels list + save logic
        # ══════════════════════════════════════════════════════════════

        panels = []
        if is_edit:
            panels.append(_make_panel_state(prefill=s.mandatory_rules[edit_index]))
        else:
            panels.append(_make_panel_state())

        panels_col = ft.Column(spacing=_s(12), scroll=ft.ScrollMode.AUTO)

        save_btn = ft.ElevatedButton(
            "Update rule" if is_edit else "Save rules",
            icon=ft.Icons.SAVE, disabled=True,
            bgcolor=self._BTN_COLOR, color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(10)),
                shape=ft.RoundedRectangleBorder(radius=6)))

        def _can_save():
            for ps in panels:
                p     = ps["person"][0]
                t     = ps["task"][0]
                jda   = ps["just_do_anything"][0]
                jrest = ps["just_rest"][0]
                any_h = any(ps["sel_hours"][d] for d in days)
                if not p or not any_h:
                    return False
                if not jda and not jrest and not t:
                    return False
            return True

        def _refresh_save_btn():
            save_btn.disabled = not _can_save()
            try: save_btn.update()
            except Exception: pass

        def _rebuild_all_panels():
            removable = len(panels) > 1
            widgets = []
            for pidx, ps in enumerate(panels):
                w = _build_panel_widget(ps, pidx, removable=removable)
                widgets.append(w)
                _rebuild_person_chips(ps, pidx, panels)
                _rebuild_task_chips(ps, pidx, panels)
                _rebuild_hours_section(ps, pidx, panels)
                _rebuild_preview(ps, pidx, panels)
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
            if not _can_save():
                return
            new_rules = [
                {
                    "person":           ps["person"][0],
                    "task":             ps["task"][0],
                    "just_do_anything": ps["just_do_anything"][0],
                    "just_rest":        ps["just_rest"][0],
                    "hours": {d: sorted(ps["sel_hours"][d])
                              for d in days if ps["sel_hours"][d]},
                }
                for ps in panels
            ]
            if is_edit:
                s.mandatory_rules[edit_index] = new_rules[0]
            else:
                s.mandatory_rules.extend(new_rules)
            self._page.close(dlg)
            self.validate_and_refresh()

        def _on_cancel(ev):
            self._page.close(dlg)

        save_btn.on_click = _on_save
        _rebuild_all_panels()
        _refresh_save_btn()

        content = ft.Column([
            ft.Text(
                "Force a person to work on a specific task, any available task, "
                "or force them to rest at chosen hours. "
                "Tasks in red = missing skill. "
                "Hours in red = person unavailable or task has no demand.",
                size=_s(11), italic=True, color=ft.Colors.GREY_600),
            ft.Divider(height=_s(2)),
            ft.Column([panels_col, and_btn],
                      spacing=_s(10), tight=True, expand=True,
                      scroll=ft.ScrollMode.AUTO),
        ], spacing=_s(8), expand=True, width=_s(1200), height=_s(640))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.ASSIGNMENT_IND, size=_s(20), color=self._BTN_COLOR),
                ft.Text("Edit Mandatory Rule" if is_edit else "Add Mandatory Rules",
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