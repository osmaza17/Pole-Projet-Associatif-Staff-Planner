"""
max_consec_hours.py  ←  tabs/dimensions/max_consec_hours.py

Manager for the "Rest & Workload Conditions" dialog.

Five per-person fields, each with global default + individual override:
  1. Max consecutive hours   (positive int or empty)
  2. Min rest hours          (positive int or empty)
  3. Capacity %              (0–100 integer)
  4. Max hours / day         (positive int or empty)  ← NEW
  5. Max hours / event       (positive int or empty)  ← NEW
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS
from ui_helpers import UIHelpers


class MaxConsecHoursManager:

    _BTN_COLOR       = "#455A64"
    _CARD_BG         = "#ECEFF1"
    _CARD_BORDER     = "#B0BEC5"
    _CAPTAIN_BORDER  = "#F9A825"
    _CAPTAIN_BG      = "#FFFDE7"

    # ── field specs: (state_global_attr, state_per_person_attr,
    #                  validation_suffix, label, hint, width, validator) ──
    _FIELDS = [
        ("consec_global_val",       "consec_per_person",           "",       "Max consec",   "e.g. 8",  58, "pos_int"),
        ("consec_global_rest",      "consec_rest_per_person",      "_rest",  "Min rest",     "e.g. 1",  58, "pos_int"),
        ("consec_global_capacity",  "consec_capacity_per_person",  "_cap",   "Capacity %",   "0–100",   58, "cap"),
        ("consec_global_max_day",   "consec_max_day_per_person",   "_mday",  "Max h/day",    "e.g. 6",  58, "pos_int"),
        ("consec_global_max_event", "consec_max_event_per_person", "_mevt",  "Max h/event",  "e.g. 20", 58, "pos_int"),
    ]

    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self._state                   = state
        self._page                    = page
        self._on_solve_blocked_update = on_solve_blocked_update

    def _get_captain_persons(self) -> set:
        captains = set()
        for rule in self._state.captain_rules:
            for c in rule.get("captains", []):
                captains.add(c)
        return captains

    @staticmethod
    def _validate_capacity(val: str) -> bool:
        v = val.strip()
        if v == "":
            return True
        try:
            n = int(v)
            return 0 <= n <= 100
        except ValueError:
            return False

    @staticmethod
    def _validate_pos_int(val: str) -> bool:
        return UIHelpers.validate_positive_int(val) or val.strip() == ""

    def _validator_for(self, kind: str):
        if kind == "cap":
            return self._validate_capacity
        return self._validate_pos_int

    def _err_msg_for(self, kind: str) -> str:
        if kind == "cap":
            return "Integer 0–100"
        return "Positive integer"

    # ══════════════════════════════════════════════════════════════════

    def open_dialog(self, e):
        s      = self._state
        people, _, _, _ = s.dims()
        groups = s.build_groups(people)
        captain_persons = self._get_captain_persons()

        BTN_COLOR   = self._BTN_COLOR
        CARD_BG     = self._CARD_BG
        CARD_BORDER = self._CARD_BORDER

        search_filter = [""]
        person_widgets: dict = {}   # p → {field_suffix → tf, "card": container}

        # ══════════════════════════════════════════════════════════════
        # GLOBAL FIELDS
        # ══════════════════════════════════════════════════════════════
        global_tfs  = {}   # suffix → TextField
        global_errs = {}   # suffix → Text

        for g_attr, _, suffix, label, hint, _, vkind in self._FIELDS:
            err = ft.Text("", color=ft.Colors.RED_400, size=_s(11), visible=False)
            global_errs[suffix] = err

            tf = ft.TextField(
                value=getattr(s, g_attr, "") or "",
                width=_s(160), height=_s(40), text_size=_s(12),
                hint_text=hint,
                label=f"Global {label.lower()}",
                label_style=ft.TextStyle(size=_s(10)),
                content_padding=ft.padding.all(_s(8)))
            global_tfs[suffix] = tf

            validator = self._validator_for(vkind)
            err_msg   = self._err_msg_for(vkind)
            err_key   = f"_global{suffix}" if suffix else "_global"

            def _make_global_handler(_attr=g_attr, _suffix=suffix,
                                     _validator=validator, _err_msg=err_msg,
                                     _err_key=err_key, _err=err):
                def handler(ev):
                    val = ev.control.value
                    setattr(s, _attr, val)
                    ok = _validator(val)
                    if ok:
                        s.validation_errors["consec"].discard(_err_key)
                        ev.control.border_color = None
                        _err.value = ""; _err.visible = False
                    else:
                        s.validation_errors["consec"].add(_err_key)
                        ev.control.border_color = ft.Colors.RED_400
                        _err.value = _err_msg; _err.visible = True
                    self._on_solve_blocked_update()
                    ev.control.update()
                    try: _err.update()
                    except Exception: pass
                    if ok:
                        _propagate_global_to_cards()
                return handler

            tf.on_change = _make_global_handler()

        def _propagate_global_to_cards():
            for p, wdg in person_widgets.items():
                if p not in s.consec_personalized_persons:
                    for g_attr, pp_attr, suffix, _, _, _, _ in self._FIELDS:
                        gval = getattr(s, g_attr, "")
                        getattr(s, pp_attr)[p] = gval
                        wdg[suffix].value = gval
                        wdg[suffix].border_color = None
                        err_key = f"{p}{suffix}" if suffix else p
                        s.validation_errors["consec"].discard(err_key)
                        try: wdg[suffix].update()
                        except Exception: pass

        # ══════════════════════════════════════════════════════════════
        # PER-PERSON GRID
        # ══════════════════════════════════════════════════════════════
        grid_container = ft.Column(spacing=_s(12), tight=True)

        def _build_person_card(p: str, group_color: str) -> ft.Container:
            is_personalized = p in s.consec_personalized_persons
            is_captain      = p in captain_persons

            card_field_tfs = {}

            field_cols = []
            for g_attr, pp_attr, suffix, label, hint, w, vkind in self._FIELDS:
                pp_dict = getattr(s, pp_attr)
                if is_personalized:
                    init_val = pp_dict.get(p, "")
                else:
                    init_val = getattr(s, g_attr, "")
                    pp_dict[p] = init_val

                tf_p = ft.TextField(
                    value=init_val,
                    width=_s(w), height=_s(34), text_size=_s(11),
                    hint_text=hint,
                    content_padding=ft.padding.all(_s(4)),
                    read_only=not is_personalized,
                    color=ft.Colors.BLACK if is_personalized else ft.Colors.GREY_600,
                    data=p)
                card_field_tfs[suffix] = tf_p

                validator = self._validator_for(vkind)
                err_key   = f"{p}{suffix}" if suffix else p

                def _make_ch(_p=p, _pp_attr=pp_attr, _suffix=suffix,
                             _validator=validator, _err_key=err_key):
                    def handler(ev):
                        v = ev.control.value
                        getattr(s, _pp_attr)[_p] = v
                        ok = _validator(v)
                        if ok:
                            s.validation_errors["consec"].discard(_err_key)
                            ev.control.border_color = None
                        else:
                            s.validation_errors["consec"].add(_err_key)
                            ev.control.border_color = ft.Colors.RED_400
                        self._on_solve_blocked_update()
                        try: ev.control.update()
                        except Exception: pass
                    return handler

                tf_p.on_change = _make_ch()

                field_cols.append(
                    ft.Column([
                        ft.Text(label, size=_s(8), color=ft.Colors.GREY_600),
                        tf_p,
                    ], spacing=_s(1), tight=True))

            # ── Personalize / Reset ───────────────────────────────────
            def _toggle(ev, _p=p):
                if _p in s.consec_personalized_persons:
                    s.consec_personalized_persons.discard(_p)
                    for g_attr, pp_attr, suffix, _, _, _, _ in self._FIELDS:
                        getattr(s, pp_attr)[_p] = getattr(s, g_attr, "")
                        err_key = f"{_p}{suffix}" if suffix else _p
                        s.validation_errors["consec"].discard(err_key)
                    self._on_solve_blocked_update()
                else:
                    s.consec_personalized_persons.add(_p)
                _rebuild_grid()

            btn_label = "Reset" if is_personalized else "Personalize"
            btn_icon  = ft.Icons.RESTART_ALT if is_personalized else ft.Icons.EDIT
            btn_color = ft.Colors.RED_700 if is_personalized else "#1565C0"

            personalize_btn = ft.Container(
                content=ft.Row([
                    ft.Icon(btn_icon, size=_s(12), color=btn_color),
                    ft.Text(btn_label, size=_s(9), color=btn_color,
                            weight=ft.FontWeight.BOLD),
                ], spacing=_s(2), tight=True),
                border=ft.border.all(1, btn_color), border_radius=12,
                padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(3)),
                on_click=_toggle, ink=True)

            # ── Name row ──────────────────────────────────────────────
            name_ctrls = [
                ft.Text(p, size=_s(11), weight=ft.FontWeight.BOLD,
                        color=group_color, overflow=ft.TextOverflow.ELLIPSIS,
                        max_lines=1, tooltip=p, expand=True)]
            if is_personalized:
                name_ctrls.append(ft.Container(
                    content=ft.Text("custom", size=_s(8), color="#1565C0",
                                    weight=ft.FontWeight.BOLD, italic=True),
                    bgcolor="#E3F2FD", border_radius=8,
                    padding=ft.padding.symmetric(horizontal=_s(5), vertical=_s(1))))

            captain_warning = None
            if is_captain:
                captain_warning = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.STAR, size=_s(10), color="#F57F17"),
                        ft.Text("Captain — rest may be overridden",
                                size=_s(8), italic=True, color="#F57F17"),
                    ], spacing=_s(3), tight=True),
                    padding=ft.padding.only(top=_s(2)))

            card_controls = [ft.Row(name_ctrls, spacing=_s(4))]
            if captain_warning:
                card_controls.append(captain_warning)
            card_controls.append(ft.Row(field_cols, spacing=_s(4), wrap=True))
            card_controls.append(personalize_btn)

            if is_captain:
                card_border = ft.border.all(2, self._CAPTAIN_BORDER)
                card_bg     = self._CAPTAIN_BG
            else:
                card_border = ft.border.all(1, CARD_BORDER)
                card_bg     = CARD_BG

            card = ft.Container(
                content=ft.Column(card_controls, spacing=_s(4), tight=True),
                width=_s(340), bgcolor=card_bg,
                border=card_border, border_radius=8,
                padding=ft.padding.all(_s(8)))

            person_widgets[p] = {**card_field_tfs, "card": card}
            return card

        def _rebuild_grid():
            query = search_filter[0].lower().strip()
            person_widgets.clear()
            sections = []
            for g_idx, (gname, members) in enumerate(groups.items()):
                g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
                visible = [p for p in members if not query or query in p.lower()]
                if not visible:
                    continue
                sections.append(ft.Container(
                    content=ft.Text(gname, size=_s(11), weight=ft.FontWeight.BOLD,
                                    color=g_color),
                    padding=ft.padding.only(top=_s(4), bottom=_s(2))))
                sections.append(ft.Row(
                    [_build_person_card(p, g_color) for p in visible],
                    spacing=_s(8), run_spacing=_s(8), wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.START))
            if not sections:
                sections = [ft.Text("No people match the search.",
                                    size=_s(11), italic=True, color=ft.Colors.GREY_500)]
            grid_container.controls = sections
            try: grid_container.update()
            except Exception: pass
            _refresh_bulk_btn()

        # ── Bulk button ───────────────────────────────────────────────
        bulk_btn = ft.Container(content=ft.Row([], spacing=_s(4), tight=True),
                                border_radius=6,
                                padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
                                ink=True)

        def _refresh_bulk_btn():
            all_pers = bool(people) and all(p in s.consec_personalized_persons for p in people)
            icon  = ft.Icons.RESTART_ALT if all_pers else ft.Icons.EDIT
            label = "Reset All" if all_pers else "Personalize All"
            color = ft.Colors.RED_700 if all_pers else "#1565C0"
            bulk_btn.content = ft.Row([
                ft.Icon(icon, size=_s(13), color=color),
                ft.Text(label, size=_s(10), color=color, weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True)
            bulk_btn.border = ft.border.all(1, color)
            try: bulk_btn.update()
            except Exception: pass

        def _on_bulk_click(ev):
            all_pers = bool(people) and all(p in s.consec_personalized_persons for p in people)
            if all_pers:
                for p in people:
                    s.consec_personalized_persons.discard(p)
                    for g_attr, pp_attr, suffix, _, _, _, _ in self._FIELDS:
                        getattr(s, pp_attr)[p] = getattr(s, g_attr, "")
                        err_key = f"{p}{suffix}" if suffix else p
                        s.validation_errors["consec"].discard(err_key)
                self._on_solve_blocked_update()
            else:
                for p in people:
                    s.consec_personalized_persons.add(p)
            _rebuild_grid()

        bulk_btn.on_click = _on_bulk_click
        _refresh_bulk_btn()

        # ── Search ────────────────────────────────────────────────────
        def _on_search_ch(ev):
            search_filter[0] = ev.control.value or ""
            _rebuild_grid()

        def _clear_search(ev):
            search_filter[0] = ""
            search_tf.value = ""
            try: search_tf.update()
            except Exception: pass
            _rebuild_grid()

        search_tf = ft.TextField(
            hint_text="Search people…", prefix_icon=ft.Icons.SEARCH,
            on_change=_on_search_ch, text_size=_s(11), height=_s(36),
            content_padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(6)),
            border_color=ft.Colors.GREY_400, focused_border_color=BTN_COLOR,
            border_radius=8, bgcolor=ft.Colors.WHITE,
            suffix=ft.IconButton(icon=ft.Icons.CLEAR, icon_size=_s(14),
                                 icon_color=ft.Colors.GREY_400, tooltip="Clear search",
                                 style=ft.ButtonStyle(padding=ft.padding.all(0)),
                                 on_click=_clear_search))

        _rebuild_grid()

        # ══════════════════════════════════════════════════════════════
        # DIALOG
        # ══════════════════════════════════════════════════════════════
        def _on_close(ev):
            self._page.close(dlg)

        global_row = ft.Container(
            content=ft.Column([
                ft.Row(
                    [ft.Column([global_tfs[sfx], global_errs[sfx]],
                               spacing=_s(2), tight=True)
                     for _, _, sfx, _, _, _, _ in self._FIELDS],
                    spacing=_s(10),
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    wrap=True),
                ft.Text(
                    "Global values auto-update all non-personalized people. "
                    "Capacity % controls workload share (100=full). "
                    "Max h/day and Max h/event are hard limits.",
                    size=_s(9), italic=True, color=ft.Colors.GREY_500),
            ], spacing=_s(6), tight=True),
            bgcolor="#ECEFF1", border_radius=8,
            border=ft.border.all(1, "#B0BEC5"),
            padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(10)))

        grid_section = ft.Column([
            ft.Row([search_tf, bulk_btn], spacing=_s(8),
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(
                content=ft.Column([grid_container], scroll=ft.ScrollMode.AUTO, expand=True),
                border=ft.border.all(1, ft.Colors.GREY_300), border_radius=8,
                padding=ft.padding.all(_s(10)), height=_s(420), expand=True),
        ], spacing=_s(8), tight=True, expand=True)

        content = ft.Column([
            ft.Text(
                "Configure rest, capacity, and workload limits per person.\n"
                "Set global defaults or click \"Personalize\" for individual overrides.",
                size=_s(11), italic=True, color=ft.Colors.GREY_600),
            ft.Divider(height=_s(4)),
            global_row,
            ft.Divider(height=_s(4)),
            grid_section,
        ], spacing=_s(8), expand=True, width=_s(1000), height=_s(620))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.BEDTIME_OUTLINED, size=_s(20), color=BTN_COLOR),
                ft.Text("Rest & Workload Conditions",
                        weight=ft.FontWeight.BOLD, size=_s(16)),
            ], spacing=_s(8)),
            content=content,
            actions=[ft.ElevatedButton(
                "Close", icon=ft.Icons.CHECK,
                bgcolor=BTN_COLOR, color=ft.Colors.WHITE,
                on_click=_on_close,
                style=ft.ButtonStyle(
                    padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(10)),
                    shape=ft.RoundedRectangleBorder(radius=6)))],
            actions_alignment=ft.MainAxisAlignment.END)
        self._page.open(dlg)