"""
max_consec_hours.py — "Rest & Workload Conditions" dialog.

Delegates field specs to rest_fields and card building to person_card.
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS
from .rest_fields import (FIELDS, validator_for, err_msg_for,
                          reset_person_to_globals, safe_update)
from .person_card import build_person_card


# ── Visual constants ──────────────────────────────────────────────
_BTN_COLOR      = "#455A64"
_SECTION_BG     = "#ECEFF1"
_SECTION_BORDER = "#B0BEC5"
_ACCENT         = "#1565C0"

# ── Descriptive tooltips for each field ───────────────────────────
_FIELD_DESCRIPTIONS = {
    "":      "Maximum consecutive working hours before a mandatory rest break.",
    "_rest": "Minimum rest hours required after reaching the consecutive limit.",
    "_cap":  "Workload capacity as a percentage (0–100). 100 = full availability.",
    "_mday": "Hard upper limit on total working hours in a single day.",
    "_mevt": "Hard upper limit on total working hours across the entire event.",
}


class MaxConsecHoursManager:

    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self._state = state
        self._page = page
        self._on_solve_blocked_update = on_solve_blocked_update

    def _get_captain_persons(self) -> set:
        captains = set()
        for rule in self._state.captain_rules:
            for c in rule.get("captains", []):
                captains.add(c)
        return captains

    # ══════════════════════════════════════════════════════════════════

    def open_dialog(self, e):
        s = self._state
        people, _, _, _ = s.dims()
        groups = s.build_groups(people)
        captain_persons = self._get_captain_persons()

        search_filter = [""]
        person_widgets: dict = {}  # person → {suffix → tf, "card": container}

        # ── Global fields ─────────────────────────────────────────
        global_tfs: dict = {}
        global_errs: dict = {}
        global_field_cols: list = []

        for g_attr, _, suffix, label, hint, _, vkind in FIELDS:
            err = ft.Text("", color=ft.Colors.RED_400, size=_s(11),
                          visible=False)
            global_errs[suffix] = err

            tf = ft.TextField(
                value=getattr(s, g_attr, "") or "",
                width=_s(200), height=_s(44), text_size=_s(13),
                hint_text=hint,
                label=label,
                label_style=ft.TextStyle(
                    size=_s(11), weight=ft.FontWeight.BOLD),
                content_padding=ft.padding.all(_s(8)),
                border_radius=6,
                border_color=_SECTION_BORDER,
                focused_border_color=_ACCENT)
            global_tfs[suffix] = tf

            _validator = validator_for(vkind)
            _err_msg = err_msg_for(vkind)
            _err_key = f"_global{suffix}" if suffix else "_global"

            def _make_handler(_attr=g_attr, _v=_validator, _em=_err_msg,
                              _ek=_err_key, _err=err):
                def handler(ev):
                    val = ev.control.value
                    setattr(s, _attr, val)
                    ok = _v(val)
                    if ok:
                        s.validation_errors["consec"].discard(_ek)
                        ev.control.border_color = _SECTION_BORDER
                        _err.value = ""; _err.visible = False
                    else:
                        s.validation_errors["consec"].add(_ek)
                        ev.control.border_color = ft.Colors.RED_400
                        _err.value = _em; _err.visible = True
                    self._on_solve_blocked_update()
                    safe_update(ev.control, _err)
                    if ok:
                        _propagate_global()
                return handler

            tf.on_change = _make_handler()

            desc = _FIELD_DESCRIPTIONS.get(suffix, "")
            global_field_cols.append(
                ft.Column([
                    tf,
                    err,
                    ft.Text(desc, size=_s(9), italic=True,
                            color=ft.Colors.GREY_600,
                            width=_s(200)),
                ], spacing=_s(2), tight=True))

        def _propagate_global():
            for p, wdg in person_widgets.items():
                if p not in s.consec_personalized_persons:
                    reset_person_to_globals(s, p)
                    # Sync widget values from state
                    for g_attr, pp_attr, suffix, _, _, _, _ in FIELDS:
                        wdg[suffix].value = getattr(s, g_attr, "")
                        wdg[suffix].border_color = None
                        safe_update(wdg[suffix])

        # ── Person grid ───────────────────────────────────────────
        grid_container = ft.Column(spacing=_s(14), tight=True)

        def _rebuild_grid():
            query = search_filter[0].lower().strip()
            person_widgets.clear()
            sections = []
            for g_idx, (gname, members) in enumerate(groups.items()):
                g_color = GROUP_HEADER_COLORS[
                    g_idx % len(GROUP_HEADER_COLORS)]
                visible = [p for p in members
                           if not query or query in p.lower()]
                if not visible:
                    continue
                sections.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.GROUP, size=_s(14), color=g_color),
                        ft.Text(gname, size=_s(13),
                                weight=ft.FontWeight.BOLD,
                                color=g_color),
                        ft.Text(f"({len(visible)} people)", size=_s(10),
                                color=ft.Colors.GREY_500, italic=True),
                    ], spacing=_s(6)),
                    padding=ft.padding.only(top=_s(6), bottom=_s(2))))
                cards_row = []
                for p in visible:
                    card, tfs = build_person_card(
                        p, g_color, s, captain_persons,
                        self._on_solve_blocked_update, _rebuild_grid)
                    person_widgets[p] = tfs
                    cards_row.append(card)
                sections.append(ft.Row(
                    cards_row, spacing=_s(10), run_spacing=_s(10),
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.START))
            if not sections:
                sections = [ft.Text("No people match the search.",
                                    size=_s(12), italic=True,
                                    color=ft.Colors.GREY_500)]
            grid_container.controls = sections
            safe_update(grid_container)
            _refresh_bulk()

        # ── Bulk personalize / reset ──────────────────────────────
        bulk_btn = ft.Container(
            content=ft.Row([], spacing=_s(4), tight=True),
            border_radius=6,
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(6)),
            ink=True)

        def _refresh_bulk():
            all_pers = bool(people) and all(
                p in s.consec_personalized_persons for p in people)
            icon = ft.Icons.RESTART_ALT if all_pers else ft.Icons.EDIT
            label = "Reset All to Global" if all_pers else "Personalize All"
            color = ft.Colors.RED_700 if all_pers else _ACCENT
            bulk_btn.content = ft.Row([
                ft.Icon(icon, size=_s(14), color=color),
                ft.Text(label, size=_s(11), color=color,
                        weight=ft.FontWeight.BOLD),
            ], spacing=_s(4), tight=True)
            bulk_btn.border = ft.border.all(1, color)
            safe_update(bulk_btn)

        def _on_bulk(ev):
            all_pers = bool(people) and all(
                p in s.consec_personalized_persons for p in people)
            if all_pers:
                for p in people:
                    s.consec_personalized_persons.discard(p)
                    reset_person_to_globals(s, p)
                self._on_solve_blocked_update()
            else:
                for p in people:
                    s.consec_personalized_persons.add(p)
            _rebuild_grid()

        bulk_btn.on_click = _on_bulk
        _refresh_bulk()

        # ── Search ────────────────────────────────────────────────
        def _on_search(ev):
            search_filter[0] = ev.control.value or ""
            _rebuild_grid()

        def _clear_search(ev):
            search_filter[0] = ""
            search_tf.value = ""
            safe_update(search_tf)
            _rebuild_grid()

        search_tf = ft.TextField(
            hint_text="Search people…", prefix_icon=ft.Icons.SEARCH,
            on_change=_on_search, text_size=_s(12), height=_s(40),
            width=_s(300),
            content_padding=ft.padding.symmetric(
                horizontal=_s(10), vertical=_s(8)),
            border_color=ft.Colors.GREY_400,
            focused_border_color=_BTN_COLOR,
            border_radius=8, bgcolor=ft.Colors.WHITE,
            suffix=ft.IconButton(
                icon=ft.Icons.CLEAR, icon_size=_s(14),
                icon_color=ft.Colors.GREY_400, tooltip="Clear search",
                style=ft.ButtonStyle(padding=ft.padding.all(0)),
                on_click=_clear_search))

        _rebuild_grid()

        # ── Dialog assembly ───────────────────────────────────────
        def _on_close(ev):
            self._page.close(dlg)

        # -- Global defaults section --
        global_section = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SETTINGS, size=_s(18), color=_BTN_COLOR),
                    ft.Text("Global Defaults", size=_s(14),
                            weight=ft.FontWeight.BOLD, color=_BTN_COLOR),
                    ft.Container(expand=True),
                    ft.Text(
                        "These values apply to all non-personalized people",
                        size=_s(10), italic=True, color=ft.Colors.GREY_600),
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=_s(2), color=_SECTION_BORDER),
                ft.Row(
                    global_field_cols,
                    spacing=_s(16),
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    wrap=True, run_spacing=_s(10)),
            ], spacing=_s(10), tight=True),
            bgcolor=_SECTION_BG, border_radius=10,
            border=ft.border.all(1, _SECTION_BORDER),
            padding=ft.padding.all(_s(16)))

        # -- Per-person section --
        person_section_header = ft.Row([
            ft.Icon(ft.Icons.PEOPLE, size=_s(18), color=_BTN_COLOR),
            ft.Text("Per-Person Overrides", size=_s(14),
                    weight=ft.FontWeight.BOLD, color=_BTN_COLOR),
            ft.Container(expand=True),
            search_tf,
            bulk_btn,
        ], spacing=_s(10),
            vertical_alignment=ft.CrossAxisAlignment.CENTER)

        person_grid_area = ft.Container(
            content=ft.Column([grid_container],
                              scroll=ft.ScrollMode.AUTO,
                              expand=True),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=10, padding=ft.padding.all(_s(14)),
            height=_s(560), expand=True, bgcolor=ft.Colors.WHITE)

        # -- Full content --
        content = ft.Column([
            global_section,
            ft.Divider(height=_s(6), color=ft.Colors.TRANSPARENT),
            person_section_header,
            person_grid_area,
        ], spacing=_s(8), expand=True,
            width=_s(1400), height=_s(820))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.BEDTIME_OUTLINED, size=_s(24),
                        color=_BTN_COLOR),
                ft.Text("Rest & Workload Conditions",
                        weight=ft.FontWeight.BOLD, size=_s(18)),
            ], spacing=_s(10)),
            content=content,
            actions=[ft.ElevatedButton(
                "Close", icon=ft.Icons.CHECK,
                bgcolor=_BTN_COLOR, color=ft.Colors.WHITE,
                on_click=_on_close,
                style=ft.ButtonStyle(
                    padding=ft.padding.symmetric(
                        horizontal=_s(20), vertical=_s(12)),
                    shape=ft.RoundedRectangleBorder(radius=6)))],
            actions_alignment=ft.MainAxisAlignment.END)
        self._page.open(dlg)