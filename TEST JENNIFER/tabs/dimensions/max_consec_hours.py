"""
max_consec_hours.py  ←  tabs/dimensions/max_consec_hours.py

Manager for the "Rest Conditions" dialog.

UI behaviour
────────────
• Two global TextFields (max consecutive hours, min rest hours) at the top.
• A grid of person cards, grouped by group, each showing:
    – Two read-only TextFields mirroring the global values.
    – A "Personalize" button that unlocks those TextFields for individual editing.
    – Once personalized, the button becomes "Reset" to re-sync with globals.
• Captain persons (referenced in any captain_rule) get a yellow border and a
  short warning that the solver may override their rest preferences.
• A search bar filters the grid.

State interaction
─────────────────
• consec_global_val / consec_global_rest   → global strings
• consec_per_person[p]                     → individual limit (str)
• consec_rest_per_person[p]                → individual rest  (str)
• consec_personalized_persons              → set of person names with overrides
"""

import flet as ft
from constants import _s, GROUP_HEADER_COLORS
from ui_helpers import UIHelpers


class MaxConsecHoursManager:

    _BTN_COLOR       = "#455A64"
    _CARD_BG         = "#ECEFF1"
    _CARD_BORDER     = "#B0BEC5"
    _CAPTAIN_BORDER  = "#F9A825"   # yellow for captains
    _CAPTAIN_BG      = "#FFFDE7"   # light yellow background for captains

    def __init__(self, state, page: ft.Page, on_solve_blocked_update):
        self._state                   = state
        self._page                    = page
        self._on_solve_blocked_update = on_solve_blocked_update

    # ── Helper: collect all captain person names ──────────────────────

    def _get_captain_persons(self) -> set:
        """Return the set of person names that appear as captains in any rule."""
        captains = set()
        for rule in self._state.captain_rules:
            for c in rule.get("captains", []):
                captains.add(c)
        return captains

    # ── Public entry point ────────────────────────────────────────────

    def open_dialog(self, e):
        s      = self._state
        people, _, _, _ = s.dims()
        groups = s.build_groups(people)
        captain_persons = self._get_captain_persons()

        BTN_COLOR   = self._BTN_COLOR
        CARD_BG     = self._CARD_BG
        CARD_BORDER = self._CARD_BORDER

        search_filter = [""]

        # ══════════════════════════════════════════════════════════════
        # Registry: keep references to per-person TextFields so we can
        # update them when the global values change.
        # {person: {"tf_limit": TextField, "tf_rest": TextField,
        #           "card": Container}}
        # ══════════════════════════════════════════════════════════════
        person_widgets: dict = {}

        # ══════════════════════════════════════════════════════════════
        # GLOBAL FIELDS — max consec + min rest
        # ══════════════════════════════════════════════════════════════

        err_limit = ft.Text("", color=ft.Colors.RED_400, size=_s(11),
                            visible=False)
        err_rest  = ft.Text("", color=ft.Colors.RED_400, size=_s(11),
                            visible=False)

        tf_global_limit = ft.TextField(
            value=s.consec_global_val or "",
            width=_s(180), height=_s(40), text_size=_s(12),
            hint_text="e.g. 8",
            label="Global max consecutive hours",
            label_style=ft.TextStyle(size=_s(10)),
            content_padding=ft.padding.all(_s(8)))

        tf_global_rest = ft.TextField(
            value=s.consec_global_rest or "1",
            width=_s(180), height=_s(40), text_size=_s(12),
            hint_text="e.g. 1",
            label="Global min rest hours",
            label_style=ft.TextStyle(size=_s(10)),
            content_padding=ft.padding.all(_s(8)))

        def _propagate_global_to_cards():
            """Push global values to every non-personalized person's TextFields."""
            global_limit = s.consec_global_val
            global_rest  = s.consec_global_rest
            for p, wdg in person_widgets.items():
                if p not in s.consec_personalized_persons:
                    # Update state dicts
                    s.consec_per_person[p]      = global_limit
                    s.consec_rest_per_person[p]  = global_rest
                    # Update UI
                    wdg["tf_limit"].value = global_limit
                    wdg["tf_rest"].value  = global_rest
                    # Clear any validation error for this person since we
                    # know the global was already validated
                    wdg["tf_limit"].border_color = None
                    wdg["tf_rest"].border_color  = None
                    s.validation_errors["consec"].discard(p)
                    s.validation_errors["consec"].discard(f"{p}_rest")
                    try:
                        wdg["tf_limit"].update()
                        wdg["tf_rest"].update()
                    except Exception:
                        pass

        def _on_global_limit_ch(ev):
            val = ev.control.value
            s.consec_global_val = val
            ok  = UIHelpers.validate_positive_int(val) or val.strip() == ""
            if ok:
                s.validation_errors["consec"].discard("_global")
                ev.control.border_color = None
                err_limit.value         = ""
                err_limit.visible       = False
            else:
                s.validation_errors["consec"].add("_global")
                ev.control.border_color = ft.Colors.RED_400
                err_limit.value         = "Must be a positive integer"
                err_limit.visible       = True
            self._on_solve_blocked_update()
            ev.control.update()
            try: err_limit.update()
            except Exception: pass
            # Propagate to all non-personalized cards
            if ok:
                _propagate_global_to_cards()

        def _on_global_rest_ch(ev):
            val = ev.control.value
            s.consec_global_rest = val
            ok  = UIHelpers.validate_positive_int(val) or val.strip() == ""
            if ok:
                s.validation_errors["consec"].discard("_global_rest")
                ev.control.border_color = None
                err_rest.value          = ""
                err_rest.visible        = False
            else:
                s.validation_errors["consec"].add("_global_rest")
                ev.control.border_color = ft.Colors.RED_400
                err_rest.value          = "Must be a positive integer"
                err_rest.visible        = True
            self._on_solve_blocked_update()
            ev.control.update()
            try: err_rest.update()
            except Exception: pass
            # Propagate to all non-personalized cards
            if ok:
                _propagate_global_to_cards()

        tf_global_limit.on_change = _on_global_limit_ch
        tf_global_rest.on_change  = _on_global_rest_ch

        # ══════════════════════════════════════════════════════════════
        # PER-PERSON GRID (grouped)
        # ══════════════════════════════════════════════════════════════

        grid_container = ft.Column(spacing=_s(12), tight=True)

        def _build_person_card(p: str, group_color: str) -> ft.Container:
            """Build a single person card with limit/rest fields and
            a Personalize/Reset button."""

            is_personalized = p in s.consec_personalized_persons
            is_captain      = p in captain_persons

            # Determine initial values: use individual if personalized,
            # else use current global values
            if is_personalized:
                init_limit = s.consec_per_person.get(p, "")
                init_rest  = s.consec_rest_per_person.get(p, "")
            else:
                init_limit = s.consec_global_val
                init_rest  = s.consec_global_rest
                # Sync state dicts to global values
                s.consec_per_person[p]     = init_limit
                s.consec_rest_per_person[p] = init_rest

            tf_limit_p = ft.TextField(
                value=init_limit,
                width=_s(68), height=_s(34), text_size=_s(11),
                hint_text="Limit",
                content_padding=ft.padding.all(_s(4)),
                read_only=not is_personalized,
                color=ft.Colors.BLACK if is_personalized else ft.Colors.GREY_600,
                data=p)

            tf_rest_p = ft.TextField(
                value=init_rest,
                width=_s(68), height=_s(34), text_size=_s(11),
                hint_text="Rest",
                content_padding=ft.padding.all(_s(4)),
                read_only=not is_personalized,
                color=ft.Colors.BLACK if is_personalized else ft.Colors.GREY_600,
                data=p)

            def _ch_limit_p(ev, _p=p):
                v  = ev.control.value
                s.consec_per_person[_p] = v
                ok = UIHelpers.validate_positive_int(v) or v.strip() == ""
                if ok:
                    s.validation_errors["consec"].discard(_p)
                    ev.control.border_color = None
                else:
                    s.validation_errors["consec"].add(_p)
                    ev.control.border_color = ft.Colors.RED_400
                self._on_solve_blocked_update()
                ev.control.update()

            def _ch_rest_p(ev, _p=p):
                v  = ev.control.value
                s.consec_rest_per_person[_p] = v
                ok = UIHelpers.validate_positive_int(v) or v.strip() == ""
                if ok:
                    s.validation_errors["consec"].discard(f"{_p}_rest")
                    ev.control.border_color = None
                else:
                    s.validation_errors["consec"].add(f"{_p}_rest")
                    ev.control.border_color = ft.Colors.RED_400
                self._on_solve_blocked_update()
                try: ev.control.update()
                except Exception: pass

            tf_limit_p.on_change = _ch_limit_p
            tf_rest_p.on_change  = _ch_rest_p

            # ── Personalize / Reset button ────────────────────────────

            def _toggle_personalize(ev, _p=p):
                if _p in s.consec_personalized_persons:
                    # Reset → remove from personalized set, sync to global
                    s.consec_personalized_persons.discard(_p)
                    s.consec_per_person[_p]      = s.consec_global_val
                    s.consec_rest_per_person[_p]  = s.consec_global_rest
                    s.validation_errors["consec"].discard(_p)
                    s.validation_errors["consec"].discard(f"{_p}_rest")
                    self._on_solve_blocked_update()
                else:
                    # Personalize → add to set, keep current values
                    s.consec_personalized_persons.add(_p)
                # Rebuild only this card for clean state
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
                border=ft.border.all(1, btn_color),
                border_radius=12,
                padding=ft.padding.symmetric(horizontal=_s(8),
                                             vertical=_s(3)),
                on_click=_toggle_personalize,
                ink=True)

            # ── Name row ──────────────────────────────────────────────

            name_row_controls = [
                ft.Text(p, size=_s(11), weight=ft.FontWeight.BOLD,
                        color=group_color,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        max_lines=1, tooltip=p,
                        expand=True),
            ]
            if is_personalized:
                name_row_controls.append(
                    ft.Container(
                        content=ft.Text("custom", size=_s(8),
                                        color="#1565C0",
                                        weight=ft.FontWeight.BOLD,
                                        italic=True),
                        bgcolor="#E3F2FD",
                        border_radius=8,
                        padding=ft.padding.symmetric(horizontal=_s(5),
                                                      vertical=_s(1))))

            # ── Captain warning ───────────────────────────────────────

            captain_warning = None
            if is_captain:
                captain_warning = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.STAR, size=_s(10),
                                color="#F57F17"),
                        ft.Text(
                            "Captain — rest may be overridden by solver",
                            size=_s(8), italic=True,
                            color="#F57F17"),
                    ], spacing=_s(3), tight=True),
                    padding=ft.padding.only(top=_s(2)))

            # ── Card content ──────────────────────────────────────────

            card_content_controls = [
                ft.Row(name_row_controls, spacing=_s(4)),
                ft.Row([
                    ft.Column([
                        ft.Text("Max consec", size=_s(8),
                                color=ft.Colors.GREY_600),
                        tf_limit_p,
                    ], spacing=_s(1), tight=True),
                    ft.Column([
                        ft.Text("Min rest", size=_s(8),
                                color=ft.Colors.GREY_600),
                        tf_rest_p,
                    ], spacing=_s(1), tight=True),
                ], spacing=_s(6)),
                personalize_btn,
            ]
            if captain_warning is not None:
                card_content_controls.insert(1, captain_warning)

            # ── Card border & bg ──────────────────────────────────────

            if is_captain:
                card_border = ft.border.all(2, self._CAPTAIN_BORDER)
                card_bg     = self._CAPTAIN_BG
            else:
                card_border = ft.border.all(1, CARD_BORDER)
                card_bg     = CARD_BG

            card = ft.Container(
                content=ft.Column(card_content_controls,
                                  spacing=_s(4), tight=True),
                width=_s(195),
                bgcolor=card_bg,
                border=card_border,
                border_radius=8,
                padding=ft.padding.all(_s(8)))

            # Register widgets for global propagation
            person_widgets[p] = {
                "tf_limit": tf_limit_p,
                "tf_rest":  tf_rest_p,
                "card":     card,
            }

            return card

        def _rebuild_grid():
            """Rebuild the full grid of person cards, grouped."""
            query = search_filter[0].lower().strip()
            person_widgets.clear()

            sections = []
            for g_idx, (gname, members) in enumerate(groups.items()):
                g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]

                # Filter members by search
                visible_members = [
                    p for p in members
                    if not query or query in p.lower()
                ]
                if not visible_members:
                    continue

                # Group header
                header = ft.Container(
                    content=ft.Text(gname, size=_s(11),
                                    weight=ft.FontWeight.BOLD,
                                    color=g_color),
                    padding=ft.padding.only(top=_s(4), bottom=_s(2)))

                cards = [
                    _build_person_card(p, g_color)
                    for p in visible_members
                ]

                card_row = ft.Row(
                    cards, spacing=_s(8), run_spacing=_s(8),
                    wrap=True,
                    vertical_alignment=ft.CrossAxisAlignment.START)

                sections.append(header)
                sections.append(card_row)

            if not sections:
                sections = [
                    ft.Text("No people match the search.",
                            size=_s(11), italic=True,
                            color=ft.Colors.GREY_500)
                ]

            grid_container.controls = sections
            try: grid_container.update()
            except Exception: pass

        # ── Search field ──────────────────────────────────────────────

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
            hint_text="Search people…",
            prefix_icon=ft.Icons.SEARCH,
            on_change=_on_search_ch,
            text_size=_s(11), height=_s(36),
            content_padding=ft.padding.symmetric(
                horizontal=_s(10), vertical=_s(6)),
            border_color=ft.Colors.GREY_400,
            focused_border_color=BTN_COLOR,
            border_radius=8, bgcolor=ft.Colors.WHITE,
            suffix=ft.IconButton(
                icon=ft.Icons.CLEAR, icon_size=_s(14),
                icon_color=ft.Colors.GREY_400, tooltip="Clear search",
                style=ft.ButtonStyle(padding=ft.padding.all(0)),
                on_click=_clear_search))

        # ── Build grid initially ──────────────────────────────────────

        _rebuild_grid()

        # ══════════════════════════════════════════════════════════════
        # DIALOG LAYOUT
        # ══════════════════════════════════════════════════════════════

        def _on_close(ev):
            self._page.close(dlg)

        # Count stats for info text
        n_personalized = len(
            s.consec_personalized_persons & set(people))
        n_captains     = len(captain_persons & set(people))

        info_parts = []
        if n_captains:
            info_parts.append(
                f"{n_captains} captain(s) highlighted in yellow")
        if n_personalized:
            info_parts.append(
                f"{n_personalized} person(s) with custom values")
        info_suffix = (" — " + ", ".join(info_parts)) if info_parts else ""

        global_row = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Column([
                        tf_global_limit,
                        err_limit,
                    ], spacing=_s(2), tight=True),
                    ft.Column([
                        tf_global_rest,
                        err_rest,
                    ], spacing=_s(2), tight=True),
                ], spacing=_s(12),
                   vertical_alignment=ft.CrossAxisAlignment.START),
                ft.Text(
                    "Changes to global values automatically update all "
                    "non-personalized people." + info_suffix,
                    size=_s(9), italic=True, color=ft.Colors.GREY_500),
            ], spacing=_s(6), tight=True),
            bgcolor="#ECEFF1", border_radius=8,
            border=ft.border.all(1, "#B0BEC5"),
            padding=ft.padding.symmetric(horizontal=_s(14),
                                          vertical=_s(10)))

        grid_section = ft.Column([
            search_tf,
            ft.Container(
                content=ft.Column([grid_container],
                                  scroll=ft.ScrollMode.AUTO,
                                  expand=True),
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=8,
                padding=ft.padding.all(_s(10)),
                height=_s(420), expand=True),
        ], spacing=_s(8), tight=True, expand=True)

        content = ft.Column([
            ft.Text(
                "Configure the maximum consecutive working hours and "
                "the minimum rest hours between shifts.\n"
                "Set global values that apply to everyone, or click "
                "\"Personalize\" on a person's card to set individual limits.",
                size=_s(11), italic=True, color=ft.Colors.GREY_600),
            ft.Divider(height=_s(4)),
            global_row,
            ft.Divider(height=_s(4)),
            grid_section,
        ], spacing=_s(8), expand=True, width=_s(900), height=_s(620))

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.BEDTIME_OUTLINED,
                        size=_s(20), color=BTN_COLOR),
                ft.Text("Rest Conditions",
                        weight=ft.FontWeight.BOLD, size=_s(16)),
            ], spacing=_s(8)),
            content=content,
            actions=[
                ft.ElevatedButton(
                    "Close",
                    icon=ft.Icons.CHECK,
                    bgcolor=BTN_COLOR, color=ft.Colors.WHITE,
                    on_click=_on_close,
                    style=ft.ButtonStyle(
                        padding=ft.padding.symmetric(
                            horizontal=_s(16), vertical=_s(10)),
                        shape=ft.RoundedRectangleBorder(radius=6))),
            ],
            actions_alignment=ft.MainAxisAlignment.END)
        self._page.open(dlg)