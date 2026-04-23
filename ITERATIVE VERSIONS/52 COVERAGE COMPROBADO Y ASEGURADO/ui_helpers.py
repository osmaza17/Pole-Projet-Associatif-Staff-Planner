import flet as ft
from constants import (
    _s, TAB_ACTIVE_BG, TAB_ACTIVE_FG, TAB_INACTIVE_BG, TAB_INACTIVE_FG,
    DANGER_RED,
)


class UIHelpers:
    W_LBL  = _s(80)
    W_CELL = _s(50)
    H_BTN  = _s(26)
    H_TF   = _s(30)

    _AVAIL_LBL  = {1: "1",  0: "0",  2: "!"}
    _AVAIL_CLR  = {1: ft.Colors.GREEN_700, 0: ft.Colors.RED_700, 2: ft.Colors.ORANGE_700}
    _AVAIL_NEXT = {1: 0, 0: 2, 2: 1}

    # ── Pure parsing ─────────────────────────────────────────────────
    @staticmethod
    def parse_pos_int(raw: str) -> int | None:
        """Parse a strictly positive int. Returns None for empty/invalid."""
        s = (raw or "").strip()
        if not s:
            return None
        try:
            n = int(s)
            return n if n > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _validate_int(value: str, min_value: int) -> bool:
        v = (value or "").strip()
        if v == "":
            return True
        try:
            n = int(v)
            return n >= min_value and str(n) == v
        except ValueError:
            return False

    @staticmethod
    def validate_nonneg_int(value: str) -> bool:
        return UIHelpers._validate_int(value, 0)

    @staticmethod
    def validate_positive_int(value: str) -> bool:
        return UIHelpers._validate_int(value, 1)

    # ── Labels ───────────────────────────────────────────────────────
    @staticmethod
    def lbl(text: str, w: int = None) -> ft.Container:
        return ft.Container(
            ft.Text(text, size=_s(11), no_wrap=True),
            width=w or UIHelpers.W_LBL)

    @staticmethod
    def plbl(name: str, group_color: str = None, w: int = None) -> ft.Container:
        return ft.Container(
            ft.Text(name, size=_s(11), no_wrap=True, color=group_color,
                    weight=ft.FontWeight.BOLD if group_color else None),
            width=w or UIHelpers.W_LBL)

    # ── Buttons ──────────────────────────────────────────────────────
    @staticmethod
    def make_reset_btn(text: str, on_click) -> ft.Container:
        return ft.Container(
            content=ft.Text(text, color=ft.Colors.WHITE,
                            weight=ft.FontWeight.BOLD, size=_s(12)),
            bgcolor=DANGER_RED, padding=_s(8), border_radius=4,
            on_click=on_click, width=_s(150), alignment=ft.alignment.center)

    @staticmethod
    def _flat_btn(label: str, bg: str, fg: str = ft.Colors.WHITE) -> ft.Container:
        return ft.Container(
            ft.Text(label, color=fg, size=_s(10),
                    text_align=ft.TextAlign.CENTER),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN,
            bgcolor=bg, alignment=ft.alignment.center, border_radius=4)

    @staticmethod
    def make_rc_btn(label: str = "·") -> ft.Container:
        return UIHelpers._flat_btn(label, ft.Colors.BLUE_GREY_400)

    @staticmethod
    def make_inc_btn(delta: int) -> ft.Container:
        label = "+1" if delta > 0 else "-1"
        color = ft.Colors.GREEN_700 if delta > 0 else ft.Colors.RED_700
        return UIHelpers._flat_btn(label, color)

    # ── Navigation dropdown (BUGFIX: dd.on_change now uses _nav too) ─
    @staticmethod
    def make_nav_dropdown(label: str, value, options: list,
                          on_change, width: int = 200) -> ft.Row:
        dd = ft.Dropdown(
            label=label, value=value,
            options=[ft.dropdown.Option(o, text_style=ft.TextStyle(size=_s(11)))
                     for o in options],
            width=_s(width), text_size=_s(11),
            label_style=ft.TextStyle(size=_s(11)),
            dense=True,
            content_padding=ft.padding.symmetric(horizontal=_s(8), vertical=_s(4)))

        def _set(new_value):
            dd.value = new_value
            dd.update()
            on_change(new_value)

        def _step(direction):
            idx = options.index(dd.value) if dd.value in options else 0
            _set(options[(idx + direction) % len(options)])

        dd.on_change = lambda e: _set(e.control.value)

        def _btn(icon, direction):
            return ft.IconButton(
                icon=icon, icon_size=_s(18),
                tooltip="Previous" if direction == -1 else "Next",
                style=ft.ButtonStyle(padding=ft.padding.all(_s(4)),
                                     shape=ft.RoundedRectangleBorder(radius=6)),
                on_click=lambda e, d=direction: _step(d))

        return ft.Row(
            [_btn(ft.Icons.CHEVRON_LEFT, -1), dd, _btn(ft.Icons.CHEVRON_RIGHT, 1)],
            spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # ── Toggles ──────────────────────────────────────────────────────
    @staticmethod
    def _toggle_container(text: str, bg: str, fg: str, key) -> ft.Container:
        return ft.Container(
            content=ft.Text(text, color=fg, size=_s(11),
                            weight=ft.FontWeight.BOLD),
            width=UIHelpers.W_CELL, height=UIHelpers.H_BTN, data=key,
            bgcolor=bg, alignment=ft.alignment.center, border_radius=4)

    @staticmethod
    def make_toggle(sd: dict, key, default: int) -> ft.Container:
        if key not in sd:
            sd[key] = default

        def _click(e, _sd=sd, _k=key):
            _sd[_k] = 1 - _sd[_k]
            e.control.content.value = str(_sd[_k])
            e.control.bgcolor = (ft.Colors.GREEN_700 if _sd[_k]
                                 else ft.Colors.RED_700)
            e.control.update()

        bg = ft.Colors.GREEN_700 if sd[key] else ft.Colors.RED_700
        c = UIHelpers._toggle_container(str(sd[key]), bg, ft.Colors.WHITE, key)
        c.on_click = _click
        return c

    @staticmethod
    def make_avail_toggle(sd: dict, key, default: int = 1) -> ft.Container:
        _lbl, _clr, _next = (UIHelpers._AVAIL_LBL, UIHelpers._AVAIL_CLR,
                             UIHelpers._AVAIL_NEXT)
        if key not in sd:
            sd[key] = default

        def _click(e, _sd=sd, _k=key):
            _sd[_k] = _next[_sd[_k]]
            nv = _sd[_k]
            e.control.content.value = _lbl[nv]
            e.control.bgcolor = _clr[nv]
            e.control.update()

        c = UIHelpers._toggle_container(_lbl[sd[key]], _clr[sd[key]],
                                        ft.Colors.WHITE, key)
        c.on_click = _click
        return c

    @staticmethod
    def make_force_toggle(sd: dict, key, default: int,
                          task_bg: str, task_fg: str) -> ft.Container:
        if key not in sd:
            sd[key] = default

        def _click(e, _sd=sd, _k=key, _tbg=task_bg, _tfg=task_fg):
            _sd[_k] = 1 - _sd[_k]
            on = bool(_sd[_k])
            e.control.content.value = str(_sd[_k])
            e.control.bgcolor = _tbg if on else ft.Colors.GREY_300
            e.control.content.color = _tfg if on else ft.Colors.GREY_600
            e.control.update()

        on = bool(sd[key])
        c = UIHelpers._toggle_container(
            str(sd[key]),
            task_bg if on else ft.Colors.GREY_300,
            task_fg if on else ft.Colors.GREY_600,
            key)
        c.on_click = _click
        return c

    # ── Tab bar ──────────────────────────────────────────────────────
    @staticmethod
    def make_tab_bar(options: list, selected, on_change) -> ft.Row:
        tabs = []
        for opt in options:
            sel = (opt == selected)
            tab = ft.Container(
                content=ft.Text(
                    str(opt), size=_s(11),
                    color=TAB_ACTIVE_FG if sel else TAB_INACTIVE_FG,
                    weight=ft.FontWeight.BOLD if sel else None,
                    no_wrap=True),
                bgcolor=TAB_ACTIVE_BG if sel else TAB_INACTIVE_BG,
                padding=ft.padding.symmetric(horizontal=_s(14), vertical=_s(7)),
                border_radius=ft.border_radius.only(top_left=6, top_right=6),
                on_click=lambda e, _o=opt: on_change(_o))
            tabs.append(tab)
        return ft.Row(tabs, spacing=_s(2),
                      scroll=ft.ScrollMode.AUTO, wrap=False)