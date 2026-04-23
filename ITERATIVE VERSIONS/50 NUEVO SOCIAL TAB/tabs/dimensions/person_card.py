"""
person_card.py — builds a single person card for the rest/workload dialog.
"""

import flet as ft
from constants import _s
from .rest_fields import (FIELDS, validator_for, err_key_for,
                          reset_person_to_globals, safe_update)

_CARD_BORDER    = "#CFD8DC"
_CAPTAIN_BORDER = "#F9A825"
_CAPTAIN_BG     = "#FFFDE7"
_ACCENT         = "#1565C0"


def _lighten(hex_color: str, factor: float = 0.82) -> str:
    """Mix *hex_color* towards white.  factor 1 → white, 0 → original."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return "#F8F9FA"          # fallback
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def _darken(hex_color: str, factor: float = 0.35) -> str:
    """Darken *hex_color* for use as a subtle border."""
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return _CARD_BORDER
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (200 - r) * factor)
    g = int(g + (200 - g) * factor)
    b = int(b + (200 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def build_person_card(
    person: str,
    group_color: str,
    state,
    captain_persons: set,
    on_solve_blocked_update,
    rebuild_grid_fn,
) -> tuple[ft.Container, dict]:
    """Return (card_container, {suffix→TextField, "card"→container})."""
    s = state
    is_personalized = person in s.consec_personalized_persons
    is_captain = person in captain_persons

    card_field_tfs: dict = {}
    field_cols = []

    for g_attr, pp_attr, suffix, label, hint, w, vkind in FIELDS:
        pp_dict = getattr(s, pp_attr)
        if is_personalized:
            init_val = pp_dict.get(person, "")
        else:
            init_val = getattr(s, g_attr, "")
            pp_dict[person] = init_val

        tf = ft.TextField(
            value=init_val,
            width=_s(66), height=_s(36), text_size=_s(11),
            text_align=ft.TextAlign.CENTER,
            hint_text=hint,
            content_padding=ft.padding.symmetric(
                horizontal=_s(4), vertical=_s(4)),
            read_only=not is_personalized,
            color=ft.Colors.BLACK if is_personalized else ft.Colors.GREY_500,
            bgcolor=ft.Colors.WHITE if is_personalized else "#EEEEEE",
            border_color=_ACCENT if is_personalized else "#E0E0E0",
            border_radius=5,
            data=person)
        card_field_tfs[suffix] = tf

        _validator = validator_for(vkind)
        _ek = err_key_for(person, suffix)

        def _make_ch(_p=person, _pp=pp_attr, _v=_validator, _ek=_ek):
            def handler(ev):
                getattr(s, _pp)[_p] = ev.control.value
                ok = _v(ev.control.value)
                if ok:
                    s.validation_errors["consec"].discard(_ek)
                    ev.control.border_color = _ACCENT
                else:
                    s.validation_errors["consec"].add(_ek)
                    ev.control.border_color = ft.Colors.RED_400
                on_solve_blocked_update()
                safe_update(ev.control)
            return handler

        tf.on_change = _make_ch()
        field_cols.append(
            ft.Column([
                ft.Text(label, size=_s(9), color=ft.Colors.GREY_700,
                        weight=ft.FontWeight.W_500),
                tf,
            ], spacing=_s(2), tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    # ── Personalize / Reset toggle ────────────────────────────────
    def _toggle(ev, _p=person):
        if _p in s.consec_personalized_persons:
            s.consec_personalized_persons.discard(_p)
            reset_person_to_globals(s, _p)
            on_solve_blocked_update()
        else:
            s.consec_personalized_persons.add(_p)
        rebuild_grid_fn()

    if is_personalized:
        btn_label = "Reset to global"
        btn_icon = ft.Icons.RESTART_ALT
        btn_color = ft.Colors.RED_700
        btn_bg = "#FFEBEE"
    else:
        btn_label = "Personalize"
        btn_icon = ft.Icons.EDIT
        btn_color = _ACCENT
        btn_bg = "#E3F2FD"

    personalize_btn = ft.Container(
        content=ft.Row([
            ft.Icon(btn_icon, size=_s(13), color=btn_color),
            ft.Text(btn_label, size=_s(10), color=btn_color,
                    weight=ft.FontWeight.BOLD),
        ], spacing=_s(4), tight=True),
        bgcolor=btn_bg,
        border=ft.border.all(1, btn_color), border_radius=6,
        padding=ft.padding.symmetric(horizontal=_s(10), vertical=_s(5)),
        on_click=_toggle, ink=True)

    # ── Name row ──────────────────────────────────────────────────
    name_ctrls = [
        ft.Text(person, size=_s(12), weight=ft.FontWeight.BOLD,
                color=group_color, overflow=ft.TextOverflow.ELLIPSIS,
                max_lines=1, tooltip=person, expand=True)]
    if is_personalized:
        name_ctrls.append(ft.Container(
            content=ft.Text("CUSTOM", size=_s(8), color=_ACCENT,
                            weight=ft.FontWeight.BOLD, italic=True),
            bgcolor="#E3F2FD", border_radius=4,
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(2))))

    card_controls: list = [
        ft.Row(name_ctrls, spacing=_s(4)),
    ]

    if is_captain:
        card_controls.append(ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.STAR, size=_s(11), color="#F57F17"),
                ft.Text("Captain — rest may be overridden",
                        size=_s(9), italic=True, color="#F57F17"),
            ], spacing=_s(4), tight=True),
            padding=ft.padding.only(top=_s(2))))

    card_controls.append(ft.Divider(height=_s(2), color="#E0E0E0"))
    card_controls.append(
        ft.Row(field_cols, spacing=_s(6), wrap=True, run_spacing=_s(4)))
    card_controls.append(personalize_btn)

    card_bg = _CAPTAIN_BG if is_captain else _lighten(group_color)
    card_border = (ft.border.all(2, _CAPTAIN_BORDER) if is_captain
                   else ft.border.all(1, _darken(group_color)))

    card = ft.Container(
        content=ft.Column(card_controls, spacing=_s(6), tight=True),
        width=_s(380), bgcolor=card_bg,
        border=card_border, border_radius=10,
        padding=ft.padding.all(_s(10)))

    card_field_tfs["card"] = card
    return card, card_field_tfs