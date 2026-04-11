"""
person_card.py — builds a single person card for the rest/workload dialog.
"""

import flet as ft
from constants import _s
from .rest_fields import (FIELDS, validator_for, err_key_for,
                          reset_person_to_globals, safe_update)

_CARD_BG        = "#ECEFF1"
_CARD_BORDER    = "#B0BEC5"
_CAPTAIN_BORDER = "#F9A825"
_CAPTAIN_BG     = "#FFFDE7"


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
            width=_s(w), height=_s(34), text_size=_s(11),
            hint_text=hint,
            content_padding=ft.padding.all(_s(4)),
            read_only=not is_personalized,
            color=ft.Colors.BLACK if is_personalized else ft.Colors.GREY_600,
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
                    ev.control.border_color = None
                else:
                    s.validation_errors["consec"].add(_ek)
                    ev.control.border_color = ft.Colors.RED_400
                on_solve_blocked_update()
                safe_update(ev.control)
            return handler

        tf.on_change = _make_ch()
        field_cols.append(
            ft.Column([
                ft.Text(label, size=_s(8), color=ft.Colors.GREY_600),
                tf,
            ], spacing=_s(1), tight=True))

    # ── Personalize / Reset toggle ────────────────────────────────
    def _toggle(ev, _p=person):
        if _p in s.consec_personalized_persons:
            s.consec_personalized_persons.discard(_p)
            reset_person_to_globals(s, _p)
            on_solve_blocked_update()
        else:
            s.consec_personalized_persons.add(_p)
        rebuild_grid_fn()

    btn_label = "Reset" if is_personalized else "Personalize"
    btn_icon = ft.Icons.RESTART_ALT if is_personalized else ft.Icons.EDIT
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

    # ── Name row ──────────────────────────────────────────────────
    name_ctrls = [
        ft.Text(person, size=_s(11), weight=ft.FontWeight.BOLD,
                color=group_color, overflow=ft.TextOverflow.ELLIPSIS,
                max_lines=1, tooltip=person, expand=True)]
    if is_personalized:
        name_ctrls.append(ft.Container(
            content=ft.Text("custom", size=_s(8), color="#1565C0",
                            weight=ft.FontWeight.BOLD, italic=True),
            bgcolor="#E3F2FD", border_radius=8,
            padding=ft.padding.symmetric(horizontal=_s(5), vertical=_s(1))))

    card_controls: list = [ft.Row(name_ctrls, spacing=_s(4))]

    if is_captain:
        card_controls.append(ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.STAR, size=_s(10), color="#F57F17"),
                ft.Text("Captain — rest may be overridden",
                        size=_s(8), italic=True, color="#F57F17"),
            ], spacing=_s(3), tight=True),
            padding=ft.padding.only(top=_s(2))))

    card_controls.append(ft.Row(field_cols, spacing=_s(4), wrap=True))
    card_controls.append(personalize_btn)

    card_bg = _CAPTAIN_BG if is_captain else _CARD_BG
    card_border = (ft.border.all(2, _CAPTAIN_BORDER) if is_captain
                   else ft.border.all(1, _CARD_BORDER))

    card = ft.Container(
        content=ft.Column(card_controls, spacing=_s(4), tight=True),
        width=_s(340), bgcolor=card_bg,
        border=card_border, border_radius=8,
        padding=ft.padding.all(_s(8)))

    card_field_tfs["card"] = card
    return card, card_field_tfs