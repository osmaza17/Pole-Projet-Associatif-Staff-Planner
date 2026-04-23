"""
rest_fields.py — field specifications, validators, and shared helpers
for rest/workload conditions.

Pure data and stateless helpers. No UI widgets.
"""

import flet as ft
from ui_helpers import UIHelpers

# ── Field spec ────────────────────────────────────────────────────
# (state_global_attr, state_per_person_attr,
#  suffix, label, hint, width, validator_kind)

FIELDS = [
    ("consec_global_val",       "consec_per_person",           "",       "Max consec",   "e.g. 8",  58, "pos_int"),
    ("consec_global_rest",      "consec_rest_per_person",      "_rest",  "Min rest",     "e.g. 1",  58, "pos_int"),
    ("consec_global_capacity",  "consec_capacity_per_person",  "_cap",   "Capacity %",   "0–100",   58, "cap"),
    ("consec_global_max_day",   "consec_max_day_per_person",   "_mday",  "Max h/day",    "e.g. 6",  58, "pos_int"),
    ("consec_global_max_event", "consec_max_event_per_person", "_mevt",  "Max h/event",  "e.g. 20", 58, "pos_int"),
]


# ── Validators ────────────────────────────────────────────────────

def validate_capacity(val: str) -> bool:
    v = val.strip()
    if v == "":
        return True
    try:
        return 0 <= int(v) <= 100
    except ValueError:
        return False


def validate_pos_int(val: str) -> bool:
    return UIHelpers.validate_positive_int(val) or val.strip() == ""


def validator_for(kind: str):
    return validate_capacity if kind == "cap" else validate_pos_int


def err_msg_for(kind: str) -> str:
    return "Integer 0–100" if kind == "cap" else "Positive integer"


# ── Shared helpers ────────────────────────────────────────────────

def err_key_for(person: str, suffix: str) -> str:
    """Canonical validation-error key for a person + field."""
    return f"{person}{suffix}" if suffix else person


def reset_person_to_globals(state, person: str) -> None:
    """Copy every global value into *person*'s per-person dicts
    and clear associated validation errors."""
    for g_attr, pp_attr, suffix, _, _, _, _ in FIELDS:
        getattr(state, pp_attr)[person] = getattr(state, g_attr, "")
        state.validation_errors["consec"].discard(
            err_key_for(person, suffix))


def safe_update(*controls: ft.Control) -> None:
    """Call .update() on each control, silencing errors from controls
    not yet attached to the page tree."""
    for c in controls:
        try:
            c.update()
        except Exception:
            pass