"""
profile_io.py
=============
Full serialization of AppState to/from a JSON string.

Every attribute declared in AppState._init_defaults() is persisted,
including runtime state (solution_history, diff_state, caches, filters,
validation errors, …). The only exception is `running_model_ref`,
which holds a live gurobipy.Model C handle that cannot be serialized;
it is always reset to [None] on load.

Non-JSON-native containers (tuple, set, dict-with-non-string-keys) are
encoded as tagged single-key dicts and rebuilt on decode. This is
fully recursive, so arbitrarily nested structures round-trip cleanly.
"""

import json
from state import AppState


# ── Attributes to persist ────────────────────────────────────────────
# Order matches AppState._init_defaults() for readability.
# `running_model_ref` is intentionally excluded (live Gurobi handle).
_FIELDS = (
    "tasks_text", "days_text",
    "avail_st", "demand_st", "skills_st", "force_st",
    "just_work_st", "social_st",
    "rotation_st", "task_duration_st",
    "hard_enemies", "hours_per_day", "groups_st",
    "captain_rules", "mandatory_rules", "quota_rules",
    "pref_order_st", "pref_enabled_st",
    "consec_global_val", "consec_global_rest",
    "consec_global_capacity",
    "consec_global_max_day", "consec_global_max_event",
    "consec_per_person", "consec_rest_per_person",
    "consec_capacity_per_person",
    "consec_max_day_per_person", "consec_max_event_per_person",
    "consec_personalized_persons", "consec_personalized",
    "avail_filter", "demand_filter", "force_filter",
    "location_names_st", "task_location_idx_st", "travel_time_st",
    "validation_errors",
    "solve_blocked", "solver_running",
    "solution_history", "diff_state", "base_run_idx",
    "weights_st", "weights_order", "weights_enabled",
    "weights_last_value", "solver_params",
    "_build_cache",
    "eq_group_mode", "group_shares_st", "group_equality_sets_st",
)


# ── Tagged recursive encoder / decoder ───────────────────────────────

_TAG_TUPLE = "__tuple__"
_TAG_SET   = "__set__"
_TAG_TDICT = "__tdict__"   # dict with non-string keys


def _encode(obj):
    # Order matters: bool is a subclass of int, but isinstance catches both.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, tuple):
        return {_TAG_TUPLE: [_encode(x) for x in obj]}
    if isinstance(obj, (set, frozenset)):
        # Sort when possible for stable JSON diffs; fall back otherwise.
        try:
            items = sorted(obj)
        except TypeError:
            items = list(obj)
        return {_TAG_SET: [_encode(x) for x in items]}
    if isinstance(obj, list):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        return {_TAG_TDICT: [[_encode(k), _encode(v)] for k, v in obj.items()]}
    raise TypeError(f"profile_io: cannot serialize type {type(obj).__name__}")


def _decode(obj):
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    if isinstance(obj, dict):
        if len(obj) == 1:
            if _TAG_TUPLE in obj:
                return tuple(_decode(x) for x in obj[_TAG_TUPLE])
            if _TAG_SET in obj:
                return {_decode(x) for x in obj[_TAG_SET]}
            if _TAG_TDICT in obj:
                return {_decode(k): _decode(v) for k, v in obj[_TAG_TDICT]}
        return {k: _decode(v) for k, v in obj.items()}
    return obj


# ── Public API ───────────────────────────────────────────────────────

def save_profile(s: AppState) -> str:
    payload = {name: _encode(getattr(s, name)) for name in _FIELDS}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def load_profile(s: AppState, json_str: str) -> None:
    data = json.loads(json_str)
    for name in _FIELDS:
        if name in data:
            setattr(s, name, _decode(data[name]))
    # Live Gurobi model handle — never persisted, always reset.
    s.running_model_ref = [None]