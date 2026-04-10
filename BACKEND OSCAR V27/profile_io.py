"""
profile_io.py
=============
Serializes / deserializes the AppState to/from a JSON string.

NOTE: backward compatibility was intentionally dropped per user request.
Old profiles that contain `quota_st` will simply lose their quota data.
"""

import json
from constants import DEFAULT_WEIGHTS
from state import AppState


# ── Tuple-key helpers ────────────────────────────────────────────────────────

def _enc(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = json.dumps(list(k), ensure_ascii=False) if isinstance(k, tuple) else k
        out[key] = v
    return out


def _dec(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            parsed = json.loads(k)
            out[tuple(parsed) if isinstance(parsed, list) else k] = v
        except (json.JSONDecodeError, ValueError):
            out[k] = v
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def save_profile(s: AppState) -> str:
    payload = {
        "tasks_text":          s.tasks_text,
        "days_text":           s.days_text,
        "groups_st":           s.groups_st,
        "hours_per_day":       s.hours_per_day,

        "avail_st":            _enc(s.avail_st),
        "demand_st":           _enc(s.demand_st),
        "skills_st":           _enc(s.skills_st),
        "force_st":            _enc(s.force_st),
        "just_work_st":        _enc(s.just_work_st),
        "social_st":           _enc(s.social_st),

        "rotation_st":         s.rotation_st,
        "hard_enemies":        s.hard_enemies,

        "consec_global_val":            s.consec_global_val,
        "consec_global_rest":           s.consec_global_rest,
        "consec_per_person":            s.consec_per_person,
        "consec_rest_per_person":       s.consec_rest_per_person,
        "consec_personalized_persons":  sorted(s.consec_personalized_persons),

        "captain_rules":       s.captain_rules,
        "mandatory_rules":     s.mandatory_rules,
        "quota_rules":         s.quota_rules,

        "weights_st":          s.weights_st,
        "weights_order":       s.weights_order,
        "weights_enabled":     s.weights_enabled,
        "weights_last_value":  s.weights_last_value,
        "solver_params":       s.solver_params,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def load_profile(s: AppState, json_str: str) -> None:
    data = json.loads(json_str)

    s.tasks_text    = data.get("tasks_text",    s.tasks_text)
    s.days_text     = data.get("days_text",     s.days_text)
    s.groups_st     = data.get("groups_st",     s.groups_st)
    s.hours_per_day = data.get("hours_per_day", s.hours_per_day)

    s.avail_st     = _dec(data.get("avail_st",     {}))
    s.demand_st    = _dec(data.get("demand_st",    {}))
    s.skills_st    = _dec(data.get("skills_st",    {}))
    s.force_st     = _dec(data.get("force_st",     {}))
    s.just_work_st = _dec(data.get("just_work_st", {}))
    s.social_st    = _dec(data.get("social_st",    {}))

    s.rotation_st  = data.get("rotation_st",  {})
    s.hard_enemies = data.get("hard_enemies", False)

    s.consec_global_val      = data.get("consec_global_val",   "")
    s.consec_global_rest     = data.get("consec_global_rest",  "1")
    s.consec_per_person      = data.get("consec_per_person",   {})
    s.consec_rest_per_person = data.get("consec_rest_per_person", {})
    s.consec_personalized_persons = set(
        data.get("consec_personalized_persons", []))
    s.consec_personalized = False

    s.captain_rules   = data.get("captain_rules",   [])
    s.mandatory_rules = data.get("mandatory_rules", [])
    s.quota_rules     = data.get("quota_rules",     [])

    s.weights_st      = data.get("weights_st",      s.weights_st)
    s.weights_order   = data.get("weights_order",   s.weights_order)
    s.weights_enabled = data.get("weights_enabled", s.weights_enabled)
    s.solver_params   = data.get("solver_params",   s.solver_params)

    raw_last = data.get("weights_last_value", None)
    if raw_last is not None:
        s.weights_last_value = raw_last
    else:
        s.weights_last_value = {}
        for k in s.weights_st:
            v = s.weights_st[k]
            s.weights_last_value[k] = v if v > 0 else DEFAULT_WEIGHTS.get(k, 1)

    for wk, wv in DEFAULT_WEIGHTS.items():
        if wk not in s.weights_st:
            s.weights_st[wk] = wv
        if wk not in s.weights_enabled:
            s.weights_enabled[wk] = True
        if wk not in s.weights_last_value:
            s.weights_last_value[wk] = wv
        if wk not in s.weights_order:
            inserted = False
            for idx, existing_key in enumerate(s.weights_order):
                if DEFAULT_WEIGHTS.get(existing_key, 0) < wv:
                    s.weights_order.insert(idx, wk)
                    inserted = True
                    break
            if not inserted:
                s.weights_order.append(wk)

    obsolete_keys = [k for k in s.weights_st if k not in DEFAULT_WEIGHTS]
    for k in obsolete_keys:
        s.weights_st.pop(k, None)
        s.weights_enabled.pop(k, None)
        s.weights_last_value.pop(k, None)
        if k in s.weights_order:
            s.weights_order.remove(k)

    # ── Reset run state ───────────────────────────────────────────────
    s.solution_history  = []
    s.diff_state        = {"ref": None, "cmp": None}
    s.base_run_idx      = None
    s.solve_blocked     = False
    s.solver_running    = False
    s.validation_errors = {"demand": set(), "consec": set(), "rules": set()}
    s.invalidate_cache()