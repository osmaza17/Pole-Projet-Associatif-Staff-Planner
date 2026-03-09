"""
profile_io.py
=============
Serializa y deserializa el AppState completo a/desde un string JSON.

Problema clave: los dicts del estado usan tuplas como claves
(ej. avail_st: {(person, hour, day): int}).
JSON solo admite str como clave, por lo que:
  - Al guardar: tuple → json.dumps(list(tuple))  →  '["Ana", "09:00", "Mon"]'
  - Al cargar:  str   → tuple(json.loads(str))   →  ("Ana", "09:00", "Mon")
"""

import json
from constants import DEFAULT_WEIGHTS
from state import AppState


# ── Helpers de serialización de claves ───────────────────────────────────────

def _enc(d: dict) -> dict:
    """Convierte claves-tupla a strings JSON para poder guardar en JSON."""
    out = {}
    for k, v in d.items():
        key = json.dumps(list(k), ensure_ascii=False) if isinstance(k, tuple) else k
        out[key] = v
    return out


def _dec(d: dict) -> dict:
    """Convierte strings JSON de vuelta a tuplas."""
    out = {}
    for k, v in d.items():
        try:
            parsed = json.loads(k)
            out[tuple(parsed) if isinstance(parsed, list) else k] = v
        except (json.JSONDecodeError, ValueError):
            out[k] = v
    return out


# ── API pública ───────────────────────────────────────────────────────────────

def save_profile(s: AppState) -> str:
    """Devuelve un string JSON con todo el estado serializable del AppState."""
    payload = {
        # ── Dimensiones ───────────────────────────────────────────────
        "tasks_text":          s.tasks_text,
        "days_text":           s.days_text,
        "groups_st":           s.groups_st,        # {nombre_grupo: "p1\np2\n..."}
        "hours_per_day":       s.hours_per_day,    # {day: "HH:MM\n..."}

        # ── Matrices (claves-tupla → string JSON) ─────────────────────
        "avail_st":            _enc(s.avail_st),   # {(p,h,d): 0|1|2}
        "demand_st":           _enc(s.demand_st),  # {(t,h,d): str}
        "skills_st":           _enc(s.skills_st),  # {(p,t):   0|1}
        "force_st":            _enc(s.force_st),   # {(p,t,h,d): 0|1}
        "just_work_st":        _enc(s.just_work_st),  # {(p,h,d): 0|1}
        "social_st":           _enc(s.social_st),  # {(p1,p2): -1|0|1}
        "quota_st":            _enc(s.quota_st),   # {(p,t,d): str}

        # ── Configuración simple ──────────────────────────────────────
        "rotation_st":         s.rotation_st,      # {task: 0|1}
        "hard_enemies":        s.hard_enemies,

        # ── Turnos consecutivos ───────────────────────────────────────
        "consec_global_val":            s.consec_global_val,
        "consec_global_rest":           s.consec_global_rest,
        "consec_per_person":            s.consec_per_person,      # {person: str}
        "consec_rest_per_person":       s.consec_rest_per_person, # {person: str}
        "consec_personalized_persons":  sorted(s.consec_personalized_persons),

        # ── Capitanes ────────────────────────────────────────────────
        "captain_rules":       s.captain_rules,
        "mandatory_rules":  s.mandatory_rules,

        # ── Pesos y solver ────────────────────────────────────────────
        "weights_st":          s.weights_st,
        "weights_order":       s.weights_order,
        "weights_enabled":     s.weights_enabled,
        "weights_last_value":  s.weights_last_value,
        "solver_params":       s.solver_params,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def load_profile(s: AppState, json_str: str) -> None:
    """
    Lee el JSON y sobreescribe el AppState completo.
    También resetea el estado de ejecución (historial, diff, etc.).
    """
    data = json.loads(json_str)

    # ── Dimensiones ───────────────────────────────────────────────────
    s.tasks_text    = data.get("tasks_text",    s.tasks_text)
    s.days_text     = data.get("days_text",     s.days_text)
    s.groups_st     = data.get("groups_st",     s.groups_st)
    s.hours_per_day = data.get("hours_per_day", s.hours_per_day)

    # ── Matrices ──────────────────────────────────────────────────────
    s.avail_st  = _dec(data.get("avail_st",  {}))
    s.demand_st = _dec(data.get("demand_st", {}))
    s.skills_st = _dec(data.get("skills_st", {}))
    s.force_st  = _dec(data.get("force_st",  {}))
    s.just_work_st = _dec(data.get("just_work_st", {}))
    s.social_st = _dec(data.get("social_st", {}))
    s.quota_st  = _dec(data.get("quota_st",  {}))

    # ── Configuración simple ──────────────────────────────────────────
    s.rotation_st  = data.get("rotation_st",  {})
    s.hard_enemies = data.get("hard_enemies", False)

    # ── Turnos consecutivos ───────────────────────────────────────────
    s.consec_global_val        = data.get("consec_global_val",   "")
    s.consec_global_rest       = data.get("consec_global_rest",  "1")
    s.consec_per_person        = data.get("consec_per_person",   {})
    s.consec_rest_per_person   = data.get("consec_rest_per_person", {})

    # New field: set of person names with individual overrides
    raw_personalized_persons = data.get("consec_personalized_persons", None)
    if raw_personalized_persons is not None:
        # New format: list of person names
        s.consec_personalized_persons = set(raw_personalized_persons)
    else:
        # ── Backward compat: migrate from old boolean format ──────────
        # If the old profile had consec_personalized=True, that meant ALL
        # people with a non-empty consec_per_person value were personalized.
        old_bool = data.get("consec_personalized", False)
        if old_bool:
            s.consec_personalized_persons = {
                p for p, v in s.consec_per_person.items()
                if v.strip()
            }
        else:
            s.consec_personalized_persons = set()

    # Legacy bool — kept in state for any code that might reference it
    s.consec_personalized = False

    # ── Capitanes ────────────────────────────────────────────────────
    s.captain_rules = data.get("captain_rules", [])
    s.mandatory_rules = data.get("mandatory_rules",  [])

    # ── Pesos y solver ────────────────────────────────────────────────
    s.weights_st      = data.get("weights_st",      s.weights_st)
    s.weights_order   = data.get("weights_order",   s.weights_order)
    s.weights_enabled = data.get("weights_enabled", s.weights_enabled)
    s.solver_params   = data.get("solver_params",   s.solver_params)

    # Backward compat: weights_last_value may not exist in old profiles.
    # For enabled weights, use their current value; for disabled, use default.
    raw_last = data.get("weights_last_value", None)
    if raw_last is not None:
        s.weights_last_value = raw_last
    else:
        # Migrate: seed from current weights_st (non-zero) or defaults
        s.weights_last_value = {}
        for k in s.weights_st:
            v = s.weights_st[k]
            s.weights_last_value[k] = v if v > 0 else DEFAULT_WEIGHTS.get(k, 1)

    # ── Backward compat: asegurar que W_CAPTAIN existe en pesos ──────
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

    # ── Resetear estado de ejecución ──────────────────────────────────
    s.solution_history  = []
    s.diff_state        = {"ref": None, "cmp": None}
    s.base_run_idx      = None
    s.solve_blocked     = False
    s.solver_running    = False
    s.validation_errors = {"demand": set(), "quota": set(), "consec": set()}
    s.invalidate_cache()