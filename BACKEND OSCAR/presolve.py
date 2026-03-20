"""
presolve.py
===========
Standalone feasibility LP + IIS analysis.
Covers ALL hard constraints present in solve_model_pace_14.py:

  1. Availability + skills         — x only created for eligible persons
  2. One task per person per hour  — x.sum(p,*,h,d) <= 1
  3. Demand coverage (hard)        — x.sum(*,t,h,d) >= demand
  4. Hard enemies                  — x[p1,t,h,d] + x[p2,t,h,d] <= 1
  5. Max consecutive hours         — sliding window sum <= max_work

All other constraints in the main model (captain rules, mandatory rules,
quota, equity, stability, gaps, social-soft) use slack variables and
therefore can NEVER cause infeasibility.

KEY DESIGN: every constraint is stored in a meta-dict keyed by a stable
integer ID (not by a name string that may contain user-defined substrings).
This prevents parsing bugs when task/person names contain the separator.

NOTE: requires Gurobi ≥ 11 for `with gp.Env(...) as env:` context manager.

Public API
----------
run_presolve(data) -> (bool, dict)

  data keys:
    people, tasks, hours (dict {day:[hours]}), days,
    availability  {(p,h,d): 0|1},
    skills        {(p,t):   0|1},
    demand        {(t,h,d): int},
    hard_enemies  bool  (optional, default False),
    social        dict  {(p1,p2): -1|0|1}  (optional),
    max_consec_hours dict {person: (max_work, min_rest)} (optional)

  Returns (True, {}) when feasible.
  Returns (False, result_dict) when infeasible.

  result_dict keys:
    simple_bottlenecks  — slots where eligible count < demand
    iis.cover_slots     — demand constraints that appear in the IIS
    iis.person_conflicts— person-hour constraints in the IIS
    iis.hour_groups     — per-hour demand vs. available pool summary
    iis.enemy_conflicts — hard-enemy pairs that block full coverage
    iis.consec_conflicts— consecutive-hours windows that block coverage
"""

import gurobipy as gp
from gurobipy import GRB


def run_presolve(data: dict) -> tuple[bool, dict]:
    people           = data["people"]
    tasks            = data["tasks"]
    hours            = data["hours"]
    days             = data["days"]
    availability     = data["availability"]
    skills           = data["skills"]
    demand           = data["demand"]
    hard_enemies     = data.get("hard_enemies", False)
    social           = data.get("social", {})
    max_consec_hours = data.get("max_consec_hours", {})

    # ── Pre-compute eligible persons per (task, hour, day) ───────────
    # A person is eligible iff available (avail >= 1) AND has the skill.
    people_set = set(people)
    eligible: dict = {}
    for j in days:
        for h in hours[j]:
            for t in tasks:
                eligible[(t, h, j)] = [
                    p for p in people
                    if availability.get((p, h, j), 0) >= 1
                    and skills.get((p, t), 0) == 1
                ]

    enemy_pairs = (
        [(p1, p2) for (p1, p2), v in social.items() if v == -1]
        if hard_enemies else [])

    # ── Constraint meta-dicts  (int ID → info dict) ──────────────────
    cover_meta:  dict[int, dict] = {}
    person_meta: dict[int, dict] = {}
    enemy_meta:  dict[int, dict] = {}
    consec_meta: dict[int, dict] = {}

    # ── Phase 1: LP solve ────────────────────────────────────────────
    try:
        with gp.Env(empty=True) as env:  # requires Gurobi ≥ 11
            env.start()

            with gp.Model(env=env) as mdl:

                # Variables: x[p, t, h, j] in [0, 1]  (continuous relaxation)
                var_keys = [
                    (p, t, h, j)
                    for j in days for h in hours[j]
                    for t in tasks
                    for p in eligible.get((t, h, j), [])
                ]
                x = mdl.addVars(var_keys, lb=0.0, ub=1.0,
                                vtype=GRB.CONTINUOUS, name="x")

                cid = 0  # monotonically increasing constraint ID

                # ── C1: one task per person per hour ─────────────────
                for j in days:
                    for h in hours[j]:
                        for p in people:
                            terms = [x[p, t, h, j] for t in tasks
                                     if (p, t, h, j) in x]
                            if not terms:
                                continue
                            mdl.addConstr(
                                gp.quicksum(terms) <= 1,
                                name=f"C{cid}")
                            person_meta[cid] = {
                                "person": p, "hour": h, "day": j}
                            cid += 1

                # ── C2: demand coverage ───────────────────────────────
                for j in days:
                    for h in hours[j]:
                        for t in tasks:
                            d = demand.get((t, h, j), 0)
                            if d <= 0:
                                continue
                            elig = eligible.get((t, h, j), [])
                            terms = [x[p, t, h, j] for p in elig
                                     if (p, t, h, j) in x]
                            # Use LinExpr() when terms is empty so Gurobi
                            # receives a valid constraint (0 >= d), not a
                            # float comparison that raises GurobiError.
                            lhs = gp.quicksum(terms) if terms else gp.LinExpr()
                            mdl.addConstr(lhs >= d, name=f"C{cid}")
                            cover_meta[cid] = {
                                "task": t, "hour": h, "day": j,
                                "required": d,
                                "eligible": len(elig),
                                "names":    list(elig),
                            }
                            cid += 1

                # ── C3: hard enemies ──────────────────────────────────
                if hard_enemies and enemy_pairs:
                    for (p1, p2) in enemy_pairs:
                        for j in days:
                            for h in hours[j]:
                                for t in tasks:
                                    if ((p1, t, h, j) in x
                                            and (p2, t, h, j) in x):
                                        mdl.addConstr(
                                            x[p1, t, h, j]
                                            + x[p2, t, h, j] <= 1,
                                            name=f"C{cid}")
                                        enemy_meta[cid] = {
                                            "person1": p1, "person2": p2,
                                            "task": t, "hour": h, "day": j,
                                        }
                                        cid += 1

                # ── C4: max consecutive hours ─────────────────────────
                for person, config in max_consec_hours.items():
                    if person not in people_set:
                        continue
                    if isinstance(config, tuple):
                        max_work, min_rest = config
                    else:
                        max_work, min_rest = int(config), 1
                    if max_work <= 0 or min_rest <= 0:
                        continue
                    window_size = max_work + min_rest
                    for j in days:
                        hours_j = hours[j]
                        if len(hours_j) < window_size:
                            continue
                        for start in range(len(hours_j) - window_size + 1):
                            window_hrs = hours_j[start:start + window_size]
                            terms = [
                                x[person, t, h, j]
                                for t in tasks for h in window_hrs
                                if (person, t, h, j) in x
                            ]
                            if not terms:
                                continue
                            mdl.addConstr(
                                gp.quicksum(terms) <= max_work,
                                name=f"C{cid}")
                            consec_meta[cid] = {
                                "person":       person,
                                "day":          j,
                                "window_hours": list(window_hrs),
                                "max_work":     max_work,
                                "min_rest":     min_rest,
                            }
                            cid += 1

                mdl.setObjective(0, GRB.MINIMIZE)
                mdl.optimize()

                if mdl.Status != GRB.INFEASIBLE:
                    return True, {}

                # ── Phase 2: IIS ─────────────────────────────────────
                print("\n[PRESOLVE] INFEASIBLE — computing IIS...\n")
                mdl.computeIIS()
                mdl.write("presolve_iis.ilp")

                iis_cover_ids:  list[int] = []
                iis_person_ids: list[int] = []
                iis_enemy_ids:  list[int] = []
                iis_consec_ids: list[int] = []

                for c in mdl.getConstrs():
                    if not c.IISConstr:
                        continue
                    name = c.ConstrName
                    if not name.startswith("C"):
                        continue
                    try:
                        cid_val = int(name[1:])
                    except ValueError:
                        continue
                    if cid_val in cover_meta:
                        iis_cover_ids.append(cid_val)
                    elif cid_val in person_meta:
                        iis_person_ids.append(cid_val)
                    elif cid_val in enemy_meta:
                        iis_enemy_ids.append(cid_val)
                    elif cid_val in consec_meta:
                        iis_consec_ids.append(cid_val)

                # ── Build result dicts ────────────────────────────────

                # 1. Simple bottlenecks: slots where eligible < demand
                #    (independent of other constraints)
                simple_bottlenecks = _build_simple_bottlenecks(
                    eligible, demand, days, hours, tasks)

                # 2. IIS cover slots
                iis_cover_slots = []
                iis_cover_keys: set = set()  # (t, h, j) — used for lookup below
                for cid_val in sorted(iis_cover_ids):
                    info = cover_meta[cid_val]
                    iis_cover_slots.append(info.copy())
                    iis_cover_keys.add((info["task"], info["hour"], info["day"]))

                # 3. IIS person conflicts (deduplicated by person+hour+day)
                iis_person_conflicts = []
                seen_phj: set = set()
                for cid_val in iis_person_ids:
                    info = person_meta[cid_val]
                    key  = (info["person"], info["hour"], info["day"])
                    if key in seen_phj:
                        continue
                    seen_phj.add(key)
                    competing = [
                        t for t in tasks
                        if (t, info["hour"], info["day"]) in iis_cover_keys
                        and info["person"] in eligible.get(
                            (t, info["hour"], info["day"]), [])
                    ]
                    iis_person_conflicts.append({
                        "person":          info["person"],
                        "hour":            info["hour"],
                        "day":             info["day"],
                        "competing_tasks": competing,
                    })

                # 4. Per-hour demand vs capacity summary (only IIS hours)
                iis_hours: set = set()
                for cid_val in iis_cover_ids:
                    info = cover_meta[cid_val]
                    iis_hours.add((info["hour"], info["day"]))

                hour_groups = []
                for (h, j) in sorted(iis_hours, key=lambda k: (k[1], k[0])):
                    tasks_in_hour = []
                    total_demand  = 0
                    for t in tasks:
                        d = demand.get((t, h, j), 0)
                        if d <= 0:
                            continue
                        elig = eligible.get((t, h, j), [])
                        tasks_in_hour.append({
                            "task":     t,
                            "demand":   d,
                            "eligible": len(elig),
                            "names":    list(elig),
                        })
                        total_demand += d
                    # unique_eligible: people who are available AND have at
                    # least one skill needed at this hour
                    unique_eligible = {
                        p for t in tasks
                        for p in eligible.get((t, h, j), [])
                    }
                    # total_eligible: sum of eligible counts per task
                    # (a person covering 2 tasks counts twice here)
                    total_eligible = sum(e["eligible"] for e in tasks_in_hour)
                    hour_groups.append({
                        "hour":            h,
                        "day":             j,
                        "total_demand":    total_demand,
                        "total_eligible":  total_eligible,
                        "unique_eligible": len(unique_eligible),
                        "tasks":           tasks_in_hour,
                    })

                # 5. Enemy conflicts (deduplicated by person-pair + day)
                enemy_conflicts = []
                seen_enemy_pairs: set = set()
                for cid_val in iis_enemy_ids:
                    info = enemy_meta[cid_val]
                    key  = (info["person1"], info["person2"], info["day"])
                    if key in seen_enemy_pairs:
                        continue
                    seen_enemy_pairs.add(key)
                    blocked_slots = [
                        {"task": enemy_meta[c]["task"],
                         "hour": enemy_meta[c]["hour"]}
                        for c in iis_enemy_ids
                        if enemy_meta[c]["person1"] == info["person1"]
                        and enemy_meta[c]["person2"] == info["person2"]
                        and enemy_meta[c]["day"]     == info["day"]
                    ]
                    enemy_conflicts.append({
                        "person1":       info["person1"],
                        "person2":       info["person2"],
                        "day":           info["day"],
                        "blocked_slots": blocked_slots,
                    })

                # 6. Consec conflicts (each cid is already unique)
                consec_conflicts = []
                for cid_val in iis_consec_ids:
                    info = consec_meta[cid_val]
                    blocked_coverage = []
                    for h in info["window_hours"]:
                        for t in tasks:
                            d = demand.get((t, h, info["day"]), 0)
                            if d > 0 and info["person"] in eligible.get(
                                    (t, h, info["day"]), []):
                                blocked_coverage.append({"task": t, "hour": h})
                    consec_conflicts.append({
                        "person":           info["person"],
                        "day":              info["day"],
                        "window_hours":     info["window_hours"],
                        "max_work":         info["max_work"],
                        "min_rest":         info["min_rest"],
                        "blocked_coverage": blocked_coverage,
                    })

                return False, {
                    "simple_bottlenecks": simple_bottlenecks,
                    "iis": {
                        "cover_slots":      iis_cover_slots,
                        "person_conflicts": iis_person_conflicts,
                        "hour_groups":      hour_groups,
                        "enemy_conflicts":  enemy_conflicts,
                        "consec_conflicts": consec_conflicts,
                    },
                }

    except gp.GurobiError as exc:
        print(f"[PRESOLVE] GurobiError: {exc}")
        problems = _build_simple_bottlenecks(eligible, demand, days, hours, tasks)
        return len(problems) == 0, {"simple_bottlenecks": problems, "iis": {}}


def _build_simple_bottlenecks(eligible, demand, days, hours, tasks) -> list:
    """Slots where eligible count < demand, regardless of other constraints."""
    out = []
    for j in days:
        for h in hours[j]:
            for t in tasks:
                d = demand.get((t, h, j), 0)
                if d <= 0:
                    continue
                elig = eligible.get((t, h, j), [])
                if len(elig) < d:
                    out.append({
                        "task":     t,
                        "hour":     h,
                        "day":      j,
                        "required": d,
                        "eligible": len(elig),
                        "names":    list(elig),
                    })
    return out