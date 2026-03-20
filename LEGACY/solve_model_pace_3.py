import time
import gurobipy as gp
from gurobipy import GRB


def solve_model(data, ui_update_callback=None, active_model_ref=None):

    # ═════════════════════════════════════════════════════════════════
    # 0. UNPACK INPUT DATA
    #    X_prev  = Anchor Plan (§2 / §J)
    # ═════════════════════════════════════════════════════════════════

    people    = data["people"];     tasks        = data["tasks"]
    hours     = data["hours"];      days         = data["days"]
    demand    = data["demand"];     availability = data["availability"]
    emergency = data["emergency"]
    skills    = data["skills"];     force        = data["force"]
    social    = data["social"];     min_quota    = data["min_quota"]
    rotation  = data["rotation"];   pref_cost    = data["pref_cost"]
    X_prev    = data["X_prev"]

    enforced_rest    = data.get("enforced_rest", False)
    # max_consec_hours: maximum consecutive working hours (model §L uses per-person
    # I_p; current implementation uses a single global value for all people).
    max_consec_hours = data.get("max_consec_hours", None)
    captains         = data.get("captains", [])
    hard_enemies     = data.get("hard_enemies", False)
    day_heuristics   = data.get("day_heuristics", 0)   # 0=coupled, 1=pacing

    W = data["weights"]
    W_COVERAGE  = W["W_COVERAGE"];  W_FORCE     = W["W_FORCE"]
    W_CAPTAIN   = W["W_CAPTAIN"];   W_EMERG     = W["W_EMERG"]
    W_STABILITY = W["W_STABILITY"]; W_EQ_DAY    = W["W_EQ_DAY"]
    W_EQ_GLOBAL = W["W_EQ_GLOBAL"]; W_ROTATION  = W["W_ROTATION"]
    W_SOCIAL    = W["W_SOCIAL"];    W_GAP       = W["W_GAP"]
    W_QUOTA     = W["W_QUOTA"];     W_PREF      = W["W_PREF"]

    # ═════════════════════════════════════════════════════════════════
    # 1. LOOKUPS GLOBALES
    #    Estructuras derivadas de los parámetros que se reutilizan
    #    a lo largo de toda la función (sets de pertenencia rápida).
    #    Se calculan UNA SOLA VEZ sobre el conjunto completo de días.
    # ═════════════════════════════════════════════════════════════════

    # Conjunto de (persona, hora, día) donde A_{p,h,d} = 1
    avail_set = {(p, h, d)
                 for (p, h, d), v in availability.items()
                 if v == 1}

    # Conjunto de (persona, tarea) donde S_{p,t} = 1
    skill_set = {(p, t)
                 for (p, t), v in skills.items()
                 if v == 1}

    # Mapa (hora, día) → hora siguiente en ese mismo día.
    # Se usa en las constraints de rotación y en la extracción de resultados.
    h_next = {(hours[d][i], d): hours[d][i + 1]
              for d in days
              for i in range(len(hours[d]) - 1)}

    # quota_keys: triples (p, t, d) con min_quota_{p,t,d} > 0.
    quota_keys = [(p, t, d)
                  for p in people for t in tasks for d in days
                  if min_quota.get((p, t, d), 0) > 0]

    # ═════════════════════════════════════════════════════════════════
    # 1b. PREPARACIÓN DE LA HEURÍSTICA DE DÍAS (PACING METHOD)
    #     Solo cuando day_heuristics == 1.
    #     Se calcula el ritmo ideal global (horas/persona/día) y se
    #     inicializan los acumuladores de horas históricas por persona.
    # ═════════════════════════════════════════════════════════════════

    if day_heuristics == 1:
        total_demand = sum(demand.get((t, h, d), 0)
                           for t in tasks for d in days for h in hours[d])
        people_global = [p for p in people
                         if any((p, h, d) in avail_set
                                for d in days for h in hours[d])]
        n_p  = len(people_global) or 1
        n_d  = len(days) or 1
        pace = total_demand / (n_p * n_d)   # horas objetivo/persona/día
        H_p  = {p: 0 for p in people}       # horas acumuladas antes del día actual
        days_iterator = [[d] for d in days] # un submodelo por día
    else:
        days_iterator = [days]              # un único modelo global

    # ─────────────────────────────────────────────────────────────────
    # Acumuladores de resultados a través de las iteraciones del bucle.
    # En modo acoplado (day_heuristics=0) el bucle se ejecuta una sola
    # vez y estos acumuladores son equivalentes a las variables del modelo.
    # ─────────────────────────────────────────────────────────────────
    all_x_vals = {}    # (p, t, h, d) → float
    all_m_vals = {}    # (t, h, d)    → float
    all_g_vals = {}    # (p, h, d)    → float

    # Asignación parcial visible para el callback en tiempo real.
    # Se rellena con None al principio y se va actualizando día a día.
    partial_assignment = {d: {p: {h: None for h in hours[d]}
                               for p in people} for d in days}

    final_status  = "Optimal"
    final_mip_gap = 0.0

    # Throttle del UI: fuera del bucle para que la cadencia sea global.
    last_ui_update = [0.0]

    # Mapa de estados Gurobi → texto (se usa dentro y fuera del bucle).
    status_map = {GRB.OPTIMAL:     "Optimal",
                  GRB.TIME_LIMIT:  "Time Limit Reached",
                  GRB.INFEASIBLE:  "Infeasible",
                  GRB.INTERRUPTED: "Interrupted by User"}

    # ═════════════════════════════════════════════════════════════════
    # BUCLE PRINCIPAL
    #   day_heuristics=0 → una sola iteración con todos los días.
    #   day_heuristics=1 → una iteración por día (submodelos desacoplados).
    # ═════════════════════════════════════════════════════════════════

    solve_start = time.monotonic()

    for loop_idx, current_days in enumerate(days_iterator):

        current_days_set = set(current_days)

        # Subconjunto de avail_set restringido a los días de esta iteración.
        # Se usa para las variables g y la constraint anti-ubicuidad.
        avail_set_day = {(p, h, d) for (p, h, d) in avail_set
                         if d in current_days_set}

        # FIX 4: people_with_avail se calcula aquí, antes de la creación de
        # variables, para poder restringir w_delta y delta a este subconjunto
        # y evitar variables sin constraint que el solver resuelve trivialmente.
        people_with_avail = [p for p in people
                             if any((p, h, d) in avail_set_day
                                    for d in current_days for h in hours[d])]

        # ─────────────────────────────────────────────────────────────
        # 2. MODELO GUROBI
        # ─────────────────────────────────────────────────────────────

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1)
        env.start()
        model = gp.Model("StaffScheduler", env=env)

        # P1 — FIX: terminar el modelo anterior Y registrar el nuevo.
        # El código original nunca asignaba active_model_ref[0] = model,
        # por lo que el nuevo modelo nunca podía ser interrumpido desde fuera.
        # Además, no comprobaba que active_model_ref[0] no fuera None antes
        # de llamar a .terminate(), lo que causaba AttributeError en la primera
        # ejecución.
        if active_model_ref is not None:
            if active_model_ref[0] is not None:
                active_model_ref[0].terminate()
            active_model_ref[0] = model

        # ─────────────────────────────────────────────────────────────
        # 3. VARIABLES DE DECISIÓN
        #    Notación alineada con el modelo matemático:
        #      x          → asignación principal        (§3)
        #      m          → missing staff               (§3 / §A)
        #      u          → unfulfilled mandate         (§3 / §B)
        #      z          → deviation / stability       (§3 / §J)
        #      q          → quota missed                (§3 / §I)
        #      g          → work restart                (§3 / §K)
        #      r          → restarts excess / real gaps (§3 / §K)
        #      j_max/min  → daily equity bounds         (§3 / §F.1)
        #      w_delta_+/-→ global equity deviation     (§3 / §F.2 coupled)
        #      delta_+/-  → pacing deviation            (§3 / §F.2 pacing)
        #      c          → consecutive rotation penalty(§3 / §G)
        #      f          → separated friends           (§3 / §H)
        #      e          → enemies together            (§3 / §H)
        #      k          → missing captain             (§3 / §C)
        # ─────────────────────────────────────────────────────────────

        # ── x: asignación principal ──────────────────────────────────────
        # x[p, t, h, d] = 1 si la persona p realiza la tarea t en la hora h
        # del día d. Solo existe donde A_{p,h,d}=1 y S_{p,t}=1.
        x_set = {(p, t, h, d)
                 for p in people for t in tasks
                 for d in current_days   for h in hours[d]
                 if (p, t) in skill_set and (p, h, d) in avail_set}

        x = model.addVars(x_set,
                          vtype=GRB.BINARY,
                          name="assignment")

        # ── m: cobertura missing ─────────────────────────────────────────
        # m[t, h, d] = número de puestos sin cubrir de la tarea t en (h, d).
        # Solo existe para combinaciones con D_{t,h,d} > 0.
        m_keys = [(t, h, d)
                  for t in tasks
                  for d in current_days
                  for h in hours[d]
                  if demand.get((t, h, d), 0) > 0]

        m = model.addVars(m_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="missing_staff")

        # ── u: mandatos incumplidos ──────────────────────────────────────
        # u[p, t, h, d] = 1 si se ignoró el mandato F_{p,t,h,d}=1.
        force_keys = [(p, t, h, d)
                      for (p, t, h, d), v in force.items()
                      if v == 1 and d in current_days_set]

        u = model.addVars(force_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="unfulfilled_mandate")

        # ── z: desviación respecto al plan anterior (X^PREV) ─────────────
        # z[p, t, h, d] = |x_nuevo - X^PREV_{p,t,h,d}|.
        z = model.addVars(x_set,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="deviation")

        # ── q: cuota de experiencia no alcanzada ─────────────────────────
        # q[p, t, d] = déficit respecto a L_{p,t,d}.
        # Indexado por día: cada día tiene su propia cuota objetivo.
        quota_ptd_keys = [(p, t, d)
                          for p, t, d in quota_keys
                          if d in current_days_set]

        q = model.addVars(quota_ptd_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="quota_missed")

        # ── g: reinicios de bloque de trabajo ────────────────────────────
        # g[p, h, d] = 1 si la persona p comienza un nuevo bloque en h, d.
        g = model.addVars(avail_set_day,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="work_restart")

        # ── r: exceso de reinicios / huecos reales (§K, §3) ──────────────
        # r[p, d] = max(0, nº_bloques[p,d] - 1).
        # Penalizar r garantiza que solo se castiga la fragmentación genuina
        # (trabajar, parar, volver a trabajar), sin penalizar a personas que
        # simplemente empiezan su turno después de la primera hora.
        r_keys = [(p, d)
                  for p in people for d in current_days
                  if any((p, h, d) in avail_set_day for h in hours[d])]

        r = model.addVars(r_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="restarts_excess")

        # ── j_max / j_min: cotas de equidad diaria (§F.1) ────────────────
        # j_max[d] y j_min[d] son el máximo y mínimo de horas trabajadas
        # por cualquier persona el día d. El solver minimiza su diferencia.
        max_hours_any_day = max((len(hours[d]) for d in current_days), default=0)

        j_max = model.addVars(current_days,
                              lb=0,
                              ub=max_hours_any_day,
                              vtype=GRB.CONTINUOUS,
                              name="j_max")

        j_min = model.addVars(current_days,
                              lb=0,
                              ub=max_hours_any_day,
                              vtype=GRB.CONTINUOUS,
                              name="j_min")

        # ── w_delta (acoplado) O delta (heurístico) ───────────────────────
        # Modo acoplado: desviación por-persona respecto al reparto ideal.
        #   w_delta_plus[p]  = horas asignadas por encima del objetivo
        #   w_delta_minus[p] = horas asignadas por debajo del objetivo
        # Modo heurístico: desviación respecto al ritmo ideal acumulado (§F.2).
        #   delta_plus[p]  = δ⁺_p (adelanto respecto al ritmo)
        #   delta_minus[p] = δ⁻_p (retraso respecto al ritmo)
        # FIX 4: ambas familias se crean solo para people_with_avail.
        if day_heuristics == 0:
            w_delta_plus  = model.addVars(people_with_avail,
                                          lb=0,
                                          vtype=GRB.CONTINUOUS,
                                          name="eq_global_over")
            w_delta_minus = model.addVars(people_with_avail,
                                          lb=0,
                                          vtype=GRB.CONTINUOUS,
                                          name="eq_global_under")
        else:
            # δ⁺_p / δ⁻_p — pacing deviation (§F.2 heuristic mode)
            delta_plus  = model.addVars(people_with_avail,
                                        lb=0,
                                        vtype=GRB.CONTINUOUS,
                                        name="delta_plus")
            delta_minus = model.addVars(people_with_avail,
                                        lb=0,
                                        vtype=GRB.CONTINUOUS,
                                        name="delta_minus")

        # ── c: penalización por horas consecutivas en tareas de rotación ─
        # c[p, t, h, d] = 1 si R_t=1 y la persona trabaja t en h y en h+1.
        consec_keys = [(p, t, h, d)
                       for p in people for t in tasks
                       for d in current_days   for h in hours[d][:-1]
                       if rotation.get(t, 0) == 1
                       and (p, t, h,              d) in x_set
                       and (h, d)                    in h_next
                       and (p, t, h_next[(h, d)], d) in x_set]

        c = model.addVars(consec_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="consecutive_rotation")

        # ── f: separación de amigos (Soc=1) ─────────────────────────────
        # f[p1, p2, t, h, d] = |x[p1] - x[p2]| en el slot (t, h, d).
        friends_keys = [(p1, p2, t, h, d)
                        for (p1, p2), sv in social.items() if sv == 1
                        for t in tasks
                        for d in current_days for h in hours[d]
                        if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]

        f = model.addVars(friends_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="separated_friends")

        # ── e: conflicto entre enemigos (Soc=-1) ─────────────────────────
        # e[p1, p2, t, h, d] = 1 si ambos enemigos trabajan el mismo slot.
        # Si hard_enemies=True, se usa constraint dura y enemies_keys queda
        # vacío; e es un tupledict vacío y no contribuye al objetivo.
        # P5 — e y k se crean siempre para eliminar if de guardia dispersos.
        enemies_scope = [(p1, p2, t, h, d)
                         for (p1, p2), sv in social.items() if sv == -1
                         for t in tasks
                         for d in current_days for h in hours[d]
                         if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]

        enemies_keys = [] if hard_enemies else enemies_scope

        e = model.addVars(enemies_keys,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="enemies_together")

        # ── k: capitán ausente ───────────────────────────────────────────
        # k[h, d] = 1 si ningún capitán está de guardia en una hora activa.
        # P5 — active_hd y k se crean siempre; si no hay capitanes ambos
        # quedan vacíos y no contribuyen al objetivo ni generan constraints.
        active_hd = [(h, d)
                     for d in current_days for h in hours[d]
                     if sum(demand.get((t, h, d), 0) for t in tasks) > 0
                     ] if captains else []

        k = model.addVars(active_hd,
                          lb=0,
                          vtype=GRB.CONTINUOUS,
                          name="missing_captain")

        # ─────────────────────────────────────────────────────────────
        # 4. FUNCIÓN OBJETIVO (§4)
        #    Minimizar la suma ponderada de todas las penalizaciones.
        #    Términos ordenados de mayor a menor coste (prioridad):
        #    Cobertura > Mandatos > Capitanes > Emergencias > Estabilidad >
        #    Equidad diaria > Equidad global > Rotación > Social >
        #    Huecos > Cuota > Preferencias
        # ─────────────────────────────────────────────────────────────

        obj = gp.LinExpr()

        # W_COVERAGE · Σ m_{t,h,d}  — puestos sin cubrir
        obj += W_COVERAGE * m.sum()

        # W_FORCE · Σ u_{p,t,h,d}  — mandatos incumplidos
        obj += W_FORCE * u.sum()

        # W_CAPTAIN · Σ k_{h,d}  — horas sin capitán (0 si k vacío)
        obj += W_CAPTAIN * k.sum()

        # W_EMERG · Σ E_{p,h,d} · x_{p,t,h,d}  — llamadas de emergencia
        obj += W_EMERG * gp.quicksum(
            emergency.get((p, h, d), 0) * x[p, t, h, d]
            for p, t, h, d in x_set)

        # W_STABILITY · Σ z_{p,t,h,d}  — cambios sobre el plan publicado
        obj += W_STABILITY * z.sum()

        # W_EQ_DAY · Σ_d (j_max_d - j_min_d)  — equidad diaria (§F.1)
        obj += W_EQ_DAY * gp.quicksum(j_max[d] - j_min[d] for d in current_days)

        # W_EQ_GLOBAL  — equidad global: desviación individual respecto al
        # reparto ideal (acoplado) o respecto al ritmo acumulado (heurístico).
        if day_heuristics == 0:
            obj += W_EQ_GLOBAL * (w_delta_plus.sum() + w_delta_minus.sum())
        else:
            obj += W_EQ_GLOBAL * (delta_plus.sum() + delta_minus.sum())

        # W_ROTATION · Σ c_{p,t,h,d}  — fatiga por rotación (0 si c vacío)
        obj += W_ROTATION * c.sum()

        # W_SOCIAL · Σ (f + e)  — conflictos sociales (0 si vacíos)
        obj += W_SOCIAL * f.sum()
        obj += W_SOCIAL * e.sum()

        # W_GAP · Σ r_{p,d}  — huecos reales en bloques de trabajo (§K).
        # Solo se penalizan bloques en exceso de 1: un turno continuo,
        # independientemente de a qué hora empiece, cuesta 0.
        obj += W_GAP * r.sum()

        # W_QUOTA · Σ q_{p,t,d}  — cuotas L_{p,t,d} no alcanzadas
        obj += W_QUOTA * q.sum()

        # W_PREF · Σ B_{p,t} · x_{p,t,h,d}  — coste de preferencias
        obj += W_PREF * gp.quicksum(
            pref_cost.get((p, t), 0) * x[p, t, h, d]
            for p, t, h, d in x_set)

        model.setObjective(obj, GRB.MINIMIZE)

        # ─────────────────────────────────────────────────────────────
        # 5. CONSTRAINTS
        # ─────────────────────────────────────────────────────────────

        # ── A. Cobertura de demanda (§A) ──────────────────────────────────
        # Σ_p x_{p,t,h,d} + m_{t,h,d} = D_{t,h,d}  ∀ t,h,d con D>0
        model.addConstrs(
            (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
             for t, h, d in m_keys),
            name="coverage")

        # Slots sin demanda: ninguna persona puede ser asignada.
        zero_demand_keys = [(t, h, d)
                            for t in tasks for d in current_days for h in hours[d]
                            if demand.get((t, h, d), 0) == 0]

        model.addConstrs(
            (x.sum('*', t, h, d) == 0
             for t, h, d in zero_demand_keys),
            name="zero_demand")

        # ── B. Mandatos individuales (§B) ─────────────────────────────────
        # Soft: 1 - x_{p,t,h,d} ≤ u_{p,t,h,d}  si A_{p,h,d}=1
        model.addConstrs(
            (1 - x[p, t, h, d] <= u[p, t, h, d]
             for p, t, h, d in force_keys if (p, t, h, d) in x_set),
            name="mandate_valid")

        # Hard: u_{p,t,h,d} ≥ 1  si A_{p,h,d}=0 (slot imposible)
        model.addConstrs(
            (u[p, t, h, d] >= 1
             for p, t, h, d in force_keys if (p, t, h, d) not in x_set),
            name="mandate_impossible")

        # ── C. Presencia de capitán (§C) ──────────────────────────────────
        # Σ_{p∈C} Σ_t x_{p,t,h,d} + k_{h,d} ≥ 1  ∀ h,d con demanda>0
        model.addConstrs(
            (gp.quicksum(x.sum(p, '*', h, d) for p in captains) + k[h, d] >= 1
             for h, d in active_hd),
            name="captain_presence")

        # ── D. Anti-ubicuidad / disponibilidad física (§D) ───────────────
        # Σ_t x_{p,t,h,d} ≤ 1  ∀ p,h,d con A_{p,h,d}=1
        model.addConstrs(
            (x.sum(p, '*', h, d) <= 1
             for p, h, d in avail_set_day),
            name="anti_ubiquity")

        # ── F.1. Equidad diaria (§F.1) ────────────────────────────────────
        # Σ_{t,h} x_{p,t,h,d} ≤ j_max_d  y  ≥ j_min_d  ∀ p,d disponibles
        eq_day_keys = [(p, d)
                       for d in current_days for p in people
                       if any((p, h, d) in avail_set_day for h in hours[d])]

        model.addConstrs(
            (x.sum(p, '*', '*', d) <= j_max[d]
             for p, d in eq_day_keys),
            name="eq_day_max")

        model.addConstrs(
            (x.sum(p, '*', '*', d) >= j_min[d]
             for p, d in eq_day_keys),
            name="eq_day_min")

        # ── F.2. Equidad global (§F.2) ────────────────────────────────────
        # Modo acoplado: desviación por-persona respecto al reparto ideal
        # (total slots de demanda / personas disponibles).
        # Modo heurístico: constraint de pacing δ⁺_p, δ⁻_p respecto a T_d.
        if day_heuristics == 0:
            total_demand_slots = sum(
                demand.get((t, h, d), 0)
                for t in tasks for d in current_days for h in hours[d])
            n_avail   = len(people_with_avail) or 1
            w_target  = total_demand_slots / n_avail

            model.addConstrs(
                (  x.sum(p, '*', '*', '*')
                 - w_delta_plus[p]
                 + w_delta_minus[p] == w_target
                 for p in people_with_avail),
                name="eq_global_target")
        else:
            # H_p + Σ_t x_{p,t,h,today} - δ⁺_p + δ⁻_p = T_d
            T_j = pace * (loop_idx + 1)
            model.addConstrs(
                (  H_p[p]
                 + x.sum(p, '*', '*', current_days[0])
                 - delta_plus[p]
                 + delta_minus[p] == T_j
                 for p in people_with_avail),
                name="pacing")

        # ── G. Fatiga por rotación (§G) ───────────────────────────────────
        # x_{p,t,h,d} + x_{p,t,h+1,d} - c_{p,t,h,d} ≤ 1  donde R_t=1
        model.addConstrs(
            (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
             for p, t, h, d in consec_keys),
            name="rotation_fatigue")

        # ── H. Social — Amigos (Soc=1) ────────────────────────────────────
        # f = |x[p1] - x[p2]|, linealizado con dos constraints.
        model.addConstrs(
            (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
             for p1, p2, t, h, d in friends_keys),
            name="friends_right")

        model.addConstrs(
            (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
             for p1, p2, t, h, d in friends_keys),
            name="friends_left")

        # ── H. Social — Enemigos (Soc=-1) ─────────────────────────────────
        if hard_enemies:
            # x_{p1,t,h,d} + x_{p2,t,h,d} ≤ 1  (restricción dura)
            model.addConstrs(
                (x[p1, t, h, d] + x[p2, t, h, d] <= 1
                 for p1, p2, t, h, d in enemies_scope),
                name="enemies_hard")
        else:
            # x_{p1,t,h,d} + x_{p2,t,h,d} - e_{p1,p2,t,h,d} ≤ 1
            model.addConstrs(
                (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
                 for p1, p2, t, h, d in enemies_keys),
                name="enemies_soft")

        # ── I. Cuota mínima de experiencia L_{p,t,d} (§I) ────────────────
        # Σ_h x_{p,t,h,d} + q_{p,t,d} ≥ target_{p,t,d}
        # donde target = min(L_{p,t,d}, |H_d|)
        quota_constr_keys = [(p, t, d, target)
                             for p, t, d in quota_keys
                             if d in current_days_set
                             if (target := min(min_quota.get((p, t, d), 0), len(hours[d]))) > 0]

        model.addConstrs(
            (x.sum(p, t, '*', d) + q[p, t, d] >= target
             for p, t, d, target in quota_constr_keys),
            name="min_quota")

        # ── J. Estabilidad / Desviación del plan X^PREV (§J) ─────────────
        # z_{p,t,h,d} ≥ X^PREV - x  y  z_{p,t,h,d} ≥ x - X^PREV
        model.addConstrs(
            (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
             for p, t, h, d in x_set),
            name="stability_right")

        model.addConstrs(
            (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
             for p, t, h, d in x_set),
            name="stability_left")

        # ── K.1. Detección de reinicios de bloque g_{p,h,d} (§K) ─────────
        # Primera hora del día: g_{p,h0,d} = Σ_t x_{p,t,h0,d}
        # Resto de horas:       g_{p,h,d}  ≥ Σ_t x_{p,t,h,d} - Σ_t x_{p,t,h-1,d}
        #
        # IMPORTANTE: la primera hora NO se penaliza en el objetivo.
        # Un turno continuo independientemente de cuándo empiece cuesta 0;
        # sin este ajuste, el solver concentraría la carga en el mínimo número
        # de personas para evitar el coste de g en la primera hora.
        gap_first_keys = [(p, d)
                          for p in people for d in current_days
                          if (p, hours[d][0], d) in avail_set_day]

        model.addConstrs(
            (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
             for p, d in gap_first_keys),
            name="restart_first_hour")

        gap_rest_keys = [(p, hours[d][i], hours[d][i - 1], d)
                         for p in people for d in current_days
                         for i in range(1, len(hours[d]))
                         if (p, hours[d][i], d) in avail_set_day]

        model.addConstrs(
            (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
             for p, h_curr, h_prev, d in gap_rest_keys),
            name="restart_subsequent_hours")

        # ── K.2. Linking r_{p,d} con el número de bloques (§K, Fórmula 3) ─
        # r_{p,d} ≥ (Σ_h g_{p,h,d}) - 1
        # Con lb=0: r_{p,d} = max(0, bloques - 1)  → ReLU del exceso.
        model.addConstrs(
            (r[p, d] >= gp.quicksum(
                g[p, h, d] for h in hours[d] if (p, h, d) in avail_set_day)
            - 1
            for p, d in r_keys),
            name="restarts_excess")

        # ── L. Descanso obligatorio — ventana deslizante (§L) ────────────
        # Σ_{τ=h}^{h+max_consec_hours} x_{p,t,τ,d} ≤ max_consec_hours  ∀ p, d, h
        # Nota: el modelo define I_p por persona; aquí se usa un valor
        # global max_consec_hours aplicado a todos (limitación actual).
        if enforced_rest and max_consec_hours is not None:
            rest_keys = [(p, d, i)
                         for d in current_days for p in people
                         if any((p, h, d) in avail_set_day for h in hours[d])
                         for i in range(len(hours[d]) - max_consec_hours)]

            model.addConstrs(
                (gp.quicksum(x.sum(p, '*', tau, d) for tau in hours[d][i:i + max_consec_hours + 1]) <= max_consec_hours
                 for p, d, i in rest_keys),
                name="enforced_rest")

        # ─────────────────────────────────────────────────────────────
        # 6. CONFIGURACIÓN DEL SOLVER Y OPTIMIZACIÓN
        # ─────────────────────────────────────────────────────────────

        model._x     = x
        model._x_set = x_set

        for param_name, param_value in data.get("solver_params", {}).items():
            try:
                model.setParam(param_name, param_value)
            except Exception as err:
                print(f"Warning: Could not set param {param_name}={param_value}: {err}")

        # Throttle: evita saturar el UI enviando como máximo una actualización
        # cada 0.5 s aunque el solver encuentre muchas soluciones seguidas.
        # El callback se define dentro del bucle para capturar current_days
        # por clausura; last_ui_update vive fuera para que la cadencia sea global.
        def intermediate_solution_callback(mdl, where):
            """Envía al UI cada nueva solución entera mejorada que encuentra el solver.
            Muestra los días ya resueltos (fijos) más el día actual en tiempo real."""
            if where == GRB.Callback.MIPSOL and ui_update_callback:
                now = time.monotonic()
                if now - last_ui_update[0] < 0.5:
                    return
                last_ui_update[0] = now
                try:
                    x_vals = mdl.cbGetSolution(mdl._x)
                    # Partir de los días ya resueltos en iteraciones anteriores
                    temp_assignment = {d: {p: dict(partial_assignment[d][p])
                                           for p in people} for d in days}
                    # Superponer el día actual con los valores live del solver
                    for d in current_days:
                        for p in people:
                            for h in hours[d]:
                                temp_assignment[d][p][h] = next(
                                    (t for t in tasks
                                     if (p, t, h, d) in mdl._x_set
                                     and x_vals[p, t, h, d] > 0.5),
                                    None)
                    ui_update_callback({"status":     "Solving (New Best Found)...",
                                        "assignment": temp_assignment})
                except Exception:
                    pass

        model.optimize(intermediate_solution_callback)

        # ─────────────────────────────────────────────────────────────
        # Acumular resultados de esta iteración y preparar la siguiente
        # ─────────────────────────────────────────────────────────────

        iter_status = status_map.get(model.Status, f"Status Code: {model.Status}")

        if model.SolCount == 0:
            # Sin solución para este submodelo: si ya hay días previos
            # resueltos devolvemos lo que hay; si no, es un fallo real.
            if all_x_vals:
                final_status = iter_status
                break
            else:
                raise Exception("No feasible solution was found before stopping/timeout.")

        # Hay al menos una solución: propagar el peor estado encontrado
        if iter_status != "Optimal":
            final_status = iter_status

        final_mip_gap = model.MIPGap

        # Acumular variables de decisión en los diccionarios globales
        all_x_vals.update({key: x[key].X for key in x_set})
        all_m_vals.update({key: m[key].X for key in m_keys})
        all_g_vals.update({key: g[key].X for key in avail_set_day})

        # Actualizar partial_assignment con los días recién resueltos
        # (disponible para el callback de la siguiente iteración)
        for d in current_days:
            for p in people:
                for h in hours[d]:
                    partial_assignment[d][p][h] = next(
                        (t for t in tasks
                         if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5),
                        None)

        # Actualizar H_p para la siguiente iteración (solo modo heurístico)
        if day_heuristics == 1:
            d_cur = current_days[0]
            for p in people:
                H_p[p] += sum(1 for h in hours[d_cur] for t in tasks
                              if (p, t, h, d_cur) in x_set
                              and x[p, t, h, d_cur].X > 0.5)

        # Si el solver fue interrumpido, detener el bucle y devolver lo acumulado
        if model.Status == GRB.INTERRUPTED:
            break

    # ═════════════════════════════════════════════════════════════════
    # 7. EXTRACCIÓN DE RESULTADOS
    #    Se usan los acumuladores all_x_vals / all_m_vals / all_g_vals
    #    en lugar de las variables del modelo (que solo cubren la última
    #    iteración). En modo acoplado el resultado es idéntico al original.
    # ═════════════════════════════════════════════════════════════════

    solve_elapsed = time.monotonic() - solve_start

    sol = {"solve_time":    solve_elapsed,
           "status":        final_status,
           "mip_gap":       final_mip_gap,
           "enforced_rest": enforced_rest}

    # ── Cuadrícula de asignaciones ────────────────────────────────────
    sol["assignment"] = {
        d: {
            p: {
                h: next(
                    (t for t in tasks
                     if all_x_vals.get((p, t, h, d), 0) > 0.5),
                    None)
                for h in hours[d]}
            for p in people}
        for d in days}

    # ── Puestos sin cubrir (m_{t,h,d} > 0) ───────────────────────────
    # Se itera directamente sobre el dict acumulado: evita el triple for
    # sobre tasks × days × hours y el .get() redundante.
    sol["missing"] = [f"{t} @ {h}, {d}: {v:.0f} missing"
                      for (t, h, d), v in all_m_vals.items()
                      if v > 0.01]

    # ── Carga de trabajo por persona ──────────────────────────────────
    assigned_counts = {p: 0 for p in people}
    for p, t, h, d in all_x_vals:
        if all_x_vals[p, t, h, d] > 0.5:
            assigned_counts[p] += 1

    sol["workload"] = assigned_counts

    # w_max / w_min: siempre a posteriori desde los conteos reales.
    wl = list(assigned_counts.values())
    sol["w_max"] = float(max(wl)) if wl else 0.0
    sol["w_min"] = float(min(wl)) if wl else 0.0

    # ── Cumplimiento de cuotas L_{p,t,d} ─────────────────────────────
    # Se reporta el cumplimiento por persona, tarea y día de forma
    # independiente, alineado con la indexación diaria de L_{p,t,d}.
    sol["quota_issues"] = []
    for p, t, d in quota_keys:
        demanded = min_quota[p, t, d]
        assigned = sum(1 for p2, t2, h2, d2 in all_x_vals
                       if p2 == p and t2 == t and d2 == d
                       and all_x_vals[p2, t2, h2, d2] > 0.5)
        status = "OK" if assigned >= demanded else "SHORTFALL"
        sol["quota_issues"].append(
            f"{status}: {p} — task '{t}' on {d}: {assigned}/{demanded} h assigned")

    # ── Huecos en los bloques de trabajo (r_{p,d} > 0) ───────────────
    # Se usa all_g_vals para reconstruir el número de bloques por persona
    # y día (equivalente al valor de r_{p,d} resuelto por el modelo).
    gaps = []
    for d in days:
        day_gaps = [f"  • {p}: {starts} blocks (Starts: {', '.join(hs)})"
                    for p in people
                    for starts, hs in [(
                        sum(1 for h in hours[d] if (p, h, d) in avail_set and all_g_vals.get((p, h, d), 0) > 0.5),
                        [h for h in hours[d] if (p, h, d) in avail_set and all_g_vals.get((p, h, d), 0) > 0.5])]
                    if starts > 1]
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # ── Incidencias sociales ──────────────────────────────────────────
    # Se pre-construye el set de slots activos para evitar el
    # cuádruple for anidado original O(parejas × tasks × days × hours).
    # active_slots se reutiliza también en emerg_issues, rot_violations
    # y la comprobación de enemigos y capitanes.
    soc_issues = []
    active_slots = {(p, t, h, d) for (p, t, h, d), v in all_x_vals.items() if v > 0.5}

    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for p1b, t, h, d in active_slots:
            if p1b == p1 and (p2, t, h, d) in active_slots:
                soc_issues.append(
                    f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    if any(sv == -1 for sv in social.values()):
        # P5 — enemies: se reconstruye el scope global desde all_x_vals.
        if not hard_enemies:
            violations = [f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                          for (p1, p2), sv in social.items() if sv == -1
                          for p1b, t, h, d in active_slots
                          if p1b == p1 and (p2, t, h, d) in active_slots]
        else:
            violations = []
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")

    sol["social_issues"] = soc_issues

    # ── Incidencias de capitanes ──────────────────────────────────────
    # P5 — k siempre existe: aquí se deriva directamente de all_x_vals.
    all_active_hd = [(h, d)
                     for d in days for h in hours[d]
                     if sum(demand.get((t, h, d), 0) for t in tasks) > 0
                     ] if captains else []
    missing_caps = [f"MISSING CAPTAIN @ {h}, {d}"
                    for h, d in all_active_hd
                    if not any((cap, t, h, d) in active_slots
                               for cap in captains for t in tasks)]
    sol["captain_issues"] = (missing_caps or
                             ["SUCCESS: All active hours have at least one captain on duty."]
                             if all_active_hd else [])

    # ── Llamadas de emergencia (E_{p,h,d}=1) ─────────────────────────
    sol["emerg_issues"] = [f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {d}"
                           for p, t, h, d in active_slots
                           if emergency.get((p, h, d), 0) == 1]

    # ── Violaciones de rotación (R_t=1) ──────────────────────────────
    rot_violations = [f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next[(h, d)]}, {d}"
                      for p, t, h, d in active_slots
                      if rotation.get(t, 0) == 1
                      and (h, d) in h_next
                      and (p, t, h_next[(h, d)], d) in active_slots]
    sol["rotation_issues"] = (rot_violations or
                              ["SUCCESS: No consecutive hours on rotation tasks."])

    return sol