import time
import gurobipy as gp
from gurobipy import GRB


def solve_model(data, ui_update_callback=None, active_model_ref=None):

    # ═════════════════════════════════════════════════════════════════
    # 0. UNPACK INPUT DATA
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
    max_consec_hours = data.get("max_consec_hours", None)
    captains         = data.get("captains", [])
    hard_enemies     = data.get("hard_enemies", False)

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
    # ═════════════════════════════════════════════════════════════════

    # Conjunto de (persona, hora, día) donde la persona está disponible
    avail_set = {(p, h, d)
                 for (p, h, d), v in availability.items()
                 if v == 1}

    # Conjunto de (persona, tarea) donde la persona tiene la habilidad requerida
    skill_set = {(p, t)
                 for (p, t), v in skills.items()
                 if v == 1}

    # Mapa (hora, día) → hora siguiente en ese mismo día.
    # Se usa en las constraints de rotación y en la extracción de resultados.
    h_next = {(hours[d][i], d): hours[d][i + 1]
              for d in days
              for i in range(len(hours[d]) - 1)}

    # ═════════════════════════════════════════════════════════════════
    # 2. MODELO GUROBI
    # ═════════════════════════════════════════════════════════════════

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

    # ═════════════════════════════════════════════════════════════════
    # 3. VARIABLES DE DECISIÓN
    #    Cada bloque declara primero las claves necesarias y a
    #    continuación crea la(s) variable(s) correspondiente(s).
    #    x es la única variable binaria; el resto son continuas
    #    (la integralidad queda inducida por x).
    # ═════════════════════════════════════════════════════════════════

    # ── x: asignación principal ──────────────────────────────────────
    # x[p, t, h, d] = 1 si la persona p realiza la tarea t en la hora h
    # del día d. Solo existe donde hay disponibilidad y habilidad.
    x_set = {(p, t, h, d)
             for p in people for t in tasks
             for d in days   for h in hours[d]
             if (p, t) in skill_set and (p, h, d) in avail_set}

    x = model.addVars(x_set,
                      vtype=GRB.BINARY,
                      name="assignment")

    # ── m: cobertura missing ─────────────────────────────────────────
    # m[t, h, d] = número de puestos sin cubrir de la tarea t en (h, d).
    # Existe para toda combinación (tarea, hora, día) con demanda.
    m_keys = [(t, h, d)
              for t in tasks
              for d in days
              for h in hours[d]]

    m = model.addVars(m_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="missing_staff_penalty")

    # ── u: mandatos incumplidos ──────────────────────────────────────
    # u[p, t, h, d] = 1 si se ignoró la orden del manager de asignar
    # a la persona p a la tarea t en (h, d). Solo para entradas force=1.
    force_keys = [(p, t, h, d)
                  for (p, t, h, d), v in force.items()
                  if v == 1]

    u = model.addVars(force_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="mandates_penalty")

    # ── z: desviación respecto al plan anterior ──────────────────────
    # z[p, t, h, d] = |x_nuevo - x_previo|. Penaliza cambios sobre el
    # plan publicado. Comparte claves con x (solo donde x puede existir).
    z = model.addVars(x_set,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="deviation_penalty")

    # ── q: cuota de experiencia no alcanzada ─────────────────────────
    # q[p, t] = 1 si la persona p no llegó a su mínimo de horas deseadas
    # en la tarea t. Solo para parejas (persona, tarea) con cuota > 0.
    quota_keys = [(p, t)
                  for p in people for t in tasks
                  if min_quota.get((p, t), 0) > 0]

    q = model.addVars(quota_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="quota_penalty")

    # ── g: reinicios de bloque de trabajo ────────────────────────────
    # g[p, h, d] = 1 si la persona p comienza un nuevo bloque de trabajo
    # en la hora h del día d (es decir, trabajó en h pero no en h-1).
    # Penalizar g equivale a forzar bloques contiguos (sin huecos).
    # P7 — g_keys tiene exactamente las mismas claves (p, h, d) que
    # avail_set. Se usa avail_set directamente y se elimina la variable
    # intermedia anti_ubiq_keys (que era otro alias del mismo conjunto).
    g = model.addVars(avail_set,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="gaps_penalty")

    # ── j_max / j_min: cotas de equidad diaria ───────────────────────
    # j_max[d] y j_min[d] son el máximo y mínimo de horas trabajadas
    # por cualquier persona el día d. El solver minimiza su diferencia.
    max_hours_any_day = max((len(hours[d]) for d in days), default=0)

    j_max = model.addVars(days,
                          lb=0,
                          ub=max_hours_any_day,
                          vtype=GRB.CONTINUOUS,
                          name="j_max_var_aux")

    j_min = model.addVars(days,
                          lb=0,
                          ub=max_hours_any_day,
                          vtype=GRB.CONTINUOUS,
                          name="j_min_var_aux")

    # ── w_max / w_min: cotas de equidad global ───────────────────────
    # w_max y w_min son el máximo y mínimo de horas totales trabajadas
    # por cualquier persona en todo el evento.
    total_hours_all_days = sum(len(hours[d]) for d in days)

    w_max = model.addVar(lb=0,
                         ub=total_hours_all_days,
                         vtype=GRB.CONTINUOUS,
                         name="w_max_var_aux")

    w_min = model.addVar(lb=0,
                         ub=total_hours_all_days,
                         vtype=GRB.CONTINUOUS,
                         name="w_min_var_aux")

    # ── c: penalización por horas consecutivas en tareas de rotación ─
    # c[p, t, h, d] = 1 si la persona p trabaja la tarea t en la hora h
    # Y en la hora h+1 del mismo día d, siendo t una tarea de rotación.
    consec_keys = [(p, t, h, d)
                   for p in people for t in tasks
                   for d in days   for h in hours[d][:-1]
                   if rotation.get(t, 0) == 1
                   and (p, t, h,              d) in x_set
                   and (h, d)                    in h_next
                   and (p, t, h_next[(h, d)], d) in x_set]

    c = model.addVars(consec_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="consecutivity_penalty")

    # ── f: separación de amigos (penaliza que no coincidan) ──────────
    # f[p1, p2, t, h, d] = |x[p1] - x[p2]| en el slot (t, h, d).
    # Solo se crea donde ambos amigos (social=1) pueden trabajar juntos.
    friends_keys = [(p1, p2, t, h, d)
                    for (p1, p2), sv in social.items() if sv == 1
                    for t in tasks
                    for d in days for h in hours[d]
                    if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]

    f = model.addVars(friends_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="friends_separated_penalty")

    # ── e: conflicto entre enemigos (penaliza que coincidan) ─────────
    # e[p1, p2, t, h, d] = 1 si ambos enemigos (social=-1) trabajan el
    # mismo slot. Si hard_enemies=True, se usa una constraint dura en
    # su lugar y enemies_keys queda vacío, por lo que e es un tupledict
    # vacío y no contribuye al objetivo ni genera constraints.
    # P5 — e y k se crean siempre (lista vacía si no aplican) para
    # eliminar todos los if de guardia dispersos en el objetivo,
    # las constraints y los resultados.
    enemies_scope = [(p1, p2, t, h, d)
                     for (p1, p2), sv in social.items() if sv == -1
                     for t in tasks
                     for d in days for h in hours[d]
                     if (p1, t, h, d) in x_set and (p2, t, h, d) in x_set]

    enemies_keys = [] if hard_enemies else enemies_scope

    e = model.addVars(enemies_keys,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="enemies_together_penalty")

    # ── k: capitán ausente ───────────────────────────────────────────
    # k[h, d] = 1 si ningún capitán está de guardia en una hora activa.
    # P5 — active_hd y k se crean siempre; si no hay capitanes ambos
    # quedan vacíos y no contribuyen al objetivo ni generan constraints.
    active_hd = [(h, d)
                 for d in days for h in hours[d]
                 if sum(demand.get((t, h, d), 0) for t in tasks) > 0
                 ] if captains else []

    k = model.addVars(active_hd,
                      lb=0,
                      vtype=GRB.CONTINUOUS,
                      name="absent_captains_penalty")

    # ═════════════════════════════════════════════════════════════════
    # 4. FUNCIÓN OBJETIVO
    #    Minimizar la suma ponderada de todas las penalizaciones.
    #    Términos ordenados de mayor a menor coste (prioridad):
    #    Cobertura > Mandatos > Capitanes > Emergencias > Estabilidad >
    #    Equidad diaria > Equidad global > Rotación > Social >
    #    Huecos > Cuota > Preferencias
    # ═════════════════════════════════════════════════════════════════

    obj = gp.LinExpr()

    # Puestos sin cubrir: máxima prioridad
    obj += W_COVERAGE * m.sum()

    # Mandatos del manager incumplidos
    obj += W_FORCE * u.sum()

    # Horas sin capitán de guardia (0 si k está vacío)
    obj += W_CAPTAIN * k.sum()

    # Uso de personal en estado de "emergencia" (disponibilidad blanda)
    obj += W_EMERG * gp.quicksum(
        emergency.get((p, h, d), 0) * x[p, t, h, d]
        for p, t, h, d in x_set)

    # Cambios respecto al plan publicado (incomodidad para el personal)
    obj += W_STABILITY * z.sum()

    # Diferencia entre la persona más y menos cargada cada día
    obj += W_EQ_DAY * gp.quicksum(j_max[d] - j_min[d] for d in days)

    # Diferencia global entre la persona más y menos cargada del evento
    obj += W_EQ_GLOBAL * (w_max - w_min)

    # Horas consecutivas en tareas de rotación (0 si c está vacío)
    obj += W_ROTATION * c.sum()

    # Amigos que no coinciden / enemigos que sí coinciden (0 si vacíos)
    obj += W_SOCIAL * f.sum()
    obj += W_SOCIAL * e.sum()

    # Huecos en el bloque de trabajo diario (reinicios de jornada)
    obj += W_GAP * g.sum()

    # Cuotas de experiencia no alcanzadas
    obj += W_QUOTA * q.sum()

    # Coste de preferencias: menor prioridad, al final de la jerarquía
    obj += W_PREF * gp.quicksum(
        pref_cost.get((p, t), 0) * x[p, t, h, d]
        for p, t, h, d in x_set)

    model.setObjective(obj, GRB.MINIMIZE)

    # ═════════════════════════════════════════════════════════════════
    # 5. CONSTRAINTS
    # ═════════════════════════════════════════════════════════════════

    # ── A. Cobertura de demanda ───────────────────────────────────────
    # Personas asignadas + faltantes = demanda requerida.
    # Si no hay suficiente personal disponible, m absorbe el déficit.
    model.addConstrs(
        (x.sum('*', t, h, d) + m[t, h, d] == demand.get((t, h, d), 0)
         for t, h, d in m_keys),
        name="coverage_constraint")

    # ── B.1. Mandatos individuales (restricción blanda) ───────────────
    # Si force=1 y el solver no asigna (x=0), u se fuerza a 1.
    # Para slots imposibles (fuera de x_set), u se fija a 1 directamente.
    model.addConstrs(
        (1 - x[p, t, h, d] <= u[p, t, h, d]
         for p, t, h, d in force_keys if (p, t, h, d) in x_set),
        name="valid_mandate_constraint")

    model.addConstrs(
        (u[p, t, h, d] >= 1
         for p, t, h, d in force_keys if (p, t, h, d) not in x_set),
        name="impossible_mandate_constraint")

    # ── B.2. Presencia de capitán (restricción blanda) ────────────────
    # En cada hora activa debe haber al menos un capitán de guardia.
    # Si active_hd está vacío, addConstrs no añade ninguna constraint.
    model.addConstrs(
        (gp.quicksum(x.sum(p, '*', h, d) for p in captains) + k[h, d] >= 1
         for h, d in active_hd),
        name="captains_constraint")

    # ── C. Anti-ubicuidad (restricción dura) ─────────────────────────
    # Una persona no puede realizar más de una tarea al mismo tiempo.
    # P7 — avail_set tiene exactamente las mismas claves (p, h, d) que
    # la antigua anti_ubiq_keys, que era un alias redundante del mismo
    # conjunto. Se itera directamente sobre avail_set.
    model.addConstrs(
        (x.sum(p, '*', h, d) <= 1
         for p, h, d in avail_set),
        name="anti_ubiquity_constraint")

    # ── E. Equidad diaria ─────────────────────────────────────────────
    # Las horas trabajadas por cada persona en un día quedan encajadas
    # entre j_min[d] y j_max[d]. El solver minimiza su diferencia.
    eq_day_keys = [(p, d)
                   for d in days for p in people
                   if any((p, h, d) in avail_set for h in hours[d])]

    model.addConstrs(
        (x.sum(p, '*', '*', d) <= j_max[d]
         for p, d in eq_day_keys),
        name="eq_day_max_constraint")

    model.addConstrs(
        (x.sum(p, '*', '*', d) >= j_min[d]
         for p, d in eq_day_keys),
        name="eq_day_min_constraint")

    # ── E. Equidad global ─────────────────────────────────────────────
    # Las horas totales de cada persona quedan encajadas entre w_min
    # y w_max. Solo se aplica a personas con al menos una disponibilidad.
    people_with_avail = [p for p in people
                         if any((p, h, d) in avail_set
                                for d in days for h in hours[d])]

    model.addConstrs(
        (x.sum(p, '*', '*', '*') <= w_max
         for p in people_with_avail),
        name="eq_global_max_constraint")

    model.addConstrs(
        (x.sum(p, '*', '*', '*') >= w_min
         for p in people_with_avail),
        name="eq_global_min_constraint")

    # ── F. Fatiga por rotación ────────────────────────────────────────
    # Si una persona trabaja la tarea t en h y en h+1 (siendo t de
    # rotación), c se fuerza a 1 y se penaliza en el objetivo.
    model.addConstrs(
        (x[p, t, h, d] + x[p, t, h_next[(h, d)], d] - c[p, t, h, d] <= 1
         for p, t, h, d in consec_keys),
        name="rotation_constraint")

    # ── G. Social — Amigos ────────────────────────────────────────────
    # f = |x[p1] - x[p2]|: se penaliza si los amigos no coinciden
    # en el mismo slot. Si friends_keys está vacío, addConstrs no añade
    # ninguna constraint.
    model.addConstrs(
        (x[p1, t, h, d] - x[p2, t, h, d] <= f[p1, p2, t, h, d]
         for p1, p2, t, h, d in friends_keys),
        name="friends_together_constraint_right")

    model.addConstrs(
        (x[p2, t, h, d] - x[p1, t, h, d] <= f[p1, p2, t, h, d]
         for p1, p2, t, h, d in friends_keys),
        name="friends_together_constraint_left")

    # ── G. Social — Enemigos ──────────────────────────────────────────
    # Modo duro: dos enemigos no pueden coincidir nunca (constraint dura).
    # Modo blando: si coinciden, e se fuerza a 1 y se penaliza.
    if hard_enemies:
        model.addConstrs(
            (x[p1, t, h, d] + x[p2, t, h, d] <= 1
             for p1, p2, t, h, d in enemies_scope),
            name="enemies_hard_constraint")
    else:
        # enemies_keys puede estar vacío si no hay parejas enemigas;
        # en ese caso addConstrs no añade nada.
        model.addConstrs(
            (x[p1, t, h, d] + x[p2, t, h, d] - e[p1, p2, t, h, d] <= 1
             for p1, p2, t, h, d in enemies_keys),
            name="enemies_soft_constraint")

    # ── H. Cuota mínima de experiencia ───────────────────────────────
    # Si la persona no alcanza su cuota deseada en la tarea t durante
    # el día d, q absorbe el déficit y se penaliza en el objetivo.
    # P6 — se reemplaza el opaco `for target in (min(...),)` por un
    # if-guard explícito, homogéneo con el resto de list comprehensions.
    quota_constr_keys = [(p, t, d, min(min_quota.get((p, t), 0), len(hours[d])))
                         for p, t in quota_keys
                         for d in days
                         if min(min_quota.get((p, t), 0), len(hours[d])) > 0]

    model.addConstrs(
        (x.sum(p, t, '*', d) + q[p, t] >= target
         for p, t, d, target in quota_constr_keys),
        name="min_quota_constraint")

    # ── I. Estabilidad / Desviación ───────────────────────────────────
    # z = |x_nuevo - x_previo|, linealizado con dos constraints.
    # Se penaliza cualquier cambio sobre el plan publicado.
    model.addConstrs(
        (z[p, t, h, d] >= X_prev.get((p, t, h, d), 0) - x[p, t, h, d]
         for p, t, h, d in x_set),
        name="deviation_constraint_right")

    model.addConstrs(
        (z[p, t, h, d] >= x[p, t, h, d] - X_prev.get((p, t, h, d), 0)
         for p, t, h, d in x_set),
        name="deviation_constraint_left")

    # ── J.1. Detección de huecos ──────────────────────────────────────
    # g[p, h, d] = 1 si la persona p empieza un nuevo bloque en la hora h.
    # Primera hora del día: g = x (siempre cuenta como inicio si trabaja).
    # Resto de horas: g >= x[h] - x[h-1] (inicio solo si vuelve a trabajar
    # tras una hora de descanso).
    gap_first_keys = [(p, d)
                      for p in people for d in days
                      if (p, hours[d][0], d) in avail_set]

    model.addConstrs(
        (g[p, hours[d][0], d] == x.sum(p, '*', hours[d][0], d)
         for p, d in gap_first_keys),
        name="gap_first_constraint")

    gap_rest_keys = [(p, hours[d][i], hours[d][i - 1], d)
                     for p in people for d in days
                     for i in range(1, len(hours[d]))
                     if (p, hours[d][i], d) in avail_set]

    model.addConstrs(
        (g[p, h_curr, d] >= x.sum(p, '*', h_curr, d) - x.sum(p, '*', h_prev, d)
         for p, h_curr, h_prev, d in gap_rest_keys),
        name="gap_rest_constraint")

    # ── J.2. Descanso obligatorio (ventana deslizante) ────────────────
    # En cualquier ventana de Y+1 horas consecutivas, la persona puede
    # trabajar como máximo Y horas (combinado con J.1 obliga al descanso).
    if enforced_rest and max_consec_hours is not None:
        Y = max_consec_hours
        rest_keys = [(p, d, i)
                     for d in days for p in people
                     for i in range(len(hours[d]) - Y)]

        model.addConstrs(
            (gp.quicksum(x.sum(p, '*', tau, d) for tau in hours[d][i:i + Y + 1]) <= Y
             for p, d, i in rest_keys),
            name="min_rest_constraint")

    # ═════════════════════════════════════════════════════════════════
    # 6. CONFIGURACIÓN DEL SOLVER Y OPTIMIZACIÓN
    # ═════════════════════════════════════════════════════════════════

    model._x     = x
    model._x_set = x_set

    for param_name, param_value in data.get("solver_params", {}).items():
        try:
            model.setParam(param_name, param_value)
        except Exception as err:
            print(f"Warning: Could not set param {param_name}={param_value}: {err}")

    # Throttle: evita saturar el UI enviando como máximo una actualización
    # cada 0.5 s aunque el solver encuentre muchas soluciones seguidas.
    last_ui_update = [0.0]

    def intermediate_solution_callback(mdl, where):
        """Envía al UI cada nueva solución entera mejorada que encuentra el solver,
        con un mínimo de 0.5 s entre actualizaciones consecutivas."""
        if where == GRB.Callback.MIPSOL and ui_update_callback:
            now = time.monotonic()
            if now - last_ui_update[0] < 0.5:
                return
            last_ui_update[0] = now
            try:
                x_vals = mdl.cbGetSolution(mdl._x)
                temp_assignment = {d: {p: {} for p in people} for d in days}
                for d in days:
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

    # ═════════════════════════════════════════════════════════════════
    # 7. EXTRACCIÓN DE RESULTADOS
    # ═════════════════════════════════════════════════════════════════

    status_map = {GRB.OPTIMAL:     "Optimal",
                  GRB.TIME_LIMIT:  "Time Limit Reached",
                  GRB.INFEASIBLE:  "Infeasible",
                  GRB.INTERRUPTED: "Interrupted by User"}
    sol = {"status":        status_map.get(model.Status, f"Status Code: {model.Status}"),
           "enforced_rest": enforced_rest}

    if model.SolCount == 0:
        raise Exception("No feasible solution was found before stopping/timeout.")

    # ── Cuadrícula de asignaciones ────────────────────────────────────
    sol["assignment"] = {
        d: {
            p: {
                h: next(
                    (t for t in tasks
                     if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5),
                    None)
                for h in hours[d]}
            for p in people}
        for d in days}

    # ── Puestos sin cubrir ────────────────────────────────────────────
    sol["missing"] = [f"{t} @ {h}, {d}: {m[t, h, d].X:.0f} missing"
                      for t, h, d in m_keys
                      if m[t, h, d].X > 0.01]

    # ── Carga de trabajo por persona ──────────────────────────────────
    # Pre-fill the dictionary with 0 for all people
    assigned_counts = {p: 0 for p in people}
    
    # Iterate through the decision variables exactly once
    for p, t, h, d in x_set:
        if x[p, t, h, d].X > 0.5:
            assigned_counts[p] += 1
            
    sol["workload"] = assigned_counts
    sol["w_max"] = w_max.X
    sol["w_min"] = w_min.X

    # ── Cumplimiento de cuotas ────────────────────────────────────────
    sol["quota_issues"] = []
    for p, t in quota_keys:
        demanded = min_quota[p, t]
        assigned = sum(1 for p2, t2, h, d in x_set
                    if p2 == p and t2 == t and x[p2, t2, h, d].X > 0.5)
        status = "OK" if assigned >= demanded else "SHORTFALL"
        sol["quota_issues"].append(
            f"{status}: {p} — task '{t}': {assigned}/{demanded} h assigned")

    # ── Huecos en los bloques de trabajo ─────────────────────────────
    gaps = []
    for d in days:
        day_gaps = [f"  • {p}: {starts} blocks (Starts: {', '.join(hs)})"
                    for p in people
                    for starts, hs in [(
                        sum(1 for h in hours[d] if (p, h, d) in avail_set and g[p, h, d].X > 0.5),
                        [h for h in hours[d] if (p, h, d) in avail_set and g[p, h, d].X > 0.5])]
                    if starts > 1]
        if day_gaps:
            gaps.append(f"--- {d} ---")
            gaps.extend(day_gaps)
    sol["gaps"] = gaps

    # ── Incidencias sociales ──────────────────────────────────────────
    soc_issues = []

    for (p1, p2), sv in social.items():
        if sv != 1:
            continue
        for t in tasks:
            for d in days:
                for h in hours[d]:
                    if ((p1, t, h, d) in x_set and x[p1, t, h, d].X > 0.5
                            and (p2, t, h, d) in x_set and x[p2, t, h, d].X > 0.5):
                        soc_issues.append(
                            f"MATCH: Friends {p1} & {p2} together @ {t}, {h}, {d}")

    if any(sv == -1 for sv in social.values()):
        # P5 — e siempre existe: si está vacío, la list comprehension
        # devuelve [] sin necesidad de un if de guardia previo.
        violations = [f"VIOLATION: Enemies {p1} & {p2} together @ {t}, {h}, {d}"
                      for p1, p2, t, h, d in enemies_keys
                      if e[p1, p2, t, h, d].X > 0.5]
        soc_issues.extend(violations)
        if not violations:
            soc_issues.append("SUCCESS: All enemies successfully separated (0 violations).")

    sol["social_issues"] = soc_issues

    # ── Incidencias de capitanes ──────────────────────────────────────
    # P5 — k siempre existe: si está vacío, la list comprehension
    # devuelve [] y captain_issues queda como lista vacía sin if de guardia.
    missing_caps = [f"MISSING CAPTAIN @ {h}, {d}"
                    for h, d in active_hd
                    if k[h, d].X > 0.5]
    sol["captain_issues"] = (missing_caps or
                             ["SUCCESS: All active hours have at least one captain on duty."]
                             if active_hd else [])

    # ── Llamadas de emergencia ────────────────────────────────────────
    sol["emerg_issues"] = [f"EMERGENCY CALL-IN: {p} @ {t}, {h}, {d}"
                           for p in people for d in days for h in hours[d]
                           if emergency.get((p, h, d), 0) == 1
                           for t in tasks
                           if (p, t, h, d) in x_set and x[p, t, h, d].X > 0.5]

    # ── Violaciones de rotación ───────────────────────────────────────
    rot_violations = [f"CONSECUTIVE: {p} doing '{t}' at {h} & {h_next[(h, d)]}, {d}"
                      for p in people for t in tasks
                      if rotation.get(t, 0) == 1
                      for d in days
                      for h in hours[d][:-1]
                      if (p, t, h,              d) in x_set and x[p, t, h,              d].X > 0.5
                      and (p, t, h_next[(h, d)], d) in x_set and x[p, t, h_next[(h, d)], d].X > 0.5]
    sol["rotation_issues"] = (rot_violations or
                              ["SUCCESS: No consecutive hours on rotation tasks."])

    return sol