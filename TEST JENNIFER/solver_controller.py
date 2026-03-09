import threading
import flet as ft
from constants import _s
from tabs.output import OutputTab
from solve_model_pace_14 import solve_model


class SolverController:

    def __init__(self, state, page: ft.Page,
                 output_tab: OutputTab,
                 on_solve_blocked_update,
                 switch_page_cb,
                 ui_lock: threading.Lock,
                 live_callbacks_sw: ft.Switch):
        self.state    = state
        self.page     = page
        self._out     = output_tab
        self._upd     = on_solve_blocked_update
        self._switch  = switch_page_cb
        self._lock    = ui_lock
        self._sw_live = live_callbacks_sw

    def do_solve(self, e):
        s = self.state
        if s.running_model_ref[0] is not None:
            s.running_model_ref[0].terminate()
        if s.solve_blocked:
            return

        people, tasks, hours, days = s.dims()
        groups = s.build_groups(people)

        # ── Build max_consec dict ─────────────────────────────────────
        # New logic: for each person, if they are in
        # consec_personalized_persons use their individual values;
        # otherwise use the global values.
        max_consec: dict = {}

        def _parse_int(raw: str, default: int = 1) -> int | None:
            """Return int if valid and >= 1, None if empty, raises if invalid."""
            raw = raw.strip()
            if not raw:
                return None
            v = int(raw)          # let ValueError propagate
            return max(1, v)      # minimum 1 always

        global_limit_raw = s.consec_global_val.strip()
        global_rest_raw  = s.consec_global_rest.strip()

        for p in people:
            if p in s.consec_personalized_persons:
                # Use individual values
                raw_limit = s.consec_per_person.get(p, "").strip()
                raw_rest  = s.consec_rest_per_person.get(p, "").strip()
            else:
                # Use global values
                raw_limit = global_limit_raw
                raw_rest  = global_rest_raw

            if not raw_limit:
                continue
            try:
                limit    = _parse_int(raw_limit)
                min_rest = _parse_int(raw_rest) if raw_rest else 1
                if limit is not None:
                    max_consec[p] = (limit, min_rest or 1)
            except ValueError:
                pass

        availability: dict = {}
        emergency   : dict = {}
        for p in people:
            for j in days:
                for h in hours[j]:
                    v = s.avail_st.get((p, h, j), 1)
                    availability[(p, h, j)] = 1 if v in (1, 2) else 0
                    emergency   [(p, h, j)] = 1 if v == 2      else 0

        demand: dict = {}
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    raw = s.demand_st.get((t, h, j), "1").strip()
                    try:    demand[(t, h, j)] = int(raw) if raw else 0
                    except: demand[(t, h, j)] = 0

        mq: dict = {}
        for p in people:
            for t in tasks:
                for j in days:
                    raw = s.quota_st.get((p, t, j), "").strip()
                    try:    mq[(p, t, j)] = int(raw) if raw else 0
                    except: mq[(p, t, j)] = 0

        X_prev     = s.build_x_prev(people, tasks, hours, days)
        using_base = s.base_run_idx is not None
        live_status = (f"Solving… (Base: Run #{s.base_run_idx+1})"
                       if using_base else "Solving...")

        # ── Derive force & just_work from mandatory_rules ─────────────
        # Replaces the old force_st / just_work_st direct reads.
        force_active     = {}
        just_work_active = {}

        for rule in s.mandatory_rules:
            rp  = rule.get("person")
            rt  = rule.get("task")
            jda = rule.get("just_do_anything", False)

            if rp not in people:
                continue
            for d, hs in rule.get("hours", {}).items():
                if d not in days:
                    continue
                for h in hs:
                    if h not in hours.get(d, []):
                        continue
                    if jda:
                        just_work_active[(rp, h, d)] = 1
                    elif rt and rt in tasks:
                        force_active[(rp, rt, h, d)] = 1

        force_dict = {
            (p, t, h, j): force_active.get((p, t, h, j), 0)
            for p in people for t in tasks
            for j in days for h in hours[j]
        }
        just_work_dict = {
            (p, h, j): just_work_active.get((p, h, j), 0)
            for p in people for j in days for h in hours[j]
        }

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, emergency=emergency,
            demand=demand,
            skills   = {(p, t): s.skills_st.get((p, t), 1)
                        for p in people for t in tasks},
            force     = force_dict,
            social   = {(p1, p2): s.social_st.get((p1, p2), 0)
                        for i, p1 in enumerate(people)
                        for p2 in people[i+1:]},
            min_quota = mq,
            pref_cost = {(p, t): 1 for p in people for t in tasks},
            rotation  = {t: s.rotation_st.get(t, 1) for t in tasks},
            X_prev    = X_prev,
            weights          = s.weights_st.copy(),
            max_consec_hours = max_consec,
            solver_params    = s.solver_params,
            hard_enemies     = s.hard_enemies,
            live_callbacks   = 1 if self._sw_live.value else 0,
            groups           = groups,
            just_work        = just_work_dict,
            captain_rules    = s.captain_rules,
        )

        self._out.rebuild(
            live_sol={"status": live_status, "assignment": {}},
            live_people=people, live_tasks=tasks, live_hours=hours,
            live_days=days, live_avail=availability, live_emerg=emergency,
            live_groups=groups)
        self._switch(7)   # Output tab now at index 7 (Force tab removed)

        def _update_ui(partial_sol):
            if not self._lock.acquire(blocking=False):
                return
            try:
                self._out.rebuild(
                    live_sol=partial_sol,
                    live_people=people, live_tasks=tasks, live_hours=hours,
                    live_days=days, live_avail=availability, live_emerg=emergency,
                    live_groups=groups)
                self.page.update()
            except Exception:
                pass
            finally:
                self._lock.release()

        def _run():
            try:
                final = solve_model(
                    data,
                    ui_update_callback=_update_ui if self._sw_live.value else None,
                    active_model_ref=s.running_model_ref)
                s.solution_history.append({
                    "sol":          final,
                    "people":       people,
                    "tasks":        tasks,
                    "hours":        hours,
                    "days":         days,
                    "availability": availability,
                    "emergency":    emergency,
                    "groups":       groups,
                })
                s.base_run_idx = None
                self._out.rebuild()
            except Exception as ex:
                self._out._ct.controls = [
                    ft.Text(f"ERROR: {ex}", color=ft.Colors.RED_400, size=_s(14))]
            finally:
                s.solver_running = False
                self._upd()
            self.page.update()

        s.solver_running = True
        self._upd()
        threading.Thread(target=_run, daemon=True).start()

    def do_stop(self, e):
        if self.state.running_model_ref[0] is not None:
            self.state.running_model_ref[0].terminate()