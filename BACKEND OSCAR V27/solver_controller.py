import sys
import threading
import traceback
import flet as ft
from constants import _s
from ui_helpers import UIHelpers
from tabs.output import OutputTab
from tabs.preferences import PreferencesTab
from solve_model_pace_15 import solve_model


class SolverController:

    def __init__(self, state, page: ft.Page,
                 output_tab: OutputTab,
                 pref_tab: PreferencesTab,
                 on_solve_blocked_update,
                 switch_page_cb,
                 ui_lock: threading.Lock,
                 live_callbacks_sw: ft.Switch):
        self.state    = state
        self.page     = page
        self._out     = output_tab
        self._pref    = pref_tab
        self._upd     = on_solve_blocked_update
        self._switch  = switch_page_cb
        self._lock    = ui_lock
        self._sw_live = live_callbacks_sw

    # ── Helpers to build solver input dicts ──────────────────────────

    def _build_consec_dicts(self, people):
        """Returns (max_consec, capacity, max_hours_per_day, max_hours_per_event)."""
        s    = self.state
        _spi = UIHelpers.parse_pos_int

        global_limit_raw = s.consec_global_val.strip()
        global_rest_raw  = s.consec_global_rest.strip()
        global_cap_raw   = getattr(s, "consec_global_capacity", "100").strip()
        global_max_day_raw   = getattr(s, "consec_global_max_day", "").strip()
        global_max_event_raw = getattr(s, "consec_global_max_event", "").strip()

        max_consec          : dict = {}
        capacity            : dict = {}
        max_hours_per_day   : dict = {}
        max_hours_per_event : dict = {}

        for p in people:
            personalized = p in s.consec_personalized_persons

            if personalized:
                raw_limit = s.consec_per_person.get(p, "").strip()
                raw_rest  = s.consec_rest_per_person.get(p, "").strip()
                raw_cap   = s.consec_capacity_per_person.get(p, "100").strip()
                raw_day   = s.consec_max_day_per_person.get(p, "").strip()
                raw_event = s.consec_max_event_per_person.get(p, "").strip()
            else:
                raw_limit = global_limit_raw
                raw_rest  = global_rest_raw
                raw_cap   = global_cap_raw
                raw_day   = global_max_day_raw
                raw_event = global_max_event_raw

            limit = _spi(raw_limit)
            if limit is not None:
                rest = _spi(raw_rest) or 1
                max_consec[p] = (limit, rest)

            try:
                capacity[p] = (max(0.0, min(1.0, int(raw_cap) / 100.0))
                               if raw_cap else 1.0)
            except ValueError:
                capacity[p] = 1.0

            v = _spi(raw_day)
            if v is not None:
                max_hours_per_day[p] = v

            v = _spi(raw_event)
            if v is not None:
                max_hours_per_event[p] = v

        return max_consec, capacity, max_hours_per_day, max_hours_per_event

    def _build_availability_demand(self, people, tasks, hours, days):
        """Returns (availability, emergency, demand)."""
        s = self.state
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
                    try:
                        demand[(t, h, j)] = int(raw) if raw else 0
                    except ValueError:
                        demand[(t, h, j)] = 0
        return availability, emergency, demand

    def _build_force_dicts(self, people, tasks, hours, days):
        """Returns (force_dict, just_work_dict, force_rest_dict)."""
        s = self.state
        force_active      = {}
        just_work_active  = {}
        force_rest_active = {}

        for rule in s.mandatory_rules:
            rp    = rule.get("person")
            rt    = rule.get("task")
            jda   = rule.get("just_do_anything", False)
            jrest = rule.get("just_rest", False)

            if rp not in people:
                continue
            for d, hs in rule.get("hours", {}).items():
                if d not in days:
                    continue
                for h in hs:
                    if h not in hours.get(d, []):
                        continue
                    if jrest:
                        force_rest_active[(rp, h, d)] = 1
                    elif jda:
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
        force_rest_dict = {
            (p, h, j): force_rest_active.get((p, h, j), 0)
            for p in people for j in days for h in hours[j]
        }
        return force_dict, just_work_dict, force_rest_dict

    # ── Main solve entry point ───────────────────────────────────────

    def do_solve(self, e):
        s = self.state
        if s.running_model_ref[0] is not None:
            s.running_model_ref[0].terminate()
        if s.solve_blocked:
            return

        people, tasks, hours, days = s.dims()
        groups = s.build_groups(people)

        max_consec, capacity, max_hours_per_day, max_hours_per_event = \
            self._build_consec_dicts(people)

        availability, emergency, demand = \
            self._build_availability_demand(people, tasks, hours, days)

        force_dict, just_work_dict, force_rest_dict = \
            self._build_force_dicts(people, tasks, hours, days)

        X_prev      = s.build_x_prev(people, tasks, hours, days)
        using_base  = s.base_run_idx is not None
        live_status = (f"Solving… (Base: Run #{s.base_run_idx+1})"
                       if using_base else "Solving...")

        # Task duration
        task_duration = {}
        for t in tasks:
            raw = s.task_duration_st.get(t, "1").strip()
            try:
                dur = max(1, int(raw)) if raw else 1
            except ValueError:
                dur = 1
            if dur > 1:
                task_duration[t] = dur

        # Task priority
        N_tasks = len(tasks)
        task_priority = {t: 1 + (N_tasks - rank)
                         for rank, t in enumerate(tasks)}

        # Displacement / Travel
        task_location = s.build_task_location(tasks)
        travel_time   = s.build_travel_time()

        # Preferences
        pref_cost = self._pref.get_pref_cost(people, tasks)

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, emergency=emergency,
            demand=demand,
            skills    = {(p, t): s.skills_st.get((p, t), 1)
                         for p in people for t in tasks},
            force     = force_dict,
            social    = {(p1, p2): s.social_st.get((p1, p2), 0)
                         for i, p1 in enumerate(people)
                         for p2 in people[i+1:]},
            pref_cost = pref_cost,
            rotation  = {t: 1 for t in tasks if s.rotation_st.get(t, "R") == "R"},
            sticky    = {t: 1 for t in tasks if s.rotation_st.get(t, "R") == "S"},
            X_prev    = X_prev,
            weights          = s.weights_st.copy(),
            max_consec_hours = max_consec,
            solver_params    = s.solver_params,
            hard_enemies     = s.hard_enemies,
            live_callbacks   = 1 if self._sw_live.value else 0,
            groups           = groups,
            just_work        = just_work_dict,
            force_rest       = force_rest_dict,
            captain_rules    = s.captain_rules,
            quota_rules      = s.quota_rules,
            task_duration    = task_duration,
            task_priority    = task_priority,
            capacity         = capacity,
            max_hours_per_day   = max_hours_per_day,
            max_hours_per_event = max_hours_per_event,
            task_location       = task_location,
            travel_time         = travel_time,
        )

        self._out.rebuild(
            live_sol={"status": live_status, "assignment": {}},
            live_people=people, live_tasks=tasks, live_hours=hours,
            live_days=days, live_avail=availability, live_emerg=emergency,
            live_groups=groups)
        self._switch(9)

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
                traceback.print_exc(file=sys.stderr)
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