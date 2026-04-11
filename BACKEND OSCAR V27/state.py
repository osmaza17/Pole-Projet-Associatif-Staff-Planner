from constants import DEFAULT_WEIGHTS, DEFAULT_SOLVER_PARAMS, DEFAULT_HOURS_TEXT


def _parse_lines(txt: str) -> list:
    """Parse newline-separated text into stripped, deduplicated, non-empty entries."""
    return list(dict.fromkeys(x.strip() for x in txt.split("\n") if x.strip()))


class AppState:
    _DEFAULT_PEOPLE = (
        "Arnaud\nNina\nJoseph\nChloé\nNiels\nZiad\nTristan\nJasmin\nMarine\nNoé\n"
        "Jules R\nGuillaume\nNoémie\nStanislas\nTérence\nManon In\nManon L\nPauline\n"
    )
    _DEFAULT_TASKS = (
        "Iron\nRenew\nClean\nBackup\nDiscard\nSweep\nArchive\nCall\nLoad\nPolish\n"
    )
    _DEFAULT_DAYS = "Mon"

    def __init__(self):
        self._init_defaults()

    def _init_defaults(self):
        self.tasks_text = self._DEFAULT_TASKS
        self.days_text  = self._DEFAULT_DAYS

        self.avail_st        : dict = {}
        self.demand_st       : dict = {}
        self.skills_st       : dict = {}
        self.force_st        : dict = {}
        self.just_work_st    : dict = {}
        self.social_st       : dict = {}
        self.rotation_st     : dict = {}
        self.task_duration_st: dict = {}
        self.hard_enemies    : bool = False
        self.hours_per_day   : dict = {}
        self.groups_st       : dict = {"Group 1": self._DEFAULT_PEOPLE}

        self.captain_rules   : list = []
        self.mandatory_rules : list = []
        self.quota_rules     : list = []

        self.pref_order_st   : dict = {}
        self.pref_enabled_st : dict = {}

        self.consec_global_val          : str = ""
        self.consec_global_rest         : str = "1"
        self.consec_global_capacity     : str = "100"
        self.consec_global_max_day      : str = ""
        self.consec_global_max_event    : str = ""

        self.consec_per_person          : dict = {}
        self.consec_rest_per_person     : dict = {}
        self.consec_capacity_per_person : dict = {}
        self.consec_max_day_per_person  : dict = {}
        self.consec_max_event_per_person: dict = {}

        self.consec_personalized_persons: set  = set()
        self.consec_personalized        : bool = False

        self.avail_filter  = None
        self.demand_filter = None
        self.force_filter  = [None, None]

        self.location_names_st    : list = ["Default"]
        self.task_location_idx_st : dict = {}
        self.travel_time_st       : dict = {}

        self.validation_errors: dict = {
            "demand": set(), "consec": set(), "rules": set()
        }

        self.solve_blocked     : bool = False
        self.solver_running    : bool = False
        self.running_model_ref : list = [None]

        self.solution_history: list = []
        self.diff_state      : dict = {"ref": None, "cmp": None}
        self.base_run_idx    : int | None = None

        self.weights_st         = DEFAULT_WEIGHTS.copy()
        self.weights_order      = list(DEFAULT_WEIGHTS.keys())
        self.weights_enabled    = {k: True for k in DEFAULT_WEIGHTS}
        self.weights_last_value = DEFAULT_WEIGHTS.copy()
        self.solver_params      = DEFAULT_SOLVER_PARAMS.copy()

        self._build_cache: dict = {}

    def reset(self):
        self._init_defaults()

    @property
    def people_text(self) -> str:
        seen = set()
        result = []
        for members_text in self.groups_st.values():
            for name in members_text.split("\n"):
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    result.append(name)
        return "\n".join(result)

    def dims(self):
        people = _parse_lines(self.people_text)
        tasks  = _parse_lines(self.tasks_text)
        days   = _parse_lines(self.days_text)
        default_hrs = _parse_lines(DEFAULT_HOURS_TEXT)
        hours = {}
        for j in days:
            raw    = self.hours_per_day.get(j, DEFAULT_HOURS_TEXT)
            parsed = _parse_lines(raw)
            hours[j] = parsed if parsed else default_hrs
        return people, tasks, hours, days

    def build_groups(self, people: list) -> dict:
        people_set = set(people)
        assigned   = set()
        result     = {}
        for gname, members_text in self.groups_st.items():
            members = [m for m in _parse_lines(members_text) if m in people_set]
            if members:
                result[gname] = members
                assigned.update(members)
        unassigned = [p for p in people if p not in assigned]
        if unassigned:
            result["Default"] = unassigned
        return result or {"Default": list(people)}

    def sync_displacement_tasks(self, tasks: list):
        task_set = set(tasks)
        for t in [t for t in self.task_location_idx_st if t not in task_set]:
            del self.task_location_idx_st[t]
        n_locs = len(self.location_names_st)
        for t in tasks:
            if t not in self.task_location_idx_st:
                self.task_location_idx_st[t] = 0
            elif self.task_location_idx_st[t] >= n_locs:
                self.task_location_idx_st[t] = 0

    def build_task_location(self, tasks: list) -> dict:
        self.sync_displacement_tasks(tasks)
        locs = self.location_names_st
        return {t: locs[self.task_location_idx_st.get(t, 0)] for t in tasks}

    def build_travel_time(self) -> dict:
        locs = self.location_names_st
        result = {}
        for (i, j), v in self.travel_time_st.items():
            v_str = str(v).strip()
            if not v_str:
                continue
            try:
                k = int(v_str)
            except ValueError:
                continue
            if k > 0 and i < len(locs) and j < len(locs):
                result[(locs[i], locs[j])] = k
                result[(locs[j], locs[i])] = k
        return result

    def _dims_hash(self) -> int:
        return hash((
            self.people_text, self.tasks_text, self.days_text,
            tuple(sorted(self.hours_per_day.items())),
        ))

    def needs_rebuild(self, tab_name: str) -> bool:
        key = self._dims_hash()
        if self._build_cache.get(tab_name) == key:
            return False
        self._build_cache[tab_name] = key
        return True

    def invalidate_cache(self):
        self._build_cache.clear()

    @property
    def person_colors(self) -> dict:
        from constants import GROUP_HEADER_COLORS
        res = {}
        people = self.dims()[0]
        groups = self.build_groups(people)
        for idx, (gname, members) in enumerate(groups.items()):
            color = GROUP_HEADER_COLORS[idx % len(GROUP_HEADER_COLORS)]
            for p in members:
                res[p] = color
        return res

    def build_x_prev(self, new_people, new_tasks, new_hours, new_days) -> dict:
        X_prev = {
            (p, t, h, j): 0
            for p in new_people for t in new_tasks
            for j in new_days for h in new_hours[j]
        }
        if self.base_run_idx is None:
            return X_prev
        if not (0 <= self.base_run_idx < len(self.solution_history)):
            return X_prev
        entry     = self.solution_history[self.base_run_idx]
        base_asgn = entry["sol"]["assignment"]
        base_p_s  = set(entry["people"])
        base_t_s  = set(entry["tasks"])
        for p in new_people:
            if p not in base_p_s:
                continue
            for j in new_days:
                if j not in base_asgn:
                    continue
                for h in new_hours[j]:
                    assigned_task = base_asgn[j].get(p, {}).get(h)
                    for t in new_tasks:
                        if t not in base_t_s:
                            continue
                        X_prev[(p, t, h, j)] = 1 if assigned_task == t else 0
        return X_prev