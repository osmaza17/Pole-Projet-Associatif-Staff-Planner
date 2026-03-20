from constants import DEFAULT_WEIGHTS, DEFAULT_SOLVER_PARAMS, DEFAULT_HOURS_TEXT


class AppState:
    _DEFAULT_PEOPLE = (
        "Arnaud\nNina\nJoseph\nChloé\nNiels\nZiad\nTristan\nJasmin\nMarine\nNoé\n"
        "Jules R\nGuillaume\nNoémie\nStanislas\nTérence\nManon In\nManon L\nPauline\n"
        "Lucie\nDarius\nMattia\nPierre\nBaptiste B\nVincent\nMadeleine\nIlhan\nMatteo\n"
        "Alexandre L\nPablo\nJenaya\nLiv\nFaustin\nKenza\nJuliette B\nSarah\n"
        "Alexandre B\nRémi\nGabi\nJeanne B\nMatthieu A\nInès\nMaxime N\nAriane\nMatthias"
    )
    _DEFAULT_TASKS = (
        "Iron\nRenew\nClean\nBackup\nDiscard\nSweep\nArchive\nCall\nLoad\nPolish\n"
        "Replace\nDefrost\nShake\nExercise\nPrepare\nInventory\nSew\nShine\nSort\n"
        "Recycle\nProgram\nWash\nRefuel\nBuy\nManage\nClean\nMeditate"
    )
    _DEFAULT_DAYS = "Mon\nTue\nWed"

    def __init__(self):
        self._init_defaults()

    def _init_defaults(self):
        """Set (or reset) every attribute to its factory default."""
        self.tasks_text   = self._DEFAULT_TASKS
        self.days_text    = self._DEFAULT_DAYS

        self.avail_st     : dict = {}
        self.demand_st    : dict = {}
        self.skills_st    : dict = {}
        self.force_st     : dict = {}
        self.just_work_st : dict = {}   # (p, h, d) → 0/1
        self.social_st    : dict = {}
        self.quota_st     : dict = {}
        self.rotation_st  : dict = {}
        self.hard_enemies : bool = False
        self.hours_per_day: dict = {}
        # {group_name: "member1\nmember2\n..."}
        # Always starts with at least one group; people are defined here.
        self.groups_st    : dict = {"Group 1": self._DEFAULT_PEOPLE}
        self.captain_rules: list = []   # [{captains:[…], tasks:[…], hours:{day:[…]}}]
        self.mandatory_rules : list = []

        self.consec_global_val   : str  = ""
        self.consec_global_rest  : str  = "1"   # global min rest hours
        self.consec_per_person   : dict = {}    # {person: str}  individual limit
        self.consec_rest_per_person: dict = {}  # {person: str}  individual rest

        # Set of person names whose consec values are individually overridden.
        # Persons NOT in this set use the global values.
        self.consec_personalized_persons: set = set()

        # Legacy field kept for backward compat during load; no longer used
        # by the UI or solver. See profile_io.load_profile() migration.
        self.consec_personalized : bool = False

        self.avail_filter  = None
        self.demand_filter = None
        self.force_filter  = [None, None]
        self.quota_filter  = None

        self.validation_errors: dict = {
            "demand": set(), "quota": set(), "consec": set()
        }

        self.solve_blocked     : bool = False
        self.solver_running    : bool = False
        self.running_model_ref : list = [None]

        self.solution_history: list = []
        self.diff_state      : dict = {"ref": None, "cmp": None}
        self.base_run_idx    : int | None = None

        self.weights_st      = DEFAULT_WEIGHTS.copy()
        self.weights_order   = list(DEFAULT_WEIGHTS.keys())
        self.weights_enabled = {k: True for k in DEFAULT_WEIGHTS}
        # Stores the last non-zero value for each weight so it can be
        # restored when re-enabling after a disable (value=0).
        self.weights_last_value = DEFAULT_WEIGHTS.copy()
        self.solver_params   = DEFAULT_SOLVER_PARAMS.copy()

        self._build_cache: dict = {}

    def reset(self):
        """Reset the entire state to factory defaults (in-place)."""
        self._init_defaults()

    # ── Dimension helpers ─────────────────────────────────────────────

    @property
    def people_text(self) -> str:
        """Derived from groups_st for backward-compatibility with other tabs."""
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
        def _parse(txt):
            return list(dict.fromkeys(x.strip() for x in txt.split("\n") if x.strip()))
        people = _parse(self.people_text)
        tasks  = _parse(self.tasks_text)
        days   = _parse(self.days_text)
        default_hrs = _parse(DEFAULT_HOURS_TEXT)
        hours = {}
        for j in days:
            raw    = self.hours_per_day.get(j, DEFAULT_HOURS_TEXT)
            parsed = _parse(raw)
            hours[j] = parsed if parsed else default_hrs
        return people, tasks, hours, days

    def build_groups(self, people: list) -> dict:
        """Return {group_name: [list_of_people]}.
        People not in any group go into an implicit 'Default' group."""
        def _parse(txt):
            return [x.strip() for x in txt.split("\n") if x.strip()]

        people_set = set(people)
        assigned   = set()
        result     = {}

        for gname, members_text in self.groups_st.items():
            members = [m for m in _parse(members_text) if m in people_set]
            if members:
                result[gname] = members
                assigned.update(members)

        unassigned = [p for p in people if p not in assigned]
        if unassigned:
            result["Default"] = unassigned

        return result or {"Default": list(people)}

    # ── Cache helpers ─────────────────────────────────────────────────

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
        # We need to build the effective groups mapped structure
        # dims() returns people, tasks, hours, days. We need just people.
        people = self.dims()[0]
        groups = self.build_groups(people)
        for idx, (gname, members) in enumerate(groups.items()):
            color = GROUP_HEADER_COLORS[idx % len(GROUP_HEADER_COLORS)]
            for p in members:
                res[p] = color
        return res

    # ── X_prev builder ────────────────────────────────────────────────

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
        entry         = self.solution_history[self.base_run_idx]
        base_asgn     = entry["sol"]["assignment"]
        base_people_s = set(entry["people"])
        base_tasks_s  = set(entry["tasks"])
        for p in new_people:
            if p not in base_people_s:
                continue
            for j in new_days:
                if j not in base_asgn:
                    continue
                for h in new_hours[j]:
                    assigned_task = base_asgn[j].get(p, {}).get(h)
                    for t in new_tasks:
                        if t not in base_tasks_s:
                            continue
                        X_prev[(p, t, h, j)] = 1 if assigned_task == t else 0
        return X_prev