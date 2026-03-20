import threading
from pathlib import Path

import flet as ft
from constants import (
    _s, SIDEBAR_WIDTH, SIDEBAR_BG, SIDEBAR_SELECTED_BG,
    SIDEBAR_TEXT_COLOR, SIDEBAR_SELECTED_TEXT,
)
from state import AppState
from solver_controller import SolverController
from profile_io import save_profile, load_profile
from presolve import run_presolve
from tabs.dimensions        import DimensionsTab
from tabs.availability_tab  import AvailabilityTab
from tabs.demand_tab        import DemandTab
from tabs.skills_tab        import SkillsTab
from tabs.quota_tab         import QuotaTab
from tabs.social_tab        import SocialTab
from tabs.configuration_tab import ConfigurationTab
from tabs.output            import OutputTab

_HERE     = Path(__file__).parent
_PROFILES = _HERE / "PROFILES"


class StaffSchedulerApp:

    def __init__(self, page: ft.Page):
        self.page      = page
        self.state     = AppState()
        self._ui_lock  = threading.Lock()
        self._solve_btn = None
        self._configure_page()
        self._build_ui()

    # ── Configuración de la página ────────────────────────────────────

    def _configure_page(self):
        p = self.page
        p.title            = "Staff Scheduler"
        p.scroll           = None
        p.window.maximized = True
        p.theme_mode       = ft.ThemeMode.LIGHT
        p.padding          = 0

    # ── Botón Solve ───────────────────────────────────────────────────

    def _update_solve_blocked(self):
        s = self.state

        if hasattr(self, '_dims_tab'):
            self._dims_tab.validate_rules()

        has_err = bool(s.validation_errors["demand"] or
                       s.validation_errors["quota"]  or
                       s.validation_errors["consec"] or
                       s.validation_errors.get("rules", set()))
        s.solve_blocked = has_err
        if self._solve_btn is None:
            return
        blocked = has_err or s.solver_running
        self._solve_btn.bgcolor  = ft.Colors.GREY_500 if blocked else "#1565C0"
        self._solve_btn.disabled = blocked
        try:
            self._solve_btn.update()
        except Exception:
            pass

    # ── Snack helper ──────────────────────────────────────────────────

    def _show_snack(self, msg: str, color: str = ft.Colors.GREEN_700):
        self._snack.content = ft.Text(msg, color=ft.Colors.WHITE)
        self._snack.bgcolor = color
        self._snack.open    = True
        self._snack.update()

    # ── Guardar perfil ────────────────────────────────────────────────

    def _do_save(self, e):
        tf = ft.TextField(
            label="Nombre del archivo",
            value="scheduler_profile",
            hint_text="sin extensión .json",
            autofocus=True,
            width=_s(320),
            text_size=_s(13),
            suffix_text=".json",
        )

        def _confirm(ev):
            name = tf.value.strip() or "scheduler_profile"
            if not name.endswith(".json"):
                name += ".json"
            self.page.close(dlg)
            _PROFILES.mkdir(exist_ok=True)
            dest = _PROFILES / name
            try:
                json_str = save_profile(self.state)
                dest.write_text(json_str, encoding="utf-8")
                self._show_snack(f"✓  Guardado: {dest.name}")
                self._scan_profiles()
            except Exception as ex:
                self._show_snack(f"✗  Error al guardar: {ex}", ft.Colors.RED_700)

        def _cancel(ev):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Guardar perfil", weight=ft.FontWeight.BOLD, size=_s(15)),
            content=tf,
            actions=[
                ft.TextButton("Cancelar", on_click=_cancel),
                ft.FilledButton("Guardar", on_click=_confirm),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.open(dlg)

    # ── Borrar perfil ─────────────────────────────────────────────────

    def _do_delete_profile(self, path: Path):
        def _confirm(ev):
            self.page.close(dlg)
            try:
                path.unlink()
                self._show_snack(f"✓  Eliminado: {path.name}")
                self._scan_profiles()
            except Exception as ex:
                self._show_snack(f"✗  Error al eliminar: {ex}", ft.Colors.RED_700)

        def _cancel(ev):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Eliminar perfil", weight=ft.FontWeight.BOLD, size=_s(15)),
            content=ft.Text(
                f'¿Seguro que quieres eliminar "{path.name}"?\nEsta acción no se puede deshacer.',
                size=_s(13),
            ),
            actions=[
                ft.TextButton("Cancelar", on_click=_cancel),
                ft.FilledButton(
                    "Eliminar",
                    on_click=_confirm,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.RED_700),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.open(dlg)

    # ── Cargar perfil (FilePicker) ────────────────────────────────────

    def _do_load(self, e):
        self._load_picker.pick_files(
            dialog_title="Cargar perfil",
            allowed_extensions=["json"],
            allow_multiple=False,
        )

    def _on_load_result(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        self._load_profile_from_path(Path(e.files[0].path))

    # ── Cargar perfil desde ruta ──────────────────────────────────────

    def _load_profile_from_path(self, path: Path):
        try:
            json_str = path.read_text(encoding="utf-8")
            load_profile(self.state, json_str)
            self._apply_loaded_profile()
            self._show_snack(f"✓  Cargado: {path.name}")
        except Exception as ex:
            self._show_snack(f"✗  Error al cargar: {ex}", ft.Colors.RED_700)

    # ── Reset a defaults ─────────────────────────────────────────────

    def _do_reset(self, e):
        def _confirm(ev):
            self.page.close(dlg)
            self.state.reset()
            self._apply_loaded_profile()
            self._show_snack("✓  Inputs reseteados a valores por defecto")

        def _cancel(ev):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Reset to defaults", weight=ft.FontWeight.BOLD, size=_s(15)),
            content=ft.Text(
                "This will discard all current inputs and restore\n"
                "the original default values.\n\n"
                "Are you sure?",
                size=_s(13),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.FilledButton(
                    "Reset",
                    on_click=_confirm,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.RED_700),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.open(dlg)

    def _apply_loaded_profile(self):
        s  = self.state
        dt = self._dims_tab

        dt.tf_tasks.value = s.tasks_text
        dt.tf_days.value  = s.days_text

        dt._build_groups_list()
        dt._build_hours_per_day()
        dt._build_rotation_list()
        dt._build_rest_list()
        dt._build_captain_rules_summary()
        dt._build_mandatory_rules_summary()

        for k, tf in self._config_tab._param_tfs.items():
            tf.value = str(s.solver_params.get(k, ""))
        self._config_tab.build()

        self._output_tab.rebuild()
        self._update_solve_blocked()
        self._switch_page(0)

    # ── Panel de perfiles: escaneo y botones ──────────────────────────

    def _scan_profiles(self):
        try:
            files = sorted(_PROFILES.glob("*.json"), key=lambda p: p.stat().st_mtime,
                           reverse=True) if _PROFILES.is_dir() else []
        except Exception:
            files = []

        btns = []
        for path in files:
            p = path
            btn = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.DESCRIPTION, size=_s(12), color="#90CAF9"),
                    ft.Text(
                        p.stem, size=_s(10), color=ft.Colors.WHITE,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True, tooltip=p.name,
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CLOSE, size=_s(11),
                                        color="#EF9A9A"),
                        width=_s(18), height=_s(18),
                        alignment=ft.alignment.center,
                        border_radius=3,
                        tooltip=f"Eliminar {p.name}",
                        on_click=lambda e, _p=p: self._do_delete_profile(_p),
                    ),
                ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor="#2E3D49",
                border=ft.border.all(1, "#455A64"),
                border_radius=4,
                padding=ft.padding.only(
                    left=_s(6), top=_s(1), bottom=_s(1), right=_s(1)),
                on_click=lambda e, _p=p: self._load_profile_from_path(_p),
                ink=True, tooltip=f"Cargar {p.name}",
            )
            btns.append(btn)

        if not btns:
            btns = [ft.Text("— ningún .json en PROFILES —",
                            size=_s(10), color="#607D8B", italic=True)]

        self._profiles_lv.controls = btns
        try:
            self._profiles_lv.update()
        except Exception:
            pass

    def _start_profile_watcher(self):
        def _watch():
            last_names: set = set()
            while True:
                try:
                    current = {p.name for p in _PROFILES.glob("*.json")} if _PROFILES.is_dir() else set()
                    if current != last_names:
                        last_names = current
                        self._scan_profiles()
                except Exception:
                    pass
                threading.Event().wait(2.5)

        t = threading.Thread(target=_watch, daemon=True)
        t.start()

    # ── Construcción de la UI ─────────────────────────────────────────

    def _build_ui(self):
        s = self.state

        # ── File pickers y snack ──────────────────────────────────────
        self._load_picker = ft.FilePicker(on_result=self._on_load_result)
        self._snack       = ft.SnackBar(content=ft.Text(""), open=False)
        self.page.overlay.extend([self._load_picker, self._snack])

        # ── Tabs ──────────────────────────────────────────────────────
        self._dims_tab   = DimensionsTab(s, self.page, self._update_solve_blocked)
        self._avail_tab  = AvailabilityTab(s, self.page, self._update_solve_blocked)
        self._demand_tab = DemandTab(s, self.page, self._update_solve_blocked)
        self._skills_tab = SkillsTab(s, self.page, self._update_solve_blocked)
        self._quota_tab  = QuotaTab(s, self.page, self._update_solve_blocked)
        self._social_tab = SocialTab(s, self.page)
        self._config_tab = ConfigurationTab(s, self.page)
        self._output_tab = OutputTab(s, self.page)

        # ── Solve / Stop ──────────────────────────────────────────────
        self._solve_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE, size=_s(18)),
                ft.Text("SOLVE", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(13)),
            ], spacing=_s(6), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#1565C0",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=8, on_click=self._do_solve,
            width=SIDEBAR_WIDTH - _s(32), alignment=ft.alignment.center)

        self._stop_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE, size=_s(18)),
                ft.Text("STOP", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(13)),
            ], spacing=_s(6), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#C62828",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(8)),
            border_radius=8, on_click=self._do_stop,
            width=SIDEBAR_WIDTH - _s(32), alignment=ft.alignment.center)

        self._presolve_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SEARCH, color=ft.Colors.WHITE, size=_s(16)),
                ft.Text("PRESOLVE", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(12)),
            ], spacing=_s(6), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#E65100",
            padding=ft.padding.symmetric(horizontal=_s(12), vertical=_s(6)),
            border_radius=8, on_click=self._do_presolve,
            width=SIDEBAR_WIDTH - _s(32), alignment=ft.alignment.center,
            tooltip="Check feasibility: can demand be covered with available staff?")

        self._live_callbacks_sw = ft.Switch(
            label="Live Preview", value=True,
            label_style=ft.TextStyle(color=ft.Colors.WHITE, size=_s(11)),
            tooltip="Show intermediate solutions while solving (slightly slower)")

        # ── Save / Load buttons (apilados verticalmente) ──────────────
        _btn_w = SIDEBAR_WIDTH - _s(32)

        self._save_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SAVE_ALT, color=ft.Colors.WHITE, size=_s(14)),
                ft.Text("Save", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(11)),
            ], spacing=_s(4), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#2E7D32",
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(7)),
            border_radius=6, on_click=self._do_save,
            width=_btn_w, alignment=ft.alignment.center,
            tooltip=f"Guardar perfil en {_PROFILES}")

        self._load_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.UPLOAD_FILE, color=ft.Colors.WHITE, size=_s(14)),
                ft.Text("Load from other folder", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(11)),
            ], spacing=_s(4), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#6A1B9A",
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(7)),
            border_radius=6, on_click=self._do_load,
            width=_btn_w, alignment=ft.alignment.center,
            tooltip="Cargar perfil desde JSON (cualquier carpeta)")

        self._reset_btn = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.RESTART_ALT, color=ft.Colors.WHITE, size=_s(14)),
                ft.Text("Reset to defaults", color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=_s(11)),
            ], spacing=_s(4), alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#C62828",
            padding=ft.padding.symmetric(horizontal=_s(6), vertical=_s(7)),
            border_radius=6, on_click=self._do_reset,
            width=_btn_w, alignment=ft.alignment.center,
            tooltip="Resetear todos los inputs a los valores por defecto")

        # ── Solver controller ─────────────────────────────────────────
        self._solver = SolverController(
            state=s, page=self.page,
            output_tab=self._output_tab,
            on_solve_blocked_update=self._update_solve_blocked,
            switch_page_cb=self._switch_page,
            ui_lock=self._ui_lock,
            live_callbacks_sw=self._live_callbacks_sw)

        # ── Page contents & builders ──────────────────────────────────
        self._page_contents = {
            0: self._dims_tab.get_container(),
            1: self._avail_tab.get_container(),
            2: self._demand_tab.get_container(),
            3: self._skills_tab.get_container(),
            4: self._quota_tab.get_container(),
            5: self._social_tab.get_container(),
            6: self._config_tab.get_container(),
            7: self._output_tab.get_container(),
        }
        self._builders = {
            1: self._avail_tab.build,
            2: self._demand_tab.build,
            3: self._skills_tab.build,
            4: self._quota_tab.build,
            5: self._social_tab.build,
            6: self._config_tab.build,
        }

        # ── Menú de navegación ────────────────────────────────────────
        _menu_def = [
            ("Dimensions",    ft.Icons.GRID_VIEW,            0),
            ("Availability",  ft.Icons.EVENT_AVAILABLE,      1),
            ("Demand",        ft.Icons.TRENDING_UP,          2),
            ("Skills",        ft.Icons.STAR_BORDER,          3),
            ("Quota",         ft.Icons.FORMAT_LIST_NUMBERED, 4),
            ("Social",        ft.Icons.PEOPLE_OUTLINE,       5),
            ("Configuration", ft.Icons.SETTINGS,             6),
            ("Output",        ft.Icons.ASSESSMENT,           7),
        ]
        self._selected_idx  = 0
        self._menu_btn_refs = []
        self._content_area  = ft.Container(
            expand=True, padding=0, content=self._page_contents[0])

        for label, icon, idx in _menu_def:
            sel = (idx == 0)
            btn = ft.Container(
                content=ft.Row([
                    ft.Icon(icon,
                            color=SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR,
                            size=_s(20)),
                    ft.Text(label, size=_s(13),
                            color=SIDEBAR_SELECTED_TEXT if sel else SIDEBAR_TEXT_COLOR,
                            weight=ft.FontWeight.BOLD if sel else None),
                ], spacing=_s(10)),
                padding=ft.padding.symmetric(horizontal=_s(16), vertical=_s(12)),
                border_radius=8,
                bgcolor=SIDEBAR_SELECTED_BG if sel else None,
                on_click=self._on_menu_click, data=idx, ink=True)
            self._menu_btn_refs.append(btn)

        # ── Panel de perfiles ─────────────────────────────────────────
        self._profiles_lv = ft.ListView(spacing=_s(2), expand=True)

        profiles_panel = ft.Column(
            controls=[
                ft.Divider(color="#455A64", height=1),
                ft.Column(
                    controls=[self._save_btn, self._load_btn, self._reset_btn],
                    spacing=_s(4),
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.FOLDER_OPEN, size=_s(13), color="#90CAF9"),
                        ft.Text("Perfiles", color="#90CAF9",
                                size=_s(11), weight=ft.FontWeight.BOLD, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.REFRESH,
                            icon_color="#90CAF9",
                            icon_size=_s(14),
                            tooltip="Actualizar lista",
                            on_click=lambda e: self._scan_profiles(),
                            style=ft.ButtonStyle(
                                padding=ft.padding.all(_s(2)),
                                shape=ft.RoundedRectangleBorder(radius=4)),
                        ),
                    ],
                    spacing=_s(4),
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self._profiles_lv,
            ],
            spacing=_s(4),
            height=_s(400),
        )

        # ── Sidebar ───────────────────────────────────────────────────
        sidebar = ft.Container(
            width=SIDEBAR_WIDTH, bgcolor=SIDEBAR_BG,
            padding=ft.padding.only(top=_s(12), bottom=_s(12),
                                    left=_s(8), right=_s(8)),
            content=ft.Column(
                controls=[
                    ft.Container(
                        ft.Text("Staff Scheduler", size=_s(16),
                                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        padding=ft.padding.only(left=_s(8), bottom=_s(4))),
                    ft.Container(
                        ft.Column([
                            self._solve_btn,
                            self._presolve_btn,
                            self._live_callbacks_sw,
                            self._stop_btn,
                        ], spacing=_s(6)),
                        padding=ft.padding.only(bottom=_s(8))),
                    ft.Divider(color="#455A64", height=1),
                    ft.Column(
                        controls=self._menu_btn_refs,
                        spacing=_s(2),
                        scroll=ft.ScrollMode.AUTO,
                        expand=True),
                    profiles_panel,
                ],
                spacing=_s(4), expand=True),
            border=ft.border.only(right=ft.border.BorderSide(1, "#455A64")))

        self.page.add(ft.Row(
            controls=[sidebar, self._content_area],
            spacing=0, expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START))

        self._dims_tab.initial_build()

        self._scan_profiles()
        self._start_profile_watcher()

    # ── Navegación ────────────────────────────────────────────────────

    def _on_menu_click(self, e):
        self._switch_page(e.control.data)

    def _switch_page(self, idx: int):
        self._selected_idx = idx
        self._content_area.content = self._page_contents[idx]
        for i, btn in enumerate(self._menu_btn_refs):
            sel = (i == idx)
            btn.bgcolor                    = SIDEBAR_SELECTED_BG if sel else None
            btn.content.controls[0].color  = (SIDEBAR_SELECTED_TEXT if sel
                                               else SIDEBAR_TEXT_COLOR)
            btn.content.controls[1].color  = (SIDEBAR_SELECTED_TEXT if sel
                                               else SIDEBAR_TEXT_COLOR)
            btn.content.controls[1].weight = ft.FontWeight.BOLD if sel else None
        if idx in self._builders:
            with self._ui_lock:
                self._builders[idx]()
        self.page.update()

    def _do_presolve(self, e):
        """Launch feasibility LP in a background thread; show result dialog."""
        s = self.state
        if s.solver_running:
            return

        people, tasks, hours, days = s.dims()

        availability: dict = {}
        for p in people:
            for j in days:
                for h in hours[j]:
                    v = s.avail_st.get((p, h, j), 1)
                    availability[(p, h, j)] = 1 if v in (1, 2) else 0

        demand: dict = {}
        for t in tasks:
            for j in days:
                for h in hours[j]:
                    raw = s.demand_st.get((t, h, j), "1").strip()
                    try:    demand[(t, h, j)] = int(raw) if raw else 0
                    except: demand[(t, h, j)] = 0

        skills = {(p, t): s.skills_st.get((p, t), 1)
                  for p in people for t in tasks}

        # ── max_consec_hours: same logic as solver_controller ─────────
        max_consec: dict = {}
        global_limit_raw = s.consec_global_val.strip()
        global_rest_raw  = s.consec_global_rest.strip()

        def _parse_int(raw: str):
            raw = raw.strip()
            if not raw:
                return None
            v = int(raw)
            return max(1, v)

        for p in people:
            if p in s.consec_personalized_persons:
                raw_limit = s.consec_per_person.get(p, "").strip()
                raw_rest  = s.consec_rest_per_person.get(p, "").strip()
            else:
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

        data = dict(
            people=people, tasks=tasks, hours=hours, days=days,
            availability=availability, skills=skills, demand=demand,
            hard_enemies=s.hard_enemies,
            social={
                (p1, p2): s.social_st.get((p1, p2), 0)
                for i, p1 in enumerate(people)
                for p2 in people[i + 1:]
            },
            max_consec_hours=max_consec,
        )

        self._presolve_btn.bgcolor  = "#78909C"
        self._presolve_btn.disabled = True
        self._presolve_btn.content.controls[1].value = "Checking..."
        self._presolve_btn.update()

        def _run():
            try:
                feasible, result = run_presolve(data)
            except Exception as ex:
                self.page.run_thread(
                    lambda: self._show_presolve_result(None, {}, str(ex)))
                return
            self.page.run_thread(
                lambda f=feasible, r=result: self._show_presolve_result(f, r))

        threading.Thread(target=_run, daemon=True).start()

    # ── helpers for _show_presolve_result ─────────────────────────────

    # ── Presolve dialog helpers ───────────────────────────────────────
    # All sizes are intentionally larger than the rest of the UI for
    # maximum readability in the dialog.

    @staticmethod
    def _ps_badge(text: str, bg: str) -> ft.Container:
        """Coloured pill badge (e.g. demand / eligible counts)."""
        return ft.Container(
            ft.Text(text, size=14, color=ft.Colors.WHITE,
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER),
            width=44, height=28, bgcolor=bg,
            border_radius=6, alignment=ft.alignment.center)

    @staticmethod
    def _ps_section_title(icon: str, text: str,
                          color: str = "#263238") -> ft.Container:
        """Bold section header with a coloured left bar."""
        return ft.Container(
            ft.Row([
                ft.Text(icon, size=18),
                ft.Text(text, size=15, weight=ft.FontWeight.BOLD,
                        color=color),
            ], spacing=8),
            padding=ft.padding.only(top=6, bottom=2))

    @staticmethod
    def _ps_box(content: ft.Control,
                border_color: str = "#FFCDD2",
                bg: str = "#FFF8F8") -> ft.Container:
        return ft.Container(
            content,
            border=ft.border.all(2, border_color),
            border_radius=8,
            padding=12,
            bgcolor=bg,
            clip_behavior=ft.ClipBehavior.HARD_EDGE)

    def _show_presolve_result(self, feasible, result: dict, error: str = ""):
        """Restore button state and open result dialog."""
        self._presolve_btn.bgcolor  = "#E65100"
        self._presolve_btn.disabled = False
        self._presolve_btn.content.controls[1].value = "PRESOLVE"
        self._presolve_btn.update()

        # ── Error ─────────────────────────────────────────────────────
        if feasible is None:
            body = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.WARNING_ROUNDED,
                            color=ft.Colors.ORANGE_700, size=32),
                    ft.Text("Gurobi error during presolve",
                            size=16, weight=ft.FontWeight.BOLD,
                            color=ft.Colors.ORANGE_700),
                ], spacing=10),
                ft.Text(error, size=13, selectable=True, color="#37474F"),
            ], spacing=12, tight=True)
            title_text  = "⚠  Presolve Error"
            title_color = ft.Colors.ORANGE_700

        # ── Feasible ──────────────────────────────────────────────────
        elif feasible:
            s = self.state
            checks = [
                "✓  Every task has enough available, skilled staff at every hour.",
                "✓  No person is the sole cover for two tasks at the same time.",
            ]
            if s.hard_enemies and any(
                    v == -1 for v in s.social_st.values()):
                checks.append(
                    "✓  Hard-enemy pairs do not block full coverage.")
            if hasattr(s, "consec_global_val") and s.consec_global_val.strip():
                checks.append(
                    "✓  Consecutive-hours limits do not block full coverage.")
            body = ft.Column([
                ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED,
                        color=ft.Colors.GREEN_700, size=64),
                ft.Text("The schedule is feasible!",
                        size=20, weight=ft.FontWeight.BOLD,
                        color=ft.Colors.GREEN_800),
                ft.Container(height=4),
                ft.Text(
                    "All demand can be fully covered given the current "
                    "configuration.",
                    size=14, color="#2E7D32",
                    text_align=ft.TextAlign.CENTER),
                ft.Container(height=8),
                self._ps_box(
                    ft.Text("\n".join(checks), size=13, color="#1B5E20"),
                    border_color="#A5D6A7", bg="#F1F8E9"),
            ], spacing=8, tight=True,
               horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            title_text  = "✓  Feasible"
            title_color = ft.Colors.GREEN_700

        # ── Infeasible ────────────────────────────────────────────────
        else:
            iis              = result.get("iis", {})
            bottlenecks      = result.get("simple_bottlenecks", [])
            cover_slots      = iis.get("cover_slots", [])
            conflicts        = iis.get("person_conflicts", [])
            hour_groups      = iis.get("hour_groups", [])
            enemy_conflicts  = iis.get("enemy_conflicts", [])
            consec_conflicts = iis.get("consec_conflicts", [])

            n_causes = sum([
                bool(bottlenecks), bool(conflicts),
                bool(enemy_conflicts), bool(consec_conflicts),
            ])

            sections = []

            # ── Banner ────────────────────────────────────────────────
            cause_summary = (
                f"{n_causes} type(s) of conflict detected — "
                "see details below."
                if n_causes else
                "See details below for possible causes."
            )
            sections.append(ft.Container(
                ft.Row([
                    ft.Icon(ft.Icons.CANCEL_ROUNDED,
                            color=ft.Colors.WHITE, size=28),
                    ft.Column([
                        ft.Text("This schedule cannot be solved as configured.",
                                size=16, weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE),
                        ft.Text(cause_summary, size=12, color="#FFCDD2"),
                    ], spacing=2, tight=True),
                ], spacing=12),
                bgcolor="#C62828", border_radius=8,
                padding=ft.padding.symmetric(horizontal=16, vertical=12)))

            # ── Section 1: Not enough staff ───────────────────────────
            if bottlenecks:
                sections.append(self._ps_section_title(
                    "🔴",
                    f"Not enough qualified staff  —  {len(bottlenecks)} slot(s)",
                    color="#B71C1C"))
                sections.append(ft.Text(
                    "Each row below is a time slot where the number of people "
                    "who are both available AND have the right skill is lower "
                    "than the number of staff required.\n"
                    "Fix: mark more people as available, assign the skill to "
                    "more people, or reduce demand for that slot.",
                    size=12, color="#546E7A"))

                COL = [60, 60, 110, 50, 50]
                def _th(label, w): return ft.Container(
                    ft.Text(label, size=12, weight=ft.FontWeight.BOLD,
                            color="#37474F"),
                    width=w)
                hdr = ft.Row([
                    _th("Day",  COL[0]),
                    _th("Hour", COL[1]),
                    _th("Task", COL[2]),
                    _th("Need", COL[3]),
                    _th("Have", COL[4]),
                    ft.Text("Eligible staff",
                            size=12, weight=ft.FontWeight.BOLD,
                            color="#37474F", expand=True),
                ], spacing=6)

                trows = [hdr, ft.Divider(height=1, color="#FFCDD2")]
                for pb in bottlenecks:
                    nobody = pb["eligible"] == 0
                    names_txt = (
                        ", ".join(pb["names"]) if pb["names"]
                        else "nobody has this skill")
                    trows.append(ft.Container(
                        ft.Row([
                            ft.Container(
                                ft.Text(pb["day"], size=13, no_wrap=True,
                                        weight=ft.FontWeight.BOLD),
                                width=COL[0]),
                            ft.Container(
                                ft.Text(pb["hour"], size=13, no_wrap=True),
                                width=COL[1]),
                            ft.Container(
                                ft.Text(pb["task"], size=13, no_wrap=True,
                                        overflow=ft.TextOverflow.ELLIPSIS,
                                        weight=ft.FontWeight.BOLD,
                                        color="#1565C0"),
                                width=COL[2]),
                            self._ps_badge(str(pb["required"]), "#C62828"),
                            self._ps_badge(
                                str(pb["eligible"]),
                                "#E65100" if pb["eligible"] > 0
                                else "#546E7A"),
                            ft.Text(names_txt, size=12, expand=True,
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    color="#546E7A", italic=nobody),
                        ], spacing=6),
                        bgcolor="#FFF3F3" if nobody else None,
                        border_radius=4,
                        padding=ft.padding.symmetric(vertical=3)))

                sections.append(self._ps_box(
                    ft.Column(trows, spacing=4, tight=True)))

            # ── Section 2: Hour-by-hour breakdown ─────────────────────
            if hour_groups:
                sections.append(self._ps_section_title(
                    "🔍",
                    f"Hour-by-hour breakdown  —  {len(hour_groups)} critical hour(s)",
                    color="#1565C0"))
                sections.append(ft.Text(
                    "These are the hours where it is mathematically impossible "
                    "to cover all demand at the same time.\n"
                    "For each hour, you can see the demand per task and how "
                    "many eligible staff exist for it.",
                    size=12, color="#546E7A"))

                hour_rows = []
                for hg in hour_groups:
                    # ok only if every individual task has enough eligible staff
                    all_tasks_ok = all(
                        tk["eligible"] >= tk["demand"]
                        for tk in hg["tasks"])
                    hour_rows.append(ft.Container(
                        ft.Row([
                            ft.Icon(ft.Icons.SCHEDULE,
                                    size=18, color="#1565C0"),
                            ft.Text(f"{hg['day']}   {hg['hour']}",
                                    size=15, weight=ft.FontWeight.BOLD,
                                    color="#1565C0"),
                            ft.Container(expand=True),
                            ft.Text("Total demand:", size=12, color="#546E7A"),
                            self._ps_badge(str(hg["total_demand"]), "#C62828"),
                            ft.Text("Unique staff available:",
                                    size=12, color="#546E7A"),
                            self._ps_badge(
                                str(hg["unique_eligible"]),
                                "#2E7D32" if all_tasks_ok else "#E65100"),
                        ], spacing=8),
                        bgcolor="#E3F2FD", border_radius=6,
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        margin=ft.margin.only(top=6)))

                    for tk in hg["tasks"]:
                        names_txt = (", ".join(tk["names"])
                                     if tk["names"] else "nobody")
                        enough = tk["eligible"] >= tk["demand"]
                        hour_rows.append(ft.Container(
                            ft.Row([
                                ft.Container(width=20),
                                ft.Icon(
                                    ft.Icons.CHECK_CIRCLE if enough
                                    else ft.Icons.ERROR,
                                    size=16,
                                    color=(ft.Colors.GREEN_600 if enough
                                           else ft.Colors.RED_700)),
                                ft.Container(
                                    ft.Text(tk["task"], size=13,
                                            weight=ft.FontWeight.BOLD,
                                            no_wrap=True,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                            color=(ft.Colors.GREEN_800 if enough
                                                   else ft.Colors.RED_900)),
                                    width=130),
                                ft.Text("needs", size=12, color="#78909C"),
                                self._ps_badge(str(tk["demand"]), "#C62828"),
                                ft.Text("eligible:", size=12, color="#78909C"),
                                self._ps_badge(
                                    str(tk["eligible"]),
                                    "#2E7D32" if enough
                                    else "#E65100" if tk["eligible"] > 0
                                    else "#546E7A"),
                                ft.Text(f"  {names_txt}", size=12,
                                        color="#546E7A", expand=True,
                                        no_wrap=True,
                                        overflow=ft.TextOverflow.ELLIPSIS),
                            ], spacing=8),
                            bgcolor="#FFF3F3" if not enough else None,
                            border_radius=4,
                            padding=ft.padding.symmetric(vertical=4)))

                sections.append(self._ps_box(
                    ft.Column(hour_rows, spacing=2, tight=True),
                    border_color="#BBDEFB", bg="#F5F9FF"))

            # ── Section 3: One person needed in two places ────────────
            if conflicts:
                sections.append(self._ps_section_title(
                    "⚠️",
                    f"One person needed in two places  —  {len(conflicts)} case(s)",
                    color="#E65100"))
                sections.append(ft.Text(
                    "Each person below is the sole eligible staff for two "
                    "different tasks at the same hour. Because a person can "
                    "only be assigned to one task at a time, both tasks "
                    "cannot be covered simultaneously.\n"
                    "Fix: train additional staff in one of the two tasks, "
                    "or reduce demand so only one of the tasks needs coverage "
                    "at that hour.",
                    size=12, color="#546E7A"))

                cf_rows = []
                for cf in conflicts:
                    tasks_str = "  and  ".join(
                        f'"{t}"' for t in cf["competing_tasks"]
                    ) or "multiple tasks"
                    cf_rows.append(ft.Container(
                        ft.Row([
                            ft.Icon(ft.Icons.PERSON_OFF,
                                    size=20, color="#E65100"),
                            ft.Column([
                                ft.Row([
                                    ft.Text(cf["person"], size=14,
                                            weight=ft.FontWeight.BOLD,
                                            color="#BF360C"),
                                    ft.Text("  ·  ", size=14,
                                            color="#78909C"),
                                    ft.Text(f"{cf['day']}  {cf['hour']}",
                                            size=14, color="#37474F"),
                                ], spacing=0),
                                ft.Text(
                                    f"Only person who can cover both:  {tasks_str}",
                                    size=12, color="#546E7A"),
                            ], spacing=2, tight=True),
                        ], spacing=12),
                        bgcolor="#FFF8F0",
                        border=ft.border.all(1, "#FFCC80"),
                        border_radius=6,
                        padding=10))

                sections.append(self._ps_box(
                    ft.Column(cf_rows, spacing=6, tight=True),
                    border_color="#FFE0B2", bg="#FFFBF5"))

            # ── Section 4: Hard-enemy conflicts ───────────────────────
            if enemy_conflicts:
                sections.append(self._ps_section_title(
                    "🚫",
                    f"Enemy pair blocks coverage  —  {len(enemy_conflicts)} pair(s)",
                    color="#6A1B9A"))
                sections.append(ft.Text(
                    "These two people are marked as hard enemies — they are "
                    "forbidden from working the same task at the same hour. "
                    "But together they are the only staff who can cover the "
                    "required demand: neither alone is enough, and they "
                    "cannot work together.\n"
                    "Fix: switch to soft enemies (penalised, not forbidden), "
                    "add a third person with the required skill, or reduce "
                    "demand at the affected slots.",
                    size=12, color="#546E7A"))

                en_rows = []
                for ec in enemy_conflicts:
                    slots_str = ",  ".join(
                        f"{s['task']} @ {s['hour']}"
                        for s in ec["blocked_slots"]) or "multiple slots"
                    en_rows.append(ft.Container(
                        ft.Row([
                            ft.Icon(ft.Icons.BLOCK, size=20,
                                    color="#6A1B9A"),
                            ft.Column([
                                ft.Row([
                                    ft.Text(ec["person1"], size=14,
                                            weight=ft.FontWeight.BOLD,
                                            color="#4A148C"),
                                    ft.Text("  ✕  ", size=14, color="#78909C"),
                                    ft.Text(ec["person2"], size=14,
                                            weight=ft.FontWeight.BOLD,
                                            color="#4A148C"),
                                    ft.Text(f"  —  {ec['day']}", size=13,
                                            color="#37474F"),
                                ], spacing=0),
                                ft.Text(
                                    f"Slots they would need to share:  {slots_str}",
                                    size=12, color="#546E7A"),
                            ], spacing=2, tight=True),
                        ], spacing=12),
                        bgcolor="#F3E5F5",
                        border=ft.border.all(1, "#CE93D8"),
                        border_radius=6,
                        padding=10))

                sections.append(self._ps_box(
                    ft.Column(en_rows, spacing=6, tight=True),
                    border_color="#CE93D8", bg="#FCF4FF"))

            # ── Section 5: Consecutive-hours conflicts ────────────────
            if consec_conflicts:
                sections.append(self._ps_section_title(
                    "⏱️",
                    f"Rest rule blocks coverage  —  {len(consec_conflicts)} case(s)",
                    color="#00695C"))
                sections.append(ft.Text(
                    "The maximum consecutive hours rule requires certain "
                    "people to take a break within a specific time window. "
                    "But during that window, they are the only person "
                    "who can cover a task that has demand.\n"
                    "Fix: increase the consecutive-hours limit for that "
                    "person, add more staff who can cover that task, "
                    "or remove demand during the forced rest slot.",
                    size=12, color="#546E7A"))

                cc_rows = []
                for cc in consec_conflicts:
                    first_h = cc["window_hours"][0]
                    last_h  = cc["window_hours"][-1]
                    blocked_str = ",  ".join(
                        f"{s['task']} @ {s['hour']}"
                        for s in cc["blocked_coverage"]) or "—"
                    cc_rows.append(ft.Container(
                        ft.Row([
                            ft.Icon(ft.Icons.HOURGLASS_DISABLED,
                                    size=20, color="#00695C"),
                            ft.Column([
                                ft.Row([
                                    ft.Text(cc["person"], size=14,
                                            weight=ft.FontWeight.BOLD,
                                            color="#004D40"),
                                    ft.Text(f"  —  {cc['day']}", size=13,
                                            color="#37474F"),
                                ], spacing=0),
                                ft.Text(
                                    f"Window {first_h} → {last_h}:  "
                                    f"max {cc['max_work']} consecutive hours, "
                                    f"then {cc['min_rest']}h rest required.",
                                    size=12, color="#546E7A"),
                                ft.Text(
                                    f"Tasks only this person can cover "
                                    f"during rest:  {blocked_str}",
                                    size=12, color="#00695C",
                                    weight=ft.FontWeight.BOLD),
                            ], spacing=2, tight=True),
                        ], spacing=12),
                        bgcolor="#E0F2F1",
                        border=ft.border.all(1, "#80CBC4"),
                        border_radius=6,
                        padding=10))

                sections.append(self._ps_box(
                    ft.Column(cc_rows, spacing=6, tight=True),
                    border_color="#80CBC4", bg="#F0FAFA"))

            # ── Section 6: Fallback (IIS returned nothing useful) ─────
            if not bottlenecks and not conflicts \
                    and not enemy_conflicts and not consec_conflicts:
                sections.append(self._ps_box(
                    ft.Column([
                        ft.Text(
                            "No obvious single cause detected.",
                            size=14, weight=ft.FontWeight.BOLD,
                            color="#37474F"),
                        ft.Text(
                            "Every individual task slot has enough eligible "
                            "staff when considered alone, but the combination "
                            "of constraints makes it impossible to assign "
                            "everyone at the same time.\n\n"
                            "Things to check:\n"
                            "  •  Skills tab — are skills assigned to too "
                            "few people?\n"
                            "  •  Availability tab — are critical people "
                            "unavailable at key hours?\n"
                            "  •  Demand tab — is demand concentrated in "
                            "hours where the staff pool is thin?",
                            size=13, color="#546E7A"),
                    ], spacing=8, tight=True)))

            body = ft.Column(sections, spacing=12, tight=True)
            title_text  = "✗  Infeasible"
            title_color = ft.Colors.RED_700

        def _close(ev):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title_text, weight=ft.FontWeight.BOLD,
                          size=18, color=title_color),
            content=ft.Container(
                ft.Column(
                    [body],
                    scroll=ft.ScrollMode.AUTO,
                    tight=True),
                width=740,
                height=min(780, self.page.window.height - 160)
                       if feasible is False else None,
                padding=ft.padding.only(top=8, right=6)),
            actions=[
                ft.FilledButton(
                    "Close",
                    on_click=_close,
                    style=ft.ButtonStyle(
                        text_style=ft.TextStyle(size=14,
                                                weight=ft.FontWeight.BOLD),
                        padding=ft.padding.symmetric(
                            horizontal=24, vertical=10)))],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.open(dlg)

    def _do_solve(self, e):
        self._solver.do_solve(e)

    def _do_stop(self, e):
        self._solver.do_stop(e)