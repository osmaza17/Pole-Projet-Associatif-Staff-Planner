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

    def _do_solve(self, e):
        self._solver.do_solve(e)

    def _do_stop(self, e):
        self._solver.do_stop(e)