import threading
from pathlib import Path

import flet as ft
from constants import (
    _s, SIDEBAR_WIDTH, SIDEBAR_BG, SIDEBAR_SELECTED_BG,
    SIDEBAR_TEXT_COLOR, SIDEBAR_SELECTED_TEXT,
    PRIMARY_BLUE, DANGER_RED, SUCCESS_GREEN, PURPLE,
    INFO_BLUE, DANGER_LIGHT, DIVIDER_COLOR,
    PROFILE_CARD_BG, PROFILE_EMPTY_FG,
    PROFILE_WATCHER_INTERVAL, DEFAULT_PROFILE_NAME,
)
from state import AppState
from solver_controller import SolverController
from profile_io import save_profile, load_profile
from tabs.dimensions    import DimensionsTab
from tabs.availability  import AvailabilityTab
from tabs.demand        import DemandTab
from tabs.skills        import SkillsTab
from tabs.rules         import RulesTab
from tabs.social        import SocialTab
from tabs.displacement  import DisplacementTab
from tabs.preferences   import PreferencesTab
from tabs.configuration import ConfigurationTab
from tabs.output        import OutputTab

_HERE     = Path(__file__).parent
_PROFILES = _HERE / "PROFILES"


class StaffSchedulerApp:

    def __init__(self, page: ft.Page):
        self.page       = page
        self.state      = AppState()
        self._ui_lock   = threading.Lock()
        self._solve_btn = None
        self._rules_tab = None
        self._configure_page()
        self._build_ui()

    def _configure_page(self):
        p = self.page
        p.title            = "Staff Scheduler"
        p.scroll           = None
        p.window.maximized = True
        p.theme_mode       = ft.ThemeMode.LIGHT
        p.padding          = 0

    # ── Generic UI helpers ───────────────────────────────────────────

    def _make_sidebar_button(self, label, icon, bg, on_click,
                             big=False, tooltip=None):
        icon_size = _s(18) if big else _s(14)
        text_size = _s(13) if big else _s(11)
        pad_v     = _s(8)  if big else _s(7)
        pad_h     = _s(12) if big else _s(6)
        radius    = 8      if big else 6
        spacing   = _s(6)  if big else _s(4)
        return ft.Container(
            content=ft.Row([
                ft.Icon(icon, color=ft.Colors.WHITE, size=icon_size),
                ft.Text(label, color=ft.Colors.WHITE,
                        weight=ft.FontWeight.BOLD, size=text_size),
            ], spacing=spacing, alignment=ft.MainAxisAlignment.CENTER),
            bgcolor=bg,
            padding=ft.padding.symmetric(horizontal=pad_h, vertical=pad_v),
            border_radius=radius, on_click=on_click,
            width=SIDEBAR_WIDTH - _s(32),
            alignment=ft.alignment.center,
            tooltip=tooltip)

    def _confirm_dialog(self, title, body, confirm_label,
                        on_confirm, danger=False):
        def _cancel(ev):
            self.page.close(dlg)

        def _confirm(ev):
            self.page.close(dlg)
            on_confirm()

        confirm_btn = ft.FilledButton(
            confirm_label, on_click=_confirm,
            style=ft.ButtonStyle(bgcolor=DANGER_RED) if danger else None)
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.BOLD, size=_s(15)),
            content=ft.Text(body, size=_s(13)) if isinstance(body, str) else body,
            actions=[ft.TextButton("Cancel", on_click=_cancel), confirm_btn],
            actions_alignment=ft.MainAxisAlignment.END)
        self.page.open(dlg)

    # ── Solve button state ───────────────────────────────────────────

    def _update_solve_blocked(self):
        s = self.state
        if self._rules_tab is not None:
            self._rules_tab.validate_rules()
        has_err = bool(s.validation_errors["demand"] or
                       s.validation_errors["consec"] or
                       s.validation_errors.get("rules", set()))
        s.solve_blocked = has_err
        if self._solve_btn is None:
            return
        blocked = has_err or s.solver_running
        self._solve_btn.bgcolor  = ft.Colors.GREY_500 if blocked else PRIMARY_BLUE
        self._solve_btn.disabled = blocked
        try:
            self._solve_btn.update()
        except Exception:
            pass

    def _show_snack(self, msg: str, color: str = SUCCESS_GREEN):
        self._snack.content = ft.Text(msg, color=ft.Colors.WHITE)
        self._snack.bgcolor = color
        self._snack.open    = True
        self._snack.update()

    # ── Save / Load / Reset profile ──────────────────────────────────

    def _do_save(self, e):
        tf = ft.TextField(
            label="Nombre del archivo", value=DEFAULT_PROFILE_NAME,
            hint_text="sin extensión .json", autofocus=True,
            width=_s(320), text_size=_s(13), suffix_text=".json")

        def _confirm(ev):
            name = tf.value.strip() or DEFAULT_PROFILE_NAME
            if not name.endswith(".json"):
                name += ".json"
            self.page.close(dlg)
            _PROFILES.mkdir(exist_ok=True)
            dest = _PROFILES / name
            try:
                dest.write_text(save_profile(self.state), encoding="utf-8")
                self._show_snack(f"✓  Guardado: {dest.name}")
                self._scan_profiles()
            except Exception as ex:
                self._show_snack(f"✗  Error al guardar: {ex}", DANGER_RED)

        def _cancel(ev):
            self.page.close(dlg)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Guardar perfil", weight=ft.FontWeight.BOLD, size=_s(15)),
            content=tf,
            actions=[ft.TextButton("Cancelar", on_click=_cancel),
                     ft.FilledButton("Guardar", on_click=_confirm)],
            actions_alignment=ft.MainAxisAlignment.END)
        self.page.open(dlg)

    def _do_delete_profile(self, path: Path):
        def _do_it():
            try:
                path.unlink()
                self._show_snack(f"✓  Eliminado: {path.name}")
                self._scan_profiles()
            except Exception as ex:
                self._show_snack(f"✗  Error al eliminar: {ex}", DANGER_RED)
        self._confirm_dialog(
            "Eliminar perfil",
            f'¿Seguro que quieres eliminar "{path.name}"?\nEsta acción no se puede deshacer.',
            "Eliminar", _do_it, danger=True)

    def _do_load(self, e):
        self._load_picker.pick_files(
            dialog_title="Cargar perfil",
            allowed_extensions=["json"], allow_multiple=False)

    def _on_load_result(self, e: ft.FilePickerResultEvent):
        if not e.files:
            return
        self._load_profile_from_path(Path(e.files[0].path))

    def _load_profile_from_path(self, path: Path):
        try:
            load_profile(self.state, path.read_text(encoding="utf-8"))
            self._apply_loaded_profile()
            self._show_snack(f"✓  Cargado: {path.name}")
        except Exception as ex:
            self._show_snack(f"✗  Error al cargar: {ex}", DANGER_RED)

    def _do_reset(self, e):
        def _do_it():
            self.state.reset()
            self._apply_loaded_profile()
            self._show_snack("✓  Inputs reseteados a valores por defecto")
        self._confirm_dialog(
            "Reset to defaults",
            "This will discard all current inputs and restore\n"
            "the original default values.\n\nAre you sure?",
            "Reset", _do_it, danger=True)

    def _apply_loaded_profile(self):
        s  = self.state
        dt = self._dims_tab
        dt.tf_tasks.value = s.tasks_text
        dt.tf_days.value  = s.days_text
        dt._build_groups_list()
        dt._build_hours_per_day()
        dt._build_rotation_list()
        self._rules_tab.rebuild_all()
        for k, tf in self._config_tab._param_tfs.items():
            tf.value = str(s.solver_params.get(k, ""))
        self._config_tab.build()
        self._output_tab.rebuild()
        self._update_solve_blocked()
        self._switch_page(0)

    # ── Profile panel ────────────────────────────────────────────────

    def _scan_profiles(self):
        try:
            files = sorted(_PROFILES.glob("*.json"),
                           key=lambda p: p.stat().st_mtime,
                           reverse=True) if _PROFILES.is_dir() else []
        except Exception:
            files = []

        btns = []
        for path in files:
            p = path
            btn = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.DESCRIPTION, size=_s(12), color=INFO_BLUE),
                    ft.Text(p.stem, size=_s(10), color=ft.Colors.WHITE,
                            no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                            expand=True, tooltip=p.name),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CLOSE, size=_s(11),
                                        color=DANGER_LIGHT),
                        width=_s(18), height=_s(18),
                        alignment=ft.alignment.center, border_radius=3,
                        tooltip=f"Eliminar {p.name}",
                        on_click=lambda e, _p=p: self._do_delete_profile(_p)),
                ], spacing=_s(3), vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=PROFILE_CARD_BG,
                border=ft.border.all(1, DIVIDER_COLOR),
                border_radius=4,
                padding=ft.padding.only(left=_s(6), top=_s(1),
                                        bottom=_s(1), right=_s(1)),
                on_click=lambda e, _p=p: self._load_profile_from_path(_p),
                ink=True, tooltip=f"Cargar {p.name}")
            btns.append(btn)

        if not btns:
            btns = [ft.Text("— ningún .json en PROFILES —",
                            size=_s(10), color=PROFILE_EMPTY_FG, italic=True)]

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
                    current = ({p.name for p in _PROFILES.glob("*.json")}
                               if _PROFILES.is_dir() else set())
                    if current != last_names:
                        last_names = current
                        self._scan_profiles()
                except Exception:
                    pass
                threading.Event().wait(PROFILE_WATCHER_INTERVAL)

        threading.Thread(target=_watch, daemon=True).start()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        s = self.state

        self._load_picker = ft.FilePicker(on_result=self._on_load_result)
        self._snack       = ft.SnackBar(content=ft.Text(""), open=False)
        self.page.overlay.extend([self._load_picker, self._snack])

        self._dims_tab   = DimensionsTab(s, self.page, self._update_solve_blocked)
        self._avail_tab  = AvailabilityTab(s, self.page, self._update_solve_blocked)
        self._demand_tab = DemandTab(s, self.page, self._update_solve_blocked)
        self._skills_tab = SkillsTab(s, self.page, self._update_solve_blocked)
        self._rules_tab  = RulesTab(s, self.page, self._update_solve_blocked)
        self._social_tab = SocialTab(s, self.page)
        self._disp_tab   = DisplacementTab(s, self.page)
        self._pref_tab   = PreferencesTab(s, self.page)
        self._config_tab = ConfigurationTab(s, self.page)
        self._output_tab = OutputTab(s, self.page)

        self._solve_btn = self._make_sidebar_button(
            "SOLVE", ft.Icons.PLAY_ARROW, PRIMARY_BLUE, self._do_solve, big=True)
        self._stop_btn = self._make_sidebar_button(
            "STOP", ft.Icons.STOP, DANGER_RED, self._do_stop, big=True)

        self._live_callbacks_sw = ft.Switch(
            label="Live Preview", value=True,
            label_style=ft.TextStyle(color=ft.Colors.WHITE, size=_s(11)),
            tooltip="Show intermediate solutions while solving (slightly slower)")

        self._save_btn = self._make_sidebar_button(
            "Save", ft.Icons.SAVE_ALT, SUCCESS_GREEN, self._do_save,
            tooltip=f"Guardar perfil en {_PROFILES}")
        self._load_btn = self._make_sidebar_button(
            "Load from other folder", ft.Icons.UPLOAD_FILE, PURPLE, self._do_load,
            tooltip="Cargar perfil desde JSON (cualquier carpeta)")
        self._reset_btn = self._make_sidebar_button(
            "Reset to defaults", ft.Icons.RESTART_ALT, DANGER_RED, self._do_reset,
            tooltip="Resetear todos los inputs a los valores por defecto")

        self._solver = SolverController(
            state=s, page=self.page,
            output_tab=self._output_tab, pref_tab=self._pref_tab,
            on_solve_blocked_update=self._update_solve_blocked,
            switch_page_cb=self._switch_page,
            ui_lock=self._ui_lock,
            live_callbacks_sw=self._live_callbacks_sw)

        self._page_contents = {
            0: self._dims_tab.get_container(),
            1: self._avail_tab.get_container(),
            2: self._demand_tab.get_container(),
            3: self._skills_tab.get_container(),
            4: self._rules_tab.get_container(),
            5: self._social_tab.get_container(),
            6: self._disp_tab.get_container(),
            7: self._pref_tab.get_container(),
            8: self._config_tab.get_container(),
            9: self._output_tab.get_container(),
        }
        self._builders = {
            1: self._avail_tab.build,    2: self._demand_tab.build,
            3: self._skills_tab.build,   4: self._rules_tab.build,
            5: self._social_tab.build,   6: self._disp_tab.build,
            7: self._pref_tab.build,     8: self._config_tab.build,
        }

        _menu_def = [
            ("Dimensions",    ft.Icons.GRID_VIEW,       0),
            ("Availability",  ft.Icons.EVENT_AVAILABLE, 1),
            ("Demand",        ft.Icons.TRENDING_UP,     2),
            ("Skills",        ft.Icons.STAR_BORDER,     3),
            ("Rules",         ft.Icons.RULE,            4),
            ("Social",        ft.Icons.PEOPLE_OUTLINE,  5),
            ("Displacement",  ft.Icons.DIRECTIONS_CAR,  6),
            ("Preferences",   ft.Icons.SORT,            7),
            ("Configuration", ft.Icons.SETTINGS,        8),
            ("Output",        ft.Icons.ASSESSMENT,      9),
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

        self._profiles_lv = ft.ListView(spacing=_s(2), expand=True)

        profiles_panel = ft.Column(
            controls=[
                ft.Divider(color=DIVIDER_COLOR, height=1),
                ft.Column(
                    controls=[self._save_btn, self._load_btn, self._reset_btn],
                    spacing=_s(4),
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.FOLDER_OPEN, size=_s(13), color=INFO_BLUE),
                        ft.Text("Perfiles", color=INFO_BLUE,
                                size=_s(11), weight=ft.FontWeight.BOLD, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.REFRESH, icon_color=INFO_BLUE,
                            icon_size=_s(14), tooltip="Actualizar lista",
                            on_click=lambda e: self._scan_profiles(),
                            style=ft.ButtonStyle(
                                padding=ft.padding.all(_s(2)),
                                shape=ft.RoundedRectangleBorder(radius=4))),
                    ],
                    spacing=_s(4),
                    vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self._profiles_lv,
            ],
            spacing=_s(4), height=_s(400))

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
                        ft.Column([self._solve_btn, self._live_callbacks_sw,
                                   self._stop_btn], spacing=_s(6)),
                        padding=ft.padding.only(bottom=_s(8))),
                    ft.Divider(color=DIVIDER_COLOR, height=1),
                    ft.Column(controls=self._menu_btn_refs,
                              spacing=_s(2), scroll=ft.ScrollMode.AUTO,
                              expand=True),
                    profiles_panel,
                ],
                spacing=_s(4), expand=True),
            border=ft.border.only(right=ft.border.BorderSide(1, DIVIDER_COLOR)))

        self.page.add(ft.Row(
            controls=[sidebar, self._content_area],
            spacing=0, expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START))

        self._dims_tab.initial_build()
        self._rules_tab.initial_build()

        self._scan_profiles()
        self._start_profile_watcher()

    # ── Navigation ───────────────────────────────────────────────────

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