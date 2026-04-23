import flet as ft
from constants import _s, GROUP_HEADER_COLORS
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab


class SkillsTab(BaseTab):

    PAGE_SIZE   = 30
    MAX_CARD_W  = 1800
    _FONT_NAME  = _s(13)
    _FONT_TASK  = _s(12)
    _CHAR_WIDTH = _s(8)
    _NAME_PAD   = _s(14)
    _MIN_NAME_W = UIHelpers.W_LBL
    _MAX_NAME_W = _s(260)

    @staticmethod
    def _to_pastel(hex_color: str, factor: float = 0.35) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r * factor + 255 * (1 - factor))
        g = int(g * factor + 255 * (1 - factor))
        b = int(b * factor + 255 * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def _calc_name_width(cls, names) -> int:
        if not names:
            return cls._MIN_NAME_W
        max_chars = max(len(n) for n in names)
        w = max_chars * cls._CHAR_WIDTH + cls._NAME_PAD
        return max(cls._MIN_NAME_W, min(cls._MAX_NAME_W, w))

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._cell_map: dict = {}
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)
        self._gesture = {"origin": None, "visited": set()}
        self._group_grids: dict = {}
        self._page_index = 0

    # ── Bajo nivel: set + repintado ─────────────────────────────────
    def _set_cell(self, p, t, nv: int):
        self.state.skills_st[(p, t)] = nv
        cell = self._cell_map.get((p, t))
        if cell is not None:
            cell.content.value = str(nv)
            cell.bgcolor = UIHelpers._AVAIL_CLR[nv]
            cell.update()

    # ── Gesto: binario 0↔1 ──────────────────────────────────────────
    def _advance(self, p, t):
        cur = self.state.skills_st.get((p, t), 1)
        nv = 1 - cur
        self._set_cell(p, t, nv)
        self._on_solve_blocked_update()

    def _begin_gesture(self, p, t):
        self._gesture["origin"] = self.state.skills_st.get((p, t), 1)
        self._gesture["visited"] = {(p, t)}
        self._advance(p, t)

    def _continue_gesture(self, p, t):
        g = self._gesture
        if g["origin"] is None:
            return
        if (p, t) in g["visited"]:
            return
        if self.state.skills_st.get((p, t), 1) != g["origin"]:
            return
        g["visited"].add((p, t))
        self._advance(p, t)

    def _end_gesture(self):
        self._gesture["origin"] = None
        self._gesture["visited"] = set()

    # ── Coordenadas locales -> (p, t) para un grupo ─────────────────
    def _coords_to_cell(self, x: float, y: float, group_people: list, tasks: list):
        if not group_people or not tasks:
            return None
        W_CELL = UIHelpers.W_CELL
        H_BTN = UIHelpers.H_BTN
        if x < 0 or y < 0:
            return None
        c, c_off = divmod(int(x), W_CELL + 2)
        if c_off >= W_CELL or c < 0 or c >= len(tasks):
            return None
        r, r_off = divmod(int(y), H_BTN + 2)
        if r_off >= H_BTN or r < 0 or r >= len(group_people):
            return None
        return (group_people[r], tasks[c])

    def _notify_and_rebuild(self):
        self.state.invalidate_cache()
        self._on_solve_blocked_update()
        self.build()

    def _bulk_set(self, keys, value):
        for k in keys:
            self.state.skills_st[k] = value

    # ── Paginación ──────────────────────────────────────────────────
    def _paginate_groups(self, groups):
        all_people = []
        for gname, members in groups.items():
            for p in members:
                all_people.append((gname, p))

        total = len(all_people)
        total_pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

        if self._page_index >= total_pages:
            self._page_index = total_pages - 1
        if self._page_index < 0:
            self._page_index = 0

        start = self._page_index * self.PAGE_SIZE
        end = min(start + self.PAGE_SIZE, total)
        page_slice = all_people[start:end]

        from collections import OrderedDict
        paged_groups = OrderedDict()
        for gname, p in page_slice:
            paged_groups.setdefault(gname, []).append(p)

        return paged_groups, total, total_pages

    def _build_page_nav(self, total, total_pages):
        if total_pages <= 1:
            return ft.Container()

        start = self._page_index * self.PAGE_SIZE + 1
        end = min((self._page_index + 1) * self.PAGE_SIZE, total)

        def _prev(e):
            if self._page_index > 0:
                self._page_index -= 1
                self.build()

        def _next(e):
            if self._page_index < total_pages - 1:
                self._page_index += 1
                self.build()

        btn_prev = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT, icon_size=_s(20),
            tooltip="Página anterior", on_click=_prev,
            disabled=self._page_index == 0)
        btn_next = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT, icon_size=_s(20),
            tooltip="Página siguiente", on_click=_next,
            disabled=self._page_index >= total_pages - 1)
        label = ft.Text(
            f"Personas {start}–{end} de {total}  (Pág {self._page_index + 1}/{total_pages})",
            size=_s(11), weight=ft.FontWeight.W_500)

        return ft.Row([btn_prev, label, btn_next],
                      alignment=ft.MainAxisAlignment.CENTER, spacing=4)

    # ── Sección de un grupo ─────────────────────────────────────────
    def _build_group_section(self, gname, group_people, tasks, g_color):
        s = self.state
        W = UIHelpers.W_CELL
        name_w = self._calc_name_width(group_people)
        header_offset = name_w + W

        group_header = ft.Container(
            content=ft.Text(gname, size=_s(12), weight=ft.FontWeight.BOLD,
                            color=g_color),
            padding=ft.padding.only(top=_s(8), bottom=_s(2)))

        task_headers = [
            ft.Container(
                ft.Text(t, size=self._FONT_TASK, no_wrap=True,
                        overflow=ft.TextOverflow.CLIP),
                width=W, clip_behavior=ft.ClipBehavior.HARD_EDGE)
            for t in tasks
        ]

        def make_col_toggle(task, members):
            btn = UIHelpers.make_rc_btn("col")
            def click(_, _t=task, _members=members):
                nv = 1 - s.skills_st.get((_members[0], _t), 1)
                self._bulk_set(((p, _t) for p in _members), nv)
                self._notify_and_rebuild()
            btn.on_click = click
            return btn

        def make_row_toggle(person):
            btn = UIHelpers.make_rc_btn("row")
            def click(_, _p=person):
                nv = 1 - s.skills_st.get((_p, tasks[0]), 1)
                self._bulk_set(((_p, t) for t in tasks), nv)
                self._notify_and_rebuild()
            btn.on_click = click
            return btn

        person_rows = []
        name_widgets = []
        row_toggle_widgets = []
        for p in group_people:
            cell_widgets = []
            for t in tasks:
                toggle = UIHelpers.make_toggle(s.skills_st, (p, t), 1)
                toggle.on_click = None
                self._cell_map[(p, t)] = toggle
                cell_widgets.append(toggle)

            person_rows.append(ft.Row(cell_widgets, spacing=2, wrap=False))
            name_widgets.append(
                ft.Container(
                    content=ft.Text(
                        p, size=self._FONT_NAME, no_wrap=True,
                        color=s.person_colors.get(p),
                        weight=ft.FontWeight.BOLD if s.person_colors.get(p) else None),
                    width=name_w,
                    height=UIHelpers.H_BTN,
                    alignment=ft.alignment.center_left))
            row_toggle_widgets.append(make_row_toggle(p))

        names_col = ft.Column(name_widgets, spacing=2, tight=True)
        row_toggles_col = ft.Column(row_toggle_widgets, spacing=2, tight=True)
        grid_column = ft.Column(person_rows, spacing=2, tight=True)

        gp_list = list(group_people)

        def _on_pan_start(e, _gp=gp_list):
            rc = self._coords_to_cell(e.local_x, e.local_y, _gp, tasks)
            if rc is None:
                return
            if self._gesture["origin"] is None:
                self._begin_gesture(*rc)
            else:
                self._continue_gesture(*rc)

        def _on_pan_update(e, _gp=gp_list):
            rc = self._coords_to_cell(e.local_x, e.local_y, _gp, tasks)
            if rc is not None:
                self._continue_gesture(*rc)

        def _on_tap_down(e, _gp=gp_list):
            rc = self._coords_to_cell(e.local_x, e.local_y, _gp, tasks)
            if rc is None:
                return
            self._begin_gesture(*rc)

        grid_area = ft.GestureDetector(
            content=grid_column,
            drag_interval=10,
            on_tap_down=_on_tap_down,
            on_pan_start=_on_pan_start,
            on_pan_update=_on_pan_update,
            on_pan_end=lambda e: self._end_gesture())

        pastel = self._to_pastel(g_color)
        pad = _s(6)

        # ── Ancho natural del contenido y tope máximo ───────────────
        n_cols = len(tasks)
        content_w = header_offset + 4 + n_cols * (W + 2)
        natural_w = content_w + 2 * pad
        card_w = min(natural_w, self.MAX_CARD_W)

        inner_column = ft.Column([
            ft.Row([ft.Container(width=header_offset)] + task_headers,
                   spacing=2, wrap=False),
            ft.Row([ft.Container(width=header_offset)] +
                   [make_col_toggle(t, group_people) for t in tasks],
                   spacing=2, wrap=False),
            ft.Row([names_col, row_toggles_col, grid_area],
                   spacing=2, wrap=False),
        ], spacing=2, tight=True)

        # Si el contenido excede el ancho máximo, scroll interno
        if natural_w > self.MAX_CARD_W:
            box_content = ft.Row(
                [ft.Container(content=inner_column, width=content_w)],
                scroll=ft.ScrollMode.AUTO,
                wrap=False,
            )
        else:
            box_content = inner_column

        matrix_box = ft.Container(
            content=box_content,
            width=card_w,
            bgcolor=pastel,
            border_radius=6,
            padding=pad,
        )

        return [group_header, matrix_box]

    # ── Build ───────────────────────────────────────────────────────
    def build(self):
        s = self.state
        people, tasks, _, _ = s.dims()

        self._cell_map.clear()
        self._end_gesture()
        self._group_grids.clear()

        groups = s.build_groups(people)

        paged_groups, total_people, total_pages = self._paginate_groups(groups)

        # ── Max name width across ALL groups on this page ───────────
        max_name_w = self._MIN_NAME_W
        for members in paged_groups.values():
            if members:
                max_name_w = max(max_name_w, self._calc_name_width(members))

        def on_reset(_):
            s.skills_st.clear()
            self._notify_and_rebuild()

        buf = [
            ft.Text("Skills Matrix", weight=ft.FontWeight.BOLD, size=_s(16)),
            UIHelpers.make_reset_btn("Reset Skills", on_reset),
            self._build_page_nav(total_people, total_pages),
        ]

        original_group_names = list(groups.keys())
        for gname, members in paged_groups.items():
            if not members:
                continue
            g_idx = original_group_names.index(gname)
            g_color = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
            buf.extend(self._build_group_section(
                gname, members, tasks, g_color))

        if total_pages > 1:
            buf.append(self._build_page_nav(total_people, total_pages))

        # Limitar el ListView al ancho máximo de tarjeta + margen
        self.set_matrix_columns(len(tasks), base_width=max_name_w)
        self._ct.width = min(self._ct.width, self.MAX_CARD_W + _s(20))
        self._ct.controls = buf
        self.page.update()