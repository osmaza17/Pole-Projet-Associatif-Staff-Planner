import flet as ft
from constants import _s, GROUP_HEADER_COLORS
from ui_helpers import UIHelpers
from tabs.base_tab import BaseTab

_LBL  = {0: "~", 1: "+", -1: "-"}
_TIP  = {0: "Neutral", 1: "Friend", -1: "Enemy"}
_CLR  = {0: "#B0B0B0", 1: "#2E7D32", -1: "#C62828"}
_NEXT = {0: 1, 1: -1, -1: 0}

# ── Notion-inspired palette ─────────────────────────────────────
_BG           = "#FFFFFF"
_BG_SIDEBAR   = "#FBFBFA"
_BG_HOVER     = "#F1F1EF"
_BG_SELECTED  = "#E3EDFB"
_BORDER       = "#E3E2DE"
_TEXT_PRI     = "#37352F"
_TEXT_SEC     = "#9B9A97"
_ACCENT       = "#2383E2"

_LEFT_W       = 230          # sidebar width
_TOGGLE_W     = 44
_TOGGLE_H     = 30


class SocialTab(BaseTab):
    """Two-panel social-relationship editor (master → detail)."""

    def __init__(self, state, page: ft.Page, on_solve_blocked_update=None):
        super().__init__(state, page)
        self._on_solve_blocked_update = on_solve_blocked_update or (lambda: None)

        # Data
        self._pidx: dict[str, int] = {}
        self._people: list[str] = []
        self._pgc: dict[str, str] = {}          # person → group colour

        # Selection / filter state
        self._selected: str | None = None
        self._search_q: str = ""
        self._rel_filter: str = "all"            # all | friends | enemies

        # Widget refs – left
        self._left_col   = ft.Column(spacing=1, expand=True,
                                     scroll=ft.ScrollMode.AUTO)
        self._left_items: dict[str, ft.Container] = {}

        # Widget refs – right
        self._right_col       = ft.Column(spacing=0, expand=True,
                                          scroll=ft.ScrollMode.AUTO)
        self._right_items: dict[str, dict] = {}  # person → {"ct","btn"}
        self._right_name_txt  = ft.Text("", size=_s(15), weight=ft.FontWeight.W_600,
                                        color=_TEXT_SEC)
        self._summary_txt     = ft.Text("", size=_s(11), color=_TEXT_SEC)
        self._right_search_q: str = ""
        self._mounted: bool = False   # True after first page.update()

    # ================================================================
    #  Key / value helpers  (canonical key keeps lower pidx first)
    # ================================================================
    def _key(self, a: str, b: str):
        return (a, b) if self._pidx[a] < self._pidx[b] else (b, a)

    def _get(self, a: str, b: str) -> int:
        return self.state.social_st.get(self._key(a, b), 0)

    def _put(self, a: str, b: str, v: int):
        self.state.social_st[self._key(a, b)] = v

    def _upd(self, *controls):
        """Call .update() only when controls are already on the page."""
        if not self._mounted:
            return
        for c in controls:
            c.update()

    # ================================================================
    #  Full rebuild (called when data changes)
    # ================================================================
    def _rebuild(self):
        self.state.invalidate_cache()
        self._on_solve_blocked_update()
        self.build()

    # ================================================================
    #  LEFT PANEL – person list
    # ================================================================
    def _build_left_item(self, p: str) -> ft.Container:
        grp_color = self._pgc.get(p, _TEXT_PRI)

        # badge counts
        fr = sum(1 for o in self._people if o != p and self._get(p, o) == 1)
        en = sum(1 for o in self._people if o != p and self._get(p, o) == -1)
        badge_parts: list[ft.Control] = []
        if fr:
            badge_parts.append(ft.Text(f"+{fr}", size=_s(10), color=_CLR[1],
                                       weight=ft.FontWeight.W_600))
        if en:
            badge_parts.append(ft.Text(f"-{en}", size=_s(10), color=_CLR[-1],
                                       weight=ft.FontWeight.W_600))

        row = ft.Row(
            [
                ft.Container(width=4, height=20, border_radius=2,
                             bgcolor=grp_color),
                ft.Text(p, size=_s(12), color=_TEXT_PRI,
                        weight=ft.FontWeight.W_500,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                        expand=True),
                *badge_parts,
            ],
            spacing=6,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        is_sel = (p == self._selected)
        ct = ft.Container(
            content=row,
            padding=ft.padding.symmetric(horizontal=8, vertical=7),
            bgcolor=_BG_SELECTED if is_sel else _BG_SIDEBAR,
            border_radius=6,
            on_click=lambda _, _p=p: self._select_person(_p),
            on_hover=lambda e, _p=p: self._hover_left(e, _p),
            animate=ft.Animation(100, ft.AnimationCurve.EASE_IN),
        )
        self._left_items[p] = ct
        return ct

    def _hover_left(self, e, p: str):
        ct = self._left_items.get(p)
        if ct is None or p == self._selected:
            return
        ct.bgcolor = _BG_HOVER if e.data == "true" else _BG_SIDEBAR
        self._upd(ct)

    # ── search filter ───────────────────────────────────────────
    def _on_search(self, e):
        self._search_q = (e.control.value or "").strip().lower()
        self._apply_left_filter()

    def _apply_left_filter(self):
        q = self._search_q
        for p, ct in self._left_items.items():
            ct.visible = (q in p.lower()) if q else True
        self._upd(self._left_col)

    # ── selection ───────────────────────────────────────────────
    def _select_person(self, p: str):
        prev = self._selected
        self._selected = p

        # highlight swap
        if prev and prev in self._left_items:
            self._left_items[prev].bgcolor = _BG_SIDEBAR
            self._upd(self._left_items[prev])
        if p in self._left_items:
            self._left_items[p].bgcolor = _BG_SELECTED
            self._upd(self._left_items[p])

        # refresh right panel
        self._right_name_txt.value = p
        self._right_name_txt.color = self._pgc.get(p, _TEXT_PRI)
        self._upd(self._right_name_txt)
        self._refresh_right()

    # ================================================================
    #  RIGHT PANEL – relationships for the selected person
    # ================================================================
    def _make_toggle(self, other: str) -> ft.Container:
        sel = self._selected
        v = self._get(sel, other) if sel else 0
        btn = ft.Container(
            content=ft.Text(_LBL[v], color=ft.Colors.WHITE,
                            size=_s(13), weight=ft.FontWeight.BOLD),
            width=_TOGGLE_W, height=_TOGGLE_H,
            bgcolor=_CLR[v],
            alignment=ft.alignment.center,
            border_radius=6,
            on_click=lambda _, _o=other: self._cycle(_o),
            tooltip=_TIP[v],
            animate=ft.Animation(100, ft.AnimationCurve.EASE),
        )
        return btn

    def _build_right_row(self, other: str) -> ft.Container:
        grp_color = self._pgc.get(other, _TEXT_PRI)
        btn = self._make_toggle(other)

        ct = ft.Container(
            content=ft.Row(
                [
                    ft.Container(width=4, height=18, border_radius=2,
                                 bgcolor=grp_color),
                    ft.Text(other, size=_s(12), color=_TEXT_PRI,
                            expand=True, no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS),
                    btn,
                ],
                spacing=8,
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=12, vertical=5),
            border=ft.border.only(bottom=ft.BorderSide(1, _BORDER)),
        )
        self._right_items[other] = {"ct": ct, "btn": btn}
        return ct

    def _refresh_right(self):
        """Rebuild right-panel rows for the current selection."""
        self._right_items.clear()
        sel = self._selected
        if sel is None:
            self._right_col.controls = [
                ft.Container(
                    ft.Text("← Select a person to edit relationships",
                            size=_s(13), color=_TEXT_SEC, italic=True),
                    padding=30, alignment=ft.alignment.center,
                )
            ]
            self._summary_txt.value = ""
            self._upd(self._right_col)
            self._upd(self._summary_txt)
            return

        rows = [self._build_right_row(p)
                for p in self._people if p != sel]
        self._right_col.controls = rows
        self._upd(self._right_col)
        self._update_summary()
        self._apply_right_filter()

    # ── cycle / bulk ────────────────────────────────────────────
    def _cycle(self, other: str):
        sel = self._selected
        if sel is None:
            return
        cur = self._get(sel, other)
        nv  = _NEXT[cur]
        self._put(sel, other, nv)

        info = self._right_items.get(other)
        if info:
            info["btn"].content.value = _LBL[nv]
            info["btn"].bgcolor       = _CLR[nv]
            info["btn"].tooltip       = _TIP[nv]
            self._upd(info["btn"])

        # Update left-panel badge for both persons
        self._refresh_left_badge(sel)
        self._refresh_left_badge(other)

        self._update_summary()
        self.state.invalidate_cache()
        self._on_solve_blocked_update()

    def _set_all(self, val: int):
        sel = self._selected
        if sel is None:
            return
        for p in self._people:
            if p == sel:
                continue
            self._put(sel, p, val)
            info = self._right_items.get(p)
            if info:
                info["btn"].content.value = _LBL[val]
                info["btn"].bgcolor       = _CLR[val]
                info["btn"].tooltip       = _TIP[val]
                self._upd(info["btn"])

        # Refresh all left badges
        for p in self._people:
            self._refresh_left_badge(p)

        self._update_summary()
        self._apply_right_filter()
        self.state.invalidate_cache()
        self._on_solve_blocked_update()

    # ── summary / filter ────────────────────────────────────────
    def _update_summary(self):
        sel = self._selected
        if sel is None:
            self._summary_txt.value = ""
            self._upd(self._summary_txt)
            return
        fr = sum(1 for p in self._people if p != sel and self._get(sel, p) == 1)
        en = sum(1 for p in self._people if p != sel and self._get(sel, p) == -1)
        self._summary_txt.value = (
            f"{fr} friend{'s' if fr != 1 else ''}  ·  "
            f"{en} enem{'ies' if en != 1 else 'y'}"
        )
        self._upd(self._summary_txt)

    def _on_rel_filter(self, e):
        self._rel_filter = e.control.value
        self._apply_right_filter()

    def _on_right_search(self, e):
        self._right_search_q = (e.control.value or "").strip().lower()
        self._apply_right_filter()

    def _apply_right_filter(self):
        sel = self._selected
        if sel is None:
            return
        q = self._right_search_q
        for p, info in self._right_items.items():
            show = True
            if self._rel_filter == "friends":
                show = self._get(sel, p) == 1
            elif self._rel_filter == "enemies":
                show = self._get(sel, p) == -1
            if show and q:
                show = q in p.lower()
            info["ct"].visible = show
        self._upd(self._right_col)

    # ── left badge refresh ──────────────────────────────────────
    def _refresh_left_badge(self, p: str):
        """Rebuild just the badge counts on a left-panel item."""
        ct = self._left_items.get(p)
        if ct is None:
            return
        row: ft.Row = ct.content
        # keep first 2 controls (color bar + name), replace rest
        base = row.controls[:2]
        fr = sum(1 for o in self._people if o != p and self._get(p, o) == 1)
        en = sum(1 for o in self._people if o != p and self._get(p, o) == -1)
        if fr:
            base.append(ft.Text(f"+{fr}", size=_s(10), color=_CLR[1],
                                weight=ft.FontWeight.W_600))
        if en:
            base.append(ft.Text(f"-{en}", size=_s(10), color=_CLR[-1],
                                weight=ft.FontWeight.W_600))
        row.controls = base
        self._upd(ct)

    # ================================================================
    #  BUILD  (called on data changes)
    # ================================================================
    def build(self):
        s = self.state
        if not s.needs_rebuild("social"):
            return
        people, _, _, _ = s.dims()

        self._mounted = False

        if len(people) < 2:
            self._ct.controls = []
            self.page.update()
            self._mounted = True
            return

        self._pidx  = {p: i for i, p in enumerate(people)}
        self._people = list(people)

        # group colours
        groups = s.build_groups(people)
        self._pgc.clear()
        for g_idx, (_, members) in enumerate(groups.items()):
            clr = GROUP_HEADER_COLORS[g_idx % len(GROUP_HEADER_COLORS)]
            for p in members:
                self._pgc[p] = clr

        # reset selection if person removed
        if self._selected not in self._pidx:
            self._selected = None

        self._left_items.clear()
        self._right_items.clear()
        self._search_q = ""
        self._right_search_q = ""
        self._rel_filter = "all"

        # ── Top bar ─────────────────────────────────────────────
        def on_reset(_):
            s.social_st.clear()
            s.hard_enemies = False
            self._rebuild()

        sw_hard = ft.Switch(
            label="Enemies: Hard Constraint",
            value=s.hard_enemies,
            on_change=lambda e: setattr(s, "hard_enemies", e.control.value),
            label_style=ft.TextStyle(size=_s(12), color=_TEXT_PRI),
        )
        top_bar = ft.Row([
            UIHelpers.make_reset_btn("Reset to Default", on_reset),
            sw_hard,
        ], spacing=16)

        # ── Left panel ──────────────────────────────────────────
        search_left = ft.TextField(
            hint_text="Search people…",
            hint_style=ft.TextStyle(size=_s(11), color=_TEXT_SEC),
            text_size=_s(12),
            dense=True,
            content_padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=6,
            border_color=_BORDER,
            focused_border_color=_ACCENT,
            on_change=self._on_search,
            prefix_icon=ft.Icons.SEARCH,
        )

        self._left_col.controls = [self._build_left_item(p) for p in people]

        left_panel = ft.Container(
            content=ft.Column([
                ft.Container(
                    ft.Row([
                        ft.Text("People", size=_s(13),
                                weight=ft.FontWeight.W_700, color=_TEXT_PRI),
                        ft.Text(str(len(people)), size=_s(11),
                                color=_TEXT_SEC),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.padding.symmetric(horizontal=10, vertical=4),
                ),
                search_left,
                self._left_col,
            ], spacing=6, expand=True),
            width=_LEFT_W,
            bgcolor=_BG_SIDEBAR,
            border=ft.border.all(1, _BORDER),
            border_radius=8,
            padding=8,
        )

        # ── Right panel ─────────────────────────────────────────
        self._right_name_txt.value = self._selected or "No selection"
        self._right_name_txt.color = (
            self._pgc.get(self._selected, _TEXT_PRI)
            if self._selected else _TEXT_SEC
        )

        filter_dd = ft.Dropdown(
            value="all",
            options=[
                ft.dropdown.Option("all", "All"),
                ft.dropdown.Option("friends", "Friends"),
                ft.dropdown.Option("enemies", "Enemies"),
            ],
            dense=True,
            content_padding=ft.padding.symmetric(horizontal=8, vertical=2),
            text_size=_s(11),
            width=110,
            border_radius=6,
            border_color=_BORDER,
            on_change=self._on_rel_filter,
        )

        search_right = ft.TextField(
            hint_text="Filter…",
            hint_style=ft.TextStyle(size=_s(11), color=_TEXT_SEC),
            text_size=_s(12),
            dense=True,
            content_padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border_radius=6,
            border_color=_BORDER,
            focused_border_color=_ACCENT,
            on_change=self._on_right_search,
            prefix_icon=ft.Icons.FILTER_LIST,
            width=160,
        )

        def _bulk_btn(label, color, val):
            return ft.Container(
                content=ft.Text(label, size=_s(11), color=ft.Colors.WHITE,
                                weight=ft.FontWeight.W_600),
                bgcolor=color, border_radius=5,
                padding=ft.padding.symmetric(horizontal=10, vertical=4),
                on_click=lambda _: self._set_all(val),
                tooltip=f"Set all to {_TIP[val]}",
            )

        bulk_row = ft.Row([
            _bulk_btn("All ~", _CLR[0], 0),
            _bulk_btn("All +", _CLR[1], 1),
            _bulk_btn("All −", _CLR[-1], -1),
        ], spacing=6)

        right_header = ft.Container(
            content=ft.Column([
                ft.Row([
                    self._right_name_txt,
                    self._summary_txt,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([
                    ft.Row([filter_dd, search_right], spacing=8),
                    bulk_row,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True),
            ], spacing=8),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            bgcolor=_BG,
            border=ft.border.only(bottom=ft.BorderSide(1, _BORDER)),
        )

        self._refresh_right()

        right_panel = ft.Container(
            content=ft.Column([
                right_header,
                self._right_col,
            ], spacing=0, expand=True),
            expand=True,
            bgcolor=_BG,
            border=ft.border.all(1, _BORDER),
            border_radius=8,
        )

        # ── Assemble ────────────────────────────────────────────
        panels = ft.Row([left_panel, right_panel],
                        spacing=12, expand=True,
                        vertical_alignment=ft.CrossAxisAlignment.START)

        self._ct.controls = [top_bar, ft.Container(content=panels, expand=True)]
        self._ct.expand = True
        self.page.update()
        self._mounted = True