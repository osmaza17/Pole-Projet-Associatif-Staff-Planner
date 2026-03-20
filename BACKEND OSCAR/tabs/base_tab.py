import flet as ft
from constants import _s
from ui_helpers import UIHelpers


class BaseTab:
    def __init__(self, state, page: ft.Page):
        self.state = state
        self.page  = page
        self._ct   = ft.ListView(spacing=_s(5))

    def set_matrix_columns(self, num_columns: int, base_width: int = 0):
        # We ensure the ListView content has enough width, adding ample padding so no column is cut.
        # W_LBL + buttons + N columns
        total_width = (base_width or UIHelpers.W_LBL) + UIHelpers.W_CELL + _s(100) + num_columns * (UIHelpers.W_CELL + 2)
        self._ct.width = total_width

    def build(self):
        raise NotImplementedError

    def get_container(self) -> ft.Container:
        return ft.Container(
            content=ft.Row(
                controls=[self._ct],
                scroll=ft.ScrollMode.ALWAYS,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.START),
            padding=_s(10), expand=True)