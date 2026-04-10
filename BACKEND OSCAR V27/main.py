import flet as ft
from app import StaffSchedulerApp

def main(page: ft.Page):
    StaffSchedulerApp(page)

ft.app(target=main)