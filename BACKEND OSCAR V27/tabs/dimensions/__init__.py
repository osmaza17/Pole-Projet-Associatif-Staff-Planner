"""Dimensions tab package — people, groups, tasks, days, hours and rules."""

from .dimensions_tab   import DimensionsTab
from .max_consec_hours import MaxConsecHoursManager

__all__ = ["DimensionsTab", "MaxConsecHoursManager"]