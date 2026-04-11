SCALE = 0.75


def _s(v: float) -> int:
    return max(1, round(v * SCALE))


# ── Solver weights & params ──────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "W_COVERAGE": 1000000, "W_FORCE": 500000,
    "W_CAPTAIN": 500000,
    "W_STABILITY": 50000,  "W_EQ_GROUP": 10000, "W_GAP": 5000,
    "W_EQ_GLOBAL": 2000,   "W_EMERG": 1000,
    "W_VARIETY": 100,      "W_SOCIAL": 10,
    "W_QUOTA": 200000,     "W_PREF": 1,
    "W_STICKY": 2000,      "W_DURATION": 5000,
    "W_ROTATION": 5000,
    "W_TRAVEL": 5000,
}
SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

DEFAULT_SOLVER_PARAMS = {
    "TimeLimit": 1200, "MIPGap": 0.001, "MIPFocus": 2,
    "Threads": 0, "Presolve": 2, "Symmetry": 2,
    "Disconnected": 2, "IntegralityFocus": 1, "Method": 3,
    "Cuts": -1, "Heuristics": 0.05,
}

DEFAULT_HOURS_TEXT = (
    "08:00\n09:00\n10:00\n11:00\n12:00\n"
    "13:00\n14:00\n15:00\n16:00\n17:00"
)

# ── App configuration ────────────────────────────────────────────────
PROFILE_WATCHER_INTERVAL = 2.5     # seconds between profile folder scans
DEFAULT_PROFILE_NAME     = "scheduler_profile"

# ── Semantic color palette ───────────────────────────────────────────
PRIMARY_BLUE     = "#1565C0"
DANGER_RED       = "#C62828"
SUCCESS_GREEN    = "#2E7D32"
PURPLE           = "#6A1B9A"
INFO_BLUE        = "#90CAF9"
DANGER_LIGHT     = "#EF9A9A"

# Sidebar / profile panel
SIDEBAR_WIDTH         = _s(200)
SIDEBAR_BG            = "#263238"
SIDEBAR_SELECTED_BG   = "#37474F"
SIDEBAR_TEXT_COLOR    = "#ECEFF1"
SIDEBAR_SELECTED_TEXT = "#4FC3F7"
DIVIDER_COLOR         = "#455A64"
PROFILE_CARD_BG       = "#2E3D49"
PROFILE_EMPTY_FG      = "#607D8B"

# Tab bar (UIHelpers.make_tab_bar)
TAB_ACTIVE_BG    = PRIMARY_BLUE
TAB_ACTIVE_FG    = "#FFFFFF"
TAB_INACTIVE_BG  = "#CFD8DC"
TAB_INACTIVE_FG  = "#37474F"

# ── Task colors ──────────────────────────────────────────────────────
TASK_COLORS = [
    ("#CE93D8","#000000"), ("#80DEEA","#000000"), ("#FFF59D","#000000"),
    ("#A5D6A7","#000000"), ("#FFAB91","#000000"), ("#90CAF9","#000000"),
    ("#F48FB1","#000000"), ("#E6EE9C","#000000"), ("#B0BEC5","#000000"),
    ("#FFCC80","#000000"), ("#80CBC4","#000000"), ("#B39DDB","#000000"),
    ("#EF9A9A","#000000"), ("#C5E1A5","#000000"), ("#81D4FA","#000000"),
    ("#FFE082","#000000"), ("#F8BBD0","#000000"), ("#BCAAA4","#000000"),
    ("#A1887F","#FFFFFF"), ("#7986CB","#FFFFFF"), ("#4DB6AC","#FFFFFF"),
    ("#FF8A65","#000000"), ("#AED581","#000000"), ("#4FC3F7","#000000"),
    ("#DCE775","#000000"), ("#BA68C8","#FFFFFF"), ("#4DD0E1","#000000"),
    ("#E57373","#000000"), ("#9575CD","#FFFFFF"), ("#FFD54F","#000000"),
]

# ── Location colors ──────────────────────────────────────────────────
LOCATION_COLORS = [
    ("#FFCDD2", "#000000"), ("#C8E6C9", "#000000"),
    ("#BBDEFB", "#000000"), ("#FFF9C4", "#000000"),
    ("#D1C4E9", "#000000"), ("#FFE0B2", "#000000"),
    ("#B2DFDB", "#000000"), ("#F8BBD0", "#000000"),
]


def loc_color(idx: int) -> tuple[str, str]:
    return LOCATION_COLORS[idx % len(LOCATION_COLORS)]


GROUP_HEADER_COLORS = [
    "#1565C0", "#2E7D32", "#6A1B9A", "#BF360C",
    "#00695C", "#4527A0", "#AD1457", "#37474F",
]

# ── Output / availability semantic colors ────────────────────────────
UNAVAIL_COLOR     = "#D32F2F"
EMERG_COLOR       = "#F57C00"
AVAIL_COLOR       = "#388E3C"
DIFF_ADD_COLOR    = "#2E7D32"
DIFF_REMOVE_COLOR = "#C62828"
DIFF_CHANGE_COLOR = "#E65100"
TRAVEL_COLOR      = "#78909C"
TRAVEL_FG_COLOR   = "#FFFFFF"
TRAVEL_LABEL      = "TRAVEL"

BASE_ACTIVE_BG = "#2E7D32"
BASE_ACTIVE_FG = "#FFFFFF"
BASE_IDLE_BG   = "#A5D6A7"
BASE_IDLE_FG   = "#000000"