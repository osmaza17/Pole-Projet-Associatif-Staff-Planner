SCALE       = 0.75
STATS_SCALE = 1.25
GRID_SCALE  = 0.75


def _s(v: float) -> int:
    return max(1, round(v * SCALE))


def _s_stats(v: float) -> int:
    return max(1, round(v * SCALE * STATS_SCALE))


def _s_grid(v: float) -> int:
    return max(1, round(v * SCALE * GRID_SCALE))


# ── Solver parameter definitions ─────────────────────────────────────
# Each param: key, default value, hint (shown under the TextField).
# DEFAULT_SOLVER_PARAMS is derived automatically at the bottom.
#
# Four groups:
#   Termination    — when to stop
#   Algorithm      — method + parallelism
#   Presolve & MIP — preprocessing and tree search
#   Numerics       — scaling (critical for large / badly-scaled models)

SOLVER_PARAM_DEFS = [
    {
        "category": "Termination",
        "color": "#C62828",
        "params": [
            {
                "key":     "TimeLimit",
                "default": 1200,
                "hint":    "Max solve time (s). E.g. 600, 3600",
            },
            {
                "key":     "MIPGap",
                "default": 0.001,
                "hint":    "Relative gap to stop. 0.01 = 1%   0.001 = 0.1%",
            },
        ],
    },
    {
        "category": "Algorithm",
        "color": "#1565C0",
        "params": [
            {
                "key":     "Threads",
                "default": 0,
                "hint":    "CPU threads.  0 = all available cores",
            },
            {
                "key":     "Method",
                "default": -1,
                "hint":    "-1 auto   0 primal   1 dual   2 barrier   3 concurrent",
            },
            {
                "key":     "MIPFocus",
                "default": 1,
                "hint":    "0 balanced   1 find solutions   2 optimality   3 best bound",
            },
            {
                "key":     "Heuristics",
                "default": 0.05,
                "hint":    "Heuristics time share.  Range: 0.0 – 1.0",
            },
            {
                "key":     "ConcurrentMIP",
                "default": 1,
                "hint":    "Parallel MIP strategies.  1 = off   2+ = concurrent",
            },
        ],
    },
    {
        "category": "Presolve & MIP",
        "color": "#2E7D32",
        "params": [
            {
                "key":     "Presolve",
                "default": 2,
                "hint":    "-1 auto   0 off   1 moderate   2 aggressive",
            },
            {
                "key":     "Symmetry",
                "default": -1,
                "hint":    "-1 auto   0 off   1 conservative   2 aggressive",
            },
            {
                "key":     "Disconnected",
                "default": -1,
                "hint":    "-1 auto   0 off   1 moderate   2 aggressive",
            },
            {
                "key":     "IntegralityFocus",
                "default": 1,
                "hint":    "Tighten integer checks.  0 = off   1 = on",
            },
            {
                "key":     "Cuts",
                "default": -1,
                "hint":    "-1 auto   0 off   1 moderate   2 aggressive   3 very agg.",
            },
        ],
    },
    {
        "category": "Numerics & Scaling",
        "color": "#00695C",
        "params": [
            {
                "key":     "ScaleFlag",
                "default": -1,
                "hint":    "-1 auto   0 off   1 geometric   2 bilinear ✓   3 both",
            },
            {
                "key":     "ObjScale",
                "default": 0,
                "hint":    "0 auto   -1 geometric mean   >0 manual factor",
            },
            {
                "key":     "NumericFocus",
                "default": 0,
                "hint":    "0 auto   1 careful   2 more   3 most careful (slower)",
            },
        ],
    },
]

# Derived automatically — single source of truth
DEFAULT_SOLVER_PARAMS: dict = {
    p["key"]: p["default"]
    for cat in SOLVER_PARAM_DEFS
    for p in cat["params"]
}

# ── Objective weights ────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "W_COVERAGE":     1000000,
    "W_RULE":          100000,
    "W_DURATION":       50000,
    "W_STABILITY":      10000,
    "W_INTERGROUP":     10000,
    "W_INTRAGROUP":      1000,
    "W_EMERG":           1000,
    "W_TRAVEL":           100,
    "W_GAP":               10,
    "W_SOCIAL":            10,
    "W_PREF":              10,
    "W_VARIETY":            1,
    "W_STICKY":             1,
    "W_ROTATION":           1,
}
SORTED_VALUES = sorted(DEFAULT_WEIGHTS.values(), reverse=True)

DEFAULT_HOURS_TEXT = (
    "08:00\n09:00\n10:00\n11:00\n12:00\n"
    "13:00\n14:00\n15:00\n16:00\n17:00"
)

# ── App configuration ────────────────────────────────────────────────
PROFILE_WATCHER_INTERVAL = 2.5
DEFAULT_PROFILE_NAME     = "scheduler_profile"

# ── Semantic color palette ───────────────────────────────────────────
PRIMARY_BLUE     = "#1565C0"
DANGER_RED       = "#C62828"
SUCCESS_GREEN    = "#2E7D32"
PURPLE           = "#6A1B9A"
INFO_BLUE        = "#90CAF9"
DANGER_LIGHT     = "#EF9A9A"

SIDEBAR_WIDTH         = _s(200)
SIDEBAR_BG            = "#263208"
SIDEBAR_SELECTED_BG   = "#37474F"
SIDEBAR_TEXT_COLOR    = "#ECEFF1"
SIDEBAR_SELECTED_TEXT = "#4FC3F7"
DIVIDER_COLOR         = "#455A64"
PROFILE_CARD_BG       = "#2E3D49"
PROFILE_EMPTY_FG      = "#607D8B"

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