from __future__ import annotations

CRITERIA_SHEET_NAME = "Criteria Sheet"
DATA_SHEET_NAME = "Data Sheet"

CRITERIA_COLUMN_NAMES = [
    "macro",
    "macro_weight_template",
    "major",
    "major_weight_template",
    "micro",
    "minor_weight_template",
]

DATA_BASE_COLUMNS = ["macro", "major", "micro", "source"]

SCORING_METHOD_RANK = "rank"
SCORING_METHOD_PERCENTILE = "percentile"
SCORING_METHOD_PERCENTILE_RANK = SCORING_METHOD_PERCENTILE
SCORING_METHOD_MINMAX = "minmax"
SCORING_METHOD_ROBUST_MINMAX = "robust_minmax"
SCORING_METHOD_LOG_ROBUST_MINMAX = "log_robust_minmax"

SCORING_METHOD_LABELS = {
    "Rank": SCORING_METHOD_RANK,
    "Percentile": SCORING_METHOD_PERCENTILE,
}

SCORING_METHOD_CLIENT_LABELS = ["Rank", "Percentile"]
SCORING_METHOD_ADVANCED_LABELS = ["Rank", "Percentile"]

MODE_CLIENT = "Client"
MODE_ADVANCED = "Advanced"
MODE_OPTIONS = [MODE_CLIENT, MODE_ADVANCED]

DIRECTION_HIGHER = "higher"
DIRECTION_LOWER = "lower"
DIRECTION_OPTIONS = [DIRECTION_HIGHER, DIRECTION_LOWER]

DEFAULT_CAPABILITY_MACROS = ["Talent", "Operating Environment", "Risk"]
DEFAULT_COST_MACRO = "Cost"

WEIGHT_TOLERANCE = 1e-6
DEFAULT_DECIMALS = 3

WEIGHTING_MODE_HELP = (
    "Simple: edit macro weights only; major/minor weights are equally split.\n"
    "Advanced: edit macro, major, and minor weights directly."
)

SCORING_METHOD_HELP = (
    "Rank: Python-computed competition ranks from raw metric values; ties share the same rank.\n"
    "Percentile: Python-computed ECDF percentile from direction-adjusted raw metric values, presented on a 0-100 indexed basis."
)

MACRO_PRESET_TARGETS = {
    "Cost-led": {
        "Cost": 0.45,
        "Talent": 0.2,
        "Operating Environment": 0.2,
        "Risk": 0.15,
    },
    "Talent-led": {
        "Talent": 0.45,
        "Operating Environment": 0.2,
        "Risk": 0.2,
        "Cost": 0.15,
    },
    "Risk-averse": {
        "Risk": 0.45,
        "Operating Environment": 0.2,
        "Talent": 0.2,
        "Cost": 0.15,
    },
    "Growth-led": {
        "Operating Environment": 0.4,
        "Talent": 0.3,
        "Risk": 0.15,
        "Cost": 0.15,
    },
}

# Preferred categorical colors from provided palette:
# yellow (bottom-left), blue (mid-left), teal (mid-right), red (mid-bottom).
SAVILLS_COLOR_SEQUENCE = [
    "#F2D500",  # yellow
    "#6D769C",  # blue
    "#4A9A8D",  # teal
    "#FF5B4F",  # red
    "#262A43",  # dark navy (fallback)
    "#6B99A2",  # muted teal (fallback)
    "#757D84",  # cool gray (fallback)
    "#6C8E70",  # sage (fallback)
    "#6E345C",  # plum (fallback)
    "#E55A00",  # orange (fallback)
    "#E7C66B",  # sand (fallback)
    "#D5D1CC",  # warm light gray (fallback)
    "#A94E8B",  # magenta (fallback)
    "#005B5F",  # deep teal (fallback)
    "#000000",  # black (fallback)
]

SAVILLS_MACRO_COLOR_MAP = {
    "Talent": "#6D769C",                 # blue
    "Operating Environment": "#4A9A8D",  # teal
    "Risk": "#FF5B4F",                   # red
    "Cost": "#F2D500",                   # yellow
}

SAVILLS_MARKET_TIER_COLOR_MAP = {
    "Primary": "#6D769C",    # blue
    "Secondary": "#4A9A8D",  # teal
    "Tertiary": "#F2D500",   # yellow
}
