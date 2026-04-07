from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
DEFAULT_WORKBOOK_PATH = ASSETS_DIR / "emea_space_occupancy_demo_dataset.xlsx"

MODULE_ID = "emea_space_occupancy"
MODULE_NAME = "EMEA Space & Occupancy Planning Studio"
MODULE_DESCRIPTION = (
    "Dynamic space and occupancy planning for executive scenario modelling, "
    "validation, and decision support across EMEA portfolios."
)
MODULE_SECTION = "Workplace Planning"


@dataclass(frozen=True)
class SheetSpec:
    name: str
    required_columns: list[str] = field(default_factory=list)
    key_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    share_columns: list[str] = field(default_factory=list)
    share_upper_bounds: dict[str, float] = field(default_factory=dict)
    non_negative_columns: list[str] = field(default_factory=list)
    blocking: bool = True
    header_tokens: list[str] = field(default_factory=list)


NAV_ITEMS: list[tuple[str, str]] = [
    ("Home", "home"),
    ("Data Upload & Validation", "data_upload_validation"),
    ("Portfolio Baseline", "portfolio_baseline"),
    ("Occupancy & Utilisation", "occupancy_utilisation"),
    ("Assumptions Manager", "assumptions_manager"),
    ("Scenario Builder", "scenario_builder"),
    ("Scenario Comparison", "scenario_comparison"),
    ("Space Planning Outputs", "space_planning_outputs"),
    ("Decision Pack", "decision_pack"),
    ("Exports & Audit", "exports_audit"),
]


SHEET_SPECS: dict[str, SheetSpec] = {
    "README": SheetSpec(
        name="README",
        blocking=False,
        required_columns=[
            "Metric",
            "Value",
            "Sheet",
            "Purpose",
            "Role in Demo",
            "Key Grain",
            "Primary Key",
            "Input / Output",
            "Notes",
        ],
        header_tokens=["Metric", "Value", "Sheet"],
    ),
    "portfolio_hierarchy": SheetSpec(
        name="portfolio_hierarchy",
        required_columns=[
            "region",
            "country",
            "city",
            "site_id",
            "site_name",
            "site_type",
            "building_id",
            "building_name",
            "floor_id",
            "floor_name",
            "floor_sequence",
            "gross_area_sqm",
            "usable_area_sqm",
            "seat_capacity",
            "criticality",
            "delivery_status",
        ],
        key_columns=["floor_id"],
        numeric_columns=["floor_sequence", "gross_area_sqm", "usable_area_sqm", "seat_capacity"],
        non_negative_columns=["floor_sequence", "gross_area_sqm", "usable_area_sqm", "seat_capacity"],
        header_tokens=["region", "country", "city", "site_id", "building_id", "floor_id"],
    ),
    "property_metrics": SheetSpec(
        name="property_metrics",
        required_columns=[
            "site_id",
            "region",
            "country",
            "city",
            "site_name",
            "site_type",
            "building_id",
            "building_name",
            "gross_area_sqm",
            "usable_area_sqm",
            "seat_capacity_total",
            "annual_property_cost_eur",
            "annual_cost_per_sqm_eur",
            "lease_start_date",
            "lease_end_date",
            "occupancy_sensors_flag",
            "desk_booking_system_flag",
            "criticality",
            "energy_rating",
        ],
        key_columns=["building_id"],
        numeric_columns=[
            "gross_area_sqm",
            "usable_area_sqm",
            "seat_capacity_total",
            "annual_property_cost_eur",
            "annual_cost_per_sqm_eur",
        ],
        date_columns=["lease_start_date", "lease_end_date"],
        non_negative_columns=[
            "gross_area_sqm",
            "usable_area_sqm",
            "seat_capacity_total",
            "annual_property_cost_eur",
            "annual_cost_per_sqm_eur",
        ],
        header_tokens=["site_id", "building_id", "seat_capacity_total", "lease_end_date"],
    ),
    "space_inventory": SheetSpec(
        name="space_inventory",
        required_columns=[
            "site_id",
            "region",
            "country",
            "city",
            "site_name",
            "building_id",
            "building_name",
            "floor_id",
            "floor_name",
            "space_type",
            "space_subtype",
            "room_count",
            "area_sqm",
            "seat_count",
            "capacity",
        ],
        key_columns=["floor_id", "space_type", "space_subtype"],
        numeric_columns=["room_count", "area_sqm", "seat_count", "capacity"],
        non_negative_columns=["room_count", "area_sqm", "seat_count", "capacity"],
        header_tokens=["site_id", "floor_id", "space_type", "space_subtype"],
    ),
    "people_demand": SheetSpec(
        name="people_demand",
        required_columns=[
            "site_id",
            "region",
            "country",
            "city",
            "site_name",
            "site_type",
            "business_unit",
            "team_cluster",
            "workstyle_category",
            "current_headcount",
            "forecast_headcount_12m",
            "forecast_headcount_24m",
            "avg_attendance_pct",
            "peak_attendance_pct",
            "remote_ratio_pct",
            "strategic_priority",
        ],
        key_columns=["site_id", "business_unit"],
        numeric_columns=[
            "current_headcount",
            "forecast_headcount_12m",
            "forecast_headcount_24m",
            "avg_attendance_pct",
            "peak_attendance_pct",
            "remote_ratio_pct",
        ],
        share_columns=["avg_attendance_pct", "peak_attendance_pct", "remote_ratio_pct"],
        share_upper_bounds={
            "avg_attendance_pct": 1.0,
            "peak_attendance_pct": 1.05,
            "remote_ratio_pct": 1.0,
        },
        non_negative_columns=["current_headcount", "forecast_headcount_12m", "forecast_headcount_24m"],
        header_tokens=["site_id", "business_unit", "current_headcount", "avg_attendance_pct"],
    ),
    "occupancy_utilisation": SheetSpec(
        name="occupancy_utilisation",
        required_columns=[
            "month",
            "site_id",
            "region",
            "country",
            "city",
            "site_name",
            "site_type",
            "current_headcount",
            "seat_capacity",
            "avg_daily_attendance",
            "peak_daily_attendance",
            "avg_desk_utilisation_pct",
            "peak_desk_utilisation_pct",
            "avg_meeting_room_utilisation_pct",
            "collaboration_space_utilisation_pct",
            "badge_swipes",
            "desk_bookings",
            "meeting_room_bookings",
        ],
        key_columns=["site_id", "month"],
        numeric_columns=[
            "current_headcount",
            "seat_capacity",
            "avg_daily_attendance",
            "peak_daily_attendance",
            "avg_desk_utilisation_pct",
            "peak_desk_utilisation_pct",
            "avg_meeting_room_utilisation_pct",
            "collaboration_space_utilisation_pct",
            "badge_swipes",
            "desk_bookings",
            "meeting_room_bookings",
        ],
        date_columns=["month"],
        share_columns=[
            "avg_desk_utilisation_pct",
            "peak_desk_utilisation_pct",
            "avg_meeting_room_utilisation_pct",
            "collaboration_space_utilisation_pct",
        ],
        share_upper_bounds={
            "avg_desk_utilisation_pct": 1.1,
            "peak_desk_utilisation_pct": 1.5,
            "avg_meeting_room_utilisation_pct": 1.25,
            "collaboration_space_utilisation_pct": 1.25,
        },
        non_negative_columns=[
            "current_headcount",
            "seat_capacity",
            "avg_daily_attendance",
            "peak_daily_attendance",
            "badge_swipes",
            "desk_bookings",
            "meeting_room_bookings",
        ],
        header_tokens=["month", "site_id", "avg_daily_attendance", "avg_desk_utilisation_pct"],
    ),
    "standards": SheetSpec(
        name="standards",
        required_columns=[
            "standard_id",
            "standard_group",
            "applicable_to",
            "seat_ratio_target",
            "desk_sharing_ratio_target",
            "sqm_per_person_target",
            "collaboration_area_pct_target",
            "meeting_seats_per_100_staff",
            "focus_seats_per_100_staff",
            "planning_utilisation_threshold_pct",
            "policy_version_or_rule",
            "effective_date",
        ],
        key_columns=["standard_id"],
        numeric_columns=[
            "seat_ratio_target",
            "desk_sharing_ratio_target",
            "sqm_per_person_target",
            "collaboration_area_pct_target",
            "meeting_seats_per_100_staff",
            "focus_seats_per_100_staff",
            "planning_utilisation_threshold_pct",
        ],
        date_columns=["effective_date"],
        share_columns=[
            "seat_ratio_target",
            "collaboration_area_pct_target",
            "planning_utilisation_threshold_pct",
        ],
        share_upper_bounds={
            "seat_ratio_target": 1.2,
            "collaboration_area_pct_target": 1.0,
            "planning_utilisation_threshold_pct": 1.1,
        },
        non_negative_columns=[
            "seat_ratio_target",
            "desk_sharing_ratio_target",
            "sqm_per_person_target",
            "collaboration_area_pct_target",
            "meeting_seats_per_100_staff",
            "focus_seats_per_100_staff",
            "planning_utilisation_threshold_pct",
        ],
        header_tokens=["standard_id", "standard_group", "desk_sharing_ratio_target", "effective_date"],
    ),
    "scenario_assumptions": SheetSpec(
        name="scenario_assumptions",
        required_columns=[
            "scenario_id",
            "scenario_name",
            "planning_horizon_months",
            "parameter_category",
            "parameter_name",
            "scope_level",
            "scope_value",
            "value",
            "unit",
            "driver_note",
            "owner",
            "version_status",
        ],
        key_columns=["scenario_id", "parameter_name", "scope_level", "scope_value"],
        numeric_columns=["planning_horizon_months", "value"],
        non_negative_columns=["planning_horizon_months"],
        header_tokens=["scenario_id", "scenario_name", "parameter_name", "scope_level", "value"],
    ),
    "scenario_outputs": SheetSpec(
        name="scenario_outputs",
        blocking=False,
        required_columns=[
            "scenario_id",
            "scenario_name",
            "site_id",
            "region",
            "country",
            "city",
            "site_name",
            "site_type",
            "forecast_headcount",
            "peak_attendance",
            "existing_seats",
            "required_seats",
            "seat_gap",
            "existing_usable_area_sqm",
            "required_area_sqm",
            "area_gap_sqm",
            "action_flag",
            "risk_rating",
            "estimated_move_complexity",
            "standards_compliance_score",
            "version_status",
        ],
        key_columns=["scenario_id", "site_id"],
        numeric_columns=[
            "forecast_headcount",
            "peak_attendance",
            "existing_seats",
            "required_seats",
            "seat_gap",
            "existing_usable_area_sqm",
            "required_area_sqm",
            "area_gap_sqm",
            "standards_compliance_score",
        ],
        non_negative_columns=[
            "forecast_headcount",
            "peak_attendance",
            "existing_seats",
            "required_seats",
            "existing_usable_area_sqm",
            "required_area_sqm",
            "standards_compliance_score",
        ],
        header_tokens=["scenario_id", "scenario_name", "site_id", "required_seats"],
    ),
    "data_dictionary": SheetSpec(
        name="data_dictionary",
        blocking=False,
        required_columns=["sheet_name", "column_name", "definition", "notes"],
        header_tokens=["sheet_name", "column_name", "definition"],
    ),
}


EXPECTED_SHEETS = list(SHEET_SPECS.keys())
CORE_MODEL_SHEETS = [name for name, spec in SHEET_SPECS.items() if spec.blocking]
OPTIONAL_SUPPORTED_SHEETS = [name for name, spec in SHEET_SPECS.items() if not spec.blocking]

SCENARIO_SCORE_WEIGHTS = {
    "capacity_fit": 30.0,
    "utilisation_fit": 20.0,
    "standards_alignment": 20.0,
    "implementation_simplicity": 15.0,
    "consolidation_efficiency": 15.0,
}

# These threshold bands intentionally stay explicit and modest so the model remains
# explainable in a live client session rather than behaving like a black box.
ACTION_THRESHOLD_DEFAULTS = {
    "expand_seat_deficit_ratio": -0.07,
    "expand_area_deficit_ratio": -0.08,
    "expand_peak_utilisation_buffer": 0.06,
    "release_seat_surplus_ratio": 0.18,
    "release_area_surplus_ratio": 0.12,
    "release_utilisation_buffer": 0.12,
    "maintain_seat_tolerance_ratio": 0.05,
    "maintain_area_tolerance_ratio": 0.06,
    "maintain_score_floor": 72.0,
    "exit_transfer_ratio": 0.75,
}

RISK_THRESHOLD_DEFAULTS = {
    "high_peak_utilisation_buffer": 0.08,
    "medium_peak_utilisation_buffer": 0.03,
    "high_capacity_gap_ratio": -0.12,
    "medium_capacity_gap_ratio": -0.05,
    "high_area_gap_ratio": -0.10,
    "medium_area_gap_ratio": -0.04,
    "high_score_floor": 60.0,
    "medium_score_floor": 75.0,
}

MOVE_COMPLEXITY_SCORE_MAP = {
    "Low": 88.0,
    "Medium": 64.0,
    "High": 38.0,
}

SPACE_PLANNING_DEFAULTS = {
    "post_24m_growth_damping": 0.5,
    "hq_growth_hub_bonus": 0.020,
    "hub_growth_hub_bonus": 0.030,
    "office_growth_hub_bonus": 0.015,
    "non_hub_consolidation_drag": -0.010,
    "minimum_peak_gap_hq": 0.09,
    "minimum_peak_gap_hub": 0.08,
    "minimum_peak_gap_office": 0.06,
    "collaboration_area_uplift_weight": 0.35,
    "meeting_area_uplift_weight": 0.10,
    "focus_area_uplift_weight": 0.08,
}

ACTION_ORDER = [
    "Expand",
    "Re-stack / Rebalance",
    "Maintain",
    "Consolidate / Release Space",
    "Exit / Merge",
]

RISK_ORDER = ["High", "Medium", "Low"]
MOVE_COMPLEXITY_ORDER = ["High", "Medium", "Low"]

ATTENDANCE_MIN_RATE = 0.2
ATTENDANCE_MAX_RATE = 1.15
DEFAULT_ROOM_SEATS = 6.0
NULL_HEAVY_THRESHOLD = 0.2

PREFERRED_SCENARIO_AUTO_KEY = "__auto__"
