from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .constants import DIRECTION_OPTIONS, WEIGHT_TOLERANCE


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def extend(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def validate_input_data(criteria_df: pd.DataFrame, raw_df: pd.DataFrame, city_columns: list[str]) -> ValidationResult:
    result = ValidationResult()
    if criteria_df.empty:
        result.errors.append("Criteria Sheet has no minor criteria rows.")
        return result
    if raw_df.empty:
        result.errors.append("Data Sheet has no raw data rows.")
        return result
    if not city_columns:
        result.errors.append("Data Sheet must include at least one city column.")
        return result

    if criteria_df["criterion_id"].duplicated().any():
        duplicate_ids = criteria_df.loc[criteria_df["criterion_id"].duplicated(), "criterion_id"].tolist()
        result.errors.append(f"Duplicate criteria found in Criteria Sheet: {duplicate_ids[:5]}")

    if raw_df["criterion_id"].duplicated().any():
        duplicate_ids = raw_df.loc[raw_df["criterion_id"].duplicated(), "criterion_id"].tolist()
        result.errors.append(f"Duplicate criteria found in Data Sheet: {duplicate_ids[:5]}")

    criteria_ids = set(criteria_df["criterion_id"])
    data_ids = set(raw_df["criterion_id"])

    missing_in_data = sorted(criteria_ids - data_ids)
    if missing_in_data:
        sample = criteria_df[criteria_df["criterion_id"] == missing_in_data[0]]
        if not sample.empty:
            sample_label = f"{sample.iloc[0]['macro']} > {sample.iloc[0]['major']} > {sample.iloc[0]['micro']}"
        else:
            sample_label = missing_in_data[0]
        result.errors.append(
            f"{len(missing_in_data)} criteria from Criteria Sheet are missing in Data Sheet "
            f"(e.g., {sample_label})."
        )

    extra_in_data = sorted(data_ids - criteria_ids)
    if extra_in_data:
        sample = raw_df[raw_df["criterion_id"] == extra_in_data[0]]
        if not sample.empty:
            sample_label = f"{sample.iloc[0]['macro']} > {sample.iloc[0]['major']} > {sample.iloc[0]['micro']}"
        else:
            sample_label = extra_in_data[0]
        result.warnings.append(
            f"{len(extra_in_data)} criteria appear in Data Sheet but not Criteria Sheet "
            f"(e.g., {sample_label}). They will be ignored."
        )

    numeric_block = raw_df[city_columns]
    missing_cells = int(numeric_block.isna().sum().sum())
    if missing_cells > 0:
        result.warnings.append(
            f"Found {missing_cells} missing or non-numeric city values; scoring will skip missing cells."
        )

    fully_missing_rows = raw_df[numeric_block.isna().all(axis=1)]
    if not fully_missing_rows.empty:
        if len(fully_missing_rows) == len(raw_df):
            result.errors.append("All criteria rows are missing numeric values across all cities.")
        else:
            first = fully_missing_rows.iloc[0]
            example = f"{first['macro']} > {first['major']} > {first['micro']}"
            result.warnings.append(
                f"{len(fully_missing_rows)} criteria rows have no numeric values across all cities "
                f"(e.g., {example}). They will be ignored in weighted averages."
            )

    return result


def validate_direction_map(direction_map: dict[str, str], valid_criterion_ids: set[str]) -> ValidationResult:
    result = ValidationResult()
    unknown = sorted(set(direction_map) - valid_criterion_ids)
    if unknown:
        result.warnings.append(
            f"{len(unknown)} direction overrides do not map to active criteria and were ignored."
        )

    invalid_values = sorted({value for value in direction_map.values() if value not in DIRECTION_OPTIONS})
    if invalid_values:
        result.errors.append(f"Invalid direction values found: {invalid_values}")

    return result


def validate_weight_sums(
    macro_weights: pd.DataFrame,
    major_weights: pd.DataFrame,
    minor_weights: pd.DataFrame,
    tolerance: float = WEIGHT_TOLERANCE,
) -> ValidationResult:
    result = ValidationResult()

    macro_sum = float(macro_weights["weight"].sum()) if not macro_weights.empty else 0.0
    if abs(macro_sum - 1.0) > tolerance:
        result.errors.append(f"Macro weights sum to {macro_sum:.4f}; expected 1.0000.")

    major_sums = major_weights.groupby("macro", dropna=False)["weight"].sum()
    for macro, total in major_sums.items():
        if abs(float(total) - 1.0) > tolerance:
            result.errors.append(f"Major weights for macro '{macro}' sum to {float(total):.4f}; expected 1.0000.")

    minor_sums = minor_weights.groupby(["macro", "major"], dropna=False)["weight"].sum()
    for (macro, major), total in minor_sums.items():
        if abs(float(total) - 1.0) > tolerance:
            result.errors.append(
                f"Minor weights for '{macro} > {major}' sum to {float(total):.4f}; expected 1.0000."
            )

    if (macro_weights["weight"] < 0).any() or (major_weights["weight"] < 0).any() or (minor_weights["weight"] < 0).any():
        result.errors.append("Weights must be non-negative.")

    return result
