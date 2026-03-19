from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


NOMIS_DATASET_BASE_URL = "https://www.nomisweb.co.uk/api/v01/dataset"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_RETRIES = 3
LAD_GEOGRAPHY_TYPE = "TYPE434"
VALUE_MEASURE = "20100"
PERCENTAGE_VALUE_MEASURE = "20599"
PERCENTAGE_NUMERATOR_MEASURE = "21001"
PERCENTAGE_DENOMINATOR_MEASURE = "21002"
FULL_RECORD_LIMIT = "2000000"
PAGE_RECORD_LIMIT = 25000

POPULATION_DATASET_ID = "NM_31_1"
APS_COUNTS_DATASET_ID = "NM_17_1"
APS_PERCENTAGES_DATASET_ID = "NM_17_5"

APS_PERCENTAGE_VARIABLE_LABELS = {
    "employment_rate": "Employment rate - aged 16-64",
    "unemployment_rate": "Unemployment rate - aged 16-64",
    "economic_activity_rate": "Economic activity rate - aged 16-64",
    "nvq4_plus_share": "% with RQF4+ - aged 16-64",
    "no_qualifications_share": "% with no qualifications (RQF) - aged 16-64",
    "professional_occupations_share": "% all in employment who are - 2: professional occupations (SOC2020)",
}


@dataclass(frozen=True)
class NomisDatasetMetadata:
    dataset_id: str
    dataset_name: str
    mnemonic: str
    last_updated: str
    def_path: Path
    overview_path: Path


@dataclass(frozen=True)
class ExtractedNomisArtifacts:
    population_metadata: NomisDatasetMetadata
    population_age_codelist_path: Path
    population_sex_codelist_path: Path
    population_data_path: Path
    aps_counts_metadata: NomisDatasetMetadata
    aps_counts_cell_codelist_path: Path
    aps_counts_data_path: Path
    aps_counts_cell_map: dict[str, int]
    aps_percentages_metadata: NomisDatasetMetadata
    aps_percentages_variable_codelist_path: Path
    aps_percentages_data_path: Path
    aps_percentages_variable_map: dict[str, int]


def _build_session(retries: int = DEFAULT_RETRIES) -> requests.Session:
    retry = Retry(
        total=retries,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Savills-DX-Talent-Analytics/1.0 (+ons-pipeline)",
            "Accept": "*/*",
        }
    )
    return session


def _build_dataset_url(dataset_id: str, suffix: str, params: dict[str, str] | None = None) -> str:
    url = f"{NOMIS_DATASET_BASE_URL}/{dataset_id}{suffix}"
    if not params:
        return url
    return f"{url}?{urlencode(params)}"


def _ensure_non_html_payload(response: requests.Response, expected: str) -> None:
    preview = response.text[:200].lstrip().lower()
    if "<!doctype html" in preview or preview.startswith("<html"):
        raise RuntimeError(
            f"Nomis returned HTML instead of {expected} for {response.url}. "
            "This usually means the endpoint or query parameters are invalid."
        )


def _download_json(session: requests.Session, url: str, destination: Path) -> dict:
    response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    _ensure_non_html_payload(response, "JSON")
    payload = response.json()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _download_csv(session: requests.Session, url: str, destination: Path) -> None:
    response = session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    _ensure_non_html_payload(response, "CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response.text, encoding="utf-8")


def _download_csv_paginated(
    session: requests.Session,
    *,
    dataset_id: str,
    params: dict[str, str],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    first_page = True
    with destination.open("w", encoding="utf-8", newline="") as handle:
        while True:
            page_url = _build_dataset_url(
                dataset_id,
                ".data.csv",
                params={
                    **params,
                    "recordlimit": str(PAGE_RECORD_LIMIT),
                    "recordoffset": str(offset),
                },
            )
            response = session.get(page_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            response.raise_for_status()
            _ensure_non_html_payload(response, "CSV")

            lines = response.text.splitlines()
            if not lines:
                break
            header, *data_lines = lines
            if first_page:
                handle.write(header + "\n")
                first_page = False
            if data_lines:
                handle.write("\n".join(data_lines) + "\n")

            if len(data_lines) < PAGE_RECORD_LIMIT:
                break
            offset += PAGE_RECORD_LIMIT


def _annotation_map(keyfamily: dict) -> dict[str, str]:
    annotations: dict[str, str] = {}
    for annotation in keyfamily.get("annotations", {}).get("annotation", []):
        title = str(annotation.get("annotationtitle", "")).strip()
        value = str(annotation.get("annotationtext", "")).strip()
        if title:
            annotations[title] = value
    return annotations


def _load_codelist_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["structure"]["codelists"]["codelist"][0]["code"]


def _entry_description(entry: dict) -> str:
    return str(entry.get("description", {}).get("value", "")).strip()


def _find_single_code(entries: list[dict], *, contains: tuple[str, ...], startswith: str | None = None) -> int:
    matches: list[int] = []
    for entry in entries:
        description = _entry_description(entry)
        lowered = description.lower()
        if startswith and not lowered.startswith(startswith.lower()):
            continue
        if all(term.lower() in lowered for term in contains):
            matches.append(int(entry["value"]))
    if len(matches) != 1:
        joined = ", ".join(str(match) for match in matches[:5]) or "none"
        terms = ", ".join(contains)
        raise ValueError(f"Expected exactly one codelist match for '{terms}', found {len(matches)} ({joined}).")
    return matches[0]


def _extract_dataset_metadata(
    session: requests.Session,
    *,
    dataset_id: str,
    raw_dir: Path,
) -> NomisDatasetMetadata:
    def_path = raw_dir / f"nomis_{dataset_id.lower()}_def.json"
    overview_path = raw_dir / f"nomis_{dataset_id.lower()}_overview.json"

    definition = _download_json(session, _build_dataset_url(dataset_id, "/def.sdmx.json"), def_path)
    _download_json(session, _build_dataset_url(dataset_id.lower(), ".overview.json"), overview_path)

    keyfamily = definition["structure"]["keyfamilies"]["keyfamily"][0]
    annotations = _annotation_map(keyfamily)
    return NomisDatasetMetadata(
        dataset_id=dataset_id,
        dataset_name=str(keyfamily["name"]["value"]),
        mnemonic=annotations.get("Mnemonic", dataset_id.lower()),
        last_updated=annotations.get("LastUpdated", ""),
        def_path=def_path,
        overview_path=overview_path,
    )


def _extract_population_raw(
    session: requests.Session,
    *,
    raw_dir: Path,
) -> tuple[NomisDatasetMetadata, Path, Path, Path]:
    metadata = _extract_dataset_metadata(session, dataset_id=POPULATION_DATASET_ID, raw_dir=raw_dir)
    age_path = raw_dir / "nomis_nm_31_1_age_codelist.json"
    sex_path = raw_dir / "nomis_nm_31_1_sex_codelist.json"

    _download_json(session, _build_dataset_url(POPULATION_DATASET_ID, "/age.def.sdmx.json"), age_path)
    _download_json(session, _build_dataset_url(POPULATION_DATASET_ID, "/sex.def.sdmx.json"), sex_path)

    age_entries = _load_codelist_entries(age_path)
    sex_entries = _load_codelist_entries(sex_path)
    all_ages_code = _find_single_code(age_entries, contains=("all ages",))
    total_sex_code = _find_single_code(sex_entries, contains=("total",))

    population_path = raw_dir / "nomis_nm_31_1_population_type434_latest.csv"
    population_url = _build_dataset_url(
        POPULATION_DATASET_ID,
        ".data.csv",
        params={
            "geography": LAD_GEOGRAPHY_TYPE,
            "sex": str(total_sex_code),
            "age": str(all_ages_code),
            "measures": VALUE_MEASURE,
            "time": "latest",
            "select": "date_name,date_code,geography_name,geography_code,sex_name,age_name,obs_value,obs_status",
            "recordlimit": FULL_RECORD_LIMIT,
        },
    )
    _download_csv(session, population_url, population_path)
    return metadata, age_path, sex_path, population_path


def _select_aps_count_cells(cell_entries: list[dict]) -> dict[str, int]:
    description_to_value = {_entry_description(entry): int(entry["value"]) for entry in cell_entries}

    return {
        "working_age_population": description_to_value["T01:22 (Aged 16-64 - All : All People )"],
        "economic_activity_count": description_to_value["T01:25 (Aged 16-64 - Economically active : All People )"],
        "employment_count": description_to_value["T01:28 (Aged 16-64 - In employment : All People )"],
        "unemployment_count": description_to_value["T01:37 (Aged 16-64 - Unemployed : All People )"],
    }

def _select_aps_percentage_variables(variable_entries: list[dict]) -> dict[str, int]:
    description_to_value = {_entry_description(entry): int(entry["value"]) for entry in variable_entries}
    missing = sorted(
        metric_key
        for metric_key, description in APS_PERCENTAGE_VARIABLE_LABELS.items()
        if description not in description_to_value
    )
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Could not resolve the full set of APS percentage variables: {joined}.")
    return {
        metric_key: description_to_value[description]
        for metric_key, description in APS_PERCENTAGE_VARIABLE_LABELS.items()
    }


def _extract_aps_counts_raw(
    session: requests.Session,
    *,
    raw_dir: Path,
) -> tuple[NomisDatasetMetadata, Path, Path, dict[str, int]]:
    metadata = _extract_dataset_metadata(session, dataset_id=APS_COUNTS_DATASET_ID, raw_dir=raw_dir)
    cell_path = raw_dir / "nomis_nm_17_1_cell_codelist_type434_latest.json"
    _download_json(
        session,
        _build_dataset_url(
            APS_COUNTS_DATASET_ID,
            "/cell.def.sdmx.json",
            params={"geography": LAD_GEOGRAPHY_TYPE, "time": "latest"},
        ),
        cell_path,
    )

    cell_entries = _load_codelist_entries(cell_path)
    aps_cell_map = _select_aps_count_cells(cell_entries)

    aps_data_path = raw_dir / "nomis_nm_17_1_t01_selected_type434_latest.csv"
    selected_cells = ",".join(str(value) for value in sorted(set(aps_cell_map.values())))

    base_params = {
        "geography": LAD_GEOGRAPHY_TYPE,
        "time": "latest",
        "cell": selected_cells,
        "measures": VALUE_MEASURE,
        "select": "date_name,date_code,geography_name,geography_code,cell,cell_name,obs_value,obs_status",
        "recordlimit": FULL_RECORD_LIMIT,
    }

    _download_csv_paginated(
        session,
        dataset_id=APS_COUNTS_DATASET_ID,
        params=base_params,
        destination=aps_data_path,
    )

    return metadata, cell_path, aps_data_path, aps_cell_map


def _extract_aps_percentages_raw(
    session: requests.Session,
    *,
    raw_dir: Path,
) -> tuple[NomisDatasetMetadata, Path, Path, dict[str, int]]:
    metadata = _extract_dataset_metadata(session, dataset_id=APS_PERCENTAGES_DATASET_ID, raw_dir=raw_dir)
    variable_path = raw_dir / "nomis_nm_17_5_variable_codelist_type434_latest.json"
    _download_json(
        session,
        _build_dataset_url(
            APS_PERCENTAGES_DATASET_ID,
            "/variable.def.sdmx.json",
            params={"geography": LAD_GEOGRAPHY_TYPE, "time": "latest"},
        ),
        variable_path,
    )

    variable_entries = _load_codelist_entries(variable_path)
    variable_map = _select_aps_percentage_variables(variable_entries)
    selected_variables = ",".join(str(value) for value in sorted(set(variable_map.values())))

    aps_percentages_path = raw_dir / "nomis_nm_17_5_selected_type434_all_periods.csv"
    base_params = {
        "geography": LAD_GEOGRAPHY_TYPE,
        "variable": selected_variables,
        "measures": ",".join(
            [PERCENTAGE_VALUE_MEASURE, PERCENTAGE_NUMERATOR_MEASURE, PERCENTAGE_DENOMINATOR_MEASURE]
        ),
        "select": (
            "date_name,date_code,geography_name,geography_code,"
            "variable,variable_name,measures,measures_name,obs_value,obs_status"
        ),
        "recordlimit": FULL_RECORD_LIMIT,
    }
    _download_csv_paginated(
        session,
        dataset_id=APS_PERCENTAGES_DATASET_ID,
        params=base_params,
        destination=aps_percentages_path,
    )
    return metadata, variable_path, aps_percentages_path, variable_map


def extract_nomis_raw(raw_dir: Path) -> ExtractedNomisArtifacts:
    session = _build_session()
    raw_dir.mkdir(parents=True, exist_ok=True)

    population_metadata, age_path, sex_path, population_data_path = _extract_population_raw(session, raw_dir=raw_dir)
    (
        aps_counts_metadata,
        aps_counts_cell_codelist_path,
        aps_counts_data_path,
        aps_counts_cell_map,
    ) = _extract_aps_counts_raw(session, raw_dir=raw_dir)
    (
        aps_percentages_metadata,
        aps_percentages_variable_codelist_path,
        aps_percentages_data_path,
        aps_percentages_variable_map,
    ) = _extract_aps_percentages_raw(session, raw_dir=raw_dir)

    return ExtractedNomisArtifacts(
        population_metadata=population_metadata,
        population_age_codelist_path=age_path,
        population_sex_codelist_path=sex_path,
        population_data_path=population_data_path,
        aps_counts_metadata=aps_counts_metadata,
        aps_counts_cell_codelist_path=aps_counts_cell_codelist_path,
        aps_counts_data_path=aps_counts_data_path,
        aps_counts_cell_map=aps_counts_cell_map,
        aps_percentages_metadata=aps_percentages_metadata,
        aps_percentages_variable_codelist_path=aps_percentages_variable_codelist_path,
        aps_percentages_data_path=aps_percentages_data_path,
        aps_percentages_variable_map=aps_percentages_variable_map,
    )
