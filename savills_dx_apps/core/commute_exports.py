from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _fig_map_png_bytes(df_emp: pd.DataFrame, office_lat: float, office_lon: float) -> bytes:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.scatter(df_emp["lon"], df_emp["lat"], s=8, alpha=0.6)
    ax.scatter([office_lon], [office_lat], s=60, marker="x")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Employee locations (scatter)")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def _plotly_employee_scatter_map_png_bytes(
    df_emp: pd.DataFrame,
    office_lat: float,
    office_lon: float,
    office_address: str,
    title: str,
) -> bytes:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:  # pragma: no cover - fallback
        raise RuntimeError("Plotly not available") from exc

    d = df_emp.copy()
    d["lat"] = pd.to_numeric(d.get("lat"), errors="coerce")
    d["lon"] = pd.to_numeric(d.get("lon"), errors="coerce")
    d["travel_time_min"] = pd.to_numeric(d.get("travel_time_min"), errors="coerce")
    d = d.dropna(subset=["lat", "lon", "travel_time_min"]).copy()

    fig = px.scatter_mapbox(
        d,
        lat="lat",
        lon="lon",
        color="travel_time_min",
        hover_data=["employeeID", "travel_time_min"],
        zoom=9,
        height=700,
        title=title,
        color_continuous_scale="RdYlGn_r",
    )
    fig.data[0].marker.colorbar = dict(title="Time (mins)")
    fig.add_trace(
        go.Scattermapbox(
            lat=[office_lat],
            lon=[office_lon],
            mode="markers",
            marker=dict(size=24, color="darkred", opacity=0.85),
            name="Office",
            hovertext=["Office: {0}".format(office_address)],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    return pio.to_image(fig, format="png", width=1200, height=700, scale=2)


def _fig_bands_png_bytes(bands_df: pd.DataFrame, office_labels: Sequence[str]) -> bytes:
    import matplotlib.pyplot as plt

    dfp = bands_df.copy()
    for c in ["le30", "b30_45", "b45_60", "gt60"]:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0.0)

    y = np.arange(len(office_labels))

    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    left = np.zeros(len(office_labels), dtype=float)
    ax.barh(y, dfp["le30"].values, left=left, label="<=30")
    left += dfp["le30"].values
    ax.barh(y, dfp["b30_45"].values, left=left, label="30-45")
    left += dfp["b30_45"].values
    ax.barh(y, dfp["b45_60"].values, left=left, label="45-60")
    left += dfp["b45_60"].values
    ax.barh(y, dfp["gt60"].values, left=left, label=">60")

    ax.set_yticks(y)
    ax.set_yticklabels(office_labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Employees (%)")
    ax.set_title("Share of employees by commute time band")
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22), frameon=False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plotly_bands_png_bytes(bands_df: pd.DataFrame, office_labels: Sequence[str]) -> bytes:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:  # pragma: no cover - fallback
        raise RuntimeError("Plotly not available") from exc

    dfp = bands_df.copy()
    for c in ["le30", "b30_45", "b45_60", "gt60"]:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="<=30", y=office_labels, x=dfp["le30"], orientation="h", marker=dict(color="#1e8449")))
    fig.add_trace(go.Bar(name="30-45", y=office_labels, x=dfp["b30_45"], orientation="h", marker=dict(color="#f1c40f")))
    fig.add_trace(go.Bar(name="45-60", y=office_labels, x=dfp["b45_60"], orientation="h", marker=dict(color="#e67e22")))
    fig.add_trace(go.Bar(name=">60", y=office_labels, x=dfp["gt60"], orientation="h", marker=dict(color="#c0392b")))

    fig.update_layout(
        barmode="stack",
        title="Share of employees by commute time band",
        xaxis=dict(title="Employees (%)", range=[0, 100], ticksuffix="%", showgrid=True, zeroline=False),
        yaxis=dict(title="", automargin=True, ticklabeloverflow="allow"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=260, r=40, t=60, b=90),
        height=max(420, 45 * len(office_labels) + 170),
    )

    return pio.to_image(fig, format="png", width=1200, height=700, scale=2)


def build_pdf_pack(
    *,
    df_valid: pd.DataFrame,
    offices: Sequence[dict],
    office_id: str,
    method: str,
    best_label: str,
    office_address: str,
) -> bytes:
    d_off = df_valid[df_valid["officeID"].astype(str) == str(office_id)].copy()

    if method != best_label:
        d = d_off[d_off["method"].astype(str) == str(method)].copy()
        df_emp = d.copy()
    else:
        if len(d_off):
            idx = d_off.groupby("employeeID")["travel_time_min"].idxmin()
            df_emp = d_off.loc[idx].copy()
        else:
            df_emp = d_off.copy()

    df_emp["travel_time_min"] = pd.to_numeric(df_emp["travel_time_min"], errors="coerce")
    df_emp = df_emp.dropna(subset=["lat", "lon", "travel_time_min"]).copy()

    off_row = next((o for o in offices if str(o.get("officeID")) == str(office_id)), None)
    if off_row is None:
        raise ValueError("Office not found for PDF pack.")
    office_lat = float(off_row["lat"])
    office_lon = float(off_row["lon"])

    bands_rows = []
    for o in offices:
        oid = str(o["officeID"])
        dd = df_valid[df_valid["officeID"].astype(str) == oid].copy()

        if method != best_label:
            dd = dd[dd["method"].astype(str) == str(method)].copy()
            vals = pd.to_numeric(dd["travel_time_min"], errors="coerce").dropna().to_numpy()
        else:
            if len(dd):
                best = dd.groupby("employeeID")["travel_time_min"].min()
                vals = pd.to_numeric(best, errors="coerce").dropna().to_numpy()
            else:
                vals = np.array([])

        n = int(vals.size)
        le30 = int(np.sum(vals <= 30)) if n else 0
        b30_45 = int(np.sum((vals > 30) & (vals <= 45))) if n else 0
        b45_60 = int(np.sum((vals > 45) & (vals <= 60))) if n else 0
        gt60 = int(np.sum(vals > 60)) if n else 0

        def pct(k: int) -> float:
            return (k / n) * 100.0 if n else 0.0

        bands_rows.append(
            {
                "officeID": oid,
                "officeLabel": str(o["address"]).split(",")[0],
                "n": n,
                "le30": pct(le30),
                "b30_45": pct(b30_45),
                "b45_60": pct(b45_60),
                "gt60": pct(gt60),
            }
        )

    bands_df = pd.DataFrame(bands_rows).sort_values("officeLabel").reset_index(drop=True)
    office_labels = bands_df["officeLabel"].astype(str).tolist()

    try:
        map_png = _plotly_employee_scatter_map_png_bytes(
            df_emp,
            office_lat,
            office_lon,
            office_address,
            title="Employees - {0} - {1}".format(office_address, method),
        )
    except Exception:
        map_png = _fig_map_png_bytes(df_emp, office_lat, office_lon)

    try:
        bands_png = _plotly_bands_png_bytes(bands_df, office_labels)
    except Exception:
        bands_png = _fig_bands_png_bytes(bands_df, office_labels)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(15 * mm, h - 18 * mm, "Commute Analysis - Summary")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, h - 26 * mm, "Office: {0}".format(office_address))
    c.drawString(15 * mm, h - 32 * mm, "Method: {0}".format(method))

    img1 = ImageReader(io.BytesIO(map_png))
    c.drawImage(img1, 15 * mm, 20 * mm, width=w - 30 * mm, height=110 * mm, preserveAspectRatio=True, mask="auto")

    vals = df_emp["travel_time_min"].dropna().to_numpy()
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, 145 * mm, "Employees (included): {0:,}".format(len(vals)))
    if len(vals):
        c.drawString(15 * mm, 139 * mm, "Median (mins): {0:.1f}".format(np.quantile(vals, 0.5)))
        c.drawString(15 * mm, 133 * mm, "P90 (mins): {0:.1f}".format(np.quantile(vals, 0.9)))
        c.drawString(15 * mm, 127 * mm, "Mean (mins): {0:.1f}".format(np.mean(vals)))

    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(15 * mm, h - 18 * mm, "Distribution by thresholds")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, h - 26 * mm, "Share of employees by commute time band - {0}".format(method))

    img2 = ImageReader(io.BytesIO(bands_png))
    c.drawImage(img2, 15 * mm, 30 * mm, width=w - 30 * mm, height=150 * mm, preserveAspectRatio=True, mask="auto")

    c.setFont("Helvetica", 9)
    c.drawString(15 * mm, 20 * mm, "Stacked bars show share of employees by band: <=30, 30-45, 45-60, >60 minutes.")

    c.save()
    return buf.getvalue()
