CITY = "verona"  

#=========================================================

import math
import os
from pathlib import Path
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.strtree import STRtree
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import sys, h3



from config.config import (
    ROOT_PATH, CRS_METRIC, CRS_WGS,
    DB_PATH, GH_PORT, WALKING_DISTANCE, EUCLID_CUTOFF,
    connect_db, gdf_to_duckdb, load_table, gpkg_to_duckdb,
    get_city_cfg, get_output_paths,
)


sys.path.insert(0, str(ROOT_PATH))
os.chdir(ROOT_PATH)

cfg   = get_city_cfg(CITY)
src   = cfg["data_sources"]
paths = get_output_paths(CITY)
paths["gpkg"] = paths["gpkg"].replace(".gpkg", "_2.gpkg")

FIG_DIR = paths["fig_dir"]
os.makedirs(FIG_DIR, exist_ok=True)

print(f"City : {CITY}")
print(f"Figures: {FIG_DIR}")

TEAL  = "#1D9E75"
BLUE  = "#378ADD"
CORAL = "#F0997B"
LGRAY = "#D3D1C7"
AMBER = "#EF9F27"
TEXT  = "#2C2C2A"
SGRAY = "#888780"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.dpi":        150,
})

paths = get_output_paths(CITY)
paths["gpkg"] = paths["gpkg"].replace(".gpkg", "_2.gpkg")

buildings_routed = gpd.read_file(paths["gpkg"], layer="buildings")
parks = gpd.read_file(paths["gpkg"], layer="parks")
results = gpd.read_file(paths["gpkg"])

con = connect_db()
boundary_gdf = load_table(con, "raw", f"{CITY}_boundary")
boundary_wgs84 = boundary_gdf.to_crs(CRS_WGS).geometry.union_all()

park_area = (
    parks[["park_id", "geometry"]]
    .copy()
    .to_crs(CRS_METRIC)
    .assign(area_ha=lambda df: df.geometry.area / 10_000)
    [["park_id", "area_ha"]]
)

results = results.merge(
    park_area.rename(columns={"park_id": "nearest_park_id"}),
    on="nearest_park_id",
    how="left",
)

total   = len(results)
avg     = results["within_300m"].mean() * 100
within  = results["within_300m"].sum()
no_park = results["nearest_park_id"].isna().sum()
routed  = results["nearest_park_id"].notna().sum()
outside_n = routed - within

print(f"Total addresses:          {total:>8,}")
print(f"Within 300m (≥1ha park):  {within:>8,}  ({within/total*100:.1f}%)")
print(f"Outside 300m:             {outside_n:>8,}  ({outside_n/total*100:.1f}%)")
print(f"No park reachable:        {no_park:>8,}  ({no_park/total*100:.1f}%)")

# ── Fig 1: H3 map ─────────────────────────────────────────────────────────────
H3_RES = 8

results_wgs = results.to_crs(CRS_WGS)
results["h3_cell"] = results_wgs.geometry.apply(
    lambda geom: h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, H3_RES)
)

h3_stats = (
    results.groupby("h3_cell")["within_300m"]
    .agg(total="count", within="sum")
    .assign(pct=lambda x: x["within"] / x["total"] * 100)
    .reset_index()
)

from shapely.geometry import Polygon as ShapelyPolygon

h3_polygons = []
for _, row in h3_stats.iterrows():
    coords = h3.cell_to_boundary(row["h3_cell"])
    verts = [[lng, lat] for lat, lng in coords]
    h3_polygons.append({
        "h3_cell": row["h3_cell"],
        "pct":     row["pct"],
        "total":   row["total"],
        "geometry": ShapelyPolygon(verts),
    })

h3_gdf = gpd.GeoDataFrame(h3_polygons, crs=CRS_WGS)

results_wgs = results.to_crs(CRS_WGS)
results["longitude"] = results_wgs.geometry.x
results["latitude"]  = results_wgs.geometry.y

norm = Normalize(vmin=0, vmax=100)
lon_range = results["longitude"].max() - results["longitude"].min()
lat_range = results["latitude"].max()  - results["latitude"].min()
lat_mid   = results["latitude"].mean()
aspect    = lon_range / (lat_range * np.cos(np.radians(lat_mid)))

fig, ax = plt.subplots(figsize=(11, 11 / aspect))
fig.patch.set_facecolor("#F5F4F0")
ax.set_facecolor("#F5F4F0")

h3_gdf.plot(column="pct", cmap="RdYlGn", linewidth=0.4,
    edgecolor="#AAAAAA", ax=ax, vmin=0, vmax=100, alpha=0.75, zorder=2)

sm = ScalarMappable(cmap="RdYlGn", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
cbar.set_label("% addresses within 300m (parks ≥1ha)", fontsize=9, color=SGRAY)
cbar.ax.tick_params(labelsize=8)

ax.set_aspect("equal")
ax.set_title(f"{CITY.title()}: park access by H3 cell (parks ≥1ha)\nRed=low · Green=high",
    fontsize=13, fontweight="500", pad=12, color=TEXT)
ax.set_xlabel("Longitude", fontsize=9, color=SGRAY)
ax.set_ylabel("Latitude",  fontsize=9, color=SGRAY)
ax.tick_params(labelsize=7)
ax.text(0.01, 0.02, f"City avg: {avg:.1f}% within 300m walk",
    transform=ax.transAxes, fontsize=7, color=SGRAY, ha="left", va="bottom")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig01_h3_walkability_map.png", dpi=180)
plt.show()

# ── Fig 2: Euclid vs Walk ─────────────────────────────────────────────────────
parks_metric = parks.to_crs(CRS_METRIC)
park_id_to_idx = {pid: i for i, pid in enumerate(parks_metric["park_id"])}

matched = results[results["nearest_park_id"].notna()].copy().reset_index(drop=True)
civico_metric = matched.to_crs(CRS_METRIC).reset_index(drop=True)

def dist_to_park_boundary(row_idx, park_id):
    pidx = park_id_to_idx.get(park_id)
    if pidx is None:
        return np.nan
    return civico_metric.geometry.iloc[row_idx].distance(
        parks_metric.geometry.iloc[pidx].boundary
    )

matched["euclid_dist_m"] = [
    dist_to_park_boundary(i, row["nearest_park_id"])
    for i, row in matched.iterrows()
]

dist_thresholds = [100, 200, 300]
x = np.arange(len(dist_thresholds))
w = 0.35

walk_pct_real   = np.array([(results["walk_dist_m"] <= t).sum() / total * 100 for t in dist_thresholds])
euclid_pct_real = np.array([(matched["euclid_dist_m"] <= t).sum() / total * 100 for t in dist_thresholds])

fig, ax = plt.subplots(figsize=(8, 4.5))

b1 = ax.bar(x - w/2, walk_pct_real,   width=w, color=TEAL, label="Network walk ≤ threshold", zorder=2, linewidth=0)
b2 = ax.bar(x + w/2, euclid_pct_real, width=w, color=BLUE, label="Euclidean ≤ threshold",    zorder=2, linewidth=0)

for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels([f"{t}m" for t in dist_thresholds], fontsize=10)
ax.set_ylabel("% of addresses compliant", fontsize=10, color=SGRAY)
ax.set_xlabel("Distance threshold", fontsize=10, color=SGRAY)
ax.set_title(f"{CITY.title()}: network walk vs. euclidean (parks ≥1ha)",
    fontsize=13, fontweight="500", pad=12, color=TEXT)
ax.set_ylim(0, max(euclid_pct_real) * 1.25)
ax.grid(axis="y", linewidth=0.4, color=LGRAY, zorder=0)
ax.legend(handles=[b1, b2], fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig02_euclid_vs_walk.png", bbox_inches="tight", dpi=150)
plt.show()

# ── Fig 3: Dot map ────────────────────────────────────────────────────────────
DOT_OUT = "#E83A2F"
DOT_IN  = "#1DB87A"

inside_df  = results[results["within_300m"]]
outside_df = results[~results["within_300m"]]

fig, ax = plt.subplots(figsize=(11, 11 / aspect))
fig.patch.set_facecolor("#F5F4F0")
ax.set_facecolor("#F5F4F0")

ax.scatter(outside_df["longitude"], outside_df["latitude"],
    s=1.8, color=DOT_OUT, alpha=0.55, linewidths=0,
    label=f"Outside 300m: {len(outside_df):,} ({len(outside_df)/total*100:.1f}%)",
    rasterized=True, zorder=2)
ax.scatter(inside_df["longitude"], inside_df["latitude"],
    s=1.8, color=DOT_IN, alpha=0.55, linewidths=0,
    label=f"Within 300m: {len(inside_df):,} ({len(inside_df)/total*100:.1f}%)",
    rasterized=True, zorder=3)

ax.set_aspect("equal")
ax.set_xlabel("Longitude", fontsize=10, color=SGRAY)
ax.set_ylabel("Latitude",  fontsize=10, color=SGRAY)
ax.set_title(f"300m park access gap — {CITY.title()} (parks ≥1ha)",
    fontsize=13, fontweight="500", pad=14, color=TEXT)
ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
ax.tick_params(labelsize=7)
for spine in ax.spines.values():
    spine.set_edgecolor(LGRAY); spine.set_linewidth(0.5)

legend = ax.legend(fontsize=7, frameon=True, framealpha=0.92,
    loc="upper right", markerscale=6, handletextpad=0.4)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_facecolor("#F5F4F0")

ax.text(0.01, 0.02, f"City avg: {avg:.1f}% within 300m walk",
    transform=ax.transAxes, fontsize=7, color=SGRAY, ha="left", va="bottom")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig03_access_gap_map.png", dpi=200, bbox_inches="tight")
plt.show()