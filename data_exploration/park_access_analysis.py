import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box
import matplotlib.patches as mpatches

# get Bologna admin boundary
bologna_admin = ox.geocode_to_gdf("Bologna, Italy")
bologna_poly  = bologna_admin.geometry.iloc[0]

CRS_METRIC   = "EPSG:32632"
results      = {}

green_tags = {
    "leisure": ["park", "garden"]
    }
green_keywords = ["Parchi", "Ville"]

# ── 1. OSM ────────────────────────────────────────────────────────────────────
print("Fetching OSM...")
osm_gdfs = []
for key, values in green_tags.items():
    for val in values:
        try:
            gdf = ox.features_from_place("Bologna, Italy", tags={key: val})
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            gdf["green_type"] = f"{key}={val}"
            # keep access column if it exists
            cols = ["geometry", "green_type", "name"]
            if "access" in gdf.columns:
                cols.append("access")
            osm_gdfs.append(gdf[cols])
        except:
            pass

osm = pd.concat(osm_gdfs).reset_index(drop=True)
osm = gpd.GeoDataFrame(osm, crs="EPSG:4326")

# filter private AFTER concat, on the DataFrame
if "access" in osm.columns:
    osm = osm[~osm["access"].isin(["private", "no"])]
    print(f"  Removed private features, remaining: {len(osm):,}")

osm = osm.to_crs(CRS_METRIC)
osm["area_km2"] = osm.geometry.area / 1e6
osm = osm[osm["area_km2"] >= 0.01]
osm["source"] = "OSM"
print(f"  OSM final: {len(osm):,} features >= 0.01 km²")
results["OSM"] = osm

# ── 2. Regione ER ─────────────────────────────────────────────────────────────
print("\nLoading Regione ER...")
er = gpd.read_file("/Users/terezasaskova/Downloads/Uso_Suolo_Dettaglio_2020_ETRS89/Uso_Suolo_Dettaglio_2020_ETRS89.shp")
# clip ER to same boundary instead of bbox

bologna_poly_25832 = gpd.GeoSeries([bologna_poly], crs="EPSG:4326").to_crs("EPSG:25832").iloc[0]

er_bologna = er[er.geometry.intersects(bologna_poly_25832)].copy()

er_green = er_bologna[
    er_bologna["DESCR"].str.contains("|".join(green_keywords), case=False, na=False)
].copy()

er_green = er_green.to_crs(CRS_METRIC)
er_green["area_km2"] = er_green.geometry.area / 1e6
er_green = er_green[er_green["area_km2"] >= 0.01]
er_green["green_type"] = er_green["DESCR"]
er_green["source"] = "Regione ER"
er_green["name"] = er_green["DESCR"]

print(f"  Regione ER final: {len(er_green):,} features >= 0.01 km²")
print(f"  Breakdown:\n{er_green['DESCR'].value_counts()}")
print(f"  Total green area: {er_green['area_km2'].sum():.2f} km²")
results["Regione ER"] = er_green

# ── 3. Comparison summary ─────────────────────────────────────────────────────
print("\n=== COMPARISON SUMMARY ===")
rows = []
for source, gdf in results.items():
    rows.append({
        "source": source,
        "n_features": len(gdf),
        "total_area_km2": round(gdf["area_km2"].sum(), 2),
        "median_size_km2": round(gdf["area_km2"].median(), 4),
        "n_large (>0.1km2)": (gdf["area_km2"] >= 0.1).sum(),
    })
print(pd.DataFrame(rows).to_string(index=False))

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Public Green Space — Bologna", fontsize=14, fontweight="bold")
colors = {"OSM": "green", "Regione ER": "coral"}

for col, (source, gdf) in enumerate(results.items()):
    color = colors[source]
    gdf.plot(ax=axes[0, col], color=color, alpha=0.6, linewidth=0.3)
    axes[0, col].set_title(f"{source} — {len(gdf):,} features")
    axes[0, col].axis("off")

    gdf["area_km2"].clip(upper=1).hist(bins=40, ax=axes[1, col], color=color, alpha=0.7)
    axes[1, col].set_title(f"{source} — Area distribution")
    axes[1, col].set_xlabel("km²")
    axes[1, col].set_ylabel("count")

plt.tight_layout()
plt.savefig("greenspace_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 4. Overlay plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Public Green Space Comparison — Bologna", fontsize=14, fontweight="bold")

# ── Left: spatial overlay ─────────────────────────────────────────────────────
ax_map = axes[0]

# same category breakdown for both sources
category_colors = {
    "leisure=park":             "#2ecc71",
    "leisure=garden":           "#27ae60",
    "leisure=nature_reserve":   "#1a6b3c",
    "landuse=recreation_ground":"#f39c12",
    "Parchi":                   "#e74c3c",
    "Ville":                    "#c0392b",
}

for source, gdf in results.items():
    for green_type, group in gdf.groupby("green_type"):
        color = category_colors.get(green_type, "#95a5a6")
        group.plot(ax=ax_map, color=color, alpha=0.5, linewidth=0.3)

# legend
patches = [
    mpatches.Patch(color=color, alpha=0.7, label=f"{cat}")
    for cat, color in category_colors.items()
]
ax_map.legend(handles=patches, loc="lower left", fontsize=8, framealpha=0.9)
ax_map.set_title("OSM (greens) vs Regione ER (reds) — spatial overlay")
ax_map.axis("off")

# ── Right: area distribution overlay ─────────────────────────────────────────
ax_hist = axes[1]
osm_data = results["OSM"]["area_km2"].clip(upper=1)
er_data  = results["Regione ER"]["area_km2"].clip(upper=1)

ax_hist.hist(osm_data, bins=40, color="green", alpha=0.5, label=f"OSM ({len(results['OSM']):,})")
ax_hist.hist(er_data,  bins=40, color="coral", alpha=0.5, label=f"Regione ER ({len(results['Regione ER']):,})")
ax_hist.set_title("Area distribution overlay (clipped at 1 km²)")
ax_hist.set_xlabel("km²")
ax_hist.set_ylabel("count")
ax_hist.legend()

plt.tight_layout()
plt.savefig("greenspace_overlay.png", dpi=150, bbox_inches="tight")
plt.show()