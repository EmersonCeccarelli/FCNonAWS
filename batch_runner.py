import os, sys, subprocess, time
import numpy as np
import pandas as pd
import pygrib
from pathlib import Path

# ── Environment ───────────────────────────────────────────────────────────────
os.chdir("/home/ec2-user/projects/FCN4AWS")
os.environ["PATH"]       = f"/home/ec2-user/.local/bin:{os.environ.get('PATH','')}"
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"

# Read CDS API key from ~/.cdsapirc rather than hardcoding
import re
cdsapirc = Path.home() / ".cdsapirc"
match = re.search(r"key:\s*(.+)", cdsapirc.read_text())
if not match:
    raise RuntimeError("Could not find key in ~/.cdsapirc — check your CDS API setup (README step 1.7)")
os.environ["CDSAPI_KEY"] = match.group(1).strip()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/ec2-user/projects/FCN4AWS")
GRIB_DIR    = BASE_DIR / "fcnv2_batch_gribs"
TRACKS_DIR  = BASE_DIR / "fcnv2_batch_tracks"
RESULTS_DIR = BASE_DIR / "fcnv2_batch_results"
LOG_FILE    = BASE_DIR / "batch_run.log"
LEAD_TIME   = 240

for d in [GRIB_DIR, TRACKS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

def log(msg):
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Load NOAA data ────────────────────────────────────────────────────────────
log("Loading NOAA data...")
hurdat = pd.read_csv(
    BASE_DIR / "noaa_tracks_clean.csv",
    parse_dates=["datetime"]
)
storm_starts = pd.read_csv(
    BASE_DIR / "noaa_storm_starts.csv",
    parse_dates=["start_datetime"]
)

# ── Build sample (minimal version for debug run) ──────────────────────────────
modern    = storm_starts[storm_starts["start_datetime"].dt.year >= 1980].copy()
ts_storms = hurdat[hurdat["status"].isin(["HU","TS"])]["storm_id"].unique()
modern    = modern[modern["storm_id"].isin(ts_storms)]

peak = hurdat.groupby("storm_id").agg(peak_wind=("max_wind_kt","max")).reset_index()

def categorize(wind):
    if wind >= 137: return "Cat5"
    elif wind >= 113: return "Cat4"
    elif wind >= 96:  return "Cat3"
    elif wind >= 83:  return "Cat2"
    elif wind >= 64:  return "Cat1"
    else: return "TS"

peak["category"] = peak["peak_wind"].apply(categorize)
modern["date_str"] = modern["start_datetime"].dt.strftime("%Y%m%d")
modern["time_str"] = modern["start_datetime"].dt.strftime("%H%M")
modern_full = modern.merge(peak, on="storm_id")

# Full stratified sample — 5 storms per decade + must-haves
modern_full["decade"] = (modern_full["start_datetime"].dt.year // 10) * 10
sample = (
    modern_full
    .groupby("decade", group_keys=False)
    .apply(lambda g: g.sample(min(5, len(g)), random_state=42))
    .reset_index(drop=True)
)
 
must_have = ["AL041992","AL081999","AL122005","AL092008",
             "AL182012","AL112017","AL052019","AL092021"]
must_have_df = modern_full[modern_full["storm_id"].isin(must_have)]
sample = pd.concat([sample, must_have_df]).drop_duplicates(
    subset="storm_id").reset_index(drop=True)
 
log(f"Sample: {len(sample)} storms | Est. {len(sample)*3/60:.1f} hrs | ~${len(sample)*3/60*0.60:.2f}")

# Debug: two storm run
# sample = modern_full[modern_full["storm_id"].isin(["AL112017", "AL122005"])].reset_index(drop=True)
# log(f"Debug sample: {len(sample)} storms — IRMA (AL112017) + KATRINA (AL122005)")

# ── Tracker parameters ────────────────────────────────────────────────────────
# NOAA-seeded search window — how far from the NOAA reference we look at all
NOAA_SEARCH_RADIUS_KM    = 500

# Continuity — how far the tracker can jump step-to-step
# Dynamic: radius = clip(last_speed * SPEED_FACTOR, MIN, MAX)
CONTINUITY_SPEED_FACTOR  = 1.5   # multiplier on last-step speed
CONTINUITY_RADIUS_MIN    = 100   # floor — stationary/slow storms
CONTINUITY_RADIUS_MAX    = 300   # ceiling — fast recurving storms

# ── Geometry helpers ──────────────────────────────────────────────────────────
def haversine_pts(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two scalar points."""
    r = np.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(dlon/2)**2)
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def dist_grid(global_lats, global_lons, ref_lat, ref_lon):
    """Haversine distance (km) from every grid point to (ref_lat, ref_lon)."""
    ref_lon_360 = ref_lon % 360
    dlat = np.radians(global_lats - ref_lat)
    dlon = np.radians(global_lons - ref_lon_360)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(ref_lat)) *
         np.cos(np.radians(global_lats)) * np.sin(dlon/2)**2)
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def haversine_col(df, lat1, lon1, lat2, lon2):
    """Vectorised haversine for a DataFrame."""
    r = np.radians
    dlat = r(df[lat2] - df[lat1])
    dlon = r(df[lon2] - df[lon1])
    a = (np.sin(dlat/2)**2 +
         np.cos(r(df[lat1])) * np.cos(r(df[lat2])) * np.sin(dlon/2)**2)
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ── Track extractor ───────────────────────────────────────────────────────────
def extract_track_guided(grib_file, noaa_track_df,
                         noaa_search_radius_km=NOAA_SEARCH_RADIUS_KM,
                         continuity_radius_min=CONTINUITY_RADIUS_MIN,
                         continuity_radius_max=CONTINUITY_RADIUS_MAX,
                         continuity_speed_factor=CONTINUITY_SPEED_FACTOR,
                         debug=True):
    """
    MSLP-guided TC tracker with:
      - NOAA-seeded search window (first-pass region of interest)
      - Dynamic continuity radius scaling with recent storm speed
      - Dead-reckoning fallback when no minimum found in window
    """
    grbs = pygrib.open(str(grib_file))
    try:
        msgs = grbs.select(shortName='msl')
    except Exception:
        msgs = grbs.select(shortName='prmsl', typeOfLevel='meanSea')

    global_lats, global_lons = msgs[0].latlons()
    noaa_track_df = noaa_track_df.copy()
    noaa_track_df["valid_time"] = pd.to_datetime(noaa_track_df["valid_time"])

    track       = []
    first_valid = None

    # Continuity state — last two picked positions for speed + dead reckoning
    prev_lat,  prev_lon  = None, None   # step N-1
    pprev_lat, pprev_lon = None, None   # step N-2

    for msg in msgs:
        valid_time = pd.Timestamp(msg.validDate)
        if first_valid is None:
            first_valid = valid_time
        try:
            step = int(msg.stepRange.split('-')[-1])
        except Exception:
            step = int((valid_time - first_valid).total_seconds() / 3600)

        data = msg.values

        # ── NOAA reference for this timestep ─────────────────────────
        time_diffs = (noaa_track_df["valid_time"] - valid_time).abs()
        nearest    = noaa_track_df.loc[time_diffs.idxmin()]
        ref_lat, ref_lon = nearest["lat"], nearest["lon"]

        # ── Step 1: NOAA-seeded coarse search window ─────────────────
        d_ref  = dist_grid(global_lats, global_lons, ref_lat, ref_lon)
        masked = np.where(d_ref <= noaa_search_radius_km, data, np.inf)

        # ── Step 2: Dynamic continuity constraint ─────────────────────
        continuity_radius      = None
        dead_reckoned_lat      = None
        dead_reckoned_lon      = None

        if prev_lat is not None:
            # Dynamic radius — scale on last observed step speed
            if pprev_lat is not None:
                last_speed_km = haversine_pts(pprev_lat, pprev_lon % 360,
                                              prev_lat,  prev_lon  % 360)
            else:
                last_speed_km = 150  # reasonable first-step default (~25 kt)

            continuity_radius = float(np.clip(
                last_speed_km * continuity_speed_factor,
                continuity_radius_min,
                continuity_radius_max
            ))

            # Dead-reckoned expected position: extrapolate velocity vector
            if pprev_lat is not None:
                dead_reckoned_lat = prev_lat + (prev_lat - pprev_lat)
                dead_reckoned_lon = prev_lon + (prev_lon - pprev_lon)
            else:
                dead_reckoned_lat = prev_lat
                dead_reckoned_lon = prev_lon

            # Apply continuity mask on top of NOAA window
            d_prev = dist_grid(global_lats, global_lons,
                               prev_lat, prev_lon % 360)
            masked = np.where(d_prev <= continuity_radius, masked, np.inf)

        # ── Step 3: Pick minimum — or dead reckon if window is empty ──
        fallback = False
        if np.all(np.isinf(masked)):
            # Continuity window + NOAA window found nothing coherent.
            # Dead reckon: place storm at extrapolated position.
            picked_lat = dead_reckoned_lat
            picked_lon = dead_reckoned_lon
            # Sample MSLP at the dead-reckoned point for logging
            d_dr   = dist_grid(global_lats, global_lons,
                               picked_lat, picked_lon % 360)
            idx_dr = np.unravel_index(np.argmin(d_dr), d_dr.shape)
            mslp_val = float(data[idx_dr]) / 100.0
            fallback = True
        else:
            idx        = np.unravel_index(np.argmin(masked), masked.shape)
            picked_lat = float(global_lats[idx])
            picked_lon = float(global_lons[idx])
            if picked_lon > 180:
                picked_lon -= 360
            mslp_val = float(data[idx]) / 100.0

        # ── Debug output ──────────────────────────────────────────────
        if debug:
            jump_km  = (haversine_pts(prev_lat, prev_lon % 360,
                                      picked_lat, picked_lon % 360)
                        if prev_lat is not None else 0)
            dist_ref = haversine_pts(ref_lat, ref_lon, picked_lat, picked_lon)
            cr_str   = f"{continuity_radius:.0f}" if continuity_radius else "n/a (step 0)"
            fb_tag   = " ◄ DEAD RECKON" if fallback else ""
            print(f"    step {step:3d}h | "
                  f"ref ({ref_lat:.1f},{ref_lon:.1f}) | "
                  f"picked ({picked_lat:.1f},{picked_lon:.1f}) | "
                  f"mslp {mslp_val:.1f} hPa | "
                  f"dist_ref {dist_ref:.0f} km | "
                  f"jump {jump_km:.0f} km | "
                  f"cont_r {cr_str} km"
                  f"{fb_tag}")

        # ── Update continuity state ───────────────────────────────────
        pprev_lat, pprev_lon = prev_lat,  prev_lon
        prev_lat,  prev_lon  = picked_lat, picked_lon

        track.append({
            "step_hours": step,
            "valid_time": valid_time,
            "lat":        picked_lat,
            "lon":        picked_lon,
            "mslp_hPa":  mslp_val,
            "fallback":   fallback,
        })

    grbs.close()
    return pd.DataFrame(track)

# ── Main batch loop ───────────────────────────────────────────────────────────
def already_done(storm_id):
    return (TRACKS_DIR / f"{storm_id}_fcnv2_track.csv").exists()

results = []
total   = len(sample)

for i, row in sample.iterrows():
    storm_id = row["storm_id"]
    date_str = row["date_str"]
    time_str = row["time_str"]
    name     = row["storm_name"]
    n        = list(sample.index).index(i) + 1

    track_out = TRACKS_DIR / f"{storm_id}_fcnv2_track.csv"
    grib_out  = GRIB_DIR   / f"{storm_id}_{date_str}_{time_str}.grib"

    if already_done(storm_id):
        log(f"[{n}/{total}] SKIP {storm_id} ({name}) — already done")
        continue

    log(f"[{n}/{total}] START {storm_id} ({name}) IC={date_str} {time_str}")
    t0 = time.time()

    cmd = [
        "ai-models",
        "--assets", str(Path.home() / ".cache/ai-models/fourcastnetv2-small"),
        "--input", "cds",
        "--date", date_str,
        "--time", time_str,
        "--lead-time", str(LEAD_TIME),
        "--path", str(grib_out),
        "fourcastnetv2-small"
    ]

    result  = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  ERROR {storm_id}: {result.stderr[-400:]}")
        continue

    log(f"  FCNv2 done in {elapsed:.0f}s")

    # Extract track
    noaa_storm = hurdat[hurdat["storm_id"] == storm_id].copy()
    noaa_storm = noaa_storm.rename(columns={"datetime": "valid_time"})
    noaa_storm["valid_time"] = pd.to_datetime(noaa_storm["valid_time"])

    try:
        fcnv2_track = extract_track_guided(grib_out, noaa_storm, debug=True)
        fcnv2_track["storm_id"]   = storm_id
        fcnv2_track["storm_name"] = name
        fcnv2_track.to_csv(track_out, index=False)
        fallback_count = fcnv2_track["fallback"].sum()
        log(f"  Track saved: {len(fcnv2_track)} steps | "
            f"dead-reckoned steps: {fallback_count}")
    except Exception as e:
        log(f"  Track extraction failed: {e}")
        grib_out.unlink(missing_ok=True)
        continue

    # Compute error (exclude dead-reckoned steps from metric)
    merged = pd.merge(
        fcnv2_track[["valid_time","step_hours","lat","lon","mslp_hPa","fallback"]],
        noaa_storm[["valid_time","lat","lon","min_pressure_mb"]],
        on="valid_time", suffixes=("_fcn","_noaa")
    )

    if len(merged) > 0:
        merged["dist_km"] = haversine_col(
            merged, "lat_fcn","lon_fcn","lat_noaa","lon_noaa")

        # Report both: all steps and tracked-only (no dead reckon)
        mean_err     = merged["dist_km"].mean()
        max_err      = merged["dist_km"].max()
        tracked_only = merged[~merged["fallback"]]
        mean_tracked = tracked_only["dist_km"].mean() if len(tracked_only) else float("nan")

        results.append({
            "storm_id":        storm_id,
            "storm_name":      name,
            "year":            row["start_datetime"].year,
            "category":        row["category"],
            "n_matched":       len(merged),
            "n_dead_reckoned": int(merged["fallback"].sum()),
            "mean_error_km":   mean_err,
            "mean_error_tracked_km": mean_tracked,
            "max_error_km":    max_err,
            "inference_sec":   elapsed
        })
        log(f"  Mean error (all):     {mean_err:.0f} km | Max: {max_err:.0f} km")
        log(f"  Mean error (tracked): {mean_tracked:.0f} km | "
            f"Dead-reckoned steps: {int(merged['fallback'].sum())}")

    grib_out.unlink(missing_ok=True)
    log(f"  GRIB deleted")

# ── Final summary ─────────────────────────────────────────────────────────────
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "batch_results.csv", index=False)

    total_sec = results_df["inference_sec"].sum()
    total_min = total_sec / 60
    total_hrs = total_min / 60
    cost      = total_hrs * 0.60

    log(f"\n{'='*60}")
    log(f"Batch complete: {len(results)}/{total} storms processed")
    log(f"Overall mean error (all steps):     {results_df['mean_error_km'].mean():.0f} km")
    log(f"Overall mean error (tracked only):  {results_df['mean_error_tracked_km'].mean():.0f} km")
    log(f"Overall mean inference time:        {results_df['inference_sec'].mean():.0f}s/storm")
    log(f"Total inference time:               {total_min:.1f} min ({total_hrs:.2f} hrs)")
    log(f"Estimated EC2 cost:                 ~${cost:.2f}")
    log(f"{'='*60}")
else:
    log("No results — all storms failed or were skipped.")