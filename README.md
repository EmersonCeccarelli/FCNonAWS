# FCN4AWS — FourCastNetV2 Hurricane Track Analysis on AWS

Evaluate NVIDIA's FourCastNetV2 (FCNv2) AI weather model against NOAA HURDAT2 ground truth for Atlantic hurricane track forecasting. Runs on an AWS EC2 GPU instance via JupyterLab.

---

## Project Overview

**Research question:** How accurately does FCNv2 predict Atlantic hurricane tracks compared to observed HURDAT2 data?

**Pipeline summary:**
```
HURDAT2 (NOAA) → noaa_historical_tracks_analysis.ipynb → noaa_data/
                                                              ↓
                                              batch_runner.py (called by notebook)
                                              ERA5 via CDS API → FCNv2 inference
                                                              ↓
                                          FCNv2_Visualizations.ipynb → plots & stats
```

---

## 1. AWS VM Setup

### Recommended instance type
- **`g4dn.xlarge`** (Tesla T4, 16GB VRAM) — sufficient for FCNv2-small inference
- AMI: Amazon Linux 2 or Ubuntu 22.04
- Storage: 100GB+ gp3 (GRIB files and weights are large)

### On a fresh instance, run these once:

```bash
# Update system
sudo yum update -y   # Amazon Linux
# or: sudo apt update && sudo apt upgrade -y   # Ubuntu

# Install system dependencies for cartopy / eccodes
sudo yum install -y gcc gcc-c++ python3-devel proj proj-devel geos geos-devel eccodes
# Ubuntu: sudo apt install -y python3-pip libproj-dev libgeos-dev libeccodes-dev

# Verify GPU
nvidia-smi
```

### Install Python dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

> **Note:** `cartopy` may require system-level `proj` and `geos` libraries (installed above).

---

## 2. Configure CDS API (ERA5 access)

Create a free account at https://cds.climate.copernicus.eu and get your API key from your profile page.

```bash
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY_HERE
verify: 1
EOF
chmod 600 ~/.cdsapirc

# Test it
python3 -c "import cdsapi; c = cdsapi.Client(); print('CDS connection OK')"
```

---

## 3. Download FCNv2 Model Weights

Weights are downloaded automatically via `ai-models` (~3.5GB, stored in `~/.cache/ai-models/`):

```bash
ai-models --download-assets fourcastnetv2-small
```

> This references NVIDIA's FCNv2 model. For background on the model architecture and training, see the original team's repo: https://github.com/ikhadir/FCN4AWS

---

## 4. Clone This Repo & Launch JupyterLab

```bash
cd ~/projects
git clone https://github.com/RalphtheWaldo/FCN4AWS.git
cd FCN4AWS

# Launch JupyterLab (use tmux/screen so it survives disconnect)
tmux new -s jupyter
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
# Ctrl+B then D to detach
```

Access via SSH tunnel from your local machine:
```bash
ssh -L 8888:localhost:8888 ec2-user@YOUR_EC2_IP
```
Then open `http://127.0.0.1:8888` in your browser.

---

## 5. Run the Pipeline

### Step 1 — Generate NOAA data
Open and run **`noaa_historical_tracks_analysis.ipynb`** top to bottom.

This downloads HURDAT2 from NOAA, cleans and parses it, and writes two files to `noaa_data/`:
- `noaa_tracks_clean.csv` — full track records for all Atlantic storms
- `noaa_storm_starts.csv` — one row per storm with genesis time, location, and category

### Step 2 — Run batch FCNv2 inference
The notebook above calls **`batch_runner.py`** directly. You can also run it standalone from a terminal (recommended for long overnight runs):

```bash
cd ~/projects/FCN4AWS
python3 batch_runner.py 2>&1 | tee batch_run.log
```

For each storm (1980+, TS/HU strength), it:
1. Downloads ERA5 initial conditions via CDS API
2. Runs FCNv2-small inference (240hr / 10-day lead time)
3. Extracts the hurricane track via MSLP minimum (NOAA-guided radius search)
4. Computes haversine error vs HURDAT2 ground truth
5. Saves results to `fcnv2_batch_results/batch_results.csv`

GRIBs are deleted after track extraction to manage disk space.

### Step 3 — Visualize results
Open and run **`FCNv2_Visualizations.ipynb`**.

Produces spaghetti plots, MSLP time series comparisons, and aggregate error statistics across all processed storms.

---

## 6. File Structure

```
FCN4AWS/
├── noaa_historical_tracks_analysis.ipynb   # Step 1: NOAA data download & cleaning
├── batch_runner.py                          # Step 2: FCNv2 batch inference engine
├── FCNv2_Visualizations.ipynb              # Step 3: plots & error analysis
├── requirements.txt
├── .gitignore
└── README.md

# Generated on first run (gitignored):
├── noaa_data/
│   ├── noaa_tracks_clean.csv
│   └── noaa_storm_starts.csv
├── fcnv2_batch_gribs/      # temp storage, deleted after extraction
├── fcnv2_batch_tracks/     # per-storm FCNv2 track CSVs
└── fcnv2_batch_results/
    └── batch_results.csv
```

---

## 7. Tips

- **Long runs:** Use `tmux` or `screen` so batch inference survives SSH disconnects.
- **Resume:** `batch_runner.py` skips storms that already have a track file in `fcnv2_batch_tracks/` — safe to restart.
- **Cost:** A `g4dn.xlarge` runs ~$0.526/hr on-demand. ~609 storms at ~3 min each ≈ 30 hrs ≈ $16 in GPU time.
- **CDS throttling:** CDS may rate-limit downloads. The batch runner handles retries automatically.
- **Weights location:** `~/.cache/ai-models/` — these persist across reboots, don't need to re-download.

---

## References

- [FourCastNetV2 (NVIDIA)](https://github.com/NVlabs/FourCastNet)
- [ai-models (ECMWF)](https://github.com/ecmwf-lab/ai-models)
- [HURDAT2 dataset (NOAA)](https://www.nhc.noaa.gov/data/#hurdat)
- [Copernicus CDS (ERA5)](https://cds.climate.copernicus.eu)
- Original FCN4AWS tutorial repo: https://github.com/ikhadir/FCN4AWS
