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

## Phase 1 — Initial VM Launch & Environment Setup

> Do this once on a fresh EC2 instance. After this phase is complete, the VM is ready for research.

### 1.1 Launch the EC2 Instance

In the AWS Console, launch a new instance with **exactly** these settings:

| Setting | Value |
|---|---|
| **AMI** | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Amazon Linux 2023) |
| **Instance type** | `g4dn.xlarge` (Tesla T4, 16GB VRAM) |
| **Storage** | 60 GB gp3 |
| **Key pair** | Create a new `.pem` key pair and save it somewhere safe on your local machine |
| **Security group** | Allow SSH. Recommended: restrict to **My IP** only. Leaving open to `0.0.0.0/0` works but exposes port 22 to the internet (will need to update the rule if your IP changes) |

> **Note:** The `g4dn.xlarge` instance type is not typically available by default and may require customer service or specific AWS approval to access. Additionally, this instance type automatically attaches a 125 GB NVMe instance store volume (ephemeral0). This cannot be removed but should be free of charge. Data stored on it is lost when the instance is stopped or terminated, which is expected and acceptable since all research code lives in the GitHub repo.

Once the instance is running, copy the **Public IPv4 address** from the AWS Console.

### 1.2 SSH Into the Instance

Run this from your local machine, substituting your `.pem` path and EC2 IP:

```bash
ssh -i "/path/to/your-key.pem" -L 8888:127.0.0.1:8888 ec2-user@<YOUR_EC2_IP>
```

The `-L 8888:127.0.0.1:8888` flag sets up the SSH tunnel so JupyterLab is accessible in your local browser at `http://127.0.0.1:8888`.

> **Note:** You should include the `-L` tunnel flag every time you SSH in if you plan to use JupyterLab.

### 1.3 Activate the PyTorch Environment

The AMI ships with a pre-built PyTorch environment. Activate it:

```bash
source /opt/pytorch/bin/activate
```

You should see `(pytorch)` appear in your terminal prompt. All subsequent commands should be run inside this environment.

### 1.4 Install JupyterLab

```bash
pip install jupyterlab
```

### 1.5 Set Up Project Directory & Clone the Repo

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/EmersonCeccarelli/FCNonAWS.git FCN4AWS
cd FCN4AWS
```
> **Note:** you will be prompted for a github username and **passkey**

> **Note:** If you plan to push changes back to GitHub from this instance, you will need to complete step 1.9 (Git credentials) before your first `git push`. Cloning does not require credentials since the repo is public.


### 1.6 Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `cartopy` requires system-level `proj` and `geos` libraries. On Amazon Linux 2023 these are typically pre-installed on the Deep Learning AMI. If you see errors, run:
> ```bash
> sudo dnf install -y proj proj-devel geos geos-devel
> ```

> **Note:** You may see dependency conflict warnings about `ml-dtypes` and `shap` requiring a newer numpy. These are pre-installed AMI packages and can be safely ignored — they do not affect the FCNv2 pipeline.

### 1.7 Configure the CDS API Key (ERA5 Access)

Create a free account at https://cds.climate.copernicus.eu and retrieve your API key from your profile page. Then:

```bash
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY_HERE
verify: 1
EOF
chmod 600 ~/.cdsapirc
```

Test it:
```bash
python3 -c "import cdsapi; c = cdsapi.Client(); print('CDS OK')"
```

### 1.8 Download FCNv2 Model Weights

The weights are ~3.3GB. Download them directly using `curl -L` (the `-L` flag is required to follow redirects):

```bash
mkdir -p ~/.cache/ai-models/fourcastnetv2-small
cd ~/.cache/ai-models/fourcastnetv2-small

curl -L -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
curl -L -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy
curl -L -O https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
```

Verify the downloads succeeded — `weights.tar` should be ~3.3GB:

```bash
ls -lh ~/.cache/ai-models/fourcastnetv2-small/
```

> **Note:** Do **not** use `ai-models --download-assets fourcastnetv2-small` — it fails due to incompatible `earthkit-data` and `numpy` versions on this AMI. The curl method above bypasses this entirely.

### 1.9 Configure Git Credentials

This must be done on **every new EC2 instance** — credentials are stored on the VM's local filesystem and are lost when the instance is terminated.

So you're not prompted for your username and token on every push:

```bash
git config --global credential.helper store
git config --global user.name "YourGitHubUsername"
git config --global user.email "your@email.com"
```

The first time you push you'll be prompted for your GitHub username and a Personal Access Token (not your password). After that, credentials are cached permanently.

> **Note:** GitHub Personal Access Tokens are generated at `github.com → Settings → Developer Settings → Personal access tokens → Tokens (classic)`. Make sure the token has `repo` scope.

### 1.10 Launch JupyterLab

```bash
cd ~/projects
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

> **Optional but recommended:** Wrap in `tmux` so JupyterLab survives SSH disconnects:
> ```bash
> tmux new -s jupyter
> cd ~/projects
> jupyter lab --no-browser --ip=127.0.0.1 --port=8888
> # Detach with Ctrl+B then D
> # Reconnect later with: tmux attach -t jupyter
> ```

---

## Phase 2 — Research Workflow

> After Phase 1, every subsequent session starts here.

### Reconnect to a Running Instance

Each time you restart your session, SSH in with the tunnel flag (the IP will likely change if the instance was stopped — check the AWS Console):

```bash
ssh -i "/path/to/your-key.pem" -L 8888:127.0.0.1:8888 ec2-user@<YOUR_EC2_IP>
source /opt/pytorch/bin/activate
tmux attach -t jupyter   # reattach to existing JupyterLab session if applicable
```

Then open `http://127.0.0.1:8888` in your browser.

### Step 1 — Generate NOAA Data

Open and run **`noaa_historical_tracks_analysis.ipynb`** top to bottom.

Downloads HURDAT2 from NOAA, cleans and parses it, and writes to `noaa_data/`:
- `noaa_tracks_clean.csv` — full track records for all Atlantic storms
- `noaa_storm_starts.csv` — one row per storm with genesis time, location, and category

### Step 2 — Run Batch FCNv2 Inference

The NOAA notebook calls **`batch_runner.py`** directly. For long overnight runs, call it from the terminal instead:

```bash
cd ~/projects/FCN4AWS
python3 batch_runner.py 2>&1 | tee batch_run.log
```

For each storm (1980+, TS/HU strength) it:
1. Downloads ERA5 initial conditions via CDS API
2. Runs FCNv2-small inference (240hr / 10-day lead time)
3. Extracts the hurricane track via MSLP minimum
4. Computes haversine error vs HURDAT2 ground truth
5. Saves results to `fcnv2_batch_results/batch_results.csv`

GRIBs are deleted after track extraction to manage disk space. The runner skips storms that already have a track file in `fcnv2_batch_tracks/` — safe to restart after interruption.

### Step 3 — Visualize Results

Open and run **`FCNv2_Visualizations.ipynb`**.

Produces spaghetti plots, MSLP time series comparisons, and aggregate error statistics.

---

## File Structure

```
FCN4AWS/
├── noaa_historical_tracks_analysis.ipynb   # Step 1: NOAA data download & cleaning
├── batch_runner.py                          # Step 2: FCNv2 batch inference engine
├── FCNv2_Visualizations.ipynb              # Step 3: plots & error analysis
├── requirements.txt
├── .gitignore
└── README.md

# Generated on first run (gitignored — not in repo):
├── noaa_data/
│   ├── noaa_tracks_clean.csv
│   └── noaa_storm_starts.csv
├── fcnv2_batch_gribs/      # temp GRIB storage, deleted after each storm
├── fcnv2_batch_tracks/     # per-storm FCNv2 track CSVs
└── fcnv2_batch_results/
    └── batch_results.csv

# Lives outside the repo (set up in Phase 1):
~/.cdsapirc                 # CDS API credentials
~/.cache/ai-models/         # FCNv2 model weights (~3.5GB)
```

---

## Cost Estimate

| Resource | Rate | Estimated total |
|---|---|---|
| `g4dn.xlarge` on-demand | ~$0.526/hr | ~609 storms × ~3 min each ≈ 30 hrs ≈ **$16** |
| EBS storage (60GB gp3) | ~$0.08/GB/month | ~**$5/month** |

Stop the instance when not in use to avoid unnecessary charges.

---

## References

- [FourCastNetV2 — NVIDIA](https://github.com/NVlabs/FourCastNet)
- [ai-models — ECMWF](https://github.com/ecmwf-lab/ai-models)
- [HURDAT2 dataset — NOAA](https://www.nhc.noaa.gov/data/#hurdat)
- [Copernicus CDS — ERA5](https://cds.climate.copernicus.eu)
- Original FCN4AWS tutorial repo: https://github.com/ikhadir/FCN4AWS
