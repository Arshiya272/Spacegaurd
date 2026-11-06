# SpaceGuard AI 
### AI-Powered Space Debris Tracking and Collision Risk Analysis System

SpaceGuard AI is an advanced, production-grade space situational awareness (SSA) application engineered to track resident space objects (active satellites and space debris), compute high-precision orbital trajectories using SGP4, screen for dangerous orbital close approaches (conjunctions), and categorize collision risks using machine learning models validated against deterministic physics baselines.

---

##  Problem Statement

Low Earth Orbit (LEO) is experiencing an exponential rise in resident space objects due to mega-constellations and historical fragmentation debris (such as the Cosmos 2251 / Iridium 33 collision and the Fengyun-1C test). Satellite operators face thousands of close-approach conjunction warnings weekly. Operational teams need:
1. High-fidelity automated orbital propagation using validated models (SGP4).
2. Rapid geometric conjunction screening with exact Time of Closest Approach (TCA) and miss distance calculation.
3. Scientifically honest risk categorization that distinguishes physical screening geometry from statistical collision probability ($P_c$).
4. Modern 3D interactive situational awareness visualization and monitoring dashboards.

---

##  Key Features

- **Authentic Orbital Data Ingestion**: Direct integration with CelesTrak using standard CCSDS Orbit Mean-Elements Messages (OMM JSON/CSV) and backward-compatible Two-Line Element (TLE) parsing.
- **High-Fidelity SGP4 Propagation**: Propagates satellite and debris positions $\vec{r}(t)$ and velocities $\vec{v}(t)$ in the TEME/ECI coordinate frame with physical boundary validation.
- **Fast Conjunction Screening Engine**: Multi-stage pipeline featuring $O(1)$ perigee/apogee altitude shell pre-filtering followed by Golden-Section sub-second TCA search.
- **Radial-InTrack-CrossTrack (RIC) Decomposition**: Computes operational encounter geometry in the target's orbital reference frame ($\Delta r_R, \Delta r_I, \Delta r_C$ and relative velocity components).
- **Segregated ML Pipeline**: Trained on 3,000 synthetic simulation encounter scenarios across LEO/MEO regimes (XGBoost & Random Forest) to avoid fabricating collision labels on real satellite data.
- **Deterministic Physics Baseline Benchmark**: Every ML prediction is systematically cross-referenced against a transparent astrodynamic baseline.
- **Interactive 3D Plotly Visualization**: 3D Earth globe with equator/polar grids, color-coded satellite constellations and debris clouds, and encounter lines.
- **Streamlit Space-Themed Dashboard**: Multi-panel monitoring interface with live telemetry cards, 3D orbit viewer, conjunction tables, ML performance charts, and data provenance audit tables.
- **100% Offline Resilience**: Shipped with pre-cached authentic CelesTrak orbital datasets for space stations, active satellites, and debris fields (Cosmos 2251, Iridium 33, Fengyun-1C).

---

## System Architecture

```text
spaceguard_ai/
│
├── config/
│   └── config.yaml               # SGP4 parameters, CelesTrak endpoints & screening thresholds
│
├── data/
│   ├── raw/                      # Authentic CelesTrak OMM JSON feeds & metadata
│   │   ├── stations.json         # Space stations (ISS, CSS)
│   │   ├── active.json           # Active payloads (Hubble, Landsat, Sentinel, Starlink)
│   │   ├── cosmos-2251-debris.json # Cosmos 2251 collision fragments
│   │   ├── iridium-33-debris.json  # Iridium 33 collision fragments
│   │   └── fengyun-1c-debris.json  # Fengyun-1C fragmentation debris
│   └── processed/
│       └── simulated_conjunction_dataset.csv # 3,000 segregated simulation encounters
│
├── docs/
│   └── methodology.md            # Detailed astrodynamic formulas & ML derivations
│
├── models/                       # Serialized models & benchmark reports
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   └── evaluation_report.json
│
├── outputs/                      # Exported conjunction assessment outputs
│   ├── conjunction_report.json
│   └── conjunction_report.csv
│
├── scripts/                      # Executable CLI workflows
│   ├── fetch_data.py             # Fetch/refresh CelesTrak orbital catalogs
│   ├── train_model.py            # Train RF & XGBoost against physics baseline
│   └── run_screening.py          # Run end-to-end conjunction screening batch
│
├── src/                          # Modular core source code
│   ├── __init__.py
│   ├── data_loader.py            # CelesTrak OMM client, schema validation & caching
│   ├── orbital_engine.py         # SGP4 propagator, TEME frames & Keplerian utilities
│   ├── conjunction.py            # Screening engine, TCA golden-section search & RIC frames
│   ├── feature_engineering.py    # Astrodynamic feature extraction & simulation generator
│   ├── ml_model.py               # Random Forest & XGBoost risk classification pipeline
│   └── visualizer.py             # Interactive 3D Plotly Earth & orbit scene generator
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                    # Streamlit multi-tab aerospace monitoring dashboard
│
├── tests/                        # Automated Pytest test suite (16 tests, 100% pass)
│   ├── test_data_loader.py
│   ├── test_orbital_engine.py
│   ├── test_conjunction.py
│   └── test_ml_model.py
│
├── .gitignore                    # Exclusions for cache, venv, and large files
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

##  Scientific Methodology & Integrity

### 1. SGP4 Limits & Coordinates
- SGP4 models secular and periodic perturbations from Earth oblateness ($J_2, J_3, J_4$), atmospheric drag ($B^*$), and gravitational resonances.
- Propagation coordinates reside natively in the **TEME** (True Equator, Mean Equinox) frame. Transformations to Earth-Fixed (ECEF) coordinates apply Greenwich Mean Sidereal Time (GMST) rotations according to the IAU-1982 standard.
- Unphysical elements ($e \ge 1$, $n \le 0$) or decayed satellites (SGP4 error codes $>0$) are safely caught and handled.

### 2. Astrodynamic Baselines vs. ML Collision Risk
- **No Fabricated Ground Truth**: Real orbital observation catalogs do not contain labeled collisions. Therefore, SpaceGuard AI **never fabricates real-world collision labels**.
- **Role of Machine Learning**: ML is deployed strictly for **risk tier prioritization** (e.g., *Low*, *Warning*, *Critical*) based on encounter geometry, relative velocity, and orbital energy differences.
- **Physics Baseline**: A transparent deterministic rule ($d_{\text{miss}} \le 10\text{ km}$, $v_{\text{rel}} \ge 2\text{ km/s}$, $\Delta r_R \le 5\text{ km}$) serves as the scientific benchmark.
- **Statistical Disclaimer**: Calculation of true collision probability ($P_c$) requires covariance matrices ($\mathbf{C}_1, \mathbf{C}_2$) from tracking radar networks. Public TLE/OMM feeds do not contain covariance data; SpaceGuard AI explicitly displays this distinction in all UI views and exports.

---

##  Machine Learning Model Benchmarks

Trained on 3,000 physically consistent simulation encounter scenarios across LEO/MEO shells with an 75/25 stratified train/test split:

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Primary Role |
| :--- | :---: | :---: | :---: | :--- |
| **Physics Baseline** | **99.73%** | **0.9969** | **0.9973** | Deterministic Astrodynamic Anchor |
| **Random Forest** | **99.73%** | **0.9969** | **0.9973** | Multi-Tree Ensemble (150 estimators) |
| **XGBoost Classifier** | **99.87%** | **0.9984** | **0.9987** | Gradient Boosted Decision Trees |

*Top Astrodynamic Features by Importance*:
1. `miss_distance_km` (Geometric proximity at TCA)
2. `radial_separation_km` (Vertical separation between orbital shells)
3. `relative_speed_km_s` (Kinetic encounter energy)
4. `altitude_diff_km` (Semi-major axis differential)

---

## Technology Stack

- **Astrodynamics & Mechanics**: SGP4, Skyfield, NumPy, SciPy
- **Data Engineering**: Pandas, Requests, PyYAML
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Interactive Visualizations**: Plotly Graph Objects, Plotly Express
- **User Interface**: Streamlit (Custom Deep Navy & Neon Aerospace Styling)
- **Testing & Quality Assurance**: Pytest

---

##  Quickstart & Installation

### 1. Clone & Enter Project Directory
```bash
git clone https://github.com/<YOUR_USERNAME>/spaceguard-ai.git
cd spaceguard-ai
```

### 2. Activate Virtual Environment & Install Dependencies
```bash
# Activate existing project venv
source .venv/bin/activate

# Or install from requirements.txt
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
*(All 16 unit and integration tests pass in ~1 second).*

---

## 🖥️ Execution Instructions

### A. Launch the Streamlit Monitoring Dashboard
```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### B. Run Batch Conjunction Screening via CLI
```bash
source .venv/bin/activate
python3 scripts/run_screening.py --hours 24 --step-sec 120 --threshold-km 150 --model xgboost
```
Outputs are written to `outputs/conjunction_report.json` and `outputs/conjunction_report.csv`.

### C. Retrain ML Models & Generate New Synthetic Datasets
```bash
source .venv/bin/activate
python3 scripts/train_model.py --samples 3000 --test-size 0.25
```

### D. Refresh CelesTrak Orbital Data (Online Mode)
```bash
source .venv/bin/activate
python3 scripts/fetch_data.py --refresh
```

---

## Dashboard Walkthrough

1. ** Overview & Telemetry**: Instant KPI cards displaying Total Tracked RSOs, Active Payloads, Tracked Debris, Screened Conjunctions, and Priority Alerts, accompanied by an altitude distribution chart and risk breakdown pie.
2. ** 3D Orbital Tracking**: Interactive 3D Earth globe rendered with continent borders, equator line, polar axis, active satellite markers (electric cyan), debris fragments (bright crimson), and highlighted orbital trajectories.
3. ** Collision Risk Analysis**: Interactive table of close-approach events with sortable miss distances, relative velocities, UTC TCA timestamps, and deep RIC frame decomposition.
4. ** Machine Learning Benchmarks**: Accuracy and F1 benchmark tables comparing XGBoost, Random Forest, and the Physics Baseline, accompanied by feature importance bar charts and confusion matrices.
5. ** Data Management & Provenance**: Audit log of all ingested CelesTrak groups, fetch epochs, validation statistics, and local disk cache status.
