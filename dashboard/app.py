"""SpaceGuard AI - Space Situational Awareness & Collision Risk Monitoring Dashboard.

Streamlit interactive multi-view aerospace dashboard.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.conjunction import ConjunctionEngine, ConjunctionEvent
from src.data_loader import CATALOG_GROUPS, CelesTrakLoader
from src.feature_engineering import (
    FEATURE_COLUMNS,
    AstrodynamicFeatureExtractor,
    SyntheticConjunctionGenerator,
)
from src.ml_model import CLASS_NAMES, ConjunctionRiskMLPipeline
from src.orbital_engine import EARTH_RADIUS_KM, OrbitalEngine
from src.visualizer import OrbitVisualizer

# Page Configuration
st.set_page_config(
    page_title="SpaceGuard AI | Orbital Risk & Debris Tracking",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Deep Navy / Electric Aerospace CSS
SPACE_CSS = """
<style>
    /* Global App Background */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .space-header {
        background: linear-gradient(90deg, #0e1e38 0%, #172a4d 50%, #0e1e38 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 240, 255, 0.08);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111A2E;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 240, 255, 0.4);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #00F0FF;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }
    
    /* Risk Badges */
    .badge-critical {
        background-color: rgba(255, 51, 102, 0.2);
        color: #FF3366;
        border: 1px solid #FF3366;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-warning {
        background-color: rgba(255, 153, 0, 0.2);
        color: #FF9900;
        border: 1px solid #FF9900;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-monitoring {
        background-color: rgba(0, 240, 255, 0.15);
        color: #00F0FF;
        border: 1px solid #00F0FF;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-low {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Disclaimer Box */
    .disclaimer-banner {
        background: rgba(255, 153, 0, 0.08);
        border-left: 4px solid #FF9900;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #CBD5E1;
        margin: 14px 0;
    }
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_orbital_engines():
    """Cache singleton orbital and visualizer instances."""
    orb_engine = OrbitalEngine()
    viz_engine = OrbitVisualizer(engine=orb_engine)
    ml_pipe = ConjunctionRiskMLPipeline(models_dir=PROJECT_ROOT / "models")
    ml_pipe.load_models()
    return orb_engine, viz_engine, ml_pipe


@st.cache_data(ttl=300)
def load_all_catalogs(use_cache: bool = True):
    """Load authentic CelesTrak active satellites and space debris catalogs."""
    loader = CelesTrakLoader(raw_dir=PROJECT_ROOT / "data" / "raw")
    catalogs = {}
    metadata = {}

    for grp_key in ["stations", "active", "cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris"]:
        recs, meta = loader.fetch_group(grp_key, fmt="json", force_refresh=not use_cache)
        catalogs[grp_key] = recs
        metadata[grp_key] = meta

    primaries = catalogs.get("stations", []) + catalogs.get("active", [])
    debris = (
        catalogs.get("cosmos-2251-debris", [])
        + catalogs.get("iridium-33-debris", [])
        + catalogs.get("fengyun-1c-debris", [])
    )

    return primaries, debris, catalogs, metadata


# Initialize Engines
orb_engine, viz_engine, ml_pipe = get_orbital_engines()


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=600&auto=format&fit=crop",
        caption="Earth Orbital Environment",
        use_container_width=True,
    )
    st.markdown("### 🛰️ **SpaceGuard AI Control**")
    st.caption("AI-Powered Orbital Debris Tracking & Collision Risk System")

    st.markdown("---")
    st.markdown("#### ⚙️ **Screening Parameters**")

    screening_hours = st.slider("Screening Window (Hours)", min_value=6, max_value=72, value=24, step=6)
    step_sec = st.select_slider("Propagation Step (Sec)", options=[30, 60, 120, 300], value=120)
    dist_thresh = st.slider("Spatial Envelope (km)", min_value=25, max_value=250, value=120, step=5)

    st.markdown("---")
    st.markdown("#### 🤖 **Risk Classifier Model**")
    selected_model = st.radio(
        "Active Inference Architecture",
        options=["XGBoost", "Random Forest", "Physics Baseline"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 🌐 **Data Ingestion Mode**")
    force_refresh = st.checkbox("Force Online CelesTrak Query", value=False)
    if st.button("🔄 Refresh Orbital Ephemerides", use_container_width=True):
        st.cache_data.clear()
        st.success("Refreshed catalog cache!")
        st.rerun()

    st.markdown("---")
    st.caption("Astrodynamic Propagator: **SGP4 (WGS-84 / TEME)**")
    st.caption("Data Mirror: **CelesTrak CCSDS OMM**")


# Load Catalogs
primaries, debris, raw_catalogs, meta_dict = load_all_catalogs(use_cache=not force_refresh)
total_objects = len(primaries) + len(debris)


# ==============================================================================
# SCREENING ENGINE EXECUTION (CACHED)
# ==============================================================================
@st.cache_data(ttl=120)
def run_cached_screening(prim_count: int, deb_count: int, hours: float, step: float, thresh: float):
    conj_engine = ConjunctionEngine(
        engine=orb_engine,
        screening_threshold_km=thresh,
        warning_threshold_km=50.0,
        critical_threshold_km=10.0,
    )
    now_utc = datetime.now(timezone.utc)
    events = conj_engine.screen_catalog(
        primary_objects=primaries,
        secondary_objects=debris,
        start_dt=now_utc,
        duration_hours=hours,
        step_seconds=step,
        max_events=30,
    )
    return events, now_utc


screening_events, screening_epoch = run_cached_screening(
    len(primaries), len(debris), screening_hours, step_sec, dist_thresh
)

# Extract Features & Predict with ML Model
feat_extractor = AstrodynamicFeatureExtractor()
event_rows = []
for ev in screening_events:
    feats = feat_extractor.extract_from_event(ev)
    ml_out = ml_pipe.predict(feats, model_type=selected_model.lower().replace(" ", "_"))
    row = ev.to_dict()
    row["ml_predicted_risk"] = ml_out["predicted_risk"]
    row["ml_probabilities"] = ml_out["probabilities"]
    row["physics_baseline_risk"] = ml_out["physics_baseline_risk"]
    event_rows.append(row)

df_events = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()


# ==============================================================================
# HEADER BANNER
# ==============================================================================
st.markdown(
    """
    <div class="space-header">
        <h1 style="margin:0; font-size:32px; color:#FFFFFF; font-weight:700;">
            🛰️ SpaceGuard AI <span style="font-size:18px; color:#00F0FF; font-weight:400;">| Orbital Debris Tracking & Collision Risk System</span>
        </h1>
        <p style="margin:6px 0 0 0; color:#94A3B8; font-size:14px;">
            Autonomous Resident Space Object (RSO) tracking, SGP4 analytical propagation,
            conjunction assessment screening, and machine learning risk stratification.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# MAIN TABS INTERFACE
# ==============================================================================
tabs = st.tabs([
    "📊 Overview & Telemetry",
    "🌐 3D Orbital Tracking",
    "⚠️ Collision Risk Analysis",
    "🤖 Machine Learning Benchmarks",
    "💾 Data Management & Provenance",
])


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------------------------
with tabs[0]:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    crit_count = len(df_events[df_events["ml_predicted_risk"] == "Critical"]) if not df_events.empty else 0
    warn_count = len(df_events[df_events["ml_predicted_risk"] == "Warning"]) if not df_events.empty else 0

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Total Tracked RSOs</div>
                <div class="metric-value">{total_objects}</div>
                <div style="font-size:12px; color:#94A3B8;">Authentic CelesTrak elements</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Active Payloads</div>
                <div class="metric-value" style="color:#00F0FF;">{len(primaries)}</div>
                <div style="font-size:12px; color:#94A3B8;">Stations & Active Sats</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Debris Fragments</div>
                <div class="metric-value" style="color:#FF3366;">{len(debris)}</div>
                <div style="font-size:12px; color:#94A3B8;">Cosmos, Iridium, Fengyun</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Conjunctions Detected</div>
                <div class="metric-value" style="color:#FFD700;">{len(df_events)}</div>
                <div style="font-size:12px; color:#94A3B8;">Threshold &le; {dist_thresh} km</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col5:
        color = "#FF3366" if crit_count > 0 else ("#FF9900" if warn_count > 0 else "#00E676")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Priority Alerts</div>
                <div class="metric-value" style="color:{color};">{crit_count + warn_count}</div>
                <div style="font-size:12px; color:#94A3B8;">{crit_count} Critical | {warn_count} Warning</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer-banner">
            <b>⚠️ Astrodynamic Verification Note:</b> Close approach metrics are derived via numerical SGP4 propagation.
            Risk levels represent algorithmic screening classifications for operational triage, not true statistical collision
            probabilities ($P_c$), which require proprietary covariance matrices from space surveillance radar networks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview Visualizations
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🛰️ Tracked Objects Altitude Distribution")
        # Compute current altitudes for all objects
        alt_data = []
        for obj in primaries:
            elems = orb_engine.extract_orbital_elements(obj)
            alt_data.append({"Name": obj.get("OBJECT_NAME"), "Type": "Active Payload", "Perigee Alt (km)": elems.perigee_altitude_km, "Apogee Alt (km)": elems.apogee_altitude_km, "Regime": elems.orbit_regime})
        for obj in debris:
            elems = orb_engine.extract_orbital_elements(obj)
            alt_data.append({"Name": obj.get("OBJECT_NAME"), "Type": "Space Debris", "Perigee Alt (km)": elems.perigee_altitude_km, "Apogee Alt (km)": elems.apogee_altitude_km, "Regime": elems.orbit_regime})

        df_alt = pd.DataFrame(alt_data)
        fig_alt = px.scatter(
            df_alt,
            x="Perigee Alt (km)",
            y="Apogee Alt (km)",
            color="Type",
            hover_name="Name",
            color_discrete_map={"Active Payload": "#00F0FF", "Space Debris": "#FF3366"},
            template="plotly_dark",
        )
        fig_alt.update_layout(paper_bgcolor="#111A2E", plot_bgcolor="#111A2E", height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_alt, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Conjunction Risk Tier Breakdown")
        if not df_events.empty:
            risk_counts = df_events["ml_predicted_risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk Tier", "Count"]
            fig_pie = px.pie(
                risk_counts,
                values="Count",
                names="Risk Tier",
                color="Risk Tier",
                color_discrete_map={"Critical": "#FF3366", "Warning": "#FF9900", "Low": "#00E676", "Monitoring": "#00F0FF"},
                hole=0.45,
                template="plotly_dark",
            )
            fig_pie.update_layout(paper_bgcolor="#111A2E", plot_bgcolor="#111A2E", height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No close encounters detected within the current spatial envelope.")


# ------------------------------------------------------------------------------
# TAB 2: 3D ORBITAL TRACKING
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🌐 Interactive 3D Earth & Resident Space Object Fleet")

    col_track_ctrl1, col_track_ctrl2, col_track_ctrl3 = st.columns([2, 2, 2])
    with col_track_ctrl1:
        search_kw = st.text_input("🔍 Search Object Name / NORAD ID", value="")
    with col_track_ctrl2:
        regime_filter = st.multiselect("Orbit Regime", options=["LEO", "MEO", "GEO", "HEO"], default=["LEO"])
    with col_track_ctrl3:
        highlight_orbit = st.selectbox(
            "Plot Full Orbital Trajectory",
            options=["None"] + [p.get("OBJECT_NAME", "SAT") for p in primaries],
            index=1 if primaries else 0,
        )

    # Compute instantaneous state vectors at now_utc
    now_utc = datetime.now(timezone.utc)
    active_states = []
    debris_states = []

    for obj in primaries:
        if search_kw and (search_kw.lower() not in str(obj.get("OBJECT_NAME", "")).lower() and search_kw not in str(obj.get("NORAD_CAT_ID", ""))):
            continue
        elems = orb_engine.extract_orbital_elements(obj)
        if regime_filter and elems.orbit_regime not in regime_filter:
            continue
        rec = orb_engine.omm_to_satrec(obj)
        state = orb_engine.propagate(rec, now_utc, omm_fallback=obj)
        active_states.append((obj, state))

    for obj in debris:
        if search_kw and (search_kw.lower() not in str(obj.get("OBJECT_NAME", "")).lower() and search_kw not in str(obj.get("NORAD_CAT_ID", ""))):
            continue
        elems = orb_engine.extract_orbital_elements(obj)
        if regime_filter and elems.orbit_regime not in regime_filter:
            continue
        rec = orb_engine.omm_to_satrec(obj)
        state = orb_engine.propagate(rec, now_utc, omm_fallback=obj)
        debris_states.append((obj, state))

    # Generate highlighted orbit loop
    trajectories = []
    if highlight_orbit != "None":
        match_obj = next((o for o in primaries if o.get("OBJECT_NAME") == highlight_orbit), None)
        if match_obj:
            rec = orb_engine.omm_to_satrec(match_obj)
            elems = orb_engine.extract_orbital_elements(match_obj)
            path_pts = orb_engine.generate_orbit_path(rec, now_utc, period_minutes=elems.orbital_period_minutes, num_points=120, omm_fallback=match_obj)
            trajectories.append((highlight_orbit, path_pts))

    # Optionally display top conjunction event on the 3D globe
    top_event = screening_events[0] if screening_events else None

    fig_3d = viz_engine.create_orbital_scene(
        active_states=active_states,
        debris_states=debris_states,
        selected_trajectories=trajectories,
        conjunction_event=top_event,
        title=f"SpaceGuard AI Live Ephemerides (UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')})",
    )
    st.plotly_chart(fig_3d, use_container_width=True, height=650)


# ------------------------------------------------------------------------------
# TAB 3: COLLISION RISK ANALYSIS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("⚠️ Close-Approach Screening Events")

    if not df_events.empty:
        display_cols = [
            "target_name",
            "chaser_name",
            "tca_iso",
            "miss_distance_km",
            "relative_speed_km_s",
            "radial_separation_km",
            "physics_risk_level",
            "ml_predicted_risk",
        ]
        
        # Format table for clean aerospace presentation
        table_df = df_events[display_cols].copy()
        table_df["miss_distance_km"] = table_df["miss_distance_km"].round(2)
        table_df["relative_speed_km_s"] = table_df["relative_speed_km_s"].round(2)
        table_df["radial_separation_km"] = table_df["radial_separation_km"].abs().round(2)
        table_df.columns = [
            "Target Object",
            "Chaser Debris",
            "Time of Closest Approach (TCA UTC)",
            "Miss Dist (km)",
            "Rel Speed (km/s)",
            "Radial Sep (km)",
            "Physics Baseline Tier",
            f"ML Pred ({selected_model})",
        ]

        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔬 Deep Astrodynamic Event Inspector")
        event_idx = st.selectbox(
            "Select Conjunction Event to Inspect",
            range(len(screening_events)),
            format_func=lambda i: f"Event #{i+1}: {screening_events[i].target_name} vs {screening_events[i].chaser_name} (Miss: {screening_events[i].miss_distance_km:.1f} km)",
        )

        selected_ev = screening_events[event_idx]
        ev_meta = df_events.iloc[event_idx]

        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            st.markdown(f"**Target Satellite:** `{selected_ev.target_name}` (ID: {selected_ev.target_id})")
            st.markdown(f"**Target Altitude at TCA:** `{selected_ev.target_altitude_km:.1f} km`")
            st.markdown(f"**Encounter Epoch (TCA):** `{selected_ev.tca.strftime('%Y-%m-%d %H:%M:%S UTC')}`")
        with e_col2:
            st.markdown(f"**Chaser Debris:** `{selected_ev.chaser_name}` (ID: {selected_ev.chaser_id})")
            st.markdown(f"**Chaser Altitude at TCA:** `{selected_ev.chaser_altitude_km:.1f} km`")
            st.markdown(f"**Miss Distance:** `{selected_ev.miss_distance_km:.2f} km`")
        with e_col3:
            st.markdown(f"**Relative Velocity:** `{selected_ev.relative_speed_km_s:.2f} km/s`")
            st.markdown(f"**Physics Baseline Tier:** `{selected_ev.physics_risk_level}`")
            st.markdown(f"**ML Predicted Tier:** `{ev_meta['ml_predicted_risk']}`")

        # RIC Frame Breakdown
        st.markdown("#### 📐 Radial-InTrack-CrossTrack (RIC) Decomposition at TCA")
        ric_col1, ric_col2 = st.columns(2)
        with ric_col1:
            st.markdown("**Relative Position Offsets:**")
            st.write(f"- **Radial Separation ($\Delta r_R$):** `{selected_ev.radial_separation_km:+.2f} km`")
            st.write(f"- **In-Track Separation ($\Delta r_I$):** `{selected_ev.intrack_separation_km:+.2f} km`")
            st.write(f"- **Cross-Track Separation ($\Delta r_C$):** `{selected_ev.crosstrack_separation_km:+.2f} km`")
        with ric_col2:
            st.markdown("**Relative Velocity Components:**")
            st.write(f"- **Radial Velocity ($\Delta v_R$):** `{selected_ev.relative_radial_vel_km_s:+.3f} km/s`")
            st.write(f"- **In-Track Velocity ($\Delta v_I$):** `{selected_ev.relative_intrack_vel_km_s:+.3f} km/s`")
            st.write(f"- **Cross-Track Velocity ($\Delta v_C$):** `{selected_ev.relative_crosstrack_vel_km_s:+.3f} km/s`")
    else:
        st.success(f"Clear skies! No close approaches detected within {dist_thresh} km over the next {screening_hours} hours.")


# ------------------------------------------------------------------------------
# TAB 4: MACHINE LEARNING BENCHMARKS
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🤖 Machine Learning Model Benchmarks & Scientific Rigor")

    st.markdown(
        """
        <div class="disclaimer-banner">
            <b>Scientific Framing & Label Integrity:</b> Real orbital conjunction catalogs do not contain authentic collision ground-truth labels.
            To avoid data contamination, SpaceGuard AI trains its machine learning models on a <b>clearly segregated synthetic simulation dataset</b>
            (3,000 physically consistent encounter scenarios across LEO/MEO shells).
            Models are systematically validated against an empirical held-out test partition and benchmarked against a transparent <b>deterministic physics baseline</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Benchmark Metrics Table
    if ml_pipe.evaluation_results:
        bench_data = []
        for name, res in ml_pipe.evaluation_results.items():
            bench_data.append({
                "Model Architecture": name,
                "Accuracy": f"{res.accuracy * 100:.2f}%",
                "Macro F1": f"{res.macro_f1:.4f}",
                "Weighted F1": f"{res.weighted_f1:.4f}",
            })
        st.table(pd.DataFrame(bench_data))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🎯 Feature Importance Ranking")
        res_to_show = ml_pipe.evaluation_results.get(selected_model, ml_pipe.evaluation_results.get("Random Forest"))
        if res_to_show and res_to_show.feature_importances:
            imp_df = pd.DataFrame(list(res_to_show.feature_importances.items()), columns=["Feature", "Importance"])
            imp_df = imp_df[imp_df["Importance"] > 0].sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", template="plotly_dark", color="Importance", color_continuous_scale="Blues")
            fig_imp.update_layout(paper_bgcolor="#111A2E", plot_bgcolor="#111A2E", height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_m2:
        st.markdown("#### 📊 Confusion Matrix on Held-Out Test Set")
        if res_to_show and res_to_show.confusion_matrix:
            cm = np.array(res_to_show.confusion_matrix)
            fig_cm = px.imshow(
                cm,
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                text_auto=True,
                color_continuous_scale="Viridis",
                template="plotly_dark",
                labels=dict(x="Predicted Risk Tier", y="Ground Truth Simulation Tier"),
            )
            fig_cm.update_layout(paper_bgcolor="#111A2E", plot_bgcolor="#111A2E", height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    # Synthetic Simulation Dataset Viewer
    st.markdown("---")
    st.markdown("#### 🧪 Synthetic Training Dataset Inspector (`data/processed/simulated_conjunction_dataset.csv`)")
    sim_path = PROJECT_ROOT / "data" / "processed" / "simulated_conjunction_dataset.csv"
    if sim_path.exists():
        df_sim = pd.read_csv(sim_path)
        st.dataframe(df_sim.head(10), use_container_width=True)
        st.caption(f"Showing 10 of {len(df_sim)} simulation encounter scenarios. Identified strictly with `data_nature = SYNTHETIC_SIMULATION`.")


# ------------------------------------------------------------------------------
# TAB 5: DATA MANAGEMENT & PROVENANCE
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("💾 Orbital Data Provenance & Catalog Management")

    st.markdown("Authentic Space Catalog feeds managed via **CelesTrak CCSDS OMM API**.")

    cat_rows = []
    for grp_key, meta in meta_dict.items():
        cat_rows.append({
            "Group Key": grp_key,
            "Group Name": meta.get("group_name", "N/A"),
            "Object Type": meta.get("object_type", "N/A"),
            "Data Source": meta.get("source", "N/A"),
            "Fetch Epoch (UTC)": meta.get("fetch_timestamp", "N/A"),
            "Format": meta.get("format", "JSON"),
            "Valid Records": meta.get("valid_count", 0),
            "Discarded": meta.get("discarded_count", 0),
            "Offline Fallback Active": meta.get("offline_mode", False),
        })

    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📂 Raw Orbital Data Files on Disk (`data/raw/`)")
    raw_files = list((PROJECT_ROOT / "data" / "raw").glob("*.json"))
    raw_file_info = []
    for rf in raw_files:
        raw_file_info.append({
            "Filename": rf.name,
            "Size (KB)": round(rf.stat().st_size / 1024.0, 2),
            "Modified": datetime.fromtimestamp(rf.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    st.table(pd.DataFrame(raw_file_info))
