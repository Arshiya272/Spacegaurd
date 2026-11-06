"""SpaceGuard AI - Interactive 3D Orbit & Conjunction Visualizer.

Generates high-performance interactive 3D astrodynamics graphics via Plotly,
featuring an accurate Earth sphere, satellite constellations, debris fields,
orbital trajectory ellipses, and close-encounter geometry at TCA.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from src.conjunction import ConjunctionEvent
from src.orbital_engine import EARTH_RADIUS_KM, OrbitalEngine, StateVector

logger = logging.getLogger(__name__)

# Aerospace Color Palette
COLOR_BG = "#0B0F19"
COLOR_EARTH_SURF = [[0.0, "#081b29"], [0.5, "#0b2545"], [1.0, "#13315c"]]
COLOR_EQUATOR = "#00bcd4"
COLOR_SATELLITE = "#00F0FF"  # Electric Cyan
COLOR_DEBRIS = "#FF3366"     # Bright Crimson
COLOR_CONJUNCTION = "#FFD700"# Neon Gold
COLOR_LINE_WARN = "#FF9900"   # Amber Warning
COLOR_GRID = "rgba(255, 255, 255, 0.1)"


class OrbitVisualizer:
    """3D Plotly-based orbital mechanics and space situational awareness visualizer."""

    def __init__(self, engine: Optional[OrbitalEngine] = None):
        self.engine = engine or OrbitalEngine()

    @staticmethod
    def _create_earth_surface(num_points: int = 40) -> go.Surface:
        """Construct 3D Earth spherical surface mesh."""
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)
        x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
        y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
        z = EARTH_RADIUS_KM * np.outer(np.ones(np.size(u)), np.cos(v))

        surface = go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=COLOR_EARTH_SURF,
            showscale=False,
            hoverinfo="skip",
            opacity=0.92,
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
            name="Earth (R=6378 km)",
        )
        return surface

    @staticmethod
    def _create_reference_rings() -> List[go.Scatter3d]:
        """Construct equator and polar reference coordinate lines."""
        theta = np.linspace(0, 2 * np.pi, 100)
        eq_x = (EARTH_RADIUS_KM + 10) * np.cos(theta)
        eq_y = (EARTH_RADIUS_KM + 10) * np.sin(theta)
        eq_z = np.zeros_like(theta)

        equator = go.Scatter3d(
            x=eq_x,
            y=eq_y,
            z=eq_z,
            mode="lines",
            line=dict(color=COLOR_EQUATOR, width=2, dash="dash"),
            hoverinfo="skip",
            name="Equator",
            showlegend=False,
        )

        polar_z = np.linspace(-EARTH_RADIUS_KM * 1.3, EARTH_RADIUS_KM * 1.3, 20)
        polar_axis = go.Scatter3d(
            x=np.zeros_like(polar_z),
            y=np.zeros_like(polar_z),
            z=polar_z,
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.3)", width=1.5, dash="dot"),
            hoverinfo="skip",
            name="Polar Axis",
            showlegend=False,
        )

        return [equator, polar_axis]

    def create_orbital_scene(
        self,
        active_states: List[Tuple[Dict[str, Any], StateVector]],
        debris_states: List[Tuple[Dict[str, Any], StateVector]],
        selected_trajectories: Optional[List[Tuple[str, np.ndarray]]] = None,
        conjunction_event: Optional[ConjunctionEvent] = None,
        title: str = "SpaceGuard AI - 3D Orbital Situational Awareness",
    ) -> go.Figure:
        """Generate interactive 3D space scene showing satellites, debris, and orbits."""
        fig = go.Figure()

        # 1. Earth mesh and reference grids
        fig.add_trace(self._create_earth_surface())
        for ring in self._create_reference_rings():
            fig.add_trace(ring)

        # 2. Active Satellites Scatter
        if active_states:
            sat_x = [s.position_km[0] for _, s in active_states if s.is_valid]
            sat_y = [s.position_km[1] for _, s in active_states if s.is_valid]
            sat_z = [s.position_km[2] for _, s in active_states if s.is_valid]
            sat_text = [
                f"<b>{meta.get('OBJECT_NAME', 'SAT')}</b><br>"
                f"NORAD ID: {meta.get('NORAD_CAT_ID', 'N/A')}<br>"
                f"Type: {meta.get('OBJECT_TYPE', 'ACTIVE')}<br>"
                f"Altitude: {s.altitude_km:.1f} km<br>"
                f"Speed: {s.speed_km_s:.2f} km/s"
                for meta, s in active_states if s.is_valid
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=sat_x,
                    y=sat_y,
                    z=sat_z,
                    mode="markers",
                    marker=dict(size=4.5, color=COLOR_SATELLITE, symbol="circle", opacity=0.9),
                    text=sat_text,
                    hoverinfo="text",
                    name=f"Active Satellites ({len(sat_x)})",
                )
            )

        # 3. Debris Objects Scatter
        if debris_states:
            deb_x = [s.position_km[0] for _, s in debris_states if s.is_valid]
            deb_y = [s.position_km[1] for _, s in debris_states if s.is_valid]
            deb_z = [s.position_km[2] for _, s in debris_states if s.is_valid]
            deb_text = [
                f"<b>{meta.get('OBJECT_NAME', 'DEBRIS')}</b><br>"
                f"NORAD ID: {meta.get('NORAD_CAT_ID', 'N/A')}<br>"
                f"Type: DEBRIS<br>"
                f"Altitude: {s.altitude_km:.1f} km<br>"
                f"Speed: {s.speed_km_s:.2f} km/s"
                for meta, s in debris_states if s.is_valid
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=deb_x,
                    y=deb_y,
                    z=deb_z,
                    mode="markers",
                    marker=dict(size=3.0, color=COLOR_DEBRIS, symbol="circle", opacity=0.85),
                    text=deb_text,
                    hoverinfo="text",
                    name=f"Space Debris ({len(deb_x)})",
                )
            )

        # 4. Trajectory Paths (Ellipses)
        if selected_trajectories:
            for name, path in selected_trajectories:
                if len(path) > 1:
                    fig.add_trace(
                        go.Scatter3d(
                            x=path[:, 0],
                            y=path[:, 1],
                            z=path[:, 2],
                            mode="lines",
                            line=dict(width=3, color=COLOR_SATELLITE),
                            name=f"Orbit: {name}",
                            hoverinfo="skip",
                        )
                    )

        # 5. Highlight Conjunction Event at TCA
        if conjunction_event is not None:
            p_tgt = np.array(conjunction_event.target_position_eci)
            p_chs = np.array(conjunction_event.chaser_position_eci)

            # Encounter Line between Target and Debris
            fig.add_trace(
                go.Scatter3d(
                    x=[p_tgt[0], p_chs[0]],
                    y=[p_tgt[1], p_chs[1]],
                    z=[p_tgt[2], p_chs[2]],
                    mode="lines+markers",
                    line=dict(color=COLOR_LINE_WARN, width=6, dash="solid"),
                    marker=dict(size=7, color=[COLOR_CONJUNCTION, COLOR_DEBRIS]),
                    name=f"Encounter ({conjunction_event.miss_distance_km:.1f} km)",
                    text=[
                        f"TARGET: {conjunction_event.target_name}",
                        f"DEBRIS: {conjunction_event.chaser_name}",
                    ],
                    hoverinfo="text",
                )
            )

        # Camera & Layout styling
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=17, color="#E0E6ED", family="monospace"),
                x=0.05,
                y=0.95,
            ),
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                font=dict(color="#C0C8D0", size=11),
                bgcolor="rgba(11, 15, 25, 0.75)",
                bordercolor="rgba(0, 240, 255, 0.2)",
                borderwidth=1,
                x=0.02,
                y=0.90,
            ),
            scene=dict(
                aspectmode="data",
                xaxis=dict(
                    title="X [km] (TEME)",
                    color="#8892B0",
                    gridcolor=COLOR_GRID,
                    showbackground=False,
                ),
                yaxis=dict(
                    title="Y [km] (TEME)",
                    color="#8892B0",
                    gridcolor=COLOR_GRID,
                    showbackground=False,
                ),
                zaxis=dict(
                    title="Z [km] (TEME)",
                    color="#8892B0",
                    gridcolor=COLOR_GRID,
                    showbackground=False,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
        )

        return fig
