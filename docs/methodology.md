# SpaceGuard AI — Scientific Methodology & Astrodynamics Reference

This document details the physical models, astrodynamic coordinate transformations, SGP4 propagation limits, close-approach screening algorithms, and machine learning formulation implemented in **SpaceGuard AI**.

---

## 1. Orbital Mechanics & Data Representation

### 1.1 CCSDS Orbit Mean-Elements Message (OMM)
SpaceGuard AI prefers the standard **CCSDS Orbit Mean-Elements Message (OMM)** in JSON or CSV format published by **CelesTrak**. The primary parameters include:
- **Semi-Major Axis ($a$)**: Derived from mean motion $\bar{n}$ via Kepler's Third Law:
  $$a = \left(\frac{\mu}{\bar{n}^2}\right)^{1/3}$$
  where $\mu = 398600.4418\text{ km}^3/\text{s}^2$ is Earth's gravitational parameter (WGS-84).
- **Eccentricity ($e$)**: Orbital eccentricity defining ellipse shape ($0 \le e < 1$).
- **Inclination ($i$)**: Orbital plane inclination relative to Earth's equatorial plane ($0 \le i \le 180^\circ$).
- **Right Ascension of the Ascending Node ($\Omega$)**: Angle from vernal equinox to ascending node.
- **Argument of Perigee ($\omega$)**: Angle from ascending node to periapsis.
- **Mean Anomaly ($M$)**: Angular position of satellite along mean orbit.
- **BSTAR ($B^*$)**: Radiation pressure and atmospheric drag parameter ($1/\text{Earth radii}$).

### 1.2 Two-Line Element (TLE) Format
Legacy TLE sets are parsed into standard OMM fields and validated against checksums and physical boundary criteria ($0 \le e < 1$, $0 \le i \le 180^\circ$, $0 < \bar{n} \le 30\text{ rev/day}$).

---

## 2. SGP4 Analytical Propagation Engine

### 2.1 Gravitational & Drag Perturbations
Simplified General Perturbations 4 (SGP4) models:
1. Earth's geopotential harmonics up to $J_2, J_3, J_4$.
2. Secular and periodic atmospheric drag variations modeled through the $B^*$ parameter.
3. Lunar-solar gravitational third-body perturbations for high-altitude regimes.

### 2.2 Native Reference Coordinate Frame: TEME
SGP4 propagates state vectors in the **True Equator, Mean Equinox (TEME)** coordinate frame:
- Origin: Earth's center of mass (Geocentric).
- $\hat{Z}$: Parallel to the True Equator pole of date.
- $\hat{X}$: Directed toward the Mean Equinox of date.
- $\hat{Y}$: Completes the right-handed Cartesian triad ($\hat{Y} = \hat{Z} \times \hat{X}$).

### 2.3 Greenwich Mean Sidereal Time (GMST) Transformation
To transform state vectors from the inertial TEME frame to Earth-Centered Earth-Fixed (ECEF) coordinates for ground-track analysis, Greenwich Mean Sidereal Time ($\theta_{\text{GMST}}$) is calculated via the IAU-1982 standard:
$$T_u = \frac{\text{JD} - 2451545.0}{36525.0}$$
$$\theta_{\text{GMST}} = 280.46061837^\circ + 360.98564736629^\circ \cdot (\text{JD} - 2451545.0) + 0.000387933^\circ T_u^2 - 2.583 \times 10^{-8\circ} T_u^3$$
Rotation matrix:
$$\vec{r}_{\text{ECEF}} = \mathbf{R}_z(\theta_{\text{GMST}}) \, \vec{r}_{\text{TEME}}$$

---

## 3. Conjunction Screening & Close-Approach Mathematics

### 3.1 Spatial Altitude Pre-Filtering (O(1))
Before executing computationally intensive time-stepped propagation over long horizons, candidate object pairs are evaluated via their orbital altitude envelope:
$$r_{\text{perigee}} = a(1 - e) - R_E, \quad r_{\text{apogee}} = a(1 + e) - R_E$$
A pair $(A, B)$ is pruned if:
$$r_{\text{perigee}, A} > r_{\text{apogee}, B} + d_{\text{buffer}} \quad \text{or} \quad r_{\text{perigee}, B} > r_{\text{apogee}, A} + d_{\text{buffer}}$$
where $d_{\text{buffer}}$ is the screening spatial threshold (typically $100\text{ km}$).

### 3.2 Time of Closest Approach (TCA) Minimization
For non-pruned candidate pairs, relative separation is sampled over discrete time steps $\Delta t$:
$$\Delta \vec{r}(t) = \vec{r}_{\text{target}}(t) - \vec{r}_{\text{debris}}(t)$$
$$d(t) = \|\Delta \vec{r}(t)\|$$
Around the discrete local minimum, sub-second continuous TCA refinement is executed via **Golden-Section Search** until temporal tolerance $\delta t \le 0.5\text{ s}$:
$$t_{\text{TCA}} = \arg \min_{t} d(t)$$

### 3.3 Radial-InTrack-CrossTrack (RIC / RTN) Frame Decomposition
To understand the encounter geometry from the target satellite's operational perspective, relative position and velocity vectors at TCA are projected onto the orthonormal RIC coordinate triad:
- **Radial unit vector ($\hat{u}_R$)**:
  $$\hat{u}_R = \frac{\vec{r}_{\text{target}}}{\|\vec{r}_{\text{target}}\|}$$
- **Cross-Track unit vector ($\hat{u}_C$)**:
  $$\hat{u}_C = \frac{\vec{r}_{\text{target}} \times \vec{v}_{\text{target}}}{\|\vec{r}_{\text{target}} \times \vec{v}_{\text{target}}\|}$$
- **In-Track unit vector ($\hat{u}_I$)**:
  $$\hat{u}_I = \hat{u}_C \times \hat{u}_R$$

The relative state decomposition:
$$\Delta r_R = \Delta \vec{r} \cdot \hat{u}_R, \quad \Delta r_I = \Delta \vec{r} \cdot \hat{u}_I, \quad \Delta r_C = \Delta \vec{r} \cdot \hat{u}_C$$
$$\Delta v_R = \Delta \vec{v} \cdot \hat{u}_R, \quad \Delta v_I = \Delta \vec{v} \cdot \hat{u}_I, \quad \Delta v_C = \Delta \vec{v} \cdot \hat{u}_C$$

---

## 4. Machine Learning Risk Stratification vs. Collision Probability

### 4.1 Distinction Between Geometric Screening and Probability of Collision ($P_c$)
In aerospace operations, the true Probability of Collision ($P_c$) requires:
1. Full 3D positional covariance matrices ($\mathbf{C}_1, \mathbf{C}_2$) derived from radar observation filters (e.g. Unscented Kalman Filters).
2. Encounter plane projection and 2D Gaussian probability density integration over the combined hard-body collision cross-section (Foster-1992 or Akella-Alfriend methods).

Because public TLE/OMM feeds do not contain covariance matrices, calculating an empirical $P_c$ from TLE data alone is scientifically invalid.

### 4.2 Machine Learning Framing
SpaceGuard AI deploys machine learning strictly for **risk tier categorization and triage prioritization**:
- **Critical Tier**: Extremely tight miss distance ($< 10\text{ km}$), high encounter kinetic energy, small radial separation ($< 5\text{ km}$).
- **Warning Tier**: Moderate miss distance ($10\text{ km} \le d \le 50\text{ km}$), overlapping altitude envelopes.
- **Monitoring Tier / Low**: Broad separation ($> 50\text{ km}$).

### 4.3 Training Data Integrity & Segregation
- **Zero Fabrication**: Real satellite observations are never falsely labelled with invented collision outcomes.
- **Segregated Simulation**: The ML models are trained on a cleanly separated synthetic simulation dataset generated across realistic LEO/MEO orbital dynamics distributions ($N = 3,000$).
- **Benchmark Baseline**: Model predictions are continuously evaluated against a transparent deterministic physics baseline rule on held-out test data.
