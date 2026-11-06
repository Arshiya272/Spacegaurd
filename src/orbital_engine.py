"""SpaceGuard AI - Orbital Mechanics Engine.

Implements high-fidelity satellite and space debris orbit propagation using SGP4,
astrodynamic coordinate transformations (TEME, ECEF, Geodetic), and state vector
derivation according to standard astrodynamic conventions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Astrodynamic Constants (WGS-84 / SGP4 standard)
MU_EARTH = 398600.4418  # Earth gravitational parameter [km^3 / s^2]
EARTH_RADIUS_KM = 6378.137  # Earth equatorial radius [km]
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS-84 Earth flattening
OMEGA_EARTH = 7.292115e-5  # Earth rotation rate [rad / s]


@dataclass
class StateVector:
    """Satellite state vector in TEME or geocentric reference frame."""
    epoch: datetime
    position_km: np.ndarray  # [x, y, z] in km
    velocity_km_s: np.ndarray  # [vx, vy, vz] in km/s
    altitude_km: float
    speed_km_s: float
    error_code: int = 0
    is_valid: bool = True
    object_id: Optional[str] = None
    object_name: Optional[str] = None


@dataclass
class OrbitalElements:
    """Classical Keplerian orbital elements and derived geometric parameters."""
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_day: float
    orbital_period_minutes: float
    perigee_altitude_km: float
    apogee_altitude_km: float
    bstar: float
    orbit_regime: str  # 'LEO', 'MEO', 'GEO', 'HEO'


class OrbitalEngine:
    """SGP4-based orbital mechanics propagator and coordinate transformation engine."""

    def __init__(self):
        # Lazy imports for SGP4 and Skyfield to ensure clean initialization
        try:
            from sgp4.api import Satrec, WGS72
            from sgp4.api import jday as sgp4_jday
            self._Satrec = Satrec
            self._WGS72 = WGS72
            self._sgp4_jday = sgp4_jday
            self._sgp4_available = True
        except ImportError:
            logger.warning("sgp4 library not installed. Falling back to analytical two-body propagator.")
            self._sgp4_available = False

    @staticmethod
    def datetime_to_jd(dt: datetime) -> Tuple[float, float]:
        """Convert a UTC datetime to Julian Date split into (jd, fr)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second + dt.microsecond * 1e-6

        # Standard Julian Date computation
        if month <= 2:
            year -= 1
            month += 12

        a = math.floor(year / 100.0)
        b = 2 - a + math.floor(a / 4.0)
        jd_day = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5
        fraction = (hour + (minute + second / 60.0) / 60.0) / 24.0

        return jd_day, fraction

    @staticmethod
    def gmst_from_jd(jd: float, fr: float) -> float:
        """Compute Greenwich Mean Sidereal Time (GMST) in radians.

        Standard IAU 1982 formula.
        """
        jd_total = jd + fr
        t_u = (jd_total - 2451545.0) / 36525.0
        gmst_sec = (
            24110.54841
            + 8640184.812866 * t_u
            + 0.093104 * (t_u ** 2)
            - 6.2e-6 * (t_u ** 3)
            + 86400.0 * fr
        )
        gmst_rad = (gmst_sec * 2.0 * math.pi / 86400.0) % (2.0 * math.pi)
        return gmst_rad

    def omm_to_satrec(self, omm_record: Dict[str, Any]):
        """Build an SGP4 Satrec instance from an OMM record or TLE lines."""
        if not self._sgp4_available:
            return None

        # Check for pre-parsed TLE lines
        l1 = omm_record.get("TLE_LINE1")
        l2 = omm_record.get("TLE_LINE2")
        if l1 and l2:
            return self._Satrec.twoline2rv(l1, l2)

        # Synthesize TLE lines from OMM fields
        l1, l2 = self.omm_to_tle_lines(omm_record)
        return self._Satrec.twoline2rv(l1, l2)

    @staticmethod
    def omm_to_tle_lines(omm: Dict[str, Any]) -> Tuple[str, str]:
        """Convert standardized OMM dictionary into valid NORAD 2-line format."""
        norad_id = int(omm.get("NORAD_CAT_ID", omm.get("OBJECT_ID", 99999)))
        norad_id_5d = f"{norad_id % 100000:05d}"
        
        # Parse ISO epoch
        epoch_str = str(omm.get("EPOCH", "2026-01-01T00:00:00"))
        try:
            if "T" in epoch_str:
                dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(epoch_str[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        yy = dt.year % 100
        start_of_year = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        day_of_year = (dt - start_of_year).total_seconds() / 86400.0 + 1.0

        epoch_tle = f"{yy:02d}{day_of_year:012.8f}"

        # BSTAR formatting
        bstar = float(omm.get("BSTAR", 0.0))
        if bstar == 0.0:
            bstar_str = " 00000-0"
        else:
            exp = int(math.floor(math.log10(abs(bstar)))) + 1 if bstar != 0 else 0
            mant = int(round((bstar / (10 ** exp)) * 100000))
            sign = "-" if bstar < 0 else " "
            exp_sign = "+" if exp >= 0 else "-"
            bstar_str = f"{sign}{abs(mant):05d}{exp_sign}{abs(exp)}"

        # Line 1 assemble
        l1_partial = f"1 {norad_id_5d}U 98067A   {epoch_tle}  .00001000  00000-0 {bstar_str} 0  999"
        # Compute checksum for line 1
        csum1 = OrbitalEngine.tle_checksum(l1_partial)
        line1 = f"{l1_partial}{csum1}"

        # Line 2 assemble
        inc = float(omm.get("INCLINATION", 0.0))
        raan = float(omm.get("RA_OF_ASC_NODE", 0.0))
        ecc = float(omm.get("ECCENTRICITY", 0.0))
        ecc_str = f"{int(round(ecc * 1e7)):07d}"
        arg_p = float(omm.get("ARG_OF_PERICENTER", 0.0))
        mean_m = float(omm.get("MEAN_ANOMALY", 0.0))
        mm = float(omm.get("MEAN_MOTION", 15.0))

        l2_partial = f"2 {norad_id_5d} {inc:8.4f} {raan:8.4f} {ecc_str} {arg_p:8.4f} {mean_m:8.4f} {mm:11.8f}00001"
        csum2 = OrbitalEngine.tle_checksum(l2_partial)
        line2 = f"{l2_partial}{csum2}"

        return line1, line2

    @staticmethod
    def tle_checksum(line: str) -> int:
        """Compute standard NORAD TLE checksum modulo 10."""
        s = 0
        for char in line:
            if char.isdigit():
                s += int(char)
            elif char == "-":
                s += 1
        return s % 10

    def extract_orbital_elements(self, omm_record: Dict[str, Any]) -> OrbitalElements:
        """Derive classical Keplerian elements, altitudes, and regime classifications."""
        mm = float(omm_record.get("MEAN_MOTION", 15.0))
        ecc = float(omm_record.get("ECCENTRICITY", 0.0))
        inc = float(omm_record.get("INCLINATION", 0.0))
        raan = float(omm_record.get("RA_OF_ASC_NODE", 0.0))
        arg_p = float(omm_record.get("ARG_OF_PERICENTER", 0.0))
        mean_anom = float(omm_record.get("MEAN_ANOMALY", 0.0))
        bstar = float(omm_record.get("BSTAR", 0.0))

        # Mean motion in radians per second
        n_rad_s = mm * (2.0 * math.pi / 86400.0)
        
        # Semi-major axis from Kepler's Third Law: a = (mu / n^2)^(1/3)
        if n_rad_s > 0:
            semi_major_axis = (MU_EARTH / (n_rad_s ** 2)) ** (1.0 / 3.0)
        else:
            semi_major_axis = EARTH_RADIUS_KM + 400.0

        period_minutes = (2.0 * math.pi / n_rad_s) / 60.0 if n_rad_s > 0 else 90.0

        # Perigee & Apogee
        perigee_r = semi_major_axis * (1.0 - ecc)
        apogee_r = semi_major_axis * (1.0 + ecc)
        perigee_alt = perigee_r - EARTH_RADIUS_KM
        apogee_alt = apogee_r - EARTH_RADIUS_KM

        # Orbit regime classification
        if perigee_alt < 2000.0:
            regime = "LEO"
        elif 35000.0 <= perigee_alt <= 36500.0 and ecc < 0.05:
            regime = "GEO"
        elif perigee_alt >= 2000.0 and ecc < 0.2:
            regime = "MEO"
        else:
            regime = "HEO"

        return OrbitalElements(
            semi_major_axis_km=semi_major_axis,
            eccentricity=ecc,
            inclination_deg=inc,
            raan_deg=raan,
            arg_perigee_deg=arg_p,
            mean_anomaly_deg=mean_anom,
            mean_motion_rev_day=mm,
            orbital_period_minutes=period_minutes,
            perigee_altitude_km=perigee_alt,
            apogee_altitude_km=apogee_alt,
            bstar=bstar,
            orbit_regime=regime,
        )

    def propagate(
        self,
        satrec: Any,
        target_dt: datetime,
        omm_fallback: Optional[Dict[str, Any]] = None,
    ) -> StateVector:
        """Propagate satellite state to a target UTC timestamp using SGP4.

        Returns StateVector with position, velocity, and altitude.
        """
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = target_dt.astimezone(timezone.utc)

        jd, fr = self.datetime_to_jd(target_dt)

        if satrec is not None and self._sgp4_available:
            e, r, v = satrec.sgp4(jd, fr)
            if e == 0:
                pos = np.array(r, dtype=float)
                vel = np.array(v, dtype=float)
                r_mag = float(np.linalg.norm(pos))
                v_mag = float(np.linalg.norm(vel))
                alt = r_mag - EARTH_RADIUS_KM

                return StateVector(
                    epoch=target_dt,
                    position_km=pos,
                    velocity_km_s=vel,
                    altitude_km=alt,
                    speed_km_s=v_mag,
                    error_code=0,
                    is_valid=True,
                )
            else:
                logger.debug("SGP4 propagation returned error code %d for %s", e, target_dt)
                # If decayed or out of range, fallback gracefully to Keplerian approximation
                if omm_fallback:
                    return self._propagate_keplerian(omm_fallback, target_dt, error_code=e)
                return StateVector(
                    epoch=target_dt,
                    position_km=np.zeros(3),
                    velocity_km_s=np.zeros(3),
                    altitude_km=0.0,
                    speed_km_s=0.0,
                    error_code=e,
                    is_valid=False,
                )

        # Fallback to analytical Keplerian propagator
        if omm_fallback:
            return self._propagate_keplerian(omm_fallback, target_dt, error_code=0)

        raise RuntimeError("No satrec or omm_fallback provided for propagation.")

    def _propagate_keplerian(
        self, omm: Dict[str, Any], target_dt: datetime, error_code: int = 0
    ) -> StateVector:
        """Unperturbed two-body analytical propagation fallback."""
        elems = self.extract_orbital_elements(omm)
        a = elems.semi_major_axis_km
        e = elems.eccentricity
        inc = math.radians(elems.inclination_deg)
        raan = math.radians(elems.raan_deg)
        arg_p = math.radians(elems.arg_perigee_deg)
        n = elems.mean_motion_rev_day * (2.0 * math.pi / 86400.0)

        # Time elapsed since epoch
        epoch_str = str(omm.get("EPOCH", "2026-01-01T00:00:00"))
        try:
            epoch_dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
        except Exception:
            epoch_dt = target_dt

        delta_t = (target_dt - epoch_dt).total_seconds()
        mean_anom = (math.radians(elems.mean_anomaly_deg) + n * delta_t) % (2.0 * math.pi)

        # Solve Kepler's equation for Eccentric Anomaly E: M = E - e*sin(E)
        ecc_anom = mean_anom
        for _ in range(10):
            f_val = ecc_anom - e * math.sin(ecc_anom) - mean_anom
            f_prime = 1.0 - e * math.cos(ecc_anom)
            delta = f_val / f_prime
            ecc_anom -= delta
            if abs(delta) < 1e-8:
                break

        # True anomaly nu
        sin_nu = (math.sqrt(max(0.0, 1.0 - e**2)) * math.sin(ecc_anom)) / (1.0 - e * math.cos(ecc_anom))
        cos_nu = (math.cos(ecc_anom) - e) / (1.0 - e * math.cos(ecc_anom))
        nu = math.atan2(sin_nu, cos_nu)

        # Orbital radius
        r = a * (1.0 - e**2) / (1.0 + e * math.cos(nu))

        # Position in orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)

        # Velocity in orbital plane
        p = a * (1.0 - e**2)
        h = math.sqrt(MU_EARTH * p) if p > 0 else 0.0
        vx_orb = - (MU_EARTH / h) * math.sin(nu) if h > 0 else 0.0
        vy_orb = (MU_EARTH / h) * (e + math.cos(nu)) if h > 0 else 0.0

        # Rotate to inertial frame via 3-1-3 Euler angles (RAAN, inc, arg_p)
        cos_O = math.cos(raan)
        sin_O = math.sin(raan)
        cos_i = math.cos(inc)
        sin_i = math.sin(inc)
        cos_w = math.cos(arg_p)
        sin_w = math.sin(arg_p)

        # Direction cosine matrix elements
        P11 = cos_O * cos_w - sin_O * sin_w * cos_i
        P12 = -cos_O * sin_w - sin_O * cos_w * cos_i
        P21 = sin_O * cos_w + cos_O * sin_w * cos_i
        P22 = -sin_O * sin_w + cos_O * cos_w * cos_i
        P31 = sin_w * sin_i
        P32 = cos_w * sin_i

        pos = np.array([
            P11 * x_orb + P12 * y_orb,
            P21 * x_orb + P22 * y_orb,
            P31 * x_orb + P32 * y_orb,
        ], dtype=float)

        vel = np.array([
            P11 * vx_orb + P12 * vy_orb,
            P21 * vx_orb + P22 * vy_orb,
            P31 * vx_orb + P32 * vy_orb,
        ], dtype=float)

        r_mag = float(np.linalg.norm(pos))
        v_mag = float(np.linalg.norm(vel))

        return StateVector(
            epoch=target_dt,
            position_km=pos,
            velocity_km_s=vel,
            altitude_km=r_mag - EARTH_RADIUS_KM,
            speed_km_s=v_mag,
            error_code=error_code,
            is_valid=(r_mag > 1000.0),
        )

    def generate_orbit_path(
        self,
        satrec: Any,
        start_dt: datetime,
        period_minutes: float = 95.0,
        num_points: int = 120,
        omm_fallback: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Sample 3D trajectory positions across a full orbital revolution.

        Returns array of shape (N, 3) in km.
        """
        import datetime as dt_module
        points = []
        step_seconds = (period_minutes * 60.0) / max(10, num_points)

        for i in range(num_points + 1):
            t = start_dt + dt_module.timedelta(seconds=i * step_seconds)
            state = self.propagate(satrec, t, omm_fallback=omm_fallback)
            if state.is_valid:
                points.append(state.position_km)

        if not points:
            return np.empty((0, 3))
        return np.array(points)
