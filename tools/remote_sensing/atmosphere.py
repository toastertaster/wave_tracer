"""Computes the atmosphere of a site fetched by fetch_site.py, with 6S.

Writes SITE_DIR/atmosphere.json: per wavelength, the atmosphere's
transmittances, spherical albedo and intrinsic (path) reflectance, for the
sun's position at --time and the camera's view direction, from 6S (Vermote et
al.; via Py6S) with the aerosol optical depth and water vapour that AERONET
measured that day (aeronet.csv), a continental aerosol model, and the
target's elevation. build_scene.py lights scenes with them (the sun through
the atmosphere and a sky dome) and has the camera see the ground through
them; toa.py adds the atmosphere's own radiance.

Quantities (dimensionless, per wavelength; μ = cos zenith):
  T_down_direct, T_up_direct   direct transmittance by scattering, exp(-τ/μ),
                               sun to ground, ground to camera
  T_down_total, T_up_total     the same plus the diffuse (scattered) part
  T_up_diffuse_rayleigh,       the diffuse upward transmittance of molecules
  T_up_diffuse_aerosol         alone and of aerosols alone (which weight their
                               environment functions; see toa.py)
  Tg_down, Tg_up, Tg_total     gaseous transmittance (absorption): sun to
                               ground, ground to camera, and both ways. In
                               absorption bands Tg_total is not
                               Tg_down Tg_up: 6S averages over its 2.5 nm
                               steps, and uses Tg_total for light that comes
                               down and goes back up.
  spherical_albedo             of the atmosphere, seen from the ground
  rho_atm                      intrinsic reflectance (with gaseous
                               absorption): the path radiance is
                               rho_atm E0 μs / π, for TOA irradiance E0

The apparent reflectance at the top of the atmosphere of a uniform Lambertian
ground of reflectance ρ is then, as in 6S,
  rho_atm + Tg_total T_down_total T_up_total ρ / (1 - ρ spherical_albedo).

Needs the 6S executable (conda-forge `sixs`) and Py6S, e.g. in a conda
environment:
    micromamba create -n rs -c conda-forge sixs python=3.12 numpy
    micromamba run -n rs pip install Py6S python-dateutil
    micromamba run -n rs python tools/remote_sensing/atmosphere.py SITE_DIR [--sixs PATH]
"""

import argparse
import csv
import datetime
import json
import math
import pathlib
import shutil
import sys

import numpy as np
from Py6S import AeroProfile, AtmosProfile, Geometry, GroundReflectance, SixS, Wavelength

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sun import solar_position  # noqa: E402


def aeronet_inputs(path):
    """AOD at 550 nm (from 500 nm and the 440-870 nm Ångström exponent) and precipitable water (cm)."""
    with open(path) as f:
        row = next(csv.DictReader(f))
    aod500 = float(row["AOD_500nm"])
    alpha = float(row["440-870_Angstrom_Exponent"])
    return aod500 * (550 / 500) ** -alpha, alpha, float(row["Precipitable_Water(cm)"]), aod500


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("site_dir", type=pathlib.Path)
    p.add_argument("--time", type=datetime.datetime.fromisoformat,
                   default=datetime.datetime(2022, 7, 19, 18, 19, 19, tzinfo=datetime.timezone.utc))
    p.add_argument("--view-zenith", type=float, default=0, help="degrees")
    p.add_argument("--view-azimuth", type=float, default=0, help="degrees clockwise from north, towards the camera")
    p.add_argument("--min-wavelength", type=float, default=400, help="nm (default 400)")
    p.add_argument("--max-wavelength", type=float, default=1000, help="nm (default 1000)")
    p.add_argument("--step", type=float, default=2.5, help="nm (default 2.5, 6S's resolution)")
    p.add_argument("--ozone", type=float, default=0.30, help="cm-atm (default .30)")
    p.add_argument("--aot550", type=float, help="instead of AERONET's")
    p.add_argument("--water", type=float, help="precipitable water, cm, instead of AERONET's")
    p.add_argument("--sixs", default=shutil.which("sixs"), help="the 6S executable (default: `sixs` on PATH)")
    args = p.parse_args()

    site = json.loads((args.site_dir / "site.json").read_text())
    lat, lon = site["centre"]["lat"], site["centre"]["lon"]
    elevation_km = site["centre"]["elevation_m"] / 1e3

    aot550, alpha, water, aod500 = (None,) * 4
    if site.get("aeronet"):
        aot550, alpha, water, aod500 = aeronet_inputs(args.site_dir / site["aeronet"]["file"])
    aot550 = args.aot550 if args.aot550 is not None else aot550
    water = args.water if args.water is not None else water
    if aot550 is None or water is None:
        sys.exit("no AERONET data: give --aot550 and --water")

    sun_zenith, sun_azimuth = solar_position(args.time, lat, lon)
    print(f"sun: zenith {sun_zenith:.3f}°, azimuth {sun_azimuth:.3f}°; AOT550 {aot550:.4f}, water {water:.3f} cm, "
          f"ozone {args.ozone} cm-atm, ground {elevation_km:.3f} km")

    s = SixS(args.sixs)
    s.geometry = Geometry.User()
    s.geometry.solar_z, s.geometry.solar_a = sun_zenith, sun_azimuth
    s.geometry.view_z, s.geometry.view_a = args.view_zenith, args.view_azimuth
    s.geometry.month, s.geometry.day = args.time.month, args.time.day
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water, args.ozone)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aot550
    s.altitudes.set_target_custom_altitude(elevation_km)
    s.altitudes.set_sensor_satellite_level()
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)

    mus, muv = math.cos(math.radians(sun_zenith)), math.cos(math.radians(args.view_zenith))
    wavelengths = np.arange(args.min_wavelength, args.max_wavelength + args.step / 2, args.step)
    keys = ["T_down_direct", "T_up_direct", "T_down_total", "T_up_total",
            "T_up_diffuse_rayleigh", "T_up_diffuse_aerosol", "Tg_down", "Tg_up", "Tg_total",
            "spherical_albedo", "rho_atm", "E0_6S"]
    out = {k: [] for k in keys}
    for i, wl in enumerate(wavelengths):
        s.wavelength = Wavelength(wl / 1e3)
        s.run()
        o = s.outputs
        tau = o.optical_depth_total.total
        out["T_down_direct"].append(math.exp(-tau / mus))
        out["T_up_direct"].append(math.exp(-tau / muv))
        out["T_down_total"].append(o.transmittance_total_scattering.downward)
        out["T_up_total"].append(o.transmittance_total_scattering.upward)
        depth = o.optical_depth_total
        out["T_up_diffuse_rayleigh"].append(o.transmittance_rayleigh_scattering.upward - math.exp(-depth.rayleigh / muv))
        out["T_up_diffuse_aerosol"].append(o.transmittance_aerosol_scattering.upward - math.exp(-depth.aerosol / muv))
        out["Tg_down"].append(o.transmittance_global_gas.downward)
        out["Tg_up"].append(o.transmittance_global_gas.upward)
        out["Tg_total"].append(o.transmittance_global_gas.total)
        out["spherical_albedo"].append(o.spherical_albedo.total)
        # the apparent reflectance of a black ground: 6S prints the intrinsic reflectance to only 3 decimals
        out["rho_atm"].append(o.values["apparent_reflectance"])
        out["E0_6S"].append(o.values["solar_spectrum"])
        if i % 40 == 0:
            print(f"  {wl:g} nm: T↓ {out['T_down_direct'][-1]:.4f}+{out['T_down_total'][-1]-out['T_down_direct'][-1]:.4f}, "
                  f"Tg {out['Tg_total'][-1]:.4f}, S {out['spherical_albedo'][-1]:.4f}, ρ_atm {out['rho_atm'][-1]:.4f}")

    atmosphere = {
        "source": "6S via Py6S; continental aerosol model",
        "time": args.time.isoformat(),
        "sun": {"zenith_deg": sun_zenith, "azimuth_deg": sun_azimuth},
        "view": {"zenith_deg": args.view_zenith, "azimuth_deg": args.view_azimuth},
        "inputs": {"aot550": aot550, "aod500": aod500, "angstrom_440_870": alpha, "water_cm": water,
                   "ozone_cm_atm": args.ozone, "ground_km": elevation_km},
        "wavelength_nm": wavelengths.tolist(),
        **out,
    }
    (args.site_dir / "atmosphere.json").write_text(json.dumps(atmosphere, indent=1) + "\n")
    print(f"wrote {args.site_dir}/atmosphere.json ({len(wavelengths)} wavelengths)")


if __name__ == "__main__":
    main()
