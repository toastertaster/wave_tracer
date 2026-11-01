"""The sun's position and distance, for tools/remote_sensing (no dependencies)."""

import datetime
import math


def solar_position(when: datetime.datetime, lat: float, lon: float):
    """Solar zenith and azimuth (clockwise from north), degrees: NOAA's solar position algorithm."""
    jd = when.timestamp() / 86400 + 2440587.5
    t = (jd - 2451545) / 36525
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    omega = 125.04 - 1934.136 * t
    app_lon = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obliq = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    obliq += 0.00256 * math.cos(math.radians(omega))
    decl = math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(app_lon)))
    y = math.tan(math.radians(obliq) / 2) ** 2
    l0r = math.radians(l0)
    eqtime = 4 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                              - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    minutes = when.hour * 60 + when.minute + when.second / 60
    hour_angle = math.radians((minutes + eqtime + 4 * lon) / 4 - 180)
    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(hour_angle)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.atan2(math.sin(hour_angle), math.cos(hour_angle) * math.sin(latr) - math.tan(decl) * math.cos(latr))
    return math.degrees(zen), (math.degrees(az) + 180) % 360


def earth_sun_distance(day: datetime.date) -> float:
    """Earth-Sun distance, AU (to ~1e-4)."""
    g = math.radians(357.529 + 0.98560028 * (day.toordinal() - datetime.date(2000, 1, 1).toordinal() - 0.5 + 1))
    return 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
