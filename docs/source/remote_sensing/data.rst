Data sources
=======================================

Data for the remote sensing scenes and for validating them, surveyed when the branch started.
None of it is committed to this repository: scenes fetch it (see the plan's milestone 1), and each source's terms are noted below.


Reference site: Railroad Valley Playa, Nevada
---------------------------------------------

A dry lakebed at 38.50° N, 115.69° W, about 1.4 km above sea level, used by CEOS to calibrate Landsat, Sentinel-2, MODIS and VIIRS.
It is about 15 × 15 km, vegetation-free, spectrally flat apart from the blue and a shortwave infrared absorption, with reflectance above 0.3, nearly Lambertian out to 30° view angles, and a mean aerosol optical depth of 0.06 at 550 nm.
Being uniform and well measured, it is a good first check of absolute radiometry; roads, oil-drilling structures and the valley's edges add spatial detail.

**Reference date: 19 July 2022.** NAIP imaged the site on 19–20 July 2022; Sentinel-2B passed over it at 18:19 UTC on 19 July (tile T11SPC, 5.5 % cloud over the tile); and AERONET's Level 2.0 data for that day give an aerosol optical depth of 0.066 at 500 nm, an Ångström exponent (440–870 nm) of 1.33, and 1.89 cm of precipitable water.

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Data
     - What
     - Terms
     - Access
   * - USGS 3DEP 1 m DEM
     - Lidar bare-earth elevation, 1 m, UTM zone 11 N (NAD83), NAVD88 heights. Project ``NV_NyeNE_D23`` (published December 2024); four 10 × 10 km tiles (``x60y426``, ``x60y427``, ``x61y426``, ``x61y427``, about 250 MB each) cover the playa.
     - Public domain
     - `TNM Access API <https://tnmaccess.nationalmap.gov/api/v1/products>`_, dataset "Digital Elevation Model (DEM) 1 meter"
   * - NAIP
     - Aerial orthophotos, 0.6 m, four 8-bit bands (red, green, blue, near infrared), 2022 (also 2019; 1 m in 2010–2017). Not radiometrically calibrated: use it for spatial detail and for classifying materials, not as reflectance.
     - Public domain
     - Microsoft Planetary Computer STAC, collection ``naip`` (e.g. ``nv_m_3811527_se_11_060_20220719``)
   * - RadCalNet (site RVUS)
     - Top-of-atmosphere and surface reflectance, 400–2500 nm at 10 nm, every 30 minutes from 09:00 to 15:00 local time, with the aerosol optical depth, Ångström exponent, pressure, water vapour and ozone used to compute them (with MODTRAN).
     - Free; registration
     - `radcalnet.org <https://www.radcalnet.org>`_ (account from admin-radcalnet@magellium.fr)
   * - AERONET (site ``Railroad_Valley``)
     - Aerosol optical depth at 340–1640 nm, Ångström exponents, precipitable water: inputs for the atmosphere tables.
     - Free; contact the site PIs before publishing
     - `AERONET web service <https://aeronet.gsfc.nasa.gov/print_web_data_help_v3.html>`_
   * - Sentinel-2 L1C/L2A
     - The same scene from orbit at 10–60 m, 13 bands: top-of-atmosphere reflectance (L1C) and surface reflectance (L2A), for comparing renders. Spectral response functions: ESA's S2-SRF (version 5.0, 2026), also per detector.
     - Free and open (Copernicus)
     - Planetary Computer collections ``sentinel-2-l1c``/``sentinel-2-l2a``; SentiWiki document library


Sun, materials and atmosphere
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Data
     - What
     - Terms
     - Access
   * - TSIS-1 HSRS
     - The top-of-atmosphere solar spectral irradiance (Coddington et al. 2021) in W/m²/nm, 202–2730 nm: 0.1 nm sampling at 0.1 nm or 1 nm resolution. A binned extension covers longer wavelengths.
     - Free; cite
     - LASP LISIRD, e.g. ``https://lasp.colorado.edu/lisird/latis/dap/tsis1_hsrs_1nm.csv``
   * - USGS Spectral Library 7 (splib07)
     - Laboratory, field and airborne reflectance spectra of minerals, soils, vegetation, coatings and man-made materials, 0.2–200 µm, in ASCII. ``materials.py`` uses its spectra convolved to an ASD spectrometer (``ASCIIdata_splib07b_cvASD.zip``: 350–2500 nm at 1 nm), e.g. Stonewall Playa's dry mud as an analogue of Railroad Valley Playa's.
     - CC0
     - `USGS data release <https://doi.org/10.5066/F7RR1WDJ>`_ (ScienceBase item ``586e8c88e4b0f5ce109fccae``)
   * - ECOSTRESS Spectral Library 1.0
     - About 3400 spectra (vegetation, soils, rocks, man-made materials, water), visible to thermal infrared. Directional-hemispherical reflectances give emissivities by Kirchhoff's law, for the thermal infrared.
     - © Caltech, all rights reserved; reproduced with acknowledgement. Fetch, don't redistribute.
     - `speclib.jpl.nasa.gov <https://speclib.jpl.nasa.gov>`_
   * - libRadtran 2.0.6
     - Radiative transfer (DISORT, MYSTIC, polarized, thermal; REPTRAN absorption): computes the atmosphere tables offline.
     - GPL: run as a separate program, never linked
     - `libradtran.org <https://www.libradtran.org>`_
   * - 6S / Py6S
     - A radiative transfer code for the solar spectrum: computes the atmosphere (``atmosphere.py``). The 6S executable is on conda-forge (``sixs``).
     - Py6S: LGPL
     - ``pip install Py6S`` (needs the 6S executable)


Telescope point-spread functions
--------------------------------

No vendor prescriptions are needed to start: the open-source optical design packages below compute field-dependent point-spread functions of published designs.

* `Optiland <https://github.com/optiland/optiland>`_ (MIT): sequential ray tracing with conic, aspheric and freeform mirrors, wavefront, PSF and MTF analysis. Its gallery includes a Cassegrain telescope (from Laikin, *Lens Design*) and a three-mirror anastigmat.
* `RayOptics <https://github.com/mjhoptics/ray-optics>`_ (BSD-3): imports Zemax ``.zmx`` and CODE V ``.seq`` files, should prescriptions become available.
* Published designs to scale: Korsch's three-mirror anastigmat (Applied Optics 16, 2074, 1977: radii, conic constants and spacings are in the paper's table), and an off-axis three-mirror anastigmat from BISME close to our example (1.75 m focal length, f/9, 7 µm TDI pixels, 2 m GSD from 500 km; Fan et al., ICSO 2017).
