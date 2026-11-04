Scene elements
=======================================

Compiled only with ``REALISTIC_REMOTE_SENSING``.


``satellite_frame`` sensor
--------------------------

A satellite's frame (staring) camera: a pinhole camera of a given focal length and pixel pitch, placed by its altitude and the direction it is seen from a target on the ground.

Scenes use a local frame on the target's ground: **x east, y up, z south** (north is −z), in metres.
The camera is at ``altitude`` above a spherical Earth, seen from ``target`` at view zenith ``view_zenith`` and view azimuth ``view_azimuth`` (clockwise from north, towards the camera), at the slant range that geometry gives.
It looks at ``target``, with north up (east to the right) in its images at nadir.

.. code-block:: xml

   <sensor type="satellite_frame" id="camera">
       <quantity name="altitude" value="500km"/>
       <quantity name="view_zenith" value="30°"/>       <!-- default 0° -->
       <quantity name="view_azimuth" value="90°"/>      <!-- default 0° -->
       <quantity name="focal_length" value="2.5m"/>
       <quantity name="pixel_pitch" value="5um"/>       <!-- 1 m GSD at nadir, from 500 km -->
       <integer name="samples" value="16"/>
       <film type="array">
           <integer name="width" value="1920"/>
           <integer name="height" value="1920"/>
           <response type="band_radiance"> ... </response>
       </film>
   </sensor>

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``focal_length``, ``pixel_pitch``
     - Pixel *i* spans the angles of one pixel pitch on the focal plane, about the optical axis through the film's centre: the vertical field of view is 2 atan(height · pitch / (2 · focal length)).
       (A ``perspective`` sensor's ``fov`` spans one pixel less, off centre by half a pixel.)
   * - ``altitude``
     - Above the target's ground. Or, instead, ``to_world``: a transform placing the camera as a perspective sensor's.
   * - ``view_zenith``, ``view_azimuth``
     - The direction from the target to the camera. Default 0° (nadir).
   * - ``target``
     - Default the origin.
   * - ``earth_radius``
     - Default 6371.0088 km (mean).
   * - ``near_clip``
     - Distance from the camera at which its rays start. Placed by ``altitude``, rays start 2 km short of the target by default, so that ray queries are computed near the scene: from hundreds of kilometres away, single-precision rounding of positions (centimetres) otherwise makes ~15% of oblique views' samples shadow themselves. Geometry farther than 2 km from the target's ground along the line of sight is then not seen: give ``near_clip`` for taller scenes.
   * - ``samples``, ``ray_trace_only``, ``alpha``, ``phase_space_extent_scale``, ``polarimetric``
     - As for the ``perspective`` sensor.

The camera is a ``perspective`` sensor with an exact film-to-camera transform, so it renders on the GPUs too.
Its beams start at ``near_clip`` as if propagated there from the camera (their extent there is the camera's pixel footprint).


``telescope`` sensor
--------------------

A satellite's camera that images the scene through a telescope's mirrors: a focal-plane array behind optics described by their surfaces, placed as a ``satellite_frame`` camera.
Each sample takes a point in its pixel and a direction towards the optics, and traces the ray backwards through the mirrors, in double precision, out towards the scene.
Rays that miss a clear aperture, or meet an obstacle (the secondary mirror's shadow, the primary's central hole), are vignetted.
The render then has the telescope's geometric optics in it: its distortion, its field-dependent geometric blur (aberrations, the focal plane's defocus off a curved field), vignetting, and the parallax of an off-axis aperture; and its mirrors' coatings: their reflectance, per wavelength and angle of incidence, and the polarization they cause.
Diffraction by the aperture is not traced: it is the detector model's (``detector.py``).

The telescope's optics are configured, not built in: a list of elements, in the order light meets them.
``tools/remote_sensing/telescope.py`` writes them for Cassegrain and Ritchey-Chrétien designs (centred, off-axis or folded) and for the Hubble Space Telescope (:doc:`tools`).

.. code-block:: xml

   <sensor type="telescope" id="camera">
       <quantity name="altitude" value="500km"/>      <!-- placed as a satellite_frame camera -->
       <quantity name="view_zenith" value="30°"/>
       <quantity name="pixel_pitch" value="5um"/>
       <quantity name="focal_length" value="2.5m"/>   <!-- optional: a check of the optics' -->
       <integer name="samples" value="16"/>
       <mirror id="primary">
           <quantity name="radius" value="-1.2m"/>
           <float name="conic" value="-1.0368"/>
           <quantity name="aperture_radius" value="0.2m"/>
           <quantity name="hole_radius" value="0.0142m"/>
           <transform name="to_telescope"/>          <!-- at the origin -->
       </mirror>
       <mirror id="secondary">
           <quantity name="radius" value="-0.382m"/>
           <float name="conic" value="-3.008"/>
           <quantity name="aperture_radius" value="0.0496m"/>
           <transform name="to_telescope"><translate z="-0.4548m"/></transform>
       </mirror>
       <transform name="focal_plane">                 <!-- film to telescope frame -->
           <rotate y="1" angle="180°"/>
           <translate z="0.15m"/>
       </transform>
       <film type="array"> ... </film>
   </sensor>

**Frames.**
The telescope's frame follows optical design software's convention: light from the scene arrives travelling towards +z.
Each element has its own frame, its vertex at the origin and its axis along local +z, placed by its ``to_telescope`` transform (rotations and translations).
A surface's radius of curvature is positive when its centre of curvature is at local +z: prescriptions from Zemax, CODE V or Optiland carry over for surfaces whose frames are not flipped (a two-mirror telescope's mirrors both have negative radii).
Pixel (*i*, *j*) of the film spans [*i*, *i*\ +1) × [*j*, *j*\ +1) pixel pitches about the film's centre, at x and y of the ``focal_plane`` transform's frame (on its plane z = 0).
The image is the one the optics form: for the film's x to be east and its y south at nadir (as a ``satellite_frame`` camera's), a two-mirror telescope's focal plane is turned half a turn about y, as above; with an odd number of mirrors the image is mirrored.

**Placement.**
The telescope's frame is placed as a ``satellite_frame`` camera is (``altitude``, ``view_zenith``, ``view_azimuth``, ``target``, ``earth_radius``, ``near_clip``): its origin is the satellite's position, and at nadir its x is east, y north and z up.
The line of sight of the film's centre (the mean direction of its rays towards the scene) is pointed at the target, so that a telescope whose field is off its axis (an off-axis three-mirror anastigmat's) is pointed as its instrument would be.
An aperture off the telescope's origin sees the ground from there: an off-axis segment 0.2 m from the axis images the ground 0.2 m from where a centred one would.
Given ``to_world`` instead of ``altitude``, it places the telescope's frame itself.

.. list-table:: Elements
   :header-rows: 1
   :widths: 25 75

   * - Element
     - Description
   * - ``<mirror>``
     - Reflects, within its clear aperture. A ray must meet each mirror in turn.
   * - ``<stop>``
     - A flat opening: a ray must pass within its aperture, in turn (an aperture or field stop).
   * - ``<obscuration>``
     - A flat disc (an annulus, with ``hole_radius``) that blocks every ray crossing it, in no particular order: a baffle or a central obscuration.

Every mirror is also an obstacle to the rays' other segments, within its clear aperture and outside its hole: the secondary's body shadows the primary, and the converging beam must pass the primary's hole.

.. list-table:: Element parameters
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``radius``
     - The vertex radius of curvature (mirrors; omit it for a flat mirror). Lengths of the prescription are read in double precision, with a unit (m, cm, mm, um, nm, km).
   * - ``conic``
     - The conic constant κ (0 sphere, −1 paraboloid, < −1 hyperboloid). Default 0.
   * - ``a4``, ``a6``, ...
     - Even aspheric coefficients, of *r*\ ⁴, *r*\ ⁶, ..., in metres (a\ :sub:`2i` in m\ :sup:`1−2i`): the surface is z = c r² / (1 + √(1 − (1+κ) c² r²)) + a\ :sub:`4` r⁴ + a\ :sub:`6` r⁶ + ...
   * - ``aperture_radius``, ``hole_radius``
     - The clear aperture: within ``aperture_radius`` of its centre, and outside ``hole_radius`` (default 0).
   * - ``aperture_centre``
     - The clear aperture's centre in the element's x and y (default the vertex): an off-axis segment of its surface, e.g. ``<vector name="aperture_centre" value="0m, 0.2m"/>``.
   * - ``to_telescope``
     - The element's frame in the telescope's (required).
   * - ``<coating>``
     - The mirror's coating (below). Default ideal.

.. list-table:: Sensor parameters
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``pixel_pitch``
     - The film's pixels, on the focal plane.
   * - ``focal_plane``
     - The focal-plane array's frame in the telescope's (required).
   * - ``focal_length``
     - Optional: refused unless the optics' focal length (below) is within 1% of it.
   * - placement
     - As for the ``satellite_frame`` sensor.
   * - ``samples``, ``ray_trace_only``, ``alpha``, ``phase_space_extent_scale``, ``polarimetric``
     - As for the ``perspective`` sensor. Beams start at the last mirror, of the pixel's angular extent (pitch over focal length, or ``alpha``) and the narrowest cross section it allows (a minimum-uncertainty beam: ~0.2 m at 550 nm for 2 µrad), then start again at ``near_clip``.

**Coatings.**
Each mirror's coating reflects light as the complex Fresnel coefficients r\ :sub:`s` and r\ :sub:`p` of its materials give, at the sample's wavelength and the ray's angle of incidence:

.. code-block:: xml

   <coating type="ideal"/>                       <!-- the default: a perfect conductor, rs = -1, rp = 1 -->
   <coating type="multilayer">
       <spectrum name="substrate" material="Al"/> <!-- a metal (wave_tracer's data), or any refractive index -->
       <layer>                                  <!-- thin films, from the outermost; none: the bare substrate -->
           <spectrum name="IOR" constant="1.38"/>
           <quantity name="thickness" value="25nm"/>
       </layer>
   </coating>

Refractive indices are wave_tracer's spectra (complex, *n* + *ik*: ``material`` from its data, ``constant="(0.958,6.69i)"``, or any other).
A bare substrate reflects as wave_tracer's own ``fresnel_reflection``; layers by their characteristic matrices (Macleod, *Thin-Film Optical Filters*), in double precision, in the same convention.
The sensor describes each mirror's coating and its reflectance at normal incidence at the film's mean wavelength.

**Polarization.**
Each reflection applies its coating's Mueller matrix, composed in s-p frames as wave_tracer's surfaces compose theirs, from the detector's frame to the frame of the light arriving from the scene.
An unpolarized scene through metal mirrors at oblique incidence (a fold mirror's) renders partially polarized, along the mirror's s-polarization; mirrors whose planes of incidence are turned from each other turn it partly circular (retardance).
A polarimetric film (``polarimetric="true"``) records the Stokes parameters about the film's axes: Q is light polarized along the film's x, less along its y; U along x + y, less along x − y.
With a scalar film, pixels record the intensity, which the coatings' diattenuation makes depend on the scene's polarization.

**Radiometry.**
Pixels record radiance relative to the film centre's throughput (the projected solid angle, on the focal plane, of the rays that pass), times the coatings' reflectances.
With ideal mirrors, a uniform scene renders as its radiance at the film's centre, as a ``satellite_frame`` camera's does, and dimmer where the optics vignette or distort the image (by the conservation of étendue); coated mirrors dim it by their reflectance, e.g. two bare aluminium mirrors to ~0.85 at 550 nm.
The render is still the radiance at the entrance pupil, for ``detector.py`` to scale by the aperture: give its instrument the aperture's diameter and obscuration that the sensor reports, a wavefront error without the design's own aberrations, which the render has, and a transmittance without the coated mirrors' reflectances, which the render has too.

**When loading**, the sensor traces its optics from the film's centre (2\ :sup:`18` rays), near it and from its corners, and refuses designs through which no ray passes from the film's centre, or whose film centre does not look towards the scene.
It logs and describes (the scene's description, e.g. ``Scene::info`` in the Rust bindings) what it found: the focal lengths along the film's x and y (from the lines of sight 10 pixels either side of the centre: the image's centroid's scale, which a telescope's coma lengthens), the entrance pupil's diameter and obscuration, the film centre's projected solid angle and geometric blur (RMS, warning beyond 10 pixels), and the corners' throughput relative to the centre's.

**Sampling.**
The rays that the optics vignette are samples of no importance: they cost noise (1.5–3% per pixel at 256 samples, for the designs of the tests).
Directions are mostly drawn from the cone of those that pass (found on a 5 × 5 grid over the film when loading), and some from the cone bounding the clear aperture nearest the focal plane, which every passing direction is in: weighted by the mixture's density, the render is unbiased.

**Limitations.**
Only backward ``plt_path``: the sensor cannot connect to scene points, as ``plt_bdpt`` and forward ``plt_path`` need (scenes are refused).
On the GPU, the kernels trace the optics in their precision: in single precision, directions leaving the telescope are accurate to ~10⁻⁷ rad, a twentieth of a pixel for the tests' designs (f = 2.5 m, 5 µm pixels), but more than one for Hubble's (f = 57.6 m; the GPU warns when it is more than a tenth): render long focal lengths on the CPU, or with ``GPU_DBL_PRECISION``.
The GPU supports up to 32 elements, 8 aspheric terms per surface, and 16 coating layers and 16 refractive indices over all coatings.
Coatings are smooth: their roughness's scatter is stray light's (to come). The focal plane is flat, and the scene loader reads transforms in single precision (their rotations are orthonormalized again; translations are good to ~0.1 µm per metre).


``band_radiance`` response
--------------------------

A multispectral response whose pixels record radiance in physical units.
Each child spectrum is a band's spectral response function *S*; the pixel records

.. math::

   \int L_\lambda(\lambda)\, S(\lambda)\, d\lambda \qquad \text{or, with } \texttt{normalize}, \qquad
   \frac{\int L_\lambda\, S\, d\lambda}{\int S\, d\lambda},

the band radiance or the band-averaged spectral radiance.
Spectral quantities in the scene (emission spectra, spectral irradiances) are densities per wavelength in the units of ``spectral_unit``: e.g., a solar irradiance in W/m²/µm with ``spectral_unit`` µm gives band-averaged radiances in W/m²/sr/µm.

.. code-block:: xml

   <response type="band_radiance">
       <string name="spectral_unit" value="um"/>   <!-- nm (default) or um -->
       <boolean name="normalize" value="true"/>    <!-- default false -->
       <spectrum type="piecewise_linear" id="green">
           <bin wavelength="520nm" value="1"/>
           <bin wavelength="600nm" value="1"/>
       </spectrum>
       ...
   </response>

Films integrate samples over wavenumber (in mm⁻¹), so each band's response is weighted by the Jacobian *dλ/dk* = 2π/*k*².
Response functions may be given at any wavelengths: they are resampled finely (at most 0.5 nm apart, at least 256 steps per band), and fall linearly to zero within one step outside their range, so that a sum of disjoint bands is zero between them.
That ramp is part of the response (and of ∫ *S* dλ): a 20 nm box band is 0.4% wider for it.

All bands share each sample's wavelength, so with *n* disjoint bands a pixel's samples are shared among them: render about *n* times as many samples as for one band.
(A pixel's wavelengths are stratified over its samples, so each band gets its share.)

An optional ``transmittance`` spectrum multiplies what every band records, wavelength by wavelength, but not ∫ *S* dλ: e.g. the atmosphere's transmittance from the ground to the camera, so that a band records the band average of the transmitted radiance.

.. code-block:: xml

   <response type="band_radiance">
       ...
       <spectrum name="transmittance" type="piecewise_linear">
           <bin wavelength="440nm" value=".83"/>
           ...
       </spectrum>
   </response>

Channels are named after their bands' ``id`` attributes (e.g. in EXR files), if every band has a distinct one.


``spectral_mix`` texture
------------------------

A linear mixture of up to four spectra whose weights vary over the surface, for reflectances of mixed materials (remote sensing's linear mixture model): at texture coordinates *uv* and wavelength λ,

.. math::

   \sum_i w_i(uv)\, S_i(\lambda),

with the weights *w*\ :sub:`1`–*w*\ :sub:`4` the R, G, B and A channels of a ``weights`` bitmap texture (a grey bitmap gives its value to R, G and B; without alpha, A is 1).
The weights are the materials' abundances (fractions of the surface): they need not sum to one, and so also carry variations in brightness.

.. code-block:: xml

   <bsdf type="diffuse">
       <texture name="reflectance" type="spectral_mix">
           <texture name="weights" type="bitmap">
               <path value="materials.exr"/>              <!-- half or float RGBA -->
               <string name="filter_type" value="bilinear"/>
               <string name="wrap_mode" value="clamp"/>
           </texture>
           <spectrum type="piecewise_linear" id="playa"> ... </spectrum>       <!-- weighted by R -->
           <spectrum type="piecewise_linear" id="concrete"> ... </spectrum>    <!-- by G -->
           <spectrum type="piecewise_linear" id="asphalt"> ... </spectrum>     <!-- by B -->
       </texture>
   </bsdf>

The ``weights`` texture may be a bitmap or a transform of one.
Weights are linear: give PNG weights ``colour_encoding`` ``linear`` (EXR files are linear already).
Spectra beyond the first four are refused; channels without a spectrum are ignored.

On the GPUs, ``spectral_mix`` textures can only be diffuse reflectances (not, e.g., roughnesses or scale factors): the kernels look up the bitmap's RGBA and take its dot product with the spectra's values at the sample's wavelength.
``materials.py`` (:doc:`tools`) writes the weights and spectra of a site.
