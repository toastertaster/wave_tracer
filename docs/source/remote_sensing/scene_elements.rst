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
