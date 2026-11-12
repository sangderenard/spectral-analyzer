# Disjoint forward origins and sensors for timed Studio exposure

The timed Spectral worker accepts `sensor_lanes`: 1–32 resolved analytical
cameras in stable order, with the main camera at index zero. Each lane owns its
own native `SourceRec12` origin population, forward field textures and buffers,
and `SensorAccumulator` output. Additional lanes may select the existing
`fixed:1` monochrome or `fixed:3` RGB transport basis independently. The existing forward and sensor shaders are
unchanged. The worker packs one captured lighting slice into every lane,
dispatches every forward population, waits for transport materialization,
then dispatches every sensor. The shared `ExposureBarrier` closes reception
and commits the slice only after all lanes finish.

The original native emitter sampler supplies each lane's complete spectral
energy from a distinct seed. The worker rejects any duplicate forward
position/direction/wavelength identity before exposure begins. Sensor strip
seeds are packed by pass, lane index and strip index; lane zero with only one
lane retains the previous seed sequence. A field or sensor failure prevents
the slice commit. No ray-arrival count drives exposure time.

The main image and HDR retain the existing names and routes. Other lanes are
stored in `lane-XX/` under the same private photo job and advertised in
`status.sensor_lanes`. The authenticated image/HDR routes accept a bounded
`lane` number and serve only the file currently advertised for that lane.

`camera_visible` controls whether the authored physical camera housing/iris/
sensor triangles enter the forward and sensor BVHs. The analytical thin-lens
camera remains active either way. Studio's Phong preview currently has no
native camera render mesh; its housing endpoint is mount-only. The browser
control currently governs exposure geometry, not a Phong camera drawing.

The Studio camera-pattern demo uses the existing `particle_styles.icosphere(0)`
faces: twenty normalized face centers at the main camera's radius, all aimed
at the same target. The normal Studio light-state recorder supplies one
timeline; all twenty sensors expose each slice in one photo job. The gallery
selector reads each lane through the authenticated artifact route.

The built-in calibration sweep uses the same path: for each declared light
case, RGB and monochrome sensors read one state slice in one job. The earlier
two-job sequence has been removed. The peanut currently has a resolved lens
pose but no authored peanut shell or clear/frosted dome.

Source verification only: focused CPU tests and syntax checks passed. No new
GPU driver link, dispatch, exposure or picture was run because the field
information prerequisite in `turing/AGENT_OBSTINANCE.md` remains open.
