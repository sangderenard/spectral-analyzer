"""Persistent rolling sensor view on the existing Spectral scene and tracer."""
import json
import time


def _ray_batch_size(live):
    # Total primary forward rays per sweep, not rays per source or the
    # renderer's smaller dispatch chunk. Bound queued work for interactive
    # use; this does not guarantee a frame rate for expensive scenes.
    value = live.get("ray_batch_size", 8192)
    if type(value) is not int or not 128 <= value <= 65536:
        raise ValueError("live ray_batch_size must be an integer between 128 and 65536")
    return value


def _scheduler_control(control, generation, state):
    """Server-authored cooperative scheduling, independent of camera edits."""
    if not control or 'scheduler_generation' not in control:
        return generation, state
    next_generation=control['scheduler_generation']
    next_state=control.get('scheduler_state')
    if type(next_generation) is not int or next_generation<0 or next_state not in ('paused','running'):
        raise ValueError('invalid live scheduler command')
    if next_generation<=generation:return generation,state
    return next_generation,next_state


def run_live_sensor(accumulator, renderer, field, output, digest, config,
                    preparation, snapshot, atomic_json, *, max_epochs=None,
                    update_camera_geometry=None, dynamic_scene_reader=None):
    live = config["live"]
    from camera_designer.studio_forward_field import ForwardFieldPublisher, validate_receive_light_field
    field_publisher = ForwardFieldPublisher(output, digest) if validate_receive_light_field(config) else None
    field_sweep = 0
    ray_batch_size = _ray_batch_size(live)
    from OpenGL.GL import glFinish
    accumulator.set_history(live["history_samples"], checkpoint=live.get("checkpoint", False))
    accumulator._aspect = live["aspect"]
    camera = config.get("camera")
    started = time.monotonic()
    last_control = last_snapshot = started - 1
    epoch = 0
    rays = sensor_sites = 0
    revision = 0
    pending_control = None
    scheduler_generation=-1
    scheduler_state='running'
    retained_pause=False
    pause_ack=None
    expires = time.time() + live["lease_seconds"]
    status = {"digest": digest, "state": "preparing", "image": None}
    cooperative_pump=None
    if config.get('client_intersections',False):
        from camera_designer.studio_cooperative_sensor import CooperativeSensorPump
        def cooperative_status(state,profile):
            atomic_json(output/'status.json',{**status,'state':'live','client_work_state':state,'client_work_profile':profile})
        cooperative_pump=CooperativeSensorPump(renderer,digest,on_state=cooperative_status)
    path = output / "live-control.json"
    while True:
        now = time.monotonic()
        if now - last_control >= .05:
            last_control = now
            try:
                control = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                control = None
            if control:
                scheduler_generation,scheduler_state=_scheduler_control(control,scheduler_generation,scheduler_state)
                expires = float(control["expires_at"])
                if control["revision"] > max(revision, pending_control["revision"] if pending_control else -1):
                    next_camera, next_live = control["camera"], control["live"]
                    if dynamic_scene_reader is not None and next_camera != camera:
                        raise ValueError("dynamic subject reader requires a new candidate for camera changes")
                    next_ray_batch_size = _ray_batch_size(next_live)
                    reset = (next_camera != camera or any(next_live.get(key) != live.get(key)
                        for key in ("history_samples", "aspect", "checkpoint"))
                        or next_ray_batch_size != ray_batch_size)
                    if reset:
                        # Finish the current sweep using its original field.
                        # The newest complete control replaces any queued edit.
                        pending_control = control
                    else:
                        # Tint/normalization is display-only. It updates the
                        # next frame without discarding linear history. A newer
                        # edit returning to active settings cancels a queued reset.
                        camera, live = next_camera, next_live
                        revision = control["revision"]
                        if field_publisher is not None:
                            field_publisher.invalidate()
                            status.pop('light_field', None)
                        pending_control = None
        if ((time.time() >= expires and cooperative_pump is None and not retained_pause and scheduler_state!='paused')
                or (max_epochs is not None and epoch >= max_epochs)):
            glFinish()
            status = {**status, "state": "canceled", "reason": "live client lease expired" if time.time() >= expires else "test epoch limit"}
            atomic_json(output / "status.json", status)
            if cooperative_pump is not None:cooperative_pump.close()
            return status
        if accumulator._row_off==0 and (scheduler_state=='paused' or ((retained_pause or cooperative_pump is not None) and time.time()>=expires)):
            # Finish already-submitted GL work before the scheduler may admit
            # another dispatching job. This releases compute ownership, NOT VRAM.
            retained_pause=True
            ack=(scheduler_generation,time.time()>=expires)
            if pause_ack!=ack:
                glFinish()
                status={**status,'state':'yielded','scheduler_generation':scheduler_generation,
                        'scheduler_state':scheduler_state,'gpu_dispatch_yielded':True,
                        'gpu_memory_released':False,'client_lease_expired':ack[1],
                        'retained_pass':accumulator._pass,'retained_row':accumulator._row_off,
                        'retained_history_bytes':accumulator._history_bytes,
                        'forward_ray_jobs':rays,'sensor_sites':sensor_sites,
                        'live_revision':revision}
                atomic_json(output/'status.json',status);pause_ack=ack
            time.sleep(.05)
            continue
        if retained_pause:
            # Same accumulator, field, seed and ring. Pending authored camera
            # changes still use their existing explicit sweep-boundary reset.
            retained_pause=False;pause_ack=None
        # One fresh, fixed-budget forward field per complete sensor sweep.
        # Every pixel in that sweep sees the same field. Retaining a finite
        # camera ring over an ever-growing field would leave old photons in
        # new samples indefinitely. Here no forward energy predates its
        # sweep: only the resulting N sensor contributions persist. Check
        # boundaries per strip, including non-square/partial-last-row films.
        for _ in range(4):
            if scheduler_state=='paused' and accumulator._row_off==0:
                break
            if accumulator._row_off == 0:
                if pending_control is not None:
                    camera, live = pending_control["camera"], pending_control["live"]
                    ray_batch_size = _ray_batch_size(live)
                    # The physical housing and both tracer BVHs must accept
                    # the same revision before any ray uses the new view.
                    if update_camera_geometry is not None:
                        update_camera_geometry(camera, accumulator)
                    accumulator.set_camera(camera["eye_m"], camera["target_m"],
                                           fov_deg=camera["fov_degrees"])
                    accumulator.set_lens(aperture_radius=camera["aperture_radius_m"],
                                         focus_dist=camera["focus_distance_m"],
                                         n_blades=camera.get("aperture_blades", 0),
                                         aperture_rot=camera.get("aperture_rotation_radians", 0))
                    accumulator.set_history(live["history_samples"], checkpoint=live.get("checkpoint", False))
                    accumulator._aspect = live["aspect"]
                    # Forward energy is additive, not normalized by batch
                    # size. Never average the old field estimate into the new
                    # ring/checkpoint, even when its capacity is unchanged.
                    accumulator.clear()
                    revision = pending_control["revision"]
                    if field_publisher is not None:
                        field_publisher.invalidate()
                        status.pop('light_field', None)
                    pending_control = None
                if dynamic_scene_reader is not None:
                    # The existing complete-sweep boundary owns both readers.
                    # Failed installation raises before another forward ray.
                    acknowledgment = dynamic_scene_reader.commit_boundary(accumulator)
                    if acknowledgment is not None:
                        if field_publisher is not None:
                            field_publisher.invalidate()
                            status.pop('light_field', None)
                        atomic_json(output / "scene-ack.json", acknowledgment)
                seed = field["_pump_seed"]
                renderer._clear_prebuilt(field)
                field["_pump_seed"] = seed
                renderer._gpu_pump_prebuilt(field, n_rays=ray_batch_size)
                field_sweep += 1
                rays += ray_batch_size
            completed_sites=min(accumulator._rows_per_frame,
                                accumulator._h-accumulator._row_off)*accumulator._w
            if cooperative_pump is not None:
                # Same retained history dispatch; no forward-field/camera edits while on loan.
                cooperative_pump.revision=f'{digest}:live:{revision}:pass:{accumulator._pass}'
                cooperative_pump.step(accumulator)
            else:accumulator.tick(strips=1)
            sensor_sites += completed_sites
        epoch += 1
        if epoch % 8 == 0:
            glFinish()
        # The snapshot's sensor alpha distinguishes unvisited pixels from
        # sampled black. Publish partial sweeps so new coverage can grow over
        # the UI; clear() prevents pixels from the old camera surviving reset.
        if now - last_snapshot >= .25:
            status = snapshot(accumulator, output, digest, epoch % 900000, now-started,
                              preparation, rays, sensor_sites,
                              transport="fixed:3", gamma=config.get("gamma",1),
                              film_response=config.get("film_response"),
                              display_params=live.get("display"), publish_status=False)
            status.update(state="live", epoch=epoch, live_revision=revision,
                          device_limits=getattr(accumulator,"_device_limits",None),
                          scheduler_generation=scheduler_generation,scheduler_state=scheduler_state,
                          gpu_dispatch_yielded=False,gpu_memory_released=False,
                          camera_settings=dict(camera),
                          camera_focus_semantics=config.get("camera_focus_semantics"),
                          history_samples=live["history_samples"],
                          history_filled=min(accumulator._pass,live["history_samples"]),
                          history_bytes=accumulator._history_bytes,
                          history_kind="checkpoint plus finite ring" if live.get("checkpoint") else "finite per-pixel sensor ray contributions",
                          checkpoint_enabled=live.get("checkpoint", False),
                          checkpoint_samples=max(0, accumulator._pass-live["history_samples"]) if live.get("checkpoint") else 0,
                          total_samples=accumulator._pass if live.get("checkpoint") else min(accumulator._pass, live["history_samples"]),
                          history_block_limit=getattr(accumulator, "_history_block_limit", None),
                          ray_batch_size=ray_batch_size,
                          forward_history=f"fresh {ray_batch_size}-ray field per sensor sweep",
                          width=accumulator._w,height=accumulator._h)
            if cooperative_pump is not None:status.update(client_work_profile=dict(cooperative_pump.profile),client_work_state='committed')
            if dynamic_scene_reader is not None and dynamic_scene_reader.ack is not None:
                status["scene_ack"] = dict(dynamic_scene_reader.ack)
            if field_publisher is not None:
                exported = field_publisher.snapshot(renderer, field, live_revision=revision,
                    sweep=field_sweep, sensor_samples=accumulator._pass,
                    ray_jobs=ray_batch_size if field.get('n_sources') else 0,
                    accumulation='fresh fixed-budget additive field per sensor sweep; no forward temporal accumulation',
                    scene_ack=status.get('scene_ack'))
                if exported is not None:
                    status['light_field'] = exported
            atomic_json(output / "status.json", status)
            last_snapshot = time.monotonic()
