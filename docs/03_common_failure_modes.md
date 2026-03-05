# Common Failure Modes

## Camera shake or vibration
- Symptom: speed spikes across all lanes.
- Fix: stabilize the camera, tighten ROI, increase lane point threshold.

## Night glare / headlights
- Symptom: noisy tracking points and false speed updates.
- Fix: reduce `feature.max_corners`, increase `feature.quality_level`, recalibrate ROI away from glare areas.

## Heavy occlusion
- Symptom: unstable lane assignment near overlaps.
- Fix: tighten lane polygons and use shorter track length for crowded scenes.

## Perspective mismatch
- Symptom: far lanes under/over-estimate speed.
- Fix: calibrate `px_to_meter` per lane, not globally.

## Low frame rate input
- Symptom: jittery speed output.
- Fix: verify source FPS and use reliable encoded footage.

## Wrong polygons after camera change
- Symptom: points appear but speed is wrong lane or zero.
- Fix: rerun ROI calibration and verify all polygons in `config/camera.yaml`.

## Wrong-looking mask despite correct clicks
- Symptom: rendered mask appears shifted or widened compared to ROI clicks.
- Fix: recalibrate using the current ROI tool and the same input video; run with `--config config/camera.yaml`.
