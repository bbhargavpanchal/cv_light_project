# Eclipse — A Hand-Controlled Light, Driven by Your Webcam

A real-time computer vision project: a glowing light rendered on your live webcam feed that you grab, drag, eclipse, and resize — all with your bare hands, no controllers or markers.

## What it does

- **Grab and drag** — clench a fist near the light and it follows your hand; open your hand and it stays exactly where you dropped it.
- **Eclipse effect** — show the back of your hand over the light and it's pixel-accurately eclipsed; turn your hand to show your palm and nothing happens. Detected by hand orientation, not distance, so it holds up at any rotation.
- **Two hands, independently tracked** — each hand eclipses (or doesn't) on its own.
- **Day/night toggle** — a small switch anchored to your head position; pinch right next to it to flip the light from sun to moon, with the whole scene fading and dimming smoothly rather than cutting.
- **Depth-aware sizing** — move your hand toward the camera while holding the light and it grows; pull back and it shrinks, based on your hand's own apparent size on screen.
- **Real rendering, not a flat circle** — radial-gradient glow with bloom, smoothed motion, and a spring-physics "pop" on pickup.

## How it works

Built entirely on classical CV and landmark geometry — no model training, no dataset:

- **MediaPipe Tasks API** (`HandLandmarker`, `FaceLandmarker`) for 21-point hand and face landmark tracking on live video
- **OpenCV** for all rendering: radial gradients, convex-hull hand silhouettes for the eclipse mask, bloom via downsample → blur → upsample
- **NumPy** for the gesture maths — fist/pinch detection, and a cross-product-based classifier that tells palm-facing from back-of-hand using three landmarks, robust to in-plane rotation

## Stack

Python · OpenCV · MediaPipe · NumPy

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Press `q` to quit. A webcam is required. The hand and face landmark models (~12 MB total) download automatically to `models/` on first run.

## Controls

| Gesture | Effect |
|---|---|
| Clench fist near the light | Grab it |
| Move a clenched fist | Drag the light |
| Open your hand / pull away | Drop the light |
| Back of hand over the light | Eclipse it |
| Palm over the light | Nothing happens |
| Pinch near the head-anchored switch | Toggle day/night |
| Move your hand toward/away from the camera while holding it | Resize the light |

## Calibration

Distance-to-size mapping is camera- and setup-dependent. While you're dragging the light, an on-screen readout shows your hand's current apparent size against the reference value the light uses for "normal" size (`REFERENCE_HAND_SCALE` in `light_control.py`). If the light feels permanently oversized or undersized at the distance you naturally hold your hand, hold it there, read the two numbers off the overlay, and nudge `REFERENCE_HAND_SCALE` to match.


## Demo

## Phase 3 a demo:

https://github.com/user-attachments/assets/722251c1-e386-4f51-abe7-779b119deb19


## Phase 3 b:

https://github.com/user-attachments/assets/e122790b-0bac-4a6d-a329-280e4f7ea125



## Phase 4:

https://github.com/user-attachments/assets/a1151f94-ec7e-4866-8de9-aed6bdcafbdc


## Phase 5a:


https://github.com/user-attachments/assets/b866ffb0-9374-4bb5-ba82-eb6e3114964e


## Phase 5b:


https://github.com/user-attachments/assets/16561e1c-c1e9-4935-8ac5-0c7e6e870487


## Phase 6: Final 


https://github.com/user-attachments/assets/337bb2ef-8ac7-4683-82c9-85eb7a97d58a





