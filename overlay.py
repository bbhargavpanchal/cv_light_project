import cv2
import numpy as np

NIGHT_TINT_COLOR = (60, 30, 10)  # BGR
NIGHT_TINT_STRENGTH = 0.35


def composite_glow(frame_bgr, glow_layer, intensity_mask):
    frame_f = frame_bgr.astype(np.float32)
    blended = frame_f + glow_layer
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_night_tint(frame_bgr, strength=NIGHT_TINT_STRENGTH):
    frame_f = frame_bgr.astype(np.float32)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    desaturated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    frame_f = frame_f * (1 - strength * 0.5) + desaturated * (strength * 0.5)

    tint = np.array(NIGHT_TINT_COLOR, dtype=np.float32)
    frame_f = frame_f * (1 - strength) + (frame_f + tint) * strength

    return np.clip(frame_f, 0, 255).astype(np.uint8)


def draw_debug(frame_bgr, fps, light, pinching_near_light, proximity_radius=90, hand_mask=None):
    out = frame_bgr
    if hand_mask is not None:
        mask_u8 = (hand_mask * 255).astype(np.uint8)
        colored = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.85, colored, 0.15, 0)

    cx, cy = int(light.position[0]), int(light.position[1])
    cv2.circle(out, (cx, cy), light.radius, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), proximity_radius, (0, 255, 255), 1, cv2.LINE_AA)

    state = "SUN" if light.is_sun else "MOON"
    color = (0, 255, 255) if pinching_near_light else (255, 255, 255)
    cv2.putText(out, f"FPS: {fps:.1f}  STATE: {state}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if pinching_near_light:
        cv2.putText(out, "TOGGLE ZONE", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return out


def render_frame(frame_bgr, light, glow_layer, intensity_mask, night_active,
                  pinching_near_light=False, proximity_radius=90, debug=False,
                  fps=0.0, hand_mask=None):
    out = frame_bgr
    if night_active:
        out = apply_night_tint(out)
    out = composite_glow(out, glow_layer, intensity_mask)
    if debug:
        out = draw_debug(out, fps, light, pinching_near_light,
                          proximity_radius=proximity_radius, hand_mask=hand_mask)
    return out
