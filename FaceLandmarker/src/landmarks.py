"""
MediaPipe FaceLandmarker v2 – landmark index constants.
478 landmarks total (468 face mesh + 10 iris).
Chỉ số miệng giống hệt FaceMesh cũ.
"""

# ── Lip contour (outer) ───────────────────────────────────────────────────────
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# ── Lip contour (inner) ───────────────────────────────────────────────────────
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

ALL_LIP_INDICES = list(
    set(UPPER_LIP_OUTER + LOWER_LIP_OUTER + UPPER_LIP_INNER + LOWER_LIP_INNER)
)

# ── Key points cho Affine Transform ──────────────────────────────────────────
MOUTH_LEFT_CORNER  = 61    # mép trái
MOUTH_RIGHT_CORNER = 291   # mép phải
MOUTH_TOP_CENTER   = 0     # đỉnh môi trên (philtrum)
MOUTH_BOT_CENTER   = 17    # đáy môi dưới

AFFINE_SRC_INDICES = [MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER, MOUTH_TOP_CENTER]
