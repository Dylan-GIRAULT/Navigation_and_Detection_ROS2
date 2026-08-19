# ===============================
# config/settings.py
# ===============================
FPS = 15
SMOOTHING_ALPHA = 0.2

INITIAL_SCORE = 1.0
PENALTY_PER_SEC = 0.10 # -0.01/s
STRONG_MULTIPLIER = 1.0 # x3
RECOVERY_PER_SEC = 0.10 # +0.02/s

# Head pose thresholds (degrees)
HEAD_YAW_FORWARD = 20 # degré max horizontal pour être considéré "regard avant"
HEAD_YAW_DISTRACT = 45 # degré max horizontal pour être considéré "distrait"
HEAD_PITCH_DOWN = 15 # degré max vertical pour être considéré "regard avant"

HEAD_YAW_FORWARD_SCORE = 0.8 # score si regard avant
HEAD_YAW_DISTRACT_SCORE = 0.6 # score si distrait

# mouth / yawn thresholds
MAR_YAWN_THRESH = 0.6 # seuil MAR pour détecter un baillement
MAR_YAWN_DUR_THRESH_SEC = 2 # durée minimale en secondes pour considérer un baillement

MAR_YAWN_SCORE = 0.7 # score si baillement détecté

# Eye / fatigue thresholds
EAR_THRESHOLD = 0.20
CLOSURE_DURATION_SEC = 2.0
PENALTY_VALUE = 0.4


EAR_BLINK_THRESH = 0.21
BLINK_CONSEC_FRAMES = 3
PERCLOS_WINDOW_SEC = 60
PERCLOS_DROWSY = 0.4

# Attention score weights
W_HEAD = 0.5
W_GAZE = 0.1
W_PHYSIO = 0.4