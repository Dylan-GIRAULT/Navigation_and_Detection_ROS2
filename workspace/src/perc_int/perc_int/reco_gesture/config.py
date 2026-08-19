class Config:
    # fold / open thresholds
    T_FOLD_SCORE = 0.7 # above -> finger considered folded
    T_EXTEND_SCORE = 0.3 # below -> finger considered extended

    T_FIST = 0.4 # mean of 4 fingers -> fist
    T_OPEN = 0.90 # open hand


    # palm orientation thresholds (degrees)
    ANGLE_VERTICAL_DEG = 35.0
    ANGLE_HORIZONTAL_DEG = 35.0

    # angle tolerance for vertical/horizontal classification
    INCLINSAISON_VERTICAL_ANGLE_TOLERANCE = 20.0
    INCLINSAISON_HORIZONTAL_ANGLE_TOLERANCE = 40.0
    ORIENTATION_ANGLE_TOLERANCE = 40.0

    # thumb classification
    THUMB_AXIS_CONFIDENCE = 0.5


    # smoothing
    EMA_ALPHA = 0.6
    DEBOUNCE_FRAMES = 4
    MAJORITY_WINDOW = 7