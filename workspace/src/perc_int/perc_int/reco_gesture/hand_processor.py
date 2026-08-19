import numpy as np
from .geom_utils import normalize
from .finger_detector import finger_fold_score
from .thumb_detector import thumb_direction_compass
from .orientation_detector import compute_palm_normal, hand_orientation
from .temporal_filter import TemporalFilter
from .config import Config

import shared
from shared.enums.enum_vehicule_action import RIGHT, LEFT, STOP, START, NO_HAND

class HandProcessor:
    def __init__(self, config=Config):
        self.cfg = config
        self.smooth_landmarks = None
        self.ema_alpha = config.EMA_ALPHA
        self.temporal = TemporalFilter(window=config.MAJORITY_WINDOW, debounce=config.DEBOUNCE_FRAMES)

    def _ema(self, prev, cur):
        if prev is None:
            return cur
        prev = np.array(prev, dtype=float)
        cur = np.array(cur, dtype=float)
        return (self.ema_alpha * cur + (1 - self.ema_alpha) * prev).tolist()

    def normalize_landmarks(self, landmarks):
        pts = [tuple(p) for p in landmarks]
        wrist = np.array(pts[0], dtype=float)
        middle_mcp = np.array(pts[9], dtype=float)
        hand_scale = np.linalg.norm(middle_mcp - wrist)
        if hand_scale == 0:
            hand_scale = 1.0
        normed = [tuple(((np.array(p) - wrist) / hand_scale).tolist()) for p in pts]
        return normed

    def compute_variable_scores(self, landmarks):
        idx = [landmarks[i] for i in (5, 6, 7, 8)]
        mid = [landmarks[i] for i in (9, 10, 11, 12)]
        ring = [landmarks[i] for i in (13, 14, 15, 16)]
        pinky = [landmarks[i] for i in (17, 18, 19, 20)]
        wrist = landmarks[0]

        s_idx = finger_fold_score(idx, wrist)
        s_mid = finger_fold_score(mid, wrist)
        s_ring = finger_fold_score(ring, wrist)
        s_pinky = finger_fold_score(pinky, wrist)

        hand_closing = "AMBIG"
        
        fist_score = float((s_idx + s_mid + s_ring + s_pinky) / 4.0)
        open_score = 1-fist_score
        var_fist = np.var([s_idx, s_mid, s_ring, s_pinky])
        if var_fist < 0.02:
            if fist_score > self.cfg.T_FIST:
                hand_closing = "FIST"
            elif open_score > self.cfg.T_OPEN:
                hand_closing = "OPEN"

        y, x, palm_normal = compute_palm_normal(landmarks)
        orient, ang_z, dir, ang_y, prof, ang_x = hand_orientation(y, x, palm_normal)

        thumb_dir_label, thumb_conf = thumb_direction_compass(landmarks)

        return {
            'fist_score': fist_score,
            'open_score': open_score,
            'hand_closing': hand_closing,
            'palm_normal': palm_normal,
            'orientation': orient,
            'orientation_ang_z': ang_z,
            'direction': dir,
            'orientation_ang_y': ang_y,
            'profile': prof,
            'orientation_ang_x': ang_x,
            'thumb_dir': thumb_dir_label,
            'thumb_conf': thumb_conf,
            'folds': {'idx': s_idx, 'mid': s_mid, 'ring': s_ring, 'pinky': s_pinky}
        }

    def decide_gesture(self, var):
        if var['hand_closing'] == 'OPEN' and var['direction'] == 'VERTICAL' and var['orientation'] == 'VERTICAL':
            return STOP, var['open_score']
        if var['hand_closing'] == 'FIST' and var['thumb_conf'] > self.cfg.THUMB_AXIS_CONFIDENCE:
            if var['direction']=='HORIZONTAL' :
                if var['orientation']=='VERTICAL' and var['profile']=='PROFILE' and var['thumb_dir'] == 'UP':
                    return START, min(var['fist_score'], var['thumb_conf'])
            elif var['direction']=='VERTICAL' :
                if var['orientation']=='VERTICAL' and var['profile']=='FRONTAL' and var['thumb_dir']=='UP' :
                    return START, min(var['fist_score'], var['thumb_conf'])
                elif (var['orientation']=='VERTICAL' and var['profile']=='PROFILE') or (var['orientation']=='HORIZONTAL' and var['profile']=='FRONTAL') :
                    if var['thumb_dir'] == 'RIGHT':
                        return RIGHT, min(var['fist_score'], var['thumb_conf'])
                    elif var['thumb_dir'] == 'LEFT':
                        return LEFT, min(var['fist_score'], var['thumb_conf'])
        return NO_HAND, 0.0

    def update(self, landmarks_raw, handedness=None):
        if landmarks_raw is None:
            return NO_HAND, 0.0, {}
        if self.smooth_landmarks is None:
            self.smooth_landmarks = landmarks_raw
        else:
            self.smooth_landmarks = self._ema(self.smooth_landmarks, landmarks_raw)

        normed = self.normalize_landmarks(self.smooth_landmarks)
        var = self.compute_variable_scores(normed)
        label, conf = self.decide_gesture(var)
        
        final_label, final_conf = self.temporal.add(label, conf)
        meta = {'vars': var, 'handedness': handedness}
        return final_label if final_label is not None else NO_HAND, final_conf, meta