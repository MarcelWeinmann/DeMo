"""Speed cap code from the orchestration's speed target (planning_req v_target, m/s), shared by the actual-cap datasets
(data/curation/full_perturbation_speed_target/planning_req/make_dataset_actual.py) and the deep prediction node
(deep_prediction_lib_cpp/deep_prediction_cpp/python/deep_predictor.py), so both map the target the same way.

Codes (the value in the parquet column speed_cap_kph and in x_attr[:, 0]; ModelForecast(use_speed_cap=True) turns
them into one-hot bits):
  60 / 80 / 100 / 130  the target lies in the level's band [LO, level + ABOVE] (m/s). Logged targets cluster just below
                       each level (54-60, 77-80, 97-100.8, 127.8-130 km/h); 132 km/h (a test speed), 108 km/h, the pit
                       limits and 90 m/s ("no limit") are no level
  SPEED_CAP_STOP_CODE  a 0 km/h target (stop), since 0 means "no cap"
  0                    no cap, or unknown (NaN)
numpy only (no torch), so the dataset tools can import it."""
import numpy as np

SPEED_CAP_STOP_CODE = 255
SPEED_CAP_BAND_LO_MPS = {60: 14.3, 80: 20.3, 100: 25.8, 130: 34.5}
SPEED_CAP_BAND_ABOVE_MPS = 0.3
SPEED_CAP_STOP_BELOW_MPS = 0.5


def speed_cap_code(v_target_mps):
    """cap code(s) for speed target(s) in m/s (scalar or array; NaN = unknown -> 0), as int16"""
    v = np.asarray(v_target_mps, dtype=np.float64)
    out = np.zeros(v.shape, np.int16)
    for level, lo in SPEED_CAP_BAND_LO_MPS.items():
        out[(v >= lo) & (v <= level / 3.6 + SPEED_CAP_BAND_ABOVE_MPS)] = level
    out[(v >= 0) & (v < SPEED_CAP_STOP_BELOW_MPS)] = SPEED_CAP_STOP_CODE
    return out
