"""v17n-4 experiment 8: first-observed-session month momentum in ZN."""
import numpy as np

FAMILY = "treasury-month-first-momentum"
PARAMS = {"minimum_move_pts": 0.046875, "stop_pts": 0.25}
T0, T1, EXIT = 30600, 32340, 54540  # 08:30, completed 08:59, evaluator flat minute


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in ("ZN", "ZF"):
            c, o, sec = df.close.to_numpy(), df.open.to_numpy(), df.ct_sec.to_numpy()
            previous_month = None
            for label, idx in df.groupby("session", sort=False).indices.items():
                month = str(label)[:7]
                trade = month != previous_month
                previous_month = month
                if not trade:
                    continue
                s = sec[idx]
                a, b = np.flatnonzero(s == T0), np.flatnonzero(s == T1)
                if not len(a) or not len(b):
                    continue
                move = c[idx[b[0]]] - o[idx[a[0]]]
                if abs(move) < PARAMS["minimum_move_pts"]:
                    continue
                side, ref = int(np.sign(move)), c[idx[b[0]]]
                for j in range(b[0], len(idx)):
                    if s[j] >= EXIT or side * (c[idx[j]] - ref) <= -PARAMS["stop_pts"]:
                        break
                    out[idx[j]] = side
        result[symbol] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 290.0) for s, df in panel.items()}
