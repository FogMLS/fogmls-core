import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MAX_JOBS, JOB_ATTRIBUTES, SLOT_LEVEL_FEATURES = 200, 6, 11
data = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1)
X = data[:, :MAX_JOBS * JOB_ATTRIBUTES].reshape(-1, MAX_JOBS, JOB_ATTRIBUTES)
y0 = MAX_JOBS * JOB_ATTRIBUTES + SLOT_LEVEL_FEATURES
Y = data[:, y0:y0 + MAX_JOBS]

rx, ry, rs = [], [], []
for s in range(X.shape[0]):
    for j in range(MAX_JOBS):
        t, lab = X[s, j], Y[s, j]
        if t.sum() == 0 and lab == 0:
            continue
        rx.append(np.append(t, j)); ry.append(lab); rs.append(s)
rx, ry, rs = np.array(rx), np.array(ry, dtype=int), np.array(rs)

split = int(0.8 * len(rx))
rf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
rf.fit(rx[:split], ry[:split])
pred = rf.predict(rx[split:])
yt, st = ry[split:], rs[split:]
shared = set(rs[:split]) & set(st)
mask = ~np.isin(st, list(shared))
print("rows total", len(rx), "train", split, "test", len(yt))
print("shared slots", sorted(shared))
print("full test accuracy %.4f" % (100 * accuracy_score(yt, pred)))
print("rows without shared slot", int(mask.sum()))
print("no-shared-slot accuracy %.4f" % (100 * accuracy_score(yt[mask], pred[mask])))
