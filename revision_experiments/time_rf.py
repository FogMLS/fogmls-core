import sys, time, joblib
sys.path.insert(0, '..')
from random_forest_baseline import parse_and_reshape
R = './FX10_BASE/'
X, Y = parse_and_reshape(R + 'GA_dataset_seed42.txt')
Xt = X[int(0.8 * len(X)):]
rf = joblib.load(R + 'random_forest_model.joblib')
T = 27
rf.predict(Xt[:T])
for rep in range(5):
    ts = []
    for i in range(1000):
        b = Xt[i * T:(i + 1) * T]
        t0 = time.perf_counter()
        rf.predict(b)
        ts.append(time.perf_counter() - t0)
    print('rep %d: avg inference per slot (27 tasks): %.6f s' % (rep, sum(ts) / len(ts)), flush=True)
