import numpy as np
from sklearn.linear_model import LinearRegression


def find_best_exponential_segment(time, od, min_points=3):
    """
    Finds the contiguous segment of (time, od) with at least `min_points`
    that yields the highest R^2 when regressing ln(od) ~ time.

    Returns a dict with:
      - 'start', 'end': indices of the best segment (end is exclusive)
      - 'r2': the R^2 score
      - 'slope', 'intercept': parameters of the fit in log-space
      - 'model': the fitted LinearRegression instance
    """
    log_od = np.log(od)
    n = len(time)
    best = {'r2': -np.inf}

    for window in range(min_points, n + 1):
        for i in range(n - window + 1):
            j = i + window
            X = time[i:j].reshape(-1, 1)
            y = log_od[i:j]

            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            if r2 > best['r2']:
                best = {
                    'start': i,
                    'end': j,
                    'r2': r2,
                    'slope': model.coef_[0],
                    'intercept': model.intercept_,
                    'model': model
                }
    return best



time = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
od = np.array([0.092, 0.104, 0.199, 0.379, 0.632, 1.1782, 1.6, 1.8728])

best = find_best_exponential_segment(time, od, min_points=3)

i, j = best['start'], best['end']
print(f"Best exponential segment is points {i} to {j - 1} (time {time[i]:.1f}–{time[j - 1]:.1f}h)")
print(f"  R²     = {best['r2']:.4f}")
print(f"  slope  = {best['slope']:.4f} (per hour in ln(OD))")
print(f"  intercept = {best['intercept']:.4f}")

# If you want to plot it:
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(time, od, 'o-', label='OD600')
plt.yscale('log')
# overlay best-fit line:
t_fit = np.linspace(time[i], time[j - 1], 100).reshape(-1, 1)
od_fit = np.exp(best['model'].predict(t_fit))
plt.plot(t_fit, od_fit, 'r--', label=f'Best fit (R²={best["r2"]:.3f})')
plt.xlabel('Time (h)')
plt.ylabel('OD600')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()
