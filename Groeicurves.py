import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Full data
time = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
od = np.array([0.092, 0.104, 0.199, 0.379, 0.632, 1.1782, 1.6, 1.8728])

# Zoomed-in data
time_zoom = np.array([1.5, 2, 2.5, 3, 3.5])
od_zoom = np.array([0.379, 0.632, 1.1782, 1.6, 1.8728])

# Linear regression on raw OD
X_zoom = time_zoom.reshape(-1, 1)
model = LinearRegression().fit(X_zoom, od_zoom)
slope = model.coef_[0]
r2 = model.score(X_zoom, od_zoom)

# Fit line for plotting
time_fit = np.linspace(time_zoom.min(), time_zoom.max(), 100).reshape(-1, 1)
od_fit = model.predict(time_fit)

# Plot
plt.figure(figsize=(10, 6))

# Full plot
plt.subplot(2, 1, 1)
plt.plot(time, od, 'o-', label='OD600', color='green')
plt.title('Bacteriële groeicurve ' + r'$\it{E.\ coli}$')
plt.xlabel('Tijd (uren)')
plt.ylabel('OD600')
plt.grid(True)
plt.legend()

# Zoomed-in with linear regression
plt.subplot(2, 1, 2)
plt.plot(time_zoom, od_zoom, 'o-', color='blue', label='OD600 (Zoomed In)')
plt.plot(time_fit, od_fit, 'r--', label=f'Regressie (R²={r2:.4f})')
plt.title('Zoomed-in groeicurve (1.5 tot 3.5 uur)')
plt.xlabel('Tijd (uren)')
plt.ylabel('OD600')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print slope
print(f"Slope of linear regression (zoomed-in segment): {slope:.4f}")
