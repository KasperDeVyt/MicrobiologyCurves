import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Input your full data ---
time = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
od = np.array([0.092, 0.104, 0.199, 0.379, 0.632, 1.1782, 1.6, 1.8728])

# --- MANUAL: define the time window for regression ---
manual_start_time = 0.5
manual_end_time = 2.5


# Find indices for the selected time segment
start_idx = np.where(time == manual_start_time)[0][0]
end_idx = np.where(time == manual_end_time)[0][0] + 1  # +1 for exclusive slicing

# Prepare data for regression
X = time[start_idx:end_idx].reshape(-1, 1)
y = np.log(od[start_idx:end_idx])  # natural log

# Fit the model
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)
slope = model.coef_[0]
intercept = model.intercept_

# Generation time using ln-based formula
generation_time = np.log(2) / slope * 60  # in minutes

# --- Print output ---
print(f"Selected segment: time {manual_start_time}–{manual_end_time} h")
print(f"  R²        = {r2:.4f}")
print(f"  Slope     = {slope:.4f} (per hour in ln(OD))")
print(f"  Intercept = {intercept:.4f}")
print(f"  Generation time = {generation_time:.2f} min")

# --- Plotting ---
plt.figure(figsize=(6, 4))
plt.plot(time, od, 'o-', label='OD600')
plt.yscale('log')

# Plot regression line
t_fit = np.linspace(manual_start_time, manual_end_time, 100).reshape(-1, 1)
od_fit = np.exp(model.predict(t_fit))
plt.plot(t_fit, od_fit, 'r--', label=f'Regression (R²={r2:.3f})')

# Labels and layout
plt.xlabel('Tijd (uren)')
plt.ylabel('OD600')
plt.title('Exponentiële voorstelling groei ' + r'$\it{E.\ coli}$')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()
