import numpy as np
import matplotlib.pyplot as plt

# --- New measured dataset ---
time = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
od = np.array([0.023, 0.025, 0.034, 0.091, 0.237, 0.522, 0.726, 1.386, 1.836])

# --- Simulated stationary phase (flat) ---
stationary_time = np.linspace(4, 5, 20)
stationary_od = np.full_like(stationary_time, od[-1])

# --- Simulated death phase (decay) ---
death_time = np.linspace(5, 6.5, 30)
death_od = od[-1] * np.exp(-0.4 * (death_time - 5))

# --- Plotting ---
plt.figure(figsize=(8, 5))

# Plot original data
plt.plot(time, od, 'o-', label='Gemeten OD600', color='blue')

# Plot predicted stationary phase
plt.plot(stationary_time, stationary_od, '--', color='green', label='Voorspelling (stationaire fase)')

# Plot predicted death phase
plt.plot(death_time, death_od, '--', color='red', label='Voorspelling (afstervingsfase)')

# Phase highlights
plt.axvspan(0, 0.5, color='lightgray', alpha=0.5, label='Lag fase')
plt.axvspan(0.5, 3.5, color='lightgreen', alpha=0.4, label='Log fase')
plt.axvspan(3.5, 5, color='gold', alpha=0.4, label='Stationaire fase')
plt.axvspan(5, 6.5, color='lightcoral', alpha=0.4, label='Afstervingsfase')

# Labels, scaling, and legend
plt.xlabel('Tijd (uren)')
plt.ylabel('OD600')
plt.title('Bacteriële groeicurve ' + r'$\it{B.\ subtilis}$')
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()
