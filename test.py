import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters and Setup
# -------------------------------

a = 0.9          # advection speed (a > 0)
ht = 1.0         # time step size
hx = 1.0         # spatial step size
gamma = a * ht / hx  # Courant number

# Create an array for k*h_x ranging from -pi to pi
k_hx = np.linspace(-np.pi, np.pi, 500)
# Recover k from k*h_x (since hx = 1, k == k_hx; but we write it generally)
ks = k_hx / hx

# -------------------------------
# 1. Upwind Scheme (ITCS)
# -------------------------------
# The update rule is: u^{n+1}_j = (1-gamma) * u^n_j + gamma * u^n_{j-1}
# For a Fourier mode u^n_j = exp(i k x_j), we have:
#   s_up = (1-gamma) + gamma * exp(-i k hx)
s_upwind = (1 - gamma) + gamma * np.exp(-1j * ks * hx)
# The dispersion relation is defined via: exp(-i omega ht) = s.
# Taking the logarithm, we get:
omega_upwind = 1j * np.log(s_upwind) / ht

# -------------------------------
# 2. Crank-Nicolson Scheme
# -------------------------------
# A centered-in-time and centered-in-space discretization gives:
#   s_CN = (1 - i*gamma*sin(k hx)) / (1 + i*gamma*sin(k hx))
s_CN = (1 - 1j * gamma * np.sin(ks * hx)) / (1 + 1j * gamma * np.sin(ks * hx))
omega_CN = 1j * np.log(s_CN) / ht

# -------------------------------
# 3. Lax-Wendroff Scheme
# -------------------------------
# The Lax-Wendroff update:
#   u^{n+1}_j = u^n_j - (gamma/2)*(u^n_{j+1} - u^n_{j-1})
#               + (gamma^2/2)*(u^n_{j+1} - 2u^n_j + u^n_{j-1})
# leads to the symbol:
#   s_LW = 1 - gamma^2
#          + (-gamma/2 + gamma^2/2)*exp(i k hx)
#          + ( gamma/2 + gamma^2/2)*exp(-i k hx)
s_LW = (1 - gamma**2
        + np.exp(1j * ks * hx) * (-gamma/2 + gamma**2/2)
        + np.exp(-1j * ks * hx) * (gamma/2 + gamma**2/2))
omega_LW = 1j * np.log(s_LW) / ht

# -------------------------------
# Compute Phase Velocity and Amplitude
# -------------------------------

# The phase velocity for a scheme is defined as:
#   c_phase = Re(omega) / k
# We need to handle k = 0 specially.
def compute_phase_velocity(omega, ks, a_exact):
    phase_vel = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(ks):
        if np.abs(k) < 1e-6:
            # For k -> 0, the exact phase velocity should be a
            phase_vel[i] = a_exact
        else:
            phase_vel[i] = np.real(omega[i]) / k
    return phase_vel

phase_upwind = compute_phase_velocity(omega_upwind, ks, a)
phase_CN     = compute_phase_velocity(omega_CN, ks, a)
phase_LW     = compute_phase_velocity(omega_LW, ks, a)

# The amplitude of the symbol (|s|) indicates the dissipation:
#   - |s| = 1 means no amplitude change (no dissipation),
#   - |s| < 1 indicates dissipation.
amp_upwind = np.abs(s_upwind)
amp_CN     = np.abs(s_CN)
amp_LW     = np.abs(s_LW)

# -------------------------------
# Plotting
# -------------------------------

plt.figure(figsize=(12, 5))

# Plot Phase Velocity vs. k*hx
plt.subplot(1, 2, 1)
plt.plot(k_hx, phase_upwind, label="Upwind (ITCS)", lw=2)
plt.plot(k_hx, phase_CN, label="Crank-Nicolson", lw=2)
plt.plot(k_hx, phase_LW, label="Lax-Wendroff", lw=2)
plt.plot(k_hx, a * np.ones_like(k_hx), 'k--', label="Exact (a = {:.2f})".format(a), lw=2)
plt.xlabel(r"$k\,h_x$", fontsize=14)
plt.ylabel("Phase Velocity", fontsize=14)
plt.title("Phase Velocity vs. $k\,h_x$", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

# Plot Amplitude of the Symbol vs. k*hx
plt.subplot(1, 2, 2)
plt.plot(k_hx, amp_upwind, label="Upwind (ITCS)", lw=2)
plt.plot(k_hx, amp_CN, label="Crank-Nicolson", lw=2)
plt.plot(k_hx, amp_LW, label="Lax-Wendroff", lw=2)
plt.xlabel(r"$k\,h_x$", fontsize=14)
plt.ylabel(r"Amplitude $|s|$", fontsize=14)
plt.title("Amplitude of the Symbol vs. $k\,h_x$", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
