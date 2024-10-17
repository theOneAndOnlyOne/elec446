# %%
# SIMULATION SETUP

import numpy as np
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from scipy import signal
from mobotpy.models import Tricycle
from mobotpy.integration import rk_four

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 20.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Angular rate [rad/s] at which to traverse the circle
THETA_D = np.pi/4
OMEGA = 0.1
v = 6.94444
# Pre-compute the desired trajectory
x_d = np.zeros((4, N))
u_d = np.zeros((2, N))
for k in range(0, N):
    x_d[0, k] = v * np.cos(THETA_D) * t[k]
    x_d[1, k] = v * np.sin(THETA_D) * t[k]
    x_d[2, k] = THETA_D
    x_d[3, k] = 0
    u_d[0, k] = v
    u_d[1, k] = 0

# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL_W = 4.75
ELL_T = 1.92

# Create a vehicle object of type DiffDrive
vehicle = Tricycle(ELL_W, ELL_T)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(4)
x_init[0] = 0.0
x_init[1] = 0.0
x_init[2] = 3 * np.pi / 4
x_init[3] = 0.0

# Setup some arrays
x = np.zeros((4, N))
u = np.zeros((2, N))
x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization
    A = np.array(
        [
            # u_d[0 , k] = linear speed of front wheel
            # u_d[1 , k] = angular rate
            [0, 0, -u_d[0, k - 1] * np.sin(x_d[2, k - 1]), 0], 
            [0, 0, u_d[0, k - 1] * np.cos(x_d[2, k - 1]), 0],
            [0, 0, 0, u_d[0, k - 1] / ELL_W],
            [0, 0, 0, 0]
        ]
    )
    B = np.array(
        [
            [np.cos(x_d[2, k - 1]), 0], 
            [np.sin(x_d[2, k - 1]), 0],
            [0, 0], 
            [0, 1]
        ]
    )

    print(A)

    print(B)

    # Compute the gain matrix to place poles of (A - BK) at p
    p = np.array([-1.0, -2.0, -1, -1.5])
    K = signal.place_poles(A, B, p)

    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    # Compute the controls (v, omega) and convert to wheel speeds (p_B, p_F)
    u_unicycle = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k]
    print("u_unicycle :", u_unicycle)
    u[:, k] = vehicle.uni2tricycle(u_unicycle)

# %%
# MAKE PLOTS

# Change some plot settings (optional)
#plt.rc("text", usetex=True)
#plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
#plt.rc("savefig", format="pdf")
# plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--")
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel("x [m]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel("y [m]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel("theta [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel("m[m/s]")
plt.xlabel("t [s]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD = vehicle.draw(x[0, 0], x[1, 0], x[2, 0], x[3,0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_BD, Y_BD, "k")
plt.fill(X_F, Y_F, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_F, Y_F, X_BD, Y_BD = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1], x[3, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_BD, Y_BD, "k")
plt.fill(X_F, Y_F, "C3", alpha=0.5, label="End")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show the plots to the screen
# plt.show()

# %%
# MAKE AN ANIMATION

# Create the animation
ani = vehicle.animate_trajectory(x, x_d, T)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "../agv-book/gifs/ch4/control_approx_linearization.gif"
# )

# Show all the plots to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
from IPython.display import display
plt.rc("animation", html="jshtml")
display(ani)
plt.close()
