
# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mobotpy.integration import rk_four
from mobotpy.models import DiffDrive

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 150
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE SETUP

# Set the track length of the differential drive vehicle [m]
ELL = 0.25

# Create a DiffDrive vehicle instance
vehicle = DiffDrive(ELL)

# %%
# CREATE A MAP OF FEATURES

# Set beacon positions as per the problem statement
beacon_positions = np.array([[4, 3], [3, -7]])

# Function to model range to beacons
def range_sensor(x, beacon_positions):
    r = np.zeros(len(beacon_positions))
    for j in range(len(beacon_positions)):
        r[j] = np.sqrt((beacon_positions[j, 0] - x[0]) ** 2 + (beacon_positions[j, 1] - x[1]) ** 2)
    return r

# Function to implement the observer for DiffDrive
def diffdrive_observer(q, u, r, beacon_positions):
    # Linearize system (Jacobian matrices)
    F = np.eye(3) + T * np.array([
        [0, 0, -0.5 * (u[0] + u[1]) * np.sin(q[2])],
        [0, 0, 0.5 * (u[0] + u[1]) * np.cos(q[2])],
        [0, 0, 0],
    ])

    H = np.zeros((len(beacon_positions), 3))
    for j in range(len(beacon_positions)):
        H[j, :] = [
            -(beacon_positions[j, 0] - q[0]) / r[j],
            -(beacon_positions[j, 1] - q[1]) / r[j],
            0,
        ]

    # Set desired poles for the observer
    desired_poles = np.array([0.7, 0.8, 0.9])

    # Compute observer gain using pole placement
    L = signal.place_poles(F.T, H.T, desired_poles).gain_matrix.T

    # Predict the next state
    q_new = q + T * vehicle.f(q, u)

    # Correct the state using the observer
    q_new = q_new + L @ (r - range_sensor(q, beacon_positions))

    return q_new

# %%
# RUN SIMULATION

# Initialize arrays for the state, input, and estimated state
x = np.zeros((3, N))
u = np.zeros((2, N))
x_hat = np.zeros((3, N))

# Set initial conditions
x[:, 0] = [-3, 2, np.pi / 6]  # Actual initial state
x_hat[:, 0] = [0, 0, 0]       # Initial estimate

# Set constant input for circular motion
v = 0.25  # linear velocity [m/s]
omega = v / 5.0  # angular velocity [rad/s] for circle of radius 5m

# Simulate the system
for k in range(1, N):
    # Set inputs for both wheels (differential drive)
    u[:, k - 1] = vehicle.uni2diff([v, omega])

    # Measure the range to each beacon
    r = range_sensor(x[:, k - 1], beacon_positions)

    # Update the estimated state
    x_hat[:, k] = diffdrive_observer(x_hat[:, k - 1], u[:, k - 1], r, beacon_positions)

    # Simulate the actual system dynamics
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

# Plot the results
plt.figure()
plt.plot(x[0, :], x[1, :], label="Actual Path")
plt.plot(x_hat[0, :], x_hat[1, :], '--', label="Estimated Path")
plt.scatter(beacon_positions[:, 0], beacon_positions[:, 1], c='r', marker='*', label="Beacons")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.legend()
plt.title("Differential Drive Vehicle Path and Estimated Path")
plt.grid()
plt.show()

# Plot the states as a function of time
fig2 = plt.figure(2)
fig2.set_figheight(6.4)
ax2a = plt.subplot(311)
plt.plot(t, x[0, :], "C0", label="Actual")
plt.plot(t, x_hat[0, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax2a, xticklabels=[])
plt.legend()

ax2b = plt.subplot(312)
plt.plot(t, x[1, :], "C0", label="Actual")
plt.plot(t, x_hat[1, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax2b, xticklabels=[])

ax2c = plt.subplot(313)
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0", label="Actual")
plt.plot(t, x_hat[2, :] * 180.0 / np.pi, "C1--", label="Estimated")
plt.ylabel(r"$\theta$ [deg]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")

plt.show()