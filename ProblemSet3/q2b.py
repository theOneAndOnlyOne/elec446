"""
Example MPC_linear_cvxpy.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import signal

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 30.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# VEHICLE MODELS

# Vehicle mass [kg]
m = 1.0

# Discrete time vehicle model
F = np.array([[1, T], [0, 1]])
G = np.array([[(T*T)/(2 * m)], [T / m]])
n = G.shape[0]

# Continuous time vehicle model (for full-state feedback)
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1 / m]])

# %%
# UNCONSTRAINED MPC CONTROLLER DESIGN

# Lookahead time steps
p = 100

# Decide on state and input cost matrices
smallQ = np.diag([1.0, 1.0])
smallR = np.diag([0.1])

# %%
# FULL-STATE FEEDBACK CONTROLLER DESIGN

# Choose some poles for the FSF controller
poles = np.array([-1.0, -2.0])

# Find the controller gain to place the poles at our chosen locations
K_FSF = signal.place_poles(A, B, poles)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEMS

# Set the desired trajectory to take a step mid way through the trajectory
x_d = np.zeros((2, N + p))
for k in range(int(N / 2), N + p):
    x_d[0, k] = 10
    x_d[1, k] = 0

# Set up some more arrays for MPC and FSF
x_MPC = np.zeros((n, N))
u_MPC = np.zeros((1, N))
xi_d = np.zeros(n * p)
x_FSF = np.zeros((n, N))
u_FSF = np.zeros((1, N))

# Set the initial conditions
x_MPC[0, 0] = 3
x_MPC[1, 0] = 0
x_FSF[0, 0] = x_MPC[0, 0]
x_FSF[1, 0] = x_MPC[1, 0]

# Simulate the the closed-loop system with MPC
for k in range(1, N):

    # Simulate the vehicle motion under MPC
    x_MPC[:, k] = F @ x_MPC[:, k - 1] + G @ u_MPC[:, k - 1]

    # Set vectors for optimization
    x = cp.Variable((n, p))
    u = cp.Variable((1, p))

    # Initialize the cost function and constraints
    J = 0
    constraints = []

    # For each lookahead step
    for j in range(0, p):

        # Increment the cost function
        J += cp.quad_form(x[:, j] - x_d[:, k + j], smallQ) + cp.quad_form(
            u[:, j], smallR
        )

        # Enter the "subject to" constraints

        # Add a constraint that enforces the system dynamics for each step k
        constraints += [x[:, j] == F @ x[:, j - 1] + G @ u[:, j - 1]]

        # Add a constraint to set the initial state x[:, 0] in the optimization problem to the current state x_MPC[:, k]
        constraints += [x[:, 0] == x_MPC[:, k]]

        #constraints += [u[:, j] >= -2.0, u[:, j] <= 2.0]
        constraints += [x[1, j] >= -2.0, x[1, j] <= 2.0]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(J), constraints)
    problem.solve(verbose=False)

    # Set the control input to the first element of the solution
    u_MPC[:, k] = u[0, 0].value

# Simulate the closed-loop system with FSF
for k in range(1, N):
    x_FSF[:, k] = F @ x_FSF[:, k - 1] + G @ u_FSF[:, k - 1]
    u_FSF[:, k] = -K_FSF.gain_matrix @ (x_FSF[:, k - 1] - x_d[:, k - 1])

# %%
# PLOT THE RESULTS

# Change some plot settings (optional)

fig1 = plt.figure(1)
ax1a = plt.subplot(311)
plt.plot(t, x_d[0, 0:N], "C2--")
plt.plot(t, x_MPC[0, :], "C0")
plt.plot(t, x_FSF[0, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"x_1 [m]")
plt.legend(["Desired", "MPC", "Full-state feedback"], loc="lower right")
plt.setp(ax1a, xticklabels=[])
ax1b = plt.subplot(312)
plt.plot(t, x_d[1, 0:N], "C2--")
plt.plot(t, x_MPC[1, :], "C0")
plt.plot(t, x_FSF[1, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"x_2 [m/s]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(313)
plt.plot(t, u_MPC[0, :], "C0")
plt.plot(t, u_FSF[0, :], "C1")
plt.grid(color="0.95")
plt.ylabel(r"u [N]")
plt.xlabel(r"t [s]")

# Save the plot
# plt.savefig("../agv-book/figs/ch4/MPC_linear_tracking_fig1.pdf")

# Show the plots
plt.show()
