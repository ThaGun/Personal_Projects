import numpy as np
import torch
import matplotlib.pyplot as plt
from simulate import generate_data, dt, dynamics
from scipy.integrate import solve_ivp
from model import SystemIDNet
from dataset import load_data
import os
BASE = os.path.dirname(os.path.abspath(__file__))

''' Load trained model + normalized data '''
_, _, stats = load_data()
model = SystemIDNet()
model.load_state_dict(torch.load(os.path.join(BASE,"best_model.pt")))
model.eval()

''' Fresh unseen test sequence generation '''
T_test = 10.0
t_span = np.arange(0, T_test, dt)
frequency = 0.1
F_test = 2.5 * np.sin(2 * np.pi * frequency * t_span)  # unseen frequency

''' True trajectory (ground truth) '''
state = [0.0, 0.0]
true_traj = [state]
for i, t in enumerate(t_span[:-1]):
    sol = solve_ivp(dynamics, [t, t+dt], state,
                    args=(F_test[i],), max_step=dt/5)
    state = sol.y[:, -1].tolist()
    true_traj.append(state)

true_traj = np.array(true_traj)

''' NN rollout '''
state = np.array([0.0, 0.0], dtype=np.float32)
nn_traj = [state.copy()]

with torch.no_grad():
    for i in range(len(t_span) - 1):
        inp = np.array([state[0], state[1], F_test[i]], dtype=np.float32)
        # Normalize input
        inp_norm = (inp - stats["X_mean"]) / stats["X_std"]
        x_t = torch.tensor(inp_norm).unsqueeze(0)
        # Predict next step
        y_norm = model(x_t).squeeze().numpy()
        # Denormalize output
        state = y_norm * stats["Y_std"] + stats["Y_mean"]
        nn_traj.append(state.copy())

nn_traj = np.array(nn_traj)

''' Plots '''
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].plot(t_span, true_traj[:, 0], label="True Trajectory", lw=2)
axs[0].plot(t_span, nn_traj[:, 0], "--", label="Neural Network Trajectory", lw=2)
axs[0].set_ylabel("Position x") 
axs[0].legend()
axs[0].grid()

axs[1].plot(t_span, true_traj[:, 1], label="True Trajectory", lw=2)
axs[1].plot(t_span, nn_traj[:, 1], "--", label="Neural Network Trajectory", lw=2)
axs[1].set_ylabel("Velocity ẋ")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t_span, F_test, label="Applied Force", lw=2)
axs[2].set_ylabel("Force F")
axs[2].set_xlabel("Time (s)")
axs[2].legend()
axs[2].grid()

plt.suptitle("Neural Network Rollout vs True Trajectory")
plt.tight_layout()
plt.show()