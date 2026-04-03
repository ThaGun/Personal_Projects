import numpy as np
from scipy.integrate import solve_ivp

# System parameters
m = 1.0   # mass (kg)
c = 0.5   # damping coefficient
k = 2.0   # spring stiffness

# Sim setup
dt = 0.05 # timestep (seconds)
T  = 20.0 # total simulation time

def dynamics(t, y, F):
    ''' ODE: m*x'' + c*x' + k*x = F '''
    x, x_dot = y
    x_ddot = (F - c * x_dot - k * x) / m
    return [x_dot, x_ddot]   # x_dot - velocity

def generate_data(n_sequences=50):
    data = []
    t_span = np.arange(0, T, dt)

    for _ in range(n_sequences):
        ''' Randon force: slow sine waves + noise '''
        freq = np.random.uniform(0.1, 1.0)
        F_t = 3.0 * np.sin(2 * np.pi * freq * t_span)
        F_t += np.random.randn(len(t_span)) * 0.5

        ''' Random initial conditions '''
        x0 = np.random.uniform(-1, 1)
        v0 = np.random.uniform(-1, 1)
        state = [x0, v0]
        seq = []

        for i, t in enumerate(t_span[:-1]):
            F = F_t[i]
            sol = solve_ivp(dynamics, [t, t+dt],
                            state, args=(F,), max_step=dt/5)
            next_state = sol.y[:,-1].tolist()
            # [x, x_dot, F, x_next, x_dot_next]
            seq.append(state + [F] + next_state)
            state = next_state
        
        data.extend(seq)
    
    return np.array(data)

if __name__ == "__main__":
    data = generate_data(n_sequences=100)
    np.save("system_data.npy", data)
    print(f"Generated {len(data)} samples")
    print(f"Shape {data.shape}")