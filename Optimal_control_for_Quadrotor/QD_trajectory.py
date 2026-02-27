import numpy as np

def Circle_trajectory(R, Omega, x_ref, k, dt):

    t = Omega * k * dt
    cr = 0.0 * k * dt
    x_ref[0] = R * np.cos(t)
    x_ref[1] = R * np.sin(t)
    x_ref[2] = 1.0 + cr
    #x_ref[6] = np.deg2rad(20)

    return x_ref

def Spiral_trajectory(R, Omega, x_ref, k, dt, expand_rate=0.05):

    t = Omega * k * dt
    cr = 0.1 * k * dt
    r = expand_rate * k * dt  # grows over time, or use R + expand_rate*t
    x_ref[0] = r * np.cos(t)
    x_ref[1] = r * np.sin(t)
    x_ref[2] = 1.0 + cr
    return x_ref

def Sinusoidal_trajectory(R, Omega, x_ref, k, dt, amplitude=1.0):

    t = Omega * k * dt
    cr = 0.0 * k * dt
    x_ref[0] = R * t / (2 * np.pi)        # forward motion
    x_ref[1] = amplitude * np.sin(t)       # lateral oscillation
    x_ref[2] = 1.0 + cr
    return x_ref

def Square_trajectory(R, Omega, x_ref, k, dt):
    t = (Omega * k * dt) % (2 * np.pi)
    cr = 0.1 * k * dt
    segment = int(t / (np.pi / 2))  # 4 segments
    s = (t % (np.pi / 2)) / (np.pi / 2)  # progress within segment [0,1]
    corners = [(R, 0), (0, R), (-R, 0), (0, -R)]  # adjust as needed
    waypoints = [(-R, -R), (R, -R), (R, R), (-R, R)]
    x0, y0 = waypoints[segment % 4]
    x1, y1 = waypoints[(segment + 1) % 4]
    x_ref[0] = x0 + s * (x1 - x0)
    x_ref[1] = y0 + s * (y1 - y0)
    x_ref[2] = 1.0 + cr
    return x_ref

def Waypoint_trajectory(waypoints, x_ref, k, dt, speed=0.5):
    distance = speed * k * dt
    cr = 0.1 * k * dt
    cumulative = 0.0
    for i in range(len(waypoints) - 1):
        seg_len = np.linalg.norm(np.array(waypoints[i+1]) - np.array(waypoints[i]))
        if cumulative + seg_len >= distance:
            s = (distance - cumulative) / seg_len
            x_ref[0] = waypoints[i][0] + s * (waypoints[i+1][0] - waypoints[i][0])
            x_ref[1] = waypoints[i][1] + s * (waypoints[i+1][1] - waypoints[i][1])
            x_ref[2] = 1.0 + cr
            return x_ref
        cumulative += seg_len
    # Hold last waypoint
    x_ref[0], x_ref[1] = waypoints[-1][0], waypoints[-1][1]
    x_ref[2] = 1.0
    return x_ref

# Usage:
# wps = [(0,0), (2,0), (2,2), (0,2)]
# Waypoint_trajectory(wps, x_ref, k, dt)

def Circle_trajectory2(x_ref, k, dt):

    x_ref[0] = 1.0 * np.cos(0.5 * k)
    x_ref[1] = 1.0 * np.sin(0.5 * k)
    x_ref[2] = 1.0

    return x_ref