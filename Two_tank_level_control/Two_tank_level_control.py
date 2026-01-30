"""      Tank Level control with 2 tanks        """
import numpy as np
import matplotlib.pyplot as plt
import time
#======================================================================================================================
'''FUNCTIONS'''

'''Sum block to compute Error signal'''
def Error_Signal_Block(Setpoint_Signal, Feedback_Signal):
    Error_Signal = Setpoint_Signal + (-Feedback_Signal)
    return Error_Signal

'''Sum block to compute Disturbance'''
def Sum_Block(Controller_Signal, Disturbance):
    Sum_Signal = Controller_Signal + Disturbance
    return Sum_Signal

'''P Controller'''
def P_Controller_Block(P_Gain, Error_Computed=None, Closed_Loop=None):
    if Closed_Loop == True:
        Proportional_Signal = P_Gain * Error_Computed
    elif Closed_Loop == False:
        Proportional_Signal = P_Gain
    return Proportional_Signal

'''PI Controller'''
def PI_Controller_Block(P_Gain, I_Time, Dt=None, Input_Signal=None):
    Proportional_Signal = P_Gain * Input_Signal
    Integral_error = 0
    Integral_error += Input_Signal * Dt
    Integral_Signal = (P_Gain / I_Time) * Integral_error
    PI_Signal = Proportional_Signal + Integral_Signal
    return PI_Signal

'''PID Controller'''
def PID_Controller_Block(P_Gain, I_Time, D_Time, Dt=None, Input_Signal=None, Prev_Input_Signal=None):
    Proportional_Signal = P_Gain * Input_Signal
    Integral_error = 0
    Integral_error += Input_Signal * Dt
    Integral_Signal = (P_Gain / I_Time) * Integral_error
    Derivative_error = (Input_Signal - Prev_Input_Signal) / Dt
    Derivative_Signal = P_Gain * D_Time * Derivative_error
    PID_Signal = Proportional_Signal + Integral_Signal + Derivative_Signal
    return PID_Signal

'''Saturation block to define Actuator'''
def Valve_Saturation(Signal):
    Output = max(0, min(100, Signal))
    return Output

#=====================================================================================================================
'''Simulation time'''
dt = 0.1                     # Step size
T = 400                      # Total Time (Seconds)    
t = np.arange(0, T+dt, dt)

'''Constant Parameters'''
max_flow = 5.0
out_flow = 3.0
dispose_flow = 0.5

'''PID Controller Parameters'''
Kp_1 = 80.0                  # Proportional gain
Kp_2 = 25.0
Ti_1 = 1.3                   # Integral time
Ti_2 = 1.2
Td_1 = 0.01                     # Derivative time
Td_2 = 0.02
prev_error_1 = 0.0           # Initial prev error for derivative
prev_error_2 = 0.0

'''Setpoint'''
# Setpoint = 100
Setpoint = np.zeros(len(t))
Setpoint[t<((25.0/100.0)*T)] = 50
Setpoint[(t>=((25.0/100.0)*T)) & (t<((50.0/100.0)*T))] = 100
Setpoint[t>=((50.0/100.0)*T)] = 40

'''Initial condition'''
level_1 = np.zeros(len(t))               # Level of tank 1
level_2 = np.zeros(len(t))               # Level of tank 2
valve_1 = np.zeros(len(t))               # Valve for tank 1
valve_2 = np.zeros(len(t))               # Valve for tank 2
saturated_valve_1 = np.zeros(len(t))
saturated_valve_2 = np.zeros(len(t))
level_1[0] = 5.0                         # Initial level of tank 1
level_2[0] = 0.0                         # Initial level of tank 2

print("\nRunning Simulation....")
print("Step size:", dt)
print("Simulation time:", T, "sec")
print("="*70)

#========================================================================================================================================
start_time = time.perf_counter()

'''Simulation loop'''
for i in range(len(t)-1):
    '''Tank 1 loop'''
    error_1 = Error_Signal_Block(Setpoint_Signal=Setpoint[i], Feedback_Signal=level_1[i])
    valve_1[i] = PID_Controller_Block(P_Gain=Kp_1, I_Time=Ti_1, D_Time=Td_1, Dt=dt, Input_Signal=error_1, Prev_Input_Signal=prev_error_1)
    prev_error_1 = error_1
    valve_1_with_disturbance = Sum_Block(Controller_Signal=valve_1[i], Disturbance=valve_2[i])
    saturated_valve_1[i] = Valve_Saturation(Signal=valve_1_with_disturbance)

    '''Tank 2 loop'''
    error_2 = Error_Signal_Block(Setpoint_Signal=Setpoint[i], Feedback_Signal=level_2[i])
    valve_2[i] = PID_Controller_Block(P_Gain=Kp_2, I_Time=Ti_2, D_Time=Td_2, Dt=dt, Input_Signal=error_2, Prev_Input_Signal=prev_error_2)
    prev_error_2 = error_2
    saturated_valve_2[i] = Valve_Saturation(Signal=valve_2[i])

    '''Flow rates'''
    in_flow_tank1 = saturated_valve_1[i] / 100 * max_flow
    in_flow_tank2 = saturated_valve_2[i] / 100 * out_flow
    
    '''Level of tank 1'''
    level_1[i+1] = level_1[i] + (in_flow_tank1 - out_flow) * dt
    
    '''Level of tank 2'''
    level_2[i+1] = level_2[i] + (in_flow_tank2 - dispose_flow) * dt

end_time = time.perf_counter()

print("\nSimulation is completed Successfully..!")
print("Time taken:", round(end_time - start_time, 6), "seconds")
print("="*70)

#===============================================================================================================================================
'''Plots'''
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,2].plot(t, level_1, label="Measured level")
axes[0,2].fill_between(t, level_1, alpha=0.1, color='b')
#axes[0,2].axhline(Setpoint, color = 'r', linestyle = '--')
axes[0,2].plot(t, Setpoint, color='k', linestyle='--', label="Setpoint")
axes[0,2].set_xlabel("Time (s)")
axes[0,2].set_ylabel("Level (cm)")
axes[0,2].set_title("Level in Tank 1")
axes[0,2].grid(True)
axes[0,2].legend()

axes[0,1].plot(t, saturated_valve_1, label="Valve position")
axes[0,1].axhline(100.0, color='r', linestyle='--')
axes[0,1].axhline(0.0, color='r', linestyle='--')
axes[0,1].set_title("Valve 1 opening position")
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("Position")
axes[0,1].grid(True)
axes[0,1].legend()

axes[0,0].plot(t, valve_1, color='g', linestyle='-.', label="Contol Signal")
axes[0,0].set_title("Controller signal for valve 1")
axes[0,0].set_xlabel("Time (s)")
axes[0,0].set_ylabel("Controller Signal")
axes[0,0].grid(True)
axes[0,0].legend()

axes[1,2].plot(t, level_2, label="Measured Level")
axes[1,2].fill_between(t, level_2, alpha=0.1, color='b')
#axes[1,2].axhline( Setpoint, color = 'r', linestyle = '--')
axes[1,2].plot(t, Setpoint, color='k', linestyle='--', label="Setpoint")
axes[1,2].set_xlabel("Time (s)")
axes[1,2].set_ylabel("Level (cm)")
axes[1,2].set_title("Level in Tank 2")
axes[1,2].grid(True)
axes[1,2].legend()

axes[1,1].plot(t, saturated_valve_2, label="Valve position")
axes[1,1].axhline(100.0, color='r', linestyle='--')
axes[1,1].axhline(0.0, color='r', linestyle='--')
axes[1,1].set_title("Valve 2 opening position")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("Position")
axes[1,1].grid(True)
axes[1,1].legend()

axes[1,0].plot(t, valve_2, color='g', linestyle='-.', label="Control Signal")
axes[1,0].set_title("Controller signal for valve 2")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Controller Signal")
axes[1,0].grid(True)
axes[1,0].legend()

fig.suptitle("Tank Level Control with 2 Tanks", fontsize='16')
plt.tight_layout()
plt.show()

#=== END ===========================================================================================================================