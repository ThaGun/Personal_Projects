''' Distillation Column Modelling '''
import numpy as np
import matplotlib.pyplot as plt
from DC_Parameters import ParametersCalc
from DC_Controller import ControlFunc
#=======================================================================================================================================
''' Parameters '''
F_con = 4.0             # Flow rate
R_con = 1.1             # Reflux ratio
V = 2.8                 # Molar ratio of vapor
xf = 0.3                # Feed composition
StageNumbers = 4        # n number of stages
FlowStages = 1          # feed stages

T = 50.0                # seconds
dt = 0.01               # Step size
t = np.arange(0, T+dt, dt)

''' Setpoint and disturbance '''
D_setpoint = 2.0        # Molar flowrate withdrawn at overhead condenser 
B_setpoint = 3.0        # Molar flowrate withdrawn at bottom reboiler

F = np.ones(len(t)) * F_con
F[(t >= 10) & (t <= 20)] = F_con + 1.0

R = np.ones(len(t)) * R_con
R[(t >= 30) & (t <= 40)] = R_con + 1.0

''' Vector Initialization '''
u = np.zeros((2))              # Control vector
x = np.zeros((16, len(t)))     # State vector
d = np.zeros((2, len(t)))      # Disturbance vector

''' Control vector '''
# u = [D; B]
u[0], u[1] = D_setpoint, B_setpoint

''' State Vector'''
# Initial conditions
x0 = np.array([[1.0] ,[0.1], [1.0] ,[0.1], [1.0] ,[0.1], [1.0] ,[0.1], 
               [1.0], [0.1], [1.0] ,[0.1], [1.0] ,[0.1], [1.0] ,[0.1]])

# x = [Hd; xd; HN; xN; Hn4; xn4; Hn3; xn3; Hn2; xn2; Hf1; xf1; H1; x1; Hb; xb]
x[:,0] = x0.flatten()

''' Disturbance vector '''
# d = [F; xf]
d[0], d[1] = F, xf

''' PID Controller parameters '''
Kp1 = 3.0
Ti1 = 2.0
Td1 = 0.0
prev_err_D = 0.0
Integral_Error_D = 0.0

Kp2 = 11.0
Ti2 = 5.0
Td2 = 0.0
prev_err_B = 0.0
Integral_Error_B = 0.0

Parameters = ParametersCalc(StageNumbers, FlowStages)

#========================================================================================================================================
''' Simulation '''
for i in range(len(t)-1):
    
    ''' Mole fraction of vapor '''
    y_N = Parameters.MoleFractionOfVapor(MoleFracOfL=x[3,i])
    y_Nmin1 = Parameters.MoleFractionOfVapor(MoleFracOfL=x[5,i])
    y_n3 = Parameters.MoleFractionOfVapor(MoleFracOfL=x[7,i])
    y_n2 = Parameters.MoleFractionOfVapor(MoleFracOfL=x[9,i])
    y_nmin1 = Parameters.MoleFractionOfVapor(MoleFracOfL=x[11,i])
    y_fmin1 = Parameters.MoleFractionOfVapor(MoleFracOfL=x[13,i])
    y_b = Parameters.MoleFractionOfVapor(MoleFracOfL=x[15,i])

    ''' Liquid flow rate '''
    L_N = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[2,i])
    L_n4 = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[4,i])
    L_n3 = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[6,i])
    L_n2 = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[8,i])
    L_f1 = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[10,i])
    L_1 = Parameters.RateOfFlowOfLiquid(LiquidHoldup=x[12,i])

    ''' Controllers '''
    Error_D = ControlFunc.Error_Signal_Block(Setpoint_Signal=-(u[0]), Feedback_Signal=x[0,i])
    PID_D, IE_D = ControlFunc.PID_Controller_Block(P_Gain=Kp1, I_Time=Ti1, D_Time=Td1, Dt=dt, Input_Signal=Error_D, Prev_Input_Signal=prev_err_D, Integral_error=Integral_Error_D)
    PID_D = ControlFunc.Saturation(Signal=PID_D)
    Integral_Error_D = IE_D
    prev_err_D = Error_D

    Error_B = ControlFunc.Error_Signal_Block(Setpoint_Signal=-(u[1]), Feedback_Signal=x[14,i])
    PID_B, IE_B = ControlFunc.PID_Controller_Block(P_Gain=Kp2, I_Time=Ti2, D_Time=Td2, Dt=dt, Input_Signal=Error_B, Prev_Input_Signal=prev_err_B, Integral_error=Integral_Error_B)
    PID_B = ControlFunc.Saturation(Signal=PID_B)
    Integral_Error_B = IE_B
    prev_err_B = Error_B

    ''' Balance equations '''
    # Condenser and reflux
    x[0,i+1] = x[0,i] + (V - R[i] - PID_D) * dt
    x[1,i+1] = x[1,i] + (V * (y_N - x[1,i]) / x[0,i]) * dt

    # Top tray
    x[2,i+1] = x[2,i] + (R[i] - L_N) * dt
    x[3,i+1] = x[3,i] + (((V * (y_Nmin1 - y_N)) + (R[i] * (x[1,i] - x[3,i])))/ x[2,i]) * dt

    # Arbitary tray
    # 4th tray
    x[4,i+1] = x[4,i] + (L_N - L_n4) * dt
    x[5,i+1] = x[5,i] + ((V * (y_n3 - y_Nmin1) + L_N * (x[3,i] - x[5,i])) / x[4,i]) * dt
    
    # 3rd tray
    x[6,i+1] = x[6,i] + (L_n4 - L_n3) * dt
    x[7,i+1] = x[7,i] + ((V * (y_n2 - y_n3) + L_n4 * (x[5,i] - x[7,i])) / x[6,i]) * dt

    # 2nd tray
    x[8,i+1] = x[8,i] + (L_n3 - L_n2) * dt
    x[9,i+1] = x[9,i] + ((V * (y_nmin1 - y_n2) + L_n3 * (x[7,i] - x[9,i])) / x[8,i]) * dt

    # Feed tray
    x[10,i+1] = x[10,i] + (L_n2 - L_f1 + d[0,i]) * dt
    x[11,i+1] = x[11,i] + ((V * (y_fmin1 - y_nmin1) + L_f1 * (x[9,i] - x[11,i]) + d[0,i] * (d[1,i] - x[11,i])) / x[10,i]) * dt

    # First tray
    x[12,i+1] = x[12,i] + (L_f1 - L_1) * dt
    x[13,i+1] = x[13,i] + ((V * (y_b - y_fmin1) + L_f1 * (x[11,i] - x[13,i])) / x[12,i]) * dt

    # Column base and Reboiler
    x[14,i+1] = x[14,i] + (L_1 - V - PID_B) * dt
    x[15,i+1] = x[15,i] + ((V * (x[15,i] - y_b) + L_1 * (x[13,i] - x[15,i])) / x[14,i]) * dt

#============================================================================================================================================    
print("Integral error of Hd:" , Integral_Error_D)
print("Integral error of Hb:" , Integral_Error_B)

''' Plots '''
fig, axes = plt.subplots(2, 3, figsize=(13, 10))

axes[0,0].plot(t, x[0], label="Hd")
axes[0,0].axhline(u[0], linestyle='--', color='r', label="Setpoint")
axes[0,0].set_ylabel("Mass balance in Condenser and reflux receiver")
axes[0,0].set_xlabel("Time (s)")
axes[0,0].legend()
axes[0,0].grid(True)

axes[1,0].plot(t, x[14], label="Hb")
axes[1,0].axhline(u[1], linestyle='--', color='r', label="Setpoint")
axes[1,0].set_ylabel("Mass balance in Column base and reboiler")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].legend()
axes[1,0].grid(True)

axes[0,1].plot(t, (x[8]), label="Component A")
axes[0,1].set_ylabel("Total volume of accumulated liquid")
axes[0,1].set_xlabel("Time (s)")
axes[0,1].legend()
axes[0,1].grid(True)

axes[1,1].plot(t, (x[8]*x[9]), label="Component A")
axes[1,1].set_ylabel("Rate of accumulation in liquid phase")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].legend()
axes[1,1].grid(True)

axes[0,2].plot(t, d[0], linestyle="-", color='m',label="F")
axes[0,2].set_ylabel("Flow rate (mol/sec)")
axes[0,2].set_xlabel("Time (s)")
axes[0,2].legend()
axes[0,2].grid(True)

axes[1,2].plot(t, R, linestyle="-", color='m', label="R")
axes[1,2].set_ylabel("Reflux ratio (mol/sec)")
axes[1,2].set_xlabel("Time (s)")
axes[1,2].legend()
axes[1,2].grid(True)

fig.suptitle("Distillation Column with PID Controllers", fontsize='16')
plt.tight_layout()
plt.show()

#===END======================================================================================================================================