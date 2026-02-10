""" Control Functions """
import numpy as np
class ControlFunc:
    def Error_Signal_Block(Setpoint_Signal, Feedback_Signal):
        Error_Signal = Setpoint_Signal + Feedback_Signal
        return Error_Signal

    def Sum_Block(Controller_Signal, Disturbance):
        Sum_Signal = Controller_Signal + Disturbance
        return Sum_Signal

    def PID_Controller_Block(P_Gain, I_Time, D_Time, Dt=None, Input_Signal=None, Prev_Input_Signal=None, Integral_error=None):
        Proportional_Signal = P_Gain * Input_Signal
        Integral_error = Integral_error + Input_Signal * Dt
        Integral_Signal = (I_Time) * Integral_error
        Derivative_error = (Input_Signal - Prev_Input_Signal) / Dt
        Derivative_Signal = D_Time * Derivative_error
        PID_Signal = Proportional_Signal + Integral_Signal + Derivative_Signal
    
        return PID_Signal, Integral_error

    def Saturation(Signal):
        Output = max(0, min(np.inf, Signal))
        return Output