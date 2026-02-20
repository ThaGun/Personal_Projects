# Two tank level control with PID controllers

## Techologies💻:
1. Python
2. NumPy

## Objective🎯:
1. To model the dynamics of two interconnected tanks for water filling with inlet valves and outlet valves.
2. Design a simple control algorithm to control and maintain the levels of two interconnected tanks.
3. Dvelope and tune custom P, PI and PID controllers for the level control.

## Control functions⚙️:
1. Summing Block (for error calculation and disturbamce calculation in feedback).
2. PID controller Block (with P, I and D mathematical models).

   $u(t)=Kp​e(t)+Ki​∫​e(τ)dτ+Kd​dtde(t)$
   ​
4. Saturator Block (for actutor appproximation).
5. Plant Block (with tank level dynamics).

## Developement▶️:
1. Differential equation for tank dynamics are solved with the Euler's method.

   $Level_k+1​=Level_k​+(Inflow−Outflow)⋅Δt$

   Where:
   
   $Inflow = valve percentage/100 × max_flow$

   $Outflow = constant disturbance$

   $Δt = time step$
   
3. A Cascade control scheme was used for the level control of two tanks simultaneously.
4. The water inlet for the 2nd tank is considered as a disturbance for the 1st tank.
5. Two PID controllers were implemented on the both inlets of the tanks.

## Results✅:

A well established control system for the level control of two tanks is developed and tested for differerent desired setpoints.

<img width="1536" height="754" alt="Figure_1" src="https://github.com/user-attachments/assets/9a747b02-b9b8-47d5-98df-8d4b5175e314" />
