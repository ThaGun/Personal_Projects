Two tank level control with PID controllers

Techologies:
1. Python
2. NumPy

Objective:
1. To model the dynamics of two interconnected tanks for water filling with inlet valves and outlet valves.
2. Design a simple control algorithm to control and maintain the levels of two interconnected tanks.
3. Dvelope and tune custom P, PI and PID controllers for the level control.

Control functions:
1. Summing Block (for error calculation and disturbamce calculation in feedback).
2. PID controller Block (with P, I and D mathematical models).
3. Saturator Block (for actutor appproximation).
4. Plant Block (with tank level dynamics).

Developement:
1. Differential equation for tank dynamics are solved with the Euler's method.
2. A Cascade control scheme was used for the level control of two tanks simultaneously.
3. The water inlet for the 2nd tank is considered as a disturbance for the 1st tank.
4. Two PID controllers were implemented on the both inlets of the tanks.

Results:

A well established control system for the level control of two tanks is developed and test foe differerent desired setpoints.
