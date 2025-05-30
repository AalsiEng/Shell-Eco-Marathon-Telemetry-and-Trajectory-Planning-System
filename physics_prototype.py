import numpy as np
tire_coeff = 0.2 #assumed value for friciton coeffecient of tire rubber
g = 9.81
def test():
    speed = int(input("Enter speed in m/s: "))
    neg_acceleration = -tire_coeff * g
    distance = (speed ** 2) / (2 * abs(neg_acceleration))
    print(f"Distance to stop: {distance:.2f} m")

test()
# This code calculates the distance required to stop a vehicle given its speed and the friction coefficient of the tires.