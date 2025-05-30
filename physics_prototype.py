import numpy as np
tire_coeff = 0.2 #assumed value for friciton coeffecient of tire rubber
g = 9.81
prim_red = 4.055
sec_red = 2.533
diff_red = 2.714
gear_1 = 2.769
gear_2 = 1.722
gear_3 = 1.272
gear_4 = 1.000
idle_rpm = 800
global on_idle
global has_started
on_idle = True
has_started = False
speed_ms = 0.0
speed_kmh = 0.0


def idle_check(rpm):
    global on_idle
    if rpm < idle_rpm:
        on_idle = True
    else:
        on_idle = False

def begin_check(on_idle):
    global has_started
    if has_started == False:
        if on_idle == True:
            has_started = False
            return has_started
        else:
            has_started = True
            return has_started
    else:
        return

def speed_calc(rpm):
    
    gear = int(input("Enter Gear (1-4): "))
    if gear == 1:
        gear_ratio = gear_1
    elif gear == 2:
        gear_ratio = gear_2
    elif gear == 3:
        gear_ratio = gear_3
    elif gear == 4:
        gear_ratio = gear_4
    else:
        print("Invalid gear selected.")
        return
    combined_ratio = prim_red * sec_red * gear_ratio * diff_red
    # Calculate speed in m/s
    speed_ms = (rpm * 0.10472) / combined_ratio  # Convert RPM to m/s
    speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
    print(f"Speed: {speed_kmh:.2f} km/h")

def friction_calculation():    
    neg_acceleration = -tire_coeff * g
    distance = (speed_ms ** 2) / (2 * abs(neg_acceleration))
    speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
    print(f"Speed: {speed_kmh:.2f} km/h")
    print(f"Distance to stop: {distance:.2f} m")


def main():
    rpm = int(input("Enter RPM: "))
    idle_check(rpm)
    begin_check(on_idle)
    if has_started:
        speed_calc(rpm)
        friction_calculation()

# This code calculates the distance required to stop a vehicle given its speed and the friction coefficient of the tires.