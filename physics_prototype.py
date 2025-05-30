import numpy as np
from math import sqrt
tire_coeff = 0.2 #assumed value for friciton coeffecient of tire rubber
g = 9.81
prim_red = 4.055
sec_red = 2.857
diff_red = 2.714
gear_1 = 2.769
gear_2 = 1.722
gear_3 = 1.272
gear_4 = 1.041
gear_5 = 0.884
idle_rpm = 800
global on_idle
global has_started
on_idle = True
has_started = False
init_speed_ms = 0.0
final_speed_ms = 0.0
speed_kmh = 0.0
drag_accel = 0.0
drag_force = 0.0
frontal_area = 1.95
drag_coeff = 0.37
mass = 190


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

def drag_effect():
    global init_speed_ms
    global drag_force
    global drag_coeff
    global frontal_area
    global g
    global mass
    global drag_accel
    drag_force = 0.5 * drag_coeff * frontal_area * (init_speed_ms ** 2)
    drag_accel = drag_force / mass
    
    

def speed_calc(rpm):
    global prim_red, sec_red, diff_red, gear_1, gear_2, gear_3, gear_4, gear_5
    global init_speed_ms, speed_kmh 
    gear = int(input("Enter Gear (1-5): "))
    if gear == 1:
        gear_ratio = gear_1
    elif gear == 2:
        gear_ratio = gear_2
    elif gear == 3:
        gear_ratio = gear_3
    elif gear == 4:
        gear_ratio = gear_4
    elif gear == 5:
        gear_ratio = gear_5
    else:
        print("Invalid gear selected.")
        return
    combined_ratio = prim_red * sec_red * gear_ratio * diff_red
    # Calculate speed in m/s
    init_speed_ms = (rpm * 0.10472) / combined_ratio  # Convert RPM to m/s
    speed_kmh = init_speed_ms * 3.6  # Convert m/s to km/h
    print(f"Speed: {speed_kmh:.2f} km/h")

def friction_calculation():
    global tire_coeff, g, init_speed_ms, speed_kmh
    global final_speed_ms, drag_accel    
    neg_acceleration = -tire_coeff * g - drag_accel
    distance = (init_speed_ms ** 2) / (2 * abs(neg_acceleration))
    speed_kmh = init_speed_ms * 3.6  # Convert m/s to km/h
    final_speed_ms = sqrt(init_speed_ms ** 2 + 2 * neg_acceleration * distance)
    print(f"Speed: {speed_kmh:.2f} km/h")
    print(f"Distance to stop: {distance:.2f} m")



def main():
    global rpm, init_speed_ms, final_speed_ms, has_started, on_idle
    rpm = int(input("Enter RPM: "))
    idle_check(rpm)
    begin_check(on_idle)
    init_speed_ms = final_speed_ms 
    if has_started == True:
        speed_calc(rpm)
        drag_effect()
        friction_calculation()

main()

# This code calculates the distance required to stop a vehicle given its speed and the friction coefficient of the tires.