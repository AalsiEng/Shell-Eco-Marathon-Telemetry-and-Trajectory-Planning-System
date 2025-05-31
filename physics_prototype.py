import numpy as np
import math
from math import sqrt
import csv
import pandas as pd
tire_coeff = 0.02 #assumed value for friciton coeffecient of tire rubber
g = 9.81
prim_red = 4.055
sec_red = 2.857
diff_red = 2.714
gear_1 = 2.769
gear_2 = 1.722
gear_3 = 1.272
gear_4 = 1.041
gear_5 = 0.884
gears = [float(gear_1), float(gear_2), float(gear_3), float(gear_4), float(gear_5)]
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
frontal_area = 1.5
drag_coeff = 0.42
mass = 190
df = pd.read_csv('125_power_curve.csv')
hp = 0
torque = 0
rpm = 0
wheel_radius = 0.23
accel = 0.0
neg_acceleration = 0.0
total_accel = 0.0
time_step = 0.1  # seconds
distance = 0.0  # Initialize distance
time = 0.0
corrected_torque = 0.0



def rpm_round(rpm):
    global df
    closest_rpm = df['rpm'].iloc[(df['rpm'] - rpm).abs().argsort()[:1]]
    if not closest_rpm.empty:
        return closest_rpm.values[0]
    else:
        print("No RPM values found.")
        return None

def get_hp(rpm):
    global df
    hp = df.loc[df['rpm'] == rpm, 'hp']
    if not hp.empty:
        print(f"HP: {hp.values[0]}")
        return hp.values[0]
    else:
        print("RPM value not found.")
        return None
    
def get_torque(rpm):
    global df
    torque = df.loc[df['rpm'] == rpm, 'torque']
    if not torque.empty:
        print(f"Torque: {torque.values[0]}")
        return torque.values[0]
    else:
        print("RPM value not found.")
        return None


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
    
    

def speed_calc(rpm, gear):
    global prim_red, sec_red, diff_red, gear_1, gear_2, gear_3, gear_4, gear_5
    global init_speed_ms, speed_kmh 
    
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

def deceleration_calculation():
    global tire_coeff, g, init_speed_ms, speed_kmh
    global final_speed_ms, drag_accel, neg_acceleration    
    neg_acceleration = -tire_coeff * g - drag_accel

def torque_gear_ratio_calculation(gear):
    global torque, corrected_torque, prim_red, sec_red, diff_red
    global gear_1, gear_2, gear_3, gear_4, gear_5
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
    corrected_torque = torque * prim_red * sec_red * gear_ratio * diff_red


def acceleration_calculation():
    global corrected_torque, wheel_radius, mass, accel, rpm
    accel = (corrected_torque * wheel_radius) / (mass * wheel_radius**2)
    

def accel_total():
    global accel, neg_acceleration, total_accel
    total_accel = accel + neg_acceleration
    
def speed_update():
    global init_speed_ms, final_speed_ms, total_accel, time_step
    final_speed_ms = init_speed_ms + total_accel * time_step
    print(f"Final speed in km/h: {final_speed_ms * 3.6:.2f}")
    if final_speed_ms < 0:
        final_speed_ms = 0
        

def distance_per_time():
    global init_speed_ms, final_speed_ms, time_step
    distance = (init_speed_ms + final_speed_ms) / 2 * time_step
    return distance

def final_speed_to_rpm(gear):
    global final_speed_ms, prim_red, sec_red, diff_red, gear_1, gear_2, gear_3, gear_4, gear_5
    
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
        return None
    combined_ratio = prim_red * sec_red * gear_ratio * diff_red
    rpm = (final_speed_ms * combined_ratio) / 0.10472  # Convert m/s to RPM
    return rpm

def gear_change(gear):
    global rpm, gears
    if rpm < 1000 and gear > 1:
        rpm = (gears[gear - 2] * rpm) / gears[gear - 1]
        gear -= 1
        print(f"Gear changed down to {gear}")
        return gear
    elif rpm > 8000 and gear < 5:
        rpm = (gears[gear] * rpm) / gears[gear - 1]  # Adjust RPM based on gear ratio   
        gear += 1
        print(f"Gear changed up to {gear}")
        return gear
    else:
        print("No gear change needed.")
        print(f"Current gear: {gear}")
        return gear


def main():
    global hp, torque, distance, time, idle_rpm, accel
    global rpm, init_speed_ms, final_speed_ms, has_started, on_idle
    rpm = int(input("Enter RPM: "))
    gear = int(input("Enter Gear (1-5): "))
    target_time = float(input("Enter time of accel in seconds: "))
    hp = int(get_hp(rpm)) + 1
    torque = int(get_torque(rpm))
    idle_check(rpm)
    begin_check(on_idle)
    init_speed_ms = final_speed_ms 
    if has_started == True:
        if on_idle == False:
            while time < target_time:
                rpm_round(rpm)
                speed_calc(rpm, gear)
                drag_effect()
                torque_gear_ratio_calculation(gear)
                deceleration_calculation()
                acceleration_calculation()
                accel_total()
                speed_update()
                rpm = final_speed_to_rpm(gear)
                gear = gear_change(gear)
                new_distance = distance_per_time()
                distance = distance + new_distance
                time += time_step
                print("RPM:", rpm)
            rpm = idle_rpm
            on_idle = True
        if on_idle == True:
            while final_speed_ms > 0:
                accel = 0
                init_speed_ms = final_speed_ms
                rpm_round(rpm)
                drag_effect()
                deceleration_calculation()
                accel_total()
                speed_update()
                gear = gear_change(gear)
                new_distance = distance_per_time()
                distance = distance + new_distance
                time += time_step
                print("RPM:", rpm, "time:", time)
                print(f"Distance traveled: {distance:.2f} meters")


    print(f"Distance traveled: {distance:.2f} meters")
    #    speed_calc(rpm)
    #    drag_effect()
    #    friction_calculation()

main()

# This code calculates the distance required to stop a vehicle given its speed and the friction coefficient of the tires.
# RESULTS STILL REQUIRE FURTHER TESTING AND VALIDATION