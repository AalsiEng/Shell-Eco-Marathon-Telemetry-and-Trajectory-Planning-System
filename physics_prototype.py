import numpy as np
import math
from math import sqrt
import csv
import pandas as pd
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
df = pd.read_csv('125_power_curve.csv')
hp = 0
torque = 0
rpm = 0
wheel_radius = 0.46
accel = 0.0
neg_acceleration = 0.0
total_accel = 0.0
time_step = 0.1  # seconds

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

def deceleration_calculation():
    global tire_coeff, g, init_speed_ms, speed_kmh
    global final_speed_ms, drag_accel, neg_acceleration    
    neg_acceleration = -tire_coeff * g - drag_accel


def acceleration_calculation():
    global hp, wheel_radius, mass, accel, rpm
    calculated_torque = (hp * 5252) / rpm  # Convert HP to torque
    accel = calculated_torque / (wheel_radius * mass)  # Calculate acceleration in m/s^2

def accel_total():
    global accel, neg_acceleration, total_accel
    total_accel = accel + neg_acceleration
    
def speed_update():
    global init_speed_ms, final_speed_ms, total_accel, time_step
    final_speed_ms = init_speed_ms + total_accel * time_step
    if final_speed_ms < 0:
        final_speed_ms = 0

def distance_per_time():
    global init_speed_ms, final_speed_ms, time_step
    distance = (init_speed_ms + final_speed_ms) / 2 * time_step
    return distance


def main():
    global hp, torque
    global rpm, init_speed_ms, final_speed_ms, has_started, on_idle
    rpm = int(input("Enter RPM: "))
    hp = int(get_hp(rpm))
    torque = int(get_torque(rpm))
    idle_check(rpm)
    begin_check(on_idle)
    init_speed_ms = final_speed_ms 
    if has_started == True:
        speed_calc(rpm)
        drag_effect()
        deceleration_calculation()
        acceleration_calculation()
        accel_total()
        speed_update()
        distance = distance_per_time()
        print(f"Distance traveled in this time step: {distance:.2f} meters")
    #    speed_calc(rpm)
    #    drag_effect()
    #    friction_calculation()

main()

# This code calculates the distance required to stop a vehicle given its speed and the friction coefficient of the tires.