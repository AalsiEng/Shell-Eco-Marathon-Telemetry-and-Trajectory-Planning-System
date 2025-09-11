import numpy as np
import math
from math import sqrt
import csv
import pandas as pd
import csv
from math import radians, sin, cos, sqrt, atan2, degrees

# TO BE VALIDATED AND TESTED
drivetrain_efficiency = 0.65
tire_coeff = 0.05 #assumed value for friciton coeffecient of tire rubber
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
idle_rpm = 2500
global on_idle
global has_started
on_idle = True
has_started = False
init_speed_ms = 0.0
final_speed_ms = 0.0
speed_kmh = 0.0
drag_accel = 0.0
drag_force = 0.0
frontal_area = 1.298
drag_coeff = 0.327335
mass = 220
df = pd.read_csv('125_power_curve.csv')
ff = pd.read_csv('driving_log.csv')
hp = 0
torque = 0
rpm = 0
wheel_radius = 0.31
accel = 0.0
neg_acceleration = 0.0
total_accel = 0.0
time_step = 0.1 # seconds
distance_covered = 0.0  # Initialize distance
time = 0.0
corrected_torque = 0.0
throttle = 0.0
corrected_power = 0
throttle_torque = 0
has_ended = False
dist = 0
rounded_rpm = 0


def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0
    # Convert coordinates to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def read_coords(filename):
    coords = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            coords.append((lat, lon))
    return coords

def throttle_calc():
    global throttle, dist, time, time_step, distance_covered, has_ended
    if distance_covered < dist:
        throttle = 100
    else:
        throttle = 0

def has_ended_basic():
    global has_ended, dist, distance_covered
    if distance_covered >= dist:
        has_ended = 1
    else:
        has_ended = 0
    print(f"Has ended: {has_ended}")

def total_distance(coords):
    distance = 0.0
    for i in range(1, len(coords)):
        distance += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        print(distance)
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):     
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    return (bearing + 360) % 360

def get_has_ended_from_csv():
    global has_ended, time
    global ff
    has_ended = ff.loc[ff['time'] == time, 'has_ended']
    if not has_ended.empty:
        has_ended = has_ended.values[0]
        print(f"Has ended: {has_ended}")
    else:
        print("No end status found for the current time.")
        has_ended = 0  

#def get_throttle_from_csv():
#   global throttle, time
#    global ff
#    throttle = ff.loc[ff['time'] == time, 'throttle']
#   if not throttle.empty:
#        throttle = throttle.values[0] * 100
#        print(f"Throttle: {throttle}")
#    else:
#        print("No throttle value found for the current time.")
 #       throttle = 0  
    
def rpm_round():
    global df, rpm, rounded_rpm
    closest_rpm = df['rpm'].iloc[(df['rpm'] - rpm).abs().argsort()[:1]]
    if not closest_rpm.empty:
        rounded_rpm = closest_rpm.values[0]
    else:
        print("No RPM values found.")
        return None

def get_hp():
    global df, rpm, rounded_rpm
    global hp
    hp = df.loc[df['rpm'] == rounded_rpm, 'hp']
    if not hp.empty:
        print(f"HP: {hp.values[0]}")
        hp =  int(hp.values[0]) * .6
    else:
        print("RPM value not found.")
        hp = 0  # Default to 0 if no value found
    
def get_torque():
    global df, rpm, torque, rounded_rpm
    torque = df.loc[df['rpm'] == rounded_rpm, 'torque']
    if not torque.empty:
        print(f"Torque: {torque.values[0]}")
        torque =  float(torque.values[0]) * .6
       


def idle_check():
    global on_idle, throttle
    if throttle == 0:
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
    drag_force = 0.5 * drag_coeff * frontal_area * (init_speed_ms * init_speed_ms) * 1.225  # Air density at sea level in kg/m^3
    drag_accel = drag_force / mass
    print(f"Drag Force: {drag_force:.2f} N")
    print(f"Drag Acceleration: {drag_accel:.2f} m/s²")
    

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
    neg_acceleration = -( tire_coeff * g) - drag_accel

def torque_gear_ratio_calculation(gear):
    global torque, corrected_torque, prim_red, sec_red, diff_red,throttle_torque, throttle, drivetrain_efficiency15
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
    throttle_torque = torque * prim_red * sec_red * gear_ratio * diff_red * drivetrain_efficiency
    corrected_torque = throttle_torque * (throttle / 100)  # Adjust torque based on throttle percentage
    print(f"Corrected Torque: {corrected_torque:.2f} Nm")
    print(f"Throttle Torque: {throttle_torque:.2f} Nm")
    print("torque:", torque)


def acceleration_calculation():
    global corrected_torque, wheel_radius, mass, accel, rpm
    accel = (corrected_torque * wheel_radius) / (mass * wheel_radius**2)
    print(f"Acceleration: {accel:.2f} m/s²")
    

def accel_total():
    global accel, neg_acceleration, total_accel
    total_accel = accel + neg_acceleration
    print(f"Total acceleration: {total_accel:.2f} m/s²")
    
def speed_update():
    global init_speed_ms, final_speed_ms, total_accel, time_step
    final_speed_ms = init_speed_ms + (total_accel / 10)
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
    print("new rpm:", rpm)
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

def braking_decel():
    global brake_decel, brake_force, brake_pedal_pos



def main():
    global hp, torque, distance_covered, time, idle_rpm, accel, gear
    global rpm, init_speed_ms, final_speed_ms, has_started, on_idle, dist
    coords = read_coords('eme_straight.csv')
    dist = total_distance(coords) * 1000
    rpm_round()
    get_hp()
    get_torque()
    #get_throttle_from_csv()
    throttle_calc()
    #idle_check()
    #begin_check(on_idle)
    #init_speed_ms = final_speed_ms 
    #if on_idle == False:
    speed_calc(rpm, gear)
    drag_effect()
    torque_gear_ratio_calculation(gear)
    deceleration_calculation()
    acceleration_calculation()
    accel_total()
    speed_update()
    rpm = final_speed_to_rpm(gear)
    gear = gear_change(gear)
    rpm_round()
    new_distance = distance_per_time()
    distance_covered = distance_covered + new_distance
    print("RPM:", rpm)    

            

            
  #  if on_idle == True:
  #      accel = 0
  #      init_speed_ms = final_speed_ms
  #      rpm_round()
  #      drag_effect()
  #      deceleration_calculation()
  #      accel_total()
  #      speed_update()
  #      gear = gear_change(gear)
  #      rpm_round()
  #      new_distance = distance_per_time()
  #      distance_covered = distance_covered + new_distance
  #      print("RPM:", rpm, "time:", time)
  #      print(f"Distance traveled: {distance_covered:.2f} meters")
       


    print(f"Distance traveled: {distance_covered:.2f} meters")
    time += time_step
    print("time step:", time_step)
    time_round() 
    print("time:", time) # Increment time by the time step
    #    speed_calc(rpm)
    #    drag_effect()
    #    friction_calculation()



def time_round():
    global time
    time = round(time, 1)  # Round time to 1 decimal place  

rpm = idle_rpm
gear = 1

main()

while has_ended == 0:
    time_round()
    has_ended_basic()
    if has_ended == 0:
        main()
        
    else:
        print("Simulation has ended.")
    
        break
# RESULTS STILL REQUIRE FURTHER TESTING AND VALIDATION
