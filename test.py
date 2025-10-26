import numpy as np  
import pandas as pd  
import math  

# === SETTINGS ===
track_width = 0.00014  # Approx track width (deg equivalent, you can adjust)
deg_to_m = 111_320     # Approx conversion deg -> meters (at equator)
file_in = 'smoothed_lusail.csv'
file_out = 'final_track_combined.csv'

# === LOAD DATA ===
data = pd.read_csv(file_in)
t_long = data['t-long'].values
t_lat = data['t-lat'].values
tn_long = data['tn-long'].values
tn_lat = data['tn-lat'].values

# === ARRAYS FOR RESULTS ===
inner_long, inner_lat = [], []
outer_long, outer_lat = [], []
seg_inner_long, seg_inner_lat = [], []
seg_outer_long, seg_outer_lat = [], []
seg_heading_deg, seg_width_m = [], []

# === LOOP THROUGH TRACK ===
for i in range(1, len(t_long)):
    dx = t_long[i] - t_long[i - 1]
    dy = t_lat[i] - t_lat[i - 1]
    norm = np.sqrt(dx**2 + dy**2)
    if norm == 0:
        continue

    # Normalize direction
    dx /= norm
    dy /= norm

    # Heading angle (in degrees)
    heading = math.degrees(math.atan2(dy, dx))

    # Perpendicular vector (90° rotated)
    perp_dx = -dy
    perp_dy = dx

    # Inner wall
    inner_long_i = t_long[i] - perp_dx * (track_width / 2)
    inner_lat_i = t_lat[i] - perp_dy * (track_width / 2)
    inner_long.append(inner_long_i)
    inner_lat.append(inner_lat_i)

    # Outer wall
    outer_long_i = t_long[i] + perp_dx * (track_width / 2)
    outer_lat_i = t_lat[i] + perp_dy * (track_width / 2)
    outer_long.append(outer_long_i)
    outer_lat.append(outer_lat_i)

    # Segment endpoints
    seg_inner_long.append(inner_long_i)
    seg_inner_lat.append(inner_lat_i)
    seg_outer_long.append(outer_long_i)
    seg_outer_lat.append(outer_lat_i)

    # Convert track width (approx degrees → meters)
    width_m = track_width * deg_to_m
    seg_width_m.append(width_m)
    seg_heading_deg.append(heading)

# === CLOSE THE LOOP ===
inner_long.append(inner_long[0])
inner_lat.append(inner_lat[0])
outer_long.append(outer_long[0])
outer_lat.append(outer_lat[0])
seg_inner_long.append(seg_inner_long[0])
seg_inner_lat.append(seg_inner_lat[0])
seg_outer_long.append(seg_outer_long[0])
seg_outer_lat.append(seg_outer_lat[0])
seg_heading_deg.append(seg_heading_deg[0])
seg_width_m.append(seg_width_m[0])

# === COMBINE INTO ONE CSV ===
combined = pd.DataFrame({
    'center-long': t_long[:len(seg_inner_long)],
    'center-lat': t_lat[:len(seg_inner_lat)],
    'inner-long': inner_long,
    'inner-lat': inner_lat,
    'outer-long': outer_long,
    'outer-lat': outer_lat,
    'seg-inner-long': seg_inner_long,
    'seg-inner-lat': seg_inner_lat,
    'seg-outer-long': seg_outer_long,
    'seg-outer-lat': seg_outer_lat,
    'seg-heading-deg': seg_heading_deg,
    'seg-width-m': seg_width_m
})

combined.to_csv(file_out, index=False)
print(f"✅ Combined track data saved to '{file_out}'")
