import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

track_width = 0.00014
# Load the CSV file  
data = pd.read_csv('smoothed_lusail.csv')  # Replace with your file name  
  
# Extract columns  
t_long = data['t-long'].values  
t_lat = data['t-lat'].values  
tn_long = data['tn-long'].values  
tn_lat = data['tn-lat'].values  
  
# Track width (adjust as needed)  
 
  
# Initialize lists for inner and outer walls  
inner_wall_long = []  
inner_wall_lat = []  
outer_wall_long = []  
outer_wall_lat = []  
  
# Loop through each point and calculate walls  
for i in range(1, len(t_long)):  
    # Compute direction vector (dx, dy)  
    dx = t_long[i] - t_long[i - 1]  
    dy = t_lat[i] - t_lat[i - 1]  
    norm = np.sqrt(dx**2 + dy**2)  
      
    # Normalize direction vector  
    dx /= norm  
    dy /= norm  
  
    # Perpendicular vector (rotated by 90 degrees)  
    perp_dx = -dy  
    perp_dy = dx  
  
    # Inner wall points  
    inner_long = t_long[i] - perp_dx * (track_width / 2)  
    inner_lat = t_lat[i] - perp_dy * (track_width / 2)  
    inner_wall_long.append(inner_long)  
    inner_wall_lat.append(inner_lat)  
  
    # Outer wall points  
    outer_long = t_long[i] + perp_dx * (track_width / 2)  
    outer_lat = t_lat[i] + perp_dy * (track_width / 2)  
    outer_wall_long.append(outer_long)  
    outer_wall_lat.append(outer_lat)  
  
# Append the first points to close the loop  
inner_wall_long.append(inner_wall_long[0])  
inner_wall_lat.append(inner_wall_lat[0])  
outer_wall_long.append(outer_wall_long[0])  
outer_wall_lat.append(outer_wall_lat[0])  
  
# Generate segmentation lines  
segmentation_lines = []  
n_segments = 2 # Number of segments between inner and outer walls  
for i in range(len(inner_wall_long) - 1):  # Adjusted to avoid the last point  
    seg_long = np.linspace(inner_wall_long[i], outer_wall_long[i], n_segments)  
    seg_lat = np.linspace(inner_wall_lat[i], outer_wall_lat[i], n_segments)  
    segmentation_lines.append((seg_long, seg_lat))

# Save to CSV
track_data = pd.DataFrame({
    'tn-long': tn_long,
    'tn-lat': tn_lat,
    'inner-long': inner_wall_long,
    'inner-lat': inner_wall_lat,
    'outer-long': outer_wall_long,
    'outer-lat': outer_wall_lat
})
track_data.to_csv('final-track-lusail.csv', index=False)
  
# Visualization  
plt.figure(figsize=(40, 40))  
plt.plot(t_long, t_lat, '-', label='Smoothed Centerline', color='blue')  
plt.plot(inner_wall_long, inner_wall_lat, '-', label='Inner Wall', color='green')  
plt.plot(outer_wall_long, outer_wall_lat, '-', label='Outer Wall', color='red')  
  
# Plot segmentation lines  
for seg_long, seg_lat in segmentation_lines:  
    plt.plot(seg_long, seg_lat, '--', color='black', linewidth=1)  
  
plt.title('Generated Track with Walls and Segmentation Lines')  
plt.xlabel('Longitude (t-long)')  
plt.ylabel('Latitude (t-lat)')  
plt.legend()  
plt.grid()  
plt.show()  