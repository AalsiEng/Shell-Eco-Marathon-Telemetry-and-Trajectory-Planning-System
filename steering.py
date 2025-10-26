# Racing Line Optimization for Shell Eco Marathon
# Based on Team Batavia's segmentation approach

# %% [markdown]
# # Racing Line Optimization using Track Segmentation
# 
# This notebook implements racing line optimization by:
# 1. Segmenting the track into straight and corner sections
# 2. Optimizing speed profiles for each segment
# 3. Calculating optimal racing line through corners
# 4. Minimizing energy consumption while maintaining target lap time

# %% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize, differential_evolution
from scipy.signal import savgol_filter
import csv
from math import radians, sin, cos, sqrt, atan2, degrees
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Track Data Loading and Preprocessing

# %% Load GPS Track Data
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def load_track_data(filename='final-track-lusail.csv'):
    """Load track GPS data from CSV"""
    try:
        df = pd.read_csv(filename)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coords = df[['latitude', 'longitude']].values
        else:
            # Assume first two columns are lat, lon
            coords = df.iloc[:, :2].values
        print(f"Loaded {len(coords)} waypoints from {filename}")
        return coords
    except FileNotFoundError:
        print(f"File {filename} not found. Creating example track...")
        # Create example track - figure-8 shape
        t = np.linspace(0, 2*np.pi, 100)
        lat_base, lon_base = 25.49, 51.45
        coords = np.column_stack([
            lat_base + 0.002 * np.sin(t),
            lon_base + 0.001 * np.sin(2*t)
        ])
        return coords

def gps_to_xy(coords):
    """Convert GPS coordinates to local XY (meters)"""
    reference_lat, reference_lon = coords[0]
    R = 6371000
    
    xy = []
    for lat, lon in coords:
        lat_ref_rad = radians(reference_lat)
        x = R * radians(lon - reference_lon) * cos(lat_ref_rad)
        y = R * radians(lat - reference_lat)
        xy.append([x, y])
    
    return np.array(xy)

def calculate_curvature(xy_points, window=5):
    """Calculate curvature at each point using finite differences"""
    xy_smooth = savgol_filter(xy_points, window_length=min(window, len(xy_points)//2*2+1), 
                               polyorder=3, axis=0)
    
    dx = np.gradient(xy_smooth[:, 0])
    dy = np.gradient(xy_smooth[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    
    return curvature

# Load and process track
coords = load_track_data('final-track-lusail.csv')
xy_track = gps_to_xy(coords)
curvature = calculate_curvature(xy_track)

# Calculate cumulative distance
distances = np.zeros(len(xy_track))
for i in range(1, len(xy_track)):
    distances[i] = distances[i-1] + np.linalg.norm(xy_track[i] - xy_track[i-1])

print(f"Track length: {distances[-1]:.2f} meters")

# %% Visualize Track
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Track shape
axes[0].plot(xy_track[:, 0], xy_track[:, 1], 'b-', linewidth=2)
axes[0].scatter(xy_track[0, 0], xy_track[0, 1], c='green', s=200, marker='o', 
                label='Start/Finish', zorder=5)
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].set_title('Track Layout')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axis('equal')

# Curvature profile
axes[1].plot(distances, curvature, 'r-', linewidth=2)
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Curvature (1/m)')
axes[1].set_title('Track Curvature Profile')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Track Segmentation (Batavia Approach)

# %% Segment Classification
@dataclass
class TrackSegment:
    """Track segment with classification and properties"""
    start_idx: int
    end_idx: int
    segment_type: str  # 'straight', 'corner', 'chicane'
    start_dist: float
    end_dist: float
    length: float
    avg_curvature: float
    max_curvature: float
    entry_speed: float = 0.0
    exit_speed: float = 0.0
    apex_speed: float = 0.0

def segment_track(xy_points, curvature, distances, 
                  curvature_threshold=0.01, min_segment_length=10.0):
    """
    Segment track into straights and corners (Batavia method)
    
    Args:
        xy_points: Track coordinates
        curvature: Curvature at each point
        distances: Cumulative distance
        curvature_threshold: Threshold to classify corners
        min_segment_length: Minimum segment length
    """
    segments = []
    in_corner = False
    segment_start = 0
    
    for i in range(1, len(curvature)):
        is_corner = curvature[i] > curvature_threshold
        
        # Transition detection
        if is_corner and not in_corner:
            # Start of corner
            if distances[i] - distances[segment_start] > min_segment_length:
                # Save previous straight
                segments.append(TrackSegment(
                    start_idx=segment_start,
                    end_idx=i-1,
                    segment_type='straight',
                    start_dist=distances[segment_start],
                    end_dist=distances[i-1],
                    length=distances[i-1] - distances[segment_start],
                    avg_curvature=np.mean(curvature[segment_start:i]),
                    max_curvature=np.max(curvature[segment_start:i])
                ))
            segment_start = i
            in_corner = True
            
        elif not is_corner and in_corner:
            # End of corner
            if distances[i] - distances[segment_start] > min_segment_length/2:
                # Save corner
                segments.append(TrackSegment(
                    start_idx=segment_start,
                    end_idx=i-1,
                    segment_type='corner',
                    start_dist=distances[segment_start],
                    end_dist=distances[i-1],
                    length=distances[i-1] - distances[segment_start],
                    avg_curvature=np.mean(curvature[segment_start:i]),
                    max_curvature=np.max(curvature[segment_start:i])
                ))
            segment_start = i
            in_corner = False
    
    # Close final segment
    if segment_start < len(curvature) - 1:
        segments.append(TrackSegment(
            start_idx=segment_start,
            end_idx=len(curvature)-1,
            segment_type='straight' if not in_corner else 'corner',
            start_dist=distances[segment_start],
            end_dist=distances[-1],
            length=distances[-1] - distances[segment_start],
            avg_curvature=np.mean(curvature[segment_start:]),
            max_curvature=np.max(curvature[segment_start:])
        ))
    
    # Detect chicanes (consecutive corners)
    for i in range(len(segments)-1):
        if segments[i].segment_type == 'corner' and segments[i+1].segment_type == 'corner':
            if segments[i+1].start_dist - segments[i].end_dist < 20:
                segments[i].segment_type = 'chicane'
                segments[i+1].segment_type = 'chicane'
    
    return segments

segments = segment_track(xy_track, curvature, distances)

print(f"\nTrack Segmentation Results:")
print(f"Total segments: {len(segments)}")
print(f"Straights: {sum(1 for s in segments if s.segment_type == 'straight')}")
print(f"Corners: {sum(1 for s in segments if s.segment_type == 'corner')}")
print(f"Chicanes: {sum(1 for s in segments if s.segment_type == 'chicane')}")

# Display segment details
segment_df = pd.DataFrame([{
    'Type': s.segment_type,
    'Length (m)': f"{s.length:.1f}",
    'Avg Curvature': f"{s.avg_curvature:.4f}",
    'Max Curvature': f"{s.max_curvature:.4f}"
} for s in segments])
print("\nSegment Details:")
print(segment_df.to_string(index=True))

# %% Visualize Segmentation
plt.figure(figsize=(16, 6))

# Plot track colored by segment type
colors = {'straight': 'green', 'corner': 'red', 'chicane': 'orange'}
for seg in segments:
    idx_range = range(seg.start_idx, seg.end_idx+1)
    plt.plot(xy_track[idx_range, 0], xy_track[idx_range, 1], 
             color=colors[seg.segment_type], linewidth=3, 
             label=seg.segment_type if seg == segments[0] or 
             seg.segment_type != segments[segments.index(seg)-1].segment_type else '')

plt.scatter(xy_track[0, 0], xy_track[0, 1], c='blue', s=300, marker='*', 
            label='Start/Finish', zorder=5, edgecolors='black')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Track Segmentation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Speed Optimization per Segment

# %% Vehicle Parameters
class VehicleParams:
    """Shell Eco Marathon vehicle parameters"""
    def __init__(self):
        # Physical parameters
        self.mass = 220.0  # kg
        self.frontal_area = 1.298  # m²
        self.drag_coeff = 0.327335
        self.rolling_resistance = 0.01
        self.g = 9.81
        
        # Performance limits
        self.max_accel = 2.0  # m/s²
        self.max_decel = 3.0  # m/s²
        self.max_lateral_accel = 1.5  # m/s² (for cornering)
        
        # Drivetrain
        self.drivetrain_efficiency = 0.65
        self.max_power = 5000  # W (approximate)
        
    def drag_force(self, v):
        """Calculate drag force at velocity v"""
        return 0.5 * self.drag_coeff * self.frontal_area * v**2 * 1.225
    
    def rolling_force(self):
        """Calculate rolling resistance force"""
        return self.rolling_resistance * self.mass * self.g
    
    def max_corner_speed(self, curvature):
        """Calculate maximum cornering speed based on lateral acceleration"""
        if curvature < 1e-6:
            return 100.0  # Effectively straight
        radius = 1.0 / curvature
        return min(sqrt(self.max_lateral_accel * radius), 30.0)
    
    def energy_consumption(self, v1, v2, distance):
        """Calculate energy consumption for speed change over distance"""
        # Kinetic energy change
        ke_change = 0.5 * self.mass * (v2**2 - v1**2)
        
        # Average speed
        v_avg = (v1 + v2) / 2
        
        # Drag and rolling resistance work
        drag_work = self.drag_force(v_avg) * distance
        rolling_work = self.rolling_force() * distance
        
        # Total energy (positive means consumption)
        if v2 > v1:
            # Accelerating
            total_energy = (ke_change + drag_work + rolling_work) / self.drivetrain_efficiency
        else:
            # Decelerating (regenerative braking not considered for eco marathon)
            total_energy = drag_work + rolling_work
        
        return total_energy

vehicle = VehicleParams()

# %% Calculate Speed Limits for Each Segment
def calculate_segment_speed_limits(segments, curvature, vehicle):
    """Calculate speed limits for each segment based on curvature"""
    for seg in segments:
        if seg.segment_type == 'straight':
            seg.apex_speed = 30.0  # Max speed for straights
        else:
            # Corner speed limited by lateral acceleration
            seg.apex_speed = vehicle.max_corner_speed(seg.max_curvature)
        
        seg.entry_speed = seg.apex_speed
        seg.exit_speed = seg.apex_speed
    
    return segments

segments = calculate_segment_speed_limits(segments, curvature, vehicle)

# Display speed limits
speed_df = pd.DataFrame([{
    'Segment': i,
    'Type': s.segment_type,
    'Length (m)': f"{s.length:.1f}",
    'Max Speed (m/s)': f"{s.apex_speed:.2f}",
    'Max Speed (km/h)': f"{s.apex_speed*3.6:.1f}"
} for i, s in enumerate(segments)])
print("\nSpeed Limits per Segment:")
print(speed_df.to_string(index=False))

# %% [markdown]
# ## 4. Racing Line Optimization

# %% Generate Optimal Racing Line through Corners
def optimize_corner_line(xy_segment, track_width=12.0, n_points=20):
    """
    Generate optimal racing line through corner using track width
    
    Strategy: Late apex for eco marathon (maximize exit speed)
    """
    # Fit spline through segment
    if len(xy_segment) < 4:
        return xy_segment
    
    tck, u = splprep([xy_segment[:, 0], xy_segment[:, 1]], s=0, k=min(3, len(xy_segment)-1))
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    
    # Calculate perpendicular offset for racing line
    # Use negative offset at entry, zero at apex, positive at exit (late apex)
    apex_idx = len(x_new) // 2
    offsets = np.zeros(len(x_new))
    
    # Entry: outside
    offsets[:apex_idx] = track_width * 0.5 * np.linspace(1, 0, apex_idx)
    # Exit: outside
    offsets[apex_idx:] = track_width * 0.5 * np.linspace(0, 1, len(x_new) - apex_idx)
    
    # Apply offsets perpendicular to track
    dx = np.gradient(x_new)
    dy = np.gradient(y_new)
    norm = np.sqrt(dx**2 + dy**2)
    
    # Perpendicular vector
    px = -dy / (norm + 1e-6)
    py = dx / (norm + 1e-6)
    
    x_racing = x_new + offsets * px
    y_racing = y_new + offsets * py
    
    return np.column_stack([x_racing, y_racing])

def generate_racing_line(xy_track, segments, track_width=3.0):
    """Generate complete racing line for entire track"""
    racing_line = []
    
    for seg in segments:
        xy_segment = xy_track[seg.start_idx:seg.end_idx+1]
        
        if seg.segment_type in ['corner', 'chicane']:
            # Optimize corner
            optimized = optimize_corner_line(xy_segment, track_width)
            racing_line.append(optimized)
        else:
            # Keep straight segments as-is
            racing_line.append(xy_segment)
    
    return np.vstack(racing_line)

racing_line = generate_racing_line(xy_track, segments, track_width=12.0)

# Visualize racing line
plt.figure(figsize=(14, 8))
plt.plot(xy_track[:, 0], xy_track[:, 1], 'b--', linewidth=2, alpha=0.5, label='Center Line')
plt.plot(racing_line[:, 0], racing_line[:, 1], 'r-', linewidth=3, label='Racing Line')

# Mark corners
for seg in segments:
    if seg.segment_type in ['corner', 'chicane']:
        xy_corner = xy_track[seg.start_idx:seg.end_idx+1]
        plt.scatter(xy_corner[:, 0], xy_corner[:, 1], c='orange', s=10, alpha=0.5)

plt.scatter(xy_track[0, 0], xy_track[0, 1], c='green', s=300, marker='*', 
            label='Start/Finish', zorder=5, edgecolors='black')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Optimized Racing Line')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Velocity Profile Optimization

# %% Forward-Backward Speed Optimization
def optimize_velocity_profile(segments, distances, vehicle, target_lap_time=None):
    """
    Optimize velocity profile using forward-backward pass
    Similar to Batavia's approach
    """
    n_points = len(distances)
    v_profile = np.zeros(n_points)
    
    # Initialize with segment speed limits
    for seg in segments:
        v_profile[seg.start_idx:seg.end_idx+1] = seg.apex_speed
    
    # Forward pass: acceleration constraints
    v_profile[0] = segments[0].entry_speed
    for i in range(1, n_points):
        ds = distances[i] - distances[i-1]
        
        # Maximum speed achievable with acceleration limit
        v_max_accel = sqrt(v_profile[i-1]**2 + 2 * vehicle.max_accel * ds)
        
        # Limit by segment speed and acceleration
        v_profile[i] = min(v_profile[i], v_max_accel)
    
    # Backward pass: braking constraints
    for i in range(n_points-2, -1, -1):
        ds = distances[i+1] - distances[i]
        
        # Maximum speed before braking point
        v_max_brake = sqrt(v_profile[i+1]**2 + 2 * vehicle.max_decel * ds)
        
        # Limit by braking capability
        v_profile[i] = min(v_profile[i], v_max_brake)
    
    # Calculate lap time
    lap_time = 0
    for i in range(1, n_points):
        ds = distances[i] - distances[i-1]
        v_avg = (v_profile[i] + v_profile[i-1]) / 2
        if v_avg > 0:
            lap_time += ds / v_avg
    
    # Calculate energy consumption
    total_energy = 0
    for i in range(1, n_points):
        ds = distances[i] - distances[i-1]
        energy = vehicle.energy_consumption(v_profile[i-1], v_profile[i], ds)
        total_energy += energy
    
    return v_profile, lap_time, total_energy

v_profile, lap_time, energy = optimize_velocity_profile(segments, distances, vehicle)

print(f"\nVelocity Profile Optimization Results:")
print(f"Lap time: {lap_time:.2f} seconds")
print(f"Energy consumption: {energy/1000:.2f} kJ")
print(f"Average speed: {distances[-1]/lap_time:.2f} m/s ({distances[-1]/lap_time*3.6:.1f} km/h)")

# %% Visualize Velocity Profile
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Velocity vs distance
axes[0].plot(distances, v_profile, 'b-', linewidth=2)
axes[0].set_xlabel('Distance (m)')
axes[0].set_ylabel('Velocity (m/s)')
axes[0].set_title('Optimized Velocity Profile')
axes[0].grid(True, alpha=0.3)

# Color code by segment type
for seg in segments:
    idx_range = range(seg.start_idx, min(seg.end_idx+1, len(distances)))
    if seg.segment_type == 'straight':
        axes[0].axvspan(distances[seg.start_idx], distances[min(seg.end_idx, len(distances)-1)], 
                       alpha=0.2, color='green')
    elif seg.segment_type == 'corner':
        axes[0].axvspan(distances[seg.start_idx], distances[min(seg.end_idx, len(distances)-1)], 
                       alpha=0.2, color='red')

# Acceleration profile
accel_profile = np.gradient(v_profile, distances)
axes[1].plot(distances, accel_profile, 'r-', linewidth=2)
axes[1].axhline(y=vehicle.max_accel, color='g', linestyle='--', label='Max Accel')
axes[1].axhline(y=-vehicle.max_decel, color='orange', linestyle='--', label='Max Decel')
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Acceleration (m/s²)')
axes[1].set_title('Acceleration Profile')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Multi-Lap Strategy Optimization

# %% Energy-Time Pareto Frontier
def optimize_strategy(segments, distances, vehicle, n_strategies=10):
    """
    Generate multiple strategies along energy-time Pareto frontier
    """
    strategies = []
    
    # Range of maximum speeds to test
    max_speeds = np.linspace(8, 20, n_strategies)
    
    for max_speed in max_speeds:
        # Adjust segment speeds
        temp_segments = []
        for seg in segments:
            temp_seg = TrackSegment(
                start_idx=seg.start_idx,
                end_idx=seg.end_idx,
                segment_type=seg.segment_type,
                start_dist=seg.start_dist,
                end_dist=seg.end_dist,
                length=seg.length,
                avg_curvature=seg.avg_curvature,
                max_curvature=seg.max_curvature
            )
            temp_seg.apex_speed = min(seg.apex_speed, max_speed)
            temp_seg.entry_speed = temp_seg.apex_speed
            temp_seg.exit_speed = temp_seg.apex_speed
            temp_segments.append(temp_seg)
        
        # Optimize velocity profile
        v_prof, lap_t, energy_cons = optimize_velocity_profile(temp_segments, distances, vehicle)
        
        strategies.append({
            'max_speed': max_speed,
            'lap_time': lap_t,
            'energy': energy_cons,
            'v_profile': v_prof
        })
    
    return strategies

strategies = optimize_strategy(segments, distances, vehicle, n_strategies=15)

# Plot Pareto frontier
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Energy vs Lap Time
lap_times = [s['lap_time'] for s in strategies]
energies = [s['energy']/1000 for s in strategies]

axes[0].plot(lap_times, energies, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Lap Time (s)')
axes[0].set_ylabel('Energy Consumption (kJ)')
axes[0].set_title('Energy-Time Pareto Frontier')
axes[0].grid(True, alpha=0.3)

# Annotate points
for i, s in enumerate(strategies[::3]):
    axes[0].annotate(f"{s['max_speed']:.1f} m/s", 
                    (s['lap_time'], s['energy']/1000),
                    xytext=(5, 5), textcoords='offset points')

# Velocity profiles comparison
for i, s in enumerate(strategies[::3]):
    axes[1].plot(distances, s['v_profile'], linewidth=2, 
                label=f"Max {s['max_speed']:.1f} m/s", alpha=0.7)

axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Velocity Profiles for Different Strategies')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Export Optimized Strategy

# %% Export Results
def export_racing_strategy(racing_line, v_profile, distances, segments, filename='racing_strategy.csv'):
    """Export racing line and velocity profile"""
    data = {
        'distance': distances,
        'x': racing_line[:, 0],
        'y': racing_line[:, 1],
        'velocity_ms': v_profile,
        'velocity_kmh': v_profile * 3.6,
    }
    
    # Add segment classification
    segment_type = np.empty(len(distances), dtype=object)
    for seg in segments:
        segment_type[seg.start_idx:seg.end_idx+1] = seg.segment_type
    data['segment_type'] = segment_type
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nRacing strategy exported to {filename}")
    print(f"Total points: {len(df)}")
    return df

strategy_df = export_racing_strategy(racing_line, v_profile, distances, segments)
print("\nFirst 10 rows:")
print(strategy_df.head(10))

# %% Summary Statistics
print("\n" + "="*60)
print("RACING LINE OPTIMIZATION SUMMARY")
print("="*60)
print(f"Track length: {distances[-1]:.2f} m")
print(f"Number of segments: {len(segments)}")
print(f"  - Straights: {sum(1 for s in segments if s.segment_type == 'straight')}")
print(f"  - Corners: {sum(1 for s in segments if s.segment_type == 'corner')}")
print(f"  - Chicanes: {sum(1 for s in segments if s.segment_type == 'chicane')}")
print(f"\nOptimal Strategy:")
print(f"  Lap time: {lap_time:.2f} s")
print(f"  Average speed: {distances[-1]/lap_time*3.6:.1f} km/h")
print(f"  Energy consumption: {energy/1000:.2f} kJ")
print(f"  Max speed: {np.max(v_profile)*3.6:.1f} km/h")
print(f"  Min speed: {np.min(v_profile)*3.6:.1f} km/h")
print("="*60)

# %% [markdown]
# ## 8. Compare with Alternative Strategies

# %% Strategy Comparison Table
comparison_df = pd.DataFrame([{
    'Strategy': f"Max {s['max_speed']:.1f} m/s",
    'Lap Time (s)': f"{s['lap_time']:.2f}",
    'Energy (kJ)': f"{s['energy']/1000:.2f}",
    'Avg Speed (km/h)': f"{distances[-1]/s['lap_time']*3.6:.1f}",
    'Energy/km (kJ)': f"{s['energy']/1000/(distances[-1]/1000):.2f}"
} for s in strategies])

print("\nStrategy Comparison:")
print(comparison_df.to_string(index=False))

# Find most efficient strategy
efficiencies = [s['energy']/1000/(distances[-1]/1000) for s in strategies]
most_efficient_idx = np.argmin(efficiencies)
print(f"\nMost energy efficient: Max {strategies[most_efficient_idx]['max_speed']:.1f} m/s")
print(f"  Energy: {strategies[most_efficient_idx]['energy']/1000:.2f} kJ")
print(f"  Lap time: {strategies[most_efficient_idx]['lap_time']:.2f} s")

# %% Final Visualization - Complete Dashboard
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Track with racing line colored by speed
ax1 = fig.add_subplot(gs[0, :2])
scatter = ax1.scatter(racing_line[:, 0], racing_line[:, 1], 
                     c=v_profile, cmap='RdYlGn', s=30, linewidths=0)
ax1.plot(xy_track[:, 0], xy_track[:, 1], 'k--', alpha=0.3, linewidth=1, label='Center Line')
ax1.scatter(racing_line[0, 0], racing_line[0, 1], c='blue', s=400, marker='*', 
           label='Start/Finish', zorder=5, edgecolors='black', linewidths=2)
cbar = plt.colorbar(scatter, ax=ax1, label='Speed (m/s)')
ax1.set_xlabel('X (m)', fontsize=11)
ax1.set_ylabel('Y (m)', fontsize=11)
ax1.set_title('Racing Line with Speed Profile', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. Velocity profile
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(distances, v_profile * 3.6, 'b-', linewidth=2)
ax2.fill_between(distances, 0, v_profile * 3.6, alpha=0.3)
ax2.set_xlabel('Distance (m)', fontsize=11)
ax2.set_ylabel('Speed (km/h)', fontsize=11)
ax2.set_title('Speed Profile', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Curvature analysis
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(distances, curvature, 'r-', linewidth=2)
ax3.fill_between(distances, 0, curvature, alpha=0.3, color='red')
ax3.set_xlabel('Distance (m)', fontsize=11)
ax3.set_ylabel('Curvature (1/m)', fontsize=11)
ax3.set_title('Track Curvature', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Energy consumption breakdown
ax4 = fig.add_subplot(gs[1, 1])
segment_energies = []
segment_labels = []
for i, seg in enumerate(segments):
    seg_energy = 0
    for j in range(seg.start_idx, min(seg.end_idx, len(v_profile)-1)):
        ds = distances[j+1] - distances[j]
        seg_energy += vehicle.energy_consumption(v_profile[j], v_profile[j+1], ds)
    segment_energies.append(seg_energy / 1000)
    segment_labels.append(f"{seg.segment_type[:4].title()}{i}")

colors_bar = ['green' if s.segment_type == 'straight' else 'red' if s.segment_type == 'corner' else 'orange' 
              for s in segments]
ax4.bar(range(len(segments)), segment_energies, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Segment', fontsize=11)
ax4.set_ylabel('Energy (kJ)', fontsize=11)
ax4.set_title('Energy per Segment', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(segments)))
ax4.set_xticklabels(segment_labels, rotation=45, ha='right', fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Pareto frontier
ax5 = fig.add_subplot(gs[1, 2])
lap_times_arr = np.array([s['lap_time'] for s in strategies])
energies_arr = np.array([s['energy']/1000 for s in strategies])
ax5.plot(lap_times_arr, energies_arr, 'bo-', linewidth=2, markersize=8)
ax5.scatter(lap_time, energy/1000, c='red', s=200, marker='*', 
           label='Selected', zorder=5, edgecolors='black', linewidths=2)
ax5.set_xlabel('Lap Time (s)', fontsize=11)
ax5.set_ylabel('Energy (kJ)', fontsize=11)
ax5.set_title('Strategy Tradeoff', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.suptitle('Shell Eco Marathon - Racing Line Optimization Dashboard', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Advanced Features - Regenerative Braking Optimization

# %% Regenerative Braking Analysis
def analyze_braking_zones(v_profile, distances, vehicle):
    """Identify and analyze braking zones for energy recovery"""
    braking_zones = []
    in_braking = False
    zone_start = 0
    
    accel = np.gradient(v_profile, distances)
    
    for i in range(1, len(accel)):
        if accel[i] < -0.1 and not in_braking:
            # Start of braking
            zone_start = i
            in_braking = True
        elif accel[i] >= -0.1 and in_braking:
            # End of braking
            braking_zones.append({
                'start_idx': zone_start,
                'end_idx': i,
                'start_dist': distances[zone_start],
                'end_dist': distances[i],
                'length': distances[i] - distances[zone_start],
                'speed_loss': v_profile[zone_start] - v_profile[i],
                'energy_recoverable': 0.5 * vehicle.mass * 
                    (v_profile[zone_start]**2 - v_profile[i]**2) * 0.5  # 50% efficiency
            })
            in_braking = False
    
    return braking_zones

braking_zones = analyze_braking_zones(v_profile, distances, vehicle)

print("\n" + "="*60)
print("BRAKING ZONES ANALYSIS")
print("="*60)
total_recoverable = sum(zone['energy_recoverable'] for zone in braking_zones)
print(f"Number of braking zones: {len(braking_zones)}")
print(f"Total recoverable energy: {total_recoverable/1000:.2f} kJ")
print(f"Percentage of total energy: {total_recoverable/energy*100:.1f}%")

if braking_zones:
    print("\nTop 5 Braking Zones by Energy:")
    sorted_zones = sorted(braking_zones, key=lambda x: x['energy_recoverable'], reverse=True)
    for i, zone in enumerate(sorted_zones[:5]):
        print(f"  {i+1}. Distance {zone['start_dist']:.1f}-{zone['end_dist']:.1f}m: "
              f"{zone['energy_recoverable']/1000:.2f} kJ, "
              f"Speed {v_profile[zone['start_idx']]*3.6:.1f}-{v_profile[zone['end_idx']]*3.6:.1f} km/h")

# %% [markdown]
# ## 10. Lap Time Simulation with Real Physics

# %% Detailed Lap Simulation
def simulate_lap(racing_line, v_profile, distances, vehicle, dt=0.1):
    """
    Detailed lap simulation with physics
    Returns time series data
    """
    simulation_data = []
    current_dist = 0
    current_time = 0
    current_speed = v_profile[0]
    
    # Interpolate velocity profile
    v_interp = interp1d(distances, v_profile, kind='linear', fill_value='extrapolate')
    
    while current_dist < distances[-1]:
        target_speed = v_interp(current_dist)
        
        # Calculate required acceleration
        speed_error = target_speed - current_speed
        if abs(speed_error) < 0.1:
            accel = 0
        else:
            accel = np.clip(speed_error / dt, -vehicle.max_decel, vehicle.max_accel)
        
        # Update speed
        current_speed += accel * dt
        current_speed = max(0.1, current_speed)
        
        # Calculate forces
        drag = vehicle.drag_force(current_speed)
        rolling = vehicle.rolling_force()
        
        # Power requirement
        if accel > 0:
            power = (vehicle.mass * accel + drag + rolling) * current_speed / vehicle.drivetrain_efficiency
        else:
            power = 0
        
        # Update distance
        current_dist += current_speed * dt
        current_time += dt
        
        simulation_data.append({
            'time': current_time,
            'distance': current_dist,
            'speed': current_speed,
            'acceleration': accel,
            'power': power,
            'drag_force': drag,
            'rolling_force': rolling
        })
    
    return pd.DataFrame(simulation_data)

sim_df = simulate_lap(racing_line, v_profile, distances, vehicle)

print("\n" + "="*60)
print("DETAILED LAP SIMULATION")
print("="*60)
print(f"Simulated lap time: {sim_df['time'].iloc[-1]:.2f} seconds")
print(f"Average power: {sim_df['power'].mean()/1000:.2f} kW")
print(f"Peak power: {sim_df['power'].max()/1000:.2f} kW")
print(f"Average speed: {sim_df['speed'].mean()*3.6:.1f} km/h")

# Plot detailed simulation
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Speed vs time
axes[0].plot(sim_df['time'], sim_df['speed']*3.6, 'b-', linewidth=2)
axes[0].set_ylabel('Speed (km/h)', fontsize=11)
axes[0].set_title('Detailed Lap Simulation', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Acceleration vs time
axes[1].plot(sim_df['time'], sim_df['acceleration'], 'r-', linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Acceleration (m/s²)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Power vs time
axes[2].plot(sim_df['time'], sim_df['power']/1000, 'g-', linewidth=2)
axes[2].fill_between(sim_df['time'], 0, sim_df['power']/1000, alpha=0.3, color='green')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Power (kW)', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Export Complete Telemetry

# %% Export Full Telemetry Data
def export_complete_telemetry(sim_df, racing_line, segments, filename='telemetry_data.csv'):
    """Export complete telemetry including position and simulation data"""
    # Interpolate position to match simulation
    from scipy.interpolate import interp1d
    
    x_interp = interp1d(distances, racing_line[:, 0], kind='linear', fill_value='extrapolate')
    y_interp = interp1d(distances, racing_line[:, 1], kind='linear', fill_value='extrapolate')
    
    sim_df['x'] = x_interp(sim_df['distance'])
    sim_df['y'] = y_interp(sim_df['distance'])
    
    # Add segment information
    def get_segment_type(dist):
        for seg in segments:
            if seg.start_dist <= dist <= seg.end_dist:
                return seg.segment_type
        return 'unknown'
    
    sim_df['segment_type'] = sim_df['distance'].apply(get_segment_type)
    
    # Calculate cumulative energy
    sim_df['energy_cumulative'] = (sim_df['power'] * 0.1).cumsum() / 1000  # kJ
    
    sim_df.to_csv(filename, index=False)
    print(f"\nComplete telemetry exported to {filename}")
    print(f"Total data points: {len(sim_df)}")
    print(f"Data frequency: {1/0.1:.0f} Hz")
    
    return sim_df

telemetry_df = export_complete_telemetry(sim_df, racing_line, segments)

# Display sample
print("\nTelemetry Sample (first 10 rows):")
print(telemetry_df[['time', 'distance', 'speed', 'power', 'segment_type']].head(10).to_string(index=False))

# %% [markdown]
# ## 12. Export Complete Telemetry

# %% [markdown]
# ## 13. Final Summary Report

# %% Generate Summary Report
print("\n" + "="*70)
print(" "*15 + "RACING LINE OPTIMIZATION - FINAL REPORT")
print("="*70)
print("\nTRACK INFORMATION:")
print(f"  Total length: {distances[-1]:.2f} m")
print(f"  Track width: 12.0 m")
print(f"  Number of waypoints: {len(xy_track)}")
print(f"  Number of segments: {len(segments)}")
print(f"    - Straights: {sum(1 for s in segments if s.segment_type == 'straight')} "
      f"({sum(s.length for s in segments if s.segment_type == 'straight'):.1f} m)")
print(f"    - Corners: {sum(1 for s in segments if s.segment_type == 'corner')} "
      f"({sum(s.length for s in segments if s.segment_type == 'corner'):.1f} m)")
print(f"    - Chicanes: {sum(1 for s in segments if s.segment_type == 'chicane')} "
      f"({sum(s.length for s in segments if s.segment_type == 'chicane'):.1f} m)")

print("\nOPTIMIZED STRATEGY:")
print(f"  Lap time: {lap_time:.2f} s")
print(f"  Average speed: {distances[-1]/lap_time*3.6:.1f} km/h")
print(f"  Maximum speed: {np.max(v_profile)*3.6:.1f} km/h")
print(f"  Minimum speed: {np.min(v_profile)*3.6:.1f} km/h")

print("\nENERGY ANALYSIS:")
print(f"  Total energy: {energy/1000:.2f} kJ")
print(f"  Energy per km: {energy/1000/(distances[-1]/1000):.2f} kJ/km")
print(f"  Potential regeneration: {total_recoverable/1000:.2f} kJ ({total_recoverable/energy*100:.1f}%)")
print(f"  Net energy: {(energy - total_recoverable)/1000:.2f} kJ")

print("\nPERFORMANCE METRICS:")
print(f"  Average acceleration: {sim_df['acceleration'].mean():.3f} m/s²")
print(f"  Max acceleration: {sim_df['acceleration'].max():.2f} m/s²")
print(f"  Max deceleration: {sim_df['acceleration'].min():.2f} m/s²")
print(f"  Average power: {sim_df['power'].mean()/1000:.2f} kW")
print(f"  Peak power: {sim_df['power'].max()/1000:.2f} kW")

print("\nFILES GENERATED:")
print(f"  1. racing_strategy.csv - Racing line and velocity profile")
print(f"  2. telemetry_data.csv - Complete telemetry data")

print("\nRECOMMENDATIONS:")
print(f"  • Focus on optimizing corners {[i for i, s in enumerate(segments) if s.segment_type=='corner' and s.max_curvature > 0.02]}")
print(f"  • Consider regenerative braking in {len(braking_zones)} braking zones")
print(f"  • Track width: 12.0m for racing line optimization")
print(f"  • Energy-time tradeoff: {len(strategies)} strategies available")

print("="*70)
print("\n✓ Optimization complete! Data exported and ready for vehicle implementation.")
print("="*70)