import csv
from math import radians, sin, cos, sqrt, atan2
from math import radians, degrees, atan2, sin, cos

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

def calculate_bearing(lat1, lon1, lat2, lon2):     
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    return (bearing + 360) % 360
    
def read_coords(filename):
    coords = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            coords.append((lat, lon))
    return coords

def total_distance(coords):
    distance = 0.0
    for i in range(1, len(coords)):
        distance += haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
    return distance

if __name__ == "__main__":
    coords = read_coords('Lusail_Coords.csv')
    dist = total_distance(coords)
    bearing = calculate_bearing(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
    print(f"Initial bearing from first to second coordinate: {bearing:.2f} degrees")
    print(f"Total distance covered: {dist:.3f} km")