import pygame
import math

pygame.init()
screen = pygame.display.set_mode((1000, 400))
clock = pygame.time.Clock()
pygame.display.set_caption("Car Gauge Cluster")

# Colors
BG_COLOR = (20, 20, 20)
GAUGE_COLOR = (200, 200, 200)
NEEDLE_COLOR = (255, 50, 50)

# Gauge settings
gauge_radius = 120
center_positions = [(200, 200), (500, 200), (800, 200)]  # positions for 3 gauges

# Ranges
speed_range = (0, 200)          # km/h
steering_range = (-90, 90)      # degrees
throttle_range = (0, 100)       # %

# Angle spans (for the needle sweeps)
angle_min, angle_max = -225, 45

def draw_gauge(center, value, value_range, label):
    # Draw circle outline
    pygame.draw.circle(screen, GAUGE_COLOR, center, gauge_radius, 3)

    # Map value to angle
    vmin, vmax = value_range
    angle = angle_min + (value - vmin) / (vmax - vmin) * (angle_max - angle_min)
    angle_rad = math.radians(angle)

    # Needle endpoint
    needle_len = gauge_radius - 20
    x = center[0] + needle_len * math.cos(angle_rad)
    y = center[1] + needle_len * math.sin(angle_rad)

    pygame.draw.line(screen, NEEDLE_COLOR, center, (x, y), 4)

    # Label + Value
    font = pygame.font.SysFont("Arial", 20)
    label_surface = font.render(f"{label}", True, (255, 255, 255))
    value_surface = font.render(f"{int(value)}", True, (255, 255, 100))
    screen.blit(label_surface, (center[0] - 40, center[1] + gauge_radius + 10))
    screen.blit(value_surface, (center[0] - 15, center[1] - 15))

running = True
t = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Simulated values (you can replace with real data later)
    speed = (math.sin(t * 0.02) + 1) * 100       # oscillates 0–200
    steering = math.sin(t * 0.05) * 90           # oscillates -90–90
    throttle = (math.cos(t * 0.03) + 1) * 50     # oscillates 0–100

    # Draw
    screen.fill(BG_COLOR)
    draw_gauge(center_positions[0], speed, speed_range, "Speed (km/h)")
    draw_gauge(center_positions[1], steering, steering_range, "Steering (°)")
    draw_gauge(center_positions[2], throttle, throttle_range, "Throttle (%)")

    pygame.display.flip()
    clock.tick(30)
    t += 1
a
pygame.quit()
