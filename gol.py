import pygame
import pygame.sndarray
import numpy as np
import sys
from collections import Counter

# ----- Configuration -----
WIDTH, HEIGHT = 800, 600   # Window dimensions in pixels
CELL_SIZE = 10             # Base cell size in pixels

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

# A simple C-major scale (middle C to the next C) in Hz:
MUSICAL_SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

# ----- Predefined Patterns -----
predefined_patterns = [
    ("Glider", [(-1, 0), (0, 1), (1, -1), (1, 0), (1, 1)]),
    ("Blinker", [(0, -1), (0, 0), (0, 1)]),
    ("Toad", [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]),
    ("Beacon", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]),
    ("Pulsar", [
        (-6, -4), (-6, -3), (-6, -2), (-6, 2), (-6, 3), (-6, 4),
        (-1, -4), (-1, -3), (-1, -2), (-1, 2), (-1, 3), (-1, 4),
        (1, -4), (1, -3), (1, -2), (1, 2), (1, 3), (1, 4),
        (6, -4), (6, -3), (6, -2), (6, 2), (6, 3), (6, 4),
        (-4, -6), (-3, -6), (-2, -6), (2, -6), (3, -6), (4, -6),
        (-4, -1), (-3, -1), (-2, -1), (2, -1), (3, -1), (4, -1),
        (-4, 1), (-3, 1), (-2, 1), (2, 1), (3, 1), (4, 1),
        (-4, 6), (-3, 6), (-2, 6), (2, 6), (3, 6), (4, 6),
    ]),
    ("R-pentomino", [
        (0, 1), (0, 2),
        (1, 0), (1, 1),
        (2, 1)
    ]),
    ("Acorn", [
        (0, 1),
        (1, 3),
        (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)
    ]),
    ("Diehard", [
        (0, 6),
        (1, 0), (1, 1),
        (2, 1), (2, 5), (2, 6), (2, 7)
    ]),
    ("Gosper Glider Gun", [
        (-4, 7),
        (-3, 5), (-3, 7),
        (-2, -5), (-2, -4), (-2, 3), (-2, 4), (-2, 17), (-2, 18),
        (-1, -6), (-1, -2), (-1, 3), (-1, 4), (-1, 17), (-1, 18),
        (0, -17), (0, -16), (0, -7), (0, -1), (0, 3), (0, 4),
        (1, -17), (1, -16), (1, -7), (1, -3), (1, -1), (1, 0), (1, 5), (1, 7),
        (2, -7), (2, -1), (2, 7),
        (3, -6), (3, -2),
        (4, -5), (4, -4)
    ]),
    ("Kok's Galaxy", [
        (-4, 1), (-4, 2),
        (-3, 0), (-3, 3),
        (-2, -1), (-2, 4),
        (-1, -1), (-1, 4),
        (0, -1),  (0, 4),
        (1, 0),   (1, 3),
        (2, 1),   (2, 2)
    ]),
    ("Clock", [
        (-1, 0), (-1, 1),
        (0, -1), (0, 2),
        (1, -1), (1, 2),
        (2, 0),  (2, 1)
    ]),
    ("Tumbler", [
        (-4, -1),
        (-3, -1), (-3, 0),
        (-2, -3), (-2, -1),
        (-1, -2), (-1, 1),
        (0, -2), (0, -1), (0, 0), (0, 2),
        (1, -3), (1, -2), (1, 1),
        (2, -1), (2, 0),
        (3, -1)
    ]),
    ("Pinwheel", [
        (0, 4), (0, 5), (0, 6),
        (1, 3), (1, 5),
        (2, 2), (2, 4), (2, 6),
        (3, 1), (3, 3), (3, 5), (3, 7),
        (4, 1), (4, 3), (4, 5), (4, 7),
        (5, 2), (5, 4), (5, 6),
        (6, 3), (6, 5),
        (7, 4), (7, 5), (7, 6)
    ]),
    ("Pentadecathlon", [
        (0,1),
        (1,0),(1,1),(1,2),
        (2,0),(2,2),
        (3,1),
        (4,1),
        (5,0),(5,2),
        (6,0),(6,1),(6,2),
        (7,1)
    ]),
    ("Queen Bee Shuttle", [
        (0,7), (0,8),
        (1,6), (1,9),
        (2,2), (2,6), (2,9),
        (3,2), (3,4), (3,8),
        (4,2), (4,4)
    ]),
    ("Block-laying Switch Engine Predecessor", [
        (0, 11), (0, 13),
        (1, 0),  (1, 1),  (1, 10),
        (2, 0),  (2, 1),  (2, 11), (2, 14),
        (3, 13), (3, 14), (3, 15),
    ]),
    ("Merzenich's p11", [
        (11,0),(12,0),
        (12,1),
        (12,2),(14,2),
        (10,3),(11,3),(13,3),(15,3),
        (9,4),(11,4),(13,4),(15,4),
        (8,5),(10,5),(13,5),(16,5),(17,5),
        (7,6),(13,6),(18,6),
        (6,7),(14,7),(15,7),(16,7),(17,7),
        (5,8),(18,8),(19,8),(20,8),
        (4,9),(10,9),(16,9),(17,9),(20,9),
        (3,10),(5,10),(9,10),(11,10),(15,10),(17,10),
        (0,11),(3,11),(4,11),(10,11),(16,11),
        (0,12),(1,12),(2,12),(15,12),
        (3,13),(4,13),(5,13),(6,13),(14,13),
        (2,14),(7,14),(13,14),
        (3,15),(4,15),(7,15),(10,15),(12,15),
        (5,16),(7,16),(9,16),(11,16),
        (5,17),(7,17),(9,17),(10,17),
        (6,18),(8,18),
        (8,19),
        (8,20),(9,20),
    ]),
    ("Pufferfish", [
        (3,0), (11,0),
        (2,1), (3,1), (4,1), (10,1), (11,1), (12,1),
        (1,2), (2,2), (5,2), (9,2), (12,2), (13,2),
        (3,3), (4,3), (5,3), (9,3), (10,3), (11,3),
        # row 4: none
        (4,5), (10,5),
        (2,6), (5,6), (9,6), (12,6),
        (0,7), (6,7), (8,7), (14,7),
        (0,8), (1,8), (6,8), (8,8), (13,8), (14,8),
        (6,9), (8,9),
        (3,10), (5,10), (9,10), (11,10),
        (4,11), (10,11),
    ])
]

# View Control
zoom = 1.0
pan_x = 0
pan_y = 0

# Instead of a 2D NumPy array, use a set of (y, x) for live cells
live_cells = set()

# ----- Pygame Init -----
pygame.mixer.pre_init(44100, -16, 2, 128)
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Conway's Game of Life - Scroll Wheel Zoom")
clock = pygame.time.Clock()

# Font
font = pygame.font.SysFont(None, 24)

# Simulation flags / variables
running_sim = False
speed = 10  # (generations per second)
current_pattern_index = 0

# For smoother panning when keys are held
keys_held = {
    pygame.K_w: False,
    pygame.K_a: False,
    pygame.K_s: False,
    pygame.K_d: False
}

# For line-drawing while dragging
last_left_cell = None
last_right_cell = None

# Time accumulation for controlled simulation steps
simulation_timer = 0.0
MAX_FPS = 60  # Render frames per second (max)


def generate_tone_sound(frequency=440, duration=0.05, volume=4096):
    """
    Generate a short tone (sine wave) with a brief fade-in/fade-out.
    """
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    fade_in_out = 0.005
    fade_samples = int(sample_rate * fade_in_out)

    buf = np.zeros((n_samples, 2), dtype=np.int16)

    for s in range(n_samples):
        t = float(s) / sample_rate
        val = volume * np.sin(2.0 * np.pi * frequency * t)

        # Fade factor
        if s < fade_samples:
            factor = s / fade_samples
        elif s > n_samples - fade_samples:
            distance_from_end = n_samples - s
            factor = distance_from_end / fade_samples
        else:
            factor = 1.0

        val *= factor
        buf[s][0] = int(val)
        buf[s][1] = int(val)

    return pygame.sndarray.make_sound(buf)


def apply_pattern(offsets):
    """
    Clear current cells and add the chosen pattern around screen center.
    """
    global live_cells
    live_cells = set()

    scaled_cell = int(CELL_SIZE * zoom)
    center_in_grid_x = ((WIDTH // 2) - pan_x) // scaled_cell
    center_in_grid_y = ((HEIGHT // 2) - pan_y) // scaled_cell

    for (dy, dx) in offsets:
        y = center_in_grid_y + dy
        x = center_in_grid_x + dx
        live_cells.add((y, x))


def update_live_cells(live_cells_set):
    """
    Standard Conway update. Return (new_live, births_count, deaths_count).
    """
    neighbor_count = Counter()
    for (y, x) in live_cells_set:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    neighbor_count[(y + dy, x + dx)] += 1

    new_live = set()
    old_live = live_cells_set
    for (y, x) in old_live:
        n = neighbor_count.get((y, x), 0)
        if n in (2, 3):
            new_live.add((y, x))

    for cell, count in neighbor_count.items():
        if count == 3 and cell not in old_live:
            new_live.add(cell)

    births_count = len(new_live - old_live)
    deaths_count = len(old_live - new_live)
    return new_live, births_count, deaths_count


def bresenham_line(x0, y0, x1, y1):
    """
    Return a list of integer points on the line from (x0, y0) -> (x1, y1).
    """
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        points.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def draw_line_of_cells(cell_start, cell_end, add_cells=True):
    """
    Draw or erase cells along a line from cell_start->cell_end (row,col).
    """
    global live_cells
    (r0, c0) = cell_start
    (r1, c1) = cell_end
    line_cells = bresenham_line(c0, r0, c1, r1)
    for (r, c) in line_cells:
        if add_cells:
            live_cells.add((r, c))
        else:
            live_cells.discard((r, c))


def zoom_in(amount=0.1):
    """
    Zoom in around screen center by `amount`.
    Negative `amount` => zoom out.
    """
    global zoom, pan_x, pan_y
    current_scaled = CELL_SIZE * zoom
    center_world_x = ((WIDTH // 2) - pan_x) / current_scaled
    center_world_y = ((HEIGHT // 2) - pan_y) / current_scaled

    zoom += amount
    zoom = max(0.2, zoom)  # keep from going too small

    new_scaled = CELL_SIZE * zoom
    # Recompute pan so that same "world center" stays at screen center
    pan_x = int((WIDTH // 2) - (center_world_x * new_scaled))
    pan_y = int((HEIGHT // 2) - (center_world_y * new_scaled))


def zoom_out(amount=0.1):
    """
    Zoom out around screen center by `amount`.
    """
    zoom_in(-amount)


# Apply initial pattern
apply_pattern(predefined_patterns[current_pattern_index][1])

while True:
    # TIME STEP
    dt_ms = clock.tick(MAX_FPS)
    dt = dt_ms / 1000.0
    simulation_timer += dt

    # EVENT HANDLING
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            # Movement keys
            if event.key in keys_held:
                keys_held[event.key] = True

            if event.key == pygame.K_SPACE:
                # Start or pause the simulation
                if not running_sim:
                    simulation_timer = 0.0
                running_sim = not running_sim

            elif event.key == pygame.K_c:
                live_cells.clear()

            elif event.key == pygame.K_LEFT:
                current_pattern_index = (current_pattern_index - 1) % len(predefined_patterns)
                apply_pattern(predefined_patterns[current_pattern_index][1])
            elif event.key == pygame.K_RIGHT:
                current_pattern_index = (current_pattern_index + 1) % len(predefined_patterns)
                apply_pattern(predefined_patterns[current_pattern_index][1])

            elif event.key == pygame.K_UP:
                speed += 1
            elif event.key == pygame.K_DOWN:
                speed = max(1, speed - 1)

            # We can still keep z/x for fallback zoom:
            elif event.key == pygame.K_z:
                zoom_in(0.1)
            elif event.key == pygame.K_x:
                zoom_out(0.1)

        elif event.type == pygame.KEYUP:
            if event.key in keys_held:
                keys_held[event.key] = False

        # MOUSE WHEEL => ZOOM
        elif event.type == pygame.MOUSEWHEEL:
            # event.y > 0 => up => zoom in; event.y < 0 => down => zoom out
            # Each 'click' of the wheel typically has event.y=1 or -1, but might be more on some setups
            zoom_in(0.1 * event.y)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            scaled_cell = int(CELL_SIZE * zoom)
            gx = (mx - pan_x) // scaled_cell
            gy = (my - pan_y) // scaled_cell

            if event.button == 1:
                last_left_cell = (gy, gx)
                draw_line_of_cells((gy, gx), (gy, gx), add_cells=True)
            elif event.button == 3:
                last_right_cell = (gy, gx)
                draw_line_of_cells((gy, gx), (gy, gx), add_cells=False)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                last_left_cell = None
            elif event.button == 3:
                last_right_cell = None

        elif event.type == pygame.MOUSEMOTION:
            mx, my = pygame.mouse.get_pos()
            scaled_cell = int(CELL_SIZE * zoom)
            gx = (mx - pan_x) // scaled_cell
            gy = (my - pan_y) // scaled_cell
            buttons = pygame.mouse.get_pressed(num_buttons=3)
            if buttons[0]:  # left
                if last_left_cell is not None:
                    draw_line_of_cells(last_left_cell, (gy, gx), add_cells=True)
                last_left_cell = (gy, gx)
            elif buttons[2]:  # right
                if last_right_cell is not None:
                    draw_line_of_cells(last_right_cell, (gy, gx), add_cells=False)
                last_right_cell = (gy, gx)

    # CONTINUOUS KEY MOVEMENT
    pan_speed = 5
    if keys_held[pygame.K_w]:
        pan_y += pan_speed
    if keys_held[pygame.K_s]:
        pan_y -= pan_speed
    if keys_held[pygame.K_a]:
        pan_x += pan_speed
    if keys_held[pygame.K_d]:
        pan_x -= pan_speed

    # SIMULATION UPDATE
    if running_sim and speed > 0:
        step_time = 1.0 / speed
        births_sum = 0
        deaths_sum = 0
        while simulation_timer >= step_time:
            simulation_timer -= step_time
            new_cells, births_count, deaths_count = update_live_cells(live_cells)
            live_cells = new_cells
            births_sum += births_count
            deaths_sum += deaths_count

        # Play sounds once per frame
        if births_sum > 0:
            freq_for_births = MUSICAL_SCALE[births_sum % len(MUSICAL_SCALE)]
            sound_births = generate_tone_sound(frequency=freq_for_births, duration=0.05)
            sound_births.play()
        if deaths_sum > 0:
            freq_for_deaths = MUSICAL_SCALE[deaths_sum % len(MUSICAL_SCALE)]
            sound_deaths = generate_tone_sound(frequency=freq_for_deaths, duration=0.05)
            sound_deaths.play()

    # RENDER
    screen.fill(BLACK)
    scaled_cell = int(CELL_SIZE * zoom)

    for (cy, cx) in live_cells:
        screen_x = cx * scaled_cell + pan_x
        screen_y = cy * scaled_cell + pan_y
        if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
            rect = pygame.Rect(screen_x, screen_y, scaled_cell - 1, scaled_cell - 1)
            pygame.draw.rect(screen, GREEN, rect)

    # STATUS INFO (TOP-LEFT)
    pattern_name = predefined_patterns[current_pattern_index][0]
    status_text = "RUNNING" if running_sim else "PAUSED"
    status_lines = [
        f"Pattern: {pattern_name}",
        f"Status: {status_text}",
        f"Speed: {speed} gen/sec",
    ]
    y_off = 10
    for line in status_lines:
        txt_surf = font.render(line, True, TEXT_COLOR)
        screen.blit(txt_surf, (10, y_off))
        y_off += 20

    # CONTROLS INFO (TOP-RIGHT)
    controls_lines = [
        "Controls:",
        "[W,A,S,D]: Pan",
        "Mouse Wheel: Zoom in/out",
        "[Up/Down]: Speed +/-",
        "[Left/Right]: Patterns",
        "[Space]: Run/Pause",
        "[C]: Clear",
        "[Left Drag]: Draw cells",
        "[Right Drag]: Erase cells",
    ]
    y_off = 10
    for line in controls_lines:
        txt_surf = font.render(line, True, TEXT_COLOR)
        w_txt = txt_surf.get_width()
        screen.blit(txt_surf, (WIDTH - 10 - w_txt, y_off))
        y_off += 20

    pygame.display.flip()
