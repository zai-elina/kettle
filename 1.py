import numpy as np
import matplotlib.pyplot as plt


def parse_file(filename):
    with open(filename) as f:
        lines = f.read(-1).strip().split('\n')

    points = []
    rectangles = []

    for line in lines:
        if line.startswith('v'):
            points += [tuple(map(float, line.split(' ')[1:]))]
        elif line.startswith('f'):
            rectangles += [tuple(map(lambda x: int(x) - 1, line.split(' ')[1:]))]

    return np.array(points), rectangles


points, rectangles = parse_file('teapot.obj')

N = 1400


def canvas():
    return np.full((N, N, 3), 230, dtype=np.uint8)


center = N // 2, N // 2


def scale_matrix(factor):
    return np.eye(3) * factor


def mirror_matrix():
    matrix = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]])
    return matrix


def shift(matrix, dx, dy, dz):
    shift_matrix = np.repeat(np.array([[dx, dy, dz]]), matrix.shape[0], axis=0)
    return matrix + shift_matrix


def fill_pixel(canvas, x, y, color):
    try:
        canvas[y][x] = color
    except IndexError:
        pass


def distance(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] + point_b[1]) ** 2)


base_color = np.array([255, 0, 255])

precomputed_colors = np.tile(
    np.expand_dims(
        np.expand_dims(base_color, axis=0),
        axis=0),
    (N, N, 1))


def compute_colors():
    indices = np.indices((N, N))
    indices = indices.astype('float64')
    global precomputed_colors
    X = indices[0]
    Y = indices[1]
    distance = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mult = np.clip((1 - distance / N), 0, 1)
    mult = np.expand_dims(mult, axis=-1)
    precomputed_colors = precomputed_colors * mult
    precomputed_colors = precomputed_colors.astype(np.uint8)


compute_colors()


def get_color(x, y):
    # mult = distance((x,y), center) / N
    # color = np.array([255, 0, 255])
    # color = color * mult
    # color = np.clip(color, 0, 255)
    # return color.astype(np.uint8)
    return precomputed_colors[y][x]


from collections import namedtuple

transform_config = namedtuple("transform_config", ["invert_y", "swap"])


def normalize_points(start, stop):
    inv = transform_config(False, False)

    x_delta = start[0] - stop[0]
    y_delta = start[1] - stop[1]
    if abs(y_delta) > abs(x_delta):
        inv = transform_config(inv.invert_y, True)
        start = (start[1], start[0])
        stop = (stop[1], stop[0])

    if start[0] > stop[0]:
        start, stop = stop, start
    if start[1] > stop[1]:
        inv = transform_config(True, inv.swap)
        start = (start[0], -start[1])
        stop = (stop[0], -stop[1])

    return start, stop, inv


def transform_coordinates(x, y, inv):
    if inv.invert_y:
        y = -y
    if inv.swap:
        x, y = y, x
    return x, y


def draw_line_with_bresenham(canvas, start, stop):
    start, stop, inv = normalize_points(start, stop)

    start_x, start_y = start
    stop_x, stop_y = stop

    start_x, start_y, stop_x, stop_y = map(int, [start_x, start_y, stop_x, stop_y])

    x_delta = stop_x - start_x
    y_delta = stop_y - start_y
    D = 2 * y_delta - x_delta
    y = start_y

    for x in range(start_x, stop_x + 1):
        color = get_color(*transform_coordinates(x, y, inv))
        fill_pixel(canvas, *transform_coordinates(x, y, inv), color)
        if D > 0:
            y = y + 1
            D = D - 2 * x_delta
        D = D + 2 * y_delta


def draw_rectangle(canvas, a, b, c):
    draw_line_with_bresenham(canvas, a[:2], b[:2])
    draw_line_with_bresenham(canvas, a[:2], c[:2])
    draw_line_with_bresenham(canvas, b[:2], c[:2])


def draw_object(canvas, points, rectangles):
    for rectangle in rectangles:
        a, b, c = rectangle
        draw_rectangle(canvas, points[a], points[b], points[c])


canvas = canvas()
points = points @ mirror_matrix()
points = points @ scale_matrix(250)
points = shift(points, center[0], center[1], 0)
draw_object(canvas, points, rectangles)
plt.figure(figsize=(14, 14))
plt.imshow(canvas)
plt.imsave("teapot.png", canvas)