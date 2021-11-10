import numpy as np
import matplotlib.pyplot as plt
import math

def read_vertex():
    with open(r"teapot.obj", "r") as f:
        vertex = [[ch for ch in line.split(' ')] for line in f if line[0] == 'v']


    for i in vertex:
        i.pop(0)
        i.pop(2)


    for i in range(len(vertex)):
        vertex[i][-1] = vertex[i][-1].strip()

    for i in vertex:
        for j in range(2):
            i[j] = float(i[j])
    vertex = np.array(vertex)
    return vertex

def read_edges():
    with open(r"teapot.obj", "r") as f:
        facet = [[ch for ch in line.split(' ')] for line in f if line[0] == 'f']

    for i in facet:
        i.pop(0)

    for i in range(len(facet)):
        facet[i][-1] = facet[i][-1].strip()

    edges = []
    for i in facet:
        v1, v2, v3 = i
        edges.append([int(v1), int(v2)])
        edges.append([int(v1), int(v3)])
        edges.append([int(v2), int(v3)])

    fset = (set(frozenset(x) for x in edges))
    edges = [list(x) for x in fset]
    return edges

def draw_pix(x,y):
    base_color = [0,255,255]
    d = (((y - N / 2) ** 2 + (x - N / 2) ** 2) ** 0.5)
    color = [int(base_color[0] * (1 - d / N)), int(base_color[1] * (1 - d / N)), int( base_color[2]* (1 - d / N))]
    img[-x][-y] = color

def bresenham(x0, y0, x1, y1, img):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if x0 < x1:
            sign_x = 1
        else:
            sign_x = -1
        if y0 < y1:
            sign_y = 1
        else:
            sign_y = -1

        error = dx - dy
        draw_pix(x1, y1)
        while (x0 != x1 or y0 != y1):
            draw_pix(x0, y0)
            error_2 = error * 2
            if error_2 > -dy:
                error -= dy
                x0 += sign_x
            if error_2 < dx:
                error += dx
                y0 += sign_y

vertex = read_vertex()

edges = read_edges()

N = 2000

img = np.full((N, N, 3), 255, dtype=np.uint8)

a = np.array([[0,-1],[1,0]])
vertex = vertex @ a
vertex_scale = vertex @ (np.eye(2) * 270)

for i in range(vertex_scale.shape[0]):
    vertex_scale[i,0] += N // 2
    vertex_scale[i,1] += N // 2

for i in edges:
    x0, y0 = int(vertex_scale[i[0]-1][0]), int(vertex_scale[i[0]-1][1])
    x1, y1 = int(vertex_scale[i[1]-1][0]), int(vertex_scale[i[1]-1][1])
    bresenham(x0, y0, x1, y1, img)


plt.figure()
plt.imshow(img)
plt.imsave("teapot.png", img)
