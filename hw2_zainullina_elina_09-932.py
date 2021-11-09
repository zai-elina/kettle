import numpy as np
import matplotlib.pyplot as plt

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

def read_gran():
    with open(r"teapot.obj", "r") as f:
        gran = [[ch for ch in line.split(' ')] for line in f if line[0] == 'f']

    for i in gran:
        i.pop(0)

    for i in range(len(gran)):
        gran[i][-1] = gran[i][-1].strip()

    for i in gran:
        for j in range(3):
            i[j] = int(i[j])
    return gran

vertex = read_vertex()
gran = read_gran()

N = 2000

center = [N // 2, N // 2]

background = np.full((N, N, 3), 0, dtype=np.uint8)
vertex_scale = vertex @ (np.eye(2) * 250) #маштабирование
centre = [N // 2, N // 2]


for i in range(vertex_scale.shape[0]):
    vertex_scale[i,0] += centre[0]
    vertex_scale[i,1] += centre[1]

print(vertex_scale)