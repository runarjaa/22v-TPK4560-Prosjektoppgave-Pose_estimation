import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from utils import *

def default_colour(c):
    if c == 'black':
        return np.array([1.0, 1.0, 1.0])

    if c == 'white':
        return np.array([0.0, 0.0, 0.0])
    
    if c == 'red':
        return np.array([1.0, 0.0, 0.0])

    if c == 'green':
        return np.array([0.0, 1.0, 0.0])

    if c == 'blue':
        return np.array([0.0, 0.0, 1.0])


def default_axis():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, 
            origin=[0, 0, 0]
        )

def default_arrow():
    return o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius = 0.02, 
            cone_radius = 0.03, 
            cylinder_height = 1.0, 
            cone_height = 0.1, 
            resolution = 20, 
            cylinder_split = 4, 
            cone_split = 1
        )

def single_point(pos ,color = False):
    point = o3d.geometry.TriangleMesh.create_sphere(
            radius = 0.02, 
            resolution = 20
        )
    point.translate(pos)
    if color != False:
        point.paint_uniform_color(default_colour(color))

    return point
