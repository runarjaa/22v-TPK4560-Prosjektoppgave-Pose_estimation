from cgi import test
import enum
import numpy as np
import open3d as o3d

from utils import *
from objects import *



def test():
    plane_1 = o3d.geometry.TriangleMesh.create_box(5,0.1,2, create_uv_map=True, map_texture_to_each_face=True)
    # plane_1.compute_triangle_normals()

    plane_2 = o3d.geometry.TriangleMesh.create_box(7,0.1,2, create_uv_map=True, map_texture_to_each_face=True)
    plane_2.translate(np.array([1,1,1]))
    color = np.array([0.5, 0.0, 0.0], dtype = np.float64).reshape((3,1))
    print(color.shape)
    plane_2.paint_uniform_color(color)
    # plane_2.compute_triangle_normals()

    sphere = single_point(np.array([1,1,1]), color = 'red')
    # sphere.compute_triangle_normals()


    geometries = []
    mat_1 = o3d.visualization.rendering.MaterialRecord()
    mat_1.base_color = np.array([1, 1, 1, .5])

    mat_2 = o3d.visualization.rendering.MaterialRecord()
    mat_2.base_color = np.array([0, 0, 0, .5])

    geometries.append({
                "name": "building_" + str(1) ,
                "geometry": plane_1,
                "material": mat_1,
                "group": "buildings"
            })
    geometries.append({
                "name": "building_" + str(2) ,
                "geometry": plane_2,
                "material": mat_2,
                "group": "buildings"
            })
    geometries.append({
                "name": "building_" + str(3) ,
                "geometry": sphere,
                "material": mat_2,
                "group": "buildings"
            })


    o3d.visualization.draw(geometries, show_skybox = False)

    # o3d.visualization.draw({'name': 'plane', 'geometry': plane, 'material': mat}, show_skybox = False)

    # plotting(geometries, mat)


def testprint():
    plane = o3d.geometry.TriangleMesh.create_box(5,0.1,2, create_uv_map=True, map_texture_to_each_face=True)
    plane.compute_triangle_normals()

    plane.paint_uniform_color(np.array([64,224,208])/255)

    sphere = single_point(np.array([1,1,1]), color = 'red')
    sphere.compute_triangle_normals()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = np.array([1, 1, 1, .5])

    o3d.visualization.draw({'name': 'plane', 'geometry': plane, 'material': mat}, show_skybox = False)


def testprint_2():
    geometry_list = []

    axis_1 = default_axis()

    arrow_1 = default_arrow()

    point_1 = single_point(np.array([1,1,1]), color = 'red')

    geometry_list.append(axis_1)
    geometry_list.append(arrow_1)
    geometry_list.append(point_1)


if __name__ == "__main__":

    print("\nHello, new run!\n")

    # o3d.visualization.gui.Application.instance.initialize()

    test()
