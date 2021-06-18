import os
import numpy as np
import cv2
from pathlib import Path

import pyrender
import trimesh


if __name__ == "__main__":
    scene = pyrender.Scene(
        ambient_light=np.array([0.15, 0.15, 0.15, 1.0])
    )

    camera_pose = np.eye(4)
    cam = pyrender.IntrinsicsCamera(fx=992, fy=992, cx=446, cy=446)
    cam_node = scene.add(cam, pose=camera_pose)

    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
                       innerConeAngle=np.pi/16*0.1, outerConeAngle=np.pi/6*0.1)
    point_l = pyrender.PointLight(color=np.ones(3), intensity=2.0)

    scene.add(direc_l, pose=camera_pose)
    scene.add(spot_l, pose=camera_pose)
    scene.add(point_l, pose=camera_pose)

    # light1 = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0)
    # scene.add(light1, pose=camera_pose)

    # light2 = pyrender.SpotLight(color=np.ones(3), intensity=2.0,
    #                             innerConeAngle=np.pi/16.0,
    #                             outerConeAngle=np.pi/6.0)
    # scene.add(light2, pose=camera_pose)

    r = pyrender.OffscreenRenderer(
        viewport_width=992, viewport_height=994
    )

    # for mesh_filename in Path("/home/rakesh/workspace/mesh_mvs/results/qualitative_eval/models").rglob("*.obj"):
    for mesh_filename in Path("/home/rakesh/workspace/mesh_mvs/results/gcn_levels").rglob("*.obj"):
        trimesh_mesh = trimesh.load(str(mesh_filename))
        trimesh.repair.fix_normals(trimesh_mesh, multibody=True)
        pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        mesh_node = scene.add(pyrender_mesh)

        rendered_color, rendered_depth = r.render(scene)
        rendered_color = cv2.cvtColor(rendered_color, cv2.COLOR_BGR2RGB)

        stem = '.'.join(str(mesh_filename).split('.')[:-1])
        # print(str(mesh_filename), stem)
        image_filename = stem + '.png'
        cv2.imwrite(image_filename, rendered_color)

        scene.remove_node(mesh_node)

    r.delete()
