import sys

sys.path.append("src")

from color.color_util import web_colors
from color.color_util import xyz_to_srgb
from color.surfaces import SpectralCone
from geometry.catalog import smooth_surfaces
from geometry.discrete.util import triangulate_surface
from graphics.blender.util import look_at,delete_objects
from scipy.spatial.transform import Rotation
import bpy
import bpy
import mathutils
import numpy as np
import open3d as o3d

def make_material(rgba):
    material = bpy.data.materials.new(name="Diffuse")
    material.use_nodes = True
    material.node_tree.nodes.remove(material.node_tree.nodes.get('Principled BSDF')) #title of the existing node when materials.new
    material_output = material.node_tree.nodes.get('Material Output')
    diffuse = material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')    #name of diffuse BSDF when added with shift+A
    diffuse.inputs['Color'].default_value = rgba #last value alpha
    material.node_tree.links.new(material_output.inputs[0], diffuse.outputs[0])
    return material

def render(
    mesh_vertices,
    mesh_faces,
    output_file,
    output_resolution_px,
    light_location,
    camera_location,
    camera_lookat,
    mesh_colors=None):
    """
    Render a scene in blender
    """
    # get a fresh scene
    delete_objects(bpy.data.objects)

    # Get the scene
    scene = bpy.context.scene

    # Use the cycles renderer (default errors out)
    # https://www.cycles-renderer.org/
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Other rendering settings
    scene.render.sequencer_gl_preview = 'RENDERED'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_file
    scene.render.resolution_x = output_resolution_px
    scene.render.resolution_y = output_resolution_px
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    # Create a light
    light_data = bpy.data.lights.new('light', type='SUN')
    light_data.angle=1
    light_data.energy=5
    light = bpy.data.objects.new('light', light_data)
    scene.collection.objects.link(light)
    light.location = mathutils.Vector(light_location)
    look_at(light, mathutils.Vector((0,0,0)))

    # Create the camera
    cam_data = bpy.data.cameras.new('camera')
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 30.0
    cam = bpy.data.objects.new('camera', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = mathutils.Vector(camera_location)
    if camera_lookat:
        look_at(cam, mathutils.Vector(camera_lookat))

    # Add a floor
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0,0,-10))
    floor_material = make_material(web_colors["cyan"].rgba_01)
    bpy.data.objects['Plane'].data.materials.append(floor_material)

    # create mesh and object
    mesh = bpy.data.meshes.new("color")
    mesh.from_pydata(mesh_vertices,[],mesh_faces)
    mesh.update(calc_edges=True)
    ob = bpy.data.objects.new("color",mesh)
    scene.collection.objects.link(ob)

    materials_lookup = {}

    ob.data.materials.append(make_material(web_colors["pink"].rgba_01))

    bpy.ops.wm.save_as_mainfile(filepath="test.blend")

    # Finally, run the render
    bpy.ops.render.render(write_still = 1)

step = 150
surface = smooth_surfaces["torus"]
verts, faces = triangulate_surface(surface, step)

render(
    mesh_vertices=verts,
    mesh_faces=faces,
    light_location=(30,30,30),
    camera_location=(30,30,30),
    camera_lookat=(0,0,0),
    output_file="test.png",
    output_resolution_px=500
)
