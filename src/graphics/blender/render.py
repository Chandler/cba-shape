import bpy
from graphics.blender.util import look_at
import bpy
import mathutils
from scipy.spatial.transform import Rotation
import numpy as np

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

    # Get the scene
    scene = bpy.context.scene

    # Remove the default cube from the scene
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)

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
    light = bpy.data.objects.new('light', light_data)
    scene.collection.objects.link(light)
    light.location = mathutils.Vector(light_location)

    # Create the camera
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = mathutils.Vector(camera_location)
    if camera_lookat:
        look_at(cam, mathutils.Vector(camera_lookat))

    # Add a floor
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0,0,-10))
    floor_material = make_material(cyan)
    bpy.data.objects['Plane'].data.materials.append(floor_material)

    # create mesh and object
    mesh = bpy.data.meshes.new("color")
    mesh.from_pydata(mesh_vertices,[],mesh_faces)
    mesh.update(calc_edges=True)
    ob = bpy.data.objects.new("color",mesh)
    scene.collection.objects.link(ob)

    materials_lookup = {}

    # Add materials to each polygon (TODO: improve)
    if mesh_colors is not None:
        for p in ob.data.polygons:
            color = list(mesh_colors[p.vertices[0]]) + [0]
            color_hash = str(color)
            if not (color_hash in materials_lookup):
                mat = bpy.data.materials.new(name='MaterialCBA{}'.format(color_hash))
                mat.diffuse_color = color
                ob.data.materials.append(mat)
                materials_lookup[color_hash] = len(ob.data.materials) - 1

            p.material_index = materials_lookup[color_hash]
    else:
        ob.data.materials.append(make_material(pink))

    bpy.ops.wm.save_as_mainfile(filepath="test.blend")

    # Finally, run the render
    bpy.ops.render.render(write_still = 1)
