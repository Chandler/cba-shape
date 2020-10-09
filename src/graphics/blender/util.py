
def look_at(obj, point):
    """ obj can be camera or light
    """
    loc = obj.location

    direction = point - loc
    # point the obj '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj.rotation_euler = rot_quat.to_euler()


def delete_objects(objects):
    # Could be used to clear objects to refresh the GUI
    # Pass in bpy.data.objects
    for objk in objects.keys():
        objects.remove(objects[objk], do_unlink=True)
