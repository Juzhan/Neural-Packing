import bpy
import os
import numpy as np
import itertools

colors = [ "#FBB5AF", "#FBE06F", "#B0E586", "#8AD4D5", "#718DD5", "#A38DDE", "#9ED68C", "#61abff", "#ffb056", '#A9CBFF' ]
colors = [ "#FBB5AF", "#FBE06F", "#D4E59C", "#8AD4D5", "#718DD5", "#A38DDE", "#9ED68C", "#61abff", "#ffb056", '#A9CBFF' ]
colors = [ "#FBB5AF", "#D4E59C", "#FBE06F", "#8AD4D5", "#718DD5", "#A38DDE", "#9ED68C", "#61abff", "#ffb056", '#A9CBFF' ]


def hsv2rgb(h, s, v):
    """
    Return [R, G, B]
    """
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = np.math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return [r, g, b]

def rgb2hsv(r, g, b):
    """
    Return [H, S, V]
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return [h, s, v]

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4

def hex_to_rgb(hex_str):
    h = int('0x' + hex_str[1:], 0)    
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return [srgb_to_linearrgb(c/0xff) for c in (r,g,b)]

def hex_to_rgba(hex_str, alpha=1):
    h = int('0x' + hex_str[1:], 0)
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return [srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha]

def set_visible_property(object, diffuse=True, glossy=True, shadow=True):
    if bpy.app.version[0] == 3:
        object.visible_diffuse = diffuse
        object.visible_glossy = glossy
        object.visible_shadow = shadow
    else:    
        object.cycles_visibility.diffuse = diffuse
        object.cycles_visibility.glossy = glossy
        object.cycles_visibility.shadow = shadow

def clear_all():
    [bpy.data.materials.remove(mat) for mat in bpy.data.materials]
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes]
    [bpy.data.curves.remove(curve) for curve in bpy.data.curves]
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects]
    
    [bpy.data.collections.remove(col) for col in bpy.data.collections]
    [bpy.data.cameras.remove(cam) for cam in bpy.data.cameras]
    [bpy.data.images.remove(img) for img in bpy.data.images]
    [bpy.data.lights.remove(light) for light in bpy.data.lights]

def do_render(render_path, format="PNG", adaptive_threshold=0.1, exr_codec='DWAA'):
    '''
    Args:
        render_path: str
    '''

    render = bpy.context.scene.render
    render.use_file_extension = True
    
    render.filepath = render_path
    
    render.image_settings.file_format = format

    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold

    if format == 'OPEN_EXR':
        bpy.context.scene.render.image_settings.exr_codec = exr_codec

    bpy.ops.render.render(write_still=True)

def set_render( resolution_x=640, resolution_y=480, engine="CYCLES", \
    samples=128, color_management='Standard', color_mode='RGBA', \
    bg_transparent=True ):
    # 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.device = 'GPU'
    render = bpy.context.scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y

    render.image_settings.color_mode = color_mode
    render.film_transparent = bg_transparent
    
    # bpy.context.scene.view_settings.view_transform = 'Filmic' # gray
    bpy.context.scene.view_settings.view_transform = color_management

def new_material(mat_name, color=(1,0,0,1), shadow_mode='OPAQUE'):

    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.name = mat_name
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links

    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    

    principled_node.inputs.get("Base Color").default_value = color
    principled_node.inputs.get("Alpha").default_value = color[3]

    # principled_node.inputs.get("Transmission").default_value = 1 - color[3]
    # if color[3] < 1:
    #     principled_node.inputs.get("Alpha").default_value = 0.5

    # principled_node.inputs.get("Roughness").default_value = 1.0
    # principled_node.inputs.get("Emission Strength").default_value = 0
    
    links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )

    if color[-1] < 1:
        mat.blend_method = 'BLEND'

    mat.shadow_method = shadow_mode
    return mat

def color_material( object, mat_name, color=(1,0,0,1), set_active=True):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        color: list[float] 4
        shadow_mode: str (only work for eevee?)
            "OPAQUE" for shadow \\
            "NONE" for none
    
    Returns:
        None
    '''
    mat = new_material(mat_name, color)
    obj_mats = object.data.materials
    if mat_name not in obj_mats:
        obj_mats.append(mat)
    return mat

def add_shape(obj_type, obj_name, obj_pos, obj_rot, obj_size, obj_color=[0.8,0.8,0.8,1], \
        shape_setting=None, \
        diffuse=False, glossy=False, shadow=True, mat_name=None ):
    '''
    Args:
        obj_type: str
            >>> 'cube'
            >>> 'sphere'
            >>> 'cone'
            >>> 'plane'
            >>> 'cylinder'
            >>> 'torus'
        obj_name: str
        obj_pos: list[float] [3]
        obj_rot: list[float] [3]
        obj_color: list[float] [4]
        shape_setting: dict
            specify for shape
    
    Returns:
        object: bpy.data.object
    '''
    if obj_name in bpy.data.objects:
        object = bpy.data.objects[obj_name]
    else:
        if obj_type == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(vertices=shape_setting['vertices'])
            init_name = 'Cylinder'

        elif obj_type == 'plane':
            bpy.ops.mesh.primitive_plane_add()
            init_name = 'Plane'

        elif obj_type == 'cube':
            bpy.ops.mesh.primitive_cube_add()
            init_name = 'Cube'

        elif obj_type == 'sphere':
            seg = 32
            rings = 32
            if shape_setting:
                seg = shape_setting['segments']
                rings = shape_setting['rings']
            bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings)
            init_name = 'Sphere'
            
        elif obj_type == 'cone':
            bpy.ops.mesh.primitive_cone_add(radius1=shape_setting['radius1'], radius2=shape_setting['radius2'], depth=shape_setting['depth'])
            init_name = 'Cone'

        elif obj_type == 'torus':
            major_segments = 48
            minor_segments = 8
            minor_radius = 0.1
            major_radius = 0.6
            if shape_setting is not None:
                if 'minor_segments' in shape_setting:
                    minor_segments = shape_setting['minor_segments']
                if 'major_segments' in shape_setting:
                    major_segments = shape_setting['major_segments']
                if 'minor_radius' in shape_setting:
                    minor_radius = shape_setting['minor_radius']
                if 'major_radius' in shape_setting:
                    major_radius = shape_setting['major_radius']

            bpy.ops.mesh.primitive_torus_add(
                major_segments=major_segments,
                minor_segments=minor_segments,
                minor_radius=minor_radius,
                major_radius=major_radius)

            init_name = 'Torus'

        for o in bpy.data.objects:
            if o.name == init_name:
                object = o
                break
        object = bpy.context.active_object

    object.name = obj_name
    
    object.data.name = obj_name

    object.scale = obj_size
    object.rotation_mode = 'XYZ'
    object.rotation_euler= obj_rot
    object.location= obj_pos

    if mat_name is None:
        mat_name = obj_name
    
    set_visible_property(object, diffuse, glossy, shadow)

    color_material(object, mat_name, obj_color)

    return object

def add_axis(origin, lens, diameter, end_width, order):
    x_len = lens[0]
    y_len = lens[1]
    z_len = lens[2]

    rot = [0,0,0]

    mat_dict = {
        0: ['x', [1,0,0,1]],
        1: ['y', [0,1,0,1]],
        2: ['z', [0,0,1,1]],
    }

    add_shape('cube', 'axis_x', origin + [x_len/2, 0, 0], rot, [x_len/2, diameter, diameter], mat_dict[order[0]][1], shadow=False, mat_name=mat_dict[order[0]][0] )
    add_shape('cube', 'axis_y', origin + [0, y_len/2, 0], rot, [diameter, y_len/2, diameter], mat_dict[order[1]][1], shadow=False, mat_name=mat_dict[order[1]][0] )
    add_shape('cube', 'axis_z', origin + [0, 0, z_len/2], rot, [diameter, diameter, z_len/2], mat_dict[order[2]][1], shadow=False, mat_name=mat_dict[order[2]][0] )

    end_size = [end_width, end_width, end_width]
    add_shape('cube', 'C_x', origin + [x_len, 0, 0], rot, end_size, mat_dict[order[0]][1], shadow=False, mat_name=mat_dict[order[0]][0] )
    add_shape('cube', 'C_y', origin + [0, y_len, 0], rot, end_size, mat_dict[order[1]][1], shadow=False, mat_name=mat_dict[order[1]][0] )
    add_shape('cube', 'C_z', origin + [0, 0, z_len], rot, end_size, mat_dict[order[2]][1], shadow=False, mat_name=mat_dict[order[2]][0] )

def obj_to_wireframe(obj, thickness=0.04, only_wire=False, offset=0):
    if obj.name in obj.modifiers:
        wire = obj.modifiers[obj.name]
    else:
        wire = obj.modifiers.new( obj.name, 'WIREFRAME' )

    wire.thickness = thickness
    wire.material_offset = offset
    wire.use_replace = only_wire
    return wire

def new_camera(name='Camera'):
    '''
    A new camera in scene
    
    Returns:
        camera_object
    '''
    if name in bpy.data.objects:
        camera_object = bpy.data.objects[name]
    else:
        # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
        camera_data = bpy.data.cameras.new(name=name)
        camera_object = bpy.data.objects.new('Camera', camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        bpy.context.scene.camera = camera_object
    return camera_object

def set_cam_pos(cam, cam_pos, look_at, ortho_scale=11):
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = ortho_scale
    
    if 'Empty' not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=look_at, scale=(1, 1, 1))

    bpy.data.objects['Empty'].location = look_at
    cam.location = cam_pos
    track = cam.constraints.new( 'TRACK_TO' )
    track.target = bpy.data.objects['Empty']

def set_hdr_background( hdr_path, rotation=[0,0,0], strength=1 ):
    '''
    Add hdr for scene
    
    Args:
        hdr: str
        rotation: list[float] [3]
    
    Returns:
        None
    '''
    world = bpy.data.worlds.get('World')
    
    world.use_nodes = True

    nodes = world.node_tree.nodes
    nodes.clear()

    links = world.node_tree.links
    
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    
    hdr_tex = nodes.new(type="ShaderNodeTexEnvironment")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    bg = nodes.new(type="ShaderNodeBackground")
    
    img = bpy.data.images.load(hdr_path)
    hdr_tex.image = img

    rotation = [ np.deg2rad(r) for r in rotation ]
    mapping.inputs['Rotation'].default_value = rotation
    bg.inputs['Strength'].default_value = strength

    links.new( tex_coord.outputs['Object'], mapping.inputs['Vector'] )
    links.new( mapping.outputs['Vector'], hdr_tex.inputs['Vector'] )
    links.new( hdr_tex.outputs['Color'], bg.inputs['Color'] )
    links.new( bg.outputs['Background'], output_node.inputs['Surface'] )

def set_world_light_rot(rotation):
    world = bpy.data.worlds.get('World')
    nodes = world.node_tree.nodes
    tex_coord = nodes["Mapping"]
    tex_coord.inputs['Rotation'].default_value = rotation

def set_cam(cam, target_width, width, max_height, img_w, img_h, single=False, ortho_scale=11, z_offset=0):
    
    bin_center = np.array([0,0,0]) + [target_width/2, target_width/2, max_height/2]
    # cam_center = bin_center * 4 - [-width/2, 0, max_height/1]
    if single:
        cam_center = bin_center * 4 - [0, 0, max_height/2]
    else:
        cam_center = bin_center * 4 - [-width * 0.8, 0, max_height * 0.5]

    width_scale = (target_width*1.0/width) * (width / 5)

    cam_center[-1] += z_offset
    bin_center[-1] += z_offset

    set_cam_pos(cam, cam_center, bin_center, ortho_scale=ortho_scale * width_scale)
    set_render( int(img_w), int(img_h), samples=500)

