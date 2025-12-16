import numpy as np
import os
from scipy.spatial.transform import Rotation
import bpy



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


def load_cubes( path, method, task_id, container_id, offset=[0,0,0]):

    
    _, _, init_sizes = load_data(path, method, task_id, 'init')

    locs, quats, sizes = load_data(path, method, task_id, container_id)

    if container_id != 'init':
        candidate_ids = [ i for i in range(len(init_sizes))]
        match_ids = []

        for s in sizes:
            sort_candi_sizes = np.sort(np.array(init_sizes[candidate_ids]), axis=1)
            
            size_diff = np.linalg.norm( np.sort(s) - sort_candi_sizes, axis=1 )
            match_id = np.argmin(size_diff)
            match_ids.append( candidate_ids[match_id] )
            candidate_ids.remove( candidate_ids[match_id] )
    else:
        match_ids = [ i for i in range(len(init_sizes))]
    
    np.random.seed(task_id)
    tmp_ids = np.random.choice( [0,1,2,3,4,5], len(init_sizes) )
    
    candidate_sets = [
        bpy.data.objects['1'],
        bpy.data.objects['2'],
        bpy.data.objects['3'],
        bpy.data.objects['4'],
        bpy.data.objects['5'],
        bpy.data.objects['6'],
    ]

    box_num = len(locs)

    # box_num = int(box_num / 2)
    skip = 1

    all_boxes = []

    for i in range(box_num):
        
        name = f"mbox_{i}"

        tmp = candidate_sets[ tmp_ids[match_ids[i]] ]

        bpy.ops.object.select_all(action="DESELECT")
        tmp.select_set( True )
        bpy.context.view_layer.objects.active = tmp

        bpy.ops.object.duplicate_move()

        new_obj = bpy.context.active_object
        new_obj.name = name
        new_obj.data.name = name + '_mesh'

        new_obj.location = locs[i * skip] + np.array(offset)
        new_obj.rotation_mode = 'QUATERNION'

        new_obj.rotation_quaternion = quats[i * skip]
        new_obj.scale = sizes[i * skip]
    
        all_boxes.append(new_obj)


    data_path = os.path.join( path, f"./images/sim/{method}/{task_id}/{container_id}")
    # do_render(data_path)
    
    return all_boxes

def load_data(path, method, task_id, container_id):
    data_path = os.path.join( path, f"./images/sim/{method}/{task_id}/{container_id}-data")

    mats = np.load( os.path.join(data_path, 'mat.npy') )
    extents = np.load( os.path.join(data_path, 'extent.npy') )
    scales = np.load( os.path.join(data_path, 'scale.npy') )

    sizes = extents * scales

    quats = []
    locs = []
    list_sizes = []
    if container_id != 'init':
        resize_into = 0.55
    else:
        resize_into = 1

    sizes *= resize_into


    for i, mat in enumerate(mats):
        they_have = False
        pos = mat[:3,3] * resize_into
        for l in locs:
            if np.linalg.norm(pos - l) < 1e-5:
                they_have = True
                break
        if not they_have:
            locs.append(pos)
            rot = mat[:3,:3]

            # if container_id != 'init' and np.random.rand() > 0.5:
            #     rot = Rotation.from_rotvec([0,0,np.pi]).as_matrix() @ rot
            
            x,y,z,w = Rotation.from_matrix(rot).as_quat()
            quats.append([w,x,y,z])
            list_sizes.append(sizes[i])

    return locs, quats, np.array(list_sizes)
