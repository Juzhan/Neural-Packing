import bpy
import bmesh

import os
import numpy as np
import itertools
import importlib
import tools
import pickle
importlib.reload(tools)
from tools import *
# from pack_tools.pack import left_botton
from scipy.spatial.transform import Rotation



def add_heightmap(container_size):
    heightmap = add_shape('plane', 'heightmap', [ container_size[0] / 2.0, container_size[1] / 2.0, 0], [0,0,0], container_size / 2.0, [1,1,1,1] )


    heightmap.select_set( True )
    bpy.context.view_layer.objects.active = heightmap

    bpy.ops.object.mode_set(mode = 'OBJECT')

    bpy.ops.object.mode_set(mode = 'EDIT')

    def view3d_find( return_area = False ):
        # returns first 3d view, normally we get from context
        for area in bpy.context.window.screen.areas:
            if area.type == 'VIEW_3D':
                v3d = area.spaces[0]
                rv3d = v3d.region_3d
                for region in area.regions:
                    if region.type == 'WINDOW':
                        if return_area: return region, rv3d, v3d, area
                        return region, rv3d, v3d
        return None, None

    region, rv3d, v3d, area = view3d_find(True)

    override = {
        'scene'  : bpy.context.scene,
        'region' : region,
        'area'   : area,
        'space'  : v3d
    }


    bpy.ops.mesh.loopcut_slide(
       override,
       MESH_OT_loopcut={
           "number_cuts":container_size[0]-1,
           "smoothness":0,
           "falloff":'INVERSE_SQUARE',
           "object_index":0,
           "edge_index":0,
           "mesh_select_mode_init":(False, True, False)
           })
    bpy.ops.mesh.loopcut_slide(
       override,
       MESH_OT_loopcut={
           "number_cuts": container_size[1]-1,
           "smoothness":0,
           "falloff":'INVERSE_SQUARE',
           "object_index":0,
           "edge_index":1,
           "mesh_select_mode_init":(False, True, False)
           })
    
    bpy.ops.mesh.select_all(action='SELECT')
    
    bpy.ops.mesh.select_all(action = 'DESELECT')

    # for v in heightmap.data.vertices:
    #     if v.co[2] > border_threshold and \
    #     v.co[0] > border_threshold and \
    #     v.co[1] > border_threshold :
    #         v.select = True
    bpy.ops.object.mode_set(mode = 'OBJECT') 
    
    heightmap.data.polygons[0].select = True
    heightmap.data.polygons[2].select = True
    heightmap.data.polygons[4].select = True
    heightmap.data.polygons[7].select = True
    heightmap.data.polygons[99].select = True
    heightmap.data.polygons[100].select = True
    heightmap.data.polygons[200].select = True
    heightmap.data.polygons[300].select = True
    
    
    return heightmap



def add_bin(box, color, container_width, container_height, reso, thickness=0.002):
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes if 'bin' in mesh.name]
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects if 'bin' in obj.name]

    box = np.array(box)

    # heightmap = add_heightmap(box)
    
    container_width = container_width * reso
    container_height = container_height * reso
    border_threshold = 0.995
    
    bin = add_shape('cube', 'bin', [0,0,0] + box/2.0, [0,0,0], box/2.0, color )
    obj_to_wireframe(bin, only_wire=True, thickness=0.04)


    bin_wire = add_shape('cube', 'bin_wire', [0,0,0] + box/2.0, [0,0,0], box/2.0, color )
    obj_to_wireframe(bin_wire, only_wire=True, thickness=thickness)
    
    bin_wire.select_set( True )
    bpy.context.view_layer.objects.active = bin_wire

    # if container_width < 10:
    if True:
        bin.select_set( True )
        bpy.context.view_layer.objects.active = bin
        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action = 'DESELECT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 
        for v in bin.data.vertices:
            if v.co[2] > border_threshold and \
            v.co[0] > border_threshold and \
            v.co[1] > border_threshold :
                v.select = True
        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 


        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action = 'DESELECT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 
        for v in bin_wire.data.vertices:
            if v.co[2] > border_threshold and \
            v.co[0] > border_threshold and \
            v.co[1] > border_threshold :
                v.select = True
        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 
        
        # loop cut
        bin_wire.select_set( True )
        bpy.context.view_layer.objects.active = bin_wire

        bpy.ops.object.mode_set(mode = 'OBJECT')

        bpy.ops.object.mode_set(mode = 'EDIT')

        def view3d_find( return_area = False ):
            # returns first 3d view, normally we get from context
            for area in bpy.context.window.screen.areas:
                if area.type == 'VIEW_3D':
                    v3d = area.spaces[0]
                    rv3d = v3d.region_3d
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            if return_area: return region, rv3d, v3d, area
                            return region, rv3d, v3d
            return None, None

        region, rv3d, v3d, area = view3d_find(True)

        override = {
            'scene'  : bpy.context.scene,
            'region' : region,
            'area'   : area,
            'space'  : v3d
        }

        if container_width == 100:
            a = 10
            b = 10
        else:
            a = 14
            b = 16

        bpy.ops.mesh.loopcut_slide(
        override,
        MESH_OT_loopcut={
            "number_cuts": a,
            "smoothness":0,
            "falloff":'INVERSE_SQUARE',
            "object_index":0,
            "edge_index":8,
            "mesh_select_mode_init":(False, True, False)
            })
        bpy.ops.mesh.loopcut_slide(
        override,
        MESH_OT_loopcut={
            "number_cuts": a,
            "smoothness":0,
            "falloff":'INVERSE_SQUARE',
            "object_index":0,
            "edge_index":5,
            "mesh_select_mode_init":(False, True, False)
            })
        bpy.ops.mesh.loopcut_slide(
        override,
        MESH_OT_loopcut={
            "number_cuts": b,
            "smoothness":0,
            "falloff":'INVERSE_SQUARE',
            "object_index":0,
            "edge_index":1,
            "mesh_select_mode_init":(False, True, False)
            })

        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.select_all(action = 'DESELECT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 
        for v in bin_wire.data.vertices:
            if v.co[2] > border_threshold and \
            abs(v.co[0]) < border_threshold and \
            abs(v.co[1]) < border_threshold :
                v.select = True

        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode = 'OBJECT') 

    return bin, bin_wire

def add_boxes(boxes, positions, rots, offset=np.array([0,0,0]), mode='color', color_offset=0, real_order=None, size_list=None ):
    
    box_num = len(boxes)
    # print(box_num)
    color = [0.2,0.2,0.2,1]

    white_color = [1,1,1,1]

    for i in range(box_num):
        
        box = boxes[i]
        pos = positions[i]
        rot = rots[i]

        if np.sum(pos) == 0 and i > 0:
            yield 0
            for pre_i in range(i):
                
                cube_name = 'box_%d' % pre_i
                if mode == 'ems':
                    cube_name = 'ems_%d' % pre_i
                    
                obj = bpy.data.objects[cube_name]
                # obj.hide_render = True

        x,y,z,w=Rotation.from_matrix(rot).as_quat()
        quat = [w,x,y,z]

        if box_num < 10:
            color_index = (i + color_offset) % (box_num+color_offset)
        else:
            color_index = (i + color_offset) % (10+color_offset)
        
        threshold = 1e-4
        # threshold = 2

        if size_list is not None:
            for si, s in enumerate(size_list):
                if (box[-1] - s[-1] < threshold and box[-1] - s[-1] >= 0):
                    if ( (box[0] - s[0] >= 0) and (box[0] - s[0] < threshold) and (box[1] - s[1] >= 0) and (box[1] - s[1] < threshold) ) or \
                          ( (box[1] - s[0] >= 0) and (box[1] - s[0] < threshold) and (box[0] - s[1] < threshold) and (box[0] - s[1] >= 0) ):
                        color_index = si
                        break
                    
        if real_order is not None:
            color_index = real_order[color_index]

        hex_color = colors[color_index % 10]
        # if i == 0:
        #     hex_color = colors[(i + color_offset) % (box_num+color_offset)]
        # elif i == 1:
        #     hex_color = colors[6]

        color = hex_to_rgba(hex_color)
        color = np.array(color)
        
        cube_name = 'box_{}'.format(i)
        if mode == 'ems':
            color[-1] = 0.5
            cube_name = 'ems_{}'.format(i)
        
            cube = add_shape('cube', cube_name, offset + pos + box/2.0, [0,0,0], box/2.0, color )
        # if i < 16:
        #     color = [0.8, 0.8, 0.8, 1]

            # location = add_shape('sphere', cube_name + '_loc', offset + pos + [box[0], box[1], 0], [0,0,0], [0.12]*3, [0.1,0.1,0.1,1] )
            location = add_shape('sphere', cube_name + '_loc', offset + pos, [0,0,0], [0.12]*3, [0.1,0.1,0.1,1] )
        else:
            cube = add_shape('cube', cube_name, offset + pos + box/2.0, [0,0,0], box/2.0 * 0.99, color )
            wire = obj_to_wireframe( cube, 0.06, only_wire=False, offset=1)


        cube.rotation_mode = 'QUATERNION'
        cube.rotation_quaternion = quat

        if mode in ['white', 'pack']:
            for j in range(i):
                if mode == 'pack':
                    if j == i-1: break

                cube_name = 'box_%d' % j
                obj = bpy.data.objects[cube_name]
                color_material( obj, 'white', white_color, True)
                obj.data.materials.clear()
                obj.data.materials.append(bpy.data.materials['white'])
                obj.data.materials.append(bpy.data.materials[cube_name])

    yield 1

def add_ems(ems, offset=np.array([0,0,0]), max_height=10):
    ems = np.array(ems)

    ems_list = []
    ems_num = len(ems)    
    for i in range(ems_num):
    # for i in [1]:
        ems_box = np.array(ems[i]).astype(float)
        ems_box[1][-1] = max_height

        box = ems_box[1] - ems_box[0]

        if box[-1] == 0:
            box[-1] = 0.2
        pos = ems_box[0]

        hex_color = colors[i+1]
        color = hex_to_rgba(hex_color)
        color[-1] = 0.5
        
        # color = [0.5, 0.5, 0.5, 0.5]

        mat_name = 'mes_{}'.format(i)
        ems_cube = add_shape('cube', mat_name, offset + pos + box/2.0, [0,0,0], box/2.0 * 0.99999, color )
        ems_list.append(ems_cube)
    
    return ems_list

def show_box_single(batch, boxes, box_ids, offset, save_path):

    def xyz(order):
        dicts = ['x', 'y', 'z']
        ret = ''
        for o in order:
            ret += dicts[o]
        return ret
    
    box_num = len(boxes)
    print(boxes.shape)
    
    real_box_ids = [ ids % box_num for ids in box_ids ]

    box_num = 10
    for i in range(box_num):
        
        cube_name = 'box_{}'.format(i)
        box_id = i
        box_id = real_box_ids[i]
        box_size = boxes[box_id]
        
        hex_color = colors[i % box_num]
        # hex_color = colors[i % box_num]
        color = hex_to_rgba(hex_color)
        color = np.array(color)

        for pi, p in enumerate(itertools.permutations([0,1,2], 3)):
            order = np.array(p)
            box = box_size[ order ]
            pos = -box / 2.0
            pos[-1] += 1

            border_color = color

            cube_center = offset + pos + box/2.0
            cube = add_shape('cube', cube_name, cube_center, [0,0,0], box/2.0 * 0.99, color )
            color_material( cube, cube_name + '_' + xyz(order), border_color, True)

            wire = obj_to_wireframe( cube, 0.06, only_wire=False, offset=pi+1)
            # add_axis(cube_center, box * 0.8, 0.1, 0.2, order)

            save_folder = os.path.join( save_path, 'batch_%d' % batch, 'box' )
            os.makedirs(save_folder, exist_ok=True)
            do_render( os.path.join(save_folder, cube_name + '_' + xyz(order) ) )

        bpy.data.objects.remove( bpy.data.objects[ cube_name ])
        bpy.data.meshes.remove( bpy.data.meshes[ cube_name ] )

def load_data(task, path, prefix=''):

    if not os.path.exists(os.path.join( path, task, f'{prefix}pack_info_ems.pkl')):
        return None
    
    # ems = np.load( os.path.join( path, task, 'pack_info_ems.npy' ), allow_pickle=True )
    with open(os.path.join( path, task, f'{prefix}pack_info_ems.pkl' ), 'rb') as f:
        ems = pickle.load(f)

    boxes = np.load( os.path.join(path, task, f'{prefix}pack_info_box.npy') )
    positions = np.load( os.path.join(path, task, f'{prefix}pack_info_pos.npy') )
    reward = -np.load( os.path.join(path, task, f'{prefix}pack_info_rew.npy') )
    # all_boxes = np.load( os.path.join(path, task, f'{prefix}pack_info_all.npy') )
    idx = np.load( os.path.join(path, task, f'{prefix}pack_info_idx.npy') )

    return boxes, positions, reward, ems


def test_pack(path, test_result, unit_scale=1, maya=False, size_list=None, container_size=[100,100,150], reverse=False):
    clear_all()
    # [0, -37, -33]
    set_hdr_background( os.path.join( path, './light/leadenhall_market_2k.hdr'), [0, -37, -33], 1.1)
    img_w, img_h = 600, 600
    
    container_width = 5
    container_height = 6
    reso = 1
    step = 1000
    thickness=0.004

    save_path = os.path.join( path, './result', 'test')

    container_width = container_size[0]
    container_length = container_size[1]
    container_height = container_size[2]

    if maya:
        data = np.load( test_result ) / unit_scale
        blocks = data[:, :3]
        positions = data[:, 3:6]
    else:
        blocks = np.load( os.path.join(path, test_result, f'pack_info_box.npy') ) / unit_scale
        positions = np.load( os.path.join(path, test_result, f'pack_info_pos.npy') ) / unit_scale

        ems_blocks = np.load( os.path.join(path, test_result, f'pack_info_ems_box.npy') ) / unit_scale
        ems_positions = np.load( os.path.join(path, test_result, f'pack_info_ems_pos.npy') ) / unit_scale

        # print(blocks, ems_blocks)
    # reward = -np.load( os.path.join(path, test_result, f'pack_info_rew.npy') )
    # print(reward)
    if reverse:
        blocks = blocks[:,[1,0,2]]
        positions = positions[:,[1,0,2]]

        ems_blocks = ems_blocks[:,[1,0,2]]
        ems_positions = ems_positions[:,[1,0,2]]

    rots = [np.eye(3) for i in range(500)]
    
    cam = new_camera()

    set_cam(cam, container_width, container_width, container_height, img_w, img_h, ortho_scale=10)

    bin, bin_wire = add_bin([container_width, container_length, container_height], [0.2,0.2,0.2,1], container_width, container_height, reso, thickness)
    # bin_wire.hide_viewport = True
    # bin_wire.hide_render = True
    # bpy.data.objects["bin_wire"].rotation_euler[2] = np.pi
    # bpy.data.objects["bin"].rotation_euler[2] = np.pi
    bpy.data.scenes["Scene"].use_nodes = False

    ctn = 0


    # ems_positions[:, :2] = container_width - ems_positions[:, :2] - ems_blocks[:, :2]
    # positions[:, :2] = container_width - positions[:, :2] - blocks[:, :2]

    # for r in add_boxes(ems_blocks[:step], ems_positions[:step], rots, mode='ems'):
    #     ctn += 1

    for r in add_boxes(blocks[:step], positions[:step], rots, mode='color', size_list=size_list):
        ctn += 1
        # break
        # do_render( os.path.join(save_path, f"{task}_{ctn}_{score:3f}") )


def render_pack(path, task_i, data_type='rand', model='tnpp', container_type='single', pack_type='last', unit_scale=1, container_size=[100,100,100], reverse=False, img_wh=[600,600]):
    
    pack_path = "./results"

    if model == 'init':
        model_str = f"{data_type}/{task_i}/{model}"
        container_size = [140, 140, 160]
    else:
        model_str = f"{data_type}/{task_i}/{container_type}/{model}_{pack_type}"
        container_size = [100, 100, 100]
    
    data_path = f"{pack_path}/data/" + model_str
    
    c_path = os.path.join(path, data_path + '.npy')
    container_num = np.load(c_path)

    print(container_num)

    for ci in range(container_num[0]):
        clear_all()
        # [0, -37, -33]
        set_hdr_background( os.path.join( path, './light/leadenhall_market_2k.hdr'), [0, -37, -33], 1.1)
        img_w, img_h = img_wh[0], img_wh[1]
        
        reso = 1
        step = 1000
        thickness=0.004

        save_path = os.path.join( path, f'{pack_path}/pack_ideal/', model_str + f'-{ci}')

        container_width = container_size[0]
        container_length = container_size[1]
        container_height = container_size[2]

        blocks = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_box.npy') ) / unit_scale
        positions = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_pos.npy') ) / unit_scale

        stop_data = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_stop.npy') )
        print(stop_data)
        if len(stop_data) > 0:
            stop_data = stop_data / unit_scale
            stop_pos = stop_data[0]
            stop_box = stop_data[1]
        else:
            stop_box = None
            stop_pos = None

        max_height = container_height
        box_add_wire = False
        
        box_num = len(blocks)

        if stop_pos is not None:
            if (stop_pos == positions[-1]).all():
                stop_pos = None
                # box_add_wire = True
                # box_num -= 1
            else:
                stop_color = [0.821, 0.092, 0.104, 0.6]
                # stop_color = [0, 0.433, 0.821, 0.3]
                max_height = stop_pos[2] + stop_box[2]
        
        box_ids = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_id.npy') )

        ems_blocks = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_ems_box.npy') ) / unit_scale
        ems_positions = np.load( os.path.join(path, data_path + f'-{ci}', f'pack_info_ems_pos.npy') ) / unit_scale

        
        rots = [np.eye(3) for i in range(500)]
        cam = new_camera()

        # set_cam(cam, container_width, container_width, max_height, img_w, img_h, ortho_scale=9 * max_height / container_height)
        set_cam(cam, container_width, container_width, container_height, img_w, img_h, ortho_scale=9)

        bin, bin_wire = add_bin([container_width, container_length, container_height], [0.2,0.2,0.2,1], container_width, container_height, reso, thickness)
        # bin_wire.hide_viewport = True
        # bin_wire.hide_render = True
        # bpy.data.objects["bin_wire"].rotation_euler[2] = np.pi
        # bpy.data.objects["bin"].rotation_euler[2] = np.pi
        bpy.data.scenes["Scene"].use_nodes = False

        ctn = 0
        size_list = None

        if size_list is None:
            size_list = None


        real_order = [ i for i in range(box_num)]

        if model != 'init':
            real_order = box_ids[:box_num]

        for r in add_boxes(blocks[:box_num], positions[:box_num], rots, mode='color', real_order=real_order, size_list=size_list):
            ctn += 1
            # break

        if stop_pos is not None:
            stop_cube = add_shape('cube', 'stop_cube', stop_pos + stop_box/2.0, [0,0,0], stop_box/2.0, [1,1,1,0.2] )
            stop_color[-1] = 1
            color_material( stop_cube, 'stop_cube_wire', stop_color, True)
            obj_to_wireframe( stop_cube, 0.06, only_wire=False, offset=1)

        # break
        do_render( save_path )

