import os
import sys
import bpy
# change to your path here
ROOT_DIR = "E:/workspace/TAP/binpacking/render"

sys.path.append(ROOT_DIR)

import importlib
import render_tools

importlib.reload(render_tools)

size_list = None

maya = False
unit_scale = 1

cs = [5,5,5]
cs = [100, 100, 100]
cs = [600, 150, 10]

task_path = "E:/workspace/TAP/binpacking/pack/test/100-100-note-test-source"
task_path = "E:/workspace/TAP/binpacking/pack/test/100-100-note-test"
#task_path = "E:/workspace/TAP/binpacking/pack/test/100-100-note-ppsg"

#task_path = "H:/Project/rl-binpacking/examples/binpacking/web_service/logs/maya_v3/230320-181842/logs_0_packed_seq.npy"
#task_path="E:/workspace/TAP/binpacking/web_service/logs/maya_v3/230423-094822/logs_0_packed_seq.npy"
task_path="E:/workspace/TAP/binpacking/test_json/taoliao/pre50/pack/0/0-taoliao.npy"
#maya = 'maya' in task_path
maya = True

# height-diff-50 230330-114825

# >>> 230330-014406

size_list = [ [43, 34, 24], [33, 29, 20], [31, 23, 16] ]
render_tools.test_pack(os.path.join(ROOT_DIR), task_path, unit_scale, maya, \
    size_list=size_list, container_size=cs)
    
bpy.data.objects["bin"].modifiers["bin"].thickness = 0.004

