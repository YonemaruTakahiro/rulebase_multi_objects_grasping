import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.modeling import collision_model as cm
import wrs.modeling.constant as const
import numpy as np
import math
import placepattern as pp
import pushing_and_grasping as pag
import itertools
import networkx as nx
import csv
import time
import copy
import termcolor

from wrs.modeling import _ode_cdhelper as oc
from wrs.motion.motion_data import MotionData
from sakamoto_method_class import Sakamoto


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


TABLE_Z = 1.096

BOX_POS = np.array([0.35, -0.1, TABLE_Z])
RS_POS = np.array([0.4, -0.47, TABLE_Z + 0.6])
CENTOR_WORKSPACE = np.array([0.35, -0.35, TABLE_Z])
DISTANCE_OF_DOUBLEPICK = 0.08
ANGLEDIFF_OF_DOUBLEPICK = np.pi / 2
ALIGN_DISTANCE = 0.02
PUSHER_THICKNESS = 0.02
objheight = 0.02
r_th = 0.08
d_th = 0.4
theta_th = 2*np.pi / 3
alpha = np.pi / 3
divnum = 8
pushing_Z = 0.015 # base is table
pusher_width = 0.085
distance_from_objcor_to_pushing_surface = 0.01

RANDOM = True

if __name__ == "__main__":

    base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
    mcm.mgm.gen_frame().attach_to(base)

    # robot
    robot = ur3d.UR3Dual()
    robot.use_rgt()
    robot.change_jaw_width(jaw_width=0.08)
    # robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
    init_tgt_pos = np.array([BOX_POS[0], BOX_POS[1], BOX_POS[2] + 0.15])
    init_tgt_rotmat = rm.rotmat_from_euler(rm.pi, 0, 0)
    mcm.mgm.gen_frame(pos=init_tgt_pos, rotmat=init_tgt_rotmat).attach_to(base)
    jnt_values_above_box = robot.ik(tgt_pos=init_tgt_pos, tgt_rotmat=init_tgt_rotmat)
    # print(f"jnt_values:{jnt_values}")
    robot.goto_given_conf(jnt_values=jnt_values_above_box)
    robot.gen_meshmodel().attach_to(base)

    # box
    box = cm.CollisionModel("/home/yonemaru/PycharmProjects/wrs/yonemaru/objects/yonemaru_box.stl")
    boxrot = rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
    boxmat4 = rm.homomat_from_posrot(pos=BOX_POS, rotmat=boxrot)
    box.homomat = boxmat4
    box.rgba = np.array([0.87, 0.65, 0.53, 1])
    box.attach_to(base)
    # obj_3dprinted for picking
    obj_3dprinted = cm.CollisionModel("/home/yonemaru/PycharmProjects/wrs/yonemaru/objects/yonemaru_object1.stl")
    #obstacle

    sakamoto = Sakamoto(base=base, robot=robot, table_z=TABLE_Z, box=box, box_pos=BOX_POS, printed_obj=obj_3dprinted,
                        objheight=objheight,
                        distance_from_objcor_to_pushing_surface=distance_from_objcor_to_pushing_surface,
                        centor_workspace=CENTOR_WORKSPACE,
                        distance_doublepick=DISTANCE_OF_DOUBLEPICK, angle_difference_doble_pick=ANGLEDIFF_OF_DOUBLEPICK,
                        align_distance=ALIGN_DISTANCE,
                        pusher_thickness=PUSHER_THICKNESS, aboveboxpos=init_tgt_pos,
                        jnt_values_above_box=jnt_values_above_box, r_th=r_th, d_th=d_th,
                        theta_th=theta_th, alpha=alpha,
                        divnum=divnum,
                        pushing_Z=pushing_Z, pusher_width=pusher_width)

    x_range = (-0.1, 0.1)
    y_range = (-0.15, 0.15)
    sakamoto.placeRandomRoundly(num=4, x_range=x_range, y_range=y_range, liftheight=0.03)

    obscmlist = [box] + sakamoto.objects_list.cdmodel_list
    # pp.writePlacePattern('placepattern.csv', centroids, angle_list)

    """make Tree"""  # toDO: should be a function
    objindex_list = list(range(len(sakamoto.objects_list.cdmodel_list)))
    candidate_list = sakamoto.getCandidate_list(objindex_list)  # 制約なしの把持する物体のくみあわせの候補を決定

    # 木の作成
    sakamoto.make_dp_table(len(objindex_list))  # 動的計画法

    start = time.time()  # debug
    sakamoto.buildTree(sakamoto.tree, sakamoto.root_node, candidate_list, sakamoto.mincost_leaf)
    elapsed_time = time.time() - start  # debug
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")  # debug
    method_list = sakamoto.mincost_leaf.node.getRouteNodesMethod()
    print(method_list, "cost:", sakamoto.mincost_leaf.node.cost)
    # showTree(tree, show=False)
    # sakamoto.mincost_leaf.node.writeMethodCSV("methodlistdouble.csv")

    num_of_try = 1
    for num in range(num_of_try):
        """get motion"""
        # action_list = ActionList()
        motion_data = MotionData(sakamoto.robot)
        # robot.goto_given_conf(jnt_values_above_box)
        # joints_value_above_box = robot.get_jnt_values()
        for method in method_list[1:]:
            print("\n", method)
            if method[1] == "double":
                i, j = method[0]
                grasp_pos = (sakamoto.objects_list.centroids[i] + sakamoto.objects_list.centroids[j]) / 2
                grasp_pos[2] += 0.05#物体の位置の調整が必要
                angle = (sakamoto.objects_list.angle_list[i] + sakamoto.objects_list.angle_list[j]) / 2
                grasp = [grasp_pos, angle, [i, j]]
                motion_data_double_grasp = sakamoto.PickandRecovMotion(grasp)
                if motion_data_double_grasp is not None:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([0, 0, 1, 1])
                    sakamoto.objects_list.cdmodel_list[j].rgba = np.array([0, 0, 1, 1])
                    motion_data.extend(motion_data_double_grasp)
                else:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([0, 0, 1, 0.8])
                    sakamoto.objects_list.cdmodel_list[j].rgba = np.array([0, 0, 1, 0.8])
                    # action_list.append(Action("double", motion_data_double_grasp))

            elif method[1] == "push":
                i, j = method[0]
                ###pushingの高さを変更
                motion_data_push_grasp = sakamoto.PushandRecovMotion(i, j)
                if motion_data_push_grasp is not None:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([0, 1, 0, 1])
                    sakamoto.objects_list.cdmodel_list[j].rgba = np.array([0, 1, 0, 1])
                    motion_data.extend(motion_data_push_grasp)
                else:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([0, 1, 0, 0.8])
                    sakamoto.objects_list.cdmodel_list[j].rgba = np.array([0, 1, 0, 0.8])
                    # action_list.append(Action("push", motion_data_push_grasp))

            elif method[1] == "mono-single":
                i = method[0][0]
                grasp_pos = sakamoto.objects_list.centroids[i]
                grasp_pos[2] += 0.05
                grasp = [sakamoto.objects_list.centroids[i], sakamoto.objects_list.angle_list[i], [i]]

                motion_data_single_grasp = sakamoto.PickandRecovMotion(grasp=grasp)
                if motion_data_single_grasp is not None:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([1, 0, 0, 1])
                    motion_data.extend(motion_data_single_grasp)
                else:
                    sakamoto.objects_list.cdmodel_list[i].rgba = np.array([1, 0, 0, 0.8])
                    # action_list.append(Action("mono-single", motion_data_single_grasp))

            else:
                warning_sentence = termcolor.colored('Warning: invalid method!', 'red')
                print(warning_sentence)

        # print(f"anime_path:{motion_data}")
        print("-----simulation-----")
        anime_data = Data(motion_data)
        print(f"len(anime_data.mot_data.mesh_list):{len(anime_data.mot_data.mesh_list)}")


    def update(anime_data, sakamoto, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            anime_data.counter = 0

        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(sakamoto.base)
        mesh_model.show_cdprim()
        if sakamoto.base.inputmgr.keymap['space']:
            anime_data.counter += 1
        # if base.inputmgr.keymap['g']:
        #     robotx.move_jntspace_path(anime_data.mot_data.jv_list, control_frequency=.005)
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data, sakamoto],
                          appendTask=True)

    base.run()
