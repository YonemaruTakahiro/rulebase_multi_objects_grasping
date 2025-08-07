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

TABLE_Z = 1.094

BOX_POS = np.array([0.4, -0.15, TABLE_Z])
RS_POS = np.array([0.4, -0.47, TABLE_Z + 0.6])
CENTOR_WORKSPACE = np.array([0.4, -0.5, TABLE_Z])
DISTANCE_OF_DOUBLEPICK = 0.08
ANGLEDIFF_OF_DOUBLEPICK = np.pi / 2
ALIGN_DISTANCE = 0.2
PUSHER_THICKNESS = 0.02

RANDOM = True

class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data



def getCandidate_list(index_list):  # [0, 1, 2] -> [(0), (1), (2), (0, 1), (1, 2), (0, 2)]
    comb_list = itertools.combinations(index_list, 2)
    candidate_list = [(n,) for n in index_list] + [comb for comb in comb_list]
    return candidate_list


class Node:
    def __init__(self, index, target_obj_pair_list=None, angle_list=None):
        self.index = index
        self.target_obj_pair_list = target_obj_pair_list
        self.angle_list = angle_list  # toDO: delete this parameter
        self.method = "not-decided"
        self.cost = 0.0
        self.route = [self]

    def getRouteNodesIndex(self):
        index_list = []
        for node in self.route:
            index_list.append(node.index)
        return index_list

    def getRouteNodesMethod(self):
        method_list = []
        for node in self.route:
            method_list.append((node.index, node.method))
        return method_list

    def writeMethodCSV(self, filepath):
        with open(filepath, 'w') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            for node in self.route:
                if node.index == "ROOT":
                    writer.writerow([str(node.index), "", str(node.method)])
                elif len(node.index) == 2:
                    writer.writerow([str(node.index[0]), str(node.index[1]), str(node.method)])
                elif len(node.index) == 1:
                    writer.writerow([str(node.index[0]), "", str(node.method)])


class MinCostLeaf:
    def __init__(self):
        self.node = None


class DPTable:
    def __init__(self, num):
        self.table = [[[None for i in range(3)] for j in range(num)] for k in range(num)]

    def set(self, idx_tuple, index, method, edgecost):
        if len(idx_tuple) == 2:
            i, j = idx_tuple[0], idx_tuple[1]
            if idx_tuple[0] > idx_tuple[1]:
                i, j = j, i
            self.table[i][j][0], self.table[i][j][1], self.table[i][j][2] = index, method, edgecost
        elif len(idx_tuple) == 1:
            i = idx_tuple[0]
            self.table[i][i][0], self.table[i][i][1], self.table[i][i][2] = index, method, edgecost

    def get(self, idx_tuple):
        if len(idx_tuple) == 2:
            i, j = idx_tuple[0], idx_tuple[1]
            if idx_tuple[0] > idx_tuple[1]:
                i, j = j, i
            return self.table[i][j]
        elif len(idx_tuple) == 1:
            i = idx_tuple[0]
            return self.table[i][i]

    def isNone(self, idx_tuple):
        if len(idx_tuple) == 2:
            i, j = idx_tuple[0], idx_tuple[1]
            if idx_tuple[0] > idx_tuple[1]:
                i, j = j, i
            return bool(self.table[i][j] == [None, None, None])
        elif len(idx_tuple) == 1:
            return bool(self.table[idx_tuple[0]][idx_tuple[0]] == [None, None, None])

    # idx_tupleは今見ている物体の組み合わせ、obj_listはすべての把持対象物体
    def getMethod(self, base, idx_tuple, aboveboxpos, target_obj_pair_1, obj_list, objangle_list,
                  DISTANCE_OF_DOUBLEPICK,
                  ANGLEDIFF_OF_DOUBLEPICK, ALIGN_DISTANCE, PUSHER_THICKNESS):
        if self.isNone(idx_tuple) == True:
            index, method, edgecost = selectMethod(base, idx_tuple, aboveboxpos, target_obj_pair_1, obj_list,
                                                   objangle_list,
                                                   DISTANCE_OF_DOUBLEPICK=DISTANCE_OF_DOUBLEPICK,
                                                   ANGLEDIFF_OF_DOUBLEPICK=ANGLEDIFF_OF_DOUBLEPICK,
                                                   ALIGN_DISTANCE=ALIGN_DISTANCE, PUSHER_THICKNESS=PUSHER_THICKNESS)
            self.set(idx_tuple, index, method, edgecost)
            return index, method, edgecost
        else:
            [index, method, edgecost] = self.get(idx_tuple)
            return index, method, edgecost


#############ここを書き換える必要############################
def selectMethod(base, index_tuple, aboveboxpos, target_obj_pair_1, obj_list, objangle_list, DISTANCE_OF_DOUBLEPICK=0.08,
                 ANGLEDIFF_OF_DOUBLEPICK=np.pi / 2, ALIGN_DISTANCE=0.2, PUSHER_THICKNESS=0.02):
    print(f"len(index_tuple):{len(index_tuple)}")
    if len(index_tuple) == 2:
        i = index_tuple[0]
        j = index_tuple[1]

        dist = np.linalg.norm(np.array(obj_list[i]._pos) - np.array(obj_list[j]._pos))
        angledif = abs(objangle_list[i] - objangle_list[j])
        if dist < DISTANCE_OF_DOUBLEPICK and angledif < ANGLEDIFF_OF_DOUBLEPICK:
            if not checkCollisionDP(obj_list[i], obj_list[j], obj_list):
                cost = np.linalg.norm(aboveboxpos - (np.array(obj_list[i]._pos) + np.array(obj_list[j]._pos)) / 2) * 2
                return (i, j), "double", cost

        if obj_list[i]._pos[1] < obj_list[j]._pos[1]:
            i, j = j, i
        objpos_pushed, _, cor, radius_init = pag.constrainedPushingPos(obj_list[i]._pos, obj_list[j]._pos,
                                                                       objangle_list[i], objangle_list[j],
                                                                       ALIGN_DISTANCE, PUSHER_THICKNESS)
        # objrot_pushed = np.dot(rm.rotmat_from_axangle([0, 0, 1], angle_list[j]), rm.rotmat_from_axangle([0, 1, 0], -np.pi/2))
        objrot_pushed = rm.rotmat_from_axangle([0, 0, 1], angle_list[j])  # 押す物体の目標姿勢
        objmat4_pushed = rm.homomat_from_posrot(pos=objpos_pushed, rotmat=objrot_pushed)
        obj_pushed = copy.deepcopy(target_obj_pair_1)
        obj_pushed.rgba = np.array([0, 1, 0, 0.6])
        obj_pushed.homomat = objmat4_pushed
        """r_th, d_th, theta_th(parameters threthoulds of constraint1) and alpha(friction angle) and num(num of complementing motions in pushing)"""
        ############################微調整が必要######################################################################3
        r_th = 0.08
        d_th = 0.4
        theta_th = np.pi / 2
        alpha = np.radians(60)  # friction angle
        divnum = 4
        pushing_Z = 0.88
        ##########################################################################################################
        if pag.constraint1(obj_list[i], obj_pushed, obj_list[j], objangle_list[i], objangle_list[j], cor, r_th, d_th,
                           theta_th) is True:
            compobj_list, compangle_list = pag.complementConfig(cor, target_obj_pair_1, obj_list[i], obj_pushed,
                                                                objangle_list[i], objangle_list[j],
                                                                divnum=divnum)  # 補間点のオブジェクトを追加

            eepos_list = pag.constraint2(base,compobj_list, compangle_list, alpha, PUSHER_WIDTH=0.085,
                                         PUSHER_THICKNESS=PUSHER_THICKNESS, PUSHING_Z=pushing_Z)
            if eepos_list is not None:
                if not checkCollisionPSNG(compobj_list, obj_list[i], obj_list[j], obj_list):
                    cost = np.linalg.norm(aboveboxpos - obj_list[i]._pos) + np.linalg.norm(
                        obj_list[i]._pos - obj_list[j]._pos) + np.linalg.norm(aboveboxpos - obj_list[j]._pos)
                    return (i, j), "push", cost
            print("constraint2 is False")
        else:
            print("constraint1 is False")

        return (i, j), "not-available", float('inf')

    elif len(index_tuple) == 1:
        i = index_tuple[0]
        cost = np.linalg.norm(aboveboxpos - np.array(obj_list[i]._pos)) * 2
        return (i,), "mono-single", cost

    else:
        print("the length of index_tuple should be 2 or 1 !")
        return (), "not-available", float('inf')


def buildTree(base, tree, parent_node, candidate_list, mincost_leaf, obj_list, objangle_list, aboveboxpos,
              makenxgraph=False):
    if parent_node.cost == float('inf'):
        return

    if len(candidate_list) == 0:
        if mincost_leaf.node is None:
            mincost_leaf.node = copy.deepcopy(parent_node)
            return
        else:
            if parent_node.cost < mincost_leaf.node.cost:
                mincost_leaf.node = copy.deepcopy(parent_node)
            return

    if mincost_leaf.node is not None:
        if parent_node.cost > mincost_leaf.node.cost:
            return

    for i_tuple in candidate_list:
        # 今見ている組み合わせの物体とその角度をノードとして追加
        newnode = Node(i_tuple, target_obj_pair_list=[obj_list[idx] for idx in i_tuple],
                       angle_list=[objangle_list[idx] for idx in i_tuple])
        newnode.index, newnode.method, edgecost = dp.getMethod(base, i_tuple, aboveboxpos,
                                                               newnode.target_obj_pair_list[0],
                                                               obj_list, objangle_list, DISTANCE_OF_DOUBLEPICK,
                                                               ANGLEDIFF_OF_DOUBLEPICK, ALIGN_DISTANCE,
                                                               PUSHER_THICKNESS)
        newnode.cost = parent_node.cost + edgecost
        newnode.route = parent_node.route + [newnode]
        if makenxgraph:
            tree.add_node(newnode,
                          label=str(newnode.index) + "\n" + newnode.method + "\n" + str(round(newnode.cost, 2)))
            tree.add_edge(parent_node, newnode, weight=edgecost, label=str(round(edgecost, 2)))
        temp_list = [j_tuple for j_tuple in candidate_list if set(i_tuple) & set(j_tuple) == set()]
        buildTree(base, tree, newnode, temp_list, mincost_leaf, obj_list, objangle_list, aboveboxpos)


def checkCollisionDP(obj_i, obj_j, obj_list, divnum=3):
    other_list = [obj for obj in obj_list if obj != obj_i and obj != obj_j]
    comp_list = []
    for n in range(0, divnum):
        pos = (divnum - n) / divnum * np.array(obj_i._pos) + n / divnum * np.array(obj_j._pos)
        obj = copy.deepcopy(obj_i)
        obj.pos = pos
        comp_list.append(obj)
    colcheck = oc.is_collided(comp_list, other_list, toggle_contacts=False)
    return colcheck


def checkCollisionPSNG(compobj_list, obj_initial, obj_target, obj_list):
    other_list = [obj for obj in obj_list if obj != obj_initial and obj != obj_target]
    colcheck = oc.is_collided(compobj_list, other_list, toggle_contacts=False)
    print("PSNG", colcheck)
    return colcheck


def readMethodCSV(filepath):
    method_list = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        for line in reader:
            if line[0] == "ROOT":
                index = "ROOT"
            else:
                if line[1] == "":
                    index = (int(line[0]),)
                else:
                    index = (int(line[0]), int(line[1]))
            method = line[2]
            method_list.append((index, method))
    return method_list


class ActionList:
    def __init__(self):
        self.aclist = []

    def append(self, action):
        self.aclist.append(action)

    def getAnimationPath(self):
        anipath = []
        for ac in self.aclist:
            if ac.method == "double":
                anipath.append(ac.path_list)
            elif ac.method == "mono-single":
                anipath.append(ac.path_list)
            elif ac.method == "push":
                anipath.append(ac.path_list)
        return anipath

    def evaluatePath(self, base, simrobot, armname="rgt", drawpoints=True):
        eepos_list = []
        pose_list = []
        for ac in self.aclist:
            if ac.method == "double" or ac.method == "mono-single":
                for path in ac.path_list:
                    if path is not None:
                        for pose in path:
                            robot.movearmfk(pose, armname)  # fkのみ
                            eepos, _ = simrobot.getee(armname)
                            eepos_list.append(eepos)
                            pose_list.append(pose)
                            if drawpoints == True:
                                rgba = np.array([0.0, 0.0, 0.0, 0.5])
                                if ac.method == "double":
                                    rgba = np.array([255.0, 0.0, 0.0, 0.5])
                                elif ac.method == "mono-single":
                                    rgba = np.array([0.0, 0.0, 255.0, 0.5])
                                base.pggen.plotSphere(base.render, pos=eepos, radius=10, rgba=rgba, plotname="sphere")
            elif ac.method == "push":
                for path_2dlst in ac.path_list:
                    if path_2dlst is not None:
                        for path in path_2dlst:
                            for pose in path:
                                robot.movearmfk(pose, armname)
                                eepos, _ = simrobot.getee(armname)
                                eepos_list.append(eepos)
                                pose_list.append(pose)
                                if drawpoints == True:
                                    base.pggen.plotSphere(base.render, pos=eepos, radius=10,
                                                          rgba=np.array([0.0, 255.0, 0.0, 0.5]), plotname="sphere")
        sum_ee = 0.0
        for i in range(len(eepos_list) - 1):
            sum_ee += np.linalg.norm(eepos_list[i + 1] - eepos_list[i])
        sum_PL = 0.0
        for i in range(len(pose_list) - 1):
            sum_PL += np.linalg.norm(np.array(pose_list[i + 1]) - np.array(pose_list[i]))
        return sum_ee, sum_PL


class Action:
    def __init__(self, method="not-decided", path_list=None):
        self.method = method
        self.path_list = path_list


###################mscの追加
##################radianに統一#################################################
if __name__ == "__main__":

    base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
    mcm.mgm.gen_frame().attach_to(base)

    # robot
    robot = ur3d.UR3Dual()
    robot.use_rgt()
    robot.change_jaw_width(jaw_width=0.08)
    robot.gen_meshmodel(toggle_jnt_frames=True).attach_to(base)
    init_tgt_pos = np.array([BOX_POS[0], BOX_POS[1], BOX_POS[2] + 0.3])
    init_tgt_rotmat = rm.rotmat_from_euler(rm.pi, 0, 0)
    mcm.mgm.gen_frame(pos=init_tgt_pos, rotmat=init_tgt_rotmat).attach_to(base)
    jnt_values_above_box = robot.ik(tgt_pos=init_tgt_pos, tgt_rotmat=init_tgt_rotmat)
    # print(f"jnt_values:{jnt_values}")
    robot.goto_given_conf(jnt_values=jnt_values_above_box)
    robot.gen_meshmodel().attach_to(base)

    # box
    box = cm.CollisionModel("/home/yonemaru/PycharmProjects/wrs/yonemaru/objects/yonemaru_box.stl",
                            cdprim_type=const.CDPrimType.CAPSULE)
    boxrot = rm.rotmat_from_axangle([0, 0, 1], np.pi / 2)
    boxmat4 = rm.homomat_from_posrot(pos=BOX_POS, rotmat=boxrot)
    box.homomat = boxmat4
    box.rgba = np.array([0.87, 0.65, 0.53, 1])
    box.attach_to(base)
    # obj_3dprinted for picking
    obj_3dprinted = cm.CollisionModel("/home/yonemaru/PycharmProjects/wrs/yonemaru/objects/yonemaru_object1.stl",
                                      cdprim_type=const.CDPrimType.CAPSULE)

    if RANDOM:  # set at random
        r_range = (0.0, 0.2)
        theta_range = (0, 2 * math.pi)
        centroids, angle_list, obj_list, objangle_list = pp.placeRandomRoundly(base, box, obj_3dprinted, 0.02, 4,
                                                                               CENTOR_WORKSPACE,
                                                                               r_range, theta_range, TABLE_Z,
                                                                               simrobot=robot)

        obscmlist = [box] + obj_list
        pp.writePlacePattern('placepattern.csv', centroids, angle_list)

        """make Tree"""  # toDO: should be a function
        objindex_list = list(range(len(obj_list)))
        candidate_list = getCandidate_list(objindex_list)  # 制約なしの把持する物体のくみあわせの候補を決定

        # 木の作成
        tree = nx.DiGraph()  # ライブラリ
        root_node = Node("ROOT")  # ライブラリ
        tree.add_node(root_node, label="ROOT")  # ライブラリ
        # base.run()
        mincost_leaf = MinCostLeaf()  # ライブラリ
        dp = DPTable(len(obj_list))  # 動的計画法

        start = time.time()  # debug
        buildTree(base,tree, root_node, candidate_list, mincost_leaf, obj_list, objangle_list, init_tgt_pos)
        elapsed_time = time.time() - start  # debug
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")  # debug
        method_list = mincost_leaf.node.getRouteNodesMethod()
        print(method_list, "cost:", mincost_leaf.node.cost)
        # showTree(tree, show=False)
        mincost_leaf.node.writeMethodCSV("methodlistdouble.csv")

    else:  # read from .csv
        centroids, angle_list, obj_list, objangle_list = pp.readPlacePattern(base, 'placepattern.csv', obj_3dprinted,
                                                                             0.02)
        obscmlist = [box] + obj_list
        method_list = readMethodCSV("methodlistsingle.csv")

    num_of_try = 1
    for num in range(num_of_try):
        """get motion"""
        # action_list = ActionList()
        motion_data=MotionData(robot)
        robot.goto_given_conf(jnt_values_above_box)
        joints_value_above_box = robot.get_jnt_values()
        for method in method_list[1:]:
            print("\n", method)
            if method[1] == "double":
                i, j = method[0]
                grasp_pos = (centroids[i] + centroids[j]) / 2
                grasp_pos[2] += 0.04
                angle = (angle_list[i] + angle_list[j]) / 2
                grasp = [grasp_pos, angle, [i, j]]
                motion_data_double_grasp = pag.PickandRecovMotion(base, robot, obscmlist, joints_value_above_box, grasp)
                if motion_data_double_grasp is not None:
                    joints_value_above_box = motion_data_double_grasp._jv_list[-1]
                    obj_list[i].rgba = np.array([0, 0, 1, 0.8])
                    obj_list[j].rgba = np.array([0, 0, 1, 0.8])
                    motion_data.extend(motion_data_double_grasp)
                    # action_list.append(Action("double", motion_data_double_grasp))

            elif method[1] == "push":
                i, j = method[0]
                ###pushingの高さを変更
                motion_data_push_grasp = pag.PushandRecovMotion(base, robot, obscmlist, joints_value_above_box, obj_3dprinted,
                                                   obj_list[i], obj_list[j], objangle_list[i], objangle_list[j],
                                                   ALIGN_DISTANCE=0.02, PUSHER_THICKNESS=0.02)
                if motion_data_push_grasp is not None:
                    obj_list[i].rgba = np.array([0, 1, 0, 0.6])
                    obj_list[j].rgba = np.array([0, 1, 0, 0.8])
                    joints_value_above_box = motion_data_push_grasp[-1]
                    motion_data.extend(motion_data_push_grasp)
                    # action_list.append(Action("push", motion_data_push_grasp))

            elif method[1] == "mono-single":
                i = method[0][0]
                grasp_pos = centroids[i]
                grasp_pos[2] += 0.04
                grasp = [centroids[i], angle_list[i], [i]]

                motion_data_single_grasp = pag.PickandRecovMotion(base=base, simrobot=robot, obscmlist=obscmlist, start_joints=joints_value_above_box, grasp=grasp)
                if motion_data_single_grasp is not None:
                    obj_list[i].rgba = np.array([0, 0, 1, 0.8])
                    joints_value_above_box = motion_data_single_grasp._jv_list[-1]
                    motion_data.extend(motion_data_single_grasp)
                    # action_list.append(Action("mono-single", motion_data_single_grasp))

            else:
                warning_sentence = termcolor.colored('Warning: invalid method!', 'red')
                print(warning_sentence)

        # print(f"anime_path:{motion_data}")
        print("-----simulation-----")
        anime_data=Data(motion_data)


    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            anime_data.counter = 0
        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(base)
        mesh_model.show_cdprim()
        if base.inputmgr.keymap['space']:
            anime_data.counter += 1
        # if base.inputmgr.keymap['g']:
        #     robotx.move_jntspace_path(anime_data.mot_data.jv_list, control_frequency=.005)
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)


    base.run()
