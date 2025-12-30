from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.modeling import collision_model as cm
import wrs.modeling.constant as const
import numpy as np
import math
import placepattern as pp
# import pushing_and_grasping as pag
import itertools
import networkx as nx
import csv
import time
import copy
import termcolor

from wrs.modeling import _ode_cdhelper as oc
from wrs.motion.primitives.approach_depart_planner import ADPlanner
# from wrs.motion.primitives.interpolated import InterplatedMotion
from wrs.motion.motion_data import MotionData
import motion as mt

import numpy as np
import copy
import math
import random
import csv
from scipy.spatial.transform import Rotation
from wrs import wd, rm, ur3d, rrtc, mgm, mcm


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


class objects_state:
    def __init__(self):
        self.cdmodel_list = []
        self.angle_list = []
        self.centroids = []


class Sakamoto:
    def __init__(self, base, robot, table_z, box, box_pos, printed_obj, objheight,
                 distance_from_objcor_to_pushing_surface, centor_workspace,
                 distance_doublepick, angle_difference_doble_pick,
                 align_distance,
                 pusher_thickness, aboveboxpos, jnt_values_above_box, r_th=0.08, d_th=0.4, theta_th=np.pi / 2,
                 alpha=np.pi / 3, divnum=4,
                 pushing_Z=0.88, pusher_width=0.085):
        ###タスクパラメータ##
        self.base = base
        self.table_z = table_z
        self.box = box
        self.box_pos = box_pos
        self.printed_obj = printed_obj
        self.centor_workspace = centor_workspace
        self.distance_doublepick = distance_doublepick
        self.angle_difference_doble_pick = angle_difference_doble_pick
        self.align_distance = align_distance
        self.pusher_thickness = pusher_thickness
        self.objheight = objheight
        self.distance_from_objcor_to_pushing_surface = distance_from_objcor_to_pushing_surface
        self.aboveboxpos = aboveboxpos
        self.jnt_values_above_box = jnt_values_above_box
        self.r_th = r_th
        self.d_th = d_th
        self.theta_th = theta_th
        self.alpha = alpha  # friction angle
        self.divnum = divnum
        self.pushing_Z = pushing_Z
        self.pusher_width = pusher_width

        # robot
        self.robot = robot

        # ad_planer
        self.adplaner = ADPlanner(self.robot)

        # object
        self.objects_list = objects_state()
        
        # obstacle
        self.obstacle_list = []
        self.obstacle_list.append(self.box)

        # DP_table
        self.dp_table = None

        # 木の初期化
        self.tree = nx.DiGraph()
        self.root_node = Node("ROOT")
        self.tree.add_node(self.root_node, label="ROOT")
        self.mincost_leaf = MinCostLeaf()

    def make_dp_table(self, num_obj):
        self.dp_table = DPTable(num_obj)

    def placeRandomRoundly(self, num, x_range: tuple, y_range: tuple, liftheight=0.03):

        centroids = []
        angle_list = []
        obj_list = []
        objangle_list = []

        picking_pose = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        while len(centroids) < num:
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            centroid = np.array([self.centor_workspace[0], self.centor_workspace[1], self.table_z]) + np.array(
                [x, y, 0])
            angle = random.uniform(-np.pi,
                                   0)  # todo change based on your coordination and robot end_effector_local_rotmat
            rotvec = np.array([0, 0, -angle])
            rot = Rotation.from_rotvec(rotvec)
            rotmat = rot.as_matrix()
            orientation_lft_moveto = np.dot(picking_pose, rotmat)
            liftpos = centroid + np.array([0, 0, liftheight])
            jntangle_rgt_liftpoint = self.robot.ik(tgt_pos=liftpos, tgt_rotmat=orientation_lft_moveto)

            if jntangle_rgt_liftpoint is not None:
                objangle = angle  # angle:rad objangle:deg
                objpos = centroid
                # objrot = np.dot(rm.rotmat_from_axangle([0, 0, 1], objangle), rm.rotmat_from_axangle([0, 1, 0], -90))
                objrot = rm.rotmat_from_axangle([0, 0, 1], objangle)
                objmat4 = rm.homomat_from_posrot(pos=objpos, rotmat=objrot)

                obj = copy.deepcopy(self.printed_obj)
                obj.rgba = np.array([1, 0, 0, 1])
                obj.homomat = objmat4

                if not obj.is_mcdwith(obj_list) and not self.box.is_mcdwith(obj_list):
                    centroids.append(centroid)
                    angle_list.append(angle)

                    mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(self.base)
                    obj.attach_to(self.base)

                    obj_list.append(obj)
                    objangle_list.append(objangle)

                self.objects_list.cdmodel_list = obj_list
                self.objects_list.angle_list = angle_list
                self.objects_list.centroids = centroids

    # return centroids, angle_list, obj_list, objangle_list

    def selectMethod(self, index_tuple, target_obj_pair_1):
        if len(index_tuple) == 2:
            i = index_tuple[0]
            j = index_tuple[1]

            dist = np.linalg.norm(
                np.array(self.objects_list.cdmodel_list[i].pos) - np.array(self.objects_list.cdmodel_list[j].pos))
            angledif = abs(self.objects_list.angle_list[i] - self.objects_list.angle_list[j])
            if dist < self.distance_doublepick and angledif < self.angle_difference_doble_pick:
                if not self.checkCollisionDP(self.objects_list.cdmodel_list[i], self.objects_list.cdmodel_list[j],
                                             self.objects_list.cdmodel_list):
                    cost = np.linalg.norm(
                        self.aboveboxpos - (self.objects_list.cdmodel_list[i].pos + self.objects_list.cdmodel_list[
                            j].pos) / 2) * 2
                    return (i, j), "double", cost
            # 座標系の向きによって調整
            # todo change this value based on workspace coordination and end_effector_local_rotmat
            if self.objects_list.cdmodel_list[i].pos[1] > self.objects_list.cdmodel_list[j].pos[1]:
                i, j = j, i

            objpos_pushed, _, cor, radius_init = self.constrainedPushingPos(i, j)
            # objrot_pushed = np.dot(rm.rotmat_from_axangle([0, 0, 1], angle_list[j]), rm.rotmat_from_axangle([0, 1, 0], -np.pi/2))
            objrot_pushed = rm.rotmat_from_axangle([0, 0, 1], self.objects_list.angle_list[j])  # 押す物体の目標姿勢
            objmat4_pushed = rm.homomat_from_posrot(pos=objpos_pushed, rotmat=objrot_pushed)
            obj_pushed = copy.deepcopy(target_obj_pair_1)
            # obj_pushed.rgba = np.array([0, 1, 0, 0.6])
            obj_pushed.homomat = objmat4_pushed
            """r_th, d_th, theta_th(parameters threthoulds of constraint1) and alpha(friction angle) and num(num of complementing motions in pushing)"""
            if self.constraint1(i, obj_pushed, j, cor):
                compobj_list, compangle_list = self.complementConfig(cor, target_obj_pair_1,
                                                                     i, obj_pushed, j)  # 補間点のオブジェクトを追加

                eepos_list = self.constraint2(compobj_list, compangle_list)
                if eepos_list is not None:
                    if not self.checkCollisionPSNG(compobj_list, self.objects_list.cdmodel_list[i],
                                                   self.objects_list.cdmodel_list[j], self.objects_list.cdmodel_list):
                        cost = np.linalg.norm(
                            self.aboveboxpos - self.objects_list.cdmodel_list[i].pos) + np.linalg.norm(
                            self.objects_list.cdmodel_list[i].pos - self.objects_list.cdmodel_list[
                                j].pos) + np.linalg.norm(self.aboveboxpos - self.objects_list.cdmodel_list[j].pos)
                        return (i, j), "push", cost
                print("constraint2 is False")
            else:
                print("constraint1 is False")

            return (i, j), "not-available", float('inf')

        elif len(index_tuple) == 1:
            i = index_tuple[0]
            cost = np.linalg.norm(self.aboveboxpos - np.array(self.objects_list.cdmodel_list[i].pos)) * 2
            return (i,), "mono-single", cost

        else:
            print("the length of index_tuple should be 2 or 1 !")
            return (), "not-available", float('inf')

    # idx_tupleは今見ている物体の組み合わせ、obj_listはすべての把持対象物体
    def getMethod(self, idx_tuple, target_obj_pair_1):
        if self.dp_table.isNone(idx_tuple) is True:
            index, method, edgecost = self.selectMethod(idx_tuple, target_obj_pair_1)
            self.dp_table.set(idx_tuple, index, method, edgecost)
            return index, method, edgecost
        else:
            [index, method, edgecost] = self.dp_table.get(idx_tuple)
            return index, method, edgecost

    def buildTree(self, tree, parent_node, candidate_list, mincost_leaf, makenxgraph=False):
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
            newnode = Node(i_tuple, target_obj_pair_list=[self.objects_list.cdmodel_list[idx] for idx in i_tuple],
                           angle_list=[self.objects_list.angle_list[idx] for idx in i_tuple])

            if self.dp_table is None:
                raise NotImplementedError

            newnode.index, newnode.method, edgecost = self.getMethod(i_tuple, newnode.target_obj_pair_list[0])
            newnode.cost = parent_node.cost + edgecost
            newnode.route = parent_node.route + [newnode]
            if makenxgraph:
                tree.add_node(newnode,
                              label=str(newnode.index) + "\n" + newnode.method + "\n" + str(round(newnode.cost, 2)))
                tree.add_edge(parent_node, newnode, weight=edgecost, label=str(round(edgecost, 2)))
            temp_list = [j_tuple for j_tuple in candidate_list if set(i_tuple) & set(j_tuple) == set()]
            self.buildTree(self.tree, newnode, temp_list, mincost_leaf)

    def checkCollisionDP(self, obj_i, obj_j, obj_list):
        other_list = [obj for obj in obj_list if obj != obj_i and obj != obj_j]
        comp_list = []
        for n in range(0, self.divnum):
            pos = (self.divnum - n) / self.divnum * np.array(obj_i.pos) + n / self.divnum * np.array(obj_j.pos)
            obj = copy.deepcopy(obj_i)
            obj.pos = pos
            comp_list.append(obj)
        colcheck = oc.is_collided(comp_list, other_list, toggle_contacts=False)
        return colcheck

    def complementConfig(self, cor, target_obj_pair_1, i, obj_pushed, j):
        initial_pos = np.array(
            [self.objects_list.cdmodel_list[i].pos[0], self.objects_list.cdmodel_list[i].pos[1],
             self.objects_list.cdmodel_list[i].pos[2]])
        obj_list = [self.objects_list.cdmodel_list[i]]
        angle_list = [self.objects_list.angle_list[i]]
        divangle = (self.objects_list.angle_list[j] - self.objects_list.angle_list[
            i]) / self.divnum  # 押す対象物体の押す前と押した後の角度(姿勢)の分割数
        for n in range(1, self.divnum):  # 押し始めから押し終わるまでの中点を計算
            R = np.array([
                [np.cos(divangle * n), -np.sin(divangle * n), 0],
                [np.sin(divangle * n), np.cos(divangle * n), 0],
                [0, 0, 1]
            ])
            pos = np.dot(R, (initial_pos - cor)) + cor
            angle = ((self.divnum - n) * self.objects_list.angle_list[i] + n * self.objects_list.angle_list[
                j]) / self.divnum  # 角度の次元における中点

            interp_objpos = pos
            interp_objrot = rm.rotmat_from_axangle([0, 0, 1], angle)
            interp_objmat4 = rm.homomat_from_posrot(interp_objpos, interp_objrot)
            interp_obj = copy.deepcopy(target_obj_pair_1)
            # interp_obj.rgba = np.array([0, 1, 0, 0.4])
            interp_obj.homomat = interp_objmat4

            obj_list.append(interp_obj)
            angle_list.append(angle)
        obj_list.append(obj_pushed)
        angle_list.append(self.objects_list.angle_list[j])
        return obj_list, angle_list

    def constraint1(self, i, obj_pushed, j, cor):
        """constraint1 : the configuration of object_pair1, obj_pushed, object_pair2 is close"""
        radius_initial = np.linalg.norm(
            np.array([self.objects_list.cdmodel_list[i].pos[0], self.objects_list.cdmodel_list[i].pos[1]]) - np.array(
                [cor[0], cor[1]]))
        radius_target = np.linalg.norm(
            np.array([self.objects_list.cdmodel_list[j].pos[0], self.objects_list.cdmodel_list[j].pos[1]]) - np.array(
                [cor[0], cor[1]]))
        r = abs(radius_target - radius_initial)  # same as |rB-rA|
        d = np.linalg.norm(
            [obj_pushed.pos[1] - self.objects_list.cdmodel_list[i].pos[1],
             obj_pushed.pos[0] - self.objects_list.cdmodel_list[i].pos[0]])  # same as ||xA'-xA||
        theta = abs(self.objects_list.angle_list[j] - self.objects_list.angle_list[i])  # same as |thetaA'-thetaA|

        print("r, r_th", r, self.r_th)
        print("d, d_th", d, self.d_th)
        print("theta, theta_th", theta, self.theta_th)
        if r < self.r_th and d < self.d_th and theta < self.theta_th:
            return True
        else:
            return False

    def constraint2(self, comp_obj_list, comp_angle_list):
        """constraint2 : constraint of e_v(unit vector of Pusher's velocity), COR(Center of Rotation)"""
        """2-1, 2-2 True -> return eepos_list  False-> return None"""
        eepos_list = []

        if (len(comp_obj_list) != len(comp_angle_list)):
            raise ValueError("the lens of comp_obj_list and angle_list are different")
        if (len(comp_obj_list) < 2):
            raise ValueError("the lens of comp_obj_list must be least 2")

        """calc push pos for each motion"""
        for i in range(len(comp_obj_list)):
            pos = np.array(comp_obj_list[i].pos)
            rotmat = rm.rotmat_from_axangle([0, 0, 1], comp_angle_list[i])
            # xF = pos + np.dot(rotmat, np.array([0, 0.01, 0.01]))
            # mgm.gen_sphere(pos=xF, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
            if i == 0:
                eepos = pos + np.dot(rotmat, np.array(
                    [self.distance_from_objcor_to_pushing_surface + self.pusher_thickness / 2 + 0.005, 0,
                     self.pushing_Z]))
                # eepos[2] = self.pushing_Z  # 押すときの高さ
            else:
                eepos = pos + np.dot(rotmat,
                                     np.array(
                                         [self.distance_from_objcor_to_pushing_surface + self.pusher_thickness / 2 + 0.005,
                                          0, self.pushing_Z]))
                # eepos[2] = self.pushing_Z  # 押すときの高さ
            # mgm.gen_sphere(pos=xF, radius=0.2, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8)
            if i == len(comp_obj_list) - 1:
                print(i, "/", len(comp_obj_list) - 1, "the last index")
                eepos[2] += 0.005
                eepos_list.append(eepos)
            else:
                """const2-1 there are no slip between pusher and object"""
                # todo change this value from .stl file
                xF1 = pos + np.dot(rotmat, np.array(
                    [self.distance_from_objcor_to_pushing_surface, -self.pusher_width / 2, self.pushing_Z]))
                xF2 = pos + np.dot(rotmat, np.array(
                    [self.distance_from_objcor_to_pushing_surface, self.pusher_width / 2, self.pushing_Z]))
                # mgm.gen_sphere(pos=xF1, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
                # mgm.gen_sphere(pos=xF2, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
                #この向きに注意
                eF = np.dot(
                    [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]],
                    np.array(xF2 - xF1))
                ev = comp_obj_list[i + 1].pos - comp_obj_list[i].pos
                # mgm.gen_arrow(spos=pos, epos=eF+pos,stick_radius=0.004, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
                # mgm.gen_arrow(spos=pos, epos=10*ev+pos,stick_radius=0.002, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
                eF = eF[0:2]
                eF = eF / np.linalg.norm(eF)
                ev = ev[0:2]
                ev = ev / np.linalg.norm(ev)
                print(f"np.dot(ev, eF):{np.dot(ev, eF)}")
                if np.dot(ev, eF) >= np.linalg.norm(ev) / np.sqrt(1 + np.tan(self.alpha) * np.tan(self.alpha)):
                    print(i, "/", len(comp_obj_list) - 1, "2-1 is True")
                else:
                    print(i, "/", len(comp_obj_list) - 1, "2-1 is False")
                    return None

                """const2-2 there are rotation on edge of pusher"""
                # todo change this value from .stl file
                ecof_list = [pos + np.dot(rotmat, np.array([0, 0.03, 0.01])),
                             pos + np.dot(rotmat, np.array([0, -0.03, 0.01])),
                             pos + np.dot(rotmat, np.array([0, 0.03, -0.01])),
                             pos + np.dot(rotmat, np.array([0, -0.03, -0.01]))]
                for ecof in ecof_list:
                    r1 = xF1 - ecof
                    r2 = xF2 - ecof
                    r1 = r1[0:2]
                    r2 = r2[0:2]
                    lambda_F = np.cross(r2, ev) / np.cross((r2 - r1), ev)
                    print("lambda_F", lambda_F)
                    xF = lambda_F * xF1 + (1 - lambda_F) * xF2
                    # mgm.gen_sphere(pos=xF, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
                    if 0 >= lambda_F or lambda_F >= 1:
                        print(i, "/", len(comp_obj_list) - 1, "2-2 is False")
                        return None
                    else:
                        print(i, "/", len(comp_obj_list) - 1, "2-2 is True")
                eepos[2] += 0.01
                eepos_list.append(eepos)

        return eepos_list

    def constrainedPushingPos(self, i, j):
        """
        get pushed object's pos in constrained pushing
        return pushed_pos(pos of A'), nexttarget_pos(pos of A''), cor(Center of Rotation), radius(radius from cor to object_pair1_pos)
        """

        rad_i = self.objects_list.angle_list[i]  # object_pair1_angle in radian
        rad_t = self.objects_list.angle_list[j]  # object_pair2_angle in radian

        """nexttarget_pos means A'', pushed_pos means A'"""  # 向きが大事
        cand1_pos = self.objects_list.cdmodel_list[j].pos + self.align_distance * np.array(
            [np.cos(rad_t), np.sin(rad_t), 0])
        cand2_pos = self.objects_list.cdmodel_list[j].pos + self.align_distance * np.array(
            [np.cos(rad_t+np.pi), np.sin(rad_t+np.pi), 0])
        # mcm.mgm.gen_sphere(pos=cand1_pos, radius=0.007, rgb=np.array([1.0, 0.0, 0.0]), alpha=0.8).attach_to(self.base)
        # mcm.mgm.gen_sphere(pos=cand2_pos, radius=0.007, rgb=np.array([0.0, 0.0, 1.0]), alpha=0.8).attach_to(self.base)
        if np.linalg.norm(cand1_pos - self.objects_list.cdmodel_list[i].pos) < np.linalg.norm(
                cand2_pos - self.objects_list.cdmodel_list[i].pos):
            nexttarget_pos = cand1_pos
        else:
            nexttarget_pos = cand2_pos

        """get COR(Center of Rotation)"""
        A = np.array([
            [np.tan(rad_i + np.pi / 2), -1.0],
            [np.tan(rad_t + np.pi / 2), -1.0]
        ])
        Y = np.array([
            [np.tan(rad_i + np.pi / 2) * self.objects_list.cdmodel_list[i].pos[0] -
             self.objects_list.cdmodel_list[i].pos[1]],
            [np.tan(rad_t + np.pi / 2) * nexttarget_pos[0] - nexttarget_pos[1]]
        ])
        cor_xy = np.linalg.solve(A, Y)
        cor = np.array([cor_xy[0][0], cor_xy[1][0], self.objects_list.cdmodel_list[i].pos[2]])
        # base.pggen.plotSphere(base.render, pos=cor, radius=20, rgba=np.array([0.0, 255.0, 0.0, 0.5]), plotname="sphere")
        radius = np.linalg.norm(
            np.array([self.objects_list.cdmodel_list[i].pos[0], self.objects_list.cdmodel_list[i].pos[1]]) - np.array(
                [cor[0], cor[1]]))
        radius_vec = np.array([nexttarget_pos[0], nexttarget_pos[1]]) - np.array([cor[0], cor[1]])  # 回転中心から次の目標位置までの距離
        radius_unitvec = radius_vec / np.linalg.norm(radius_vec)
        pushed_pos_xy = np.array([cor[0], cor[1]]) + radius * radius_unitvec
        pushed_pos = np.array(
            [pushed_pos_xy[0], pushed_pos_xy[1], self.objects_list.cdmodel_list[i].pos[2]])  # 押されて移動した物体の位置
        return pushed_pos, nexttarget_pos, cor, radius

    def checkCollisionPSNG(self, compobj_list, obj_initial, obj_target, obj_list):
        other_list = [obj for obj in obj_list if obj != obj_initial and obj != obj_target]
        colcheck = oc.is_collided(compobj_list, other_list, toggle_contacts=False)
        print("PSNG", colcheck)
        return colcheck

    @staticmethod
    def getCandidate_list(index_list):  # [0, 1, 2] -> [(0), (1), (2), (0, 1), (1, 2), (0, 2)]
        comb_list = itertools.combinations(index_list, 2)
        candidate_list = [(n,) for n in index_list] + [comb for comb in comb_list]
        return candidate_list

    def PickandRecovMotion(self, grasp,toggle_dbg=False):
        rotvec = np.array([0, 0, -grasp[1]])
        rot = Rotation.from_rotvec(rotvec)
        rotmat = rot.as_matrix()

        print("-----IK pick point-----")
        pos_rgt_moveto = grasp[0]
        picking_pose = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        picking_pose1 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        picking_pose2 = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        picking_pose_list= [picking_pose1, picking_pose2]
        use_rrt_list = [False, True]
        for use_rrt in use_rrt_list:
            print(f"-----use_rrt:{use_rrt}-----")
            for picking_pose in picking_pose_list:
                orientation_rgt_moveto = np.dot(picking_pose, rotmat)

                print("-----approach_to_objects_motion Planning-----")
                approach_to_objects_motion = self.adplaner.gen_approach(goal_tcp_pos=pos_rgt_moveto,
                                                                        goal_tcp_rotmat=orientation_rgt_moveto,
                                                                        start_jnt_values=self.jnt_values_above_box,
                                                                        linear_direction=orientation_rgt_moveto[:, 2],
                                                                        linear_distance=0.03, use_rrt=use_rrt,toggle_dbg=toggle_dbg,
                                                                        obstacle_list= self.obstacle_list
                                                                        )
                if approach_to_objects_motion is None:
                    print("cannot calc approach_to_objects_motion")
                    continue

                close_gripper_motion = MotionData(self.robot)
                close_ev_list = []
                close_jnts_list = []
                # print(f"approach_to_objects_motion.jv_list():{approach_to_objects_motion.jv_list}")
                for i in range(85):
                    close_ev_list.append(0.085 - (i * 0.085 / 85))
                    close_jnts_list.append(approach_to_objects_motion.jv_list[-1])
                close_gripper_motion.extend(close_jnts_list, close_ev_list)
                
                print("-----return_to_start_motion Planning-----")
                return_to_start_motion = self.adplaner.gen_depart_from_given_conf(start_jnt_values=approach_to_objects_motion.jv_list[-1],
                                                                    end_jnt_values= self.jnt_values_above_box,
                                                                    linear_direction=-orientation_rgt_moveto[:, 2],
                                                                    linear_distance=.03,
                                                                    ee_values=close_gripper_motion.ev_list[-1],
                                                                    use_rrt=use_rrt,
                                                                    obstacle_list= self.obstacle_list,
                                                                    toggle_dbg=toggle_dbg)
                if return_to_start_motion is None:
                    print("cannot calc return_to_start_motion")
                    continue

                open_gripper_motion = MotionData(self.robot)
                open_ev_list = []
                open_jnts_list = []
                for i in range(85):
                    open_ev_list.append(i * 0.085 / 85)
                    open_jnts_list.append(return_to_start_motion.jv_list[-1])
                open_gripper_motion.extend(open_jnts_list, open_ev_list)
                
                if approach_to_objects_motion is not None or close_gripper_motion is not None or return_to_start_motion is not None or open_gripper_motion is not None:
                    print("-----Pick and Recover Motion Planning Succeeded-----")
                    return approach_to_objects_motion + close_gripper_motion + return_to_start_motion + open_gripper_motion

                
        if approach_to_objects_motion is None or close_gripper_motion is None or return_to_start_motion is None or open_gripper_motion is None:
            print("failed to plan the pick and recover motions")
            return None
      

    def PushandRecovMotion(self, i, j):
        """path list of pushing"""
        push_motion_data = self.RotatingPushMotion(i, j)

        if push_motion_data is None:
            print("push_motion_data is None")
            return None
        jntangle_after_push = push_motion_data.jv_list[-1]  # pushingの最後の関節角
        eepos, eerot = self.robot.fk(jnt_values=jntangle_after_push)

        eepos_approach = np.array(
            [(eepos[0] + self.objects_list.cdmodel_list[j].pos[0]) / 2,
             (eepos[1] + self.objects_list.cdmodel_list[j].pos[1]) / 2,
             eepos[2]])

        self.objects_list.cdmodel_list[j].rgba = np.array([1, 1, 0, 1])

        picking_rot = np.dot(np.array([
            [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
            [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
            [0, 0, 1]
        ]), eerot)

        lift_up_motion = self.adplaner.gen_linear_depart(start_tcp_pos=eepos, start_tcp_rotmat=eerot,
                                                         direction=-eerot[:, 2], distance=.08, toggle_dbg=True)

        if lift_up_motion is None:
            print("cannot calc return_to_start_motion")
            return None

        print("-----approach_to_objects_motion Planning-----")
        approach_to_objects_motion = self.adplaner.gen_approach(goal_tcp_pos=eepos_approach,
                                                                goal_tcp_rotmat=picking_rot,
                                                                start_jnt_values=lift_up_motion.jv_list[-1],
                                                                linear_direction=picking_rot[:, 2],
                                                                linear_distance=.08, use_rrt=False, toggle_dbg=True)
        if approach_to_objects_motion is None:
            print("cannot calc return_to_start_motion")
            return None

        close_gripper_motion = MotionData(self.robot)
        close_ev_list = []
        close_jnts_list = []
        # print(f"approach_to_objects_motion.jv_list():{approach_to_objects_motion.jv_list}")
        for i in range(85):
            close_ev_list.append(0.085 - (i * 0.085 / 85))
            close_jnts_list.append(approach_to_objects_motion.jv_list[-1])
        close_gripper_motion.extend(close_jnts_list, close_ev_list)

        print("-----return_to_start_motion Planning-----")
        return_to_start_motion = self.adplaner.gen_depart(start_tcp_pos=eepos_approach,
                                                          start_tcp_rotmat=picking_rot,
                                                          end_jnt_values=self.jnt_values_above_box,
                                                          linear_distance=.03, use_rrt=False, toggle_dbg=True)
        if return_to_start_motion is None:
            print("cannot calc return_to_start_motion")
            return None

        open_gripper_motion = MotionData(self.robot)
        open_ev_list = []
        open_jnts_list = []
        for i in range(85):
            open_ev_list.append(i * 0.085 / 85)
            open_jnts_list.append(return_to_start_motion.jv_list[-1])
        open_gripper_motion.extend(open_jnts_list, open_ev_list)

        return push_motion_data + lift_up_motion + approach_to_objects_motion + close_gripper_motion + return_to_start_motion + open_gripper_motion

    def RotatingPushMotion(self, i, j):
        # objpos_pushed押されて移動した物体の位置　radius_initはpair1とcorの距離
        objpos_pushed, _, cor, radius_init = self.constrainedPushingPos(i, j)

        objrot_pushed = rm.rotmat_from_axangle([0, 0, 1], self.objects_list.angle_list[j])
        # mcm.mgm.gen_frame(pos=objpos_pushed, rotmat=objrot_pushed, ax_radius=0.005).attach_to(base)
        # mcm.mgm.gen_sphere(pos=cor, radius=0.01, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
        objmat4_pushed = rm.homomat_from_posrot(objpos_pushed, objrot_pushed)
        obj_pushed = copy.deepcopy(self.printed_obj)
        obj_pushed.rgba = np.array([0, 1, 0, 0.8])
        obj_pushed.homomat = objmat4_pushed
        obj_pushed.attach_to(self.base)

        if self.constraint1(i, obj_pushed, j, cor):
            print("constraint1 is True")
        else:
            print("constraint1 is False")
            return None

        obj_list, angle_list = self.complementConfig(cor, self.printed_obj, i, obj_pushed,
                                                     j)  # complete config between init and target
        for i, obj in enumerate(obj_list[1:-1]):
            obj.rgba = np.array([1, 1, 0, 0.2 + (0.8 / len(obj_list) * i)])
            obj.attach_to(self.base)

        eepos_list = self.constraint2(obj_list, angle_list)  ###押し動作の開始から終わりまでのpositionリスト

        if eepos_list == None:
            print("constraint2 is False")
            return None
        else:
            print("constraint2 is True")
            # for pos in eepos_list:
            #     print(f"pos:{pos}")
            #     mcm.mgm.gen_sphere(pos=pos, radius=0.004, rgb=np.array([0.0, 0.0, 0.0]), alpha=1).attach_to(self.base)

        """calc jntangle_lift (eepos_list[0] + z) and approahing path"""
        PUSHING_ROTMAT = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

        print("-----approach_to_objects_motion_for_pushing Motion Planning-----")
        eepos = eepos_list[0]
        eerot = np.dot(PUSHING_ROTMAT, rm.rotmat_from_axangle([0, 0, -1], angle_list[0]))
        mcm.mgm.gen_frame(pos=eepos, rotmat=eerot).attach_to(self.base)
        approach_to_objects_motion = self.adplaner.gen_approach(goal_tcp_pos=eepos,
                                                                goal_tcp_rotmat=eerot,
                                                                start_jnt_values=self.jnt_values_above_box,
                                                                linear_direction=eerot[:, 2],
                                                                linear_distance=0.03,
                                                                toggle_dbg=True)
        if approach_to_objects_motion is None:
            print("cannot calc approach_to_objects_motion")
            return None

        print("-----Pushing Motion Planning-----")
        push_motion = MotionData(self.robot)
        """calc pose for each eepos, angle"""
        jntangle_list = []
        for i in range(len(eepos_list)):
            eepos = eepos_list[i]
            eerot = np.dot(PUSHING_ROTMAT, rm.rotmat_from_axangle([0, 0, -1], angle_list[i]))
            mcm.mgm.gen_frame(pos=eepos, rotmat=eerot).attach_to(self.base)
            jntangle = self.robot.ik(tgt_pos=eepos, tgt_rotmat=eerot)
            if jntangle is not None:
                jntangle_list.append(jntangle)

        if jntangle_list is None or len(jntangle_list) != len(eepos_list):
            print("cannot get path to jntangle_list")
            return None

        """calc pushing path"""
        pushing_path_list = []
        temp_path = mt.callLAJNTPath(approach_to_objects_motion.jv_list[-1], jntangle_list[0], self.robot,
                                     discretedist=10)
        pushing_path_list.extend(temp_path)
        for i in range(len(jntangle_list) - 1):
            temp_path = mt.callLAJNTPath(jntangle_list[i], jntangle_list[i + 1], self.robot, discretedist=10)
            pushing_path_list.extend(temp_path)

        push_motion.extend(pushing_path_list)

        return approach_to_objects_motion + push_motion
