import os
import numpy as np
import copy
import math
import random
import csv
from scipy.spatial.transform import Rotation
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from panda3d.core import *
from wrs.modeling import collision_model



def placeObj(base, colmodel, objpos, objrot, r, g, b, a, axislength=40, axisalpha=0.8):
    """

    Args:
        objpos: numpy array 3 by 1
        objrot: numpy array 3 by 3

    Returns:

    """
    objmat4 = rm.homomat_from_posrot(pos=objpos,rotmat=objrot)
    obj_real = copy.deepcopy(colmodel)
    obj_real.rgba = np.array([r, g, b, a])
    obj_real.homomat=objmat4
    mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(base)
    obj_real.attach_to(base)
    return obj_real


def fitCube(base, colmodel, objheight, pos:np.array, objangle, r, g, b, a):
    objpos = pos - np.array([0, 0, objheight])
    objrot = np.dot(rm.rotmat_from_axangle([0, 0, 1], objangle), rm.rotmat_from_axangle([0, 1, 0], -90))
    objmat4 = rm.homomat_from_posrot(pos=objpos,rotmat=objrot)
    obj_real = copy.deepcopy(colmodel)
    obj_real.rgba = np.array([r, g, b, a])
    obj_real.homomat=objmat4
    mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(base)
    obj_real.attach_to(base)
    return obj_real, objangle


def placeRandom(base, colmodel, objheight, num, x_range: tuple, y_range: tuple, z, simrobot, liftheight=30):
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
        centroid = np.array([random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1]), z])
        angle = random.uniform(0.00, math.pi/2)  #angle:rad objangle:deg
        rotvec = np.array([0, 0, -angle])
        rot = Rotation.from_rotvec(rotvec)
        rotmat = rot.as_matrix()
        orientation_lft_moveto = np.dot(picking_pose, rotmat)
        liftpos = centroid + np.array([0, 0, liftheight])
        jntangle_lft_liftpoint = simrobot.ik(pos=liftpos, rotmat=orientation_lft_moveto)

        if jntangle_lft_liftpoint is not None:
            objangle = math.degrees(angle)  #angle:rad objangle:deg
            objpos = centroid - np.array([0, 0, objheight])
            objrot = np.dot(rm.rotmat_from_axangle([0, 0, 1], objangle), rm.rotmat_from_axangle([0, 1, 0], -90))
            objmat4 = rm.homomat_from_posrot(pos=objpos,rotmat=objrot)
            obj = copy.deepcopy(colmodel)
            obj.rgba = np.array([1, 0, 0, 1])
            obj.homomat=objmat4

            if not obj.is_mcdwith(obj_list):
                centroids.append(centroid)
                angle_list.append(angle)
                mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(base)
                obj.attach_to(base)
                obj_list.append(obj)
                objangle_list.append(objangle)

    return centroids, angle_list, obj_list, objangle_list


def placeRandomRoundly(base, box, colmodel, objheight, num, center, r_range: tuple, theta_range: tuple, z, simrobot, liftheight=0.03):
    """

    Args:
        center: numpy array 3 by 1
    """

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
        r = random.uniform(r_range[0], r_range[1])
        theta = random.uniform(theta_range[0], theta_range[1])
        centroid = np.array([center[0], center[1], z]) + np.array([r * math.cos(theta), r * math.sin(theta), 0])
        angle = random.uniform(0.00, math.pi)  #angle:rad objangle:deg
        rotvec = np.array([0, 0, -angle])
        rot = Rotation.from_rotvec(rotvec)
        rotmat = rot.as_matrix()
        orientation_lft_moveto = np.dot(picking_pose, rotmat)
        liftpos = centroid + np.array([0, 0, liftheight])
        jntangle_rgt_liftpoint = simrobot.ik(tgt_pos=liftpos, tgt_rotmat=orientation_lft_moveto)

        if jntangle_rgt_liftpoint is not None:
            objangle = angle  #angle:rad objangle:deg
            objpos = centroid + np.array([0, 0, objheight])
            # objrot = np.dot(rm.rotmat_from_axangle([0, 0, 1], objangle), rm.rotmat_from_axangle([0, 1, 0], -90))
            objrot = rm.rotmat_from_axangle([0, 0, 1], objangle)
            objmat4=rm.homomat_from_posrot(pos=objpos,rotmat=objrot)

            obj = copy.deepcopy(colmodel)
            obj.rgba=np.array([1, 0, 0, 1])
            obj.homomat=objmat4

            if not obj.is_mcdwith(obj_list) and not box.is_mcdwith(obj_list):
                centroids.append(centroid)
                angle_list.append(angle)

                mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(base)
                obj.attach_to(base)

                obj_list.append(obj)
                objangle_list.append(objangle)

    return centroids, angle_list, obj_list, objangle_list


def writePlacePattern(filepath, centroids, angle_list):
    with open(filepath, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for i in range(len(centroids)):
            writer.writerow([centroids[i][0], centroids[i][1], centroids[i][2],  angle_list[i]])


def readPlacePattern(base, filepath, colmodel, objheight):
    centroids = []
    angle_list = []  # rad
    obj_list = []
    objangle_list = []  # degree
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        for line in reader:
            centroids.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
            angle_list.append(float(line[3]))

    for i in range(len(centroids)):
        objangle_list.append(math.degrees(angle_list[i]))
        objpos = centroids[i] - np.array([0, 0, objheight])
        objrot = np.dot(rm.rotmat_from_axangle([0, 0, 1], math.degrees(angle_list[i])), rm.rotmat_from_axangle([0, 1, 0], -90))
        objmat4 = rm.homomat_from_posrot(objpos, objrot)
        obj = copy.deepcopy(colmodel)
        obj.setColor(1, 1, 0, 0.5)
        obj.homomat=objmat4
        mcm.mgm.gen_frame(pos=objpos, rotmat=objrot).attach_to(base)
        obj.attach_to(base)
        obj_list.append(obj)

    return centroids, angle_list, obj_list, objangle_list