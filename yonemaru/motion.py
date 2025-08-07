import numpy as np
import math
from wrs.motion.probabilistic.rrt_connect import RRTConnect
from wrs.motion.primitives.interpolated import InterplatedMotion


def callRRTConnect(startjnts, goaljnts, robot, obscmlist):
    # print("start:", startjnts)
    # print("goal", goaljnts)

    rrtc = RRTConnect(robot)
    print(f"startjnts:{startjnts},goaljnts:{goaljnts},obscmlist:{obscmlist}")
    path = rrtc.plan(start_conf=startjnts,
                     goal_conf=goaljnts,
                     obstacle_list=obscmlist,
                     ext_dist=.1,
                     max_time=30,
                     smoothing_n_iter=100)
    if path is None:
        print("cannot calc path")
        return None
    return path


def callLAJNTPath(startjnts, goaljnts, robot, discretedist=10):
    startpos, _ =robot.fk(jnt_values=startjnts)
    goalpos, _ = robot.fk(jnt_values=goaljnts)
    vecsg = goalpos - startpos
    dist = np.linalg.norm(vecsg)
    if (dist < 1e-6):
        return [startjnts, goaljnts]
    njnts = int(math.ceil(dist / discretedist))
    path = [startjnts]
    for i in range(1, njnts):
        pose = ((njnts - i) / njnts) * startjnts + (i / njnts) * goaljnts
        path.append(pose)
    path.append(goaljnts)
    return path


# def numik_Nearest(robot, pos, rotmat, msc, armname="lft", tryinversion=False):
#     """
#     calc numikmsc which changes little base and shoulder angle
#
#     :param robot:simrobot
#     :param pos: nparray 3
#     :param rotmat: ndarray mat 3 by 3
#     :param msc: nparray 6 (jointangle)
#     """
#
#     """try rotA and rotB(Rotated 180 degrees around the z-axis)"""
#     rotA = rotmat
#     warning_sentence = termcolor.colored('Warning: Could NOT calc numikmsc', 'red')
#
#     jntangle = robot.numikmsc(pos, rotA, msc, armname=armname)
#     if tryinversion == True:
#         rotB = np.dot(rotA, rm.rodrigues([0, 0, 1], 180))
#         if jntangle is None:
#             jntangle = robot.numikmsc(pos, rotB, msc, armname=armname)
#             if jntangle is None:
#                 print(warning_sentence)
#                 jntangle = robot.numik(pos, rotA, armname=armname)
#                 if jntangle is None:
#                     jntangle = robot.numik(pos, rotB, armname=armname)
#                     if jntangle is None:
#                         print("There is no ik")
#                         return None
#     else:
#         if jntangle is None:
#             print(warning_sentence)
#             jntangle = robot.numik(pos, rotA, armname=armname)
#             if jntangle is None:
#                 print("There is no ik")
#                 return None
#
#
#     jntsrng_list = robot.getarmjntsrng(armname=armname)
#     for i in range(len(jntsrng_list)):
#         if tryinversion == True and i == 5:
#             cand_list = [jntangle[i] - 360, jntangle[i] - 180, jntangle[i], jntangle[i] + 180, jntangle[i] + 360]
#         else:
#             cand_list = [jntangle[i] - 360, jntangle[i], jntangle[i] + 360]
#         cand_inrange_list = []
#         for cand in cand_list:
#             if jntsrng_list[i][0] < cand and cand < jntsrng_list[i][1]:
#                 cand_inrange_list.append(cand)
#
#         "choose the nearest from  msc[i]"
#         index = np.abs(np.asarray(cand_inrange_list) - msc[i]).argmin()
#         jntangle[i] = cand_inrange_list[index]
#
#
#     return jntangle
#
def nearestJoint(jntangle, msc):
    for i in range(len(jntangle)):
        cand_list = [jntangle[i] - 2*np.pi, jntangle[i], jntangle[i] + 2*np.pi]
        cand_inrange_list = []
        for cand in cand_list:
            if jntsrng_list[i].motion_range[0] < cand and cand < jntsrng_list[i].motion_range[1]:
                cand_inrange_list.append(cand)

        "choose the nearest from  msc[i]"
        if len(cand_inrange_list) == 0:
            raise ValueError(f"cand_inrange_list is empty at i={i}, msc[i]={msc[i]}")
        index = np.abs(np.array(cand_inrange_list) - msc[i]).argmin()
        print(f"cand_inrange_list:{cand_inrange_list},msc[i]:{msc[i]},np.abs(np.asarray(cand_inrange_list) - msc[i]){np.abs(np.asarray(cand_inrange_list) - msc[i])},jntangle:{jntangle},cand_inrange_list:{cand_inrange_list},index:{index}")
        jntangle[i] = cand_inrange_list[index]

    return jntangle


#
def getLinearMotion(startpos, goalpos, rotmat, msc, robot):
    im = InterplatedMotion(robot)
    path_tmp = im.gen_linear_motion(start_tcp_pos=startpos, start_tcp_rotmat=rotmat, goal_tcp_pos=goalpos,
                                    goal_tcp_rotmat=rotmat)

    # path = []
    # seed_joints=None
    # """convet list of ndarray to list of list"""
    # for i in range(len(path_tmp)):
    #     print(f"path_tmp[i]:{path_tmp[i]}")
    #     if i == 0:
    #         joints = nearestJoint(robot, path_tmp[i], msc, armname="lft")
    #         seed_joints=joints
    #     else:
    #         joints = nearestJoint(robot, path_tmp[i], seed_joints, armname="lft")
    #         seed_joints = joints
    #
    #     path.append(joints)
    # return path
    return path_tmp
#
# def getJointNormDiff(jnt_a, jnt_b):
#     jnt_a = np.array(jnt_a)
#     jnt_b = np.array(jnt_b)
#     return np.linalg.norm(jnt_b-jnt_a)
