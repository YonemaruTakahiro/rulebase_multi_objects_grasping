import numpy as np
from scipy.spatial.transform import Rotation
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
from wrs.modeling import collision_model as cm
import numpy as np
import copy
import motion as mt
from wrs.motion.primitives.interpolated import InterplatedMotion
from wrs.motion.motion_data import MotionData


def constrainedPushingPos(object_pair1_pos, object_pair2_pos, object_pair1_angle, object_pair2_angle,
                          ALIGN_DISTANCE=0.2,
                          PUSHER_THICKNESS=10):
    """
    get pushed object's pos in constrained pushing
    return pushed_pos(pos of A'), nexttarget_pos(pos of A''), cor(Center of Rotation), radius(radius from cor to object_pair1_pos)
    """

    rad_i = object_pair1_angle  # object_pair1_angle in radian
    rad_t = object_pair2_angle  # object_pair2_angle in radian

    """nexttarget_pos means A'', pushed_pos means A'"""  # 向きが大事
    cand1_pos = object_pair2_pos + ALIGN_DISTANCE * np.array([np.cos(rad_t), np.sin(rad_t), 0])
    cand2_pos = object_pair2_pos + ALIGN_DISTANCE * np.array([np.cos(-rad_t), np.sin(-rad_t), 0])
    if np.linalg.norm(cand1_pos - object_pair1_pos) < np.linalg.norm(cand2_pos - object_pair1_pos):
        nexttarget_pos = cand1_pos
    else:
        nexttarget_pos = cand2_pos

    """get COR(Center of Rotation)"""
    A = np.array([
        [np.tan(rad_i + np.pi / 2), -1.0],
        [np.tan(rad_t + np.pi / 2), -1.0]
    ])
    Y = np.array([
        [np.tan(rad_i + np.pi / 2) * object_pair1_pos[0] - object_pair1_pos[1]],
        [np.tan(rad_t + np.pi / 2) * nexttarget_pos[0] - nexttarget_pos[1]]
    ])
    cor_xy = np.linalg.solve(A, Y)
    cor = np.array([cor_xy[0][0], cor_xy[1][0], object_pair1_pos[2]])
    # base.pggen.plotSphere(base.render, pos=cor, radius=20, rgba=np.array([0.0, 255.0, 0.0, 0.5]), plotname="sphere")
    radius = np.linalg.norm(np.array([object_pair1_pos[0], object_pair1_pos[1]]) - np.array([cor[0], cor[1]]))
    radius_vec = np.array([nexttarget_pos[0], nexttarget_pos[1]]) - np.array([cor[0], cor[1]])  # 回転中心から次の目標位置までの距離
    radius_unitvec = radius_vec / np.linalg.norm(radius_vec)
    pushed_pos_xy = np.array([cor[0], cor[1]]) + radius * radius_unitvec
    pushed_pos = np.array([pushed_pos_xy[0], pushed_pos_xy[1], object_pair1_pos[2]])  # 押されて移動した物体の位置
    return pushed_pos, nexttarget_pos, cor, radius


def complementConfig(cor, target_obj_pair_1, object_pair1, obj_pushed, object_pair1_angle, object_pair2_angle,
                     divnum=4):
    initial_pos = np.array([object_pair1._pos[0], object_pair1._pos[1], object_pair1._pos[2]])
    obj_list = [object_pair1]
    angle_list = [object_pair1_angle]
    divangle = (object_pair2_angle - object_pair1_angle) / divnum  # 押す対象物体の押す前と押した後の角度(姿勢)の分割数
    for n in range(1, divnum):  # 押し始めから押し終わるまでの中点を計算
        R = np.array([
            [np.cos(divangle * n), -np.sin(divangle * n), 0],
            [np.sin(divangle * n), np.cos(divangle * n), 0],
            [0, 0, 1]
        ])
        pos = np.dot(R, (initial_pos - cor)) + cor
        print(f"interp_pos:{pos}")
        angle = ((divnum - n) * object_pair1_angle + n * object_pair2_angle) / divnum  # 角度の次元における中点

        interp_objpos = pos
        interp_objrot = rm.rotmat_from_axangle([0, 0, 1], angle)
        interp_objmat4 = rm.homomat_from_posrot(interp_objpos, interp_objrot)
        interp_obj = copy.deepcopy(target_obj_pair_1)
        interp_obj.rgba = np.array([1, 1, 0, 0.8])
        interp_obj.homomat = interp_objmat4

        # display in RotatingPushingMotion
        # base.pggen.plotAxis(base.render, objpos, objrot, length=40, alpha=0.5)
        # obj.reparentTo(base.render)

        obj_list.append(interp_obj)
        angle_list.append(angle)
    obj_list.append(obj_pushed)
    angle_list.append(object_pair2_angle)
    return obj_list, angle_list


def constraint1(object_pair1, obj_pushed, object_pair2, object_pair1_angle, object_pair2_angle, cor, r_th, d_th,
                theta_th):
    """constraint1 : the configuration of object_pair1, obj_pushed, object_pair2 is close"""
    radius_initial = np.linalg.norm(
        np.array([object_pair1._pos[0], object_pair1._pos[1]]) - np.array([cor[0], cor[1]]))
    radius_target = np.linalg.norm(
        np.array([object_pair2._pos[0], object_pair2._pos[1]]) - np.array([cor[0], cor[1]]))
    r = abs(radius_target - radius_initial)  # same as |rB-rA|
    d = np.linalg.norm(
        [obj_pushed._pos[1] - object_pair1._pos[1], obj_pushed._pos[0] - object_pair1._pos[0]])  # same as ||xA'-xA||
    theta = abs(object_pair2_angle - object_pair1_angle)  # same as |thetaA'-thetaA|

    print("r, r_th", r, r_th)
    print("d, d_th", d, d_th)
    print("theta, theta_th", theta, theta_th)
    if r < r_th and d < d_th and theta < theta_th:
        return True
    else:
        return False


def constraint2(base, compobj_list, angle_list, alpha, PUSHER_WIDTH=0.085, PUSHER_THICKNESS=0.02, PUSHING_Z=1.1):
    """constraint2 : constraint of e_v(unit vector of Pusher's velocity), COR(Center of Rotation)"""
    """2-1, 2-2 True -> return eepos_list  False-> return None"""
    eepos_list = []

    if (len(compobj_list) != len(angle_list)):
        raise ValueError("the lens of compobj_list and angle_list are different")
    if (len(compobj_list) < 2):
        raise ValueError("the lens of compobj_list must be least 2")

    """calc push pos for each motion"""
    for i in range(len(compobj_list)):
        pos = np.array(compobj_list[i]._pos)
        rotmat = rm.rotmat_from_axangle([0, 0, 1], angle_list[i])
        # xF = pos + np.dot(rotmat, np.array([0, 0.01, 0.01]))
        # mgm.gen_sphere(pos=xF, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
        if i == 0:
            eepos = pos + np.dot(rotmat, np.array([0, 0, PUSHER_THICKNESS * 2]))
            eepos[2] = PUSHING_Z  # 押すときの高さ
        else:
            eepos = pos + np.dot(rotmat, np.array([0, 0, PUSHER_THICKNESS]))
            eepos[2] = PUSHING_Z  # 押すときの高さ
        # mgm.gen_sphere(pos=xF, radius=0.2, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8)
        if i == len(compobj_list) - 1:
            print(i, "/", len(compobj_list) - 1, "the last index")
            eepos[2] += 0.01
            eepos_list.append(eepos)
        else:
            """const2-1 there are no slip between pusher and object"""
            # todo change this value from .stl file
            xF1 = pos + np.dot(rotmat, np.array([0.01, -PUSHER_WIDTH / 2, 0.01]))
            xF2 = pos + np.dot(rotmat, np.array([0.01, PUSHER_WIDTH / 2, 0.01]))
            # mgm.gen_sphere(pos=xF1, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
            # mgm.gen_sphere(pos=xF2, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)

            eF = np.dot(
                [[np.cos(np.pi / 2), np.sin(np.pi / 2), 0], [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]],
                np.array(xF2 - xF1))
            ev = compobj_list[i + 1]._pos - compobj_list[i]._pos
            # mgm.gen_arrow(spos=pos, epos=eF+pos,stick_radius=0.004, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
            # mgm.gen_arrow(spos=pos, epos=10*ev+pos,stick_radius=0.002, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
            eF = eF[0:2]
            eF = eF / np.linalg.norm(eF)
            ev = ev[0:2]
            ev = ev / np.linalg.norm(ev)
            if abs(np.dot(ev, eF)) >= np.linalg.norm(ev) / np.sqrt(1 + np.tan(alpha) * np.tan(alpha)):
                print(i, "/", len(compobj_list) - 1, "2-1 is True")
            else:
                print(i, "/", len(compobj_list) - 1, "2-1 is False")
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
                xF = lambda_F * xF1 + (1 - lambda_F) * xF2
                # mgm.gen_sphere(pos=xF, radius=0.002, rgb=np.array([0.0, 1.0, 0.0]), alpha=0.8).attach_to(base)
                if 0 >= lambda_F or lambda_F >= 1:
                    print(i, "/", len(compobj_list) - 1, "2-2 is False")
                    return None
                else:
                    print(i, "/", len(compobj_list) - 1, "2-2 is True")
            eepos[2] += 0.01
            eepos_list.append(eepos)

    return eepos_list


"""get pushing motions e_v(unit vector of Pusher's velocity) and COR constrained"""


def RotatingPushMotion(base, robot, obscmlist, start_joints, obj_3dprinted, object_pair1, object_pair2,
                       object_pair1_angle,
                       object_pair2_angle, ALIGN_DISTANCE=0.2, PUSHER_THICKNESS=0.2):
    # objpos_pushed押されて移動した物体の位置　radius_initはpair1とcorの距離
    objpos_pushed, _, cor, radius_init = constrainedPushingPos(object_pair1._pos, object_pair2._pos, object_pair1_angle,
                                                               object_pair2_angle, ALIGN_DISTANCE, PUSHER_THICKNESS)

    objrot_pushed = rm.rotmat_from_axangle([0, 0, 1], object_pair2_angle)
    # mcm.mgm.gen_frame(pos=objpos_pushed, rotmat=objrot_pushed, ax_radius=0.005).attach_to(base)
    # mcm.mgm.gen_sphere(pos=cor, radius=0.01, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)
    objmat4_pushed = rm.homomat_from_posrot(objpos_pushed, objrot_pushed)
    obj_pushed = copy.deepcopy(obj_3dprinted)
    obj_pushed.rgba = np.array([1, 1, 0, 0.6])
    obj_pushed.homomat = objmat4_pushed
    obj_pushed.attach_to(base)

    """r_th, d_th, theta_th(parameters threthoulds of constraint1) and alpha(friction angle) and num(num of complementing motions in pushing)"""
    r_th = 0.08
    d_th = 0.4
    theta_th = np.pi / 2
    alpha = np.radians(60)  # friction angle
    divnum = 4
    pushing_Z = 1.1
    if constraint1(object_pair1, obj_pushed, object_pair2, object_pair1_angle, object_pair2_angle, cor, r_th, d_th,
                   theta_th):
        print("constraint1 is True")
    else:
        print("constraint1 is False")
        return None

    obj_list, angle_list = complementConfig(cor, obj_3dprinted, object_pair1, obj_pushed, object_pair1_angle,
                                            object_pair2_angle,
                                            divnum=divnum)  # complete config between init and target
    #
    # for obj in obj_list:
    #     obj.attach_to(base)
    #     mcm.mgm.gen_sphere(pos=obj._pos, radius=0.01, rgb=np.array([0.0, 1.0, 1.0]), alpha=0.8).attach_to(base)

    eepos_list = constraint2(base, obj_list, angle_list, alpha, PUSHER_WIDTH=0.085, PUSHER_THICKNESS=0.01,
                             PUSHING_Z=pushing_Z)  ###押し動作の開始から終わりまでのpositionリスト

    if eepos_list == None:
        print("constraint2 is False")
        return None
    else:
        print("constraint2 is True")
        for pos in eepos_list:
            print(f"pos:{pos}")
            mcm.mgm.gen_sphere(pos=pos, radius=0.004, rgb=np.array([0.0, 0.0, 0.0]), alpha=1).attach_to(base)

    """calc jntangle_lift (eepos_list[0] + z) and approahing path"""
    PICKING_ROTMAT = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    path_list = []  # path_list:zの値ごとのスタート位置から押し動作開始位置までのjointsリスト
    obscmlist = obscmlist + obj_list
    z_list = [0.01, 0.015, 0.02, 0.005, 0.008]
    for z in z_list:
        eepos = eepos_list[0]  # 押し動作開始位置
        # mgm.gen_sphere(pos=eepos, radius=0.1, rgb=np.array([0.0, 0.0, 1.0]), alpha=0.8).attach_to(base)
        eerot = np.dot(PICKING_ROTMAT, rm.rotmat_from_axangle([0, 0, -1], angle_list[0]))
        jntangle_lift = robot.ik(tgt_pos=eepos + np.array([0, 0, z]), tgt_rotmat=eerot)

        if jntangle_lift is not None:
            path = mt.callRRTConnect(start_joints, jntangle_lift, robot, obscmlist)
            if path is not None:
                print("jntangle_lift", jntangle_lift)
                print("select jntangle_lift z as ", z)
                path_list.extend(path._jv_list)
                break

    if path_list is None:
        print("cannot get path to jntangle_lift")
        return None

    """calc pose for each eepos, angle"""
    jntangle_list = []
    for i in range(len(eepos_list)):
        eepos = eepos_list[i]
        eepos[2] += 0.03
        eerot = np.dot(PICKING_ROTMAT, rm.rotmat_from_axangle([0, 0, -1], angle_list[i]))
        mcm.mgm.gen_frame(pos=eepos, rotmat=eerot).attach_to(base)
        jntangle = robot.ik(tgt_pos=eepos, tgt_rotmat=eerot)
        jntangle_list.append(jntangle)

        # if i == 0:
        #     jntangle = robot.ik(tgt_pos=eepos, tgt_rotmat=eerot)
        #     jntangle_list.append(jntangle)
        #     print("jntangle:", jntangle)
        # else:
        #     jntangle = robot.ik(tgt_pos=eepos, tgt_rotmat=eerot)
        #     jntangle_list.append(jntangle)
        #     print("jntangle:", jntangle)
    if jntangle_list is None:
        print("cannot get path to jntangle_list")
        return None

    """calc pushing path"""
    path = mt.callLAJNTPath(jntangle_lift, jntangle_list[0], robot, discretedist=10)
    path_list.extend(path)
    for i in range(len(jntangle_list) - 1):
        path = mt.callLAJNTPath(jntangle_list[i], jntangle_list[i + 1], robot, discretedist=10)
        path_list.extend(path)

    # display COR, obj_pushed, complementConfig
    mcm.mgm.gen_frame(pos=objpos_pushed, rotmat=objrot_pushed).attach_to(base)
    obj_pushed.attach_to(base)
    mgm.gen_sphere(pos=cor, radius=0.003, rgb=np.array([0.0, 0.0, 1.0]), alpha=0.8)
    for obj in obj_list:
        mcm.mgm.gen_frame(pos=np.array(obj._pos), rotmat=np.array(obj._rotmat)).attach_to(base)
        obj.attach_to(base)

    push_motion_data = MotionData(robot)
    print(f"aaaaaaaaaaaaaaaaaaapath_list:{path_list}")
    push_motion_data.extend(path_list)

    return push_motion_data


def PickandRecovMotion(base, simrobot, obscmlist, start_joints, grasp):
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
    orientation_rgt_moveto = np.dot(picking_pose, rotmat)

    jntangle_rgt_moveto = simrobot.ik(tgt_pos=pos_rgt_moveto, tgt_rotmat=orientation_rgt_moveto)
    if jntangle_rgt_moveto is None:
        print("There is no ik")
        return None
    # simrobot.goto_given_conf(jntangle_rgt_moveto)

    print("IK joint angle of the right arm", jntangle_rgt_moveto)

    print("-----IK lift point-----")
    z_list = [0.03, 0.04, 0.05, 0.01, 0.02]
    for z in z_list:
        pos_rgt_liftpoint = grasp[0] + np.array([0, 0, z])
        jntangle_rgt_liftpoint = simrobot.ik(tgt_pos=pos_rgt_liftpoint, tgt_rotmat=orientation_rgt_moveto)
        if jntangle_rgt_liftpoint is not None:
            simrobot.goto_given_conf(jntangle_rgt_liftpoint)
            print("IK joint angle of the right arm", jntangle_rgt_liftpoint)
            break
    if jntangle_rgt_liftpoint is None:
        print("There is no ik")
        return None

    print("-----Motion Planning-----")
    print("---path_a---")

    motion_data_a = mt.callRRTConnect(startjnts=start_joints, goaljnts=jntangle_rgt_liftpoint, robot=simrobot,
                                      obscmlist=obscmlist)

    print(f"motion_data_a:{motion_data_a}")
    if motion_data_a is None:
        print("cannot calc path_a")
        return None

    pick_joints_value = motion_data_a._jv_list[-1]
    simrobot.change_jaw_width(jaw_width=0.0)
    motion_data_a.extend([pick_joints_value])

    print("---path_b---")
    motion_data_b = mt.getLinearMotion(startpos=pos_rgt_liftpoint, goalpos=pos_rgt_moveto,
                                       rotmat=orientation_rgt_moveto, msc=motion_data_a._jv_list[-1], robot=simrobot)
    if len(motion_data_b._jv_list) == 0:
        print("path_b is []")
        return None

    print("---path_c---")
    motion_data_c = MotionData(simrobot)
    motion_data_c.extend(motion_data_b._jv_list[::-1])

    print("---path_d---")
    motion_data_d = MotionData(simrobot)
    motion_data_d.extend(motion_data_a._jv_list[::-1])
    #
    # if path_d is None:
    #     print("cannot calc path_d")
    #     return None

    pick_joints_value = motion_data_d._jv_list[-1]
    simrobot.change_jaw_width(jaw_width=0.08)
    motion_data_a.extend([pick_joints_value])

    # return [path_a, path_b, path_c, path_d]
    return motion_data_a + motion_data_b + motion_data_c + motion_data_d


def PushandRecovMotion(base, robot, obscmlist, start_joints, obj_3dprinted, object_pair1, object_pair2,
                       object_pair1_angle,
                       object_pair2_angle, ALIGN_DISTANCE=0.02, PUSHER_THICKNESS=0.02):
    """path list of pushing"""
    push_motion_data = RotatingPushMotion(base, robot, obscmlist, start_joints, obj_3dprinted,
                                          object_pair1, object_pair2, object_pair1_angle, object_pair2_angle,
                                          ALIGN_DISTANCE=ALIGN_DISTANCE, PUSHER_THICKNESS=PUSHER_THICKNESS)

    if push_motion_data is None:
        print("push_motion_data is None")
        return None
    jntangle_after_push = push_motion_data._jv_list[-1]  # pushingの最後の関節角
    eepos, eerot = robot.fk(jnt_values=jntangle_after_push)

    im = InterplatedMotion(robot)

    """lift up"""
    z_list = [0.01, 0.015, 0.02, 0.005, 0.008, 0]
    for z in z_list:
        # liftpath = mt.getLinearMotion(eepos, eepos + np.array([0, 0, z]), eerot, jntangle_after_push, robot)
        lift_motion_data = im.gen_linear_motion(start_tcp_pos=eepos, start_tcp_rotmat=eerot,
                                                goal_tcp_pos=eepos + np.array([0, 0, z]),
                                                goal_tcp_rotmat=eerot)
        if len(lift_motion_data._jv_list) > 0:
            jntangle_after_lift = lift_motion_data._jv_list[-1]
            eepos_after_lift, _ = robot.fk(jnt_values=jntangle_after_lift)
            break
    if len(lift_motion_data._jv_list) == 0:
        print("lift_motion_data is []")
        return None

    """approach and down"""
    eepos_approach = np.array(
        [(eepos_after_lift[0] + object_pair2._pos[0]) / 2, (eepos_after_lift[1] + object_pair2._pos[1]) / 2,
         eepos_after_lift[2]])

    parallel_motion_data = im.gen_linear_motion(start_tcp_pos=eepos_after_lift, start_tcp_rotmat=eerot,
                                                goal_tcp_pos=eepos_approach,
                                                goal_tcp_rotmat=eerot)
    # parallel_motion_data = mt.getLinearMotion(eepos_after_lift, eepos_approach, eerot, jntangle_after_lift, robot)
    if parallel_motion_data._jv_list is None:
        print("parallel_motion_data is []")
        return None
    jntangle_after_parallelmove = parallel_motion_data._jv_list[-1]
    eepos_down = np.array([eepos_approach[0], eepos_approach[1], eepos[2]])
    # downpath = mt.getLinearMotion(eepos_approach, eepos_down, eerot, jntangle_after_parallelmove, robot)
    down_motion_data = im.gen_linear_motion(start_tcp_pos=eepos_approach, start_tcp_rotmat=eerot,
                                            goal_tcp_pos=eepos_down,
                                            goal_tcp_rotmat=eerot)
    if down_motion_data is None:
        print("down_motion_data is []")
        return None

    """up and recov(to above the box)"""
    up_motion_data = MotionData(robot)
    up_motion_data.extend(down_motion_data._jv_list[::-1])
    jntangle = up_motion_data._jv_list[-1]
    recov_motion_data = mt.callRRTConnect(jntangle, start_joints, robot, obscmlist=None)  # 掴んでから箱にもどるまでの経路
    if recov_motion_data is None:
        print("recov_motion_data is None")
        return None

    return push_motion_data + lift_motion_data + parallel_motion_data + down_motion_data + up_motion_data + recov_motion_data
