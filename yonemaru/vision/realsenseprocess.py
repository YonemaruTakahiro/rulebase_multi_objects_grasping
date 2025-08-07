import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from enum import IntEnum

##########pointclouds処理######################
def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


def preprocess_point_cloud(points, use_cuda=True):
    num_points = 1024

    extrinsics_matrix1 = np.array([[1, 0, 0, 0],
                                  [0, np.cos(np.deg2rad(139)), -np.sin(np.deg2rad(139)), 0],
                                  [0, np.sin(np.deg2rad(139)), np.cos(np.deg2rad(139)), 0],
                                  [0., 0., 0., 1.]])

    extrinsics_matrix2 = np.array([[np.cos(np.deg2rad(90)), -np.sin(np.deg2rad(90)), 0, 0],
                                   [np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90)), 0, 0],
                                   [0, 0, 1, 0],
                                   [0., 0., 0., 1.]])
    WORK_SPACE = [
        [0, 0.6],
        [-0.25, 0.25],
        [-0.7, 0]
    ]

    # scale
    # point_xyz = points[..., :3] * 0.0002500000118743628
    point_xyz = points[..., :3]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix1)
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix2)



    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    # crop
    # points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
    #                          (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
    #                          (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1])&
                         (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]))]

    temp=points

    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))

    return temp,points


def preproces_image(image):
    img_size = 84

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1)  # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0)  # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = []
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print('Sensor: {}, {}'.format(name, serial))
        print('Supported video formats:')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        video_type, w, h, fps, fmt))
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class RealsesneProcessor:
    def __init__(
        self,
        first_D435_serial,
        second_D435_serial,
        total_frame,
        store_frame=False,
        out_directory=None,
        save_hand=False,
        enable_visualization=True,
    ):
        self.first_D435_serial = first_D435_serial
        self.second_D435_serial = second_D435_serial
        self.store_frame = store_frame
        self.out_directory = out_directory
        self.total_frame = total_frame
        self.save_hand = save_hand
        self.enable_visualization = enable_visualization
        self.rds = None

        self.color_buffer = []
        self.depth_buffer = []

        self.pose_buffer = []
        self.pose2_buffer = []
        self.pose3_buffer = []

        self.pose2_image_buffer = []
        self.pose3_image_buffer = []

        # self.rightHandJoint_buffer = []
        # self.leftHandJoint_buffer = []
        # self.rightHandJointOri_buffer = []
        # self.leftHandJointOri_buffer = []
    
    def get_rs_D435_config(self, D435_serial, D435_pipeline):
        D435_config = rs.config()
        D435_config.enable_device(D435_serial)
        D435_config.enable_stream(rs.stream.pose)

        return D435_config
        
    def configure_stream(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        color_profiles, depth_profiles = get_profiles()
        w, h, fps, fmt = depth_profiles[1]
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        w, h, fps, fmt = color_profiles[18]
        config.enable_stream(rs.stream.color, w, h, fmt, fps)

        # Configure the D435 1 stream
        ctx = rs.context()#デバイスの検出
        self.D435_pipeline = rs.pipeline(ctx)
        D435_config = rs.config()
        D435_config.enable_device(self.first_D435_serial)

        # Configure the D435 2 stream
        ctx_2 = rs.context()#デバイスの検出
        self.D435_pipeline_2 = rs.pipeline(ctx_2)
        D435_config_2 = self.get_rs_D435_config(
            self.second_D435_serial, self.D435_pipeline_2
        )

        self.D435_pipeline.start(D435_config)
        self.D435_pipeline_2.start(D435_config_2)

        pipeline_profile = self.pipeline.start(config)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.vis = None
        if self.enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.get_view_control().change_field_of_view(step=1.0)

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = (
            np.asanyarray(aligned_depth_frame.get_data()) // 4
        )  # L515 camera need to divide by 4 to get metric in meter
        color_image = np.asanyarray(color_frame.get_data())

        rgbd = None
        if enable_visualization:
            depth_image_o3d = o3d.geometry.Image(depth_image)
            color_image_o3d = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d,
                depth_image_o3d,
                depth_trunc=4.0,
                convert_rgb_to_intensity=False,
            )
        return rgbd, depth_image, color_image

    def process_frame(self):
        frame_count = 0
        first_frame = True

        try:
            while frame_count < self.total_frame:
                t265_frames = self.t265_pipeline.wait_for_frames()
                t265_frames_2 = self.t265_pipeline_2.wait_for_frames()
                t265_frames_3 = self.t265_pipeline_3.wait_for_frames()
                rgbd, depth_frame, color_frame = self.get_rgbd_frame_from_realsense()

                # get pose data for t265 1
                pose_4x4 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames
                )
                pose_4x4_2 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames_2
                )
                pose_4x4_3 = RealsesneProcessor.frame_to_pose_conversion(
                    input_t265_frames=t265_frames_3
                )

                if self.save_hand:
                    # get hand joint data
                    leftHandJointXyz = np.frombuffer(
                        self.rds.get("rawLeftHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)
                    rightHandJointXyz = np.frombuffer(
                        self.rds.get("rawRightHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)
                    leftHandJointOrientation = np.frombuffer(
                        self.rds.get("rawLeftHandJointOrientation"), dtype=np.float64
                    ).reshape(21, 4)
                    rightHandJointOrientation = np.frombuffer(
                        self.rds.get("rawRightHandJointOrientation"), dtype=np.float64
                    ).reshape(21, 4)

                corrected_pose = pose_4x4 @ between_cam

                # Convert to Open3D format L515
                o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    1280,
                    720,
                    898.2010498046875,
                    897.86669921875,
                    657.4981079101562,
                    364.30950927734375,
                )

                if first_frame:
                    if self.enable_visualization:
                        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, o3d_depth_intrinsic
                        )
                        pcd.transform(corrected_pose)

                        rgbd_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.3
                        )
                        rgbd_mesh.transform(corrected_pose)
                        rgbd_previous_pose = copy.deepcopy(corrected_pose)

                        chest_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.3
                        )
                        chest_mesh.transform(pose_4x4)
                        chest_previous_pose = copy.deepcopy(pose_4x4)

                        left_hand_mesh = (
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                        )
                        left_hand_mesh.transform(pose_4x4_2)
                        left_hand_previous_pose = copy.deepcopy(pose_4x4_2)

                        right_hand_mesh = (
                            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                        )
                        right_hand_mesh.transform(pose_4x4_3)
                        right_hand_previous_pose = copy.deepcopy(pose_4x4_3)

                        self.vis.add_geometry(pcd)
                        self.vis.add_geometry(rgbd_mesh)
                        self.vis.add_geometry(chest_mesh)
                        self.vis.add_geometry(left_hand_mesh)
                        self.vis.add_geometry(right_hand_mesh)

                        view_params = (
                            self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                        )
                    first_frame = False
                else:
                    if self.enable_visualization:
                        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, o3d_depth_intrinsic
                        )
                        new_pcd.transform(corrected_pose)

                        rgbd_mesh.transform(np.linalg.inv(rgbd_previous_pose))
                        rgbd_mesh.transform(corrected_pose)
                        rgbd_previous_pose = copy.deepcopy(corrected_pose)

                        chest_mesh.transform(np.linalg.inv(chest_previous_pose))
                        chest_mesh.transform(pose_4x4)
                        chest_previous_pose = copy.deepcopy(pose_4x4)

                        left_hand_mesh.transform(np.linalg.inv(left_hand_previous_pose))
                        left_hand_mesh.transform(pose_4x4_2)
                        left_hand_previous_pose = copy.deepcopy(pose_4x4_2)

                        right_hand_mesh.transform(
                            np.linalg.inv(right_hand_previous_pose)
                        )
                        right_hand_mesh.transform(pose_4x4_3)
                        right_hand_previous_pose = copy.deepcopy(pose_4x4_3)

                        pcd.points = new_pcd.points
                        pcd.colors = new_pcd.colors

                        self.vis.update_geometry(pcd)
                        self.vis.update_geometry(rgbd_mesh)
                        self.vis.update_geometry(chest_mesh)
                        self.vis.update_geometry(left_hand_mesh)
                        self.vis.update_geometry(right_hand_mesh)

                        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
                            view_params
                        )

                if self.enable_visualization:
                    self.vis.poll_events()
                    self.vis.update_renderer()

                if self.store_frame:
                    self.depth_buffer.append(copy.deepcopy(depth_frame))
                    self.color_buffer.append(copy.deepcopy(color_frame))

                    self.pose_buffer.append(copy.deepcopy(pose_4x4))
                    self.pose2_buffer.append(copy.deepcopy(pose_4x4_2))
                    self.pose3_buffer.append(copy.deepcopy(pose_4x4_3))

                    if self.save_hand:
                        self.rightHandJoint_buffer.append(
                            copy.deepcopy(rightHandJointXyz)
                        )
                        self.leftHandJoint_buffer.append(
                            copy.deepcopy(leftHandJointXyz)
                        )
                        self.rightHandJointOri_buffer.append(
                            copy.deepcopy(rightHandJointOrientation)
                        )
                        self.leftHandJointOri_buffer.append(
                            copy.deepcopy(leftHandJointOrientation)
                        )

                frame_count += 1
                print("streamed frame {}".format(frame_count))
        except Exception as e:
            print("An error occurred:", e)
        finally:
            self.D435_pipeline.stop()
            self.D435_pipeline_2.stop()
            # self.t265_pipeline_3.stop()
            self.pipeline.stop()
            if self.enable_visualization:
                self.vis.destroy_window()

            if self.store_frame:
                print("saving frames...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            save_frame,
                            frame_id,
                            self.out_directory,
                            self.color_buffer,
                            self.depth_buffer,
                            self.pose_buffer,
                            self.pose2_buffer,
                            self.pose3_buffer,
                            self.rightHandJoint_buffer,
                            self.leftHandJoint_buffer,
                            self.rightHandJointOri_buffer,
                            self.leftHandJointOri_buffer,
                            self.save_hand,
                        )
                        for frame_id in range(frame_count)
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        print(future.result(), f" total frame: {frame_count}")