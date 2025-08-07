import pyrealsense2 as rs
import sys
import numpy as np
import cv2
import open3d as o3d
# from open3d.core import PinholeCameraIntrinsic
from enum import IntEnum
import time
from realsenseprocess import *
from yonemaru.utils.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
sys.path.append('../maskrcnn/Mask_RCNN/samples/block')
import maskrcnn.Mask_RCNN.samples.block.estimte_block as est

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 1
    HighAccuracy = 5
    HighDensity = 4
    MediumDensity = 5

# RealSenseカメラの設定
pipeline = rs.pipeline()
config = rs.config()

WIDTH=640
HEIGHT=480

# カメラのストリームを設定（RGBと深度）
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 60)#30は毎秒フレーム数
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 60)



##depthとcolorの画角がずれないようにalignを生成
align=rs.align(rs.stream.color)
# パイプラインを開始
pipeline_profile=pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

# get camera intrinsics
intr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
#

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters

firstfirst = True
recording=False
with KeystrokeCounter() as key_counter:
    try:
        while True:
            press_events = key_counter.get_press_events()
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='s'):
                    recording = True
                    print("saving")
                    key_counter.clear()
            # 1つ目のフレームを取得
            t1=time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames=align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                continue

            # カラーカメラの内部パラメータを取得
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics



            ##深度フレームの取得
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                continue


            # BGR画像をNumPy配列に変換
            image_bgr = np.asanyarray(color_frame.get_data())
            color_image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            depth_image = np.asanyarray(depth_frame.get_data())



            depth=o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)



            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)




            color_pcd = np.concatenate((np.array(pcd.points), np.array(pcd.colors)), axis=-1)

            ####処理#######
            processed_pcd1,processed_pcd2=preprocess_point_cloud(color_pcd,use_cuda=True)

            # # update pointcloud visualization
            pcd_vis.points = o3d.utility.Vector3dVector(color_pcd[:, :3])
            pcd_vis.colors = o3d.utility.Vector3dVector(color_pcd[:, 3:])


            if firstfirst:
                # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.000000001, origin=[0, 0, 0])

                # ポイントサイズを変更する
                render_option = vis.get_render_option()
                render_option.point_size = 15.0  # ← 点を大きく表示（デフォルトは1.0）
                vis.add_geometry(pcd_vis)
                # vis.add_geometry(axis)
                view_ctl = vis.get_view_control()
                view_ctl.set_lookat([0, 0, 0])  # 注視点（見ている中心）
                view_ctl.set_front([-1, 0, 1])  # 視線ベクトル（前方向）
                view_ctl.set_up([0, 0, 1])  # 上方向ベクトル
                view_ctl.set_zoom(0.03)

                firstfirst = False
            else:
                vis.update_geometry(pcd_vis)
                if recording is True:
                    vis.poll_events()
                    vis.update_renderer()
                    vis.capture_screen_image("data_collect_pointcloud_image1.png", do_render=False)

                    pcd_vis.points = o3d.utility.Vector3dVector(processed_pcd2[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(processed_pcd2[:, 3:])
                    vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()
                    vis.capture_screen_image("data_collect_pointcloud_image2.png", do_render=False)


                    pcd_vis.points = o3d.utility.Vector3dVector(processed_pcd1[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(processed_pcd1[:, 3:])
                    vis.update_geometry(pcd_vis)
                    vis.poll_events()
                    vis.update_renderer()
                    vis.capture_screen_image("data_collect_pointcloud_image3.png", do_render=False)
                    break

            vis.poll_events()
            vis.update_renderer()
            # #
            t2 = time.time()

            print(f"周期:{t2-t1}")

            # 手のトラッキングを行う
            # frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            # o3d.visualization.draw_geometries([pcd])




            # if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            #     break


    finally:
        # パイプラインを停止
        pipeline.stop()
        cv2.destroyAllWindows()

