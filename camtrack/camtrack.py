#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences, 
    TriangulationParameters, 
    Correspondences, 
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)



def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    MAX_REPROJECTION_ERROR = 1.6

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    frame_count = len(corner_storage)
    t_vecs = [None] * frame_count
    t_vecs[known_view_1[0]] = known_view_1[1].t_vec
    t_vecs[known_view_2[0]] = known_view_2[1].t_vec
    print(known_view_1[1].t_vec, known_view_2[1].t_vec)
    view_mats = [None] * frame_count # позиции камеры на всех кадрах
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    points_cloud = {} # 'id': coords [., ., .] 3d coordinates, rays [(frame, coords)] - for new points addition

    #     TriangulationParameters = namedtuple(
    #     'TriangulationParameters',
    #     ('max_reprojection_error', 'min_triangulation_angle_deg', 'min_depth')
    # )
    def triangulate(frame_0, frame_1, params=TriangulationParameters(2, 1e-3, 1e-4), ids_to_remove=None):
        corrs = build_correspondences(corner_storage[frame_0], corner_storage[frame_1])
        return triangulate_correspondences(corrs, view_mats[frame_0], view_mats[frame_1], intrinsic_mat, params)
    # pts3d ids med_cos

    # инициализируем облако точек по 2 положениям
    points3d, ids, median_cos = triangulate(known_view_1[0], known_view_2[0])
    for id, point3d in zip(ids, points3d):
        points_cloud[id] = point3d
    

    def find_camera_pose(frame_id):
        # сначала находим инлаеры ранзаком, потом на инлаерах делаем пнп с итеративной мин кв
        corners = corner_storage[frame_id]
        points3d = []
        points2d = []
        for id, point2d in zip(corners.ids.flatten(), corners.points):
            if id in points_cloud.keys():
                points3d.append(points_cloud[id])
                points2d.append(point2d)
        # чтобы епнп работал
        points3d = np.array(points3d, dtype=np.float64)
        points2d = np.array(points2d, dtype=np.float64)
        if len(points3d) < 4:
            return None
        try:
            success, R, t, inliers = cv2.solvePnPRansac(objectPoints=points3d, imagePoints=points2d, cameraMatrix=intrinsic_mat, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9995, reprojectionError=MAX_REPROJECTION_ERROR)
        except Exception as e:
            print(points3d.shape, points2d.shape)
            raise e
        if not success:
            return None

        inliers = np.asarray(inliers ,dtype=np.int32).flatten()
        points3d = np.array(points3d)
        points2d = np.array(points2d)
        points3d = points3d[inliers]
        points2d = points2d[inliers]
        retval, R, t = cv2.solvePnP(objectPoints=points3d, imagePoints=points2d, cameraMatrix=intrinsic_mat, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=R, tvec=t)
        return R, t, len(inliers)

    def frames_without_pose():
        frames = []
        for ind, mat in enumerate(view_mats):
            if mat is None:
                frames.append(ind)
        return frames

    

    frames_of_corner = {}
    for i, corners in enumerate(corner_storage):
        for id_in_list, j in enumerate(corners.ids.flatten()):
            if j not in frames_of_corner.keys():
                frames_of_corner[j] = [[i, id_in_list]]
            else:
                frames_of_corner[j].append([i, id_in_list])

    def try_add_new_point_to_cloud(corner_id):
        best_frames = [None, None]
        frames = []
        for frame in frames_of_corner[corner_id]:
            if view_mats[frame[0]] is not None:
                frames.append(frame)
        if len(frames) < 2:
            return
        
        min_len = 0

        for frame_1 in frames:
            for frame_2 in frames:
                if frame_1 == frame_2:
                    continue
                
                


    while len(frames_without_pose()) > 0:
        # пока есть не найденные положения камеры будем искать лучшее для восстановления
        # т.е. такое, где больше всего инлаеров после применения пнп ранзака
        wanted_frames = frames_without_pose()
        inliers_amount = []
        max_col = -1
        best_frame_id = -1
        best_R = None
        best_t = None
        for frame_id in wanted_frames:
            result = find_camera_pose(frame_id)
            if result is not None:
                R, t, col = result
                if col > max_col:
                    max_col = col
                    best_frame_id = frame_id
                    best_R = R
                    best_t = t
        if max_col == -1:
            # больше позиции камер не находятся
            break
        print('Now we add camera pose on {}-th frame, camera pose was calculated by {} inliers'.format(best_frame_id, max_col))
        ttl_poses_calculated = np.sum([mat is not None for mat in view_mats])
        percent = "{:.0f}".format(ttl_poses_calculated / frame_count * 100.0)
        print('{}% poses calculated'.format(percent))
        view_mats[best_frame_id] = rodrigues_and_translation_to_view_mat3x4(best_R, best_t)
        print('{} points in 3D points cloud'.format(len(points_cloud)))

    last_view_mat = None
    for i in range(frame_count):
        if view_mats[i] is None:
            view_mats[i] = last_view_mat
        else:
            last_view_mat = view_mats[i]


    # corners_0 = corner_storage[0]
    # ids = []
    # points = []
    # for i in corners_0.ids:
    #     ids.append(i[0])
    #     points.append([0, 0, 0])
    
    ids = []
    points = []
    for id in points_cloud.keys():
        ids.append(id)
        points.append(points_cloud[id])

    point_cloud_builder = PointCloudBuilder(np.array(ids, dtype=np.int32), np.array(points))

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
