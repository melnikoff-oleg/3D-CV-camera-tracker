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
    rodrigues_and_translation_to_view_mat3x4, 
    compute_reprojection_errors
)




def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    # функция для проверки что матрица поворота дейтсвительно матрица поворота
    def check_if_mat_is_rot_mat(mat):
        mat = mat[:, :3]
        # print(mat.shape)
        det = np.linalg.det(mat)
        diff1 = abs(1 - det)
        x = mat @ mat.T
        diff2 = abs(np.sum(np.eye(3) - x))
        eps = 0.001
        # print(diff1, diff2)
        return diff1 < eps and diff2 < eps


    MAX_REPROJECTION_ERROR = 1.6
    MIN_DIST_TRIANGULATE = 0.01

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    t_vecs = [None] * frame_count
    view_mats = [None] * frame_count # позиции камеры на всех кадрах

    # для каждого уголка находим все кадры на котором он есть и его порядковый номер в каждом из них
    cur_corners_occurencies = {}
    frames_of_corner = {}
    for i, corners in enumerate(corner_storage):
        for id_in_list, j in enumerate(corners.ids.flatten()):
            cur_corners_occurencies[j] = 0
            if j not in frames_of_corner.keys():
                frames_of_corner[j] = [[i, id_in_list]]
            else:
                frames_of_corner[j].append([i, id_in_list])

    #по двум кадрам находим общие точки и восстанавливаем их позиции в 3D
    def triangulate(frame_0, frame_1, params=TriangulationParameters(2, 1e-3, 1e-4), ids_to_remove=None):
        corrs = build_correspondences(corner_storage[frame_0], corner_storage[frame_1])
        return triangulate_correspondences(corrs, view_mats[frame_0], view_mats[frame_1], intrinsic_mat, params)

    # debug tool
    # real_t0 = known_view_1[1].t_vec
    # real_t1 = known_view_2[1].t_vec
    # real_viewmat0 = pose_to_view_mat3x4(known_view_1[1])
    # real_viewmat1 = pose_to_view_mat3x4(known_view_2[1])

    #константы для инициализации
    BASELINE_THRESHOLD = 0.9
    REPROJECTION_ERROR_THRESHOLD = 1.4
    MIN_3D_POINTS = 10
    frame_steps = [9, 30, 40]

    #нахождение относительного движения между двумя данными кадрами
    def get_first_cam_pose(frame_0, frame_1):
        corrs = build_correspondences(corner_storage[frame_0], corner_storage[frame_1])
        e_mat, _ = cv2.findEssentialMat(corrs.points_1, corrs.points_2, intrinsic_mat, cv2.RANSAC, 0.999, 2.0)

        def essential_mat_to_camera_view_matrix(e_mat):
            U, S, VH = np.linalg.svd(e_mat, full_matrices=True)
            W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)
            u3 = U[:, 2]
            possible_R = [U @ W @ VH, U @ W.T @ VH]
            possible_t = [u3, -u3]
            best_R = None
            best_t = None
            max_positive_z = 0
            for R in possible_R:
                for t in possible_t:
                    view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
                    view_mat1 = np.hstack((R, t.reshape((3, 1))))
                    view_mats[frame_0] = view_mat0
                    view_mats[frame_1] = view_mat1
                    points3d, ids, median_cos = triangulate(frame_0, frame_1)
                    cur_positive_z = 0
                    for pt in points3d:
                        pt = np.append(pt, np.zeros((1))).reshape((4, 1))
                        pt_transformed0 = (view_mat0 @ pt).flatten()
                        z0 = pt_transformed0[2]
                        pt_transformed1 = (view_mat1 @ pt).flatten()
                        z1 = pt_transformed1[2]
                        cur_positive_z += (z0 > 0.1) + (z1 > 0.1)
                    if cur_positive_z > max_positive_z:
                        max_positive_z = cur_positive_z
                        best_R = R
                        best_t = t
            return best_R, best_t
                    
        R, t = essential_mat_to_camera_view_matrix(e_mat)
        baseline = np.linalg.norm(t)


        view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
        view_mat1 = np.hstack((R, t.reshape((3, 1))))
        view_mats[frame_0] = view_mat0
        view_mats[frame_1] = view_mat1
        points3d, ids, median_cos = triangulate(frame_0, frame_1)
        points0 = []
        points1 = []
        for i in ids:
            for j in frames_of_corner[i]:
                if j[0] == frame_0:
                    points0.append(j[1])
                if j[0] == frame_1:
                    points1.append(j[1])
        points0 = np.array(points0)
        points1 = np.array(points1)
        sum_error = compute_reprojection_errors(points3d, corner_storage[frame_0].points[points0], intrinsic_mat @ view_mat0) + compute_reprojection_errors(points3d, corner_storage[frame_1].points[points1], intrinsic_mat @ view_mat1)
        reprojection_error = np.mean(sum_error)
        points3d_count = len(points3d)

        view_mats[frame_0] = None
        view_mats[frame_1] = None

        #проверяем насколько хороша данная пара кадров, и заодно проверяем что матрица R - реально поворот
        if baseline < BASELINE_THRESHOLD or reprojection_error > REPROJECTION_ERROR_THRESHOLD or points3d_count < MIN_3D_POINTS or not check_if_mat_is_rot_mat(np.hstack((R, t.reshape((3, 1))))):
            return False, baseline, reprojection_error, points3d_count, None, None
        else:
            return True, baseline, reprojection_error, points3d_count, R, t



    #делаем перебор пар кадров для инициализации
    INF = 1e9
    best_frame_step = 5
    best = [-INF, INF, -INF]
    ind = None
    for frame_step in frame_steps:
        for i in range(frame_count // 2):
            if i + frame_step < frame_count * 0.85:
                ok, baseline, reproection_error, points3d_count, Rx, tx = get_first_cam_pose(i, i + frame_step)
                if ok:
                    best = [baseline, reproection_error, points3d_count]
                    ind = i
                    best_frame_step = frame_step
                    if frame_count > 100:
                        break
    


    if ind is None:
        raise NotImplementedError()

    #инициализируемся по лучшей найденной паре
    frame_0 = ind
    frame_1 = ind + best_frame_step
    ok, baseline, reproection_error, points3d_count, R, t = get_first_cam_pose(frame_0, frame_1)
    print('INITIALIZATION RESULTS:')
    print('FRAMES: frame_0={}, frame_1={}, total_frames={}'.format(frame_0, frame_1, frame_count))
    print(ok, baseline, reproection_error, points3d_count, R, t)
    view_mat0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    view_mat1 = np.hstack((R, t.reshape((3, 1))))
    view_mats[frame_0] = view_mat0
    view_mats[frame_1] = view_mat1
    t_vecs[frame_0] = np.zeros((3), dtype=np.float64)
    t_vecs[frame_1] = t


    points_cloud = {} # 'id': coords [., ., .] 3d coordinates, rays [(frame, coords)] - for new points addition



    # инициализируем облако точек по 2 положениям
    points3d, ids, median_cos = triangulate(frame_0, frame_1)
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

    

    def try_add_new_point_to_cloud(corner_id):
        frames = []
        for frame in frames_of_corner[corner_id]:
            if view_mats[frame[0]] is not None:
                frames.append(frame)
        if len(frames) < 2:
            return
        
        max_dist = 0
        best_frames = [None, None]
        best_ids = [None, None]

        for frame_1 in frames:
            for frame_2 in frames:
                if frame_1 == frame_2:
                    continue
                if t_vecs[frame_1[0]] is None or t_vecs[frame_2[0]] is None:
                    continue
                t_vec_1 = t_vecs[frame_1[0]]
                t_vec_2 = t_vecs[frame_2[0]]
                dist = np.linalg.norm(t_vec_1 - t_vec_2)
                if max_dist < dist:
                    max_dist = dist
                    best_frames = [frame_1[0], frame_2[0]]
                    best_ids = [frame_1[1], frame_2[1]]
        if max_dist > MIN_DIST_TRIANGULATE:
            ids = np.array([corner_id])
            points1 = corner_storage[best_frames[0]].points[np.array([best_ids[0]])]
            points2 = corner_storage[best_frames[1]].points[np.array([best_ids[1]])]
            corrs = corrs = Correspondences(
                ids,
                points1,
                points2
            )
            points3d, ids, med_cos = triangulate_correspondences(corrs, view_mats[best_frames[0]], view_mats[best_frames[1]], intrinsic_mat, parameters=TriangulationParameters(2, 1e-3, 1e-4))
            if len(points3d) > 0:
                points_cloud[ids[0]] = points3d[0]
                cur_corners_occurencies.pop(ids[0], None)
                
    def add_corners_occurencies(frame_id):
        two_and_greater = []
        for id in corner_storage[frame_id].ids.flatten():
            if id in cur_corners_occurencies.keys():
                cur_corners_occurencies[id] += 1
                if cur_corners_occurencies[id] >= 2:
                    two_and_greater.append(id)
        return two_and_greater

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
        try_add = add_corners_occurencies(best_frame_id)
        for corner in try_add:
            try_add_new_point_to_cloud(corner)
        print('Now we add camera pose on {}-th frame, camera pose was calculated by {} inliers'.format(best_frame_id, max_col))
        ttl_poses_calculated = np.sum([mat is not None for mat in view_mats])
        percent = "{:.0f}".format(ttl_poses_calculated / frame_count * 100.0)
        print('{}% poses calculated'.format(percent))
        view_mats[best_frame_id] = rodrigues_and_translation_to_view_mat3x4(best_R, best_t)
        t_vecs[best_frame_id] = best_t
        print('{} points in 3D points cloud'.format(len(points_cloud)))

    last_view_mat = None
    for i in range(frame_count):
        if view_mats[i] is None:
            view_mats[i] = last_view_mat
        else:
            last_view_mat = view_mats[i]

    #check that all matrices are rotation matrices
    for ind, i in enumerate(view_mats):
        if not check_if_mat_is_rot_mat(i):
            print('In frame {}/{} we have matrix wich is not rotation matrix:('.format(ind + 1, frame_count))
            raise NotImplementedError()


    
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
