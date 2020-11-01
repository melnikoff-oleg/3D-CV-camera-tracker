#! /usr/bin/env python3
 
__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]
 
import click
import cv2
import numpy as np
import pims
 
from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli
 
 
class _CornerStorageBuilder:
 
    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()
 
    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)
 
    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))
 
MAX_CORNERS = 4000
PYRAMID_DEPTH = 3
BLOCK_SIZE = 6
WIN_SIZE = (BLOCK_SIZE * 3, BLOCK_SIZE * 3)
MIN_DISTANCE_C = 5
QUALITY_LEVEL = 0.06
 
def get_points_mask(img_shape, points, sizes):
    mask = np.full(img_shape, 255).astype(np.uint8)
    for i in range(len(points)):
        x, y = points[i]
        size = sizes[i]
        mask = cv2.circle(img=mask, center=(int(x), int(y)), radius=int(size), color=0, thickness=-1)
    return mask
 
def remove_old_points(image_0, image_1, ids, points, sizes, depth):
    depth, image_0_pyramid = cv2.buildOpticalFlowPyramid((image_0 * 255).astype(np.uint8), WIN_SIZE, PYRAMID_DEPTH, None, False)
    depth, image_1_pyramid = cv2.buildOpticalFlowPyramid((image_1 * 255).astype(np.uint8), WIN_SIZE, PYRAMID_DEPTH, None, False)
    next_pts = None
    status = None
    err = None
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(image_0_pyramid[0], image_1_pyramid[0], np.array(points, dtype=np.float32).reshape(-1, 2), None, status, err, WIN_SIZE, PYRAMID_DEPTH, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.002))
    status = np.ravel(status)
    cur_points = (status == 1)
    a = list(np.asarray(ids)[cur_points])
    b = list(np.asarray(next_pts)[cur_points])
    c = list(np.asarray(sizes)[cur_points])
    return (a, b, c)
 
def add_new_points(image_1, ids, points, sizes, next_id, mask, first_frame=False):
    depth, image_1_pyramid = cv2.buildOpticalFlowPyramid((image_1 * 255).astype(np.uint8), WIN_SIZE, PYRAMID_DEPTH, None, False)
    for level in range(depth):
        # 0.00075 if first_frame else 0.075,
        new_points = cv2.goodFeaturesToTrack(image_1_pyramid[level], maxCorners=MAX_CORNERS - len(points), qualityLevel=0.00075 if first_frame else 0.075, minDistance=(2 ** depth) * MIN_DISTANCE_C, mask=mask, blockSize=BLOCK_SIZE)
        if new_points is not None:
            new_points = new_points.reshape(-1, 2).astype(np.float32)
        if new_points is not None:
            for (x, y) in new_points:
                if len(points) == MAX_CORNERS:
                    break
                if mask[int(y), int(x)] > 0:
                    ids.append(next_id)
                    next_id += 1
                    points.append((int(x) * (2 ** level), int(y) * (2 ** level)))
                    sizes.append((2 ** level) * BLOCK_SIZE)
                    cv2.circle(mask, (int(x), int(y)), BLOCK_SIZE, color=0, thickness=-1)
        mask = cv2.pyrDown(mask).astype(np.uint8)
    return ids, points, sizes, next_id
 
def _build_impl(frame_sequence: pims.FramesSequence, builder: _CornerStorageBuilder) -> None:
    image_0 = None
    depth, image_0_pyramid = PYRAMID_DEPTH, None
    ids = []
    points = []
    sizes = []
    img_shape = frame_sequence[0].shape
    next_id = 0
 
    for frame, image_1 in enumerate(frame_sequence):
        if len(points) > 0:
            ids, points, sizes = remove_old_points(image_0, image_1, ids, points, sizes, PYRAMID_DEPTH)
        mask = get_points_mask(img_shape, points, sizes)
        ids, points, sizes, next_id = add_new_points(image_1, ids, points, sizes, next_id, mask, frame == 0)
        builder.set_corners_at_frame(frame, FrameCorners(np.array(ids), np.array(points), np.array(sizes)))
        image_0 = image_1
 
 
def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
 
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()
 
 
if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter