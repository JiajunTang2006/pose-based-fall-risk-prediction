"""
姿态后端适配层的单元测试。

这里主要测试 YOLO COCO 17 点到项目内部 MediaPipe 33 点格式的映射。
测试不依赖 ultralytics 或真实模型文件，所以在普通测试环境里也能稳定运行。
"""

import unittest

from fall_prediction.landmarks import (
    LANDMARK_COUNT,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Landmark,
)
from fall_prediction.pose import coco17_to_mediapipe_landmarks, visible_landmark_bbox


class YOLOPoseMappingTest(unittest.TestCase):
    def test_coco17_keypoints_are_mapped_to_mediapipe33_slots(self):
        xy = [[0.0, 0.0] for _ in range(17)]
        xy[5] = [50.0, 100.0]    # COCO left shoulder
        xy[6] = [150.0, 100.0]   # COCO right shoulder
        xy[11] = [60.0, 200.0]   # COCO left hip
        xy[12] = [140.0, 200.0]  # COCO right hip
        conf = [0.0 for _ in range(17)]
        conf[5] = 0.91
        conf[6] = 0.92
        conf[11] = 0.81
        conf[12] = 0.82

        landmarks = coco17_to_mediapipe_landmarks(
            xy=xy,
            conf=conf,
            image_width=200,
            image_height=400,
        )

        self.assertEqual(len(landmarks), LANDMARK_COUNT)
        self.assertAlmostEqual(landmarks[LEFT_SHOULDER].x, 0.25)
        self.assertAlmostEqual(landmarks[LEFT_SHOULDER].y, 0.25)
        self.assertAlmostEqual(landmarks[RIGHT_SHOULDER].x, 0.75)
        self.assertAlmostEqual(landmarks[RIGHT_SHOULDER].y, 0.25)
        self.assertAlmostEqual(landmarks[LEFT_HIP].x, 0.30)
        self.assertAlmostEqual(landmarks[LEFT_HIP].y, 0.50)
        self.assertAlmostEqual(landmarks[RIGHT_HIP].x, 0.70)
        self.assertAlmostEqual(landmarks[RIGHT_HIP].y, 0.50)
        self.assertAlmostEqual(landmarks[LEFT_SHOULDER].visibility, 0.91)

    def test_missing_mediapipe_only_keypoints_stay_invisible(self):
        xy = [[10.0, 10.0] for _ in range(17)]

        landmarks = coco17_to_mediapipe_landmarks(
            xy=xy,
            conf=None,
            image_width=100,
            image_height=100,
        )

        self.assertEqual(landmarks[LEFT_HEEL].visibility, 0.0)

    def test_visible_landmark_bbox_uses_only_reliable_points(self):
        landmarks = [Landmark(0.0, 0.0, visibility=0.0) for _ in range(LANDMARK_COUNT)]
        landmarks[LEFT_SHOULDER] = Landmark(0.25, 0.25, visibility=0.9)
        landmarks[RIGHT_SHOULDER] = Landmark(0.75, 0.25, visibility=0.9)
        landmarks[LEFT_HIP] = Landmark(0.30, 0.50, visibility=0.9)
        landmarks[RIGHT_HIP] = Landmark(0.70, 0.50, visibility=0.9)

        bbox = visible_landmark_bbox(landmarks, image_width=200, image_height=400)

        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertLess(x1, 50)
        self.assertLess(y1, 100)
        self.assertGreater(x2, 150)
        self.assertGreater(y2, 200)


if __name__ == "__main__":
    unittest.main()
