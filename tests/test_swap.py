import numpy as np
from faceforge.core import Face
from faceforge.swap import geometric_swap_frame, mark_generated


def test_mark_generated_keeps_dimensions():
    blank = np.zeros((150, 250, 3), dtype=np.uint8)
    result = mark_generated(blank)
    assert result.shape == blank.shape
    assert result.sum() > 0


def test_geometric_swap_synthetic_aligned_faces():
    src = np.zeros((240, 240, 3), dtype=np.uint8)
    src[40:200, 40:200] = (15, 180, 210)
    dst = np.full((240, 240, 3), (60, 95, 115), dtype=np.uint8)
    points = np.asarray([[84, 94], [148, 94], [115, 123], [92, 157], [140, 157]], dtype=np.float32)
    face = Face(np.asarray([55, 50, 180, 195], dtype=np.float32), points, .99)
    result = geometric_swap_frame(src, dst, face, face)
    assert result.shape == dst.shape
    assert result.dtype == np.uint8
