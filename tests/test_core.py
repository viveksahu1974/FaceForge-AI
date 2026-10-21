import numpy as np
import pytest
from faceforge.core import Face, cosine, rgb_image
from faceforge.gallery import FaceGallery


def test_cosine_basics():
    assert cosine(np.array([1., 0.]), np.array([1., 0.])) == pytest.approx(1.0)
    assert cosine(np.array([1., 0.]), np.array([0., 1.])) == pytest.approx(0.0)


def test_rgb_image_validation():
    with pytest.raises(ValueError):
        rgb_image(np.zeros((8, 8), dtype=np.uint8))
    assert rgb_image(np.zeros((8, 8, 3), dtype=np.uint8)).size == (8, 8)


def test_gallery_enrollment_and_matching(tmp_path):
    gallery = FaceGallery(tmp_path / 'nested' / 'gallery.json')
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    gallery.enroll('Alice', vec)
    loaded = FaceGallery(tmp_path / 'nested' / 'gallery.json')
    assert loaded.search(vec) == ('Alice', pytest.approx(1.0))
    other = np.zeros(512, dtype=np.float32)
    other[1] = 1.0
    assert loaded.search(other)[0] == 'Unknown'
    assert loaded.delete('Alice')
    assert not FaceGallery(tmp_path / 'nested' / 'gallery.json').entries


def test_face_area():
    assert Face(np.array([1, 2, 11, 22]), np.zeros((5, 2)), 0.9).area == 200
