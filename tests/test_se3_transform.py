# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Se3Transform class."""

import numpy as np
import pytest

from sfmtool._sfmtool import RotQuaternion, Se3Transform


class TestFromAxisAngle:
    def test_rotation_around_z_axis_90_degrees(self):
        transform = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        point = np.array([1.0, 0.0, 0.0])
        result = transform.apply_to_point(point)
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_rotation_around_x_axis_180_degrees(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 0, 0]), np.pi)
        point = np.array([0.0, 1.0, 0.0])
        result = transform.apply_to_point(point)
        expected = np.array([0.0, -1.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_rotation_around_y_axis_90_degrees(self):
        transform = Se3Transform.from_axis_angle(np.array([0, 1, 0]), np.pi / 2)
        point = np.array([1.0, 0.0, 0.0])
        result = transform.apply_to_point(point)
        expected = np.array([0.0, 0.0, -1.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_zero_angle_is_identity(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 2, 3]), 0.0)
        point = np.array([5.0, -3.0, 7.0])
        result = transform.apply_to_point(point)
        assert np.allclose(result, point, atol=1e-10)

    def test_axis_is_normalized(self):
        transform1 = Se3Transform.from_axis_angle(np.array([0, 0, 10]), np.pi / 2)
        transform2 = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        point = np.array([1.0, 0.0, 0.0])
        result1 = transform1.apply_to_point(point)
        result2 = transform2.apply_to_point(point)
        assert np.allclose(result1, result2, atol=1e-10)

    def test_zero_axis_raises_error(self):
        with pytest.raises(ValueError):
            Se3Transform.from_axis_angle(np.array([0, 0, 0]), np.pi / 2)

    def test_has_identity_translation_and_scale(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        assert np.allclose(transform.translation, np.zeros(3))
        assert transform.scale == 1.0

    def test_arbitrary_axis_rotation(self):
        axis = np.array([1, 1, 1])
        angle = 2 * np.pi / 3
        transform = Se3Transform.from_axis_angle(axis, angle)
        result = transform.apply_to_point(np.array([1.0, 0.0, 0.0]))
        assert np.allclose(result, np.array([0.0, 1.0, 0.0]), atol=1e-10)
        result = transform.apply_to_point(np.array([0.0, 1.0, 0.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 1.0]), atol=1e-10)


class TestApplyToPoints:
    def test_apply_to_single_point(self):
        transform = Se3Transform(
            rotation=RotQuaternion(1, 0, 0, 0),
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
        )
        point = np.array([1.0, 1.0, 1.0])
        result = transform.apply_to_point(point)
        expected = np.array([3.0, 4.0, 5.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_apply_to_multiple_points(self):
        transform = Se3Transform(
            translation=np.array([10.0, 0.0, 0.0]),
            scale=1.0,
        )
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        result = transform.apply_to_points(points)
        expected = np.array([[10.0, 0.0, 0.0], [11.0, 2.0, 3.0], [9.0, -2.0, -3.0]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_apply_to_points_matches_individual(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 2, 3]), 0.5)
        transform.translation = np.array([1.0, -1.0, 2.0])
        transform.scale = 1.5
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        batch_result = transform.apply_to_points(points)
        for i, point in enumerate(points):
            individual_result = transform.apply_to_point(point)
            assert np.allclose(batch_result[i], individual_result, atol=1e-10)


class TestCompose:
    def test_compose_translations(self):
        t1 = Se3Transform(translation=np.array([1.0, 0.0, 0.0]))
        t2 = Se3Transform(translation=np.array([0.0, 2.0, 0.0]))
        composed = t1.compose(t2)
        point = np.array([0.0, 0.0, 0.0])
        result = composed.apply_to_point(point)
        expected = np.array([1.0, 2.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_compose_scales(self):
        t1 = Se3Transform(scale=2.0)
        t2 = Se3Transform(scale=3.0)
        composed = t1.compose(t2)
        assert composed.scale == 6.0

    def test_compose_rotations(self):
        t1 = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        t2 = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        composed = t1.compose(t2)
        point = np.array([1.0, 0.0, 0.0])
        result = composed.apply_to_point(point)
        expected = np.array([-1.0, 0.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_compose_equals_sequential_apply(self):
        t1 = Se3Transform.from_axis_angle(np.array([1, 0, 0]), np.pi / 4)
        t1.translation = np.array([1.0, 2.0, 3.0])
        t1.scale = 1.5
        t2 = Se3Transform.from_axis_angle(np.array([0, 1, 0]), np.pi / 3)
        t2.translation = np.array([-1.0, 0.0, 2.0])
        t2.scale = 0.8
        composed = t1.compose(t2)
        point = np.array([5.0, -3.0, 2.0])
        intermediate = t1.apply_to_point(point)
        sequential_result = t2.apply_to_point(intermediate)
        composed_result = composed.apply_to_point(point)
        assert np.allclose(sequential_result, composed_result, atol=1e-10)


class TestInverse:
    def test_inverse_translation(self):
        transform = Se3Transform(translation=np.array([5.0, -3.0, 2.0]))
        inverse = transform.inverse()
        point = np.array([1.0, 2.0, 3.0])
        transformed = transform.apply_to_point(point)
        recovered = inverse.apply_to_point(transformed)
        assert np.allclose(recovered, point, atol=1e-10)

    def test_inverse_rotation(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 2, 3]), 0.7)
        inverse = transform.inverse()
        point = np.array([1.0, 2.0, 3.0])
        transformed = transform.apply_to_point(point)
        recovered = inverse.apply_to_point(transformed)
        assert np.allclose(recovered, point, atol=1e-10)

    def test_inverse_scale(self):
        transform = Se3Transform(scale=2.5)
        inverse = transform.inverse()
        assert np.isclose(inverse.scale, 1.0 / 2.5)
        point = np.array([1.0, 2.0, 3.0])
        transformed = transform.apply_to_point(point)
        recovered = inverse.apply_to_point(transformed)
        assert np.allclose(recovered, point, atol=1e-10)

    def test_inverse_combined(self):
        transform = Se3Transform.from_axis_angle(np.array([0, 1, 0]), np.pi / 6)
        transform.translation = np.array([10.0, -5.0, 3.0])
        transform.scale = 2.0
        inverse = transform.inverse()
        point = np.array([7.0, -2.0, 8.0])
        transformed = transform.apply_to_point(point)
        recovered = inverse.apply_to_point(transformed)
        assert np.allclose(recovered, point, atol=1e-10)

    def test_inverse_zero_scale_raises(self):
        transform = Se3Transform(scale=0.0)
        with pytest.raises(ValueError, match="scale=0 cannot be inverted"):
            transform.inverse()

    def test_compose_with_inverse_is_identity(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 1, 0]), 0.5)
        transform.translation = np.array([1.0, 2.0, 3.0])
        transform.scale = 1.5
        inverse = transform.inverse()
        composed = transform.compose(inverse)
        point = np.array([5.0, -3.0, 7.0])
        result = composed.apply_to_point(point)
        assert np.allclose(result, point, atol=1e-10)


class TestDictSerialization:
    def test_round_trip(self):
        original = Se3Transform.from_axis_angle(np.array([1, 2, 3]), 0.8)
        original.translation = np.array([5.0, -3.0, 2.0])
        original.scale = 2.5
        dict_repr = original.to_dict()
        recovered = Se3Transform.from_dict(dict_repr)
        point = np.array([1.0, 2.0, 3.0])
        original_result = original.apply_to_point(point)
        recovered_result = recovered.apply_to_point(point)
        assert np.allclose(original_result, recovered_result, atol=1e-10)

    def test_to_dict_structure(self):
        transform = Se3Transform(
            rotation=RotQuaternion(0.5, 0.5, 0.5, 0.5),
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
        )
        d = transform.to_dict()
        assert "rotation" in d
        assert np.isclose(d["rotation"]["w"], 0.5)
        assert np.isclose(d["rotation"]["x"], 0.5)
        assert np.isclose(d["rotation"]["y"], 0.5)
        assert np.isclose(d["rotation"]["z"], 0.5)
        assert "translation" in d
        assert np.allclose(d["translation"], [1.0, 2.0, 3.0])
        assert "scale" in d
        assert d["scale"] == 2.0

    def test_from_dict_default_scale(self):
        d = {
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "translation": [0.0, 0.0, 0.0],
        }
        transform = Se3Transform.from_dict(d)
        assert transform.scale == 1.0


class TestDefaultValues:
    def test_default_identity_transform(self):
        transform = Se3Transform()
        point = np.array([5.0, -3.0, 7.0])
        result = transform.apply_to_point(point)
        assert np.allclose(result, point, atol=1e-10)

    def test_default_rotation_is_identity(self):
        transform = Se3Transform()
        assert np.isclose(transform.rotation.w, 1.0)
        assert np.isclose(transform.rotation.x, 0.0)
        assert np.isclose(transform.rotation.y, 0.0)
        assert np.isclose(transform.rotation.z, 0.0)

    def test_default_translation_is_zero(self):
        transform = Se3Transform()
        assert np.allclose(transform.translation, np.zeros(3))

    def test_default_scale_is_one(self):
        transform = Se3Transform()
        assert transform.scale == 1.0


class TestMatmulOperator:
    def test_matmul_compose_transforms(self):
        t1 = Se3Transform(translation=np.array([1.0, 0.0, 0.0]))
        t2 = Se3Transform(translation=np.array([0.0, 2.0, 0.0]))
        composed = t1 @ t2
        assert isinstance(composed, Se3Transform)
        point = np.array([0.0, 0.0, 0.0])
        assert np.allclose(
            composed.apply_to_point(point), t1.compose(t2).apply_to_point(point)
        )

    def test_matmul_single_point(self):
        transform = Se3Transform(
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
        )
        point = np.array([1.0, 1.0, 1.0])
        result = transform @ point
        expected = transform.apply_to_point(point)
        assert np.allclose(result, expected)

    def test_matmul_multiple_points(self):
        transform = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 4)
        transform.translation = np.array([5.0, 0.0, 0.0])
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = transform @ points
        expected = transform.apply_to_points(points)
        assert np.allclose(result, expected)

    def test_matmul_chaining_transforms(self):
        t1 = Se3Transform.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        t2 = Se3Transform(translation=np.array([1.0, 0.0, 0.0]))
        t3 = Se3Transform(scale=2.0)
        combined = t1 @ t2 @ t3
        point = np.array([1.0, 0.0, 0.0])
        p1 = t1.apply_to_point(point)
        p2 = t2.apply_to_point(p1)
        p3 = t3.apply_to_point(p2)
        result = combined @ point
        assert np.allclose(result, p3, atol=1e-10)

    def test_matmul_invalid_array_shape_1d(self):
        transform = Se3Transform()
        with pytest.raises(ValueError, match="Array must have shape"):
            transform @ np.array([1.0, 2.0])

    def test_matmul_invalid_array_shape_2d(self):
        transform = Se3Transform()
        with pytest.raises(ValueError, match="Array must have shape"):
            transform @ np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_matmul_invalid_type(self):
        transform = Se3Transform()
        with pytest.raises(TypeError):
            transform @ "invalid"
        with pytest.raises(TypeError):
            transform @ 42

    def test_matmul_matches_compose(self):
        t1 = Se3Transform.from_axis_angle(np.array([1, 2, 3]), 0.5)
        t1.translation = np.array([1.0, -1.0, 2.0])
        t1.scale = 1.5
        t2 = Se3Transform.from_axis_angle(np.array([0, 1, 0]), 0.3)
        t2.translation = np.array([0.0, 5.0, -1.0])
        t2.scale = 0.8
        matmul_result = t1 @ t2
        compose_result = t1.compose(t2)
        point = np.array([7.0, -3.0, 2.0])
        assert np.allclose(
            matmul_result.apply_to_point(point),
            compose_result.apply_to_point(point),
            atol=1e-10,
        )

    def test_matmul_matches_apply_to_point(self):
        transform = Se3Transform.from_axis_angle(np.array([1, 0, 0]), np.pi / 3)
        transform.translation = np.array([10.0, -5.0, 3.0])
        transform.scale = 2.0
        point = np.array([1.0, 2.0, 3.0])
        assert np.allclose(
            transform @ point, transform.apply_to_point(point), atol=1e-10
        )

    def test_matmul_matches_apply_to_points(self):
        transform = Se3Transform.from_axis_angle(np.array([0, 1, 0]), np.pi / 6)
        transform.translation = np.array([1.0, 2.0, 3.0])
        transform.scale = 0.5
        points = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        assert np.allclose(
            transform @ points, transform.apply_to_points(points), atol=1e-10
        )
