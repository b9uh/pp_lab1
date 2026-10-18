import numpy as np
from grader_contracts.numpy_tasks import MatrixVectorBatchInput
from numpy_tasks import *


def test_sum_prod():
    matrices = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 1], [2, 3]],
    ])
    vectors = np.array([
        [[1], [0]],
        [[0], [1]],
        [[2], [2]],
    ])
    expected = np.array([[27], [21]])
    result = sum_prod(MatrixVectorBatchInput(matrices=matrices, vectors=vectors))
    assert result.shape == (2, 1)
    assert np.array_equal(result, expected)

def test_binarize():
    matrix = np.array([
        [1, 5, 3],
        [7, 2, 9],
    ])
    original = matrix.copy()

    result = binarize(BinarizeInput(matrix=matrix, threshold=4))
    expected = np.array([
        [0, 1, 0],
        [1, 0, 1],
    ])
    assert np.array_equal(result, expected)

    assert np.array_equal(matrix, original)

    assert np.array_equal(
        binarize(BinarizeInput(matrix=np.array([[4]]), threshold=4)),
        np.array([[0]]),
    )

def test_unique_rows():
    matrix = np.array([
        [7, 2, 7, 5],
        [4, 4, 9, 1],
        [8, 3, 8, 3],
    ])

    assert unique_rows(MatrixInput(matrix=matrix)) == [[2, 5, 7], [1, 4, 9], [3, 8]]

def test_unique_columns():
    matrix = np.array([
        [7, 2, 7, 5],
        [4, 4, 9, 1],
        [8, 3, 8, 3],
    ])

    assert unique_columns(MatrixInput(matrix=matrix)) == [[4, 7, 8], [2, 3, 4], [7, 8, 9], [1, 3, 5]]

def test_matrix_statistics_shape():
    data = RandomMatrixInput(rows=4, columns=3, mean=0.0, std=1.0, seed=67)
    stats = matrix_statistics(data)

    assert stats.matrix.shape == (4, 3)
    assert stats.row_means.shape == (4,)
    assert stats.column_means.shape == (3,)
    assert stats.row_variances.shape == (4,)
    assert stats.column_variances.shape == (3,)

def test_matrix_statistics_reproducible():
    data = RandomMatrixInput(rows=5, columns=5, mean=10.0, std=2.0, seed=52)
    a = matrix_statistics(data).matrix
    b = matrix_statistics(data).matrix
    assert np.array_equal(a, b)

def test_matrix_statistics_close_to_params():
    data = RandomMatrixInput(rows=1000, columns=1000, mean=5.0, std=1.5, seed=0)
    stats = matrix_statistics(data)
    assert abs(stats.matrix.mean() - 5.0) < 0.1
    assert abs(stats.matrix.std() - 1.5) < 0.1

def test_chess_example():
    result = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    assert np.array_equal(result, expected)

def test_draw_rectangle():
    data = RectangleInput(
        width=2, height=2,
        image_height=4, image_width=4,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
    image = draw_rectangle(data)
    assert image.shape == (4, 4, 3)
    assert np.array_equal(image[1:3, 1:3], np.array([[[255, 0, 0]] * 2] * 2))
    assert np.array_equal(image[0, 0], np.array([0, 0, 0]))

def test_draw_ellipse():
    data = EllipseInput(
        semi_axis_x=2, semi_axis_y=1,
        image_height=5, image_width=5,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0),
    )
    image = draw_ellipse(data)
    assert image.shape == (5, 5, 3)
    assert np.array_equal(image[2, 2], np.array([255, 0, 0]))
    assert np.array_equal(image[0, 0], np.array([0, 0, 0]))
    assert np.array_equal(image[2, 4], np.array([255, 0, 0]))

def test_analyze_time_series():
    values = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0])
    data = TimeSeriesInput(values=values, window=3)
    stats = analyze_time_series(data)

    assert np.isclose(stats.mean, np.mean(values))
    assert np.isclose(stats.variance, np.var(values))
    assert np.isclose(stats.std, np.std(values))

    assert np.array_equal(stats.local_maxima_indices, np.array([1, 3, 5]))
    assert np.array_equal(stats.local_minima_indices, np.array([2, 4]))

    expected_ma = np.array([2.0, 10/3, 11/3, 5.0, 11/3])
    assert np.allclose(stats.moving_average, expected_ma)

def test_one_hot():
    result = one_hot(OneHotInput(labels=np.array([0, 2, 3, 0]), class_count=4))
    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    test_sum_prod()
    test_binarize()
    test_unique_rows()
    test_unique_columns()
    test_matrix_statistics_shape()
    test_matrix_statistics_reproducible()
    test_matrix_statistics_close_to_params()
    test_chess_example()
    test_draw_rectangle()
    test_draw_ellipse()
    test_analyze_time_series()
    test_one_hot()
    print("OK")