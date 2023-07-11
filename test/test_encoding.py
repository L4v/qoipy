from qoipy import qoipy
import numpy as np


RGB = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
BRG = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
GBR = [[0, 255, 0], [0, 0, 255], [255, 0, 0]]


def mock_input():
    input_array = [RGB, BRG, GBR, BRG, GBR, RGB, GBR, RGB, BRG]

    return input_array


def test_rgb_encoding():
    input_array = mock_input()

    expected_encoded_bytes = bytes(
        b"qoif\x00\x00\x00\x03\x00\x00\x00\t\x03\x00Zvm\xc020\xc0.2.20\xc0.2\xc00.0.2\xc00.\xc020\x00\x00\x00\x00\x00\x00\x00\x01"
    )
    encoded_bytes = qoipy.encode(np.array(input_array), 0)
    print(f"Expected: {expected_encoded_bytes}")
    print(f"Encoded: {encoded_bytes}")

    assert expected_encoded_bytes == encoded_bytes
