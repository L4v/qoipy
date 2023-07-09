from qoipy import qoipy


def mock_input_array() -> bytes:
    header = b"qoif\x00\x00\x00\x04\x00\x00\x00\x04\x03\x00"
    input_array = bytes(header + b"ZvmV.20n0.2&20.5" + qoipy.QOI_END_MARKER)
    print(f'mock: {input_array}')
    return input_array


def test_rgb_decoding_from_bytearray():
    input_array = mock_input_array()

    # input_array = bytes(b'qoif\x00\')

    expected_decoded_array = [
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]],
        [[0, 0, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 0, 255]],
        [[0, 255, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255], [255, 255, 255, 255]],
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 0, 0, 255]],
    ]
    decoded_array = qoipy.decode(memoryview(input_array))
    assert expected_decoded_array == decoded_array
