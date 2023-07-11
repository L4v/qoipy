from __future__ import annotations
from PIL import Image
import numpy as np
import ctypes


PIXEL_CACHE_SIZE = 60
QOI_HEADER_SIZE = 14
QOI_END_MARKER_SIZE = 8
QOI_2BIT_MASK = 0xC0  # 0x11000000
QOI_OP_RGB = 0xFE  # 0x11111110
QOI_OP_RGBA = 0xFF  # 0x11111111
QOI_OP_INDEX = 0x00  # 0x00000000
QOI_OP_DIFF = 0x40  # 0x01000000
QOI_OP_LUMA = 0x80  # 0x10000000
QOI_OP_RUN = 0xC0  # 0x11000000
QOI_END_MARKER = b"\x00\x00\x00\x00\x00\x00\x00\x01"
QOI_MAGIC = b"qoif"
QOI_MAX_RUN_LENGTH = 62


class InvalidHeaderError(Exception):
    def __init__(self, message="Invalid header"):
        self.message = message
        super().__init__(self.message)


class Header:
    def __init__(self, magic: bytes, width, height, channels, colorspace: int) -> None:
        self.magic = magic
        self.width = width
        self.height = height
        self.channels = channels
        self.colorspace = colorspace

    def __repr__(self) -> str:
        return f"""Magic: {self.magic}
Width: {self.width}
Height: {self.height}
Channels: {self.channels}
Colorspace: {self.colorspace}"""

    @classmethod
    def frombytes(cls, header_bytes: bytes) -> Header:
        headerview = memoryview(header_bytes)
        if len(headerview) != 14:
            raise InvalidHeaderError(f"Invalid header length: {len(headerview)}")

        magic = headerview[0:4].tobytes()
        if magic != QOI_MAGIC:
            raise InvalidHeaderError(f"Invalid magic value: {magic}")

        width = int.from_bytes(headerview[4:8].tobytes())
        height = int.from_bytes(headerview[8:12].tobytes())

        if width * height < 0:
            raise InvalidHeaderError(f"Invalid width: {width} or height: {height}")

        channels = headerview[12]

        if channels < 3 or channels > 4:
            raise InvalidHeaderError(f"Invalid number of channels: {channels}")

        colorspace = headerview[13]

        if colorspace < 0 or colorspace > 1:
            raise InvalidHeaderError(f"Invalid colorspace: {colorspace}")

        return Header(magic, width, height, channels, colorspace)

    def tobytes(self) -> bytes:
        return (
            bytes(self.magic)
            + self.width.to_bytes(4, "big")
            + self.height.to_bytes(4, "big")
            + bytes([self.channels, self.colorspace])
        )


def get_pixel_hash_mod_64(r, g, b, a: int = 255) -> int:
    return (r * 3 + g * 5 + b * 7 + a * 11) % 64


def get_diffs(byte: int) -> tuple[int, int, int]:
    diff_r = ((byte & 0x30) >> 4) - 2
    diff_g = ((byte & 0x0C) >> 2) - 2
    diff_b = (byte & 0x03) - 2
    return diff_r, diff_g, diff_b


def get_luma_diffs(first_byte, second_byte: int) -> tuple[int, int, int]:
    diff_g = (first_byte & 0x3F) - 32
    diff_r = ((second_byte & 0xF0) >> 4) + diff_g - 8
    diff_b = (second_byte & 0x0F) + diff_g - 8
    return diff_r, diff_g, diff_b


def decode(fileview: memoryview) -> list[list[list[list]]]:
    fileview_size = len(fileview)
    number_of_chunks = fileview_size - QOI_HEADER_SIZE - QOI_END_MARKER_SIZE

    header_bytes = fileview[:14]
    header = Header.frombytes(header_bytes)

    bytesview = fileview[QOI_HEADER_SIZE : len(fileview) - QOI_END_MARKER_SIZE]
    byte_index = 0
    remaining_bytes = number_of_chunks
    current_pixel = [0, 0, 0, 255]
    decoded_pixels = []
    pixel_cache = [[] for _ in range(PIXEL_CACHE_SIZE)]
    while byte_index < len(bytesview):
        byte = bytesview[byte_index]
        bytes_read = 1

        if byte == QOI_OP_RGBA:
            current_pixel = bytesview[byte_index : byte_index + 4].tolist()
            bytes_read += 4
            decoded_pixels.append(current_pixel.copy())
        elif byte == QOI_OP_RGB:
            current_pixel[0:3] = bytesview[byte_index : byte_index + 3].tolist()
            bytes_read += 3
            decoded_pixels.append(current_pixel.copy())
        elif (byte & QOI_2BIT_MASK) == QOI_OP_RUN:
            run_length = (byte & 0x3F) + 1
            for _ in range(run_length):
                decoded_pixels.append(current_pixel.copy())
        elif (byte & QOI_2BIT_MASK) == QOI_OP_INDEX:
            pixel_index = byte & 0x3F
            current_pixel = pixel_cache[pixel_index].copy()
            decoded_pixels.append(current_pixel.copy())
        elif (byte & QOI_2BIT_MASK) == QOI_OP_DIFF:
            diffs = get_diffs(byte)
            for i, diff in enumerate(diffs):
                current_pixel[i] = (current_pixel[i] + diff) % 256
            decoded_pixels.append(current_pixel.copy())
        elif (byte & QOI_2BIT_MASK) == QOI_OP_LUMA:
            second_byte = bytesview[byte_index + 1]
            bytes_read += 1
            diffs = get_luma_diffs(byte, second_byte)
            for i, diff in enumerate(diffs):
                current_pixel[i] = (current_pixel[i] + diff) % 256
            decoded_pixels.append(current_pixel.copy())

        pixel_index = get_pixel_hash_mod_64(*current_pixel)
        pixel_cache[pixel_index] = current_pixel.copy()
        remaining_bytes -= bytes_read
        byte_index += bytes_read

    decoded_image = []
    for row in range(header.height):
        row_of_pixels = []
        for column in range(header.width):
            row_of_pixels.append(decoded_pixels[row * header.width + column])
        decoded_image.append(row_of_pixels)

    return decoded_image


class InvalidFormatError(Exception):
    def __init__(self, message="Invalid format"):
        self.message = message
        super().__init__(self.message)


def signed_8bit_wraparound(x) -> int:
    return ctypes.c_int8(x).value


def encode(pixels, colorspace: int) -> bytes:
    if len(pixels.shape) != 3:
        raise InvalidFormatError(
            "Invalid pixel array shape, should be: (height, width, channels"
        )

    height = pixels.shape[0]
    width = pixels.shape[1]
    channels = pixels.shape[2]
    if channels < 3 or channels > 4:
        raise InvalidFormatError(f"Invalid number of channels: {channels}")

    header = Header(QOI_MAGIC, width, height, channels, colorspace)
    data = bytearray()

    pixel_cache = [[] for _ in range(PIXEL_CACHE_SIZE)]
    previous_pixel = [0, 0, 0, 255]
    current_pixel = [0, 0, 0, 255]
    pixel_index = 0

    pixels = pixels.reshape(width * height, channels)
    run_length = 0

    while pixel_index < len(pixels):
        current_pixel[:channels] = pixels[pixel_index]
        print(f'Current pixel: {current_pixel}')
        if current_pixel == previous_pixel:
            print(f'Run++, {run_length}')
            run_length += 1
            if run_length == QOI_MAX_RUN_LENGTH:
                print('RUN MAX')
                data.append(QOI_OP_RUN | (run_length - 1))
                run_length = 0

        else:
            if run_length > 0:
                print(f'RUN {run_length}, curr: {current_pixel}')
                data.append(QOI_OP_RUN | (run_length - 1))
                run_length = 0

            cache_index = get_pixel_hash_mod_64(*current_pixel)
            cached_pixel = pixel_cache[cache_index]
            if current_pixel == cached_pixel:
                print(f'INDEX: {cache_index}, curr: {current_pixel}, to append: {QOI_OP_INDEX | cache_index}')
                data.append(QOI_OP_INDEX | cache_index)
            else:
                pixel_cache[cache_index] = current_pixel.copy()
                if current_pixel[3] == previous_pixel[3]:
                    diff_r = signed_8bit_wraparound(
                        current_pixel[0] - previous_pixel[0]
                    )
                    diff_g = signed_8bit_wraparound(
                        current_pixel[1] - previous_pixel[1]
                    )
                    diff_b = signed_8bit_wraparound(
                        current_pixel[2] - previous_pixel[2]
                    )
                    diff_r_g = signed_8bit_wraparound(diff_r - diff_g)
                    diff_b_g = signed_8bit_wraparound(diff_b - diff_g)

                    if (
                        -2 <= diff_r <= 1
                        and -2 <= diff_g <= 1
                        and -2 <= diff_b <= 1
                    ):
                        print('DIFF')
                        dr = (diff_r + 2) << 4
                        dg = (diff_g + 2) << 2
                        db = diff_b + 2
                        data.append(
                            QOI_OP_DIFF
                            | ((diff_r + 2) << 4)
                            | ((diff_g + 2) << 2)
                            | (diff_b + 2)
                        )
                    elif (
                        -32 <= diff_g <= 31
                        and -8 <= diff_r_g <= 7
                        and -8 <= diff_b_g <= 7
                    ):
                        print('LUMA')
                        data.extend(
                            [
                                QOI_OP_LUMA | (diff_g + 32),
                                ((diff_r_g + 8) << 4) | (diff_b_g + 8),
                            ]
                        )
                else:
                    if channels == 3:
                        print('RGB')
                        data.extend([QOI_OP_RGB, *current_pixel[:3]])
                    else:
                        print('RGBA')
                        data.extend([QOI_OP_RGBA, *current_pixel[:4]])
        pixel_index += 1
        previous_pixel = current_pixel.copy()

    return header.tobytes() + bytes(data) + bytes(QOI_END_MARKER)


def qoi_file_to_png(input_filename, output_filename: str) -> None:
    with open(input_filename, "rb") as file:
        fileview = memoryview(file.read())
        decoded_image = decode(fileview)
        print(decoded_image)
        decoded_image = np.array(decoded_image, dtype=np.uint8)
        print(decoded_image)
        img = Image.fromarray(decoded_image, "RGBA")
        img.save(output_filename)


def main() -> None:
    qoi_file_to_png("./imagetest.qoi", "converted.png")
    return


if __name__ == "__main__":
    main()
