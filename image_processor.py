"""
图像处理模块
包含图像编码、解码、格式转换等处理逻辑
"""
import zlib
import os
from PIL import Image
import numpy as np
import torch

from utils.image import image_to_bytes, bytes_to_image
from utils.encoding import encode_steganography, decode_steganography


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        pass

    def process_image_to_steganography(self, file_path, compression_level, steganography_width, steganography_height, use_alpha):
        """处理普通图像转换为隐写图像"""
        try:
            # Load and process image
            with Image.open(file_path) as img:
                # Handle alpha channel based on configuration
                if use_alpha:
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                # Convert to numpy array
                img_array = np.array(img)

                # Convert image to bytes
                image_bytes = image_to_bytes(img_array)

                # Compress with zlib using configured compression level
                compressed_data = zlib.compress(image_bytes, compression_level)

                # Convert to steganography image with configured dimensions
                width = steganography_width if steganography_width > 0 else None
                height = steganography_height if steganography_height > 0 else None

                steganography_image = encode_steganography(compressed_data, width=width, height=height, use_alpha=use_alpha)

                return steganography_image

        except Exception as e:
            raise Exception(f"处理图像时出错: {str(e)}")

    def process_steganography_to_image(self, file_path):
        """处理隐写图像转换为普通图像"""
        try:
            # Load steganography image
            steganography_image = self._load_steganography_image(file_path)

            # Decode bytes from steganography image
            compressed_data = decode_steganography(steganography_image)

            # Decompress with zlib
            image_bytes = zlib.decompress(compressed_data)

            # Convert bytes back to image
            image_tensor, mask = bytes_to_image(image_bytes)

            return image_tensor

        except Exception as e:
            raise Exception(f"处理隐写图像时出错: {str(e)}")

    def _load_steganography_image(self, file_path):
        """加载隐写图像并转换为张量"""
        with Image.open(file_path) as img:
            img_array = np.array(img)

        # Convert to float32 [0, 1] range
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return image_tensor

    def save_steganography_image(self, steganography_image, output_path):
        """保存隐写图像"""
        # Convert tensor to PIL image and save
        image_array = steganography_image[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        # Determine image mode based on number of channels
        if image_array.shape[2] == 4:
            pil_image = Image.fromarray(image_array, 'RGBA')
        else:
            pil_image = Image.fromarray(image_array, 'RGB')

        pil_image.save(output_path)

    def save_regular_image(self, image_tensor, output_path):
        """保存普通图像"""
        # Convert tensor to PIL image and save
        image_array = image_tensor[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image_array)
        pil_image.save(output_path)

    def get_output_path(self, input_path, suffix, custom_extension=None):
        """生成输出文件路径"""
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        if custom_extension is not None:
            ext = f".{custom_extension}"
        return os.path.join(directory, f"{name}{suffix}{ext}")