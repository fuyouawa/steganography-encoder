"""
图像处理模块
包含图像编码、解码、格式转换等处理逻辑
"""
import base64
import zlib
import os
from PIL import Image
import numpy as np
import torch

from utils.image import image_to_bytes, bytes_to_image, bytes_to_noise_image, noise_image_to_bytes


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        self.format_mapping = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }

    def process_image_to_noise(self, file_path, compression_level, noise_width, noise_height, use_alpha):
        """处理普通图像转换为噪点图像"""
        try:
            # Determine image format from file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            image_format = self.format_mapping.get(file_ext, 'image/png')

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

                # Add URL prefix with detected format
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                data_url = f"data:{image_format};base64,{base64_data}"
                data_url_bytes = data_url.encode('utf-8')

                # Compress with zlib using configured compression level
                compressed_data = zlib.compress(data_url_bytes, compression_level)

                # Convert to noise image with configured dimensions
                width = noise_width if noise_width > 0 else None
                height = noise_height if noise_height > 0 else None

                noise_image = bytes_to_noise_image(compressed_data, width=width, height=height, use_alpha=use_alpha)

                return noise_image

        except Exception as e:
            raise Exception(f"处理图像时出错: {str(e)}")

    def process_noise_to_image(self, file_path):
        """处理噪点图像转换为普通图像"""
        try:
            # Load noise image
            noise_image = self._load_noise_image(file_path)

            # Convert noise image to bytes
            compressed_data = noise_image_to_bytes(noise_image)

            # Decompress with zlib
            data_url_bytes = zlib.decompress(compressed_data)

            # Extract base64 data from data URL
            data_url = data_url_bytes.decode('utf-8')

            # Parse image format from data URL prefix
            if data_url.startswith("data:"):
                # Find the semicolon after the format
                semicolon_pos = data_url.find(";")
                if semicolon_pos != -1:
                    # Extract the format part (e.g., "image/png")
                    format_part = data_url[5:semicolon_pos]

                    # Check if it's a supported image format
                    supported_formats = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff']
                    if format_part in supported_formats:
                        # Extract base64 data
                        base64_prefix = f"data:{format_part};base64,"
                        base64_data = data_url[len(base64_prefix):]
                    else:
                        raise ValueError(f"不支持的图像格式: {format_part}")
                else:
                    raise ValueError("无效的数据URL格式")
            else:
                raise ValueError("无效的数据URL格式")

            # Decode base64 to get image bytes
            image_bytes = base64.b64decode(base64_data)

            # Convert bytes back to image
            image_tensor, mask = bytes_to_image(image_bytes)

            return image_tensor

        except Exception as e:
            raise Exception(f"处理噪点图像时出错: {str(e)}")

    def _load_noise_image(self, file_path):
        """加载噪点图像并转换为张量"""
        with Image.open(file_path) as img:
            img_array = np.array(img)

        # Convert to float32 [0, 1] range
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return image_tensor

    def save_noise_image(self, noise_image, output_path):
        """保存噪点图像"""
        # Convert tensor to PIL image and save
        image_array = noise_image[0].detach().cpu().numpy()
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