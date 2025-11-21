import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import base64
import zlib
from PIL import Image
import io
import os
from utils.image import image_to_bytes, bytes_to_image, bytes_to_noise_image, noise_image_to_bytes

class DragDropBox:
    def __init__(self, parent, label_text, drop_callback):
        self.frame = tk.Frame(parent, bd=2, relief="solid", width=300, height=200)
        self.frame.pack_propagate(False)
        self.label = tk.Label(self.frame, text=label_text, font=("Arial", 12))
        self.label.pack(expand=True)
        self.drop_callback = drop_callback

        # Enable drag and drop using tkinterdnd2
        self.frame.drop_target_register(DND_FILES)
        self.frame.dnd_bind('<<Drop>>', self.on_drop)

        # Also bind click event for file dialog fallback
        self.frame.bind("<Button-1>", self.on_click)

    def on_drop(self, event):
        # Handle file drop
        self.frame.config(bg="white")

        # Get the dropped file path
        file_path = event.data

        # tkinterdnd2 returns file paths in a specific format
        # On Windows, it might be in curly braces or as a string
        if file_path.startswith('{') and file_path.endswith('}'):
            # Remove curly braces and split multiple files
            file_path = file_path[1:-1]
            files = file_path.split('} {')
            # Use the first file
            file_path = files[0]

        # Clean up the file path (remove any quotes or extra characters)
        file_path = file_path.strip().strip('"')

        if file_path and os.path.exists(file_path):
            self.drop_callback(file_path)

    def on_click(self, _):
        # Fallback: open file dialog on click
        file_path = filedialog.askopenfilename()
        if file_path:
            self.drop_callback(file_path)

class NoiseImageEncoder:
    def __init__(self, root):
        self.root = root
        self.root.title("噪点图像编码器")
        self.root.geometry("700x600")

        # Configuration variables
        self.compression_level = tk.IntVar(value=-1)
        self.noise_width = tk.IntVar(value=0)
        self.noise_height = tk.IntVar(value=0)
        self.use_alpha = tk.BooleanVar(value=False)

        # Create configuration frame
        self.create_config_frame(root)

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create drag and drop boxes
        self.create_drag_drop_boxes(main_frame)

        # Status label
        self.status_label = tk.Label(root, text="拖拽图像文件到上方方框", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=10)

    def create_config_frame(self, parent):
        """Create configuration options frame at the top"""
        config_frame = tk.LabelFrame(parent, text="编码选项", font=("Arial", 10, "bold"), padx=10, pady=10)
        config_frame.pack(fill="x", padx=20, pady=(20, 10))

        # Compression level
        tk.Label(config_frame, text="压缩等级:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        compression_frame = tk.Frame(config_frame)
        compression_frame.grid(row=0, column=1, sticky="w")

        compression_scale = tk.Scale(compression_frame, from_=-1, to=9, orient="horizontal",
                                   variable=self.compression_level, showvalue=True, length=200)
        compression_scale.pack(side="left")
        tk.Label(compression_frame, text="(-1=默认, 0=无压缩, 9=最大压缩)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Noise image dimensions
        tk.Label(config_frame, text="噪点图像尺寸:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        dimensions_frame = tk.Frame(config_frame)
        dimensions_frame.grid(row=1, column=1, sticky="w", pady=(10, 0))

        tk.Label(dimensions_frame, text="宽:").pack(side="left")
        width_entry = tk.Entry(dimensions_frame, textvariable=self.noise_width, width=6)
        width_entry.pack(side="left", padx=(2, 10))

        tk.Label(dimensions_frame, text="高:").pack(side="left")
        height_entry = tk.Entry(dimensions_frame, textvariable=self.noise_height, width=6)
        height_entry.pack(side="left", padx=(2, 0))

        tk.Label(dimensions_frame, text="(0=不指定)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Alpha channel option
        tk.Label(config_frame, text="使用alpha通道:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        alpha_check = tk.Checkbutton(config_frame, variable=self.use_alpha)
        alpha_check.grid(row=2, column=1, sticky="w", pady=(10, 0))

    def create_drag_drop_boxes(self, parent):
        # Left box for regular images -> noise images
        left_frame = tk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.left_box = DragDropBox(left_frame, "拖拽普通图像文件到这里\n(转换为噪点图像)", self.process_image_to_noise)
        self.left_box.frame.pack(fill="both", expand=True)

        # Right box for noise images -> regular images
        right_frame = tk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.right_box = DragDropBox(right_frame, "拖拽噪点图像文件到这里\n(转换为普通图像)", self.process_noise_to_image)
        self.right_box.frame.pack(fill="both", expand=True)

    def process_image_to_noise(self, file_path):
        try:
            self.status_label.config(text="正在处理图像...")

            # Determine image format from file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            format_mapping = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff'
            }
            image_format = format_mapping.get(file_ext, 'image/png')

            # Load and process image
            with Image.open(file_path) as img:
                # Handle alpha channel based on configuration
                if self.use_alpha.get():
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                # Convert to numpy array
                import numpy as np
                img_array = np.array(img)

                # Convert image to bytes
                image_bytes = image_to_bytes(img_array)

                # Add URL prefix with detected format
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                data_url = f"data:{image_format};base64,{base64_data}"
                data_url_bytes = data_url.encode('utf-8')

                # Compress with zlib using configured compression level
                compression_level = self.compression_level.get()
                compressed_data = zlib.compress(data_url_bytes, compression_level)

                # Convert to noise image with configured dimensions
                width = self.noise_width.get()
                height = self.noise_height.get()
                use_alpha = self.use_alpha.get()

                # Use 0 to indicate no specification (None)
                noise_width = width if width > 0 else None
                noise_height = height if height > 0 else None

                noise_image = bytes_to_noise_image(compressed_data, width=noise_width, height=noise_height, use_alpha=use_alpha)

                # Save noise image
                output_path = self.get_output_path(file_path, "_noise")
                self.save_noise_image(noise_image, output_path)

                self.status_label.config(text=f"转换成功！噪点图像已保存到: {output_path}")
                messagebox.showinfo("成功", f"噪点图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理图像时出错:\n{str(e)}")

    def process_noise_to_image(self, file_path):
        try:
            self.status_label.config(text="正在处理噪点图像...")

            # Load noise image
            noise_image = self.load_noise_image(file_path)

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

            # Save regular image
            output_path = self.get_output_path(file_path, "_decoded")
            self.save_regular_image(image_tensor, output_path)

            self.status_label.config(text=f"转换成功！普通图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"普通图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理噪点图像时出错:\n{str(e)}")

    def get_output_path(self, input_path, suffix):
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        return os.path.join(directory, f"{name}{suffix}{ext}")

    def save_noise_image(self, noise_image, output_path):
        # Convert tensor to PIL image and save
        import numpy as np
        image_array = noise_image[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        # Determine image mode based on number of channels
        if image_array.shape[2] == 4:
            pil_image = Image.fromarray(image_array, 'RGBA')
        else:
            pil_image = Image.fromarray(image_array, 'RGB')

        pil_image.save(output_path)

    def load_noise_image(self, file_path):
        # Load image and convert to tensor
        import numpy as np
        with Image.open(file_path) as img:
            img_array = np.array(img)

        # Convert to float32 [0, 1] range
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        import torch
        image_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return image_tensor

    def save_regular_image(self, image_tensor, output_path):
        # Convert tensor to PIL image and save
        import numpy as np
        image_array = image_tensor[0].detach().cpu().numpy()
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image_array)
        pil_image.save(output_path)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = NoiseImageEncoder(root)
    root.mainloop()
