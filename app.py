"""
主应用类模块
包含应用程序的主要逻辑和界面
"""
import tkinter as tk
from tkinter import messagebox

from gui_components import DragDropBox
from config_manager import ConfigManager
from image_processor import ImageProcessor


class SteganographyEncoder:
    """隐写图像编码器主应用类"""

    def __init__(self, root):
        self.root = root
        self.root.title("隐写图像编码器")
        self.root.geometry("700x600")

        # Initialize managers
        self.config_manager = ConfigManager()
        self.image_processor = ImageProcessor()

        # Create configuration frame
        self.config_manager.create_config_frame(root)

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Create drag and drop boxes
        self.create_drag_drop_boxes(main_frame)

        # Status label
        self.status_label = tk.Label(root, text="拖拽图像文件到上方方框", font=("Arial", 10))
        self.status_label.pack(side="bottom", pady=10)

    def create_drag_drop_boxes(self, parent):
        """创建拖拽框"""
        # Left box for regular images -> noise images
        left_frame = tk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.left_box = DragDropBox(left_frame, "拖拽普通图像文件到这里\n(转换为隐写图像)", self.process_image_to_steganography)
        self.left_box.frame.pack(fill="both", expand=True)

        # Right box for steganography images -> regular images
        right_frame = tk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.right_box = DragDropBox(right_frame, "拖拽隐写图像文件到这里\n(转换为普通图像)", self.process_steganography_to_image)
        self.right_box.frame.pack(fill="both", expand=True)

    def process_image_to_steganography(self, file_path):
        """处理普通图像转换为隐写图像"""
        try:
            self.status_label.config(text="正在处理图像...")

            # Get configuration
            compression_level = self.config_manager.get_compression_level()
            steganography_width = self.config_manager.get_noise_width()
            steganography_height = self.config_manager.get_noise_height()
            use_alpha = self.config_manager.get_use_alpha()

            # Process image
            steganography_image = self.image_processor.process_image_to_steganography(
                file_path, compression_level, steganography_width, steganography_height, use_alpha
            )

            # Save steganography image
            output_path = self.image_processor.get_output_path(file_path, "_steganography", custom_extension="png")
            self.image_processor.save_steganography_image(steganography_image, output_path)

            self.status_label.config(text=f"转换成功！隐写图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"隐写图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理图像时出错:\n{str(e)}")

    def process_steganography_to_image(self, file_path):
        """处理隐写图像转换为普通图像"""
        try:
            self.status_label.config(text="正在处理隐写图像...")

            # Process steganography image
            image_tensor = self.image_processor.process_steganography_to_image(file_path)

            # Save regular image
            output_path = self.image_processor.get_output_path(file_path, "_decoded")
            self.image_processor.save_regular_image(image_tensor, output_path)

            self.status_label.config(text=f"转换成功！普通图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"普通图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理隐写图像时出错:\n{str(e)}")