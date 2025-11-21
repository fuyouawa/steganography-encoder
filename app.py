"""
主应用类模块
包含应用程序的主要逻辑和界面
"""
import tkinter as tk
from tkinter import messagebox

from gui_components import DragDropBox
from config_manager import ConfigManager
from image_processor import ImageProcessor


class NoiseImageEncoder:
    """噪点图像编码器主应用类"""

    def __init__(self, root):
        self.root = root
        self.root.title("噪点图像编码器")
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

        self.left_box = DragDropBox(left_frame, "拖拽普通图像文件到这里\n(转换为噪点图像)", self.process_image_to_noise)
        self.left_box.frame.pack(fill="both", expand=True)

        # Right box for noise images -> regular images
        right_frame = tk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.right_box = DragDropBox(right_frame, "拖拽噪点图像文件到这里\n(转换为普通图像)", self.process_noise_to_image)
        self.right_box.frame.pack(fill="both", expand=True)

    def process_image_to_noise(self, file_path):
        """处理普通图像转换为噪点图像"""
        try:
            self.status_label.config(text="正在处理图像...")

            # Get configuration
            compression_level = self.config_manager.get_compression_level()
            noise_width = self.config_manager.get_noise_width()
            noise_height = self.config_manager.get_noise_height()
            use_alpha = self.config_manager.get_use_alpha()

            # Process image
            noise_image = self.image_processor.process_image_to_noise(
                file_path, compression_level, noise_width, noise_height, use_alpha
            )

            # Save noise image
            output_path = self.image_processor.get_output_path(file_path, "_noise")
            self.image_processor.save_noise_image(noise_image, output_path)

            self.status_label.config(text=f"转换成功！噪点图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"噪点图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理图像时出错:\n{str(e)}")

    def process_noise_to_image(self, file_path):
        """处理噪点图像转换为普通图像"""
        try:
            self.status_label.config(text="正在处理噪点图像...")

            # Process noise image
            image_tensor = self.image_processor.process_noise_to_image(file_path)

            # Save regular image
            output_path = self.image_processor.get_output_path(file_path, "_decoded")
            self.image_processor.save_regular_image(image_tensor, output_path)

            self.status_label.config(text=f"转换成功！普通图像已保存到: {output_path}")
            messagebox.showinfo("成功", f"普通图像已保存到:\n{output_path}")

        except Exception as e:
            self.status_label.config(text="处理失败")
            messagebox.showerror("错误", f"处理噪点图像时出错:\n{str(e)}")