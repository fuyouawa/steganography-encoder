"""
配置管理模块
管理应用程序的配置选项
"""
import tkinter as tk


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        # Configuration variables
        self.compression_level = tk.IntVar(value=-1)
        self.steganography_width = tk.IntVar(value=0)
        self.steganography_height = tk.IntVar(value=0)
        self.use_alpha = tk.BooleanVar(value=False)
        self.top_margin_ratio = tk.DoubleVar(value=20.0)  # 20% top margin (stored as percentage)
        self.bottom_margin_ratio = tk.DoubleVar(value=20.0)  # 20% bottom margin (stored as percentage)

    def create_config_frame(self, parent):
        """创建配置选项框架"""
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

        # Steganography image dimensions
        tk.Label(config_frame, text="隐写图像尺寸:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        dimensions_frame = tk.Frame(config_frame)
        dimensions_frame.grid(row=1, column=1, sticky="w", pady=(10, 0))

        tk.Label(dimensions_frame, text="宽:").pack(side="left")
        width_entry = tk.Entry(dimensions_frame, textvariable=self.steganography_width, width=6)
        width_entry.pack(side="left", padx=(2, 10))

        tk.Label(dimensions_frame, text="高:").pack(side="left")
        height_entry = tk.Entry(dimensions_frame, textvariable=self.steganography_height, width=6)
        height_entry.pack(side="left", padx=(2, 0))

        tk.Label(dimensions_frame, text="(0=不指定)", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        # Alpha channel option
        tk.Label(config_frame, text="使用alpha通道:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        alpha_check = tk.Checkbutton(config_frame, variable=self.use_alpha)
        alpha_check.grid(row=2, column=1, sticky="w", pady=(10, 0))

        # Margin ratios
        tk.Label(config_frame, text="上预留区域:").grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        top_margin_frame = tk.Frame(config_frame)
        top_margin_frame.grid(row=3, column=1, sticky="w", pady=(10, 0))

        top_margin_scale = tk.Scale(top_margin_frame, from_=0, to=100, orient="horizontal",
                                   variable=self.top_margin_ratio, showvalue=True, length=150, resolution=1)
        top_margin_scale.pack(side="left")
        tk.Label(top_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        tk.Label(config_frame, text="下预留区域:").grid(row=4, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        bottom_margin_frame = tk.Frame(config_frame)
        bottom_margin_frame.grid(row=4, column=1, sticky="w", pady=(10, 0))

        bottom_margin_scale = tk.Scale(bottom_margin_frame, from_=0, to=100, orient="horizontal",
                                      variable=self.bottom_margin_ratio, showvalue=True, length=150, resolution=1)
        bottom_margin_scale.pack(side="left")
        tk.Label(bottom_margin_frame, text="%", font=("Arial", 8)).pack(side="left", padx=(5, 0))

        return config_frame

    def get_compression_level(self):
        """获取压缩等级"""
        return self.compression_level.get()

    def get_noise_width(self):
        """获取隐写图像宽度"""
        return self.steganography_width.get()

    def get_noise_height(self):
        """获取隐写图像高度"""
        return self.steganography_height.get()

    def get_use_alpha(self):
        """获取是否使用alpha通道"""
        return self.use_alpha.get()

    def get_top_margin_ratio(self):
        """获取上预留区域百分比（返回小数）"""
        return self.top_margin_ratio.get() / 100.0

    def get_bottom_margin_ratio(self):
        """获取下预留区域百分比（返回小数）"""
        return self.bottom_margin_ratio.get() / 100.0