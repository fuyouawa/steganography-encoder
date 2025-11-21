"""
噪点图像编码器 - 应用程序入口点
"""
from tkinterdnd2 import TkinterDnD
from app import NoiseImageEncoder


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = NoiseImageEncoder(root)
    root.mainloop()
