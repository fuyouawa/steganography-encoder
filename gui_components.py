"""
GUI组件模块
包含拖拽框等GUI组件类
"""
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES


class DragDropBox:
    """拖拽文件框组件"""

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
        """处理文件拖拽事件"""
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

        if file_path and self._file_exists(file_path):
            self.drop_callback(file_path)

    def on_click(self, _):
        """点击事件处理 - 打开文件对话框"""
        file_path = filedialog.askopenfilename()
        if file_path:
            self.drop_callback(file_path)

    def _file_exists(self, file_path):
        """检查文件是否存在"""
        import os
        return os.path.exists(file_path)