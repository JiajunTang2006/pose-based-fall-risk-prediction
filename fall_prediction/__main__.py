"""
程序的入口文件。

当你执行 `python -m fall_prediction` 时，Python 会自动运行这个文件。
它会调用 video_app 模块中的 main() 函数，启动整个跌倒预测程序。
"""

from .video_app import main


if __name__ == "__main__":
    main()
