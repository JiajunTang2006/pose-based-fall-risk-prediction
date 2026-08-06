"""
从预测结果 CSV 文件中绘制特征曲线图。

这个工具脚本用于可视化分析——你运行跌倒预测后得到了一个 CSV 文件，
这个脚本可以读取 CSV 并画出各特征随时间变化的曲线，方便分析和调试。

使用方式：
    # 默认用法（读取 outputs/predictions.csv，输出 outputs/feature_curves.png）
    python -m fall_prediction.plot_features outputs/predictions.csv

    # 指定输出路径
    python -m fall_prediction.plot_features outputs/predictions.csv --output my_chart.png

生成的图表包含 4 个子图（从上到下）：
    1. smoothed_risk_score → 平滑风险分数曲线
    2. torso_angle          → 躯干倾斜角度曲线
    3. vertical_velocity    → 垂直速度曲线
    4. aspect_ratio         → 宽高比曲线
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_csv(csv_path: str | Path, output_path: str | Path) -> None:
    """
    读取 CSV 文件并绘制特征曲线图。

    参数:
        csv_path:    跌倒预测输出的 CSV 文件路径
        output_path: 输出图片路径（PNG 格式）
    """
    import matplotlib

    # 使用非交互式后端，只负责把图保存成 PNG。
    # 这样即使在终端、远程环境或没有显示窗口的环境里也能正常画图。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 读取 CSV 文件 ----
    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    # ---- 提取数据 ----
    # 横轴：时间（秒）
    time = [float(row["time"]) for row in rows]
    # 纵轴：4 个关键特征
    series = {
        "risk_score": [float(row["smoothed_risk_score"]) for row in rows],
        "torso_angle": [float(row["torso_angle"]) for row in rows],
        "vertical_velocity": [float(row["vertical_velocity"]) for row in rows],
        "aspect_ratio": [float(row["aspect_ratio"]) for row in rows],
    }

    # ---- 创建图表 ----
    # 4 行 1 列的子图布局，共享 X 轴，图片大小 10×8 英寸
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 8), sharex=True)
    for axis, (name, values) in zip(axes, series.items(), strict=True):
        axis.plot(time, values, linewidth=1.8)  # 画曲线
        axis.set_ylabel(name)                    # Y 轴标签
        axis.grid(True, alpha=0.25)              # 添加浅色网格线
    axes[-1].set_xlabel("time (s)")  # 最底部的子图：X 轴标签
    fig.tight_layout()               # 自动调整子图间距

    # ---- 保存图片 ----
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)  # 自动创建目录
    fig.savefig(output, dpi=160)  # dpi=160 保证清晰度
    plt.close(fig)                # 关闭图形，释放内存
    print(f"Chart saved to {output}")


def main() -> None:
    """
    命令行入口。

    用法:
        python -m fall_prediction.plot_features <csv_path> [--output <output_path>]
    """
    parser = argparse.ArgumentParser(description="Plot feature curves from fall prediction CSV output.")
    parser.add_argument("csv_path")                             # 必选参数：CSV 文件路径
    parser.add_argument("--output", default="outputs/feature_curves.png")  # 可选：输出图片路径
    args = parser.parse_args()
    plot_csv(args.csv_path, args.output)


if __name__ == "__main__":
    main()
