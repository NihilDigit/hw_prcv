"""统一绘图配置：HarmonyOS Sans + SciencePlots"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 注册 HarmonyOS Sans 字体
FONT_PATH = PROJECT_ROOT / "HarmonyOS_Sans_Regular.ttf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = "HarmonyOS Sans"
else:
    FONT_NAME = "sans-serif"


def setup_style(style: str = "science", usetex: bool = False):
    """
    设置绘图样式

    Args:
        style: SciencePlots 样式，可选 "ieee", "science", "nature" 等
        usetex: 是否使用 LaTeX 渲染（需要系统安装 LaTeX）
    """
    import scienceplots  # noqa: F401

    plt.style.use(["science", style, "no-latex"] if not usetex else ["science", style])

    # 覆盖字体设置
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME, "DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,  # 解决负号显示问题
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def savefig(fig, name: str, formats: list[str] = ["png"], output_dir: Path | None = None):
    """
    保存图片到多种格式

    Args:
        fig: matplotlib figure 对象
        name: 文件名（不含扩展名）
        formats: 输出格式列表
        output_dir: 输出目录，默认为 PROJECT_ROOT/figures
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(output_dir / f"{name}.{fmt}")


# 自动设置默认样式
setup_style()
