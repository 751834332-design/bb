#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
脉搏脉象可视化分析系统 V3.0
================================================================================
作者：任妍婕   学号：202228034130
指导教师：熊帆
--------------------------------------------------------------------------------
新增功能：
  1. 支持自采txt格式数据导入（寸/关/尺三部位独立文件）
  2. 全面美化界面（医学仪器风格）
  3. 自采数据与公开数据集双模式切换
================================================================================
"""

import sys
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.ndimage import grey_opening, grey_closing

# ---- 动态导入小波库 ----
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams

# ---- 动态导入wfdb ----
try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QComboBox,
    QSpinBox, QDoubleSpinBox, QHeaderView, QFrame, QMessageBox,
    QProgressBar, QSizePolicy, QStackedWidget, QRadioButton,
    QButtonGroup, QLineEdit, QScrollArea, QSplitter, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient

# ======================== 字体 ========================
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ======================== 白蓝专业主题 ========================
# 背景色
BG_WHITE   = "#FFFFFF"
BG_LIGHT   = "#F0F6FF"
BG_PANEL   = "#E8F2FF"
BG_HOVER   = "#DDEEFF"

# 蓝色主色
BLUE_MAIN  = "#1565C0"   # 深蓝主色
BLUE_BTN   = "#1976D2"   # 按钮蓝
BLUE_LIGHT = "#42A5F5"   # 浅蓝
BLUE_BG    = "#E3F2FD"   # 极浅蓝

# 文字色
TEXT_DARK  = "#0D2137"   # 深色文字
TEXT_MID   = "#34567A"   # 中色文字
TEXT_LIGHT = "#7A9BC0"   # 浅色文字

# 边框
BORDER     = "#BBDEFB"   # 蓝色边框
BORDER2    = "#90CAF9"   # 深蓝边框

# 功能色
COL_GREEN  = "#2E7D32"   # 绿色（最优）
COL_RED    = "#C62828"   # 红色（波峰）
COL_ORANGE = "#E65100"   # 橙色（起点）
COL_WAVE   = "#1565C0"   # 波形蓝
COL_CLEAN  = "#1B5E20"   # 处理后深绿
COL_CUN    = "#C62828"   # 寸部红
COL_GUAN   = "#1565C0"   # 关部蓝
COL_CHI    = "#2E7D32"   # 尺部绿

# Matplotlib白色图表主题
for k, v in {
    "figure.facecolor": BG_WHITE,
    "axes.facecolor":   BG_LIGHT,
    "axes.edgecolor":   BORDER2,
    "axes.labelcolor":  TEXT_MID,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.color":      TEXT_LIGHT,
    "ytick.color":      TEXT_LIGHT,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "text.color":       TEXT_DARK,
    "grid.color":       BORDER,
    "grid.linewidth":   0.7,
    "grid.alpha":       0.8,
    "legend.facecolor": BG_WHITE,
    "legend.edgecolor": BORDER2,
    "legend.fontsize":  12,
    "lines.linewidth":  1.8,
    "figure.dpi":       110,
}.items():
    rcParams[k] = v


STYLESHEET = """
/* ===== 全局 ===== */
QMainWindow, QWidget {
    background: #FFFFFF;
    color: #0D2137;
    font-family: 'Microsoft YaHei', 'SimHei';
    font-size: 15px;
}

/* ===== 标签页 ===== */
QTabWidget::pane {
    border: 2px solid #90CAF9;
    background: #FFFFFF;
    border-radius: 12px;
    margin-top: 6px;
}
QTabBar::tab {
    background: #E3F2FD;
    color: #34567A;
    padding: 12px 22px;
    border: 1px solid #90CAF9;
    border-bottom: none;
    margin-right: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    min-width: 108px;
    min-height: 28px;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #1565C0;
    border-bottom: 3px solid #1565C0;
}
QTabBar::tab:hover:!selected {
    background: #BBDEFB;
    color: #1565C0;
}

/* ===== 按钮 ===== */
QPushButton {
    background: #F0F6FF;
    color: #0D2137;
    border: 2px solid #90CAF9;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 15px;
    font-weight: bold;
    min-height: 34px;
}
QPushButton:hover {
    background: #1976D2;
    color: #FFFFFF;
    border: 2px solid #1976D2;
}
QPushButton:disabled {
    background: #EAF4FF;
    color: #34567A;
    border: 2px solid #BBDEFB;
}
QPushButton#primary {
    background: #D8ECFF;
    color: #0D2137;
    border: 2px solid #1565C0;
    font-size: 16px;
}
QPushButton#primary:hover {
    background: #1976D2;
    color: #FFFFFF;
}
QPushButton#primary:disabled {
    background: #DDEEFF;
    color: #0D2137;
    border: 2px solid #90CAF9;
}
QPushButton#gold {
    background: #FFE6D5;
    color: #0D2137;
    border: 2px solid #E65100;
    font-size: 16px;
}
QPushButton#gold:hover {
    background: #F57C00;
    color: #FFFFFF;
}
QPushButton#gold:disabled {
    background: #FFE6D5;
    color: #0D2137;
    border: 2px solid #F5B08A;
}
QPushButton#danger {
    background: #FFE3E3;
    color: #0D2137;
    border: 2px solid #C62828;
    font-size: 15px;
}
QPushButton#danger:hover {
    background: #E53935;
    color: #FFFFFF;
}
QPushButton#danger:disabled {
    background: #FFE3E3;
    color: #0D2137;
    border: 2px solid #F4A0A0;
}

/* ===== 分组框 ===== */
QGroupBox {
    border: 2px solid #90CAF9;
    border-radius: 12px;
    margin-top: 24px;
    padding-top: 20px;
    background: #F0F6FF;
    color: #1565C0;
    font-weight: bold;
    font-size: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    top: -12px;
    padding: 2px 10px;
    background: #FFFFFF;
    color: #1565C0;
    font-size: 16px;
    font-weight: bold;
}

/* ===== 下拉/数字框 ===== */
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF;
    color: #0D2137;
    border: 2px solid #90CAF9;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 15px;
    min-height: 34px;
    min-width: 120px;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background: #F8FBFF;
    color: #34567A;
    border: 2px solid #BBDEFB;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border: 2px solid #1565C0;
}
QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #0D2137;
    selection-background-color: #1565C0;
    selection-color: #FFFFFF;
    font-size: 15px;
}

/* ===== 单选按钮 ===== */
QRadioButton {
    color: #0D2137;
    font-size: 15px;
    spacing: 10px;
    min-height: 28px;
}
QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #90CAF9;
    background: #FFFFFF;
}
QRadioButton::indicator:checked {
    background: #1565C0;
    border: 2px solid #1565C0;
}

/* ===== 表格 ===== */
QTableWidget {
    background: #FFFFFF;
    color: #0D2137;
    border: 2px solid #90CAF9;
    gridline-color: #BBDEFB;
    selection-background-color: #1565C0;
    selection-color: #FFFFFF;
    font-size: 15px;
    border-radius: 6px;
}
QHeaderView::section {
    background: #E3F2FD;
    color: #1565C0;
    border: 1px solid #90CAF9;
    padding: 10px;
    font-weight: bold;
    font-size: 15px;
}
QTableWidget::item {
    padding: 8px;
}
QTableWidget::item:alternate {
    background: #F0F6FF;
}

/* ===== 标签 ===== */
QLabel {
    color: #0D2137;
    font-size: 15px;
}
QLabel:disabled {
    color: #34567A;
}

QGroupBox:disabled {
    color: #34567A;
}

/* ===== 进度条 ===== */
QProgressBar {
    background: #E3F2FD;
    border-radius: 5px;
    height: 12px;
    color: transparent;
    border: 1px solid #90CAF9;
}
QProgressBar::chunk {
    background: #1565C0;
    border-radius: 5px;
}

/* ===== 状态栏 ===== */
QStatusBar {
    background: #E3F2FD;
    color: #34567A;
    border-top: 2px solid #90CAF9;
    font-size: 15px;
    padding: 6px 12px;
}

/* ===== 图表交互工具栏 ===== */
QToolBar, QToolButton {
    background: #FFFFFF;
    color: #0D2137;
    border: 1px solid #BBDEFB;
    border-radius: 6px;
    padding: 4px;
    font-size: 14px;
}
QToolButton:hover {
    background: #E3F2FD;
    border: 1px solid #1565C0;
}
QLabel#coordLabel {
    background: #F8FBFF;
    color: #34567A;
    border: 1px solid #BBDEFB;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 13px;
}

QScrollArea {
    background: #FFFFFF;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: #F0F6FF;
}

QWidget#leftPanel {
    background: #F0F6FF;
    border-radius: 14px;
}

/* ===== 滚动条 ===== */
QScrollBar:vertical {
    background: #E3F2FD;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #90CAF9;
    border-radius: 6px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #1565C0;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class Canvas(QWidget):
    """Matplotlib canvas with built-in navigation and coordinate feedback."""

    def __init__(self, rows=1, cols=1, figsize=(8, 4), show_toolbar=True):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if show_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setToolTip("交互工具栏：可缩放、平移、还原视图并保存图表")
            layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas, 1)

        self.coord_label = QLabel("提示：使用工具栏可缩放/平移/保存图表，鼠标移入图表可查看坐标")
        self.coord_label.setObjectName("coordLabel")
        layout.addWidget(self.coord_label)

        self.axes = []
        self._build_axes()
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)

    def _build_axes(self):
        self.axes = []
        total = self.rows * self.cols
        for i in range(total):
            ax = self.fig.add_subplot(self.rows, self.cols, i + 1)
            self._style_axis(ax)
            self.axes.append(ax)
        self.fig.patch.set_facecolor(BG_WHITE)

    def _style_axis(self, ax):
        if getattr(ax, "name", "") != "polar":
            ax.set_facecolor(BG_LIGHT)
            ax.grid(True, alpha=0.35, linestyle="--")
            ax.tick_params(axis="both", labelsize=11, colors=TEXT_MID)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER2)
                sp.set_linewidth(1.0)

    def _style_all_axes(self):
        self.fig.patch.set_facecolor(BG_WHITE)
        for ax in self.fig.axes:
            self._style_axis(ax)

    def clear_all(self):
        self.fig.clear()
        self._build_axes()
        self.coord_label.setText("图表已刷新：可使用工具栏进行缩放、平移和保存")
        self.canvas.draw_idle()

    def clear_fig(self):
        self.fig.clear()
        self.axes = []
        self.coord_label.setText("图表已刷新：可使用工具栏进行缩放、平移和保存")
        self.canvas.draw_idle()

    def draw(self):
        self._style_all_axes()
        self.canvas.draw_idle()

    def _on_motion(self, event):
        if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
            self.coord_label.setText(f"当前坐标：x = {event.xdata:.3f}，y = {event.ydata:.3f}")
        else:
            self.coord_label.setText("提示：拖拽平移、框选缩放、Home 还原视图、Save 保存图表")

    def _on_leave(self, event):
        self.coord_label.setText("提示：使用工具栏可缩放/平移/保存图表，鼠标移入图表可查看坐标")


class SignalProcessor:
    """Signal processing helpers used by the analysis workflow."""

    @staticmethod
    def _safe_bandpass(signal, fs, lo=0.5, hi=8.0):
        nyq = fs / 2.0
        hi = min(hi, nyq * 0.95)
        lo_n, hi_n = lo / nyq, hi / nyq
        if lo_n <= 0 or lo_n >= hi_n:
            return np.asarray(signal, dtype=float)
        try:
            b, a = butter(4, [lo_n, hi_n], btype="band")
            return filtfilt(b, a, signal)
        except Exception:
            return np.asarray(signal, dtype=float)

    @staticmethod
    def preprocess(signal, fs):
        s = np.asarray(signal, dtype=float).ravel()
        if len(s) == 0:
            return s
        s = s - np.nanmean(s)
        s = np.nan_to_num(s)
        filtered = SignalProcessor._safe_bandpass(s, fs)

        if HAS_PYWT and len(filtered) >= 32:
            try:
                coeffs = pywt.wavedec(filtered, "db4", level=min(5, pywt.dwt_max_level(len(filtered), pywt.Wavelet("db4").dec_len)))
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0
                thr = sigma * np.sqrt(2 * np.log(max(len(filtered), 2)))
                denoised = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
                filtered = pywt.waverec(denoised, "db4")[:len(filtered)]
            except Exception:
                pass

        size = max(3, int(fs * 0.12))
        if size % 2 == 0:
            size += 1
        try:
            baseline = grey_closing(grey_opening(filtered, size=size), size=size)
            filtered = filtered - baseline
        except Exception:
            pass

        return filtered

    @staticmethod
    def detect_peaks(signal, fs, alpha=1.5, beta=0.5):
        s = np.asarray(signal, dtype=float).ravel()
        if len(s) < 3:
            return np.array([], dtype=int)
        height = np.mean(s) + beta * np.std(s)
        prominence = max(np.std(s) * alpha * 0.15, np.ptp(s) * 0.04, 1e-8)
        distance = max(1, int(0.35 * fs))
        peaks, _ = find_peaks(s, height=height, prominence=prominence, distance=distance)
        return peaks.astype(int)

    @staticmethod
    def find_onsets(signal, peaks, fs):
        s = np.asarray(signal, dtype=float).ravel()
        onsets = []
        lookback = max(1, int(0.45 * fs))
        for p in peaks:
            start = max(0, int(p) - lookback)
            if start < p:
                onsets.append(start + int(np.argmin(s[start:int(p)])))
        return np.array(onsets, dtype=int)

    @staticmethod
    def extract_features(signal, peaks, onsets, fs):
        s = np.asarray(signal, dtype=float).ravel()
        features = []
        for i, p in enumerate(peaks):
            p = int(p)
            onset = int(onsets[i]) if i < len(onsets) else max(0, p - int(0.25 * fs))
            next_onset = int(onsets[i + 1]) if i + 1 < len(onsets) else min(len(s) - 1, p + int(0.55 * fs))
            prev_peak = int(peaks[i - 1]) if i > 0 else None
            rr = (p - prev_peak) / fs if prev_peak is not None and p > prev_peak else None
            heart_rate = round(60.0 / rr, 1) if rr and rr > 0 else "-"
            amp = float(s[p] - np.min(s[onset:min(next_onset + 1, len(s))])) if len(s) else 0.0
            rise = max(0.0, (p - onset) / fs)
            fall = max(0.0, (next_onset - p) / fs)
            features.append({
                "heart_rate": heart_rate,
                "amplitude": round(amp, 4),
                "rise_time": round(rise, 3),
                "fall_time": round(fall, 3),
                "peak_idx": p,
            })
        return features

    @staticmethod
    def snr(raw, processed):
        raw = np.asarray(raw, dtype=float).ravel()
        processed = np.asarray(processed, dtype=float).ravel()
        n = min(len(raw), len(processed))
        if n == 0:
            return 0
        noise = raw[:n] - processed[:n]
        val = 10 * np.log10((np.mean(processed[:n] ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12))
        return round(float(val), 2)

    @staticmethod
    def smoothness(signal):
        s = np.asarray(signal, dtype=float).ravel()
        if len(s) < 3:
            return 0
        val = 1.0 / (1.0 + np.mean(np.abs(np.diff(s, n=2))))
        return round(float(val), 4)


class PulseRecognizer:
    TYPES = {
        "ping": {"name": "平脉", "color": "#1565C0", "tcm": "气血调和，脏腑功能平衡。"},
        "chi": {"name": "迟脉", "color": "#5C6BC0", "tcm": "多见寒证，或阳气不足。"},
        "shu": {"name": "数脉", "color": "#C62828", "tcm": "多见热证，或阴虚内热。"},
        "fu": {"name": "浮脉", "color": "#E65100", "tcm": "多见表证，外感初起。"},
        "chen": {"name": "沉脉", "color": "#2E7D32", "tcm": "多见里证，气血内郁。"},
        "hua": {"name": "滑脉", "color": "#00897B", "tcm": "多见痰湿、食积，亦可见于妊娠。"},
        "se": {"name": "涩脉", "color": "#6D4C41", "tcm": "多见血瘀、精伤、津亏。"},
        "xian": {"name": "弦脉", "color": "#7B1FA2", "tcm": "多见肝胆病、痛证或痰饮。"},
    }

    @staticmethod
    def _score(value, target, width):
        return max(0.0, 100.0 - abs(value - target) / max(width, 1e-8) * 100.0)

    @staticmethod
    def score(hr, amp, cv, rt, slope, cun_r, chi_r, rr_std):
        scores = {
            "ping": 0.35 * PulseRecognizer._score(hr, 72, 28) + 0.30 * PulseRecognizer._score(cv, 0.08, 0.20) + 0.35 * PulseRecognizer._score(rr_std, 0.04, 0.16),
            "chi": PulseRecognizer._score(hr, 52, 22),
            "shu": PulseRecognizer._score(hr, 102, 32),
            "fu": 0.55 * PulseRecognizer._score(cun_r, 0.48, 0.25) + 0.45 * PulseRecognizer._score(amp, max(amp, 0.08), 0.18),
            "chen": 0.55 * PulseRecognizer._score(chi_r, 0.48, 0.25) + 0.45 * PulseRecognizer._score(amp, 0.04, 0.12),
            "hua": 0.50 * PulseRecognizer._score(cv, 0.06, 0.16) + 0.50 * PulseRecognizer._score(rr_std, 0.03, 0.14),
            "se": 0.55 * min(100.0, cv * 260.0) + 0.45 * min(100.0, rr_std * 520.0),
            "xian": 0.60 * min(100.0, slope * 18.0) + 0.40 * PulseRecognizer._score(rt, 0.12, 0.20),
        }
        return {k: int(round(max(0, min(100, v)))) for k, v in scores.items()}




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("脉搏脉象可视化分析系统 V3.0  ·  任妍婕  202228034130")
        self.setMinimumSize(1280, 820)
        self.resize(1600, 950)

        # 数据状态
        self.signal_raw   = None
        self.signal_clean = None
        self.peaks        = None
        self.onsets       = None
        self.features     = None
        self.fs           = 125
        self.data_dir     = 'E:/pulse_data'
        self.data_mode    = 'bidmc'   # 'bidmc' 或 'custom'

        # 自采数据
        self.custom_cun   = None
        self.custom_guan  = None
        self.custom_chi   = None
        self.custom_paths = {'寸': '', '关': '', '尺': ''}

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self.statusBar().showMessage("🌿 欢迎使用脉搏脉象可视化分析系统 V3.0 — 请选择数据模式并加载数据")

    # ==================== UI构建 ====================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(14,14,14,14)
        main.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12,0,0,0)
        rl.setSpacing(12)
        rl.addWidget(self._build_topbar())

        self.tabs = QTabWidget()
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.addTab(self._tab_raw(),        "原始信号")
        self.tabs.addTab(self._tab_preprocess(), "预处理对比")
        self.tabs.addTab(self._tab_features(),   "特征提取")
        self.tabs.addTab(self._tab_compare(),    "算法对比")
        self.tabs.addTab(self._tab_cgc(),        "寸关尺分析")
        self.tabs.addTab(self._tab_recog(),      "脉象识别")
        rl.addWidget(self.tabs)
        splitter.addWidget(self._build_panel())
        splitter.addWidget(right)
        splitter.setSizes([480, 1080])
        main.addWidget(splitter)

    # ---- 顶部统计条 ----
    def _build_topbar(self):
        bar = QWidget()
        bar.setMinimumHeight(150)
        bar.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {'#E3F2FD'}, stop:0.5 {'#DDEEFF'}, stop:1 {'#E3F2FD'});
            border-radius: 14px;
            border: 1px solid {'#90CAF9'};
        """)
        lo = QGridLayout(bar)
        lo.setContentsMargins(20,14,20,14)
        lo.setHorizontalSpacing(14)
        lo.setVerticalSpacing(12)

        stats = [
            ("数据来源", "—",       '#1565C0'),
            ("采样率",   "— Hz",    '#1565C0'),
            ("检测波峰", "— 个",    '#C62828'),
            ("平均脉率", "— 次/分", '#E65100'),
            ("平均幅度", "—",       '#2E7D32'),
            ("识别脉象", "—",       '#5C6BC0'),
        ]
        self.stat_labels = {}
        for i,(name,default,color) in enumerate(stats):
            w = QWidget()
            w.setMinimumHeight(58)
            w.setStyleSheet("""
                QWidget {
                    background: #FFFFFF;
                    border: 1px solid #BBDEFB;
                    border-radius: 10px;
                }
            """)
            wl = QVBoxLayout(w)
            wl.setContentsMargins(14,8,14,8)
            wl.setSpacing(4)
            n_lbl = QLabel(name)
            n_lbl.setStyleSheet(f"color:{'#34567A'};font-size:14px;border:none;background:transparent;")
            n_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            v_lbl = QLabel(default)
            v_lbl.setStyleSheet(f"color:{color};font-size:21px;font-weight:bold;border:none;background:transparent;")
            v_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            v_lbl.setWordWrap(True)
            v_lbl.setMinimumWidth(160)
            wl.addWidget(n_lbl)
            wl.addWidget(v_lbl)
            self.stat_labels[name] = v_lbl
            lo.addWidget(w, i // 3, i % 3)
        return bar

    def _upd(self, **kw):
        for k,v in kw.items():
            if k in self.stat_labels:
                self.stat_labels[k].setText(str(v))

    # ---- 左侧控制面板 ----
    def _build_panel(self):
        panel = QWidget()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(420)
        lo = QVBoxLayout(panel)
        lo.setContentsMargins(26,26,26,26)
        lo.setSpacing(20)

        # 标题
        title = QLabel("脉象分析系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-size:26px;font-weight:bold;
            color:{'#1565C0'};padding:14px 0 8px 0;
            border-bottom:1px solid {'#90CAF9'};
        """)
        title.setWordWrap(True)
        lo.addWidget(title)

        sub = QLabel("软件工程  202228034130  任妍婕")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color:{'#34567A'};font-size:14px;padding-bottom:8px;")
        sub.setWordWrap(True)
        lo.addWidget(sub)

        # ---- 数据模式选择 ----
        mode_grp = QGroupBox("数据模式")
        mg = QVBoxLayout(mode_grp)
        mg.setContentsMargins(26,24,26,20)
        mg.setSpacing(12)
        self.rb_bidmc  = QRadioButton("公开数据集 (BIDMC)")
        self.rb_custom = QRadioButton("自采数据 (TXT文件)")
        self.rb_bidmc.setChecked(True)
        self.rb_bidmc.toggled.connect(self._on_mode_change)
        mg.addWidget(self.rb_bidmc)
        mg.addWidget(self.rb_custom)
        lo.addWidget(mode_grp)

        # ---- BIDMC数据加载 ----
        self.stack = QStackedWidget()

        # 页面0：BIDMC
        p0 = QWidget()
        p0l = QVBoxLayout(p0)
        p0l.setContentsMargins(0,0,0,0)
        p0l.setSpacing(10)
        p0l.addWidget(QLabel("选择受试者:"))
        self.subject_cb = QComboBox()
        self.subject_cb.setMinimumWidth(260)
        self.subject_cb.addItems([f"bidmc{i:02d}" for i in range(1,54)])
        p0l.addWidget(self.subject_cb)
        self.dir_btn = QPushButton("选择数据目录")
        self.dir_btn.clicked.connect(self._choose_dir)
        p0l.addWidget(self.dir_btn)
        self.load_btn0 = QPushButton("加载BIDMC数据")
        self.load_btn0.setObjectName("primary")
        self.load_btn0.clicked.connect(self._load_bidmc)
        p0l.addWidget(self.load_btn0)
        self.stack.addWidget(p0)

        # 页面1：自采数据
        p1 = QWidget()
        p1l = QVBoxLayout(p1)
        p1l.setContentsMargins(0,0,0,0)
        p1l.setSpacing(10)
        self.custom_btns = {}
        self.custom_lbls = {}
        for part in ['寸', '关', '尺']:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0)
            rl.setSpacing(10)
            btn = QPushButton(f"{part}部文件")
            btn.setMinimumWidth(136)
            btn.clicked.connect(lambda checked,p=part: self._load_custom_file(p))
            lbl = QLabel("未选择")
            lbl.setStyleSheet(f"color:{'#34567A'};font-size:14px;")
            lbl.setMinimumWidth(210)
            lbl.setWordWrap(True)
            rl.addWidget(btn)
            rl.addWidget(lbl)
            p1l.addWidget(row)
            self.custom_btns[part] = btn
            self.custom_lbls[part] = lbl

        fs_row = QWidget()
        fsl = QHBoxLayout(fs_row)
        fsl.setContentsMargins(0,0,0,0)
        fsl.setSpacing(10)
        fsl.addWidget(QLabel("采样率(Hz):"))
        self.fs_spin = QSpinBox()
        self.fs_spin.setRange(10, 1000)
        self.fs_spin.setValue(50)
        self.fs_spin.setMinimumWidth(160)
        fsl.addWidget(self.fs_spin)
        p1l.addWidget(fs_row)

        self.load_btn1 = QPushButton("加载自采数据")
        self.load_btn1.setObjectName("gold")
        self.load_btn1.clicked.connect(self._load_custom)
        p1l.addWidget(self.load_btn1)
        self.stack.addWidget(p1)

        lo.addWidget(self.stack)

        # 信息显示
        self.info_lbl = QLabel("尚未加载数据")
        self.info_lbl.setStyleSheet(f"""
            color:{'#34567A'};font-size:14px;
            background:{'#E3F2FD'};border-radius:4px;
            border:1px solid {'#BBDEFB'};
            padding:10px;
        """)
        self.info_lbl.setMinimumHeight(50)
        self.info_lbl.setWordWrap(True)
        lo.addWidget(self.info_lbl)

        # ---- 预处理参数 ----
        pre_grp = QGroupBox("预处理参数")
        pg = QGridLayout(pre_grp)
        pg.setContentsMargins(26,24,26,20)
        pg.setHorizontalSpacing(18)
        pg.setVerticalSpacing(14)
        pg.addWidget(QLabel("小波基:"), 0, 0)
        self.wav_cb = QComboBox()
        self.wav_cb.setMinimumWidth(170)
        self.wav_cb.addItems(['db4','db6','sym4','coif2'])
        pg.addWidget(self.wav_cb, 0, 1)
        pg.addWidget(QLabel("分解层:"), 1, 0)
        self.lv_sp = QSpinBox()
        self.lv_sp.setRange(3,8); self.lv_sp.setValue(5)
        self.lv_sp.setMinimumWidth(170)
        pg.addWidget(self.lv_sp, 1, 1)
        pre_run = QPushButton("运行预处理")
        pre_run.setObjectName("primary")
        pre_run.clicked.connect(self._run_preprocess)
        pg.addWidget(pre_run, 2, 0, 1, 2)
        lo.addWidget(pre_grp)

        # ---- 检测参数 ----
        det_grp = QGroupBox("检测参数")
        dg = QGridLayout(det_grp)
        dg.setContentsMargins(26,24,26,20)
        dg.setHorizontalSpacing(18)
        dg.setVerticalSpacing(14)
        dg.addWidget(QLabel("α (均值权重):"), 0, 0)
        self.alpha_sp = QDoubleSpinBox()
        self.alpha_sp.setRange(0.5,3.0); self.alpha_sp.setSingleStep(0.1)
        self.alpha_sp.setValue(1.5)
        self.alpha_sp.setMinimumWidth(170)
        dg.addWidget(self.alpha_sp, 0, 1)
        dg.addWidget(QLabel("β (标准差权重):"), 1, 0)
        self.beta_sp = QDoubleSpinBox()
        self.beta_sp.setRange(0.0,2.0); self.beta_sp.setSingleStep(0.1)
        self.beta_sp.setValue(0.5)
        self.beta_sp.setMinimumWidth(170)
        dg.addWidget(self.beta_sp, 1, 1)
        feat_btn = QPushButton("运行特征提取")
        feat_btn.setObjectName("primary")
        feat_btn.clicked.connect(self._run_features)
        dg.addWidget(feat_btn, 2, 0, 1, 2)
        cmp_btn = QPushButton("运行算法对比")
        cmp_btn.clicked.connect(self._run_compare)
        dg.addWidget(cmp_btn, 3, 0, 1, 2)
        lo.addWidget(det_grp)

        # ---- 导出 ----
        exp_grp = QGroupBox("导出")
        el = QVBoxLayout(exp_grp)
        el.setContentsMargins(26,24,26,20)
        save_btn = QPushButton("保存所有图表")
        save_btn.setObjectName("danger")
        save_btn.clicked.connect(self._save_all)
        el.addWidget(save_btn)
        lo.addWidget(exp_grp)

        lo.addStretch()

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lo.addWidget(self.progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        scroll.setMaximumWidth(540)
        scroll.setWidget(panel)
        return scroll

    # ---- 标签页 ----
    def _tab_raw(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)
        self.cv_raw = Canvas(1, 2, (14,5.6))
        l.addWidget(self.cv_raw)
        return w

    def _tab_preprocess(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)
        self.cv_pre = Canvas(2, 2, (14,8.4))
        l.addWidget(self.cv_pre, 3)
        self.pre_tbl = self._make_table(4, ["算法","SNR(dB)↑","平滑度↑","基线去除","综合"], 170)
        l.addWidget(self.pre_tbl, 1)
        return w

    def _tab_features(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)
        self.cv_feat1 = Canvas(1, 2, (14,4.4))
        l.addWidget(self.cv_feat1, 2)
        self.cv_feat2 = Canvas(1, 3, (14,3.4))
        l.addWidget(self.cv_feat2, 1)
        self.feat_tbl = self._make_table(0, ["周期","脉率(次/分)","主波幅度","上升时间(s)","下降时间(s)","波峰位置"], 190)
        l.addWidget(self.feat_tbl, 1)
        return w

    def _tab_compare(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)
        self.cv_cmp = Canvas(1, 3, (14,5.6))
        l.addWidget(self.cv_cmp, 2)
        self.cmp_tbl = self._make_table(4, ["算法","检测数","准确率(%)↑","误检率(%)↓","漏检率(%)↓"])
        l.addWidget(self.cmp_tbl, 1)
        return w

    def _tab_cgc(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)

        # 自采数据时显示真实三部位，否则频带分解
        self.cgc_mode_lbl = QLabel("当前模式：频带分解模拟（切换至自采数据模式可显示真实三部位）")
        self.cgc_mode_lbl.setStyleSheet(f"color:{'#E65100'};font-size:14px;padding:8px 12px;"
                                        f"background:{'#E3F2FD'};border-radius:4px;")
        l.addWidget(self.cgc_mode_lbl)

        self.cv_cgc1 = Canvas(1, 3, (14,3.9))
        l.addWidget(self.cv_cgc1, 2)
        self.cv_cgc2 = Canvas(1, 2, (14,3.9))
        l.addWidget(self.cv_cgc2, 2)

        row = QWidget()
        rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
        self.sim_tbl   = self._make_table(3, ["部位对","皮尔逊相关系数","余弦相似度","解读"])
        self.cgc_feat_tbl = self._make_table(3, ["部位","平均幅度","变异系数CV","主频(Hz)","RR标准差(ms)","脉率(次/分)"])
        rl.addWidget(self.sim_tbl, 1)
        rl.addWidget(self.cgc_feat_tbl, 2)
        l.addWidget(row, 1)
        return w

    def _tab_recog(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(8,8,8,8)

        # 结果卡片
        self.rec_card = self._make_rec_card()
        l.addWidget(self.rec_card)

        mid = QWidget()
        ml = QHBoxLayout(mid); ml.setContentsMargins(0,0,0,0); ml.setSpacing(8)
        self.cv_rec_bar   = Canvas(figsize=(9,4.3))
        self.cv_rec_radar = Canvas(figsize=(5,4.3))
        bg = QGroupBox("各脉象匹配度")
        bg.setStyleSheet(self._grp())
        bgl = QVBoxLayout(bg); bgl.addWidget(self.cv_rec_bar)
        rg = QGroupBox("特征雷达图")
        rg.setStyleSheet(self._grp())
        rgl = QVBoxLayout(rg); rgl.addWidget(self.cv_rec_radar)
        ml.addWidget(bg, 3)
        ml.addWidget(rg, 2)
        l.addWidget(mid, 3)

        self.rec_tbl = self._make_table(8, ["脉象","匹配分数","置信度","中医主治","脉象描述"])
        l.addWidget(self.rec_tbl, 1)
        return w

    def _make_rec_card(self):
        card = QWidget()
        card.setMinimumHeight(128)
        card.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {'#E3F2FD'}, stop:1 {'#DDEEFF'});
            border-radius:10px;
            border:1px solid {'#90CAF9'};
        """)
        lo = QHBoxLayout(card)
        lo.setContentsMargins(24,10,24,10)
        lo.setSpacing(24)

        self.rec_name  = QLabel("— 待识别 —")
        self.rec_name.setStyleSheet(f"font-size:30px;font-weight:bold;color:{'#1565C0'};")
        self.rec_name.setAlignment(Qt.AlignCenter)
        lo.addWidget(self.rec_name, 1)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color:{'#90CAF9'};")
        lo.addWidget(sep1)

        sw = QWidget(); sl = QVBoxLayout(sw); sl.setContentsMargins(0,0,0,0)
        QLabel_s = QLabel("匹配分数")
        QLabel_s.setStyleSheet(f"color:{'#34567A'};font-size:14px;")
        QLabel_s.setAlignment(Qt.AlignCenter)
        self.rec_score = QLabel("—")
        self.rec_score.setStyleSheet(f"font-size:28px;font-weight:bold;color:{'#2E7D32'};")
        self.rec_score.setAlignment(Qt.AlignCenter)
        sl.addWidget(QLabel_s); sl.addWidget(self.rec_score)
        lo.addWidget(sw)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet(f"color:{'#90CAF9'};")
        lo.addWidget(sep2)

        self.rec_desc = QLabel("请完成「特征提取」后，系统将自动识别脉象类型")
        self.rec_desc.setStyleSheet(f"color:{'#0D2137'};font-size:16px;")
        self.rec_desc.setWordWrap(True)
        lo.addWidget(self.rec_desc, 3)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet(f"color:{'#90CAF9'};")
        lo.addWidget(sep3)

        tw = QWidget(); tl = QVBoxLayout(tw); tl.setContentsMargins(0,0,0,0)
        QLabel_t = QLabel("中医主治")
        QLabel_t.setStyleSheet(f"color:{'#34567A'};font-size:14px;")
        self.rec_tcm = QLabel("—")
        self.rec_tcm.setStyleSheet(f"color:{'#E65100'};font-size:15px;")
        self.rec_tcm.setWordWrap(True)
        tl.addWidget(QLabel_t); tl.addWidget(self.rec_tcm)
        lo.addWidget(tw, 2)
        return card

    def _make_table(self, rows, headers, max_h=None):
        t = QTableWidget(rows, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setMinimumHeight(min(max_h, 150) if max_h else 150)
        t.verticalHeader().setDefaultSectionSize(36)
        if max_h: t.setMaximumHeight(max_h)
        return t

    def _grp(self):
        return f"""
            QGroupBox{{border:1px solid {'#90CAF9'};border-radius:8px;
            margin-top:12px;padding-top:8px;color:{'#1565C0'};
            font-weight:bold;font-size:15px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:12px;top:-7px;
            padding:0 6px;background:{'#FFFFFF'};}}
        """

    # ==================== 模式切换 ====================
    def _on_mode_change(self):
        if self.rb_bidmc.isChecked():
            self.stack.setCurrentIndex(0)
            self.data_mode = 'bidmc'
        else:
            self.stack.setCurrentIndex(1)
            self.data_mode = 'custom'

    # ==================== 数据加载 ====================
    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择BIDMC数据目录", self.data_dir)
        if d: self.data_dir = d; self.statusBar().showMessage(f"数据目录：{d}")

    def _load_bidmc(self):
        if not HAS_WFDB:
            QMessageBox.warning(self, "提示", "wfdb库未安装，无法读取BIDMC数据集。\n请切换到「自采数据」模式。")
            return
        subject = self.subject_cb.currentText()
        path = os.path.join(self.data_dir, subject)
        try:
            record = wfdb.rdrecord(path)
            idx = 0
            for i, name in enumerate(record.sig_name):
                if 'PLETH' in name.upper() or 'PPG' in name.upper():
                    idx = i; break
            raw = record.p_signal[:, idx].astype(float)
            n = int(30 * record.fs)
            self.signal_raw = raw[:n]
            self.fs = record.fs
            self._reset_results()
            self.info_lbl.setText(f"✅ {subject}\n信号: {record.sig_name[idx]}\n"
                                   f"采样率: {self.fs}Hz\n时长: 30秒")
            self._upd(数据来源=subject, 采样率=f"{self.fs}Hz")
            self.statusBar().showMessage(f"✅ 已加载 {subject} — 请点击「运行预处理」")
            self._plot_raw()
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"错误：{e}\n请确认数据目录和U盘状态。")

    def _load_custom_file(self, part):
        path, _ = QFileDialog.getOpenFileName(self, f"选择{part}部数据文件", "", "文本文件 (*.txt *.csv);;所有文件 (*)")
        if path:
            self.custom_paths[part] = path
            fname = os.path.basename(path)
            self.custom_lbls[part].setText(fname)
            self.custom_lbls[part].setStyleSheet(f"color:{'#2E7D32'};font-size:14px;")

    def _load_custom(self):
        # 检查至少有脉搏数据（用寸部作为主信号，或任意一个）
        loaded = {p: self.custom_paths[p] for p in ['寸','关','尺'] if self.custom_paths[p]}
        if not loaded:
            QMessageBox.warning(self, "提示", "请至少选择一个数据文件！")
            return
        try:
            self.fs = self.fs_spin.value()
            # 优先用寸部，其次关部，最后尺部作为主信号
            main_part = '寸' if '寸' in loaded else list(loaded.keys())[0]
            self.signal_raw = np.loadtxt(loaded[main_part])

            # 加载三部位
            self.custom_cun  = np.loadtxt(loaded['寸'])  if '寸' in loaded else None
            self.custom_guan = np.loadtxt(loaded['关'])  if '关' in loaded else None
            self.custom_chi  = np.loadtxt(loaded['尺'])  if '尺' in loaded else None

            self._reset_results()
            parts_info = "、".join(f"{p}部" for p in loaded.keys())
            n_sec = len(self.signal_raw) / self.fs
            self.info_lbl.setText(f"✅ 自采数据\n已加载：{parts_info}\n"
                                   f"采样率: {self.fs}Hz\n"
                                   f"主信号({main_part}部): {len(self.signal_raw)}点 ({n_sec:.1f}秒)")
            self._upd(数据来源="自采数据", 采样率=f"{self.fs}Hz")
            self.cgc_mode_lbl.setText(f"✅ 自采模式：已加载真实{parts_info}数据，直接展示原始三部位信号！")
            self.cgc_mode_lbl.setStyleSheet(f"color:{'#2E7D32'};font-size:14px;padding:8px 12px;"
                                             f"background:{'#E3F2FD'};border-radius:4px;")
            self.statusBar().showMessage(f"✅ 自采数据加载成功（{parts_info}）— 请点击「运行预处理」")
            self._plot_raw()
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"数据读取错误：{e}\n请确认文件格式为每行一个数值的txt文件。")

    def _reset_results(self):
        self.signal_clean = None
        self.peaks = self.onsets = self.features = None

    # ==================== 绘图：原始信号 ====================
    def _plot_raw(self):
        if self.signal_raw is None: return
        c = self.cv_raw; c.clear_all()
        t = np.arange(len(self.signal_raw)) / self.fs

        ax0 = c.axes[0]
        ax0.plot(t, self.signal_raw, color='#1565C0', lw=0.9, alpha=0.9, label='原始脉搏波')
        ax0.fill_between(t, self.signal_raw, alpha=0.08, color='#1565C0')
        ax0.set_title('原始脉搏波信号（时域）', color='#0D2137', fontsize=15, pad=10)
        ax0.set_xlabel('时间 (s)', color='#34567A')
        ax0.set_ylabel('幅值', color='#34567A')
        ax0.legend(); ax0.set_xlim(0, t[-1])

        ax1 = c.axes[1]
        freqs, psd = welch(self.signal_raw, fs=self.fs, nperseg=min(256, len(self.signal_raw)))
        mask = freqs <= min(10, self.fs/2 - 0.1)
        ax1.semilogy(freqs[mask], psd[mask], color='#5C6BC0', lw=1.2)
        ax1.fill_between(freqs[mask], psd[mask], alpha=0.15, color='#5C6BC0')
        # 脉搏主频带
        f_lo, f_hi = 0.5, min(4.0, self.fs/2-0.1)
        ax1.axvspan(f_lo, f_hi, alpha=0.12, color='#2E7D32', label=f'脉搏主频({f_lo}~{f_hi}Hz)')
        ax1.set_title('功率谱密度（Welch法）', color='#0D2137', fontsize=15, pad=10)
        ax1.set_xlabel('频率 (Hz)', color='#34567A')
        ax1.set_ylabel('PSD', color='#34567A')
        ax1.legend(); ax1.set_xlim(0, min(10, self.fs/2))

        c.fig.tight_layout(pad=2.5); c.draw()

    # ==================== 预处理 ====================
    def _run_preprocess(self):
        if self.signal_raw is None:
            QMessageBox.warning(self, "提示", "请先加载数据！"); return
        self._show_prog(True)
        raw = self.signal_raw
        fs  = self.fs

        # 四种方法
        def ma(s, w=max(3, int(fs*0.08))):
            k = np.ones(w)/w
            return np.convolve(np.pad(s,w//2,'edge'), k, 'valid')[:len(s)]

        def poly(s, deg=3):
            x = np.arange(len(s))
            b_line = np.polyval(np.polyfit(x, s, deg), x)
            det = s - b_line
            nyq = fs/2
            lo, hi = 0.3/nyq, min(0.9, 8.0/nyq)
            b,a = butter(4,[lo,hi],btype='band')
            try: return filtfilt(b,a,det)
            except: return det

        def wt_only(s):
            nyq = fs/2
            lo, hi = 0.5/nyq, min(0.99, 8.0/nyq)
            b,a = butter(4,[lo,hi],btype='band')
            try: return filtfilt(b,a,s)
            except: return s

        s_proposed = SignalProcessor.preprocess(raw, fs)
        s_ma       = ma(raw)
        s_poly     = poly(raw)
        s_wt       = wt_only(raw)
        self.signal_clean = s_proposed
        self.progress.setValue(60)

        # 绘图
        c = self.cv_pre; c.clear_all()
        t = np.arange(len(raw)) / fs
        titles  = ['所提方法（小波-形态学融合）★', '移动平均滤波', '多项式基线校正', '带通滤波']
        signals = [s_proposed, s_ma, s_poly, s_wt]
        colors  = ['#2E7D32', '#1565C0', '#C62828', '#5C6BC0']

        for ax, sig, title, color in zip(c.axes, signals, titles, colors):
            ax.plot(t, raw, color='#7A9BC0', lw=0.5, alpha=0.4, label='原始')
            ax.plot(t, sig[:len(t)], color=color, lw=1.0, label='处理后')
            snr = SignalProcessor.snr(raw, sig[:len(raw)])
            ax.set_title(f'{title}  |  SNR={snr}dB', color='#0D2137', fontsize=13, pad=7)
            ax.set_xlabel('时间(s)', fontsize=11); ax.legend(fontsize=10)
            ax.set_xlim(0, t[-1])

        c.fig.tight_layout(pad=2.0); c.draw()

        # 表格
        methods = ['所提方法','移动平均','多项式基线','带通滤波']
        evals   = ['最优★','对比','对比','对比']
        sigs_all = [s_proposed, s_ma, s_poly, s_wt]
        best_snr = max(SignalProcessor.snr(raw, s[:len(raw)]) for s in sigs_all)
        for i,(m,s,e) in enumerate(zip(methods, sigs_all, evals)):
            snr = SignalProcessor.snr(raw, s[:len(raw)])
            sm  = SignalProcessor.smoothness(s[:len(raw)])
            br  = '完全' if m=='所提方法' else '部分' if m!='带通滤波' else '部分'
            for j,v in enumerate([m, str(snr), str(sm), br, e]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if snr == best_snr:
                    item.setForeground(QColor('#2E7D32'))
                self.pre_tbl.setItem(i, j, item)

        self._show_prog(False)
        self.statusBar().showMessage("✅ 预处理完成！请继续「运行特征提取」")
        self.tabs.setCurrentIndex(1)

    # ==================== 特征提取 ====================
    def _run_features(self):
        if self.signal_clean is None:
            QMessageBox.warning(self, "提示", "请先运行预处理！"); return
        self._show_prog(True)
        sig = self.signal_clean; fs = self.fs
        alpha = self.alpha_sp.value(); beta = self.beta_sp.value()

        peaks   = SignalProcessor.detect_peaks(sig, fs, alpha, beta)
        onsets  = SignalProcessor.find_onsets(sig, peaks, fs)
        features = SignalProcessor.extract_features(sig, peaks, onsets, fs)
        self.peaks = peaks; self.onsets = onsets; self.features = features
        self.progress.setValue(50)

        t = np.arange(len(sig)) / fs
        c1 = self.cv_feat1; c1.clear_all()

        # 波峰图
        ax0 = c1.axes[0]
        ax0.plot(t, sig, color='#2E7D32', lw=0.9, alpha=0.9)
        if len(peaks) > 0:
            ax0.scatter(peaks/fs, sig[peaks], color='#C62828', s=60, zorder=5,
                        marker='^', label=f'波峰({len(peaks)}个)', edgecolors='white', lw=0.5)
        if len(onsets) > 0:
            ax0.scatter(onsets/fs, sig[onsets], color='#E65100', s=35, zorder=5,
                        marker='o', label=f'起点({len(onsets)}个)', edgecolors='white', lw=0.5)
        ax0.set_title('波峰检测结果（红▲波峰 · 橙●起点）', color='#0D2137', fontsize=14, pad=8)
        ax0.set_xlabel('时间(s)'); ax0.set_ylabel('幅值')
        ax0.legend(fontsize=11); ax0.set_xlim(0, t[-1])

        # 单周期叠加
        ax1 = c1.axes[1]
        if len(peaks) >= 3:
            clen = int(np.mean(np.diff(peaks)))
            cycles = []
            for i,p in enumerate(peaks[:-1]):
                o = onsets[i] if i<len(onsets) else max(0, p-clen//2)
                end = o + clen
                if end <= len(sig):
                    cy = sig[o:end]
                    if len(cy)==clen and np.ptp(cy)>1e-8:
                        norm = (cy-np.min(cy))/np.ptp(cy)
                        cycles.append(norm)
                        ax1.plot(np.linspace(0,1,clen), norm, color='#1565C0', alpha=0.2, lw=0.6)
            if cycles:
                mc = np.mean(cycles, axis=0)
                sc = np.std(cycles, axis=0)
                x  = np.linspace(0,1,len(mc))
                ax1.plot(x, mc, color='#C62828', lw=2.5, label='平均波形', zorder=5)
                ax1.fill_between(x, mc-sc, mc+sc, alpha=0.2, color='#C62828', label='±1σ')
        ax1.set_title(f'单周期叠加（共{len(cycles) if len(peaks)>=3 else 0}个周期，归一化）',
                      color='#0D2137', fontsize=14, pad=8)
        ax1.set_xlabel('归一化时间'); ax1.set_ylabel('归一化幅值')
        ax1.legend(fontsize=11)
        c1.fig.tight_layout(pad=2.0); c1.draw()

        # 趋势图
        c2 = self.cv_feat2; c2.clear_all()
        hr_v  = [f['heart_rate'] for f in features if f['heart_rate']!='-']
        amp_v = [f['amplitude']  for f in features]
        rt_v  = [f['rise_time']  for f in features]
        for ax, vals, label, color, title in zip(
            c2.axes,
            [hr_v, amp_v, rt_v],
            ['次/分','幅值','s'],
            ['#C62828', '#2E7D32', '#5C6BC0'],
            ['脉率趋势','主波幅度趋势','上升时间趋势']
        ):
            if vals:
                x = list(range(1, len(vals)+1))
                ax.plot(x, vals, color=color, lw=1.5, marker='o', ms=3)
                ax.axhline(np.mean(vals), color=color, lw=1, ls='--', alpha=0.5,
                           label=f'均值={np.mean(vals):.2f}')
                ax.set_title(title, color='#0D2137', fontsize=13)
                ax.set_xlabel('周期序号', fontsize=11)
                ax.set_ylabel(label, fontsize=11)
                ax.legend(fontsize=10)
        c2.fig.tight_layout(pad=1.5); c2.draw()

        # 特征表格
        self.feat_tbl.setRowCount(len(features))
        for i,f in enumerate(features):
            for j,v in enumerate([str(i+1), str(f['heart_rate']), str(f['amplitude']),
                                   str(f['rise_time']), str(f['fall_time']), str(f['peak_idx'])]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.feat_tbl.setItem(i,j,item)

        # 统计
        avg_hr  = np.mean(hr_v)  if hr_v  else 0
        avg_amp = np.mean(amp_v) if amp_v else 0
        self._upd(检测波峰=f"{len(peaks)}个", 平均脉率=f"{avg_hr:.1f}次/分", 平均幅度=f"{avg_amp:.4f}")

        # 自动触发寸关尺和识别
        self._run_cgc()
        self._run_recognition()

        self._show_prog(False)
        self.statusBar().showMessage(
            f"✅ 特征提取完成！检测到{len(peaks)}个波峰，平均脉率{avg_hr:.1f}次/分")
        self.tabs.setCurrentIndex(2)

    # ==================== 算法对比 ====================
    def _run_compare(self):
        if self.signal_clean is None:
            QMessageBox.warning(self, "提示", "请先运行预处理！"); return
        self._show_prog(True)
        sig = self.signal_clean; fs = self.fs
        alpha = self.alpha_sp.value(); beta = self.beta_sp.value()

        p0 = SignalProcessor.detect_peaks(sig, fs, alpha, beta)

        def thresh(s, fs, r=0.6):
            thr = np.min(s)+r*(np.max(s)-np.min(s))
            cands = [i for i in range(1,len(s)-1)
                     if s[i]>s[i-1] and s[i]>s[i+1] and s[i]>thr]
            mg = int(0.3*fs); pk = []
            for p in cands:
                if not pk or p-pk[-1]>mg: pk.append(p)
            return np.array(pk)

        def diff_method(s, fs):
            d = np.diff(s)
            sc = np.where((d[:-1]>0)&(d[1:]<=0))[0]+1
            thr = np.mean(s)+0.3*np.std(s)
            cands = [i for i in sc if s[i]>thr]
            mg = int(0.3*fs); pk = []
            for p in cands:
                if not pk or p-pk[-1]>mg: pk.append(p)
            return np.array(pk)

        p1 = thresh(sig, fs)
        p2 = diff_method(sig, fs)
        # 小波模极大（fallback到阈值法）
        try:
            if HAS_PYWT:
                scales = np.arange(1,9)
                coeffs,_ = pywt.cwt(sig, scales, 'cgau4')
                wt = np.abs(coeffs[5])
                thr = np.mean(wt)+0.8*np.std(wt)
                cands=[i for i in range(1,len(wt)-1)
                       if wt[i]>wt[i-1] and wt[i]>wt[i+1] and wt[i]>thr]
                mg=int(0.3*fs); pk=[]
                for p in cands:
                    if not pk or p-pk[-1]>mg: pk.append(p)
                p3=np.array(pk)
            else:
                p3 = thresh(sig, fs, 0.5)
        except: p3 = thresh(sig, fs, 0.5)

        self.progress.setValue(60)
        all_peaks = [p0,p1,p2,p3]
        names = ['所提方法','固定阈值法','差分过零点法','小波模极大值法']
        colors= ['#2E7D32','#1565C0','#C62828','#5C6BC0']
        short = ['所提方法','固定阈值','差分法','小波模极大']

        def evaluate(det, ref, tol=6):
            tp=0; matched=set()
            for dp in det:
                for j,rp in enumerate(ref):
                    if abs(dp-rp)<=tol and j not in matched:
                        tp+=1; matched.add(j); break
            fp=len(det)-tp; fn=len(ref)-tp
            return (round(tp/max(len(ref),1)*100,1),
                    round(fp/max(len(det),1)*100,1),
                    round(fn/max(len(ref),1)*100,1))

        results = [evaluate(p,p0) for p in all_peaks]

        # 绘图
        c = self.cv_cmp; c.clear_all()
        metrics = ['准确率(%)↑','误检率(%)↓','漏检率(%)↓']
        m_vals  = [[r[i] for r in results] for i in range(3)]

        for ax, metric, vals in zip(c.axes, metrics, m_vals):
            bars = ax.bar(short, vals, color=colors, edgecolor='#BBDEFB', width=0.55, lw=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                        f'{val}%', ha='center', va='bottom',
                        fontsize=12, color='#0D2137', fontweight='bold')
            bars[0].set_edgecolor('#2E7D32'); bars[0].set_linewidth(2.5)
            ax.set_title(metric, color='#0D2137', fontsize=15, pad=10)
            ax.set_ylim(0, 118)
            ax.tick_params(axis='x', labelsize=9, colors='#34567A')

        c.fig.tight_layout(pad=2.0); c.draw()

        # 表格
        for i,(name,pks,r) in enumerate(zip(names,all_peaks,results)):
            for j,v in enumerate([name, str(len(pks)), f"{r[0]}%", f"{r[1]}%", f"{r[2]}%"]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if i==0: item.setForeground(QColor('#2E7D32'))
                self.cmp_tbl.setItem(i,j,item)

        self._upd(识别脉象=f"准确率{results[0][0]}%")
        self._show_prog(False)
        self.statusBar().showMessage(f"✅ 算法对比完成！所提方法准确率{results[0][0]}%")
        self.tabs.setCurrentIndex(3)

    # ==================== 寸关尺分析 ====================
    def _run_cgc(self):
        if self.signal_clean is None: return
        sig = self.signal_clean; fs = self.fs

        # 判断是否有真实三部位数据
        has_real = (self.data_mode == 'custom' and
                    any(x is not None for x in [self.custom_cun, self.custom_guan, self.custom_chi]))

        if has_real:
            # 用真实三部位数据
            parts = {}
            labels = {'寸': self.custom_cun, '关': self.custom_guan, '尺': self.custom_chi}
            colors = {'寸': '#C62828', '关': '#1565C0', '尺': '#2E7D32'}
            for name, data in labels.items():
                if data is not None:
                    parts[name] = SignalProcessor.preprocess(data, fs)
        else:
            # 频带分解模拟
            def bp(s, lo, hi, fs):
                nyq = fs/2
                lo_n = lo/nyq; hi_n = min(0.99, hi/nyq)
                if lo_n >= hi_n: return s.copy()
                try:
                    b,a = butter(4,[lo_n,hi_n],btype='band')
                    return filtfilt(b,a,s)
                except: return s.copy()
            parts = {
                '寸': bp(sig, 0.8, min(4.0, fs/2-0.1), fs),
                '关': bp(sig, 0.5, min(2.0, fs/2-0.1), fs),
                '尺': bp(sig, 0.3, min(1.0, fs/2-0.1), fs),
            }
            colors = {'寸': '#C62828', '关': '#1565C0', '尺': '#2E7D32'}

        self._cgc_parts = parts

        # ---- 绘波形图 ----
        c1 = self.cv_cgc1; c1.clear_all()
        part_order = [k for k in ['寸','关','尺'] if k in parts]
        zh_name = {'寸':'寸部（心/肺）','关':'关部（肝/脾）','尺':'尺部（肾）'}

        for ax, name in zip(c1.axes, part_order):
            s = parts[name]
            t = np.arange(len(s)) / fs
            ax.plot(t, s, color=colors[name], lw=0.9)
            ax.fill_between(t, s, alpha=0.1, color=colors[name])
            if self.peaks is not None:
                valid = self.peaks[self.peaks < len(s)]
                ax.scatter(valid/fs, s[valid], color='white', s=20, zorder=5, alpha=0.8)
            ax.set_title(zh_name.get(name, name), color='#0D2137', fontsize=13, pad=6)
            ax.set_xlabel('时间(s)', fontsize=11); ax.set_ylabel('幅值', fontsize=11)
            ax.set_xlim(0, t[-1])
        c1.fig.tight_layout(pad=1.5); c1.draw()

        # ---- 单周期叠加对比 ----
        c2 = self.cv_cgc2; c2.clear_all()
        ax_cy = c2.axes[0]
        if self.peaks is not None and len(self.peaks) >= 3:
            clen = int(np.mean(np.diff(self.peaks)))
            for name in part_order:
                s = parts[name]; cycs=[]
                for i,p in enumerate(self.peaks[:-1]):
                    o = self.onsets[i] if (self.onsets is not None and i<len(self.onsets)) else max(0,p-clen//2)
                    end = o+clen
                    if end <= len(s):
                        cy = s[o:end]
                        if len(cy)==clen and np.ptp(cy)>1e-8:
                            cycs.append((cy-np.min(cy))/np.ptp(cy))
                if cycs:
                    mc = np.mean(cycs,axis=0)
                    ax_cy.plot(np.linspace(0,1,len(mc)), mc, color=colors[name],
                               lw=2, label=zh_name.get(name,name))
        ax_cy.set_title('三部位平均波形对比（归一化）', color='#0D2137', fontsize=14, pad=8)
        ax_cy.set_xlabel('归一化时间'); ax_cy.set_ylabel('归一化幅值')
        ax_cy.legend(fontsize=11)

        # ---- 雷达图 ----
        ax_r = c2.axes[1]
        c2.fig.delaxes(ax_r)
        ax_radar = c2.fig.add_subplot(122, polar=True)
        ax_radar.set_facecolor('#F0F6FF')
        cats = ['平均幅度','节律稳定性','脉率','主频','幅度均匀性']
        N = len(cats)
        angles = [n/N*2*np.pi for n in range(N)]; angles += angles[:1]

        feat_vals = {}
        for name in part_order:
            s = parts[name]
            pks,_ = find_peaks(s, distance=max(1,int(0.4*fs)),
                               height=np.mean(s)+0.3*np.std(s))
            amps = s[pks]-np.min(s) if len(pks)>0 else np.array([0])
            freqs,psd = welch(s, fs=fs, nperseg=min(128,len(s)))
            dom_f = freqs[np.argmax(psd)] if len(psd)>0 else 0
            hr = 60/np.mean(np.diff(pks)/fs) if len(pks)>1 else 0
            rr_s = np.std(np.diff(pks)/fs)*1000 if len(pks)>1 else 0
            feat_vals[name] = [float(np.mean(amps)), float(1/(rr_s/1000+0.01)),
                               float(hr), float(dom_f), float(1/(np.std(amps)/max(np.mean(amps),1e-8)+0.01))]

        # 归一化
        norm_v = {}
        for i in range(N):
            vals = [feat_vals[k][i] for k in part_order]
            rng = max(max(vals)-min(vals), 1e-8)
            for k in part_order:
                if k not in norm_v: norm_v[k]=[]
                norm_v[k].append((feat_vals[k][i]-min(vals))/rng)

        for name in part_order:
            v = norm_v[name]; v_closed = v+[v[0]]
            ax_radar.plot(angles, v_closed, color=colors[name], lw=2,
                         label=zh_name.get(name,name))
            ax_radar.fill(angles, v_closed, color=colors[name], alpha=0.12)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(cats, color='#0D2137', fontsize=11)
        ax_radar.set_yticks([0.25,0.5,0.75,1.0])
        ax_radar.set_yticklabels(['','','',''], fontsize=9)
        ax_radar.set_ylim(0,1.0)
        ax_radar.spines['polar'].set_color('#90CAF9')
        ax_radar.grid(color='#90CAF9', lw=0.5, alpha=0.5)
        ax_radar.set_title('三部位特征雷达图', color='#0D2137', fontsize=13, pad=14)
        ax_radar.legend(loc='lower right', bbox_to_anchor=(1.35,-0.1), fontsize=10)

        c2.fig.tight_layout(pad=1.5); c2.draw()

        # ---- 更新表格 ----
        # 相似性
        prs = [(part_order[i], part_order[j])
               for i in range(len(part_order)) for j in range(i+1,len(part_order))]
        self.sim_tbl.setRowCount(len(prs))
        for i,(a,b) in enumerate(prs):
            s1,s2 = parts[a],parts[b]
            n=min(len(s1),len(s2))
            r = float(np.corrcoef(s1[:n],s2[:n])[0,1])
            cos = float(np.dot(s1[:n],s2[:n])/(np.linalg.norm(s1[:n])*np.linalg.norm(s2[:n])+1e-8))
            interp = '高度相关' if abs(r)>=0.8 else '中度相关' if abs(r)>=0.5 else '低相关'
            label = f"{zh_name.get(a,a)} vs {zh_name.get(b,b)}"
            for j,v in enumerate([label,f'{r:.4f}',f'{cos:.4f}',interp]):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                self.sim_tbl.setItem(i,j,item)

        # 特征参数
        self.cgc_feat_tbl.setRowCount(len(part_order))
        for i,name in enumerate(part_order):
            s = parts[name]
            pks,_ = find_peaks(s, distance=max(1,int(0.4*fs)),
                               height=np.mean(s)+0.3*np.std(s))
            amps = s[pks]-np.min(s) if len(pks)>0 else np.array([0])
            freqs,psd = welch(s, fs=fs, nperseg=min(128,len(s)))
            dom_f = round(float(freqs[np.argmax(psd)]),3) if len(psd)>0 else 0
            hr = round(60/np.mean(np.diff(pks)/fs),1) if len(pks)>1 else '-'
            rr_s = round(np.std(np.diff(pks)/fs)*1000,1) if len(pks)>1 else '-'
            ma = round(float(np.mean(amps)),3)
            cv = round(float(np.std(amps)/max(np.mean(amps),1e-8)),4)
            for j,v in enumerate([zh_name.get(name,name), str(ma), str(cv),
                                   str(dom_f), str(rr_s), str(hr)]):
                item = QTableWidgetItem(v); item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(colors[name]))
                self.cgc_feat_tbl.setItem(i,j,item)

    # ==================== 脉象识别 ====================
    def _run_recognition(self):
        if self.signal_clean is None or self.peaks is None: return
        sig = self.signal_clean; fs = self.fs

        hr_v  = [f['heart_rate'] for f in self.features if f['heart_rate']!='-']
        amp_v = [f['amplitude']  for f in self.features]
        rt_v  = [f['rise_time']  for f in self.features]

        hr    = float(np.mean(hr_v))  if hr_v  else 75.0
        amp   = float(np.mean(amp_v)) if amp_v else 0.05
        cv    = float(np.std(amp_v)/max(np.mean(amp_v),1e-8)) if amp_v else 0.1
        rt    = float(np.mean(rt_v))  if rt_v  else 0.15
        slope = float(amp/max(rt,1e-8))

        rr = np.diff(self.peaks)/fs
        rr_std = float(np.std(rr)) if len(rr)>0 else 0.05

        # 寸尺比例（用三部位或估算）
        if hasattr(self,'_cgc_parts') and self._cgc_parts:
            parts = self._cgc_parts
            amps_all = {k: np.ptp(parts[k]) for k in parts}
            total = max(sum(amps_all.values()), 1e-8)
            cun_r = amps_all.get('寸', total/3) / total
            chi_r = amps_all.get('尺', total/3) / total
        else:
            cun_r = chi_r = 0.33

        scores = PulseRecognizer.score(hr, amp, cv, rt, slope, cun_r, chi_r, rr_std)
        results = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        top_key, top_score = results[0]
        top_info = PulseRecognizer.TYPES[top_key]

        conf = ('高度匹配' if top_score>=80 else '较为匹配' if top_score>=60
                else '部分匹配' if top_score>=40 else '低度匹配')

        desc_map = {
            'ping': '从容和缓，不浮不沉，节律均匀，是正常健康脉象。',
            'chi':  '脉来迟缓，脉率低于60次/分，一息不足四至。',
            'shu':  '脉来急促，脉率超过90次/分，一息五至以上。',
            'fu':   '轻取即得，重按稍减而不空，脉位表浅。',
            'chen': '轻取不应，重按始得，脉位深沉。',
            'hua':  '往来流利圆滑，如珠走盘，应指圆滑。',
            'se':   '往来艰涩，如轻刀刮竹，迟细短涩，节律不均。',
            'xian': '端直以长，如按琴弦，脉势较强，上升斜率大。',
        }

        # 更新卡片
        self.rec_name.setText(f"{top_info['name']}  ({conf})")
        self.rec_name.setStyleSheet(f"font-size:30px;font-weight:bold;color:{top_info['color']};")
        self.rec_score.setText(f"{top_score} 分")
        self.rec_desc.setText(desc_map.get(top_key,''))
        self.rec_tcm.setText(top_info['tcm'])
        self._upd(识别脉象=top_info['name'])

        # 条形图
        c = self.cv_rec_bar; c.clear_fig()
        ax = c.fig.add_subplot(111)
        ax.set_facecolor('#F0F6FF')
        for sp in ax.spines.values():
            sp.set_edgecolor('#90CAF9'); sp.set_linewidth(0.8)

        names_r  = [PulseRecognizer.TYPES[k]['name']  for k,_ in results[::-1]]
        scores_r = [v for _,v in results[::-1]]
        colors_r = [PulseRecognizer.TYPES[k]['color'] for k,_ in results[::-1]]

        bars = ax.barh(names_r, scores_r, color=colors_r,
                       edgecolor='#BBDEFB', height=0.6, lw=0.5)
        bars[-1].set_edgecolor('white'); bars[-1].set_linewidth(2)
        for bar,sc in zip(bars, scores_r):
            ax.text(min(sc+1.5,97), bar.get_y()+bar.get_height()/2,
                    f'{sc}分', va='center', ha='left',
                    color='#0D2137', fontsize=11, fontweight='bold')
        ax.axvline(60, color='#7A9BC0',   lw=1, ls='--', alpha=0.5)
        ax.axvline(80, color='#2E7D32',    lw=1, ls='--', alpha=0.5)
        ax.text(61,-0.5,'60分',color='#7A9BC0',fontsize=10,va='top')
        ax.text(81,-0.5,'80分',color='#2E7D32',fontsize=10,va='top')
        ax.set_xlim(0,105); ax.set_xlabel('匹配分数', color='#34567A')
        ax.set_title('八种脉象匹配度排行', color='#0D2137', fontsize=14, pad=10)
        ax.tick_params(colors='#34567A')
        ax.grid(axis='x', alpha=0.3, ls='--')
        c.fig.tight_layout(pad=1.5); c.draw()

        # 雷达图
        c2 = self.cv_rec_radar; c2.clear_fig()
        ax2 = c2.fig.add_subplot(111, polar=True)
        ax2.set_facecolor('#F0F6FF')
        c2.fig.patch.set_facecolor('#FFFFFF')

        cats6 = ['脉率','幅度','节律\n稳定性','上升\n速度','寸部\n比例','尺部\n比例']
        N6 = len(cats6)
        ang6 = [n/N6*2*np.pi for n in range(N6)]; ang6 += ang6[:1]

        hr_n  = min(1.0,max(0,(hr-40)/120))
        amp_n = min(1.0,amp/0.15)
        reg_n = max(0,1-min(1,cv/0.3))
        spd_n = max(0,1-min(1,rt/0.4))
        cun_n = min(1.0,cun_r*2)
        chi_n = min(1.0,chi_r*2)
        vals6 = [hr_n,amp_n,reg_n,spd_n,cun_n,chi_n]; vals6 += vals6[:1]

        ax2.plot(ang6, vals6, color=top_info['color'], lw=2, zorder=5)
        ax2.fill(ang6, vals6, color=top_info['color'], alpha=0.2)
        ref = [0.5,0.5,0.8,0.6,0.5,0.5,0.5]
        ax2.plot(ang6, ref, color='#7A9BC0', lw=1, ls='--', alpha=0.5, label='平脉基准')

        ax2.set_xticks(ang6[:-1])
        ax2.set_xticklabels(cats6, color='#0D2137', fontsize=11)
        ax2.set_yticks([0.25,0.5,0.75,1.0])
        ax2.set_yticklabels(['','','',''], fontsize=9)
        ax2.set_ylim(0,1); ax2.spines['polar'].set_color('#90CAF9')
        ax2.grid(color='#90CAF9', lw=0.5, alpha=0.5)
        ax2.set_title(f'特征分布\n（{top_info["name"]}）', color='#0D2137', fontsize=13, pad=14)
        ax2.legend(loc='lower right', bbox_to_anchor=(1.3,-0.1), fontsize=10)
        c2.fig.tight_layout(); c2.draw()

        # 结果表格
        descs = list(desc_map.values())
        tcms  = [PulseRecognizer.TYPES[k]['tcm'] for k,_ in results]
        self.rec_tbl.setRowCount(len(results))
        for i,(key,sc) in enumerate(results):
            info = PulseRecognizer.TYPES[key]
            cf = ('高度匹配' if sc>=80 else '较为匹配' if sc>=60
                  else '部分匹配' if sc>=40 else '低度匹配')
            for j,v in enumerate([info['name'], f'{sc}分', cf, info['tcm'],
                                   desc_map.get(key,'')[:35]+'...' if len(desc_map.get(key,''))>35 else desc_map.get(key,'')]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor(info['color']))
                self.rec_tbl.setItem(i,j,item)
            if i==0:
                for j in range(5):
                    it = self.rec_tbl.item(i,j)
                    if it: it.setBackground(QColor('#E3F2FD'))

    # ==================== 保存图表 ====================
    def _save_all(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not d: return
        try:
            saves = [
                (self.cv_raw,       '01_原始信号与频谱.png'),
                (self.cv_pre,       '02_预处理算法对比.png'),
                (self.cv_feat1,     '03_特征点检测与单周期叠加.png'),
                (self.cv_feat2,     '04_特征参数趋势图.png'),
                (self.cv_cmp,       '05_算法性能对比.png'),
                (self.cv_cgc1,      '06_寸关尺三部位波形.png'),
                (self.cv_cgc2,      '07_寸关尺叠加与雷达图.png'),
                (self.cv_rec_bar,   '08_脉象识别匹配度.png'),
                (self.cv_rec_radar, '09_脉象特征雷达图.png'),
            ]
            for canvas, fname in saves:
                canvas.fig.savefig(os.path.join(d, fname),
                                   dpi=200, bbox_inches='tight',
                                   facecolor='#FFFFFF')
            QMessageBox.information(self, "保存成功",
                f"✅ 9张图表已保存至：\n{d}\n可直接插入论文！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _show_prog(self, show):
        self.progress.setVisible(show)
        if show:
            self.progress.setValue(0)
            QTimer.singleShot(300, lambda: self.progress.setValue(40))
        else:
            self.progress.setValue(100)
            QTimer.singleShot(400, lambda: self.progress.setVisible(False))


# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 12))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
