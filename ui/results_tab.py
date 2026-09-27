"""
Experiment Results Dashboard Tab for AI Waste Doctor.
Displays total trials, accuracy breakdown per environmental condition,
embedded matplotlib charts, and controls for refreshing or clearing experiment logs.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QScrollArea
)

from data.experiment_logger import ExperimentLogger
from ui.theme import ThemeManager
from ui.widgets import MetricCard

# Try importing Matplotlib Qt Canvas
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    FigureCanvas = QWidget
    MATPLOTLIB_AVAILABLE = False


class MatplotlibCanvas(FigureCanvas):
    """Embedded matplotlib canvas styled with app's dark navy theme."""

    def __init__(self, parent=None):
        t = ThemeManager.get_theme()
        self.fig = Figure(figsize=(6, 3), dpi=100, facecolor=t['bg_card'])
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_facecolor(t['bg_main'])

    def plot_accuracy_by_condition(self, condition_metrics: dict):
        """Render bar chart of accuracy % per environmental condition."""
        t = ThemeManager.get_theme()
        self.ax.clear()
        self.ax.set_facecolor(t['bg_main'])

        if not condition_metrics:
            self.ax.text(
                0.5, 0.5, "No Experiment Data Logged",
                color=t['text_secondary'], fontsize=12, fontweight="bold",
                ha="center", va="center", transform=self.ax.transAxes
            )
            self.draw()
            return

        conditions = list(condition_metrics.keys())
        accuracies = [stats["accuracy"] for stats in condition_metrics.values()]

        # Shorten labels for clean display
        short_labels = [c.replace(" Lighting", " Lgt").replace(" Background", " Bkg") for c in conditions]

        colors = [t['accent_primary'] if acc >= 75.0 else (t['highlight'] if acc >= 60.0 else t['error']) for acc in accuracies]

        bars = self.ax.bar(short_labels, accuracies, color=colors, width=0.5, edgecolor=t['border'])

        # Annotate bars with accuracy numbers
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                color=t['text_primary'], fontsize=9, fontweight="bold"
            )

        self.ax.set_ylim(0, 110)
        self.ax.set_ylabel("Accuracy (%)", color=t['text_secondary'], fontsize=10, fontweight="bold")
        self.ax.set_title("ACCURACY BY ENVIRONMENTAL CONDITION", color=t['accent_primary'], fontsize=11, fontweight="bold", pad=10)
        self.ax.tick_params(colors=t['text_secondary'], labelsize=8)
        self.ax.spines["bottom"].set_color(t['border'])
        self.ax.spines["left"].set_color(t['border'])
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        self.fig.tight_layout()
        self.draw()


class ResultsTab(QWidget):
    """Results Dashboard tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = ExperimentLogger()

        self._build_ui()
        self._connect_signals()
        self.refresh_dashboard()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        t = ThemeManager.get_theme()

        # Header Title & Action Buttons
        top_bar = QHBoxLayout()
        title_lbl = QLabel("EXPERIMENT RESULTS DASHBOARD")
        title_lbl.setObjectName("H2")

        self.btn_refresh = QPushButton("🔄 REFRESH RESULTS")
        self.btn_clear = QPushButton("🗑️ CLEAR EXPERIMENT DATA")
        self.btn_clear.setObjectName("DangerButton")

        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_refresh)
        top_bar.addWidget(self.btn_clear)

        # Metrics Cards Row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)

        self.card_total = MetricCard("TOTAL TESTS", "0", "🧪", t['accent_secondary'])
        self.card_correct = MetricCard("CORRECT PREDICTIONS", "0", "✅", t['accent_primary'])
        self.card_incorrect = MetricCard("INCORRECT PREDICTIONS", "0", "❌", t['error'])
        self.card_accuracy = MetricCard("OVERALL ACCURACY", "0.0%", "🎯", t['highlight'])

        metrics_layout.addWidget(self.card_total)
        metrics_layout.addWidget(self.card_correct)
        metrics_layout.addWidget(self.card_incorrect)
        metrics_layout.addWidget(self.card_accuracy)

        # Content Split Layout: Left Table + Right Matplotlib Chart
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left Table Card (Condition Breakdown)
        table_card = QFrame()
        table_card.setProperty("class", "CardFrame")
        table_layout = QVBoxLayout(table_card)

        tbl_header = QLabel("ACCURACY BY TESTING CONDITION")
        tbl_header.setObjectName("H3")

        self.cond_table = QTableWidget()
        self.cond_table.setColumnCount(4)
        self.cond_table.setHorizontalHeaderLabels(["Testing Condition", "Total", "Correct", "Accuracy %"])
        self.cond_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        table_layout.addWidget(tbl_header)
        table_layout.addWidget(self.cond_table, 1)

        # Right Chart Card
        chart_card = QFrame()
        chart_card.setProperty("class", "CardFrame")
        chart_layout = QVBoxLayout(chart_card)

        if MATPLOTLIB_AVAILABLE:
            self.chart_canvas = MatplotlibCanvas(self)
            chart_layout.addWidget(self.chart_canvas, 1)
        else:
            no_chart_lbl = QLabel("Matplotlib is not installed. Table breakdown shown above.")
            no_chart_lbl.setStyleSheet(f"color: {t['text_secondary']}; font-weight: 700;")
            chart_layout.addWidget(no_chart_lbl)

        content_layout.addWidget(table_card, 5)
        content_layout.addWidget(chart_card, 5)

        main_layout.addLayout(top_bar)
        main_layout.addLayout(metrics_layout)
        main_layout.addLayout(content_layout, 1)

    def _connect_signals(self):
        self.btn_refresh.clicked.connect(self.refresh_dashboard)
        self.btn_clear.clicked.connect(self.on_clear_clicked)

    @Slot()
    def refresh_dashboard(self):
        """Recalculate experiment metrics and update metrics cards, table, and charts."""
        metrics = self.logger.calculate_metrics()

        total = metrics["total_tests"]
        correct = metrics["correct_count"]
        incorrect = metrics["incorrect_count"]
        overall_acc = metrics["overall_accuracy"]
        cond_metrics = metrics["condition_metrics"]

        # Update cards
        self.card_total.set_value(str(total))
        self.card_correct.set_value(str(correct))
        self.card_incorrect.set_value(str(incorrect))
        self.card_accuracy.set_value(f"{overall_acc:.1f}%")

        # Update condition table
        self.cond_table.setRowCount(len(cond_metrics))
        for row_idx, (cond_name, stats) in enumerate(cond_metrics.items()):
            self.cond_table.setItem(row_idx, 0, QTableWidgetItem(cond_name))
            self.cond_table.setItem(row_idx, 1, QTableWidgetItem(str(stats["total"])))
            self.cond_table.setItem(row_idx, 2, QTableWidgetItem(str(stats["correct"])))
            
            acc_item = QTableWidgetItem(f"{stats['accuracy']:.1f}%")
            if stats['accuracy'] >= 75.0:
                acc_item.setForeground(Qt.GlobalColor.green)
            else:
                acc_item.setForeground(Qt.GlobalColor.yellow)
            self.cond_table.setItem(row_idx, 3, acc_item)

        # Render bar chart if Matplotlib is available
        if MATPLOTLIB_AVAILABLE and hasattr(self, "chart_canvas"):
            self.chart_canvas.plot_accuracy_by_condition(cond_metrics)

    @Slot()
    def on_clear_clicked(self):
        """Prompt user confirmation before clearing experiment CSV data."""
        reply = QMessageBox.question(
            self,
            "Clear Experiment Data",
            "Are you sure you want to clear all recorded experiment trial data?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.logger.clear_results()
            self.refresh_dashboard()
            QMessageBox.information(self, "Data Cleared", "All experiment trial data has been successfully reset.")
