"""
Scientific Experiment Mode Tab for AI Waste Doctor.
Allows students to test AI classification under different environmental conditions,
compare true vs predicted classes, and log trial data to CSV for accuracy evaluation.
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QLineEdit
)

from data.experiment_logger import ExperimentLogger
from ui.theme import ThemeManager


class ExperimentTab(QWidget):
    """Scientific experiment testing & logging tab."""

    results_updated = Signal()

    ENVIRONMENT_CONDITIONS = [
        "Normal Lighting",
        "Dim Lighting",
        "Harsh Lighting",
        "Different Background",
        "Tilted Object",
        "Partial Occlusion",
        "Custom Condition"
    ]

    def __init__(self, classifier, parent=None):
        super().__init__(parent)
        self.classifier = classifier
        self.logger = ExperimentLogger()

        self.latest_predicted_class = ""
        self.latest_confidence = 0.0

        self._build_ui()
        self._connect_signals()
        self.refresh_table()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # =========================================================================
        # LEFT PANEL – EXPERIMENT FORM
        # =========================================================================
        t = ThemeManager.get_theme()

        form_card = QFrame()
        form_card.setProperty("class", "CardFrame")

        form_layout = QVBoxLayout(form_card)
        form_layout.setSpacing(14)

        form_title = QLabel("🧪 SCIENTIFIC EXPERIMENT FORM")
        form_title.setObjectName("H3")

        form_desc = QLabel("Test how environmental conditions affect AI accuracy. Select the ground truth class and condition, then click Save Test Result.")
        form_desc.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        form_desc.setWordWrap(True)

        # 1. True Class Dropdown
        lbl_true = QLabel("True Ground-Truth Class:")
        lbl_true.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.combo_true_class = QComboBox()
        self.populate_class_dropdown()

        # 2. Testing Condition Dropdown
        lbl_cond = QLabel("Environmental Testing Condition:")
        lbl_cond.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.combo_condition = QComboBox()
        self.combo_condition.addItems(self.ENVIRONMENT_CONDITIONS)

        # Custom Condition Text Box (visible if Custom selected)
        self.txt_custom_cond = QLineEdit()
        self.txt_custom_cond.setPlaceholderText("Specify custom condition...")
        self.txt_custom_cond.setVisible(False)

        # 3. Predicted Class Display
        lbl_pred = QLabel("AI Predicted Class (Auto-Filled):")
        lbl_pred.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.lbl_predicted_val = QLabel("No Object Scanned Yet")
        self.lbl_predicted_val.setStyleSheet(f"""
            background-color: {t['bg_card']};
            color: {t['accent_primary']};
            font-weight: 800;
            font-size: 14px;
            padding: 10px;
            border: 1px solid {t['border']};
            border-radius: 6px;
        """)

        # 4. Confidence Display
        lbl_conf = QLabel("AI Confidence:")
        lbl_conf.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.lbl_confidence_val = QLabel("0.0%")
        self.lbl_confidence_val.setStyleSheet(f"""
            background-color: {t['bg_card']};
            color: {t['accent_secondary']};
            font-weight: 800;
            font-size: 14px;
            padding: 10px;
            border: 1px solid {t['border']};
            border-radius: 6px;
        """)

        # 5. Correct Prediction Calculation Box
        lbl_correct = QLabel("Correct Prediction?")
        lbl_correct.setStyleSheet(f"font-weight: 700; color: {t['text_primary']};")
        self.lbl_correct_val = QLabel("Pending Comparison")
        self.lbl_correct_val.setStyleSheet(f"""
            background-color: {t['bg_card']};
            color: {t['text_secondary']};
            font-weight: 800;
            font-size: 14px;
            padding: 10px;
            border: 1px solid {t['border']};
            border-radius: 6px;
        """)

        # Save Button
        self.btn_save = QPushButton("💾 SAVE TEST RESULT")
        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['accent_primary']};
                font-size: 14px;
                font-weight: 800;
                padding: 12px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {t['accent_secondary']};
            }}
        """)

        form_layout.addWidget(form_title)
        form_layout.addWidget(form_desc)
        form_layout.addWidget(lbl_true)
        form_layout.addWidget(self.combo_true_class)
        form_layout.addWidget(lbl_cond)
        form_layout.addWidget(self.combo_condition)
        form_layout.addWidget(self.txt_custom_cond)
        form_layout.addWidget(lbl_pred)
        form_layout.addWidget(self.lbl_predicted_val)
        form_layout.addWidget(lbl_conf)
        form_layout.addWidget(self.lbl_confidence_val)
        form_layout.addWidget(lbl_correct)
        form_layout.addWidget(self.lbl_correct_val)
        form_layout.addStretch()
        form_layout.addWidget(self.btn_save)

        # =========================================================================
        # RIGHT PANEL – RECENT EXPERIMENT LOG TABLE
        # =========================================================================
        right_card = QFrame()
        right_card.setProperty("class", "CardFrame")

        right_layout = QVBoxLayout(right_card)

        tbl_title = QLabel("RECENT EXPERIMENT LOGS (data/results.csv)")
        tbl_title.setObjectName("H3")

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "True Class", "Predicted Class", "Conf %", "Condition", "Correct?"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)

        right_layout.addWidget(tbl_title)
        right_layout.addWidget(self.table, 1)

        main_layout.addWidget(form_card, 4)
        main_layout.addWidget(right_card, 6)

    def populate_class_dropdown(self):
        """Populate ground truth class dropdown from classifier labels."""
        self.combo_true_class.clear()
        if self.classifier.labels:
            self.combo_true_class.addItems(self.classifier.labels)
        else:
            self.combo_true_class.addItems(["Organic Waste", "Paper/Cardboard", "Plastic/Metal"])

    def _connect_signals(self):
        self.combo_true_class.currentIndexChanged.connect(self._evaluate_correctness)
        self.combo_condition.currentIndexChanged.connect(self._on_condition_changed)
        self.btn_save.clicked.connect(self.on_save_clicked)

    def update_live_prediction(self, predicted_class: str, confidence: float):
        """Update form fields with latest camera scan prediction."""
        self.latest_predicted_class = predicted_class.strip()
        self.latest_confidence = confidence

        self.lbl_predicted_val.setText(predicted_class or "No Object Scanned Yet")
        self.lbl_confidence_val.setText(f"{confidence:.1f}%")

        self._evaluate_correctness()

    def _on_condition_changed(self, idx: int):
        cond_text = self.combo_condition.currentText()
        self.txt_custom_cond.setVisible(cond_text == "Custom Condition")

    def _evaluate_correctness(self):
        true_cls = self.combo_true_class.currentText().strip()
        pred_cls = self.latest_predicted_class.strip()
        
        t = ThemeManager.get_theme()

        if not pred_cls or pred_cls == "No Object Scanned Yet":
            self.lbl_correct_val.setText("Pending Scan")
            self.lbl_correct_val.setStyleSheet(f"""
                background-color: {t['bg_card']}; color: {t['text_secondary']}; font-weight: 800; font-size: 14px; padding: 10px; border: 1px solid {t['border']}; border-radius: 6px;
            """)
            return

        is_match = (true_cls.lower() == pred_cls.lower())
        if is_match:
            self.lbl_correct_val.setText("YES  (Match)")
            self.lbl_correct_val.setStyleSheet(f"""
                background-color: {t['bg_secondary']}; color: {t['accent_primary']}; font-weight: 800; font-size: 14px; padding: 10px; border: 1px solid {t['accent_primary']}; border-radius: 6px;
            """)
        else:
            self.lbl_correct_val.setText("NO  (Mismatch)")
            self.lbl_correct_val.setStyleSheet(f"""
                background-color: {t['error']}; color: {t['bg_main']}; font-weight: 800; font-size: 14px; padding: 10px; border: 1px solid {t['error']}; border-radius: 6px;
            """)

    @Slot()
    def on_save_clicked(self):
        """Log trial data to CSV."""
        true_cls = self.combo_true_class.currentText().strip()
        pred_cls = self.latest_predicted_class.strip()

        if not pred_cls:
            QMessageBox.warning(self, "Missing Prediction", "Please capture or wait for an object scan prediction first.")
            return

        cond = self.combo_condition.currentText()
        if cond == "Custom Condition":
            cond = self.txt_custom_cond.text().strip() or "Custom Condition"

        self.logger.log_result(
            true_class=true_cls,
            predicted_class=pred_cls,
            confidence=self.latest_confidence,
            condition=cond
        )

        self.refresh_table()
        self.results_updated.emit()
        QMessageBox.information(self, "Trial Saved", f"Successfully logged trial result for '{cond}'.")

    def refresh_table(self):
        """Reload CSV rows into summary table."""
        results = self.logger.get_all_results()
        self.table.setRowCount(len(results))

        for row_idx, data in enumerate(reversed(results)):  # Display latest first
            self.table.setItem(row_idx, 0, QTableWidgetItem(data.get("timestamp", "")))
            self.table.setItem(row_idx, 1, QTableWidgetItem(data.get("true_class", "")))
            self.table.setItem(row_idx, 2, QTableWidgetItem(data.get("predicted_class", "")))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"{data.get('confidence', '')}%"))
            self.table.setItem(row_idx, 4, QTableWidgetItem(data.get("condition", "")))
            
            correct_item = QTableWidgetItem(data.get("correct", ""))
            if data.get("correct", "").lower() == "yes":
                correct_item.setForeground(Qt.GlobalColor.green)
            else:
                correct_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row_idx, 5, correct_item)
