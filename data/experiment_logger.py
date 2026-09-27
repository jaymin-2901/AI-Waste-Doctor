"""
Experiment Logger module for AI Waste Doctor.
Logs scientific experiment trial data to CSV and computes accuracy metrics per condition.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from config import CSV_FILE


class ExperimentLogger:
    """Manages recording, reading, metrics calculation, and clearing of experiment results."""

    HEADERS = ["timestamp", "true_class", "predicted_class", "confidence", "condition", "correct"]

    def __init__(self, csv_path=None):
        self.csv_path = Path(csv_path) if csv_path else CSV_FILE
        self.ensure_csv_file()

    def ensure_csv_file(self):
        """Create CSV file with header if it doesn't exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADERS)
            except Exception as e:
                print(f"[ExperimentLogger] Error initializing CSV file: {e}")

    def log_result(self, true_class: str, predicted_class: str, confidence: float, condition: str) -> dict:
        """
        Record a single experiment trial.
        
        Returns:
        --------
        dict: Logged record entry
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_correct = "Yes" if true_class.strip().lower() == predicted_class.strip().lower() else "No"
        confidence_val = round(float(confidence), 1)

        row = [timestamp, true_class, predicted_class, confidence_val, condition, is_correct]

        try:
            self.ensure_csv_file()
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"[ExperimentLogger] Logged trial: {row}")
        except Exception as e:
            print(f"[ExperimentLogger] Error logging result to CSV: {e}")

        return {
            "timestamp": timestamp,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence_val,
            "condition": condition,
            "correct": is_correct
        }

    def get_all_results(self) -> list:
        """Read all logged experiment records from CSV file."""
        self.ensure_csv_file()
        results = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and "timestamp" in row:
                        results.append(row)
        except Exception as e:
            print(f"[ExperimentLogger] Error reading CSV results: {e}")
        return results

    def calculate_metrics(self) -> dict:
        """
        Compute overall and condition-specific accuracy statistics.
        
        Returns:
        --------
        dict containing:
            total_tests: int
            correct_count: int
            incorrect_count: int
            overall_accuracy: float (%)
            condition_metrics: dict {condition: {"total": int, "correct": int, "accuracy": float}}
        """
        results = self.get_all_results()
        total_tests = len(results)

        if total_tests == 0:
            return {
                "total_tests": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "overall_accuracy": 0.0,
                "condition_metrics": {}
            }

        correct_count = 0
        condition_stats = {}

        for row in results:
            cond = row.get("condition", "Normal Lighting")
            is_correct = row.get("correct", "No").strip().lower() == "yes"

            if cond not in condition_stats:
                condition_stats[cond] = {"total": 0, "correct": 0, "accuracy": 0.0}

            condition_stats[cond]["total"] += 1

            if is_correct:
                correct_count += 1
                condition_stats[cond]["correct"] += 1

        incorrect_count = total_tests - correct_count
        overall_accuracy = round((correct_count / total_tests) * 100.0, 1)

        # Compute accuracy per condition
        for cond, stats in condition_stats.items():
            tot = stats["total"]
            cor = stats["correct"]
            stats["accuracy"] = round((cor / tot) * 100.0, 1) if tot > 0 else 0.0

        return {
            "total_tests": total_tests,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "overall_accuracy": overall_accuracy,
            "condition_metrics": condition_stats
        }

    def clear_results(self) -> bool:
        """Reset CSV log file (clears all recorded trial rows)."""
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
            print("[ExperimentLogger] Cleared experiment CSV log.")
            return True
        except Exception as e:
            print(f"[ExperimentLogger] Error clearing CSV file: {e}")
            return False
