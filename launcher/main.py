"""
PhosLab Pipeline Launcher — punto de entrada
"""
import sys
from pathlib import Path
from launcher import PipelineLauncher
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PipelineLauncher()
    window.show()
    sys.exit(app.exec())