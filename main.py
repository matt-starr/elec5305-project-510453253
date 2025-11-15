import sys
import librosa
import numpy as np
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QLabel, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from analyse import AudioAnalyser

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    - finished: No data
    - error: tuple (exctype, value, traceback.format_exc())
    - result: object data returned from processing
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class AnalysisWorker(QRunnable):
    '''
    Worker thread for running the audio analysis.
    '''
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = WorkerSignals()

    def run(self):
        try:
            analyser = AudioAnalyser(self.file_path)
            if not analyser.load_audio():
                raise ValueError("Could not load the audio file.")
            
            analyser.extract_bpm()
            analyser.extract_chromagram()
            
            self.signals.result.emit(analyser)
        except Exception as e:
            self.signals.error.emit((type(e), e, e.__traceback__))
        finally:
            self.signals.finished.emit()


class MplCanvas(FigureCanvas):
    """A custom Matplotlib canvas widget to embed in a PyQt6 application."""
    def __init__(self, parent=None, width=80, height=60, dpi=100):
        # Create a new Matplotlib figure
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Add two subplots, stacked vertically
        self.axes1 = fig.add_subplot(211) # For the waveform
        self.axes2 = fig.add_subplot(212) # For the chromagram
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    """The main window of the Music Analyser application."""
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ELEC5305 Music Analyser")
        self.setGeometry(100, 100, 1000, 800)

        # --- Thread Pool for background tasks ---
        self.threadpool = QThreadPool()

        # --- Central Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # --- Load File Button ---
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # --- File Info Label ---
        self.file_label = QLabel("No file loaded.")
        layout.addWidget(self.file_label)

        # --- Header Row for Musical Features ---
        header_layout = QHBoxLayout()
        
        self.bpm_label = QLabel("BPM: --")
        self.key_label = QLabel("Key: --")
        self.instruments_label = QLabel("Instruments: N/A")
        
        header_layout.addWidget(self.bpm_label)
        header_layout.addWidget(self.key_label)
        header_layout.addWidget(self.instruments_label)
        
        layout.addLayout(header_layout)
        
        # --- Matplotlib Display ---
        self.canvas = MplCanvas(self, width=80, height=60, dpi=100)
        layout.addWidget(self.canvas)

    def open_file_dialog(self):
        """
        Opens a file dialog to allow the user to select an audio file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3);;All Files (*)"
        )

        if file_path:
            self.file_label.setText(f"Loading: {file_path.split('/')[-1]}...")
            self.start_analysis(file_path)

    def start_analysis(self, file_path):
        """
        Starts the audio analysis in a background thread.
        """
        self.load_button.setEnabled(False)
        self.bpm_label.setText("BPM: Analyzing...")
        self.key_label.setText("Key: Analyzing...")

        worker = AnalysisWorker(file_path)
        worker.signals.result.connect(self.analysis_complete)
        worker.signals.error.connect(self.analysis_error)
        worker.signals.finished.connect(self.analysis_finished)
        
        self.threadpool.start(worker)

    def analysis_complete(self, analyser):
        """
        This function is called when the analysis worker has finished.
        It updates the GUI with the results.
        """
        estimated_key = analyser.estimate_key(analyser.chromagram)
        
        self.file_label.setText(f"Loaded: {analyser.file_path.split('/')[-1]}")
        self.bpm_label.setText(f"BPM: {analyser.bpm:.2f}")
        self.key_label.setText(f"Key: {estimated_key}")

        # --- Clear and update plots ---
        self.canvas.axes1.cla()
        self.canvas.axes2.cla()

        librosa.display.waveshow(analyser.y, sr=analyser.sr, ax=self.canvas.axes1, color='blue', alpha=0.5)
        self.canvas.axes1.set_title("Waveform")
        self.canvas.axes1.set_xlabel(None)
        self.canvas.axes1.set_ylabel("Amplitude")
        self.canvas.axes1.set_xlim(0, len(analyser.y) / analyser.sr)
        self.canvas.axes1.grid(True)

        librosa.display.specshow(analyser.chromagram, y_axis='chroma', x_axis='time', ax=self.canvas.axes2)
        self.canvas.axes2.set_title("Chromagram")
        self.canvas.axes2.set_xlabel("Time (s)")
        self.canvas.axes2.set_ylabel("Pitch Class")
        
        self.canvas.draw()

    def analysis_error(self, error_tuple):
        """Handles errors from the worker thread."""
        error_message = f"An error occurred: {error_tuple[1]}"
        self.show_error_dialog(error_message)
        self.file_label.setText("Error processing file.")
        self.bpm_label.setText("BPM: --")
        self.key_label.setText("Key: --")

    def analysis_finished(self):
        """Called when the worker thread has finished."""
        self.load_button.setEnabled(True)

    def show_error_dialog(self, message):
        """Displays an error message in a dialog box."""
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Error")
        dlg.setText(message)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()

# --- Main execution block ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
