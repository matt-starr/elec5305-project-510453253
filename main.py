import sys
import librosa
import numpy as np
import pygame
import pyqtgraph as pg
from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, QTimer, QThread
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QLabel, QMessageBox, QProgressBar
)
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
            analyser.time_signature = analyser.estimate_time_signature()
            
            self.signals.result.emit(analyser)
        except Exception as e:
            self.signals.error.emit((type(e), e, e.__traceback__))
        finally:
            self.signals.finished.emit()

class PlaybackWorker(QObject):
    """Worker for handling audio playback in a separate thread."""
    positionChanged = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.is_paused = False
        self._is_running = False
        self.timer = None

    def load_file(self, file_path):
        self.current_file = file_path
        pygame.mixer.init()
        pygame.mixer.music.load(self.current_file)

    def play(self):
        if self.current_file:
            if not self._is_running:
                pygame.mixer.music.play()
                self._is_running = True
                self.is_paused = False
                
                # This timer will run in the new thread's event loop
                self.timer = QTimer()
                self.timer.setInterval(100)
                self.timer.timeout.connect(self.update_position)
                self.timer.start()
            elif self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                if self.timer:
                    self.timer.start()

    def pause(self):
        if self._is_running and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            if self.timer:
                self.timer.stop()

    def stop(self):
        if self._is_running:
            pygame.mixer.music.stop()
            self._is_running = False
            self.is_paused = False
            if self.timer:
                self.timer.stop()
            self.positionChanged.emit(0)

    def update_position(self):
        if pygame.mixer.music.get_busy() and self._is_running and not self.is_paused:
            pos_ms = pygame.mixer.music.get_pos()
            pos_s = pos_ms / 1000.0
            self.positionChanged.emit(pos_s)
        elif self._is_running: # Song finished
            self.stop()
            self.finished.emit()


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

        # --- Playback Control Buttons ---
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        
        self.play_button.clicked.connect(self.play_audio)
        self.pause_button.clicked.connect(self.pause_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.pause_button)
        playback_layout.addWidget(self.stop_button)
        layout.addLayout(playback_layout)

        # --- File Info Label ---
        self.file_label = QLabel("No file loaded.")
        layout.addWidget(self.file_label)

        # --- Header Row for Musical Features ---
        header_layout = QHBoxLayout()
        
        self.bpm_label = QLabel("BPM: --")
        self.key_label = QLabel("Key: --")
        self.time_signature_label = QLabel("Time Signature: --")
        self.instruments_label = QLabel("Instruments: N/A")

        # --- Style for feature labels ---
        feature_label_style = "font-size: 14pt; font-weight: bold;"
        self.bpm_label.setStyleSheet(feature_label_style)
        self.key_label.setStyleSheet(feature_label_style)
        self.time_signature_label.setStyleSheet(feature_label_style)
        self.instruments_label.setStyleSheet(feature_label_style)
        
        header_layout.addWidget(self.bpm_label)
        header_layout.addWidget(self.key_label)
        header_layout.addWidget(self.time_signature_label)
        header_layout.addWidget(self.instruments_label)
        
        layout.addLayout(header_layout)
        
        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # --- Plotting Widgets ---
        self.waveform_plot = pg.PlotWidget()
        self.chromagram_plot = pg.PlotWidget()
        
        self.waveform_plot.setBackground(None)
        self.chromagram_plot.setBackground(None)

        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setTitle('Waveform')
        
        self.chromagram_plot.setLabel('left', 'Pitch Class')
        self.chromagram_plot.setLabel('bottom', 'Time (s)')
        self.chromagram_plot.setTitle('Chromagram')

        layout.addWidget(self.waveform_plot)
        layout.addWidget(self.chromagram_plot)

        # --- Playback Indicator Lines ---
        self.waveform_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.chromagram_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.waveform_plot.addItem(self.waveform_line, ignoreBounds=True)
        self.chromagram_plot.addItem(self.chromagram_line, ignoreBounds=True)
        self.waveform_line.hide()
        self.chromagram_line.hide()

        # --- Audio Playback Thread ---
        self.playback_thread = QThread()
        self.playback_worker = PlaybackWorker()
        self.playback_worker.moveToThread(self.playback_thread)

        self.playback_worker.positionChanged.connect(self.update_playback_position)
        self.playback_worker.finished.connect(self.on_playback_finished)

        self.playback_thread.start()

        self.current_file = None

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
        self.progress_bar.setVisible(True)
        self.load_button.setEnabled(False)
        self.bpm_label.setText("BPM: Analyzing...")
        self.key_label.setText("Key: Analyzing...")
        self.time_signature_label.setText("Time Signature: Analyzing...")

        worker = AnalysisWorker(file_path)
        worker.signals.result.connect(self.analysis_complete)
        worker.signals.error.connect(self.analysis_error)
        worker.signals.finished.connect(self.analysis_finished)
        
        self.threadpool.start(worker)

    def analysis_complete(self, analyser):
        """
        This function is called when the analysis worker has finished.
        It updates the GUI with the results using pyqtgraph.
        """
        estimated_key = analyser.estimate_key(analyser.chromagram)
        
        self.file_label.setText(f"Loaded: {analyser.file_path.split('/')[-1]}")
        self.bpm_label.setText(f"BPM: {np.median(analyser.bpm):.2f}")
        self.key_label.setText(f"Key: {estimated_key}")
        self.time_signature_label.setText(f"Time Signature: {analyser.time_signature}")

        # --- Clear and update plots ---
        self.waveform_plot.clear()
        self.chromagram_plot.clear()

        # --- Add playback lines back to plots after clearing ---
        self.waveform_plot.addItem(self.waveform_line, ignoreBounds=True)
        self.chromagram_plot.addItem(self.chromagram_line, ignoreBounds=True)
        self.waveform_line.setPos(0)
        self.chromagram_line.setPos(0)

        # --- Plot 1: Waveform ---
        time_axis = np.linspace(0, len(analyser.y) / analyser.sr, num=len(analyser.y))
        self.waveform_plot.plot(time_axis, analyser.y, pen=pg.mkPen(color='#5A2A82', width=1))

        # --- Plot 2: Chromagram ---
        img = pg.ImageItem(image=analyser.chromagram)
        self.chromagram_plot.addItem(img)

        # Set y-axis ticks for chromagram
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        y_axis = self.chromagram_plot.getAxis('left')
        ticks = [(i, pitch) for i, pitch in enumerate(pitch_classes)]
        y_axis.setTicks([ticks])
        
        # Set the color map for the image
        cmap = pg.colormap.get('viridis')
        img.setLookupTable(cmap.getLookupTable())
        
        # Scale the image correctly
        img.setRect(0, 0, time_axis[-1], 12)
        self.chromagram_plot.setYRange(0, 12)
        self.chromagram_plot.setXRange(0, time_axis[-1])

        self.current_file = analyser.file_path
        self.playback_worker.load_file(self.current_file)

    def play_audio(self):
        if self.current_file:
            self.playback_worker.play()
            self.waveform_line.show()
            self.chromagram_line.show()
        else:
            self.show_error_dialog("No audio file loaded.")

    def pause_audio(self):
        self.playback_worker.pause()

    def stop_audio(self):
        self.playback_worker.stop()
        self.waveform_line.setPos(0)
        self.chromagram_line.setPos(0)
        self.waveform_line.hide()
        self.chromagram_line.hide()

    def update_playback_position(self, pos_s):
        self.waveform_line.setPos(pos_s)
        self.chromagram_line.setPos(pos_s)

    def on_playback_finished(self):
        self.stop_audio()

    def analysis_error(self, error_tuple):
        """Handles errors from the worker thread."""
        error_message = f"An error occurred: {error_tuple[1]}"
        self.show_error_dialog(error_message)
        self.file_label.setText("Error processing file.")
        self.bpm_label.setText("BPM: --")
        self.key_label.setText("Key: --")
        self.time_signature_label.setText("Time Signature: --")
        self.waveform_plot.clear()
        self.chromagram_plot.clear()

    def analysis_finished(self):
        """Called when the worker thread has finished."""
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)

    def show_error_dialog(self, message):
        """Displays an error message in a dialog box."""
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Error")
        dlg.setText(message)
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.exec()

    def closeEvent(self, event):
        self.playback_thread.quit()
        self.playback_thread.wait()
        super().closeEvent(event)

# --- Main execution block ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # --- Dark Theme with Purple Accents ---
    app.setStyleSheet("""
        QWidget {
            background-color: #2E2E2E;
            color: #F0F0F0;
            font-family: Arial, sans-serif;
        }
        QMainWindow {
            background-color: #252525;
        }
        QPushButton {
            background-color: #5A2A82;
            color: #FFFFFF;
            border: 1px solid #6A3A92;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #6A3A92;
        }
        QPushButton:pressed {
            background-color: #4A1A72;
        }
        QPushButton:disabled {
            background-color: #404040;
            color: #808080;
        }
        QLabel {
            color: #E0E0E0;
            padding: 2px;
        }
        QProgressBar {
            border: 1px solid #5A2A82;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #5A2A82;
            width: 10px; 
            margin: 0.5px;
        }
    """)

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())