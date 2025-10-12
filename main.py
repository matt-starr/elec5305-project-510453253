import sys
import librosa
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, 
    QWidget, QFileDialog, QLabel
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """A custom Matplotlib canvas widget to embed in a PyQt6 application."""
    def __init__(self, parent=None, width=5, height=6, dpi=100):
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

        # --- Central Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # --- Load File Button ---
        self.load_button = QPushButton("Load Audio File")
        # Connect the button's 'clicked' signal to the open_file_dialog method
        self.load_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.load_button)

        # --- File Info Label ---
        self.file_label = QLabel("No file loaded.")
        layout.addWidget(self.file_label)
        
        # --- Matplotlib Display ---
        # This is the canvas where the plots will be drawn
        self.canvas = MplCanvas(self, width=80, height=60, dpi=100)
        layout.addWidget(self.canvas)

    def open_file_dialog(self):
        """
        Opens a file dialog to allow the user to select an audio file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",  # Start directory
            "Audio Files (*.wav *.mp3);;All Files (*)"
        )

        if file_path:
            self.file_label.setText(f"Loaded: {file_path.split('/')[-1]}")
            self.plot_features(file_path)

    def plot_features(self, file_path):
        """
        Loads an audio file and plots its waveform and chromagram.

        Args:
            file_path (str): The path to the audio file.
        """
        try:
            y, sr = librosa.load(file_path)
            
            # --- Clear previous plots ---
            self.canvas.axes1.cla()
            self.canvas.axes2.cla()

            # --- Plot 1: Waveform ---
            librosa.display.waveshow(y, sr=sr, ax=self.canvas.axes1, color='blue', alpha=0.5)
            self.canvas.axes1.set_title("Waveform")
            self.canvas.axes1.set_xlabel(None) # Remove x-label to avoid overlap
            self.canvas.axes1.set_ylabel("Amplitude")
            self.canvas.axes1.grid(True)
            
            # --- Plot 2: Chromagram ---
            # Extract the chromagram
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            # Display the chromagram
            img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=self.canvas.axes2)
            self.canvas.axes2.set_title("Chromagram")
            self.canvas.axes2.set_xlabel("Time (s)")
            self.canvas.axes2.set_ylabel("Pitch Class")

            # Adjust layout to prevent titles/labels from overlapping
            # self.canvas.figure.tight_layout()
            
            # Redraw the canvas to show the new plots
            self.canvas.draw()
            
        except Exception as e:
            self.file_label.setText(f"Error loading file: {e}")
            print(f"Error: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
