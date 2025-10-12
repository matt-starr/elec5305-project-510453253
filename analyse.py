import librosa
import numpy as np

class AudioAnalyser:
    """
    A class to analyse audio files and extract musical features.

    Attributes:
        file_path (str): The path to the audio file.
        y (np.ndarray): The audio time series (waveform).
        sr (int): The sample rate of the audio.
        bpm (float): The estimated tempo in beats per minute.
        chromagram (np.ndarray): The chromagram of the audio.
    """

    def __init__(self, file_path):
        """
        Initializes the AudioAnalyser with the path to an audio file.

        Args:
            file_path (str): The full path to the .wav or .mp3 file.
        """
        self.file_path = file_path
        self.y = None
        self.sr = None
        self.bpm = 0.0
        self.chromagram = None

    def load_audio(self):
        """
        Loads the audio file into a numpy array.

        Handles potential errors if the file cannot be loaded.
        """
        try:
            # librosa.load reads the audio file and returns the waveform (y)
            # and the sample rate (sr).
            self.y, self.sr = librosa.load(self.file_path)
            print(f"Successfully loaded {self.file_path}")
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def extract_bpm(self):
        """
        Extracts the beats per minute (BPM) from the audio.

        This must be called after load_audio().
        """
        if self.y is not None:
            # librosa.beat.tempo estimates the tempo from the audio signal.
            # It returns an array, so we take the first element.
            self.bpm = librosa.feature.tempo(y=self.y, sr=self.sr)[0]
        else:
            print("Audio not loaded. Please call load_audio() first.")

    def extract_chromagram(self):
        """
        Creates a chromagram from the audio signal.

        A chromagram represents the 12 different pitch classes (C, C#, D, etc.)
        and is very useful for key detection. This must be called after load_audio().
        """
        if self.y is not None:
            # librosa.feature.chroma_stft computes a chromagram from a waveform.
            self.chromagram = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        else:
            print("Audio not loaded. Please call load_audio() first.")
            
    def run_analysis(self):
        """
        Runs the full analysis pipeline.
        """
        if self.load_audio():
            self.extract_bpm()
            self.extract_chromagram()
            self.print_results()

    def print_results(self):
        """
        Prints the results of the analysis to the console.
        """
        print("\n--- Analysis Results ---")
        if self.bpm is not None:
            print(f"Estimated BPM: {self.bpm:.2f}")
        
        if self.chromagram is not None:
            # The chromagram is a 2D array (12 pitch classes x time frames).
            # For a simple summary, we can average it over time to see the
            # overall energy of each pitch class.
            mean_chroma = np.mean(self.chromagram, axis=1)
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            print("\nChromagram (average energy per pitch class):")
            for i, pitch in enumerate(pitch_classes):
                print(f"{pitch:<3}: {mean_chroma[i]:.4f}")
        print("------------------------\n")


if __name__ == '__main__':
    # This is an example of how to use the class.
    # To run this, you must have an audio file (e.g., 'song.wav') in the same
    # directory as this script, or provide the full path to a file.
    
    # IMPORTANT: Replace 'path/to/your/song.wav' with an actual file path.
    # If you don't have a WAV file, you can install `ffmpeg` and librosa
    # will be able to open MP3s and other formats as well.
    try:
        song_name = "Come Away With Me - Norah Jones.mp3"
        analyser = AudioAnalyser(f'media/{song_name}') # <-- REPLACE WITH YOUR FILE
        analyser.run_analysis()
    except Exception as e:
        print(f"\nError: Could not find or process the audio file.")
        print(f"Error details: {e}")