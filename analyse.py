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
            self.bpm = librosa.feature.tempo(y=self.y, sr=self.sr)[0]
        else:
            print("Audio not loaded. Please call load_audio() first.")

    def extract_chromagram(self):
        """
        Creates a chromagram from the audio signal.

        A chromagram represents the 12 different pitch classes (C, C#, D, etc.)
        """
        if self.y is not None:
            # librosa.feature.chroma_stft computes a chromagram from a waveform.
            self.chromagram = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        else:
            print("Audio not loaded. Please call load_audio() first.")
            
    def estimate_key(self, chroma_features):
        """
        Estimates the key from chromagram features.
        """
        # Pitch class names
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Define major and minor key profiles
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        # Sum chroma features across time to get an overall pitch distribution
        chroma_sum = np.sum(chroma_features, axis=1)
        
        correlations = []
        # Correlate with all 12 major and 12 minor keys
        for i in range(12):
            # Major keys
            major_key_profile = np.roll(major_profile, i)
            correlations.append(np.corrcoef(chroma_sum, major_key_profile)[0, 1])
            # Minor keys
            minor_key_profile = np.roll(minor_profile, i)
            correlations.append(np.corrcoef(chroma_sum, minor_key_profile)[0, 1])
            
        # Find the best match
        best_match_index = np.argmax(correlations)
        key_index = best_match_index // 2
        is_major = best_match_index % 2 == 0
        
        key = pitches[key_index]
        mode = "Major" if is_major else "Minor"
        
        return f"{key} {mode}"
    
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
    
    try:
        song_name = "Come Away With Me - Norah Jones.mp3"
        analyser = AudioAnalyser(f'media/{song_name}')
        analyser.run_analysis()
    except Exception as e:
        print(f"\nError: Could not find or process the audio file.")
        print(f"Error details: {e}")