# ELEC5305 Music Analyser GUI

**Author:** Matt Starr (510453253)  
**Course:** ELEC5305 - University of Sydney  
**Status:** In Development

---

### Description

This project is a desktop GUI application designed for musicians and music enthusiasts. It allows users to load an audio file (`.wav`, `.mp3`) and analyses it to extract and display key musical features. The goal is to provide a simple, open-source tool to help with music transcription and analysis, automating tasks like finding a song's key and tempo.
W


### Features

* ✅ **Audio File Loading:** Load and process common audio formats (`.wav`, `.mp3`).
* ✅ **Waveform Display:** Visualize the amplitude waveform of the entire track.
* ✅ **Chromagram Plot:** Display a chromagram to show the intensity of pitch classes over time.
* ✅ **BPM Estimation:** Automatically calculate the song's tempo in Beats Per Minute.
* ✅ **Key Estimation:** Predict the musical key of the song (e.g., C# Minor, A Major).
* ⏳ **Instrument Identification:** (Planned Feature) Future versions will include a machine learning model to predict the instruments present in the song.

---

### Requirements

* **Backend:** Python 3.9+
* **GUI Framework:** PyQt6
* **Audio Analysis:** `librosa`
* **Plotting:** `matplotlib`
* **Numerical Operations:** `numpy`
* **Package Management:** `uv`

---

### Prerequisites

* Python 3.9 or newer.
* [uv](https://github.com/astral-sh/uv) - an extremely fast Python package installer and resolver.

### Installation and Usage

These instructions assume you have `uv` installed and available in your system's PATH.

**1. Clone the Repository**

First, clone the project to your local machine.

```bash
git clone [https://github.com/matt-starr/elec5305-project-510453253.git](https://github.com/matt-starr/elec5305-project-510453253.git)
cd elec5305-project-510453253
```

**2. Set up the virtual environment and dowload dependencies**

```bash
uv sync
```

**3. Run the application**

Execute the main script using `uv run`. This command runs the script within the managed virtual environment.

```bash
uv run main.py
```
