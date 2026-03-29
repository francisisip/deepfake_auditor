# Deepfake Auditor

Welcome to the Deepfake Auditor. This project is a forensics tool designed to check if a video is an authentic recording or a deepfake.

Instead of just giving a yes or no answer (acting like a black box), it tries to explain its reasoning. It does this by showing you:
1. **Grad-CAM heatmaps**: To show exactly which parts of a face triggered the deepfake alert.
2. **Frequency Analysis (FFT)**: To spot weird upsampling patterns that sometimes happen when AI generates faces.
3. **Ablation Testing**: To double-check its own work by blurring the suspected fake areas and seeing if its confidence drops significantly.

## How to Run

You'll need Python 3.11 for this since it's what we tested on.

First, set up your virtual environment so you don't mess up your system packages:
```bash
python3.11 -m venv torch-env
source torch-env/bin/activate
```

*(Note: If you're on Windows, use `torch-env\Scripts\activate` instead).*

Make sure you have all the necessary packages installed:
```bash
pip install -r requirements.txt
```

Then, start the Streamlit web app:
```bash
python3 -m streamlit run app/app.py
```

Once it's running, it should pop open in your browser. Just upload a video file and hit analyze.