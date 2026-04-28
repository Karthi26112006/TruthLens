# TruthLens
TruthLens is a sophisticated media verification system designed to combat the rising tide of digital misinformation. By leveraging deep learning and computer vision, the platform identifies manipulated content—specifically deepfakes—to ensure the authenticity of digital media.

# Refinement: 
Paste the content here, and I can polish the language to make it sound more professional and technically "sharp."

# Feature Expansion: 
If you've added new modules (like a specific detection algorithm or a new database schema), I can help you document those sections.

# Visual Enhancements: 
I can suggest how to structure your Installation and Usage sections using clear Markdown blocks so anyone can run your code easily.

# Project Context: 
If you're preparing this for a specific presentation or repository, I can help tailor the "Future Scope" section to impress the reviewers.

## Project Structure

```text
TruthLens/
├── app.py                # Main Flask application & AI model integration
├── forensics.py          # Core forensic analysis (EXIF, OpenCV, FFT)
├── LICENSE               # MIT License file
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies
│
├── static/               # Static assets
│   ├── css/
│   │   └── styles.css    # Cyber-themed UI styling
│   ├── img/              # Backgrounds and icons
│   └── uploads/          # Temporary storage for analyzed media
│
├── templates/            # Flask HTML templates
│   ├── base.html         # Main layout wrapper
│   ├── index.html        # Upload landing page
│   ├── processing.html   # Neural network analysis animation
│   └── results.html      # Forensic report dashboard
│
└── tests/                # Verification scripts
    ├── test_dima806.py   # Testing the primary AI classifier
    └── test_model.py     # Testing fallback detection models
