from transformers import pipeline
from PIL import Image
import numpy as np

try:
    print("Loading model...")
    pipe = pipeline('image-classification', model='prithivMLmods/Deep-Fake-Detector-Model')
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    print(pipe(img))
except Exception as e:
    print("Error:", e)
