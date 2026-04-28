import ssl
from transformers import pipeline
from PIL import Image
import numpy as np

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Loading model...")
ai_classifier = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
print("Model loaded.")

img_obj = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
result = ai_classifier(img_obj)
print("Result:", result)
