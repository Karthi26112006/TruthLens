import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def check_exif_data(filepath):
    """
    Checks for the presence and validity of EXIF metadata.
    AI-generated images typically lack camera metadata (e.g., Make, Model, DateTimeOriginal).
    """
    anomalies = []
    has_critical_exif = False
    
    try:
        img = Image.open(filepath)
        exif_info = img.getexif()
        
        if not exif_info:
            anomalies.append("No EXIF metadata found. Highly suspicious for original photographs (Possible AI/Deepfake).")
            return False, anomalies
            
        exif_data = {}
        for tag_id, value in exif_info.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_data[tag] = value
            
        # Critical tags that usually indicate a real camera picture
        critical_tags = ['Make', 'Model', 'DateTimeOriginal', 'GPSInfo']
        found_tags = [tag for tag in critical_tags if tag in exif_data]
        
        if len(found_tags) == 0:
            anomalies.append("Missing camera signatures (Make/Model). Image might be synthetic or heavily scrubbed.")
        else:
            has_critical_exif = True
            
        # Check for software manipulation signatures
        if 'Software' in exif_data:
            software = str(exif_data['Software']).lower()
            if any(ai_tool in software for ai_tool in ['midjourney', 'dall-e', 'stable diffusion', 'adobe']):
                anomalies.append(f"Editing/AI Software detected in EXIF: {exif_data['Software']}")
                has_critical_exif = False
                
    except Exception as e:
        anomalies.append(f"Error parsing metadata: {str(e)}")
        
    return has_critical_exif, anomalies

def analyze_edges_opencv(filepath):
    """
    Uses OpenCV for forensic edge and color analysis.
    AI images often have unusual symmetry, uniform blur, or strange edge distributions.
    """
    anomalies = []
    is_unnatural = False
    
    try:
        # Load image with OpenCV
        img = cv2.imread(filepath)
        if img is None:
            return False, ["Could not read image for OpenCV forensics analysis."]
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (Blur/Focus check)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            anomalies.append(f"Extremely low edge variance ({laplacian_var:.1f}). Overly smooth textures often found in AI generation.")
            is_unnatural = True
        elif laplacian_var > 3000:
            anomalies.append(f"Unusually high high-frequency noise ({laplacian_var:.1f}). Possible adversarial or GAN noise injection.")
            is_unnatural = True
            
        # 2. Color Histogram Analysis (Detecting unnatural color distributions)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        saturation_variance = np.var(hist_s)
        
        if saturation_variance > 1e7:
            anomalies.append("Unnatural hyper-saturation peaks detected. Color palette looks synthetic.")
            is_unnatural = True
            
    except Exception as e:
        pass
        
    return is_unnatural, anomalies

def detect_deepfake(filepath):
    """
    Wrapper for deepfake-detector library.
    Identifies manipulated media/faces.
    """
    try:
        from deepfake_detector import image_prediction
        # Predict uses a model to return a score between 0 and 1
        result = image_prediction(filepath)
        score = result.get('score', 0)
        if score > 0.6:
            return True, [f"deepfake-detector flagged manipulation (Confidence: {score*100:.1f}%)"]
        return False, []
    except ImportError:
        # Graceful fallback if library isn't installed or fails
        anomalies = []
        return False, anomalies
    except Exception as e:
        print(f"Deepfake detector warning: {e}")
        return False, []

def check_ai_or_not(filepath):
    """
    Simulated implementation of AI-Or-Not API or lightweight library.
    Flags synthetic content based on lightweight heuristics.
    """
    # In a real environment, this would be an API call to aiornot.com or an internal ONNX model
    try:
        img = Image.open(filepath)
        # Lightweight heuristic: pure square aspect ratios exactly 512x512, 1024x1024 are strongly tied to default AI outputs
        width, height = img.size
        # Known common AI output resolutions
        ai_resolutions = [(512, 512), (1024, 1024), (1024, 1536), (1536, 1024), (768, 768)]
        
        if (width, height) in ai_resolutions:
            return True, [f"AI-Or-Not Heuristic: Found exact default AI generation canvas size {width}x{height}."]
            
        return False, []
    except Exception:
        return False, []

def comprehensive_image_scan(filepath):
    """
    Runs all forensic modules and returns an aggregated score and anomalies.
    Provides evidence based on EXIF, OpenCV forensics, deepfake-detector, and ai-or-not.
    """
    all_anomalies = []
    fake_probability = 0
    
    # 1. EXIF Analysis
    has_exif, exif_anomalies = check_exif_data(filepath)
    if not has_exif:
        fake_probability += 30
    if exif_anomalies:
        all_anomalies.extend(exif_anomalies)
        
    # 2. OpenCV Forensics
    is_unnatural, cv_anomalies = analyze_edges_opencv(filepath)
    if is_unnatural:
        fake_probability += 25
    if cv_anomalies:
        all_anomalies.extend(cv_anomalies)
        
    # 3. Deepfake Library Detection
    is_deepfake, df_anomalies = detect_deepfake(filepath)
    if is_deepfake:
        fake_probability += 40
    if df_anomalies:
        all_anomalies.extend(df_anomalies)
        
    # 4. AI-Or-Not 
    is_ai, ai_anomalies = check_ai_or_not(filepath)
    if is_ai:
        fake_probability += 20
    if ai_anomalies:
        all_anomalies.extend(ai_anomalies)
        
    fake_probability = min(99, fake_probability)
    
    if fake_probability > 60:
        status = "AI MODIFIED"
    elif fake_probability > 30:
        status = "EDITED"
    else:
        status = "REAL"
        
    return fake_probability, status, all_anomalies
