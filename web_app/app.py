import os
import sys
import cv2
import easyocr
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Add the root project directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_image
from src.detect import get_plate_candidates
from src.extract import extract_and_enhance
from src.ocr import perform_ocr

app = Flask(__name__)

# Configure where to save uploaded images so the HTML can display them
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)

print("Initializing Web OCR Engine...")
reader = easyocr.Reader(['en'], gpu=False)


# Route 1: The Landing Page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Route 2: The Application Page
@app.route('/app', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        # 1. Handle the file upload
        if 'file' not in request.files:
            return render_template('app.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('app.html', error="No file selected")

        if file:
            # 2. Save the original image securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 3. Run the ALPR pipeline
            img = cv2.imread(filepath)
            if img is None:
                return render_template('app.html', error="Could not read the image file")

            resized_img, gray, bfilter = preprocess_image(img)
            edged, candidates = get_plate_candidates(bfilter)

            # Initialize default fallback values BEFORE the loop
            possible_plates = ["PLATE_NOT_FOUND"] 
            confidence = 0.0
            is_valid = False
            cropped_filename = None

            # 1. Primary Method: Scan mathematically detected candidates
            for (x, y, w, h, contour) in candidates:
                roi = gray[y:y+h, x:x+w]
                result = reader.readtext(roi, detail=0)
                if result:
                    cropped_color, enhanced_gray = extract_and_enhance(resized_img, contour)
                    if cropped_color is not None:
                        possible_plates, confidence, is_valid = perform_ocr(enhanced_gray, reader)
                        if is_valid:
                            cropped_filename = f"crop_{filename}"
                            crop_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], cropped_filename)
                            cv2.imwrite(crop_path, cropped_color)
                            break 
                            
            # 2. THE FALLBACK MECHANISM
            # If the primary method failed
            if not is_valid:
                fallback_plates, fallback_conf, fallback_valid = perform_ocr(gray, reader)
                if fallback_valid:
                    possible_plates = fallback_plates
                    confidence = fallback_conf
                    is_valid = True
                    cropped_filename = f"fallback_{filename}"
                    crop_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], cropped_filename)
                    cv2.imwrite(crop_path, resized_img)

            # 3. Render the page with the final results
            return render_template('app.html', 
                                   original_image=filename,
                                   cropped_image=cropped_filename,
                                   possible_plates=possible_plates,
                                   confidence=round(confidence * 100, 2),
                                   is_valid=is_valid)

    # For a standard GET request, just show the empty form
    return render_template('app.html')

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)