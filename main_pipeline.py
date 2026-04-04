import cv2
import os
import easyocr
import re
from src.preprocess import preprocess_image
from src.detect import get_plate_candidates
from src.extract import extract_and_enhance
from src.segment import segment_characters
from src.ocr import perform_ocr

print("Initializing OCR Engine...")
reader = easyocr.Reader(['en'], gpu=False)

def run_pipeline(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    resized_img, gray, bfilter = preprocess_image(img)
    edged, candidates = get_plate_candidates(bfilter)

    plate_found = False
    final_contour = None

    for (x, y, w, h, contour) in candidates:
        roi = gray[y:y+h, x:x+w]
        result = reader.readtext(roi, detail=0)
        text = "".join(result).upper()
        text = re.sub(r'[^A-Z0-9]', '', text) 
        
        if len(text) >= 4:
            final_contour = contour
            plate_found = True
            break

    if plate_found:
        cv2.drawContours(resized_img, [final_contour], -1, (0, 255, 0), 3)
        cropped_color, enhanced_gray = extract_and_enhance(resized_img, final_contour)
        
        if cropped_color is not None:
            # Stage 4: Segmentation (Visual only)
            thresh, char_boxes = segment_characters(enhanced_gray)
            for (cx, cy, cw, ch) in char_boxes:
                cv2.rectangle(cropped_color, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
            
            # Stage 5: OCR & Validation
            possible_plates, confidence, is_valid = perform_ocr(enhanced_gray, reader)
            
            print("\n" + "="*40)
            print("        ALPR EXTRACTION RESULTS        ")
            print("="*40)
            
            # If we generated multiple suggestions, print them all
            if len(possible_plates) > 1:
                print("AMBIGUOUS STATE CODE DETECTED.")
                print("Suggested Valid Combinations:")
                for plate in possible_plates:
                    print(f" -> {plate}")
            else:
                print(f"Detected Plate Number : {possible_plates[0]}")
                
            print(f"Confidence Score      : {confidence * 100:.2f}%")
            print(f"Valid Indian Format   : {is_valid}")
            print("="*40 + "\n")

    else:
        print("Failed: Could not detect plate.")

    print("Press any key on the image windows to close them.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_filename = input("Enter the image filename (e.g., vehicle_0001.jpg): ")
    path_to_image = os.path.join("input_images", image_filename)
    
    if os.path.exists(path_to_image):
        run_pipeline(path_to_image)
    else:
        print(f"File not found at {path_to_image}")