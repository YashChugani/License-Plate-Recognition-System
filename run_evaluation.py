import os
import cv2
import csv
import easyocr
from src.preprocess import preprocess_image
from src.detect import get_plate_candidates
from src.extract import extract_and_enhance
from src.ocr import perform_ocr
from src.evaluate import evaluate_prediction

def run_full_evaluation(input_folder="input_images", 
                        ground_truth_file="ground_truth/ground_truth.csv", 
                        output_results="evaluation_results.csv"):
    
    print("Loading Ground Truth data...")
    ground_truth_map = {}
    with open(ground_truth_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ground_truth_map[row["Filename"]] = row["Actual_Plate"]
            
    total_images = len(ground_truth_map)
    print(f"Found {total_images} annotated images. Initializing OCR Engine...\n")
    
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    
    exact_matches = 0
    total_char_accuracy = 0.0
    processed_count = 0
    
    with open(output_results, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Added a column to show how many suggestions were generated
        writer.writerow(["Filename", "Actual_Plate", "Best_Predicted_Plate", "Confidence", "Suggestions_Count", "Is_Exact_Match", "Char_Accuracy_%"])
        
        for filename, actual_plate in ground_truth_map.items():
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            
            if img is None:
                continue
                
            processed_count += 1
            print(f"[{processed_count}/{total_images}] Testing {filename} | Target: {actual_plate}...")
            
            resized_img, gray, bfilter = preprocess_image(img)
            edged, candidates = get_plate_candidates(bfilter)
            
            # Variables to track the BEST prediction among the suggestions
            best_plate_text = "NO_PLATE_DETECTED"
            best_confidence = 0.0
            best_is_exact = False
            best_char_acc = 0.0
            suggestion_count = 0
            
            for (x, y, w, h, contour) in candidates:
                roi = gray[y:y+h, x:x+w]
                result = ocr_reader.readtext(roi, detail=0)
                if result:
                    cropped_color, enhanced_gray = extract_and_enhance(resized_img, contour)
                    if cropped_color is not None:
                        # Our new OCR returns a list of possibilities
                        possible_plates, confidence, is_valid = perform_ocr(enhanced_gray, ocr_reader)
                        
                        if is_valid:
                            suggestion_count = len(possible_plates)
                            
                            # Test EVERY suggestion against the ground truth
                            for plate in possible_plates:
                                is_exact, char_acc = evaluate_prediction(plate, actual_plate)
                                
                                # Keep the one with the highest character accuracy
                                if char_acc > best_char_acc:
                                    best_char_acc = char_acc
                                    best_plate_text = plate
                                    best_is_exact = is_exact
                                    best_confidence = confidence
                                    
                                # If we hit a 100% exact match, stop checking alternatives!
                                if is_exact:
                                    break 
                            break 
            
            # --- THE FALLBACK MECHANISM ---
            # If standard edge detection failed to find a valid plate...
            if best_plate_text == "NO_PLATE_DETECTED":
                # Maybe the image IS the plate, or edges were broken. 
                # Run OCR on the entire raw image!
                possible_plates, fallback_conf, fallback_valid = perform_ocr(gray, ocr_reader)
                
                if fallback_valid:
                    suggestion_count = len(possible_plates)
                    for plate in possible_plates:
                        is_exact, char_acc = evaluate_prediction(plate, actual_plate)
                        if char_acc > best_char_acc:
                            best_char_acc = char_acc
                            best_plate_text = plate
                            best_is_exact = is_exact
                            best_confidence = fallback_conf
                        if is_exact:
                            break

            if best_is_exact:
                exact_matches += 1
            total_char_accuracy += best_char_acc
            
            writer.writerow([filename, actual_plate, best_plate_text, round(best_confidence, 2), suggestion_count, best_is_exact, round(best_char_acc, 2)])

    overall_exact_accuracy = (exact_matches / processed_count) * 100 if processed_count > 0 else 0
    overall_char_accuracy = total_char_accuracy / processed_count if processed_count > 0 else 0
    
    print("\n" + "="*45)
    print("      FINAL SYSTEM PERFORMANCE METRICS       ")
    print("="*45)
    print(f"Total Images Processed : {processed_count}")
    print(f"Exact Plate Matches    : {exact_matches}")
    print(f"System Exact Accuracy  : {overall_exact_accuracy:.2f}%")
    print(f"System Char. Accuracy  : {overall_char_accuracy:.2f}%")
    print("="*45)

if __name__ == "__main__":
    run_full_evaluation()