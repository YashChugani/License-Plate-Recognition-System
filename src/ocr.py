import re

# Dictionaries for common OCR mistakes
char_to_int = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'T': '7'}
int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B', '7': 'T'}

# Official Indian State & UT Codes
VALID_STATES = ["AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TG", "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY", "BH", "TS"]

def correct_characters(text):
    """
    Applies positional logic to fix common OCR mistakes based on standard Indian formats.
    """
    corrected = ""
    
    # Check for standard 10-character format (e.g., MH12AB1234)
    if len(text) == 10:
        for i, char in enumerate(text):
            if i in [0, 1, 4, 5]: # Must be Letters
                corrected += int_to_char.get(char, char)
            elif i in [2, 3, 6, 7, 8, 9]: # Must be Numbers
                corrected += char_to_int.get(char, char)
            else:
                corrected += char
        return corrected
        
    # Check for 9-character format (e.g., MH12A1234)
    elif len(text) == 9:
        for i, char in enumerate(text):
            if i in [0, 1, 4]: # Must be Letters
                corrected += int_to_char.get(char, char)
            elif i in [2, 3, 5, 6, 7, 8]: # Must be Numbers
                corrected += char_to_int.get(char, char)
            else:
                corrected += char
        return corrected
        
    return text

def get_state_suggestions(ocr_state):
    """Finds valid state codes that are only 1 character different from the OCR typo."""
    suggestions = []
    for valid_state in VALID_STATES:
        # Calculate simple character differences (e.g., 'HH' vs 'MH' = 1 difference)
        diff = sum(1 for a, b in zip(ocr_state, valid_state) if a != b)
        if diff == 1:
            suggestions.append(valid_state)
    return suggestions

def extract_indian_plate(text):
    """
    Extracts the plate and returns a LIST of possible plates if the state code is ambiguous.
    """
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    pattern_standard = r'^([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})$'
    pattern_bh = r'^([0-9]{2}BH[0-9]{4}[A-Z]{1,2})$'
    
    possible_plates = []
    
    for window_size in [10, 9]:
        for i in range(len(clean_text) - window_size + 1):
            chunk = clean_text[i : i + window_size]
            corrected_chunk = correct_characters(chunk)
            
            # Check if it matches Bharat series (skips state code check)
            if re.match(pattern_bh, corrected_chunk):
                return [corrected_chunk], True
                
            # Check standard series
            if re.match(pattern_standard, corrected_chunk):
                state_code = corrected_chunk[:2]
                
                # If the state code is perfect, return just that one
                if state_code in VALID_STATES:
                    return [corrected_chunk], True
                
                # If state code is invalid, generate alternatives
                suggestions = get_state_suggestions(state_code)
                if suggestions:
                    for valid_state in suggestions:
                        # Replace the typo with the valid state suggestion
                        alt_plate = valid_state + corrected_chunk[2:]
                        possible_plates.append(alt_plate)
                    return possible_plates, True
                else:
                    # If it's totally mangled but matches regex, return it anyway
                    return [corrected_chunk], True

    return [clean_text], False

def perform_ocr(enhanced_plate, reader):
    results = reader.readtext(enhanced_plate)
    if not results:
        return ["NO_TEXT_FOUND"], 0.0, False
        
    raw_text = ""
    total_confidence = 0.0
    for (bbox, text, prob) in results:
        raw_text += text
        total_confidence += prob
    avg_confidence = total_confidence / len(results)
    
    # this now returns a list of possible plates
    possible_plate_texts, is_valid = extract_indian_plate(raw_text)
    
    return possible_plate_texts, avg_confidence, is_valid