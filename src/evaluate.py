def calculate_levenshtein_distance(s1, s2):
    """
    Calculates the minimum number of single-character edits (insertions, 
    deletions, or substitutions) required to change string s1 into string s2.
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def evaluate_prediction(predicted_plate, actual_plate):
    """
    Compares the predicted plate against the ground truth and returns 
    the exact match boolean and the character-level accuracy percentage.
    """
    # Clean up both strings just in case
    predicted = str(predicted_plate).strip().upper()
    actual = str(actual_plate).strip().upper()
    
    # 1. Exact Match Metric
    is_exact_match = (predicted == actual)
    
    # 2. Character Accuracy Metric
    max_len = max(len(predicted), len(actual))
    
    if max_len == 0:
        return is_exact_match, 0.0
        
    distance = calculate_levenshtein_distance(predicted, actual)
    
    # Calculate percentage of characters correctly identified
    character_accuracy = ((max_len - distance) / max_len) * 100.0
    
    # Ensure it doesn't drop below 0% for completely mangled predictions
    character_accuracy = max(0.0, character_accuracy)
    
    return is_exact_match, character_accuracy