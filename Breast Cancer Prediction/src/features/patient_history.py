import json
import os
from datetime import datetime

PATIENT_HISTORY_FILE = "data/patient_history.json"

def _initialize_history_file(filepath):
    """Initializes the patient history JSON file if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump({}, f) # Empty dictionary to start with

def load_patient_history(filepath=PATIENT_HISTORY_FILE):
    """
    Loads patient history from a JSON file.
    Returns a dictionary of patient records.
    """
    _initialize_history_file(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)

def save_patient_history(data, filepath=PATIENT_HISTORY_FILE):
    """
    Saves patient history data to a JSON file.
    data: Dictionary of patient records.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def add_mammogram_result(patient_id, result, filepath=PATIENT_HISTORY_FILE):
    """
    Adds a new mammogram result to a patient's history.
    patient_id: Unique identifier for the patient.
    result: A dictionary containing the current mammogram's predictions and metadata.
            Expected keys: 'date', 'cancer_probability', 'risk_category', 'severity_stage', 'birads_score', 'confidence'.
            'date' should be in ISO format (e.g., '2026-01-08T12:30:00').
    """
    history = load_patient_history(filepath)
    if patient_id not in history:
        history[patient_id] = []
    history[patient_id].append(result)
    # Sort results by date to maintain chronological order
    history[patient_id].sort(key=lambda x: datetime.fromisoformat(x['date']))
    save_patient_history(history, filepath)

def get_patient_timeline(patient_id, filepath=PATIENT_HISTORY_FILE):
    """
    Retrieves all mammogram results for a given patient, sorted by date.
    Returns a list of result dictionaries.
    """
    history = load_patient_history(filepath)
    return history.get(patient_id, [])

def compare_results(patient_id, current_result, filepath=PATIENT_HISTORY_FILE):
    """
    Compares the current mammogram result with the most recent past result for a patient.
    Returns a dictionary summarizing the comparison or None if no past results exist.
    """
    timeline = get_patient_timeline(patient_id, filepath)
    if len(timeline) < 1:
        return None # No past results to compare with
    
    # The most recent past result will be the last one in the sorted timeline (excluding current if it's already added).
    # For this function, assume current_result is not yet added to the timeline.
    last_result = timeline[-1]
    
    comparison = {
        "previous_date": last_result.get('date'),
        "current_date": current_result.get('date'),
        "cancer_probability_change": current_result.get('cancer_probability') - last_result.get('cancer_probability'),
        "risk_category_change": None, # Categorical, need mapping or specific logic
        "severity_stage_change": current_result.get('severity_stage') - last_result.get('severity_stage'),
        "birads_score_change": current_result.get('birads_score') - last_result.get('birads_score'),
        "overall_trend": "Stable" # Placeholder, will be determined by changes
    }

    # More sophisticated logic for categorical changes and overall trend needed
    # Example for risk category: if current is higher than previous, then worsening
    # For simplicity, just showing numerical diffs for now.
    if comparison["cancer_probability_change"] > 0.05: # Arbitrary threshold
        comparison["overall_trend"] = "Worsening (Increased Cancer Probability)"
    elif comparison["cancer_probability_change"] < -0.05:
        comparison["overall_trend"] = "Improving (Decreased Cancer Probability)"

    return comparison

def calculate_trends(patient_id, filepath=PATIENT_HISTORY_FILE):
    """
    Calculates health trends based on a patient's entire mammogram history.
    Returns a dictionary of trends (e.g., average change, min/max values).
    """
    timeline = get_patient_timeline(patient_id, filepath)
    if len(timeline) < 2:
        return {"message": "Not enough data to calculate trends."}
    
    trends = {
        "cancer_probability_trend": [],
        "severity_stage_trend": [],
        "birads_score_trend": [],
        # Add other trends as needed
    }

    for i in range(1, len(timeline)):
        prev = timeline[i-1]
        curr = timeline[i]

        if 'cancer_probability' in prev and 'cancer_probability' in curr:
            trends["cancer_probability_trend"].append(curr['cancer_probability'] - prev['cancer_probability'])
        if 'severity_stage' in prev and 'severity_stage' in curr:
            trends["severity_stage_trend"].append(curr['severity_stage'] - prev['severity_stage'])
        if 'birads_score' in prev and 'birads_score' in curr:
            trends["birads_score_trend"].append(curr['birads_score'] - prev['birads_score'])
    
    # Summarize trends (e.g., mean change, overall direction)
    summary_trends = {}
    for key, changes in trends.items():
        if changes:
            avg_change = sum(changes) / len(changes)
            if avg_change > 0.01: # Arbitrary threshold
                summary_trends[key] = f"Increasing (Avg change: {avg_change:.2f})"
            elif avg_change < -0.01:
                summary_trends[key] = f"Decreasing (Avg change: {avg_change:.2f})"
            else:
                summary_trends[key] = f"Stable (Avg change: {avg_change:.2f})"
        else:
            summary_trends[key] = "No sufficient data for trend."
            
    return summary_trends

if __name__ == "__main__":
    # Example Usage
    print("Testing patient history tracking functions...")

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Clean up previous history for a fresh test
    if os.path.exists(PATIENT_HISTORY_FILE):
        os.remove(PATIENT_HISTORY_FILE)
    print("Cleaned up old history file.")

    # Add first result for Patient A
    result1_A = {
        "date": datetime(2024, 1, 15).isoformat(),
        "cancer_probability": 0.1,
        "risk_category": 1, # Low
        "severity_stage": 0,
        "birads_score": 2,
        "confidence": 0.85
    }
    add_mammogram_result("PatientA", result1_A)
    print("Added first result for PatientA.")

    # Add second result for Patient A
    result2_A = {
        "date": datetime(2025, 1, 20).isoformat(),
        "cancer_probability": 0.15,
        "risk_category": 2, # Medium
        "severity_stage": 1,
        "birads_score": 3,
        "confidence": 0.90
    }
    add_mammogram_result("PatientA", result2_A)
    print("Added second result for PatientA.")

    # Get timeline for Patient A
    timeline_A = get_patient_timeline("PatientA")
    print("\nPatientA Timeline:")
    for res in timeline_A:
        print(res)

    # Compare current result with previous for Patient A (using result2_A as current)
    comparison_A = compare_results("PatientA", result2_A)
    print("\nPatientA Comparison (current vs previous):")
    if comparison_A:
        print(comparison_A)
    else:
        print("No previous results for comparison.")

    # Calculate trends for Patient A
    trends_A = calculate_trends("PatientA")
    print("\nPatientA Trends:")
    print(trends_A)

    # Add result for Patient B
    result1_B = {
        "date": datetime(2024, 5, 10).isoformat(),
        "cancer_probability": 0.05,
        "risk_category": 1, 
        "severity_stage": 0,
        "birads_score": 1,
        "confidence": 0.92
    }
    add_mammogram_result("PatientB", result1_B)
    print("\nAdded first result for PatientB.")

    history = load_patient_history()
    print("\nFull Patient History:")
    print(json.dumps(history, indent=4))

    print("Patient history tracking functions tested successfully.")

