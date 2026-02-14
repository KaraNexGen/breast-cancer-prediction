import streamlit as st
import torch
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import os

# Import modules from src
from src.models.cnn_models import HybridModel
from src.data.download_and_preprocess import preprocess_image
from src.features.patient_history import (load_patient_history, save_patient_history,
                                          add_mammogram_result, get_patient_timeline,
                                          compare_results, calculate_trends)
from src.utils.grad_cam import GradCAM, visualize_heatmap

# --- Configuration --- #
CONFIG = {
    'model_path': 'trained_models/hybrid_model.pth',
    'image_size': (224, 224),
    'num_cancer_prob_classes': 2, # Benign/Malignant
    'num_risk_classes': 3,       # Low/Medium/High
    'num_severity_classes': 5,   # Stage 0-IV
    'num_birads_classes': 6,      # BI-RADS 0-5 (or 0-6 including known cancer)
    'grad_cam_target_layer': 'whole_mammogram_cnn.resnet.layer4' # Adjust as needed
}

# Map numerical labels to descriptive strings
CANCER_PROB_MAP = {0: "Benign", 1: "Malignant"}
RISK_CATEGORY_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
SEVERITY_STAGE_MAP = {0: "Stage 0", 1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}
BIRADS_SCORE_MAP = {
    0: "BI-RADS 0: Incomplete",
    1: "BI-RADS 1: Negative",
    2: "BI-RADS 2: Benign Finding",
    3: "BI-RADS 3: Probably Benign",
    4: "BI-RADS 4: Suspicious Abnormality",
    5: "BI-RADS 5: Highly Suggestive of Malignancy",
    6: "BI-RADS 6: Known Biopsy-Proven Malignancy"
}

# --- Model Loading (Cached to avoid reloading on every rerun) --- #
@st.cache_resource
def load_model(model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(
        num_cancer_prob_classes=config['num_cancer_prob_classes'],
        num_risk_classes=config['num_risk_classes'],
        num_severity_classes=config['num_severity_classes'],
        num_birads_classes=config['num_birads_classes'],
        num_clinical_features=3 # Ensure this matches the model's __init__
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.success(f"Model loaded from {model_path}")
    else:
        st.warning(f"Model file not found at {model_path}. Using untrained model weights.")
        st.info("Please train the model using `python src/train.py` first.")
    
    model.eval() # Set to evaluation mode
    return model, device

model, device = load_model(CONFIG['model_path'], CONFIG)
grad_cam = GradCAM(model, CONFIG['grad_cam_target_layer'])

# --- Helper Functions --- #
def make_prediction(model, device, processed_image, clinical_features):
    # Prepare inputs for the model
    # The model expects lesion_img, whole_mammogram_img, clinical_features
    # For now, we use the same processed_image for both image inputs.
    lesion_img_tensor = torch.tensor(processed_image).unsqueeze(0).float().to(device) # Add batch dim
    whole_mammogram_img_tensor = torch.tensor(processed_image).unsqueeze(0).float().to(device)
    clinical_features_tensor = torch.tensor(clinical_features).unsqueeze(0).float().to(device)

    with torch.no_grad():
        cancer_prob_output, risk_category_output, severity_stage_output, birads_score_output = \
            model(lesion_img_tensor, whole_mammogram_img_tensor, clinical_features_tensor)

    # Convert logits to probabilities and predicted classes
    cancer_prob_softmax = torch.softmax(cancer_prob_output, dim=1).cpu().numpy()[0]
    risk_category_softmax = torch.softmax(risk_category_output, dim=1).cpu().numpy()[0]
    severity_stage_softmax = torch.softmax(severity_stage_output, dim=1).cpu().numpy()[0]
    birads_score_softmax = torch.softmax(birads_score_output, dim=1).cpu().numpy()[0]

    predicted_cancer_prob = np.argmax(cancer_prob_softmax)
    predicted_risk_category = np.argmax(risk_category_softmax)
    predicted_severity_stage = np.argmax(severity_stage_softmax)
    predicted_birads_score = np.argmax(birads_score_softmax)
    
    # Confidence can be the max probability of the predicted class
    confidence_cancer_prob = cancer_prob_softmax[predicted_cancer_prob]
    
    return {
        "cancer_probability_raw": cancer_prob_softmax,
        "predicted_cancer_prob_idx": predicted_cancer_prob,
        "predicted_cancer_prob_str": CANCER_PROB_MAP.get(predicted_cancer_prob, "Unknown"),
        "confidence_cancer_prob": confidence_cancer_prob,
        "predicted_risk_category_idx": predicted_risk_category,
        "predicted_risk_category_str": RISK_CATEGORY_MAP.get(predicted_risk_category, "Unknown"),
        "predicted_severity_stage_idx": predicted_severity_stage,
        "predicted_severity_stage_str": SEVERITY_STAGE_MAP.get(predicted_severity_stage, "Unknown"),
        "predicted_birads_score_idx": predicted_birads_score,
        "predicted_birads_score_str": BIRADS_SCORE_MAP.get(predicted_birads_score, "Unknown"),
    }

def get_health_suggestions(risk_level, birads_score):
    suggestions = [
        "Maintain a healthy lifestyle with balanced diet and regular exercise.",
        "Perform regular self-breast exams and report any changes to your doctor.",
        "Discuss your family history and personal risk factors with a healthcare provider."
    ]
    if risk_level == "High Risk" or birads_score >= 4:
        suggestions.append("Consider more frequent screenings or advanced imaging as recommended by your doctor.")
        suggestions.append("Consult with a specialist for personalized risk assessment and management.")
    elif risk_level == "Medium Risk" or birads_score == 3:
        suggestions.append("Follow up with your doctor for recommended screening intervals.")
    
    suggestions.append("Remember, this system is for decision support only and not a substitute for professional medical advice.")
    return suggestions

# --- Streamlit App Layout --- #
st.set_page_config(layout="wide", page_title="AI-Assisted Mammography Decision Support")

st.title("AI-Assisted Breast Cancer Mammography Decision Support System")

st.markdown("""
This system provides AI-assisted insights for mammography analysis. 
**It is for decision support only and not a medical diagnosis or treatment system.** 
Always consult with a qualified healthcare professional for medical advice.
""")

st.sidebar.header("Patient Information")
patient_id = st.sidebar.text_input("Patient ID", "AnonymousPatient")

# Clinical Features Input
st.sidebar.subheader("Clinical Features")
age = st.sidebar.slider("Age", 18, 100, 50)
breast_density = st.sidebar.selectbox("Breast Density (ACR Category)", [1, 2, 3, 4], index=1)
family_history = st.sidebar.checkbox("Family History of Breast Cancer")

clinical_features_input = [age, breast_density, 1 if family_history else 0]

# --- Main Content Area --- #

# Image Upload
st.header("Upload Mammogram Image")
uploaded_file = st.file_uploader("Choose a DICOM (.dcm) or PNG/JPG image...", type=["dcm", "png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

original_image_display = None
processed_image_for_gradcam = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Determine file type and process
    if uploaded_file.name.endswith(".dcm"):
        try:
            # Save to a temporary file for pydicom to read
            temp_dcm_path = f"temp_{uploaded_file.name}"
            with open(temp_dcm_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            original_image_display = preprocess_image(temp_dcm_path, target_size=CONFIG['image_size'], enable_augmentation=False) # Get the processed grayscale image
            processed_image_for_gradcam = original_image_display # Use this for Grad-CAM
            os.remove(temp_dcm_path) # Clean up temp file

        except Exception as e:
            st.error(f"Error processing DICOM file: {e}")
            original_image_display = None
    else: # PNG/JPG
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) # Read as grayscale
        if img is not None:
            original_image_display = cv2.resize(img, CONFIG['image_size'], interpolation=cv2.INTER_AREA)
            processed_image_for_gradcam = original_image_display # Use this for Grad-CAM
        else:
            st.error("Could not decode image. Please check the file format.")

    if original_image_display is not None:
        with col1:
            st.subheader("Original/Processed Mammogram")
            st.image(original_image_display, caption="Uploaded Mammogram", use_column_width=True)
            
        # --- Make Prediction --- #
        if st.button("Analyze Mammogram"):
            st.subheader("AI Prediction Results")
            with st.spinner("Analyzing image and clinical data..."):
                prediction_results = make_prediction(model, device, processed_image_for_gradcam, clinical_features_input)
                
                # Save current result to patient history
                current_timestamp = datetime.now().isoformat()
                current_patient_result = {
                    "date": current_timestamp,
                    "cancer_probability": float(prediction_results["cancer_probability_raw"][1]), # Probability of Malignant
                    "risk_category": int(prediction_results["predicted_risk_category_idx"]),
                    "severity_stage": int(prediction_results["predicted_severity_stage_idx"]),
                    "birads_score": int(prediction_results["predicted_birads_score_idx"]),
                    "confidence": float(prediction_results["confidence_cancer_prob"])
                }
                add_mammogram_result(patient_id, current_patient_result)

                st.success("Analysis Complete!")

                # Display Prediction Summary
                st.markdown("### Risk Summary")
                st.write(f"**Cancer Probability:** {prediction_results['predicted_cancer_prob_str']} (Confidence: {prediction_results['confidence_cancer_prob']:.2f})")
                st.write(f"**Risk Category:** {prediction_results['predicted_risk_category_str']}")
                st.write(f"**Severity Stage:** {prediction_results['predicted_severity_stage_str']}")
                st.write(f"**BI-RADS Score:** {prediction_results['predicted_birads_score_str']}")

                # Grad-CAM Visualization
                with col2:
                    st.subheader("Grad-CAM Heatmap")
                    # Input for GradCAM is (lesion_img, whole_mammogram_img, clinical_features)
                    # We use the same processed_image_for_gradcam for both image inputs
                    grad_cam_input_tensor = (torch.tensor(processed_image_for_gradcam).unsqueeze(0).float().to(device),
                                             torch.tensor(processed_image_for_gradcam).unsqueeze(0).float().to(device),
                                             clinical_features_tensor)
                    
                    # Generate heatmap for the predicted cancer probability class
                    heatmap = grad_cam.generate_heatmap(grad_cam_input_tensor, target_category=prediction_results['predicted_cancer_prob_idx'])
                    superimposed_img = visualize_heatmap(original_image_display, heatmap)
                    st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
                    st.markdown("Heatmap highlights regions most influential for the AI's prediction.")
                
                # --- Patient History Timeline & Trends --- #
                st.header("Patient History & Trends")
                patient_timeline = get_patient_timeline(patient_id)
                
                if patient_timeline:
                    st.subheader("Mammogram History Timeline")
                    # Convert to DataFrame for easier display and plotting
                    history_df = pd.DataFrame(patient_timeline)
                    history_df['date'] = pd.to_datetime(history_df['date'])
                    history_df = history_df.sort_values(by='date')
                    
                    st.dataframe(history_df.set_index('date'))

                    st.subheader("Health Progress Graphs")
                    # Plotting trends for cancer probability and severity
                    if not history_df.empty:
                        st.line_chart(history_df, x='date', y=['cancer_probability', 'severity_stage', 'birads_score'])
                    
                    st.subheader("Comparison with Previous Result")
                    if len(patient_timeline) > 1:
                        comparison = compare_results(patient_id, current_patient_result)
                        if comparison:
                            st.write(f"**Previous Exam Date:** {datetime.fromisoformat(comparison['previous_date']).strftime('%Y-%m-%d')}")
                            st.write(f"**Current Exam Date:** {datetime.fromisoformat(comparison['current_date']).strftime('%Y-%m-%d')}")
                            st.write(f"**Cancer Probability Change:** {comparison['cancer_probability_change']:.2f}")
                            st.write(f"**Severity Stage Change:** {comparison['severity_stage_change']}")
                            st.write(f"**BI-RADS Score Change:** {comparison['birads_score_change']}")
                            st.write(f"**Overall Trend:** {comparison['overall_trend']}")
                    else:
                        st.info("No previous results to compare with.")

                    st.subheader("Long-term Trends")
                    trends_summary = calculate_trends(patient_id)
                    for key, value in trends_summary.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No history found for this patient. Upload a mammogram to start tracking!")
                
                # --- Non-Medical Health Suggestions --- #
                st.header("General Health & Awareness Suggestions")
                risk_str = RISK_CATEGORY_MAP.get(prediction_results['predicted_risk_category_idx'], "Unknown")
                birads_idx = prediction_results['predicted_birads_score_idx']
                suggestions = get_health_suggestions(risk_str, birads_idx)
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")

else:
    st.info("Please upload a mammogram image to begin analysis.")


