import os
import tarfile
import pydicom
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def download_cbis_ddsm(data_dir="data/CBIS-DDSM"):
    """
    Instructions for downloading the CBIS-DDSM dataset.
    Since the dataset is large and often requires manual download or Kaggle API setup,
    we'll provide instructions here.
    """
    print("Please download the CBIS-DDSM dataset manually from one of these links:")
    print("- https://www.cancerimagingarchive.net/collection/cbis-ddsm/")
    print("- https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    print(f"After downloading, place the tar.gz files (e.g., CBIS-DDSM-Breast-Cancer-Mammography-Digital-Images.tar.gz) ")
    print(f"or extracted folders into the '{data_dir}' directory.")
    print("For Kaggle, you might need to use the Kaggle API to download the dataset.")

def extract_cbis_ddsm(data_dir="data/CBIS-DDSM"):
    """
    Extracts all tar.gz files in the specified data directory.
    """
    print(f"Extracting files from {data_dir}...")
    for item in os.listdir(data_dir):
        if item.endswith(".tar.gz"):
            file_path = os.path.join(data_dir, item)
            print(f"Extracting {file_path}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print(f"Finished extracting {file_path}.")

def load_dicom_image(dicom_path):
    """Loads a DICOM image and returns it as a NumPy array."""
    dicom = pydicom.dcmread(dicom_path)
    # The pixel array might need scaling or windowing depending on the specific DICOM file
    data = dicom.pixel_array
    # Normalize to 0-255 if needed, or based on DICOM's PhotometricInterpretation
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    return data.astype(np.uint8)


def preprocess_image(image_path, target_size=(224, 224), enable_augmentation=False):
    """
    Loads, preprocesses, and optionally augments a single mammogram image.
    This is a placeholder for more sophisticated preprocessing.
    """
    # Load image (assuming it's a DICOM file for now)
    img = load_dicom_image(image_path)
    
    # Convert to grayscale if not already (DICOM should be, but for general robustness)
    if len(img.shape) == 3 and img.shape[2] == 3: # If it's a color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalization: Min-Max Scaling to 0-255 (already done in load_dicom_image, but can be re-applied)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # ROI Extraction (Enhanced Placeholder)
    # This is a basic approach. For CBIS-DDSM, it's best to use provided ROI masks if available.
    # Here, we attempt to find the breast region by thresholding and taking the largest contour.
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY) # Adjust threshold as needed
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the breast region
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y+h, x:x+w]
    else:
        print(f"Warning: No significant contours found for ROI extraction in {image_path}. Using full image.")
    
    # Resizing
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Data Augmentation (Enhanced Placeholder)
    if enable_augmentation:
        # Random rotation
        angle = np.random.randint(-15, 15) # Rotate by -15 to +15 degrees
        M = cv2.getRotationMatrix2D((target_size[0] / 2, target_size[1] / 2), angle, 1)
        img = cv2.warpAffine(img, M, target_size)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1) 
            
        # Random vertical flip (less common for mammograms, but can be an option)
        if np.random.rand() > 0.8: # Lower probability
            img = cv2.flip(img, 0)
            
        # Random brightness adjustment
        img = cv2.convertScaleAbs(img, alpha=1.0 + np.random.uniform(-0.2, 0.2), beta=np.random.randint(-20, 20))

        # (Optional) Random zoom/scale - more complex to implement without losing context
        # For now, we'll stick to rotation and flips.

    # Ensure image is 3-channel for CNN input if needed later, or keep as grayscale
    # For PyTorch, images are often expected as (C, H, W). We'll handle this in the DataLoader.
    # For now, return as grayscale (H, W)
    
    return img

def create_dataframe(data_dir="data/CBIS-DDSM"):
    """
    Creates a DataFrame containing paths to images and relevant labels.
    This function needs to be adapted based on the actual file structure
    and metadata files provided within the CBIS-DDSM dataset after extraction.
    """
    image_paths = []
    labels = []
    patient_ids = []
    mammography_types = [] # e.g., 'full mammogram', 'ROI'
    pathology_labels = [] # e.g., 'BENIGN', 'MALIGNANT', 'NORMAL'

    # This part will heavily depend on the extracted folder structure
    # For CBIS-DDSM, metadata is often in CSV files
    # Example: 'calc_case_description.csv', 'mass_case_description.csv'

    # --- Placeholder for loading metadata CSVs and linking to DICOM files ---
    # This will require careful inspection of the extracted CBIS-DDSM structure.
    # For demonstration, let's assume a simplified structure.

    # Example of navigating through the extracted folders
    # This assumes a structure like:
    # CBIS-DDSM/Calc_cases/Calc-0001/1-1.dcm
    # CBIS-DDSM/Mass_cases/Mass-0001/1-1.dcm
    
    # You will need to locate the actual patient and image IDs and match them
    # with the DICOM files and pathology information from the CSVs.
    
    # Let's assume we are iterating through the image directories after extraction
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                image_paths.append(dicom_path)
                
                # Extract basic info from path (this will need refinement)
                # This is a very simplistic way; actual parsing of metadata CSVs is crucial
                if "Calc_cases" in root:
                    mammography_types.append("calcification")
                elif "Mass_cases" in root:
                    mammography_types.append("mass")
                else:
                    mammography_types.append("unknown")

                # Placeholder for label extraction - actual labels come from CSVs
                # Example: if 'BENIGN' or 'MALIGNANT' is in the path or linked via metadata
                if "BENIGN" in root.upper():
                    pathology_labels.append("BENIGN")
                elif "MALIGNANT" in root.upper():
                    pathology_labels.append("MALIGNANT")
                else:
                    pathology_labels.append("UNKNOWN")
                
                # Placeholder for patient ID - actual IDs come from CSVs
                patient_ids.append(os.path.basename(os.path.dirname(root)))
                labels.append(0) # Placeholder label

    # Load metadata CSVs
    try:
        calc_cases = pd.read_csv(os.path.join(data_dir, "calc_case_description.csv"))
        mass_cases = pd.read_csv(os.path.join(data_dir, "mass_case_description.csv"))
        # The DICOM_metadata.csv is often found within the extracted folders,
        # or can be generated by scanning DICOM headers.
        # For simplicity, let's assume it's directly in data_dir if available,
        # or we'll infer paths later.
        
        # Combine calc and mass cases, adding a 'type' column to distinguish
        calc_cases['case_type'] = 'calcification'
        mass_cases['case_type'] = 'mass'
        
        # Standardize column names if they differ, or select common ones
        # Example: assume 'patient_id' is common or can be mapped
        
        all_cases = pd.concat([calc_cases, mass_cases], ignore_index=True)
        
        # Clean up column names (example, adjust based on actual CSV headers)
        all_cases.columns = all_cases.columns.str.strip().str.lower().str.replace(' ', '_')

        # Map pathology to a numerical label
        all_cases['label'] = all_cases['pathology'].apply(
            lambda x: 1 if 'MALIGNANT' in str(x).upper() else (0 if 'BENIGN' in str(x).upper() else -1)
        ) # -1 for unknown/normal, will need to refine

        # Now, link DICOM image paths. This is the trickiest part as naming conventions vary.
        # We need to find a way to match entries in all_cases to actual .dcm files.
        # Often, DICOM file paths contain identifiers that can be matched to CSV entries.

        # Let's create a mapping from patient/image identifiers to file paths
        # This will be highly dependent on the exact naming convention after extraction.
        
        # Placeholder: Iterate through extracted files and try to match
        image_data = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".dcm"):
                    dicom_path = os.path.join(root, file)
                    
                    # Try to infer patient_id, image_id, or other keys from path
                    # This is a generic attempt; specific CBIS-DDSM structure parsing is needed.
                    
                    # Example of extracting components from path:
                    # e.g., .../CBIS-DDSM/Calc_cases/Calc-0001/000000.dcm
                    parts = dicom_path.split(os.sep)
                    
                    patient_id_from_path = None
                    image_id_from_path = None # This might correspond to 'image file path' in CSV
                    
                    for i, part in enumerate(parts):
                        if 'Calc-' in part or 'Mass-' in part:
                            patient_id_from_path = part
                        # Further logic to extract image_id if available in path
                        # e.g., if 'full mammogram' or 'ROI' is in the path.

                    image_data.append({
                        'image_path': dicom_path,
                        'patient_id_from_path': patient_id_from_path,
                        # Add other inferred IDs here
                    })
        
        image_df = pd.DataFrame(image_data)
        
        # Merge with all_cases. This merge key will need to be carefully chosen.
        # It's usually a combination of patient_id and image identifiers.
        # For CBIS-DDSM, the CSVs often have columns like 'image file path', 'cropped image file path', 'ROI file path'.
        # We need to use these to link.

        # Let's assume 'patient_id' is the primary key for merging for now.
        # This will need to be refined based on actual CBIS-DDSM CSV column names
        # and how they link to DICOM paths.

        # A common approach is to parse the 'image file path', 'cropped image file path', 'ROI file path'
        # columns in the CSVs to construct the full path to the DICOM files.
        # For simplicity in this placeholder, let's assume we can merge on patient_id for now,
        # but this needs significant refinement with actual data structure.

        # If we had 'image file path' in the CSV, we'd do something like:
        # all_cases['image_full_path'] = all_cases['image_file_path'].apply(
        #     lambda x: os.path.join(data_dir, x) if pd.notna(x) else None
        # )
        # final_df = pd.merge(image_df, all_cases, left_on='image_path', right_on='image_full_path', how='left')

        # Since we are just inferring from paths for now, let's just use image_df
        # and add placeholder columns for now. This needs to be replaced with actual CSV parsing.
        final_df = image_df # This will be the starting point, to be enriched by CSVs

        # For CBIS-DDSM, you will typically find columns in the CSVs that contain
        # relative paths to the DICOM files. You'll need to join these paths with `data_dir`.
        
        # Example of how to link if 'image_path' in CSV matches part of the full path
        # This is highly dependent on the exact CSV structure and file organization.
        
        # A more robust approach would be to parse the CSVs, construct expected DICOM paths,
        # and then verify if those paths exist.

        # Placeholder: Add dummy labels for now if merging is complex
        if 'label' not in final_df.columns:
            final_df['label'] = 0 # Default to 0, will be filled by CSV data
        if 'pathology' not in final_df.columns:
            final_df['pathology'] = 'UNKNOWN'

        # Attempt to merge based on patient_id from path
        # This needs to be robustly handled once the exact CSVs and their linking columns are known.
        
        # For now, let's create a simplified DataFrame using patient IDs and paths found.
        # The true linking will happen when we fully parse the CBIS-DDSM metadata.

        print("Please review the CBIS-DDSM dataset's CSV files (calc_case_description.csv, mass_case_description.csv) ")
        print("and adapt the `create_dataframe` function to correctly link DICOM paths with metadata.")

        # Simplified DataFrame for now, will be replaced by a proper merge
        df = final_df # Starting with image paths found
        # Add a dummy label if not already present
        if 'label' not in df.columns:
            df['label'] = 0
        if 'pathology' not in df.columns:
            df['pathology'] = 'UNKNOWN'
        if 'mammography_type' not in df.columns:
            df['mammography_type'] = 'UNKNOWN'
        if 'patient_id' not in df.columns:
            df['patient_id'] = df['patient_id_from_path'] # Using inferred patient ID

        return df

    except FileNotFoundError:
        print("Metadata CSV files not found. Ensure 'calc_case_description.csv' and 'mass_case_description.csv'")
        print(f"are in the '{data_dir}' directory after extraction.")
        return pd.DataFrame() # Return empty DataFrame if CSVs are missing
    except Exception as e:
        print(f"An error occurred while creating the DataFrame: {e}")
        return pd.DataFrame()



if __name__ == "__main__":
    DATA_DIR = "data/CBIS-DDSM"
    
    # 1. Download instructions
    download_cbis_ddsm(DATA_DIR)
    
    # 2. Extraction
    # Make sure to place the .tar.gz files in data/CBIS-DDSM before running this
    # extract_cbis_ddsm(DATA_DIR)
    
    # 3. Create DataFrame with image paths and metadata
    # This step will need to be run AFTER extraction and manual placement of files.
    # cbis_df = create_dataframe(DATA_DIR)
    # print(f"Generated DataFrame with {len(cbis_df)} entries.")
    # print(cbis_df.head())

    # Example of processing a single image (after extraction)
    # if not cbis_df.empty:
    #     sample_image_path = cbis_df['image_path'].iloc[0]
    #     print(f"Processing sample image: {sample_image_path}")
    #     processed_img = preprocess_image(sample_image_path)
    #     print(f"Processed image shape: {processed_img.shape}")
    #     # You can save or display the processed image here for verification

