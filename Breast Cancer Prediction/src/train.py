import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import os

from src.data.datasets import CBISDDSMDataset, create_dataloaders
from src.data.download_and_preprocess import create_dataframe
from src.models.cnn_models import HybridModel

# Configuration / Hyperparameters
CONFIG = {
    'batch_size': 16,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'image_size': (224, 224),
    'num_workers': 4,
    'data_dir': 'data/CBIS-DDSM',
    'model_save_path': 'trained_models/hybrid_model.pth',
    'num_cancer_prob_classes': 2, # Benign/Malignant
    'num_risk_classes': 3,       # Low/Medium/High
    'num_severity_classes': 5,   # Stage 0-IV
    'num_birads_classes': 6      # BI-RADS 0-5 (or 0-6 including known cancer)
}

def train_model(model, train_loader, val_loader, criterion_prob, criterion_risk, 
                criterion_severity, criterion_birads, optimizer, device, config):
    
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        train_preds_prob, train_labels_prob = [], []
        
        for i, (lesion_img, whole_mammogram_img, clinical_features, 
                 cancer_prob_labels, risk_labels, severity_labels, birads_labels) in enumerate(train_loader):

            # Move data to device
            lesion_img = lesion_img.to(device)
            whole_mammogram_img = whole_mammogram_img.to(device)
            clinical_features = clinical_features.to(device)
            cancer_prob_labels = cancer_prob_labels.to(device)
            risk_labels = risk_labels.to(device)
            severity_labels = severity_labels.to(device)
            birads_labels = birads_labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass through hybrid model
            cancer_prob_output, risk_category_output, severity_stage_output, birads_score_output = \n                model(lesion_img, whole_mammogram_img, clinical_features)
            
            # Calculate multi-task losses
            loss_prob = criterion_prob(cancer_prob_output, cancer_prob_labels)
            loss_risk = criterion_risk(risk_category_output, risk_labels)
            loss_severity = criterion_severity(severity_stage_output, severity_labels)
            loss_birads = criterion_birads(birads_score_output, birads_labels)

            # Combined loss (you might want to weight these losses)
            total_loss = loss_prob + loss_risk + loss_severity + loss_birads
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * lesion_img.size(0)
            
            # Store predictions and labels for metrics
            train_preds_prob.extend(torch.argmax(cancer_prob_output, dim=1).cpu().numpy())
            train_labels_prob.extend(cancer_prob_labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_prob = accuracy_score(train_labels_prob, train_preds_prob)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {epoch_loss:.4f}, Train Acc (Prob): {epoch_acc_prob:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_preds_prob, val_labels_prob = [], []
        
        with torch.no_grad():
            for lesion_img, whole_mammogram_img, clinical_features, \
                cancer_prob_labels, risk_labels, severity_labels, birads_labels in val_loader:

                lesion_img = lesion_img.to(device)
                whole_mammogram_img = whole_mammogram_img.to(device)
                clinical_features = clinical_features.to(device)
                cancer_prob_labels = cancer_prob_labels.to(device)
                risk_labels = risk_labels.to(device)
                severity_labels = severity_labels.to(device)
                birads_labels = birads_labels.to(device)
                
                cancer_prob_output, risk_category_output, severity_stage_output, birads_score_output = \n                    model(lesion_img, whole_mammogram_img, clinical_features)

                loss_prob = criterion_prob(cancer_prob_output, cancer_prob_labels)
                loss_risk = criterion_risk(risk_category_output, risk_labels)
                loss_severity = criterion_severity(severity_stage_output, severity_labels)
                loss_birads = criterion_birads(birads_score_output, birads_labels)
                
                total_loss = loss_prob + loss_risk + loss_severity + loss_birads
                
                val_running_loss += total_loss.item() * lesion_img.size(0)
                
                val_preds_prob.extend(torch.argmax(cancer_prob_output, dim=1).cpu().numpy())
                val_labels_prob.extend(cancer_prob_labels.cpu().numpy())
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc_prob = accuracy_score(val_labels_prob, val_preds_prob)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Val Loss: {val_epoch_loss:.4f}, Val Acc (Prob): {val_epoch_acc_prob:.4f}")

        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Model saved to {config['model_save_path']}")

    print("Training complete.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data to create the full DataFrame
    # This part requires the CBIS-DDSM dataset to be downloaded and extracted.
    # For demonstration, we'll use a dummy DataFrame structure for now.
    # In a real scenario, you'd call create_dataframe and then split it.
    print("Preparing dummy DataFrame and splitting for training/validation...")
    
    # Dummy DataFrame creation (mimicking the expected structure)
    num_samples = 100
    dummy_data = {
        'image_path': [os.path.join(CONFIG['data_dir'], f"dummy_patient/dummy_image_{i}.dcm") for i in range(num_samples)],
        'patient_id': [f"P{i}" for i in range(num_samples)],
        'mammography_type': ["mass" if i % 2 == 0 else "calcification" for i in range(num_samples)],
        'pathology': ["BENIGN" if i % 2 == 0 else "MALIGNANT" for i in range(num_samples)],
        'label': [0 if i % 2 == 0 else 1 for i in range(num_samples)], # Binary for cancer_prob
        'risk_category': np.random.randint(0, CONFIG['num_risk_classes'], num_samples),
        'severity_stage': np.random.randint(0, CONFIG['num_severity_classes'], num_samples),
        'birads_score': np.random.randint(0, CONFIG['num_birads_classes'], num_samples),
        'clinical_age': np.random.randint(30, 80, num_samples),
        'clinical_breast_density': np.random.randint(1, 5, num_samples),
        'clinical_family_history': np.random.randint(0, 2, num_samples),
    }
    full_df = pd.DataFrame(dummy_data)

    # Simulate train/validation split
    train_val_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']) # 0.25 of 0.8 is 0.2
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test' # For future use
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Now, the CBISDDSMDataset and create_dataloaders will need to be updated
    # to handle multiple labels and potentially separate ROI/whole images.
    # For initial training script, let's assume the dataset provides the primary label for now.
    
    # Update: CBISDDSMDataset needs to be modified to return a tuple of labels
    # For now, let's proceed with the current dataset structure and adjust `train_model`.
    
    # You would pass the 'final_df' to create_dataloaders
    train_loader, val_loader = create_dataloaders(
        final_df[final_df['split'] != 'test'], # Don't pass test data to train/val loaders initially
        CONFIG['data_dir'], 
        batch_size=CONFIG['batch_size'], 
        target_size=CONFIG['image_size'],
        train_aug=True, 
        val_aug=False, 
        num_workers=CONFIG['num_workers']
    )

    # 2. Initialize model, loss functions, and optimizer
    model = HybridModel(
        num_cancer_prob_classes=CONFIG['num_cancer_prob_classes'],
        num_risk_classes=CONFIG['num_risk_classes'],
        num_severity_classes=CONFIG['num_severity_classes'],
        num_birads_classes=CONFIG['num_birads_classes'],
        num_clinical_features=3 # Adjust based on actual clinical features used
    ).to(device)

    # Loss functions for each task
    # For multi-class classification, CrossEntropyLoss is suitable.
    # Weights can be added to balance tasks if needed.
    criterion_prob = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()
    criterion_birads = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 3. Train the model
    train_model(model, train_loader, val_loader, criterion_prob, criterion_risk, 
                criterion_severity, criterion_birads, optimizer, device, CONFIG)

    print("Training script finished.")

if __name__ == "__main__":
    main()

