import torch
import torch.nn as nn
import torchvision.models as models

class LesionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LesionCNN, self).__init__()
        # Using a pre-trained ResNet-18 as a backbone for transfer learning
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Use DEFAULT for latest best weights
        
        # Modify the first convolutional layer for single-channel input (grayscale mammograms)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Freeze pre-trained layers initially if desired
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
            
        # Replace the final fully connected layer for our specific number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

class WholeMammogramCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(WholeMammogramCNN, self).__init__()
        # Using a pre-trained ResNet-50 for more capacity for global context
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the first convolutional layer for single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Freeze pre-trained layers initially if desired
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
            
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

class HybridModel(nn.Module):
    def __init__(self, num_lesion_classes=2, num_whole_classes=2, num_clinical_features=3, 
                 num_cancer_prob_classes=2, num_risk_classes=3, num_severity_classes=5, num_birads_classes=6):
        super(HybridModel, self).__init__()
        
        self.lesion_cnn = LesionCNN(num_classes=num_lesion_classes) # Output features, not final classification
        self.whole_mammogram_cnn = WholeMammogramCNN(num_classes=num_whole_classes) # Output features

        # Remove the final classification layers from the base CNNs to extract features
        self.lesion_cnn.resnet.fc = nn.Identity() 
        self.whole_mammogram_cnn.resnet.fc = nn.Identity()
        
        # Fusion layer for image features and clinical features
        # Assuming lesion_cnn output features are 512 (ResNet-18 last layer before FC)
        # Assuming whole_mammogram_cnn output features are 2048 (ResNet-50 last layer before FC)
        # Total image features = 512 + 2048 = 2560
        self.image_feature_dim = 512 + 2048 # Adjust based on actual backbone output feature size
        self.total_fusion_dim = self.image_feature_dim + num_clinical_features
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.total_fusion_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Multi-task learning heads
        # 1. Cancer probability (Benign vs Malignant)
        self.cancer_prob_head = nn.Linear(512, num_cancer_prob_classes) # 2 classes: benign/malignant
        
        # 2. Risk category (Low / Medium / High)
        self.risk_category_head = nn.Linear(512, num_risk_classes) # 3 classes
        
        # 3. Estimated disease severity stage (Stage 0-IV)
        self.severity_stage_head = nn.Linear(512, num_severity_classes) # 5 classes
        
        # 4. BI-RADS-like suspicion score (e.g., 0-6 or specific categories)
        self.birads_score_head = nn.Linear(512, num_birads_classes) # 6 classes (0-5, 6 for known cancer)

        # 5. Prediction confidence (can be a regression task or implied by softmax output)
        # For now, we'll let the softmax outputs from the above heads imply confidence.
        # If a dedicated confidence score is needed, another head with a single output neuron (regression) could be added.
        
    def forward(self, lesion_img, whole_mammogram_img, clinical_features):
        lesion_features = self.lesion_cnn(lesion_img)
        whole_mammogram_features = self.whole_mammogram_cnn(whole_mammogram_img)
        
        # Flatten features if they are not already (e.g., if a pooling layer precedes fc in backbone)
        lesion_features = torch.flatten(lesion_features, 1)
        whole_mammogram_features = torch.flatten(whole_mammogram_features, 1)
        
        combined_image_features = torch.cat((lesion_features, whole_mammogram_features), dim=1)
        
        # Concatenate image features with clinical features
        fused_features = torch.cat((combined_image_features, clinical_features), dim=1)
        
        # Pass through fusion MLP
        fused_features = self.fusion_mlp(fused_features)
        
        # Multi-task outputs
        cancer_prob_output = self.cancer_prob_head(fused_features)
        risk_category_output = self.risk_category_head(fused_features)
        severity_stage_output = self.severity_stage_head(fused_features)
        birads_score_output = self.birads_score_head(fused_features)
        
        return cancer_prob_output, risk_category_output, severity_stage_output, birads_score_output

if __name__ == "__main__":
    # Test the models
    print("Testing LesionCNN...")
    lesion_model = LesionCNN(num_classes=2)
    dummy_lesion_input = torch.randn(1, 1, 224, 224) # Batch size 1, 1 channel, 224x224
    lesion_output = lesion_model(dummy_lesion_input)
    print(f"LesionCNN output shape: {lesion_output.shape}")

    print("\nTesting WholeMammogramCNN...")
    whole_mammogram_model = WholeMammogramCNN(num_classes=2)
    dummy_whole_mammogram_input = torch.randn(1, 1, 224, 224) # Batch size 1, 1 channel, 224x224
    whole_mammogram_output = whole_mammogram_model(dummy_whole_mammogram_input)
    print(f"WholeMammogramCNN output shape: {whole_mammogram_output.shape}")

    print("\nTesting HybridModel...")
    hybrid_model = HybridModel(
        num_cancer_prob_classes=2, 
        num_risk_classes=3, 
        num_severity_classes=5, 
        num_birads_classes=6
    )
    dummy_lesion_input_hybrid = torch.randn(1, 1, 224, 224)
    dummy_whole_mammogram_input_hybrid = torch.randn(1, 1, 224, 224)
    dummy_clinical_features = torch.randn(1, 3) # Batch size 1, 3 clinical features
    
    cancer_prob, risk_cat, severity, birads = hybrid_model(
        dummy_lesion_input_hybrid, dummy_whole_mammogram_input_hybrid, dummy_clinical_features
    )
    print(f"HybridModel Cancer Probability output shape: {cancer_prob.shape}")
    print(f"HybridModel Risk Category output shape: {risk_cat.shape}")
    print(f"HybridModel Severity Stage output shape: {severity.shape}")
    print(f"HybridModel BI-RADS output shape: {birads.shape}")

