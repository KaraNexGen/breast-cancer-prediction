import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval() # Ensure model is in evaluation mode
        self.target_layer = None
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self._register_hooks(target_layer_name)

    def _register_hooks(self, target_layer_name):
        def forward_hook(module, input, output):
            self.activations = output.detach() # Capture activations

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach() # Capture gradients

        # Find the target layer by name. This might need to be adjusted based on the model structure.
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                self.target_layer.register_forward_hook(forward_hook)
                self.target_layer.register_backward_hook(backward_hook)
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found in model.")

    def generate_heatmap(self, input_tensor, target_category=None):
        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass
        # Assuming input_tensor is (1, C, H, W) for a single image
        # For our HybridModel, input_tensor will be a tuple: (lesion_img, whole_mammogram_img, clinical_features)
        # We need to decide which CNN's output we want to generate the heatmap for.
        # Let's assume we want to generate for the whole_mammogram_cnn's output for now.
        
        # If input_tensor is a tuple (for HybridModel)
        if isinstance(input_tensor, tuple):
            lesion_img, whole_mammogram_img, clinical_features = input_tensor
            # We need to run the specific CNN whose target_layer we are interested in
            # For example, if target_layer is in whole_mammogram_cnn:
            model_output = self.model.whole_mammogram_cnn(whole_mammogram_img)
            # If we wanted for lesion_cnn:
            # model_output = self.model.lesion_cnn(lesion_img)
        else:
            # Assume a single image input for a standalone CNN (e.g., if testing just LesionCNN)
            model_output = self.model(input_tensor)

        # If target_category is not specified, use the predicted class
        if target_category is None:
            target_category = torch.argmax(model_output).item()

        # Backward pass to get gradients for the target class
        # Assuming model_output is for the cancer probability head for simplicity
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_category] = 1 # Assuming batch size 1
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get the averaged gradients across the width and height dimensions
        guided_gradients = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Element-wise multiply the activations by the averaged gradients
        cam = (guided_gradients * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU to remove negative values and get final heatmap
        cam = F.relu(cam)
        
        # Resize heatmap to input image size
        cam = F.interpolate(cam, size=(input_tensor[1].shape[2], input_tensor[1].shape[3]), mode='bilinear', align_corners=False)
        # Normalize heatmap to 0-1
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy() # Remove batch and channel dims, convert to numpy

def visualize_heatmap(original_image, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    original_image: NumPy array (H, W) or (H, W, 1) or (H, W, 3)
    heatmap: NumPy array (H, W) with values between 0 and 1
    alpha: Transparency factor for the heatmap
    """
    # Convert original image to 3 channels if it's grayscale
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Normalize original image to 0-255 if not already
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Convert heatmap to 3 channels and scale to 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img

if __name__ == "__main__":
    # Example usage (requires a trained model and an input image)
    print("Grad-CAM utility created. Example usage below (requires model and input)...")

    # Dummy model and input for demonstration
    from src.models.cnn_models import HybridModel

    # Initialize a dummy HybridModel
    dummy_model = HybridModel(
        num_cancer_prob_classes=2, 
        num_risk_classes=3, 
        num_severity_classes=5, 
        num_birads_classes=6
    )
    # For Grad-CAM, we need to specify a target convolutional layer.
    # For WholeMammogramCNN (ResNet50), a good target layer could be 'resnet.layer4' or 'resnet.avgpool' before FC.
    # Let's pick 'resnet.layer4' for the whole_mammogram_cnn as an example.
    
    # You need to ensure the target_layer_name is correct for your model's architecture.
    # You can print(dummy_model.whole_mammogram_cnn.resnet) to inspect layers.
    target_layer_name = 'whole_mammogram_cnn.resnet.layer4'
    grad_cam = GradCAM(dummy_model, target_layer_name)

    # Dummy inputs
    dummy_lesion_img = torch.randn(1, 1, 224, 224)
    dummy_whole_mammogram_img = torch.randn(1, 1, 224, 224)
    dummy_clinical_features = torch.randn(1, 3)
    dummy_input = (dummy_lesion_img, dummy_whole_mammogram_img, dummy_clinical_features)

    # Generate heatmap for a target class (e.g., class 1 for malignant)
    # In a real scenario, this would be the predicted class or a class of interest.
    target_category = 1 
    heatmap = grad_cam.generate_heatmap(dummy_input, target_category)

    print(f"Generated heatmap shape: {heatmap.shape}")

    # Visualize the heatmap (requires an actual image)
    # dummy_original_image = np.zeros((224, 224), dtype=np.uint8) # Replace with actual image
    # superimposed_img = visualize_heatmap(dummy_original_image, heatmap)
    # cv2.imshow("Grad-CAM", superimposed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

