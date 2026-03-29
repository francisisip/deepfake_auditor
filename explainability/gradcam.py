import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_gradcam_heatmap(model, input_tensor, original_face_img):
    # We specifically target the final convolutional layer in EfficientNet.
    # That's typically where the best high-level spatial reasoning lives before the network flattens out.
    target_layers = [model.features[-1]]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # In our binary model, index 1 corresponds to "FAKE" and 0 to "AUTHENTIC".
        # We explicitly ask Grad-CAM to show us what triggered the "FAKE" nodes.
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # GradCAM naturally returns a batch of heatmaps, but since we only ever 
        # pass in a single face image at a time, we just snatch the first index.
        grayscale_cam = grayscale_cam[0, :]
        
        # The show_cam_on_image utility expects the image to be normalized between 0 and 1,
        # so we divide the original pixel values (0-255) by 255
        normalized_face = original_face_img.astype(np.float32) / 255.0
        
        return show_cam_on_image(normalized_face, grayscale_cam, use_rgb=True)