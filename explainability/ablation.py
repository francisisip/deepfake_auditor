import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def perform_ablation(model, original_face, heatmap, device):
    """
    Masks the most activated region from the heatmap and re-evaluates.
    """
    # Sometimes the heatmap comes back strictly bounded 0-1, sometimes 0-255 depending on how
    # GradCAM was scaled beforehand. We just normalize it all to 0-1 here to be safe.
    if heatmap.max() > 1:
        heatmap_norm = heatmap / 255.0
    else:
        heatmap_norm = heatmap
        
    # We're building a boolean mask grabbing the top 20% "hottest" pixels.
    # These are the pixels the model stared at the most when deciding it was a deepfake.
    mask = heatmap_norm > np.percentile(heatmap_norm, 80)
    
    # We copy the face and just black out the masked regions.
    # If the model was actually focused on this artifact, throwing it out should 
    # cause the confidence score to completely tank.
    obscured_face = original_face.copy()
    obscured_face[mask] = 0
    
    # We have to re-run the whole normalization pipeline because we manipulated the raw image array
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    obscured_pil = Image.fromarray(obscured_face)
    obscured_tensor = transform(obscured_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(obscured_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        new_fake_prob = probabilities[1].item() * 100
        
    return obscured_face, new_fake_prob
