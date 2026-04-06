import streamlit as st
import tempfile
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import sys

# We need to import our custom explainability modules from the parent directory structure
from preprocessing.face_extractor import extract_face
from explainability.gradcam import generate_gradcam_heatmap
from explainability.ablation import perform_ablation

st.set_page_config(page_title="Deepfake Auditor", page_icon="shield", layout="wide")

@st.cache_resource
def load_model():
    # Caching the model so we don't reload it into memory on every user interaction. 
    # Also checking for Apple Silicon support (MPS) which drastically speeds up inference on newer Macs.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # We trained our custom weights on top of EfficientNet-B4, so we have to instantiate it here
    # and adapt the final classifier layer to our binary output (Authentic vs Fake).
    model = efficientnet_b4(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # Path to Kaggle weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "deepfake_efficientnet_v4.pth"))
    
    # Using map_location ensures we don't crash if the weights were originally saved from a CUDA GPU
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    
    return model, device

# Load the model and device configuration
model, device = load_model()

# The CNN requires 224x224 crops and standard ImageNet normalization to work properly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Deepfake Auditor")
st.markdown("Upload a video to check if it's a real or deepfake. We'll show you why we think so using visual and frequency analysis.")

uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Streamlit uploads are held in memory, but OpenCV needs a physical file path
    # to read video frames. We bridge this by dumping the buffer into a temp file.
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    tfile.close()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Video")
        st.video(video_path)
        
    with col2:
        st.subheader("Analysis Results")
        
        if st.button("Analyze Video"):
            with st.spinner("Extracting frames and executing model inference..."):
                cap = cv2.VideoCapture(video_path)
                fake_probabilities = []
                frames_processed = 0
                all_analyzed_frames = []
                
                # Process variables to track the singular face that looked the *most* fake.
                # We do this because explainability on a harmless frame isn't useful for the user.
                highest_fake_prob = -1.0
                best_face = None
                best_tensor = None
                
                # Reading every single frame is too computationally expensive for a live app.
                # Here we figure out a skip interval to sample roughly 10 spread-out frames.
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                skip = max(1, total_frames // 10) 
                
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                        
                    if i % skip == 0:
                        face = extract_face(frame)
                        
                        if face is not None:
                            face_pil = Image.fromarray(face)
                            input_tensor = transform(face_pil).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output = model(input_tensor)
                                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                                fake_prob = probabilities[1].item() * 100
                                fake_probabilities.append(fake_prob)
                                all_analyzed_frames.append((face, fake_prob))

                            frames_processed += 1
                            
                            # If this face scored higher on the fake scale than previous ones,
                            # save it. We'll use this specific face later for the heatmaps and FFT.
                            if fake_prob > highest_fake_prob:
                                highest_fake_prob = fake_prob
                                best_face = face
                                best_tensor = input_tensor
                            
                cap.release()
                
                # Present results
                if frames_processed > 0:
                    avg_fake_prob = sum(fake_probabilities) / len(fake_probabilities)
                    
                    st.write(f"Frames analyzed: **{frames_processed}**")

                    st.markdown("---")
                    st.subheader("All Analyzed Frames")
                    st.write("Here are the individual faces extracted and their respective deepfake scores:")

                    # Create a grid of 5 columns
                    cols = st.columns(5)
                    for idx, (face_img, prob) in enumerate(all_analyzed_frames):
                        with cols[idx % 5]:
                            # Color-code the caption based on the result
                            label = "FAKE" if prob > 50 else "REAL"
                            st.image(face_img, caption=f"{label}: {prob:.1f}%")
                    
                    if avg_fake_prob > 50:
                        st.error(f"Deepfake Detected! ({avg_fake_prob:.1f}% confidence)")
                        st.progress(int(avg_fake_prob))
                    else:
                        st.success(f"Looks Authentic! ({(100 - avg_fake_prob):.1f}% confidence)")
                    
                    st.markdown("---")
                    
                    st.subheader("Spatial Explanation (Grad-CAM)")
                    st.write("This heatmap shows which parts of the face led to the prediction. The bright red and yellow areas are our main focus points.")
                    
                    with st.spinner("Generating spatial heatmap..."):
                        heatmap_img = generate_gradcam_heatmap(model, best_tensor, best_face)
                        
                        cam_col1, cam_col2 = st.columns(2)
                        with cam_col1:
                            st.image(best_face, caption="Extracted Face", width='stretch')
                        with cam_col2:
                            st.image(heatmap_img, caption="Focus Heatmap", width='stretch')
                            
                    st.markdown("---")
                    st.subheader("Ablation Testing")
                    st.write("To verify the model's focus, we obscure the highly activated regions and see how much the confidence drops.")
                    
                    with st.spinner("Running ablation test..."):
                        obscured_face, new_fake_prob = perform_ablation(model, best_face, heatmap_img, device)
                        delta = highest_fake_prob - new_fake_prob
                        
                        ab_col1, ab_col2 = st.columns(2)
                        with ab_col1:
                            st.image(obscured_face, caption="Masked Face")
                        with ab_col2:
                            st.metric(label="New Deepfake Confidence", value=f"{new_fake_prob:.1f}%", delta=f"-{delta:.1f}%")

                else:
                    st.warning("Couldn't find clear faces in this video. Please make sure the lighting is okay and faces are visible.")
                


    # Cleanup the temporary video file
    try:
        if os.path.exists(video_path):
            os.unlink(video_path)
    except PermissionError:
        pass