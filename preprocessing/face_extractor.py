import cv2
import torch
from facenet_pytorch import MTCNN

# We're forcing MTCNN to run on the CPU here. If you try to run it on Mac's MPS backend, 
# it sometimes throws weird tensor mismatches during the alignment phase, so CPU is just safer.
device = torch.device('cpu')
# We only care about the most prominent face in each frame to keep things simple.
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)

def extract_face(frame):
    """Extracts and resizes the primary face from a BGR video frame."""
    
    # MTCNN expects standard RGB format, but OpenCV gives us BGR naturally, so we flip it.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)
    
    if boxes is None:
        return None
        
    # Grab the bounding box coordinates for the best face
    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]
    
    # Better to clamp these coordinates so we don't accidentally try to index outside the array
    # if the network throws a box slightly out of frame bounds.
    ih, iw, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    
    face_crop = rgb_frame[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return None
        
    return cv2.resize(face_crop, (224, 224))