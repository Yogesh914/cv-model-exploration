import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import mediapipe as mp
from torchvision import transforms

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file"""

    def __init__(self, n_frames=None, resize=None):
        """Constructor"""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        self.n_frames = n_frames
        self.resize = resize
        self.face_transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __call__(self, filename):
        """Load frames from videos and detect faces"""

        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.arange(0, v_len - 1, self.n_frames).astype(int)
        
        faces = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    frame = cv2.resize(frame, (0, 0), fx=self.resize, fy=self.resize)

                results = self.face_detection.process(frame)
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, iw-x)
                        h = min(h, ih - y)

                        if w>0 and h>0: 
                            face = frame[y:y+h, x:x+w]
                            face_pil = Image.fromarray(face)
                            face_tensor = self.face_transform(face_pil)
                            faces.append(face_tensor)
                else:
                    continue
        
        v_cap.release()

        return faces
