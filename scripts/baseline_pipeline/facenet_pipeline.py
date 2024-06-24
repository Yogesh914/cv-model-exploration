from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
import cv2
import numpy as np
import pandas as pd
import torch
import os
import shutil
from MediaPipeDetection import DetectionPipeline
from tqdm import tqdm 
import traceback


torch_home = get_torch_home()
model_dir = os.path.join(torch_home, 'checkpoints')
os.makedirs(model_dir, exist_ok=True)
cached_file = os.path.join(model_dir, os.path.basename("../models/vggface2.pt"))
if not os.path.exists(cached_file):
    shutil.copy("../models/20180402-114759-vggface2.pt", model_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_facial_embeddings(video_path, detection_pipeline):

    faces = detection_pipeline(video_path)
    embeddings = []
    for face in faces:
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face)
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
            del face
    
    torch.cuda.empty_cache()
    return np.array(embeddings).squeeze()

def main(input_csv_path="../data/file_paths.csv", output_np_folder="../data/facial_embeddings", log_path="../data/failed_videos.log"):

    df = pd.read_csv(input_csv_path)
    #df = df[df['file_path'] == "../data/video/20010/20010_01.mp4"]
    failed_videos = []
    detection_pipeline = DetectionPipeline(resize=0.25)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
        file_path = row['file_path']
        try:
            outputs = get_facial_embeddings(file_path, detection_pipeline)
        except Exception:
            print(f"Error processing video: {file_path}")
            print(traceback.format_exc())
            failed_videos.append(file_path)
            continue
        file_name = os.path.basename(file_path).replace('.mp4', '.npy').replace('.MOV', '.npy')
        output_path = os.path.join(output_np_folder, file_name)
        np.save(output_path, outputs)

    with open(log_path, "w") as log_file:
        for video in failed_videos:
            log_file.write(f"{video}\n")


if __name__ == "__main__":
    main()
    
