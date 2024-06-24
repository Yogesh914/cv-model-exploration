import os
from pydub import AudioSegment
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.models import wavlm_large
from tqdm import tqdm
import traceback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

model = wavlm_large()
model.eval()
model.to(device)

def video_to_audio(video_path, audio_path):
    video = AudioSegment.from_file(video_path)
    video.export(audio_path, format="wav")

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    return waveform

def extract_embeddings(waveform, model):
    with torch.no_grad():
        embeddings = model(waveform)
    return embeddings[0].cpu().detach().numpy()

def main(input_csv_path="../data/file_paths.csv", output_np_folder="../data/audio_embeddings", log_path="../data/failed_audio.log"):
    
    df = pd.read_csv(input_csv_path)
    failed_audio = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
        video_path = row['file_path']
        try:
            audio_path = video_path.replace(os.path.splitext(video_path)[-1], ".wav")
            video_to_audio(video_path, audio_path)
            waveform = preprocess_audio(audio_path).to(device)
            embeddings = extract_embeddings(waveform, model)
            del waveform
            os.remove(audio_path)
        except Exception:
            print(f"Error processing video: {video_path}")
            print(traceback.format_exc())
            failed_audio.append(video_path)
            continue
        file_name = os.path.basename(audio_path).replace('.wav', '.npy')
        output_path = os.path.join(output_np_folder, file_name)
        np.save(output_path, embeddings)

    with open(log_path, "w") as log_file:
        for video in failed_audio:
            log_file.write(f"{video}\n")


if __name__ == "__main__":
    main()