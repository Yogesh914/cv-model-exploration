import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from moviepy.editor import VideoFileClip
import numpy as np
import os
from pydub import AudioSegment
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import noisereduce as nr
import warnings
import sys

"""
Transcribes a list of videos using the OpenAI Whisper model and saves the transcriptions
as a new column in the input dataframe CSV.

Note: Using a <video_folder> that contains the video files is necessary.

Usage: transcribe_videos.py --input_csv <input_csv> 

Options:
    --input_csv       Path to the input CSV file containing video file paths.
"""

warnings.filterwarnings("ignore")

model = AutoModelForSpeechSeq2Seq.from_pretrained("./models/whisper_large_v3")
processor = AutoProcessor.from_pretrained("./models/whisper_large_v3")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=440,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device=device,
    torch_dtype=torch.float16
)

def process_video(video_file):
    if video_file.endswith('.mp4'):
        audio_segment = AudioSegment.from_file(video_file, format="mp4")
    else:
        audio_segment = AudioSegment.from_file(video_file, format="mov")
    audio_segment = audio_segment.set_frame_rate(16000)
    audio_array = np.array(audio_segment.get_array_of_samples())

    if audio_segment.channels == 2:
        audio_array = audio_array.reshape((-1, 2))
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32) / (2**15)

    audio_array = nr.reduce_noise(y=audio_array, sr=16000)

    return audio_array, audio_segment.frame_rate

class AudioDataset(Dataset):
    def __init__(self, video_files):
        self.video_files = video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        audio_array, sample_rate = process_video(video_file)
        audio_tensor = torch.from_numpy(audio_array)
        return audio_tensor, sample_rate

def main(input_csv_path):
    df = pd.read_csv(input_csv_path)
    video_files = df['filepath'].tolist()

    dataset = AudioDataset(video_files)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    transcriptions = []
    for batch in dataloader:
        audio_tensors, sample_rates = batch
        audio_tensor = audio_tensors[0]
        sample_rate = sample_rates[0].item()
        audio_numpy = audio_tensor.numpy()
        audio_dict = {"path": "audio_file", "array": audio_numpy, "sampling_rate": sample_rate}
        result = pipe(audio_dict, batch_size=1)
        transcription = result["text"]
        transcriptions.append(transcription)

    df['Transcriptions'] = transcriptions
    df.to_csv(input_csv_path, index=False)
    print(f"Transcriptions added to {input_csv_path}")

if __name__ == "__main__":
    input_csv_path = sys.argv[1]
    main(input_csv_path)