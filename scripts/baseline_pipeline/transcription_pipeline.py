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
from tqdm import tqdm

"""
Transcribes a list of videos using the OpenAI Whisper model and saves the transcriptions
as a new column in the input dataframe CSV.

Note: Using a <video_folder> that contains the video files is necessary.

Usage: transcribe_videos.py --input_csv <input_csv> 
"""

warnings.filterwarnings("ignore")

model = AutoModelForSpeechSeq2Seq.from_pretrained("../models/whisper-large-v3")
processor = AutoProcessor.from_pretrained("../models/whisper-large-v3")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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
    torch_dtype=torch.float32
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

def transcribe(path):
    try:
        audio_array, frame_rate = process_video(path)
        transcription = pipe({"array":audio_array, "sampling_rate":frame_rate}, generate_kwargs={"language": "english"})["text"]
        return transcription
    except Exception as e:
        print(f"Error Processing Video: {path}")
        return None

def main(input_csv_path="../data/file_paths.csv", output_csv_path="../data/transcriptions.csv"):
    df = pd.read_csv(input_csv_path)
    df["error"] = False
    df['transcription'] = None

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        transcription = transcribe(row['file_path'])
        if transcription is None:
            df.at[idx, "error"] = True
        df.at[idx, "transcription"] = transcription

    df.to_csv(output_csv_path, index=False)
    print(f"Transcriptions added to {output_csv_path}")

if __name__ == "__main__":
    main()