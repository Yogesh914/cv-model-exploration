import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.editor import VideoFileClip
import numpy as np
import os
from pydub import AudioSegment
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import noisereduce as nr

"""
Transcribes a list of videos using the OpenAI Whisper model and saves the transcriptions
as a new column in the input dataframe CSV.

Requires the following packages to be installed:
- torch
- transformers
- moviepy
- numpy
- pandas
- pydub
- noisereduce

Note: Using a <video_folder> that contains the video files is necessary.

Usage: transcribe_videos.py --input_csv <input_csv> --output_csv <output_csv> --video_folder <video_folder>

Options:
    --input_csv       Path to the input CSV file containing video file paths.
    --output_csv      Path to the output CSV file where the processed data will be saved.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, use_safetensors=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

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

def transcribe_videos(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    video_files = df['file_path'].tolist()
    
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
    df.to_csv(output_csv, index=False)
    print(f"Transcriptions saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe videos and add transcriptions to dataframe")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file containing video file paths")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file where the processed data will be saved")

    args = parser.parse_args()
    transcribe_videos(args.input_csv, args.output_csv, args.video_folder)