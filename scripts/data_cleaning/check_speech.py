import os
import pydub
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import pandas as pd

recognizer = sr.Recognizer()

def video_to_audio(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def transcribe(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None


def analyze_transcription(transcription):
    if not transcription:
        return False
    words = transcription.split()
    if len(words) < 3:
        return False
    gibberish_ratio = sum (1 for word in words if len(word) <= 2) / len(words)
    if gibberish_ratio > 0.5:
        return False
    return True


def main():
    df = pd.read_csv("../data/file_paths.csv").head(3)
    quality_results = []

    for video_path in df["file_path"]:
        print(video_path)
        if os.path.exists(video_path) and video_path.endswith(('.mp4', '.MOV')):
            audio_path = video_path.replace(os.path.splitext(video_path)[-1], ".wav")
            #video_path = video_path.replace("/", "\\")
            #audio_path = audio_path.replace("/", "\\")
            
            video_to_audio(video_path, audio_path)
            transcription = transcribe(audio_path)
            is_good = analyze_transcription(transcription)
            quality_results.append(is_good)
            os.remove(audio_path)
        else:
            quality_results.append(False)
    
    df['quality'] = quality_results
    df.to_csv("../data/quality_check.csv", index=False)

if __name__ == "__main__":
    main()

