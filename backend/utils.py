import os
import cv2
import subprocess

def extract_audio(video_path, output_wav_path):
    subprocess.call([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000",
        "-ac", "1", "-vn", output_wav_path
    ])

def extract_frames(video_path, output_dir, step=30):
    video_capture = cv2.VideoCapture(video_path)
    count = 0
    saved = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % step == 0:
            path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
        count += 1
    video_capture.release()
    return saved
