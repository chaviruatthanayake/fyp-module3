import os
import whisper
from moviepy.editor import VideoFileClip

os.environ["IMAGEIO_FFMPEG_EXE"] = "C:\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"

def process_video(file_path, upload_id, output_dir) -> str:
    audio_path = os.path.join(output_dir, f"{upload_id}.wav")
    video = None
    try:
        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile(audio_path, logger=None)

        # Load Whisper model
        model = whisper.load_model("base")  # or "medium", "large" for better accuracy
        result = model.transcribe(audio_path)

        return result["text"]

    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        return ""

    finally:
        if video:
            video.close()
        if 'audio' in locals():
            audio.close()
