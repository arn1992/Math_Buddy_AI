from faster_whisper import WhisperModel
import pyaudio
import time
import numpy as np
import uuid
import edge_tts
import os
import asyncio  # Required for async operations

# --- Local Speech-to-Text (STT) Functionality using Faster Whisper ---
try:
    print("Loading Faster Whisper model (this may take a moment)...")
    # Using device="cpu" explicitly for broader compatibility.
    # If you have a CUDA-enabled GPU and appropriate drivers/libraries, you can try device="cuda"
    # and compute_type="float16" for potentially faster inference.
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Faster Whisper model loaded.")
except Exception as e:
    print(f"🚨 Error loading Faster Whisper model: {e}")
    print("Please ensure you have enough memory and the model files can be downloaded.")
    print("If you encounter issues, try a smaller model like 'tiny' or 'base.en'.")
    whisper_model = None  # Set to None if loading fails, so we can handle it later

def get_voice_input_local_whisper(listen_duration=10):
    """
    Records audio from the microphone using PyAudio and transcribes it using Faster Whisper.

    Args:
        listen_duration (int): The duration of the recording in seconds.

    Returns:
        str: The transcribed text, or an error message if transcription fails.
    """
    if whisper_model is None:
        return "ERROR_WHISPER_MODEL_NOT_LOADED"

    # Audio settings
    RATE = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024  # Buffer size for audio chunks

    p = None
    stream = None
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print(f"🎙️ Listening for {listen_duration} seconds... (Press Ctrl+C to stop early and continue the program)")

        frames = []
        start_time = time.time()

        while time.time() - start_time < listen_duration:
            try:
                # Read audio data from the stream. exception_on_overflow=False prevents
                # an immediate crash if the buffer overflows, allowing for more robust handling.
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            except IOError as e:
                # This can happen if the audio device is disconnected, busy, or has issues.
                print(f"🚨 Audio stream error: {e}. Attempting to stop recording.")
                break  # Exit loop on persistent audio error
            except KeyboardInterrupt:
                # Catch Ctrl+C specifically during recording to stop gracefully
                print("\nRecording interrupted by user (Ctrl+C). Processing available audio...")
                break  # Exit the recording loop

        print("🔁 Processing audio...")

        # Concatenate recorded frames and convert to float32
        if not frames:
            print("No audio data recorded.")
            return "ERROR_NO_AUDIO_RECORDED"

        # Normalize audio to float32 between -1.0 and 1.0, as expected by Whisper
        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0

        # Transcribe using Faster Whisper
        # You can specify language="en" if you only expect English, which can improve accuracy.
        # beam_size can be adjusted for accuracy vs. speed. 5 is a good default.
        segments, _ = whisper_model.transcribe(audio_np, language="en", beam_size=5)

        # Join all transcribed segments into a single string
        transcribed_text = " ".join([segment.text.strip() for segment in segments])

        if not transcribed_text.strip():
            print("Transcription was empty.")
            return "ERROR_EMPTY_TRANSCRIPTION"

        print(f"🎙️ Transcribed: '{transcribed_text}'")
        return transcribed_text

    except Exception as e:
        print(f"🚨 Error during voice input or transcription: {e}")
        print("Please ensure PyAudio is correctly installed and your microphone is configured.")
        print("On Linux, you might need `sudo apt-get install portaudio19-dev python3-pyaudio`.")
        print("On macOS, `brew install portaudio` then `pip install pyaudio`.")
        print("On Windows, you might need to install Visual C++ build tools or use pre-compiled wheels.")
        return "ERROR_TRANSCRIPTION_FAILED"
    finally:
        # Always ensure the audio stream and PyAudio are properly closed
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()


# --- Text-to-Speech (TTS) Functionality using Edge TTS ---
async def speak_message(text):
    """
    Converts text to speech and plays it using Edge TTS.
    Requires 'edge-tts' library.
    Note: The playback method `os.system('afplay ...')` is macOS specific.
    For other OS, you might need a different command-line player (e.g., 'start' on Windows, 'aplay' or 'mpv' on Linux)
    or a cross-platform Python audio playback library.
    """
    print(f"🔊 Speaking: {text}")
    output_file = f"temp_{uuid.uuid4().hex}.mp3"

    try:
        communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
        await communicate.save(output_file)

        # Play the audio using macOS native player (afplay)
        # IMPORTANT: This command is macOS specific.
        # For Windows, you might use: os.system(f"start {output_file}")
        # For Linux, you might use: os.system(f"aplay {output_file}") or install 'mpv' and use os.system(f"mpv --no-video {output_file}")
        os.system(f"afplay {output_file}")

    except asyncio.CancelledError:
        # This exception is raised when the task is cancelled, typically by Ctrl+C
        print("\n🔊 Speech playback cancelled by user.")
    except Exception as e:
        print(f"🚨 Error during text-to-speech with Edge TTS: {e}")
        print("Please ensure you have 'edge-tts' installed (`pip install edge-tts`).")
        print(
            "Also, check your system's audio playback command (e.g., 'afplay' on macOS, 'start' on Windows, 'aplay' on Linux).")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
        # No time.sleep needed here as await communicate.save() handles waiting for file creation
        # and os.system() blocks until the command finishes.