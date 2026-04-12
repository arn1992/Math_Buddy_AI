import os
import asyncio
import pyaudio
import numpy as np
import time
from faster_whisper import WhisperModel
import edge_tts
import uuid
import sys
import threading # Import threading to run blocking input in a separate thread

# Global variable for Whisper model. It will be initialized once.
whisper_model = None
try:
    # Load the Whisper model once when the script starts
    print("Loading Faster Whisper model (base.en)... This may take a moment.")
    # Changed from "tiny.en" to "base.en" for better accuracy.
    # You can change to "small.en", etc., for even better accuracy if your system can handle it.
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")  # Using CPU for broader compatibility
    print("Faster Whisper model loaded.")
except Exception as e:
    print(f"Failed to load Faster Whisper model: {e}")
    print(
        "⚠️ Voice input will not be available. Please ensure model files can be downloaded and enough memory is available.")
    # Set to None so main script can check this and fall back to text input
    whisper_model = None

# --- Text-to-Speech (TTS) Configuration ---
# Ensure EDGE_TTS_VOICE is set in your environment variables, e.g., 'en-US-JennyNeural'
# Fallback to a common voice if not set
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")


async def speak_message(text: str):
    """
    Converts text to speech and plays it.
    Uses edge-tts to generate audio and a system command for playback.
    """
    if not text:
        return

    # Generate a unique filename for the audio file
    audio_filename = f"temp_audio_{uuid.uuid4()}.mp3"
    try:
        # Use edge-tts to create the audio file
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        await communicate.save(audio_filename)

        # Play the audio file using a system command (platform-dependent)
        if os.name == 'nt':  # Windows
            os.system(f"start {audio_filename}")
        elif os.name == 'posix':  # macOS, Linux
            os.system(f"afplay {audio_filename}")  # macOS
            # For Linux, you might use 'aplay' or 'mpg123'
            # os.system(f"mpg123 {audio_filename}")
        else:
            print(f"Audio playback not supported on this OS: {os.name}")

        # Small delay to allow audio to start playing
        await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error speaking message: {e}")
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_filename):
            os.remove(audio_filename)


async def get_voice_input_local_whisper(timeout: int = 20) -> str:
    """
    Records audio from the microphone, transcribes it using Faster Whisper,
    and returns the transcribed text. Includes a timeout for silence.
    Returns "TIMEOUT_ERROR" if no speech is detected within the timeout.
    Returns "ERROR_VOICE_INPUT_..." for other voice input errors.
    Returns an empty string "" if interrupted by pressing Enter.
    """
    if whisper_model is None:
        return "ERROR_WHISPER_MODEL_NOT_LOADED"

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Whisper's preferred sample rate

    p = pyaudio.PyAudio()
    stream = None
    frames = []
    transcribed_text = ""
    start_time = time.time()

    # Event to signal when Enter is pressed
    enter_pressed_event = threading.Event()

    def listen_for_enter():
        """Function to run in a separate thread to listen for Enter key press."""
        try:
            input() # This blocks until Enter is pressed
        except EOFError: # Catch EOFError if stdin is closed unexpectedly
            pass
        finally:
            enter_pressed_event.set() # Set the event when Enter is pressed or input stream closes

    # Start the thread to listen for Enter key press
    enter_thread = threading.Thread(target=listen_for_enter, daemon=True)
    enter_thread.start()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print(f"Listening for {timeout} seconds (Press Enter to finish early)...")
        await speak_message(f"Listening for {timeout} seconds. You can press Enter to finish early.")

        while True:
            current_time = time.time()

            # Check for overall timeout
            if current_time - start_time > timeout:
                print(f"Timeout: No speech detected within {timeout} seconds.")
                transcribed_text = "TIMEOUT_ERROR"
                break

            # Check if Enter was pressed (non-blocking check of the event)
            if enter_pressed_event.is_set():
                print("\nInput finished (Enter pressed).")
                break # Exit the audio recording loop

            try:
                # Yield control to the event loop to allow the Enter key thread to signal
                await asyncio.sleep(0.01) # Small sleep to allow other tasks (like the Enter listener) to run

                # Read audio data. Removed the 'timeout' argument as it's causing the error.
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                print(f"IOError during stream read: {e}. This might indicate an audio input problem.")
                transcribed_text = f"ERROR_VOICE_INPUT_IOERROR_{e}"
                break # Break on IOError
            except Exception as e:
                print(f"Unexpected error during stream read: {e}")
                transcribed_text = f"ERROR_VOICE_INPUT_UNEXPECTED_{e}"
                break

    except Exception as e: # This outer except will catch other unexpected errors during setup/loop
        print(f"Error during voice input setup: {e}")
        return f"ERROR_VOICE_INPUT_SETUP_{e}"
    finally:
        # Ensure stream is stopped and closed only if it was successfully opened and is active
        if stream:
            if stream.is_active(): # Check if stream is active before stopping
                stream.stop_stream()
            stream.close()
        p.terminate() # Terminate PyAudio instance

    # Final transcription of any remaining frames
    if frames:
        wav_bytes = b''.join(frames)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = whisper_model.transcribe(audio_np, beam_size=5)
        # Append to existing transcribed_text if it's not an error/timeout
        if not transcribed_text.startswith("ERROR_") and transcribed_text != "TIMEOUT_ERROR":
            transcribed_text += " " + " ".join([segment.text for segment in segments]).strip()
        # If transcribed_text is already an error/timeout, it remains as is.
    elif not transcribed_text: # If no frames were recorded and no error/timeout was set
        transcribed_text = "" # Ensure it's an empty string if nothing was recorded

    return transcribed_text.strip()
