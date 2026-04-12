import os
import asyncio
import pyaudio
import time
import uuid
import threading
import httpx  # For making HTTP requests to Google Cloud TTS (batch)
import base64  # For encoding audio data
import traceback # Re-added for debugging purposes

# Imports for Google Cloud Streaming Speech-to-Text
from google.cloud import speech # The main client library for Speech-to-Text

# --- Google Cloud API Configuration ---
# IMPORTANT: Replace with your actual Google Cloud API Key
# For client libraries, it's often better to authenticate via environment variables
# or service accounts. For simplicity and direct control for this console app, we'll keep it here.
GOOGLE_API_KEY = "AIzaSyCsrPAIytMCgeO6pxLNEqLOqQg5dpfPp5E"

# Google Cloud Text-to-Speech API endpoint (still using for httpx)
TTS_API_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"

# --- Text-to-Speech (TTS) Configuration ---
TTS_VOICE_CONFIG = {
    "languageCode": "en-US",
    "name": "en-US-Wavenet-D",
    "ssmlGender": "FEMALE"
}
TTS_AUDIO_CONFIG = {
    "audioEncoding": "MP3"
}

async def speak_message(text: str):
    """
    Converts text to speech using Google Cloud Text-to-Speech API and plays it.
    """
    if not text:
        return

    payload = {
        "input": {"text": text},
        "voice": TTS_VOICE_CONFIG,
        "audioConfig": TTS_AUDIO_CONFIG
    }

    audio_filename = f"temp_audio_{uuid.uuid4()}.mp3"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(TTS_API_URL, json=payload, timeout=10.0)
            response.raise_for_status()

            audio_content_base64 = response.json().get("audioContent")
            if audio_content_base64:
                audio_data = base64.b64decode(audio_content_base64)
                with open(audio_filename, "wb") as f:
                    f.write(audio_data)

                if os.name == 'nt':  # Windows
                    os.system(f"start {audio_filename}")
                elif os.name == 'posix':  # macOS, Linux
                    os.system(f"afplay {audio_filename}")
                else:
                    print(f"Audio playback not supported on this OS: {os.name}")

                await asyncio.sleep(0.1)
            else:
                print("Error: No audio content received from Google Cloud TTS.")

    except httpx.RequestError as e:
        print(f"Error speaking message (HTTP request failed): {e}")
    except httpx.HTTPStatusError as e:
        print(f"Error speaking message (HTTP status error): {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"Error speaking message: {e}")
    finally:
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

async def get_voice_input_google_cloud(timeout: int = 50) -> str:
    """
    Records audio from the microphone using PyAudio and sends it to
    Google Cloud Streaming Speech-to-Text API for real-time transcription.
    The recording stops when Enter is pressed or the global timeout is reached.
    Returns the full transcribed text.
    Returns "TIMEOUT_ERROR" if no speech is detected within the timeout.
    Returns "ERROR_VOICE_INPUT_..." for other voice input errors.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Google Cloud STT preferred sample rate

    transcribed_text = ""
    start_time = time.time()

    # Event to signal when Enter is pressed
    enter_pressed_event = threading.Event()

    def listen_for_enter():
        """Function to run in a separate thread to listen for Enter key press."""
        try:
            input()
        except EOFError:
            pass
        finally:
            enter_pressed_event.set()

    enter_thread = threading.Thread(target=listen_for_enter, daemon=True)
    enter_thread.start()

    # ---------- SYNCHRONOUS AUDIO GENERATOR -----------------
    def audio_generator():
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()

            # --- Microphone Setup: Using default input device, mirroring Whisper setup ---
            # This relies on the OS to select the default microphone.
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            if not stream.is_active():
                print("🚨 PyAudio stream is not active immediately after opening. Potential microphone issue or resource conflict.")
                return

            print("🎤 Listening via default input device...")

            while True:
                current_loop_time = time.time()
                if current_loop_time - start_time > timeout:
                    print(f"\nTimeout: Recording limit reached after {timeout} seconds. Stopping audio generation.")
                    break
                if enter_pressed_event.is_set():
                    print("\nInput finished (Enter pressed). Stopping audio generation.")
                    break

                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    if not data or len(data) == 0:
                        time.sleep(0.01)
                        continue
                    yield speech.StreamingRecognizeRequest(audio_content=data)
                    time.sleep(0.01)
                except IOError as e:
                    print(f"IOError during stream read in generator: {e}. Audio input problem. Stopping audio generation.")
                    break
                except Exception as e:
                    print(f"Unexpected error during audio generation in generator: {e}. Stopping audio generation.")
                    break
        except Exception as e:
            print(f"🚨 Error during PyAudio or stream setup in generator: {e}")
            traceback.print_exc()
        finally:
            if stream:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            if p:
                p.terminate()

    try:
        client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False
        )

        print(f"Listening for {timeout} seconds (Press Enter to finish early)...")

        requests_iterable = audio_generator()
        responses_iterator = client.streaming_recognize(streaming_config, requests_iterable)

        received_final_result = False
        response_start_time = time.time()
        response_processing_timeout = timeout + 10

        try:
            for response in responses_iterator:
                if time.time() - response_start_time > response_processing_timeout:
                    print(f"Timeout: No final STT result received within {response_processing_timeout} seconds.")
                    break
                if not response.results:
                    continue
                result = response.results[0]
                if result.is_final:
                    if result.alternatives and len(result.alternatives) > 0:
                        transcribed_text = result.alternatives[0].transcript
                        received_final_result = True
                        break
        except Exception as e:
            print(f"🚨 Error iterating STT responses: {e}")
            traceback.print_exc()
            transcribed_text = f"ERROR_STT_RESPONSE_ITERATION_{e}"

        if not received_final_result and not transcribed_text:
            transcribed_text = "TIMEOUT_ERROR"

    except Exception as e:
        print(f"🚨 Error during streaming STT process: {e}")
        traceback.print_exc()
        transcribed_text = f"ERROR_STT_STREAMING_{e}"
    finally:
        pass

    return transcribed_text.strip()
