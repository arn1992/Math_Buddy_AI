import os
import json
import time
import httpx
import traceback
import asyncio  # Required for async operations
import edge_tts
import uuid  # For generating unique filenames for audio

# Imports for Local Speech-to-Text (Faster Whisper)
import numpy as np
import pyaudio  # For microphone input
from faster_whisper import WhisperModel

# Assuming agents.py and functions.py are in the same directory or accessible
# You'll need to ensure these imports match your actual file structure
from crewai import Task, Crew, Process
from functions import *

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


# --- Main Workflow Runner ---
async def run_math_buddy():  # Changed to async function
    try:  # Added top-level try-except for KeyboardInterrupt
        welcome_message = "Welcome to your 4th Grade Math Buddy! Let's conquer math together!"
        print(f"👋 {welcome_message}")
        await speak_message(welcome_message)
        print("Type your math problem to begin. Type 'quit' to stop anytime.\n")

        # Check if Whisper model loaded successfully at startup
        if whisper_model is None:
            print("⚠️ Voice input will not be available due to Faster Whisper model loading failure.")

        while True:  # Outer loop for new problems
            restart_session_flag = False  # Flag to signal a full session restart

            input_choice_message = "How would you like to input the problem or control the session?"
            print(f"\n{input_choice_message}")
            await speak_message(input_choice_message)
            # Updated prompt for numerical choices
            user_choice_initial = input(
                "1. Quit\n2. Restart\n3. Continue with text\n4. Continue with voice\nEnter your choice (1-4): ").strip()

            problem = ""
            if user_choice_initial == "1":  # Check for numerical input
                goodbye_message = "See you next time, Math Explorer!"
                print(f"👋 {goodbye_message}")
                await speak_message(goodbye_message)
                break  # Exit the main loop and end the program
            elif user_choice_initial == "2":  # Check for numerical input
                restart_message = "Okay, let's restart with a fresh problem!"
                print(f"🔁 {restart_message}")
                await speak_message(restart_message)
                continue  # Restart the outer loop for a new problem
            elif user_choice_initial == "3":  # Check for numerical input
                problem = input("Enter your 4th-grade math problem: ")
            elif user_choice_initial == "4":  # Check for numerical input
                if whisper_model is None:
                    voice_unavailable_message = "Voice input is not available. Please choose 'continue with text'."
                    print(voice_unavailable_message)
                    await speak_message(voice_unavailable_message)
                    continue
                await speak_message("Please speak your math problem now.")
                print("🎙️ Please speak your math problem now...")
                problem = get_voice_input_local_whisper().lower()
                if problem.startswith("ERROR_"):
                    error_message = f"Voice input failed: {problem}. Please try typing instead or check your setup."
                    print(error_message)
                    await speak_message(error_message)
                    continue
            else:
                invalid_choice_message = "Invalid choice. Please enter a number between 1 and 4."
                print(invalid_choice_message)
                await speak_message(invalid_choice_message)
                continue

            if not problem.strip():  # Handle empty input after method selection
                empty_problem_message = "Problem cannot be empty. Please try again."
                print(empty_problem_message)
                await speak_message(empty_problem_message)
                continue

            conversation_history = []
            conversation_history.append(("Student", problem))

            # These variables will store the AI_Educator's internal state and outputs
            # Moved inside the outer while True loop to reset for each new problem
            problem_breakdown_data = {}
            learning_steps = []
            student_learning_state = {
                "misconceptions_this_session": [],
                "current_problem_steps_completed": 0,
                "overall_difficulty": "neutral"
            }
            math_expression = "Mathematical representation not available."  # Initialize or reset for new problem

            planning_message = "AI Educator is initiating problem comprehension and planning..."
            print(f"\n{planning_message}\n")
            await speak_message(planning_message)

            # --- NEW TASK: Mathematical Problem Representation ---
            # This task will extract and present the mathematical core of the problem.
            mathematical_representation_task = Task(
                description=(
                    f"Analyze the student's math problem: '{problem}'.\n"
                    "Your task is to extract the core mathematical part of the problem and represent it in a clear, concise mathematical way.\n"
                    "This could be an equation, a set of operations, a visual description of quantities, or a clear statement of the mathematical question.\n"
                    "Focus ONLY on the mathematical essence, stripping away the word problem narrative.\n"
                    "Example output: '5 + 3 = ?' or 'Find the difference between 10 and 4.' or 'A rectangle with length 8 and width 5, find area.'\n"
                    "Do NOT solve the problem or provide steps. Just the mathematical representation."
                ),
                agent=ai_educator_agent,
                expected_output="A clear mathematical representation of the problem (e.g., an equation, a description of operations, or a visual representation)."
            )

            # Run a small crew just for this new task
            representation_crew = Crew(
                agents=[ai_educator_agent],
                tasks=[mathematical_representation_task],
                process=Process.sequential,
                verbose=True
            )

            try:
                math_representation_output = run_crew_with_retry(representation_crew,
                                                                 "mathematical problem representation")
                if math_representation_output:
                    math_expression = math_representation_output.raw_output if hasattr(math_representation_output,
                                                                                       'raw_output') else str(
                        math_representation_output)
                    # We'll print it later, inside the step loop
                    conversation_history.append(("AI Educator (Math Representation)", math_expression))
                else:
                    warning_math_rep = "Could not generate a mathematical representation for the problem."
                    print(f"\n⚠️ {warning_math_rep}")
                    await speak_message(warning_math_rep)
            except Exception as e:
                error_math_rep = f"Error generating mathematical representation: {e}"
                print(f"🚨 {error_math_rep}")
                await speak_message(error_math_rep)
                traceback.print_exc()

            # --- Phase 1: Problem Comprehension & Diagnostic Analysis ---
            # The AI_Educator's task to analyze the problem and output its internal structured understanding.
            initial_analysis_task = Task(
                description=(
                    f"As the AI Educator, your first comprehensive task is to analyze the student's problem: '{problem}'.\n\n"
                    "Your output MUST be a JSON object conforming to the 'problem_breakdown' structure from your internal thought process.\n"
                    "Example of expected output (full JSON object with only 'problem_breakdown' structure populated):\n"
                    "```json\n"
                    "{\n"
                    "  \"scaffolding_stage\": \"initial_analysis\",\n"
                    "  \"action_taken\": \"problem_breakdown\",\n"
                    "  \"educator_response\": {\n"
                    "    \"tone\": \"neutral\",\n"
                    "    \"message\": \"I've analyzed your problem and mapped out the core elements for our learning journey.\",\n"
                    "    \"structured_data\": {\n"
                    "      \"problem_type\": \"arithmetic_word_problem\",\n"
                    "      \"core_concepts_required\": [\"addition\", \"subtraction\"],\n"
                    "      \"prerequisite_knowledge_check\": [\"reading comprehension\", \"basic number operations\"],\n"
                    "      \"potential_misconceptions\": [\"confusing operations\", \"missing key information\"],\n"
                    "      \"key_information_given\": [\"number of items\", \"items given away\"],\n"
                    "      \"explicit_questions\": [\"how many remaining\"],\n"
                    "      \"high_level_approaches\": [\"drawing a diagram\", \"writing an equation\"]\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "```"
                ),
                agent=ai_educator_agent,
                expected_output="A JSON object following the specified 'problem_breakdown' structure within the overall output format.",
                output_json=AIResponse
            )

            initial_crew_analysis = Crew(
                agents=[ai_educator_agent],
                tasks=[initial_analysis_task],
                process=Process.sequential,
                verbose=True
            )

            try:
                initial_analysis_output_obj = run_crew_with_retry(initial_crew_analysis, "initial problem analysis")
                if initial_analysis_output_obj is None:
                    failed_analysis_message = "Failed to get initial problem analysis. Moving to the next problem."
                    print(f"\n{failed_analysis_message}")
                    await speak_message(failed_analysis_message)
                    continue

                # Robustly parse the LLM's raw output
                parsed_raw_output = parse_llm_output_robustly(
                    initial_analysis_output_obj.raw_output if hasattr(initial_analysis_output_obj,
                                                                      'raw_output') else str(
                        initial_analysis_output_obj))

                if parsed_raw_output is None:
                    parse_error_message = f"Error: Could not parse initial analysis response as JSON or Python literal. Could not process initial problem analysis. Try a different problem."
                    print(
                        f"🚨 {parse_error_message} Raw output:\n{initial_analysis_output_obj.raw_output if hasattr(initial_analysis_output_obj, 'raw_output') else str(initial_analysis_output_obj)}")
                    await speak_message(parse_error_message)
                    continue

                try:
                    # Convert parsed dict/list back to a JSON string for Pydantic validation
                    initial_analysis_data = AIResponse.model_validate_json(json.dumps(parsed_raw_output))
                except Exception as e:
                    validation_error_message = f"Error validating initial analysis response against Pydantic model. Could not process initial problem analysis. Try a different problem."
                    print(f"🚨 {validation_error_message} Parsed data attempting to validate:\n{parsed_raw_output}")
                    await speak_message(validation_error_message)
                    traceback.print_exc()
                    continue

                # Access data from the validated Pydantic model
                problem_breakdown_data = initial_analysis_data.educator_response.structured_data.problem_breakdown.model_dump() if initial_analysis_data.educator_response.structured_data.problem_breakdown else {}

            except Exception as e:
                unhandled_error_message = f"An unhandled error occurred during initial problem analysis: {e}. It seems we hit a major snag. Let's try starting fresh with a new problem."
                print(f"🚨 {unhandled_error_message}")
                await speak_message(unhandled_error_message)
                traceback.print_exc()
                continue

            # --- Phase 1.5 (Still part of initial setup): Design Learning Steps ---
            # The AI_Educator then generates the pedagogical steps based on its own analysis.
            design_learning_path_task = Task(
                description=(
                    "As the AI Educator, based on your detailed problem breakdown (provided below), "
                    "your next task is to design a structured, step-by-step learning path for a 4th-grade student. "
                    "This path should guide them through solving the problem independently.\n\n"
                    f"Problem Breakdown: {json.dumps(problem_breakdown_data)}\n\n"
                    "**Purpose:** To dynamically guide the student through the problem-solving process and conceptual understanding.\n"
                    "**Functions (initial phase):**\n"
                    "  * Create a structured, step-by-step learning path (3-5 steps) to guide a 4th-grade student through solving this problem independently.\n"
                    "  * Each step should be phrased as a clear question or instruction that promotes critical thinking and engagement.\n"
                    "  * Do NOT reveal any numeric calculations or the final answer directly.\n"
                    "  * Reflect a logical pedagogical sequence suitable for 9-10 year olds, building understanding gradually.\n"
                    "  * Anticipate common learning hurdles and encourage productive struggle, informed by `predicted_misconceptions` from the breakdown.\n"
                    "  * Ensure these steps build student confidence and align with 4th-grade math practices.\n"
                    "\n"
                    "Your output MUST be a JSON object containing the `scaffolding_stage`, `action_taken`, and `educator_response`.\n"
                    "For `action_taken`, use 'problem_breakdown' (as this is still part of the initial problem setup).\n"
                    "For `structured_data`, include a single key 'learning_steps' with a numbered list of steps.\n"
                    "Example of expected output (full JSON object with 'learning_steps' in structured_data):\n"
                    "```json\n"
                    "{\n"
                    "  \"scaffolding_stage\": \"initial_analysis\",\n"
                    "  \"action_taken\": \"problem_breakdown\",\n"
                    "  \"educator_response\": {\n"
                    "    \"tone\": \"supportive\",\n"
                    "    \"message\": \"Based on our analysis, here’s how we’ll break down this problem:\",\n"
                    "    \"structured_data\": {\n"
                    "      \"learning_steps\": [\n"
                    "        \"1. What is the problem asking you to find?\",\n"
                    "        \"2. What numbers are important and what do they mean?\",\n"
                    "        \"3. What math operation will help you solve this?\"\n"
                    "      ]\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "```"
                ),
                agent=ai_educator_agent,
                context=[initial_analysis_task],  # This explicitly links to the output of Task 1.1
                expected_output="A JSON object following the specified structure with 'learning_steps'.",
                output_json=AIResponse
            )

            steps_crew = Crew(
                agents=[ai_educator_agent],
                tasks=[design_learning_path_task],
                process=Process.sequential,
                verbose=True
            )

            try:
                learning_steps_output_obj = run_crew_with_retry(steps_crew, "designing learning path")
                if learning_steps_output_obj is None:
                    failed_design_message = "Failed to design learning steps. Moving to the next problem."
                    print(f"\n{failed_design_message}")
                    await speak_message(failed_design_message)
                    continue

                parsed_steps_output = parse_llm_output_robustly(
                    learning_steps_output_obj.raw_output if hasattr(learning_steps_output_obj, 'raw_output') else str(
                        learning_steps_output_obj))

                if parsed_steps_output is None:
                    parse_error_steps_message = f"Error: Could not parse learning steps response as JSON or Python literal. Could not process learning steps. Try a different problem."
                    print(
                        f"🚨 {parse_error_steps_message} Raw output:\n{learning_steps_output_obj.raw_output if hasattr(learning_steps_output_obj, 'raw_output') else str(learning_steps_output_obj)}")
                    await speak_message(parse_error_steps_message)
                    continue

                try:
                    learning_steps_data = AIResponse.model_validate_json(json.dumps(parsed_steps_output))
                    learning_steps = learning_steps_data.educator_response.structured_data.learning_steps
                except Exception as e:
                    validation_error_steps_message = f"Error validating learning steps output. Could not process learning steps. Try a different problem."
                    print(
                        f"🚨 {validation_error_steps_message} Parsed data attempting to validate:\n{parsed_steps_output}")
                    await speak_message(validation_error_steps_message)
                    traceback.print_exc()
                    continue

                learning_steps_intro = "Here are the learning steps we will follow:"
                print(f"\n--- AI Educator: {learning_steps_intro} ---")
                await speak_message(learning_steps_intro)
                for i, step in enumerate(learning_steps):
                    # For printing to console, keep the full step_message for clarity
                    print_message = f"Step {i + 1}: {step}"
                    print(f"📘 {print_message}")

                    # For speaking, remove the "X. " prefix from the 'step' content
                    # This assumes 'step' always starts with "X. " where X is a number.
                    spoken_content = step.split('. ', 1)[1] if '. ' in step else step
                    spoken_message = f"Step {i + 1}: {spoken_content}"
                    await speak_message(spoken_message)
                print("\n")

            except Exception as e:
                unhandled_error_steps_message = f"An unhandled error occurred during learning path design: {e}. It seems we hit a major snag. Let's try starting fresh with a new problem."
                print(f"🚨 {unhandled_error_steps_message}")
                await speak_message(unhandled_error_steps_message)
                traceback.print_exc()
                continue

            # --- Phase 2: Adaptive Scaffolding & Targeted Intervention (Interactive Loop) ---
            lets_go_message = "Let's go step-by-step."
            print(lets_go_message)
            await speak_message(lets_go_message)

            current_step_index = 0
            while current_step_index < len(learning_steps):
                current_guidance_step = learning_steps[current_step_index]
                math_problem_display = f"Mathematical Problem: {math_expression}"
                print(f"\n--- {math_problem_display} ---")
                await speak_message(math_problem_display)

                # Remove the leading "X. " from current_guidance_step for speaking
                spoken_guidance_content = current_guidance_step.split('. ', 1)[
                    1] if '. ' in current_guidance_step else current_guidance_step
                current_focus_message_spoken = f"Current focus, Step {current_step_index + 1}: {spoken_guidance_content}"

                print(
                    f"📘 Current focus, Step {current_step_index + 1}: {current_guidance_step}")  # Print full step with number
                await speak_message(current_focus_message_spoken)  # Speak without duplicate number

                # Consolidated input prompt with numerical choices
                user_choice = input(
                    "🧒 Your turn (1. Quit, 2. Restart, 3. Type Answer, 4. Voice Answer, 5. Hint, 6. Done): ").strip()

                user_input_for_history = ""  # This will store the actual content for conversation history

                if user_choice == "1":  # Quit
                    goodbye_message = "See you next time, Math Explorer!"
                    print(f"👋 {goodbye_message}")
                    await speak_message(goodbye_message)
                    return
                elif user_choice == "2":  # Restart
                    restart_message = "Okay, let's restart with a fresh problem!"
                    print(f"🔁 {restart_message}")
                    await speak_message(restart_message)
                    restart_session_flag = True  # Set the flag to true
                    break  # Break out of the inner loop
                elif user_choice == "6":  # Done
                    buddy_response_message = "✅ Fantastic! Moving on to the next step."
                    print(f"💬 AI Educator: {buddy_response_message}")
                    await speak_message(buddy_response_message)
                    conversation_history.append(("AI Educator", buddy_response_message))
                    current_step_index += 1
                    continue
                elif user_choice == "5":  # Hint
                    user_input_for_history = "hint"  # Record the hint command
                    conversation_history.append(("Student", user_input_for_history))

                    # --- Hint Generation Task ---
                    hint_task = Task(
                        description=(
                            f"The student explicitly asked for a hint. Their current learning step is: '{current_guidance_step}'.\n"
                            f"The full problem steps are: {json.dumps(learning_steps)}.\n"
                            f"The conversation history: {format_history_for_llm(conversation_history)}\n\n"
                            "Your task is to craft a subtle, guiding hint or a clear, thought-provoking question that helps the student move forward "
                            "without giving away the answer. Ensure the hint is age-appropriate (4th grade) and directly relevant to the current step.\n"
                            "The hint should feel like a friendly nudge, not a lecture."
                        ),
                        agent=ai_educator_agent,
                        expected_output="A gentle, pedagogically sound hint or guiding question (e.g., 'Think about what the word 'total' means in this problem.', 'Can you draw a picture to show the numbers?', 'What operation helps us combine things?')."
                    )
                    hint_crew = Crew(
                        agents=[ai_educator_agent],
                        tasks=[hint_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    try:
                        hint_output_obj = run_crew_with_retry(hint_crew, "explicit hint generation")
                        if hint_output_obj:
                            hint = hint_output_obj.raw_output if hasattr(hint_output_obj, 'raw_output') else str(
                                hint_output_obj)
                            print(f"💬 AI Educator: {hint}")
                            await speak_message(hint)
                            conversation_history.append(("AI Educator", hint))
                        else:
                            hint_fail_message = "Could not generate a hint after retries. Providing a generic nudge."
                            generic_nudge = "Let's think about this step a bit differently. What else could you try?"
                            print(f"❗️ {hint_fail_message}")
                            print(f"💬 AI Educator: {generic_nudge}")
                            await speak_message(generic_nudge)
                            conversation_history.append(("AI Educator", generic_nudge))
                    except Exception as e:
                        hint_error_message = f"Unexpected error during explicit hint generation flow: {e}"
                        stumped_message = "I'm a little stumped right now. Could you please try rephrasing your thought, or I can give you a different kind of hint?"
                        print(f"🚨 {hint_error_message}")
                        traceback.print_exc()
                        print(f"💬 AI Educator: {stumped_message}")
                        await speak_message(stumped_message)
                        conversation_history.append(("AI Educator", stumped_message))
                    continue  # Stay on the same step after providing a hint, wait for new student input

                elif user_choice == "3":  # Type Answer
                    user_input_for_history = input("Type your answer: ").lower().strip()
                elif user_choice == "4":  # Voice Answer
                    if whisper_model is None:
                        voice_unavailable_message = "Voice input is not available. Please choose 'type-answer'."
                        print(voice_unavailable_message)
                        await speak_message(voice_unavailable_message)
                        continue
                    await speak_message("Please speak your answer now.")
                    print("🎙️ Please speak your answer now...")
                    user_input_for_history = get_voice_input_local_whisper().lower().strip()
                    if user_input_for_history.startswith("ERROR_"):
                        error_message = f"Voice input failed: {user_input_for_history}. Please try typing instead or check your setup."
                        print(error_message)
                        await speak_message(error_message)
                        continue
                else:
                    invalid_option_message = "Invalid option. Please choose from 1, 2, 3, 4, 5, or 6."
                    print(invalid_option_message)
                    await speak_message(invalid_option_message)
                    continue

                conversation_history.append(("Student", user_input_for_history))

                # --- Core Scaffolding Interaction Task (combines evaluation, hint, feedback) ---
                scaffolding_interaction_task = Task(
                    description=(
                        f"As the AI Educator, your task is to dynamically guide the student through the problem-solving process and conceptual understanding "
                        f"based on their latest input and the current learning step. You will output a structured JSON response reflecting your internal decision-making.\n\n"
                        f"**Input for this interaction:**\n"
                        f"- `Student_Problem`: '{problem}'\n"
                        f"- `Student_Current_Input`: '{user_input_for_history}'\n"
                        f"- `Current_Learning_Step`: '{current_guidance_step}' (Step {current_step_index + 1} of {len(learning_steps)})\n"
                        f"- `All_Learning_Steps`: {json.dumps(learning_steps)}\n"
                        f"- `Interaction_History_Summary`: {format_history_for_llm(conversation_history)}\n"
                        f"- `Student_Learning_State`: {json.dumps(student_learning_state)} (includes `misconceptions_this_session`, `current_problem_steps_completed`)\n\n"
                        "**Your Output MUST be a JSON object conforming to the overall structure from your internal thought process.**\n"
                        "**Populate these fields based on your decision:**\n"
                        "  * `scaffolding_stage`: Reflects the current pedagogical phase (e.g., 'hinting_phase', 'feedback_phase').\n"
                        "  * `action_taken`: Your specific action (e.g., 'provide_hint', 'evaluate_response').\n"
                        "  * `educator_response`: Your conversational message and structured data.\n"
                        "    * `tone`: ('supportive' | 'encouraging' | 'neutral' | 'celebratory').\n"
                        "    * `message`: The conversational message you will say to the student.\n"
                        "    * `structured_data`: This object will vary based on `action_taken`:\n"
                        "      - If `action_taken` is 'evaluate_response': Provide `response_assessment` ('Correct' | 'Partially Correct' | 'Incorrect'), `assessment_justification`, `process_analysis`, `constructive_feedback`, and `scaffolding_adjustment_recommendation` ('Continue_Main_Problem' | 'Generate_New_Hint' | 'Generate_New_Activity' | 'Review_Prior_Concept' | 'Re_explain_problem_part' | 'Confirm_Mastery').\n"
                        "      - If `action_taken` is 'provide_hint': Provide `hint_level_chosen` ('Tier 1: Conceptual Reminder', etc.), `hint_content`, `rationale_for_hint`, `expected_student_action`.\n"
                        "      - If `action_taken` is 'suggest_activity': Provide `activity_type_chosen`, `activity_content`, `expected_learning_outcome`, `guidance_for_student`.\n"
                        "      - (And similarly for `re_explain_problem_part`, `review_prior_concept` if you choose those actions).\n\n"
                        "**Core Logic:**\n"
                        "1. **Evaluate `Student_Current_Input`**: Analyze it against `Current_Learning_Step` and `All_Learning_Steps` to determine understanding.\n"
                        "2. **Determine `scaffolding_adjustment_recommendation`**: Based on evaluation, decide the next pedagogical move.\n"
                        "3. **Generate `educator_response`**: Craft a message (and associated structured data) corresponding to the chosen action. If a hint is needed, provide one. If on track, provide positive feedback."
                        "**Constraint:** All interactions must consistently promote student confidence, foster independent reasoning, and build toward mastery without directly revealing solutions. Prioritize guiding questions and conceptual understanding over direct answers."
                    ),
                    agent=ai_educator_agent,
                    expected_output=(
                        "A JSON object conforming to the detailed internal thought process schema. Example:\n"
                        "```json\n"
                        "{\n"
                        "  \"scaffolding_stage\": \"feedback_phase\",\n"
                        "  \"action_taken\": \"evaluate_response\",\n"
                        "  \"educator_response\": {\n"
                        "    \"tone\": \"supportive\",\n"
                        "    \"message\": \"That's a great start! You're thinking about the right numbers.\",\n"
                        "    \"structured_data\": {\n"
                        "      \"response_assessment\": \"Partially Correct\",\n"
                        "      \"assessment_justification\": \"Student identified correct numbers but misapplied operation.\",\n"
                        "      \"process_analysis\": \"Student added instead of multiplied.\",\n"
                        "      \"constructive_feedback\": \"Remember what the word 'total' means in this problem. What operation best helps us find a 'total' when combining groups?\",\n"
                        "      \"scaffolding_adjustment_recommendation\": \"Generate_New_Hint\"\n"
                        "    }\n"
                        "  }\n"
                        "}\n"
                        "```"
                    ),
                    output_json=AIResponse
                )

                scaffolding_crew = Crew(
                    agents=[ai_educator_agent],
                    tasks=[scaffolding_interaction_task],
                    process=Process.sequential,
                    verbose=True
                )

                try:
                    scaffolding_output_obj = run_crew_with_retry(scaffolding_crew,
                                                                 f"scaffolding interaction for step {current_step_index + 1}")
                    if scaffolding_output_obj:
                        parsed_scaffolding_output = parse_llm_output_robustly(
                            scaffolding_output_obj.raw_output if hasattr(scaffolding_output_obj, 'raw_output') else str(
                                scaffolding_output_obj))

                        if parsed_scaffolding_output is None:
                            parse_error_scaffolding_message = f"Error: Could not parse scaffolding interaction response as JSON or Python literal. I'm having a little trouble understanding. Could you please rephrase your thought?"
                            print(
                                f"🚨 {parse_error_scaffolding_message} Raw output:\n{scaffolding_output_obj.raw_output if hasattr(scaffolding_output_obj, 'raw_output') else str(scaffolding_output_obj)}")
                            await speak_message(
                                "I'm having a little trouble understanding. Could you please rephrase your thought?")
                            conversation_history.append(("AI Educator",
                                                         "I'm having a little trouble understanding. Could you please rephrase your thought?"))
                            continue

                        try:
                            ai_educator_response_data = AIResponse.model_validate_json(
                                json.dumps(parsed_scaffolding_output))

                            educator_response_section = ai_educator_response_data.educator_response
                            structured_data_section = educator_response_section.structured_data
                            action_taken = ai_educator_response_data.action_taken
                            message = educator_response_section.message

                            recommendation = None
                            if action_taken == 'evaluate_response' and structured_data_section.evaluation_data:
                                recommendation = structured_data_section.evaluation_data.scaffolding_adjustment_recommendation
                                if structured_data_section.evaluation_data.response_assessment == 'Incorrect':
                                    misconception_detail = structured_data_section.evaluation_data.process_analysis
                                    student_learning_state['misconceptions_this_session'].append(misconception_detail)
                            elif action_taken == 'provide_hint' and structured_data_section.hint_data:
                                recommendation = "Generate_New_Hint"  # Explicitly set for loop control
                            elif action_taken == 'suggest_activity' and structured_data_section.activity_data:
                                recommendation = "Generate_New_Activity"  # Explicitly set for loop control

                            print(f"💬 AI Educator: {message}")
                            await speak_message(message)
                            conversation_history.append(("AI Educator", message))

                            if recommendation == "Continue_Main_Problem":
                                current_step_index += 1
                            elif recommendation == "Confirm_Mastery":
                                current_step_index = len(learning_steps)
                            # For other recommendations (hint, activity), we stay on the same step
                        except Exception as e:
                            validation_error_scaffolding_message = f"Error parsing or validating scaffolding output: {e}. I'm having a little trouble understanding. Could you please rephrase your thought?"
                            print(
                                f"🚨 {validation_error_scaffolding_message} Raw output: {scaffolding_output_obj.raw_output}")
                            traceback.print_exc()
                            await speak_message(
                                "I'm having a little trouble understanding. Could you please rephrase your thought?")
                            conversation_history.append(("AI Educator",
                                                         "I'm having a little trouble understanding. Could you please rephrase your thought;"))
                    else:
                        generic_nudge_message = "AI Educator could not provide a structured response. Providing generic nudge."
                        generic_nudge_content = "Let's keep thinking about this step. What's another way to approach it?"
                        print(f"❗️ {generic_nudge_message}")
                        print(f"💬 AI Educator: {generic_nudge_content}")
                        await speak_message(generic_nudge_content)
                        conversation_history.append(("AI Educator", generic_nudge_content))
                except Exception as e:
                    unexpected_error_scaffolding_message = f"Unexpected error during scaffolding interaction: {e}. I'm a little stumped right now. Could you please try rephrasing your thought?"
                    print(f"🚨 {unexpected_error_scaffolding_message}")
                    traceback.print_exc()
                    await speak_message("I'm a little stumped right now. Could you please try rephrasing your thought?")
                    conversation_history.append(("AI Educator",
                                                 "I'm a little stumped right now. Could you please try rephrasing your thought?"))

            # If the restart_session_flag is set, continue the outer loop to restart the session
            if restart_session_flag:
                continue

            # --- Phase 3: Progress Monitoring & Mastery Validation ---
            if current_step_index >= len(learning_steps):
                final_message_1_content = "Amazing! You've navigated through all the steps like a true math whiz!"
                print(f"\n🎉 {final_message_1_content}")
                await speak_message(final_message_1_content)
                conversation_history.append(("AI Educator", final_message_1_content))

                mastery_validation_task = Task(
                    description=(
                        f"As the AI Educator, your task is to confirm the student's mastery and provide a comprehensive reflection. "
                        f"The student has completed all the learning steps for problem: '{problem}'.\n"
                        f"Review the full conversation history: {format_history_for_llm(conversation_history)} "
                        f"and the student's final learning state: {json.dumps(student_learning_state)}.\n"
                        "Your output MUST be a JSON object conforming to the 'confirm_mastery' structure from your internal thought process.\n"
                        "Example of expected output (full JSON object with 'confirm_mastery' structured_data):\n"
                        "```json\n"
                        "{\n"
                        "  \"scaffolding_stage\": \"mastery_confirmation\",\n"
                        "  \"action_taken\": \"confirm_mastery\",\n"
                        "  \"educator_response\": {\n"
                        "    \"tone\": \"celebratory\",\n"
                        "    \"message\": \"Fantastic work today! Let's reflect on your amazing progress.\",\n"
                        "    \"structured_data\": {\n"
                        "      \"overall_mastery_confirmation\": \"Concept mastered: Addition with regrouping.\",\n"
                        "      \"goal_attainment_breakdown\": [\n"
                        "        {\"goal\": \"Identify key numbers\", \"met\": true, \"evidence\": \"Student correctly extracted numbers in Step 1\"},\n"
                        "        {\"goal\": \"Apply addition strategy\", \"met\": true, \"evidence\": \"Student successfully added numbers with correct regrouping\"}\n"
                        "      ],\n"
                        "      \"summary_of_understanding\": \"The student demonstrated a solid understanding of addition word problems, particularly in setting up the problem and executing the operation. Initial hesitation was overcome with guided questioning.\",\n"
                        "      \"next_steps_suggestion\": \"Practice more multi-step word problems involving addition and subtraction.\"\n"
                        "    }\n"
                        "  }\n"
                        "}\n"
                        "```"
                    ),
                    agent=ai_educator_agent,
                    expected_output="A JSON object following the specified 'confirm_mastery' structure within the overall output format.",
                    output_json=AIResponse
                )

            reflection_crew = Crew(
                agents=[ai_educator_agent],
                tasks=[mastery_validation_task],
                process=Process.sequential,
                verbose=True  # Set to True for debugging reflection
            )
            try:
                summary_obj = run_crew_with_retry(reflection_crew, "final reflection and mastery validation")
                if summary_obj:
                    parsed_summary_output = parse_llm_output_robustly(
                        summary_obj.raw_output if hasattr(summary_obj, 'raw_output') else str(summary_obj))

                    if parsed_summary_output is None:
                        parse_error_reflection_message = f"Error: Could not parse final reflection response as JSON or Python literal. Great job finishing this problem! Keep up the amazing work!"
                        print(
                            f"🚨 {parse_error_reflection_message} Raw output:\n{summary_obj.raw_output if hasattr(summary_obj, 'raw_output') else str(summary_obj)}")
                        await speak_message("Great job finishing this problem! Keep up the amazing work!")
                        conversation_history.append(
                            ("AI Educator", "Great job finishing this problem! Keep up the amazing work!"))
                        continue

                    try:
                        summary_data = AIResponse.model_validate_json(json.dumps(parsed_summary_output))

                        summary_message = summary_data.educator_response.message
                        summary_details = summary_data.educator_response.structured_data.mastery_confirmation_data

                        print(f"✨ AI Educator: {summary_message}")
                        await speak_message(summary_message)
                        if summary_details:
                            overall_mastery = f"Overall Mastery: {summary_details.overall_mastery_confirmation}"
                            summary_of_understanding = f"Summary: {summary_details.summary_of_understanding}"
                            next_steps = f"Next Steps: {summary_details.next_steps_suggestion}"
                            print(overall_mastery)
                            await speak_message(overall_mastery)
                            print(summary_of_understanding)
                            await speak_message(summary_of_understanding)
                            print(next_steps)
                            await speak_message(next_steps)
                        conversation_history.append(("AI Educator", summary_message))
                    except Exception as e:
                        validation_error_reflection_message = f"JSON parsing/validation error from AI Educator's reflection response: {e}. Great job finishing this problem! Keep up the amazing work!"
                        print(f"🚨 {validation_error_reflection_message} Raw output:\n{parsed_summary_output}")
                        traceback.print_exc()
                        await speak_message("Great job finishing this problem! Keep up the amazing work!")
                        conversation_history.append(
                            ("AI Educator", "Great job finishing this problem! Keep up the amazing work!"))
                else:
                    reflection_fail_message = "Could not generate a reflection after retries. Providing a generic closing message."
                    generic_closing_message = "Great job finishing this problem! Keep up the amazing work!"
                    print(f"❗️ {reflection_fail_message}")
                    print(f"✨ AI Educator: {generic_closing_message}")
                    await speak_message(generic_closing_message)
                    conversation_history.append(("AI Educator", generic_closing_message))
            except Exception as e:
                unexpected_error_reflection_message = f"Unexpected error during reflection flow: {e}. Great job finishing this problem! Keep up the amazing work!"
                print(f"� {unexpected_error_reflection_message}")
                traceback.print_exc()
                await speak_message("Great job finishing this problem! Keep up the amazing work!")
                conversation_history.append(
                    ("AI Educator", "Great job finishing this problem! Keep up the amazing work!"))

            final_message_2_content = "Now, armed with your step-by-step understanding, take a moment to solve the full problem on your own. You've got all the tools you need!"
            print(final_message_2_content)
            await speak_message(final_message_2_content)
            conversation_history.append(("AI Educator", final_message_2_content))

            final_message_3_content = "Ready for another challenge? Just type in your next problem!"
            print(final_message_3_content)
            await speak_message(final_message_3_content)
            conversation_history.append(("AI Educator", final_message_3_content))

    except KeyboardInterrupt:
        print("\n👋 Math Buddy session interrupted by user. Goodbye for now!")
    except Exception as e:
        print(f"🚨 An unhandled critical error occurred during the Math Buddy session: {e}")
        traceback.print_exc()
        print("\nIt seems we hit a major snag. Let's try starting fresh with a new problem.")

    print("👋 Goodbye for now, Math Explorer! Keep that brain sharp!")


if __name__ == "__main__":
    asyncio.run(run_math_buddy())  # Run the async main function
