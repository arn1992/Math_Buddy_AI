# 4th Grade AI Math Buddy
## Overview
The 4th Grade AI Math Buddy is an intelligent and adaptive tutoring system designed to help 4th-grade students learn and master mathematical concepts through guided discovery and problem-solving. Leveraging the power of CrewAI and a single, unified "AI Educator" agent, this system provides personalized scaffolding, adaptive hints, and constructive feedback to foster independent thinking and a resilient growth mindset in mathematics.

The application has been refactored from a monolithic script into a modular structure for better organization and maintainability.

## Features
1. Single AI Educator Agent: A highly sophisticated AI orchestrates the entire tutoring process, combining problem understanding, adaptive scaffolding, and progress monitoring capabilities.

2. Structured Learning Paths: Problems are broken down into pedagogically sound, step-by-step guidance tailored for 4th graders.

3. Adaptive Scaffolding: Dynamically adjusts support based on student input, offering hints, suggesting activities, or re-explaining concepts without giving direct answers.

4. Real-time Progress Evaluation: Assesses student responses and classifies their understanding as 'ON_TRACK', 'NEEDS_HINT', or 'MISCONCEPTION'.

5. Structured AI Outputs: The AI's internal thought process and responses are formalized using Pydantic models, ensuring reliable data exchange and enabling robust logic.

6. Robust Error Handling: Includes retry mechanisms for API calls and robust parsing for LLM outputs to enhance stability.

7. Empathetic and Supportive Tone: The AI maintains a patient and encouraging demeanor to build student confidence.

## Project Structure
The project is now organized into four main Python scripts:

1. api.py: (Assumed) Contains configurations related to external APIs, specifically the Gemini LLM API key and its initialization.

2. agents.py: Defines the AI_Educator agent with its role, goal, backstory, and LLM configuration.

3. functions.py: Holds utility functions such as format_history_for_llm for conversation formatting, run_crew_with_retry for robust execution of CrewAI, and parse_llm_output_robustly for handling varied LLM output formats. It also contains the Pydantic models for structured AI responses.

4. main.py: The main entry point of the application. It orchestrates the interaction loop, initializes tasks, creates and runs the CrewAI crew, and manages the conversation flow with the student.

5. voice_input_output.py: Encapsulates the Speech-to-Text (STT) functionality using Faster Whisper and Text-to-Speech (TTS) functionality using Edge TTS. It provides get_voice_input_local_whisper and speak_message functions for easy integration.

6. main_stt_tts.py: An alternative entry point that integrates Speech-to-Text (STT) and Text-to-Speech (TTS) for voice-enabled interaction by importing functions from voice_input_output.py.

## Usage
1. To run the Math Buddy application, execute the main.py script:
```
python main.py
```
Follow the prompts in your terminal:

   a. Enter a 4th-grade math problem when asked.
   
   b. Respond to the AI Educator's guiding questions or hints.
   
   c. You can type done to move to the next logical step (if the AI deems it appropriate based on your progress).
   
   d. Type hint if you need a nudge.
   
   e. Type restart to start a new problem.

   f. Type quit to exit the application.

2. Running the Voice-Enabled Version (main_stt_tts.py)
To run the Math Buddy with Speech-to-Text and Text-to-Speech capabilities:

```
python main_stt_tts.py
```
   Follow the prompts in your terminal:
   
   The AI Educator will speak its messages.
   
   When prompted for input, you will have options to:
   
   a. Quit: Exit the application.
   
   b. Restart: Start a new problem session.
   
   c. Continue with text: Type your math problem or answer.
   
   d. Continue with voice: Speak your math problem or answer into your microphone.
   
   e. Hint: Request a voice-based hint from the AI Educator.
   
   f. Done: Indicate you've completed a step and are ready to move on.

   Press Ctrl+C during voice recording to stop early and process the recorded audio. The application is designed to handle this interruption gracefully.


## How it Works (Under the Hood)
The application leverages CrewAI's powerful agentic framework. The single AI_Educator agent processes all interactions.

Initial Problem Analysis: When a problem is entered, the AI_Educator runs a task to comprehend the problem deeply, diagnose potential misconceptions, and generate an initial set of learning steps. This output is a structured JSON, validated by Pydantic models.

Adaptive Interaction Loop: For each student input, the AI_Educator executes a task that:
   a.  Evaluates the student's response against the current learning step.
   b.  Determines the next pedagogical action (e.g., provide a hint, suggest an activity, or advance to the next step).
   c.  Generates a conversational message and structured data (validated by Pydantic models) to guide the student.

Mastery Validation: Once all learning steps are completed, the AI_Educator performs a final task to summarize the student's learning journey, confirm mastery, and suggest next steps for continued growth.

The robust parsing logic ensures that even if the LLM output is not perfectly JSON-formatted (e.g., Python dictionary strings), the application can still interpret it correctly and proceed. The main_stt_tts.py script extends this by integrating local speech recognition (Faster Whisper) and cloud-based text-to-speech (Edge TTS) for a richer user experience, with these functionalities now cleanly separated into voice_io_module.py.