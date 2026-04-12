import os
import json
import time
import httpx
import traceback
import asyncio  # Required for async operations
import re  # Added for robust text formatting
import sys

# CrewAI imports
from crewai import Task, Crew, Process

# Local imports from your project structure
# Assuming voice_input_tts.py is in the same directory and contains these functions
from voice_input_google_tts import *
from evaluator_agents import math_evaluator_agent  # Import the instantiated agent
from evaluator_functions import (
    run_crew_with_retry,
    parse_llm_output_robustly,
    AIResponse,  # Main Pydantic model for AI response
    EvaluationData,  # Pydantic model for structured evaluation data
    StepEvaluation  # Pydantic model for individual step evaluation
)


# Helper function to format spoken math expressions into symbols for STT input
def _format_math_expression(text: str) -> str:
    """
    Converts spoken mathematical terms and numbers (as words) into symbolic format.
    Example: "four plus four minus five" -> "4 + 4 - 5"
    """
    # Convert common number words to digits
    num_word_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }
    for word, digit in num_word_map.items():
        # Use regex to replace whole words only, case-insensitive
        text = re.sub(r'\b' + re.escape(word) + r'\b', digit, text, flags=re.IGNORECASE)

    # Replace common operator words with symbols
    operator_map = {
        "plus": "+", "add": "+", "and": "+",
        "minus": "-", "subtract": "-", "take away": "-",
        "times": "*", "multiply by": "*",
        "divided by": "/", "divide by": "/",
        "equals": "=", "is": "=", "are": "="
    }
    for word, symbol in operator_map.items():
        # Use regex to replace whole words only, case-insensitive
        text = re.sub(r'\b' + re.escape(word) + r'\b', symbol, text, flags=re.IGNORECASE)

    # Remove non-alphanumeric characters except for math symbols and spaces
    # This will help clean up punctuation like '.' from "4." and other noise.
    cleaned_text = "".join(char for char in text if char.isalnum() or char in "+-*/=(). ")

    # Replace multiple spaces with a single space and strip leading/trailing spaces
    cleaned_text = " ".join(cleaned_text.split()).strip()

    return cleaned_text


# NEW helper function to format math expressions into spoken words for TTS output
def _format_math_for_tts(text: str) -> str:
    """
    Converts mathematical symbols into spoken words for better TTS pronunciation.
    Example: "4+5-6*2" -> "4 plus 5 minus 6 times 2"
    """
    # Replace symbols with words, ensuring order to avoid partial matches (e.g., `-` before `+`)
    text = text.replace("-", " minus ")
    text = text.replace("+", " plus ")
    text = text.replace("*", " times ")
    text = text.replace("/", " divided by ")
    text = text.replace("=", " equals ")

    # Clean up multiple spaces that might result from replacements
    return re.sub(r'\s+', ' ', text).strip()


async def run_math_evaluator():
    """
    Main asynchronous function to run the Math Evaluator application.
    Guides the student through problem input, solution scope, step input,
    and provides detailed evaluation and guidance.
    """
    try:  # Outermost try block starts here
        welcome_message = "Welcome to your Math Evaluator! Let's review your math problems together."
        print(f"👋 {welcome_message}")
        await speak_message(welcome_message)
        print("Type 'quit' at any prompt to stop anytime.\n")

        # --- Initial Input Method Selection ---
        preferred_input_method = ""
        while preferred_input_method not in ["type", "speak"]:
            input_method_choice_message = "How would you prefer to input your answers for this session?"
            print(f"\n{input_method_choice_message}")
            await speak_message(input_method_choice_message)
            user_choice_method = input(
                "1. Type (text input)\n2. Speak (voice input)\n3. Quit\nEnter your choice (1-3): ").strip()

            if user_choice_method == "1":
                preferred_input_method = "type"
                print("You've chosen text input for this session.")
                await speak_message("You've chosen text input for this session.")
            elif user_choice_method == "2":
                preferred_input_method = "speak"
                print("You've chosen voice input for this session.")
                await speak_message("You've chosen voice input for this session.")
            elif user_choice_method == "3":
                goodbye_message = "See you next time, Math Explorer!"
                print(f"👋 {goodbye_message}")
                await speak_message(goodbye_message)
                return  # Exit the main function
            else:
                invalid_choice_message = "Invalid choice. Please enter 1, 2, or 3."
                print(invalid_choice_message)
                await speak_message(invalid_choice_message)

        while True:  # Outer loop for new evaluation sessions
            conversation_history = []  # Reset history for each new problem evaluation

            # --- Function to get input based on preferred method ---
            async def get_user_input(prompt_message: str, voice_prompt: str = None, default_text: str = None,
                                     voice_timeout: int = 50) -> str:  # Changed default timeout to 50
                await speak_message(prompt_message)
                print(f"\n{prompt_message}")  # Print the prompt for visibility

                if preferred_input_method == "type":
                    user_input = input("Your response: ").strip()
                    if not user_input and default_text is not None:
                        return default_text
                    return user_input
                else:  # preferred_input_method == "speak"
                    await speak_message(voice_prompt if voice_prompt else prompt_message)
                    print(f"🎙️ {voice_prompt if voice_prompt else prompt_message}...")

                    # Assuming get_voice_input_google_cloud is in voice_input_tts
                    spoken_input_raw = await get_voice_input_google_cloud(timeout=voice_timeout)

                    print(f"🎙️ Raw Transcribed: '{spoken_input_raw}'")

                    if spoken_input_raw == "TIMEOUT_ERROR":
                        timeout_message = "I didn't hear anything. Please type your response instead."
                        print(f"⌛ {timeout_message}")
                        await speak_message(timeout_message)
                        user_input = input("Voice input timed out. Please type your response: ").strip()
                        if not user_input and default_text is not None:
                            return default_text
                        return user_input
                    elif spoken_input_raw.startswith("ERROR_"):
                        error_message = f"Voice input failed: {spoken_input_raw}. Please try typing instead or check your setup."
                        print(f"🚨 {error_message}")
                        await speak_message(error_message)
                        user_input = input("Voice input failed. Please type your response: ").strip()
                        if not user_input and default_text is not None:
                            return default_text
                        return user_input
                    else:
                        formatted_input = _format_math_expression(spoken_input_raw)
                        print(f"🎙️ Formatted: '{formatted_input}'")

                        if not formatted_input:
                            print(
                                "No clear speech detected or input was empty after formatting. Falling back to text input.")
                            await speak_message("I couldn't quite catch that. Please type your answer.")
                            user_input = input("Type your response: ").strip()
                            if not user_input and default_text is not None:
                                return default_text
                            return user_input
                        else:
                            return formatted_input

            # --- 1. Get the Problem ---
            problem = ""
            while not problem:
                problem = await get_user_input(
                    "What math problem would you like to evaluate today?",
                    "Please speak your math problem now.",
                    voice_timeout=50
                )
                if problem.lower() == 'quit':
                    goodbye_message = "See you next time, Math Explorer!"
                    print(f"👋 {goodbye_message}")
                    await speak_message(goodbye_message)
                    return
                if not problem:
                    empty_problem_message = "Problem cannot be empty. Please try again."
                    print(empty_problem_message)
                    await speak_message(empty_problem_message)

            # --- 2. Ask about Solution Scope (Whole/Portion) ---
            solution_scope = ""
            while solution_scope not in ["whole", "portion"]:
                scope_choice_raw = await get_user_input(
                    "Did you solve the whole problem, or just a portion of it?\n1. Whole Problem\n2. Portion of Problem",
                    "Please speak 'one' for whole problem, or 'two' for portion of problem.",
                    voice_timeout=10
                )
                if scope_choice_raw.lower() == 'quit':
                    goodbye_message = "See you next time, Math Explorer!"
                    print(f"👋 {goodbye_message}")
                    await speak_message(goodbye_message)
                    return

                if "1" in scope_choice_raw or "whole" in scope_choice_raw:
                    solution_scope = "whole"
                elif "2" in scope_choice_raw or "portion" in scope_choice_raw:
                    solution_scope = "portion"
                else:
                    invalid_choice_message = "Invalid choice. Please enter '1' for whole or '2' for portion."
                    print(invalid_choice_message)
                    await speak_message(invalid_choice_message)

            # --- 3. Get Student Steps ---
            student_steps_raw = ""
            while not student_steps_raw:
                student_steps_raw = await get_user_input(
                    "Great! Now, please tell me the steps you took to solve the problem, one by one. You can list them or describe your process.",
                    "Please speak your steps now.",
                    voice_timeout=50
                )
                if student_steps_raw.lower() == 'quit':
                    goodbye_message = "See you next time, Math Explorer!"
                    print(f"👋 {goodbye_message}")
                    await speak_message(goodbye_message)
                    return
                if not student_steps_raw:
                    empty_steps_message = "Steps cannot be empty. Please try again."
                    print(empty_steps_message)
                    await speak_message(empty_steps_message)

            # Start a new conversation history for the problem, which will be updated on each turn
            problem_conversation_history = [student_steps_raw]

            # --- 4. Model's Internal Solution Generation (Ground Truth) ---
            print("\nAI Evaluator is generating the optimal solution steps for comparison...")
            await speak_message("I'm preparing the optimal solution steps for your problem.")

            generate_correct_steps_task = Task(
                description=(
                    f"Generate the complete, step-by-step optimal solution for the following 4th-grade math problem: '{problem}'.\n"
                    "Each step should be clear, concise, and pedagogically sound for a 4th-grade student.\n"
                    "Crucially, for each step, focus on describing the *process* or *operation* without explicitly stating the numerical result of that step or the final answer. "
                    "The goal is to provide a conceptual path, not the solution itself. "
                    "For example, instead of '8 - 5 equals 3', say 'Now, we subtract 5 from 8'. Do not reveal the final answer."
                    "Output a JSON object with a single key 'optimal_steps' which is a list of strings, where each string is a clear step.\n"
                    "Example: {{ \"optimal_steps\": [\"Step 1: Understand what the problem is asking.\", \"Step 2: Identify the numbers and what they represent.\", \"Step 3: Choose the correct operation (addition, subtraction, multiplication, or division).\", \"Step 4: Perform the calculation.\", \"Step 5: Write down the final answer with units.\"] }}"
                ),
                agent=math_evaluator_agent,
                expected_output="A JSON object with a list of optimal solution steps.",
                output_json_schema={"type": "object",
                                    "properties": {"optimal_steps": {"type": "array", "items": {"type": "string"}}}}
            )

            correct_steps_crew = Crew(
                agents=[math_evaluator_agent],
                tasks=[generate_correct_steps_task],
                process=Process.sequential,
                verbose=True
            )

            model_correct_steps = []
            try:
                correct_steps_output_obj = run_crew_with_retry(correct_steps_crew, "generating optimal steps")
                if correct_steps_output_obj:
                    parsed_correct_steps = parse_llm_output_robustly(
                        correct_steps_output_obj.raw_output if hasattr(correct_steps_output_obj, 'raw_output') else str(
                            correct_steps_output_obj)
                    )
                    if parsed_correct_steps and "optimal_steps" in parsed_correct_steps:
                        model_correct_steps = parsed_correct_steps["optimal_steps"]
                    else:
                        print(
                            "⚠️ Could not parse optimal steps from AI Evaluator. Proceeding with evaluation, but it might be less accurate.")
                        await speak_message(
                            "I'm having a little trouble generating my own solution steps right now. I'll do my best to evaluate with what I have!")
                else:
                    print("🚨 Failed to generate optimal steps after retries. Evaluation accuracy may be impacted.")
                    await speak_message(
                        "I couldn't generate the optimal steps for this problem. I'll still try to evaluate your work, but it might be harder to give precise feedback.")
            except Exception as e:
                print(f"🚨 Error generating optimal steps: {e}")
                traceback.print_exc()
                await speak_message(
                    "An error occurred while generating the optimal steps. I'll do my best to evaluate your work anyway!")

            # --- 5. Evaluation Task (Initial or Iterative) ---
            problem_solved = False
            while not problem_solved:
                print("\nAI Evaluator is now evaluating your steps...")
                await speak_message("I'm carefully reviewing your steps now.")

                print(f"\nOriginal Math Problem: {problem}")
                await speak_message(f"The original math problem is: {_format_math_for_tts(problem)}")

                evaluation_task = Task(
                    description=(
                        f"As the MathEvaluator, you will evaluate a student's solution for the problem: '{problem}'.\n"
                        f"You will be provided with the student's *entire conversation history* regarding their solution.\n"
                        f"The history is: {json.dumps(problem_conversation_history)}\n\n"
                        f"Your task is to:\n"
                        "1. **CRITICAL FIRST STEP**: Synthesize a single, definitive, and coherent set of student steps for evaluation. Review the entire `conversation_history`. If a student's later input appears to be a correction or re-statement of a previous step, you MUST replace the old step with the new one to create a clean, logical set of steps to evaluate.\n"
                        f"2. Compare this synthesized set of steps to the `Optimal solution steps` (provided for your reference only): {json.dumps(model_correct_steps)}\n"
                        "3. For each of the student's identified steps, determine if it is **correct, partially correct, or incorrect**.\n"
                        "4. If a step is incorrect or partially correct, provide a clear and concise `reason_if_wrong` and `correct_guidance`. The `correct_guidance` should be a gentle nudge or a guiding question, not a direct answer.\n"
                        "5. **Calculate `percentage_correct`** based on the overall correctness of the consolidated steps.\n"
                        "6. **CRUCIAL**: If the consolidated steps constitute a full and correct solution, `overall_assessment` MUST be 'Correct' and `remaining_steps_guidance` MUST be an empty list (`[]`).\n"
                        "7. **ABSOLUTELY CRITICAL**: If the consolidated steps are NOT yet a complete solution, you MUST provide `remaining_steps_guidance` as a list of clear, actionable guiding questions for the student's *next logical action*. This list MUST NEVER be empty if `overall_assessment` is not 'Correct'.\n"
                        "8. Provide an `overall_assessment` and a general, encouraging `feedback_message`.\n\n"
                        "Your output MUST be a JSON object conforming to the `AIResponse` Pydantic model.\n"
                        "**Crucially, for the top-level fields, ensure the following literal values are used:**\n"
                        "  - `scaffolding_stage`: \"problem_evaluation\"\n"
                        "  - `action_taken`: \"evaluate_solution\"\n"
                        "  - `educator_response`: This object must contain `tone`, `message`, and `structured_data`.\n"
                        "  - `structured_data` (nested within `educator_response`): This object must contain `overall_assessment`, `percentage_correct`, `feedback_message`, `step_by_step_evaluation`, and optionally `remaining_steps_guidance`."
                    ),
                    agent=math_evaluator_agent,
                    expected_output="A JSON object conforming to the AIResponse model with EvaluationData.",
                    output_json=AIResponse
                )

                evaluation_crew = Crew(
                    agents=[math_evaluator_agent],
                    tasks=[evaluation_task],
                    process=Process.sequential,
                    verbose=True
                )

                try:
                    evaluation_output_obj = run_crew_with_retry(evaluation_crew, "solution evaluation")
                    if evaluation_output_obj is None:
                        fail_message = "Failed to evaluate your solution. This might be due to an unclear problem or steps. Please try again with a different problem or rephrase your steps."
                        print(f"🚨 {fail_message}")
                        await speak_message(fail_message)
                        problem_solved = True
                        break

                    parsed_evaluation_output = parse_llm_output_robustly(
                        evaluation_output_obj.raw_output if hasattr(evaluation_output_obj, 'raw_output') else str(
                            evaluation_output_obj)
                    )

                    if parsed_evaluation_output is None:
                        parse_error_message = "Error: Could not parse evaluation response. I'm having trouble understanding your steps. Could you please rephrase them?"
                        print(
                            f"🚨 {parse_error_message} Raw output:\n{evaluation_output_obj.raw_output if hasattr(evaluation_output_obj, 'raw_output') else str(evaluation_output_obj)}")
                        await speak_message(parse_error_message)
                        problem_solved = True
                        break

                    try:
                        ai_response_model = AIResponse.model_validate_json(json.dumps(parsed_evaluation_output))
                        evaluation_results: EvaluationData = ai_response_model.educator_response.structured_data

                        if evaluation_results is None:
                            raise ValueError(
                                "structured_data is None within educator_response after Pydantic validation.")

                        print(f"\n--- AI Evaluator: {evaluation_results.feedback_message} ---")
                        await speak_message(evaluation_results.feedback_message)

                        print(
                            f"You've got {evaluation_results.percentage_correct}% of the problem correct so far! Keep up the great work!")
                        await speak_message(
                            f"You've got {evaluation_results.percentage_correct} percent of the problem correct so far! Keep up the great work!")

                        print("\nLet's look at your steps in detail:")
                        await speak_message("Let's look at your steps in detail.")

                        for i, step_eval in enumerate(evaluation_results.step_by_step_evaluation):
                            print(f"\nYour Step {i + 1}: {step_eval.student_step}")
                            if step_eval.is_correct:
                                print("Status: ✅ Correct!")
                                await speak_message(f"Your step {i + 1} is correct!")
                            else:
                                print("Status: ❌ Incorrect or needs adjustment.")
                                await speak_message(f"Your step {i + 1} is incorrect or needs adjustment.")
                                if step_eval.reason_if_wrong:
                                    print(f"Reason: {step_eval.reason_if_wrong}")
                                    await speak_message(f"Here's why: {step_eval.reason_if_wrong}")
                                if step_eval.correct_guidance:
                                    print(f"Proper Guidance: {step_eval.correct_guidance}")
                                    await speak_message(f"Here's what you could do: {step_eval.correct_guidance}")

                        if evaluation_results.overall_assessment == "Correct":
                            print("\nGreat job! You've successfully completed the problem!")
                            await speak_message("Great job! You've successfully completed the problem!")
                            print("Thank you for solving this problem!")
                            await speak_message("Thank you for solving this problem!")
                            problem_solved = True
                        else:
                            if evaluation_results.remaining_steps_guidance:
                                print(
                                    "\nFantastic job on those steps! You're on the right track! Now, let's look at what's next to complete the problem:")
                                await speak_message(
                                    "Fantastic job on those steps! You're on the right track! Now, let's look at what's next to complete the problem:")
                                for i, remaining_step in enumerate(evaluation_results.remaining_steps_guidance):
                                    print(f"Next Hint {i + 1}: {remaining_step}")
                                    await speak_message(f"Next Hint {i + 1}: {remaining_step}")
                                print("\nWhat are your next steps to solve the problem? (Or type 'quit' to exit)")

                                new_input = await get_user_input(
                                    "What are your next steps to solve the problem? (Or type 'quit' to exit)",
                                    "Please speak your next steps now.",
                                    voice_timeout=50
                                )
                                if new_input.lower() == 'quit' or new_input.lower() == 'exit':
                                    goodbye_message = "See you next time, Math Explorer!"
                                    print(f"👋 {goodbye_message}")
                                    await speak_message(goodbye_message)
                                    return
                                if not new_input:
                                    print("No new steps provided. Ending current problem session.")
                                    await speak_message("No new steps provided. Ending current problem session.")
                                    problem_solved = True
                                else:
                                    problem_conversation_history.append(new_input)
                            else:
                                print(
                                    "\nIt seems I couldn't generate specific next steps right now, but your solution is not yet complete. Please review your previous steps and provide new ones to complete the problem, or type 'quit' to exit. If you cannot provide new steps, the session for this problem will end.")
                                await speak_message(
                                    "It seems I couldn't generate specific next steps right now, and your solution is not yet complete. Please review your previous steps and provide new ones to complete the problem, or type quit to exit. If you cannot provide new steps, the session for this problem will end.")

                                new_input = await get_user_input(
                                    "Please provide your next steps to complete the problem, or type 'quit' to exit.",
                                    "Please speak your next steps now.",
                                    voice_timeout=50
                                )
                                if new_input.lower() == 'quit' or new_input.lower() == 'exit':
                                    goodbye_message = "See you next time, Math Explorer!"
                                    print(f"👋 {goodbye_message}")
                                    await speak_message(goodbye_message)
                                    return
                                if not new_input:
                                    print("No new steps provided. Ending current problem session.")
                                    await speak_message("No new steps provided. Ending current problem session.")
                                    problem_solved = True
                                else:
                                    problem_conversation_history.append(new_input)
                    except Exception as e:
                        print(f"🚨 Error validating or processing Pydantic model: {e}")
                        traceback.print_exc()
                        await speak_message("I'm having a little trouble with my internal evaluation process. Please try again or rephrase your steps.")
                        problem_solved = True
                except Exception as e:
                    print(f"🚨 An unexpected error occurred during the evaluation process: {e}")
                    traceback.print_exc()
                    await speak_message("An unexpected error occurred during the evaluation. Let's try starting fresh with a new problem.")
                    problem_solved = True

            if not problem_solved:
                 print("\nNo more steps provided, or an error occurred. Ending current problem session.")
                 await speak_message("No more steps provided, or an error occurred. Ending current problem session.")

            while True:
                user_action_choice = input(
                    "\nWhat would you like to do next?\n1. Evaluate another problem\n2. Quit\nEnter your choice (1-2): "
                ).strip().lower()

                if user_action_choice == "1":
                    break
                elif user_action_choice in ("2", 'quit', 'exit'):
                    goodbye_message = "See you next time, Math Explorer! Keep that brain sharp!"
                    print(f"👋 {goodbye_message}")
                    await speak_message(goodbye_message)
                    return
                else:
                    invalid_choice_message = "Invalid choice. Please enter '1' to evaluate another problem, or '2' to quit."
                    print(invalid_choice_message)
                    await speak_message(invalid_choice_message)

    except Exception as e:
        print(f"🚨 An unhandled critical error occurred during the Math Evaluator session: {e}")
        traceback.print_exc()
        print("\nIt seems we hit a major snag. Let's try starting fresh with a new problem.")

    print("👋 Goodbye for now, Math Explorer! Keep that brain sharp!")


if __name__ == "__main__":
    asyncio.run(run_math_evaluator())
