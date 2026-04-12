import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List, Literal, Union, Dict, Any
import json
import time
import httpx
import traceback
import ast # Import for literal_eval
from evaluator_agents import math_evaluator_agent # Import the agent instance

# --- Pydantic Models for Structured Output ---
# These models define the exact JSON structure the AI_Educator is expected to output.

class StepEvaluation(BaseModel):
    """
    Represents the evaluation of a single step provided by the student.
    """
    student_step: str = Field(..., description="The exact step provided by the student.")
    is_correct: bool = Field(..., description="True if the student's step is correct, False otherwise.")
    reason_if_wrong: Union[str, None] = Field(None, description="Explanation of why the step is wrong, if applicable. Provide specific details.")
    correct_guidance: Union[str, None] = Field(None, description="The correct approach or a guiding question for the next part of the step, if the student's step was incorrect or incomplete. Avoid giving direct answers.")

class EvaluationData(BaseModel):
    """
    Contains the comprehensive evaluation results for the student's solution.
    """
    overall_assessment: Literal["Correct", "Partially Correct", "Incorrect"] = Field(..., description="Overall assessment of the student's provided solution.")
    percentage_correct: int = Field(..., description="Percentage of the provided solution that is correct (0-100).")
    feedback_message: str = Field(..., description="A general, encouraging feedback message to the student.")
    step_by_step_evaluation: List[StepEvaluation] = Field(..., description="Detailed evaluation for each student-provided step.")
    remaining_steps_guidance: Union[List[str], None] = Field(None, description="If the student solved only a portion, these are the remaining steps for them to complete the problem, phrased as guiding questions/instructions. Each item in the list is one guiding step.")

# NEW Pydantic Model for the content within 'educator_response'
class EducatorResponseContent(BaseModel):
    """
    Defines the structure for the 'educator_response' field, including structured_data.
    """
    tone: Literal["supportive", "encouraging", "neutral", "celebratory"] = Field(..., description="The tone of the AI's message.")
    message: str = Field(..., description="The conversational message to the student.")
    structured_data: Union[EvaluationData, None] = Field(None, description="Structured data related to the action taken, conforming to EvaluationData.")


class AIResponse(BaseModel):
    """
    The top-level Pydantic model for the AI Educator's structured response.
    Updated to correctly map structured_data within educator_response.
    """
    scaffolding_stage: Literal["problem_evaluation", "feedback_delivery", "solution_completion"] = Field(..., description="The current pedagogical stage (e.g., 'problem_evaluation', 'feedback_delivery', 'solution_completion').")
    action_taken: Literal["evaluate_solution", "provide_feedback", "guide_completion"] = Field(..., description="The specific action taken by the AI Educator (e.g., 'evaluate_solution', 'provide_feedback', 'guide_completion').")
    # Updated to use the new EducatorResponseContent model
    educator_response: EducatorResponseContent = Field(..., description="The conversational message and structured data.")
    # Removed the top-level structured_data field as it's nested within educator_response


# --- Helper Functions (reused from previous project, with minor adjustments) ---

def format_history_for_llm(history: List[tuple]) -> str:
    """
    Formats conversation history into a string suitable for LLM context.
    Ensures clear separation between speaker and message.
    """
    formatted_lines = []
    for speaker, message in history:
        formatted_lines.append(f"{speaker}: {message}")
    return "\n".join(formatted_lines)

def run_crew_with_retry(crew_instance: Crew, task_context_label: str, max_retries: int = 3, delay: int = 2) -> Union[Any, None]:
    """
    Runs a CrewAI crew with retry logic for robustness against transient errors.
    Returns the crew's result object or None if all retries fail.
    """
    for attempt in range(max_retries + 1):
        try:
            print(f"Attempt {attempt + 1}/{max_retries + 1} for {task_context_label}...")
            result = crew_instance.kickoff()
            print(f"--- {task_context_label} successful. ---")
            return result
        except Exception as e:
            print(f"Error during {task_context_label} (attempt {attempt + 1}): {e}")
            traceback.print_exc()
            if attempt < max_retries:
                print(f"Retrying {task_context_label} in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"'{task_context_label}' failed persistently after {max_retries + 1} attempts due to an unexpected error. Failing.")
                return None
    print(f"--- {task_context_label} failed after all retries. ---")
    return None

def parse_llm_output_robustly(raw_output: Any) -> Union[Dict[str, Any], None]:
    """
    Attempts to parse LLM output, first by stripping markdown code blocks,
    then as direct JSON, and finally as a Python literal (like a dict string).
    Returns a dictionary or None on failure to parse as structured data.
    """
    if isinstance(raw_output, dict):  # If CrewAI already returned a dict/Pydantic model
        return raw_output
    if not isinstance(raw_output, str):
        # If it's not a string or dict, it's not parsable as JSON/literal here.
        return None

    # Step 1: Attempt to strip markdown code block fences
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json") and cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[len("```json"): -len("```")].strip()
    elif cleaned_output.startswith("```") and cleaned_output.endswith("```"):
        # Catch generic code blocks too, though 'json' is preferred.
        cleaned_output = cleaned_output[len("```"): -len("```")].strip()

    # Step 2: Try direct JSON parsing on the cleaned output
    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        pass  # Not valid JSON, proceed to literal_eval

    # Step 3: Try parsing as a Python literal (e.g., dict with single quotes)
    if cleaned_output.strip().startswith(('{', '[')):
        try:
            # Safely evaluate string containing a Python literal
            parsed_literal = ast.literal_eval(cleaned_output)
            if isinstance(parsed_literal, (dict, list)):
                return parsed_literal
        except (ValueError, SyntaxError):
            pass  # Not a valid Python literal, proceed

    return None  # Could not parse as structured data (dict or list)
