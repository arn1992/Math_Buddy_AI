import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List, Literal, Union, Dict, Any
import json
import time
import httpx
import traceback
import ast  # Import for literal_eval
from agents import *

# --- Pydantic Models for Structured Output (from previous successful iteration) ---
# These models define the exact JSON structure the AI_Educator is expected to output.

class ProblemBreakdownData(BaseModel):
    """Structured internal representation of the problem's anatomy and predicted student challenges."""
    problem_type: str = Field(...,
                              description="The type of math problem (e.g., 'arithmetic_word_problem', 'geometry_area').")
    core_concepts_required: List[str] = Field(...,
                                              description="List of core mathematical concepts needed (e.g., 'addition', 'subtraction', 'multiplication', 'division', 'fractions', 'area').")
    prerequisite_knowledge_check: List[str] = Field(...,
                                                    description="List of assumed or needed prior knowledge/skills (e.g., 'reading comprehension', 'basic number operations').")
    potential_misconceptions: List[str] = Field(...,
                                                description="List of common errors or misunderstandings associated with this problem type.")
    key_information_given: List[str] = Field(...,
                                             description="List of key quantities, numbers, and relevant data extracted from the problem.")
    explicit_questions: List[str] = Field(...,
                                          description="Precise restatement of the explicit question(s) the student needs to answer.")
    high_level_approaches: List[str] = Field(...,
                                             description="General strategies to solve the problem (e.g., 'drawing a diagram', 'writing an equation').")


class LearningStepsData(BaseModel):
    """A list of pedagogically sound, step-by-step instructions for the student."""
    learning_steps: List[str] = Field(..., description="A numbered list of clear, guiding steps for the student.")


class HintData(BaseModel):
    """Structured data for providing a hint to the student."""
    hint_level_chosen: str = Field(...,
                                   description="The tier/type of hint chosen (e.g., 'Tier 1: Conceptual Reminder', 'Tier 2: Strategy Suggestion').")
    hint_content: str = Field(..., description="The actual hint content or guiding question.")
    rationale_for_hint: str = Field(...,
                                    description="Brief explanation for why this specific hint was chosen at this moment.")
    expected_student_action: str = Field(...,
                                         description="What the AI Educator hopes the student will do next after receiving the hint.")


class ActivityData(BaseModel):
    """Structured data for suggesting a mini-activity to the student."""
    activity_type_chosen: str = Field(...,
                                      description="The type of activity suggested (e.g., 'Simplified Analogous Problem', 'Concept Comparison').")
    activity_content: str = Field(..., description="The description or content of the mini-activity.")
    expected_learning_outcome: str = Field(...,
                                           description="What the AI Educator expects the student to learn or practice from this activity.")
    guidance_for_student: str = Field(...,
                                      description="Instructions for the student on how to engage with the activity.")


class EvaluationData(BaseModel):
    """Structured data for evaluating a student's response."""
    response_assessment: Literal["Correct", "Partially Correct", "Incorrect"] = Field(...,
                                                                                      description="Assessment of the student's most recent response.")
    assessment_justification: str = Field(..., description="Specific reason for the assessment.")
    process_analysis: str = Field(..., description="Analysis of the student's thought process or errors, if any.")
    constructive_feedback: str = Field(..., description="Actionable and encouraging feedback for the student.")
    scaffolding_adjustment_recommendation: Literal[
        "Continue_Main_Problem",
        "Generate_New_Hint",
        "Generate_New_Activity",
        "Review_Prior_Concept",
        "re_explain_problem_part",  # Changed to lowercase 're_explain_problem_part'
        "Confirm_Mastery"
    ] = Field(..., description="The next pedagogical action recommended based on the evaluation.")


class GoalAttainmentItem(BaseModel):
    """Represents a single learning goal and whether it was met."""
    goal: str = Field(..., description="Description of the learning goal.")
    met: bool = Field(..., description="Whether the goal was met (true/false).")
    evidence: str = Field(..., description="Evidence supporting whether the goal was met.")


class MasteryConfirmationData(BaseModel):
    """Structured data for confirming mastery and providing a final reflection."""
    overall_mastery_confirmation: str = Field(...,
                                              description="Overall statement confirming the student's mastery of the concept.")
    goal_attainment_breakdown: List[GoalAttainmentItem] = Field(...,
                                                                description="Breakdown of specific learning goals and their attainment.")
    summary_of_understanding: str = Field(...,
                                          description="A comprehensive summary of the student's understanding and progress during the session.")
    next_steps_suggestion: str = Field(...,
                                       description="Suggestions for logical next steps for the student's continued learning.")


class ReExplanationData(BaseModel):
    """Structured data for re-explaining a specific part of the problem."""
    explanation_focus: str = Field(...,
                                   description="What part of the problem needs re-explaining (e.g., 'problem wording', 'context').")
    explanation_content: str = Field(..., description="The content of the re-explanation, simplified and clear.")
    follow_up_question: str = Field(...,
                                    description="A follow-up question to check understanding after the re-explanation.")


class ReviewConceptData(BaseModel):
    """Structured data for reviewing a prior concept."""
    concept_to_review: str = Field(..., description="The specific prior concept to review.")
    review_content: str = Field(..., description="Brief explanation or example of the concept.")
    review_question: str = Field(..., description="A question to gauge understanding of the reviewed concept.")


class EducatorStructuredData(BaseModel):
    """Union of all possible structured data types based on the action taken."""
    # Use Optional and default to None to allow for flexible JSON output,
    # as only one of these will be present at a time based on 'action_taken'
    problem_breakdown: Union[ProblemBreakdownData, None] = None
    learning_steps: Union[List[str], None] = None  # Learning steps are simpler strings
    hint_data: Union[HintData, None] = None
    activity_data: Union[ActivityData, None] = None
    evaluation_data: Union[EvaluationData, None] = None
    mastery_confirmation_data: Union[MasteryConfirmationData, None] = None
    re_explanation_data: Union[ReExplanationData, None] = None
    review_concept_data: Union[ReviewConceptData, None] = None


class EducatorResponse(BaseModel):
    """The conversational message and structured data from the AI Educator."""
    tone: Literal["supportive", "encouraging", "neutral", "celebratory"] = Field(...,
                                                                                 description="The tone of the AI Educator's message.")
    message: str = Field(..., description="The conversational message to the student.")
    structured_data: EducatorStructuredData = Field(..., description="Structured data relevant to the action taken.")


class AIResponse(BaseModel):
    """The top-level Pydantic model for the AI Educator's full response."""
    scaffolding_stage: Literal[
        "initial_analysis",
        "hinting_phase",
        "activity_phase",
        "feedback_phase",
        "mastery_confirmation"
    ] = Field(..., description="The current pedagogical stage of the scaffolding process.")
    action_taken: Literal[
        "problem_breakdown",
        "provide_hint",
        "suggest_activity",
        "evaluate_response",
        "confirm_mastery",
        "re_explain_problem_part",
        "review_prior_concept"
    ] = Field(..., description="The specific action taken by the AI Educator.")
    educator_response: EducatorResponse = Field(..., description="The AI Educator's response details.")


# --- Utility Functions ---

def format_history_for_llm(history):
    """Formats the conversation history into a readable string for the LLM."""
    return "\n".join([f"{who}: {what}" for who, what in history])


def run_crew_with_retry(crew, task_context_label, max_retries=3, base_delay=2):
    """
    Runs a CrewAI crew with retry logic for API/network errors and provides detailed error messages.
    """
    for attempt in range(max_retries + 1):
        try:
            print(f"\n--- Running Crew for: {task_context_label} (Attempt {attempt + 1}/{max_retries + 1}) ---")
            result = crew.kickoff()
            print(f"--- {task_context_label} Succeeded. ---")
            return result
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            print(f"🚨 HTTP Status Error {status_code} for '{task_context_label}': {e}")
            if status_code in [500, 502, 503, 504] and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            elif status_code in [400, 401, 403]:
                print("This usually indicates an API key problem, billing issue, or malformed request payload.")
                print(
                    "Please check your GEMINI_API_KEY, its status, and the structure of inputs being sent to the LLM.")
                traceback.print_exc()
                return None
            else:
                print("This is an unhandled HTTP status error. Failing immediately.")
                traceback.print_exc()
                return None
        except httpx.RequestError as e:
            print(f"🚨 Network/Request Error for '{task_context_label}': {e}")
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(
                    f"Persistent Network/Request Error for '{task_context_label}' after {max_retries + 1} attempts. Failing.")
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"🚨 An unexpected error occurred during '{task_context_label}' (Attempt {attempt + 1}): {e}")
            print(
                "This could be due to LLM output format mismatch, an internal CrewAI issue, or a bug in task context.")
            traceback.print_exc()
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(
                    f"'{task_context_label}' failed persistently after {max_retries + 1} attempts due to an unexpected error. Failing.")
                return None
    print(f"--- {task_context_label} failed after all retries. ---")
    return None


def parse_llm_output_robustly(raw_output: Any) -> Union[Dict[str, Any], None]:
    """
    Attempts to parse LLM output, first as direct JSON, then as a Python literal (like a dict string),
    and finally as raw text. Returns a dictionary or None on failure to parse as structured data.
    """
    if isinstance(raw_output, dict):  # If CrewAI already returned a dict/Pydantic model
        return raw_output
    if not isinstance(raw_output, str):
        # If it's not a string or dict, it's not parsable as JSON/literal here.
        return None

    # 1. Try direct JSON parsing
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass  # Not valid JSON, proceed to literal_eval

    # 2. Try parsing as a Python literal (e.g., dict with single quotes)
    if raw_output.strip().startswith(('{', '[')):
        try:
            # Safely evaluate string containing a Python literal
            parsed_literal = ast.literal_eval(raw_output)
            if isinstance(parsed_literal, (dict, list)):
                return parsed_literal
        except (ValueError, SyntaxError):
            pass  # Not a valid Python literal, proceed

    return None  # Could not parse as structured data
