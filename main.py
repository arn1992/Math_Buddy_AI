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
from functions import *

# --- Main Workflow Runner ---
def run_math_buddy():
    print("👋 Welcome to your 4th Grade Math Buddy! Let's conquer math together!")
    print("Type your math problem to begin. Type 'quit' to stop anytime.\n")

    while True:
        problem = input("Enter your 4th-grade math problem: ")
        if problem.lower() == "quit":
            break

        conversation_history = []
        conversation_history.append(("Student", problem))

        # These variables will store the AI_Educator's internal state and outputs
        problem_breakdown_data = {}
        learning_steps = []
        student_learning_state = {
            "misconceptions_this_session": [],
            "current_problem_steps_completed": 0,
            "overall_difficulty": "neutral"
        }

        print("\nAI Educator is initiating problem comprehension and planning...\n")

        # --- Task 1: Problem Comprehension & Diagnostic Analysis (Initial Call) ---
        initial_setup_task = Task(
            description=(
                f"As the AI Educator, your first comprehensive task is to analyze the student's problem: '{problem}'.\n"
                "Then, based on that analysis, design a structured, step-by-step learning path for a 4th-grade student.\n\n"
                "**Purpose:** To establish a deep internal understanding of any given student problem, identifying its "
                "fundamental components, anticipating potential learning obstacles, and outlining the initial guided steps.\n"
                "**Functions:**\n"
                "  1. **Problem Comprehension & Diagnostic Analysis:**\n"
                "     * Identify the subject domain, core concepts, key quantities, contextual theme, implicit constraints, and explicit questions.\n"
                "     * **Crucially:** Diagnose common misconceptions or prerequisite knowledge gaps for this problem type.\n"
                "  2. **Initial Step Generation (Adaptive Scaffolding - Part 1):**\n"
                "     * Create a structured, step-by-step learning path (3-5 steps).\n"
                "     * Each step should be a guiding question/instruction, not revealing answers.\n"
                "     * Reflect a logical pedagogical sequence suitable for 9-10 year olds.\n"
                "     * Ensure steps build confidence and align with 4th-grade math practices.\n\n"
                "Your output MUST be a JSON object conforming to the following structure for your internal thought process.\n"
                "Populate both `problem_breakdown` and `learning_steps` within `structured_data`.\n"
                "```json\n"
                "{\n"
                "  \"scaffolding_stage\": \"initial_analysis\",\n"
                "  \"action_taken\": \"problem_breakdown\",\n"
                "  \"educator_response\": {\n"
                "    \"tone\": \"neutral\",\n"
                "    \"message\": \"I've analyzed your problem and mapped out the core elements for our learning journey. Here are the steps we'll take:\",\n"
                "    \"structured_data\": {\n"
                "      \"problem_breakdown\": {\n"
                "        \"problem_type\": \"arithmetic_word_problem\",\n"
                "        \"core_concepts_required\": [\"addition\", \"subtraction\"],\n"
                "        \"prerequisite_knowledge_check\": [\"reading comprehension\", \"basic number operations\"],\n"
                "        \"potential_misconceptions\": [\"confusing operations\", \"missing key information\"],\n"
                "        \"key_information_given\": [\"number of items\", \"items given away\"],\n"
                "        \"explicit_questions\": [\"how many remaining\"],\n"
                "        \"high_level_approaches\": [\"drawing a diagram\", \"writing an equation\"]\n"
                "      },\n"
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
            expected_output="A JSON object combining problem_breakdown and learning_steps.",
            output_json=AIResponse  # Using the Pydantic model for output validation
        )

        initial_setup_crew = Crew(
            agents=[ai_educator_agent],
            tasks=[initial_setup_task],
            process=Process.sequential,
            verbose=True
        )

        try:
            initial_output_obj = run_crew_with_retry(initial_setup_crew, "initial problem setup")
            if initial_output_obj is None:
                print(
                    "\nFailed to get initial problem analysis and steps after multiple attempts. Moving to the next problem.")
                continue

            # Robustly parse the LLM's raw output
            parsed_raw_output = parse_llm_output_robustly(
                initial_output_obj.raw_output if hasattr(initial_output_obj, 'raw_output') else str(initial_output_obj))

            if parsed_raw_output is None:
                print(
                    f"🚨 Error: Could not parse initial setup response as JSON or Python literal. Raw output:\n{initial_output_obj.raw_output if hasattr(initial_output_obj, 'raw_output') else str(initial_output_obj)}")
                print("Could not process initial problem analysis and steps. Try a different problem.")
                continue

            try:
                # Convert parsed dict/list back to a JSON string for Pydantic validation
                initial_data = AIResponse.model_validate_json(json.dumps(parsed_raw_output))
            except Exception as e:
                print(f"🚨 Error validating initial setup response against Pydantic model: {e}")
                print(f"Parsed data attempting to validate:\n{parsed_raw_output}")
                traceback.print_exc()
                print("Could not process initial problem analysis and steps. Try a different problem.")
                continue

            # Access data from the validated Pydantic model
            problem_breakdown_data = initial_data.educator_response.structured_data.problem_breakdown.model_dump() if initial_data.educator_response.structured_data.problem_breakdown else {}
            learning_steps = initial_data.educator_response.structured_data.learning_steps if initial_data.educator_response.structured_data.learning_steps else []
            initial_message = initial_data.educator_response.message

            if not learning_steps:
                print("⚠️ I couldn't generate clear steps for that problem. Try a different one.")
                continue

            print(f"💬 AI Educator: {initial_message}")
            conversation_history.append(("AI Educator", initial_message))
            conversation_history.extend([("AI Educator (Step)", s) for s in learning_steps])

            print("\n--- Problem Breakdown & Learning Path ---")
            for i, step in enumerate(learning_steps):
                print(f"📘 Step {i + 1}: {step}")
            print(
                "\nLet's go step-by-step. Just tell me your thoughts, or type 'done' if you're ready for the next step, 'hint' for a nudge, or 'restart' to start over.")

            current_step_index = 0
            while current_step_index < len(learning_steps):
                current_guidance_step = learning_steps[current_step_index]
                print(f"\n📘 Current focus (Step {current_step_index + 1}): {current_guidance_step}")

                user_input = input(
                    "🧒 Your turn (e.g., 'I think 5+6 is 11', 'hint', 'done', 'quit', 'restart'): ").lower()
                conversation_history.append(("Student", user_input))

                if user_input == "quit":
                    print("👋 See you next time, Math Explorer!")
                    return
                elif user_input == "restart":
                    print("🔁 Okay, let's restart with a fresh problem!")
                    break
                elif user_input == "done":
                    buddy_response = "✅ Fantastic! Moving on to the next step."
                    print(buddy_response)
                    conversation_history.append(("AI Educator", buddy_response))
                    current_step_index += 1
                    continue

                # --- Task 2: Adaptive Scaffolding & Targeted Intervention (Interactive Loop) ---
                scaffolding_interaction_task = Task(
                    description=(
                        f"As the AI Educator, your task is to dynamically guide the student through the problem-solving process and "
                        f"conceptual understanding based on their latest input and the current learning step. You will output a "
                        f"structured JSON response reflecting your internal decision-making.\n\n"
                        f"**Dynamic Scaffolding Task & Workflow (Internal Thought Process):**\n\n"
                        f"You will operate in a continuous, responsive loop, adapting your response based on the student's input and your internal assessment of their current stage of learning and needs. Your interactions will flow through distinct pedagogical phases.\n\n"
                        f"**Input for each interaction:**\n"
                        f"* `Student_Problem`: '{problem}'\n"
                        f"* `Student_Current_Input`: '{user_input}'\n"
                        f"* `Interaction_History_Summary`: {format_history_for_llm(conversation_history)}\n"
                        f"* `Learning_Goals_for_Problem`: {json.dumps(learning_steps)} "  # Using learning steps as learning goals
                        f"- `Student_Learning_State`: {json.dumps(student_learning_state)} (includes `misconceptions_this_session`, `current_problem_steps_completed`)\n"
                        f"- `Problem_Breakdown_Data`: {json.dumps(problem_breakdown_data)}\n\n"
                        f"**Output Format:** Your response MUST be a JSON object conforming to the structure from your internal thought process.\n"
                        f"Populate fields based on the chosen `action_taken` (e.g., 'evaluate_response', 'provide_hint', 'suggest_activity').\n\n"
                        f"```json\n"
                        f"{{\n"
                        f"  \"scaffolding_stage\": \"initial_analysis\" | \"hinting_phase\" | \"activity_phase\" | \"feedback_phase\" | \"mastery_confirmation\",\n"
                        f"  \"action_taken\": \"problem_breakdown\" | \"provide_hint\" | \"suggest_activity\" | \"evaluate_response\" | \"confirm_mastery\" | \"re_explain_problem_part\" | \"review_prior_concept\",\n"
                        f"  \"educator_response\": {{\n"
                        f"    \"tone\": \"supportive\" | \"encouraging\" | \"neutral\" | \"celebratory\",\n"
                        f"    \"message\": \"Your conversational message to the student.\",\n"
                        f"    \"structured_data\": {{\n"
                        f"      // This object will be populated based on the \"action_taken\"\n"
                        f"      // If \"problem_breakdown\":\n"
                        f"      \"problem_type\": \"...\",\n"
                        f"      \"core_concepts_required\": [\"...\", \"...\"],\n"
                        f"      \"prerequisite_knowledge_check\": [\"\", \"\"], // What knowledge is assumed or needed?\n"
                        f"      \"potential_misconceptions\": [\"\", \"\"], // What common errors might arise?\n"
                        f"      \"key_information_given\": [\"\", \"\"],\n"
                        f"      \"explicit_questions\": [\"\", \"\"],\n"
                        f"      \"high_level_approaches\": [\"\", \"\"] // General strategies like \"draw a diagram\"\n"
                        f"      // If \"provide_hint\":\n"
                        f"      \"hint_level_chosen\": \"Tier X: ...\", // e.g., \"Tier 1: Conceptual Reminder\"\n"
                        f"      \"hint_content\": \"...\",\n"
                        f"      \"rationale_for_hint\": \"...\", // Brief explanation for *why* this hint now\n"
                        f"      \"expected_student_action\": \"...\" // What are you hoping the student does next?\n"
                        f"      // If \"suggest_activity\":\n"
                        f"      \"activity_type_chosen\": \"...\", // e.g., \"Simplified Analogous Problem\", \"Concept Comparison\"\n"
                        f"      \"activity_content\": \"...\",\n"
                        f"      \"expected_learning_outcome\": \"...\",\n"
                        f"      \"guidance_for_student\": \"...\"\n"
                        f"      // If \"evaluate_response\":\n"
                        f"      \"response_assessment\": \"Correct\" | \"Partially Correct\" | \"Incorrect\",\n"
                        f"      \"assessment_justification\": \"...\", // Specific reason for the assessment\n"
                        f"      \"process_analysis\": \"...\", // Analysis of their thought process/errors\n"
                        f"      \"constructive_feedback\": \"...\",\n"
                        f"      \"scaffolding_adjustment_recommendation\": \"Continue_Main_Problem\" | \"Generate_New_Hint\" | \"Generate_New_Activity\" | \"Review_Prior_Concept\" | \"re_explain_problem_part\" | \"Confirm_Mastery\"\n"
                        f"      // If \"confirm_mastery\":\n"
                        f"      \"overall_mastery_confirmation\": \"...\",\n"
                        f"      \"goal_attainment_breakdown\": [{{\"goal\": \"...\", \"met\": true/false, \"evidence\": \"...\"}}, ...],\n"
                        f"      \"summary_of_understanding\": \"...\",\n"
                        f"      \"next_steps_suggestion\": \"...\"\n"
                        f"      // If \"re_explain_problem_part\":\n"
                        f"      \"explanation_focus\": \"...\", // What part of the problem needs re-explaining?\n"
                        f"      \"explanation_content\": \"...\",\n"
                        f"      \"follow_up_question\": \"...\"\n"
                        f"      // If \"review_prior_concept\":\n"
                        f"      \"concept_to_review\": \"...\",\n"
                        f"      \"review_content\": \"...\", // Brief explanation or example of the concept\n"
                        f"      \"review_question\": \"...\"\n"
                        f"    }}\n"
                        f"  }}\n"
                        f"}}\n"
                        f"```\n"
                        f"**Core Logic for this task:**\n"
                        f"1. **Evaluate `Student_Current_Input`**: Analyze it against `Current_Learning_Step` and `All_Learning_Steps` to determine understanding.\n"
                        f"2. **Determine `scaffolding_adjustment_recommendation`**: Based on evaluation, decide the next pedagogical move.\n"
                        f"3. **Generate `educator_response`**: Craft a message (and associated structured data) corresponding to the chosen action. If a hint is needed, provide one. If on track, provide positive feedback."
                        f"**Constraint:** All interactions must consistently promote student confidence, foster independent reasoning, and build toward mastery without directly revealing solutions. Prioritize guiding questions and conceptual understanding over direct answers."
                    ),
                    agent=ai_educator_agent,
                    expected_output="A JSON object conforming to the detailed internal thought process schema.",
                    output_json=AIResponse  # Using the Pydantic model for output validation
                )

                scaffolding_crew = Crew(
                    agents=[ai_educator_agent],
                    tasks=[scaffolding_interaction_task],
                    process=Process.sequential,
                    verbose=True
                )

                buddy_response_message = "I'm experiencing an issue and can't respond right now."
                processed_successfully = False
                recommendation = None  # Initialize recommendation here

                try:
                    scaffolding_output_obj = run_crew_with_retry(scaffolding_crew,
                                                                 f"scaffolding interaction for step {current_step_index + 1}")
                    if scaffolding_output_obj:
                        # Robustly parse the LLM's raw output
                        parsed_raw_output = parse_llm_output_robustly(
                            scaffolding_output_obj.raw_output if hasattr(scaffolding_output_obj, 'raw_output') else str(
                                scaffolding_output_obj))

                        if parsed_raw_output is None:
                            print(
                                f"🚨 Error: Could not parse scaffolding interaction response as JSON or Python literal. Raw output:\n{scaffolding_output_obj.raw_output if hasattr(scaffolding_output_obj, 'raw_output') else str(scaffolding_output_obj)}")
                            buddy_response_message = "AI Educator could not process the response due to unexpected format."
                            print(f"💬 AI Educator: {buddy_response_message}")
                            conversation_history.append(("AI Educator", buddy_response_message))
                            continue  # Continue to next loop iteration (re-prompts same step)

                        try:
                            # Convert parsed dict/list back to a JSON string for Pydantic validation
                            ai_educator_response_data = AIResponse.model_validate_json(json.dumps(parsed_raw_output))
                        except Exception as e:
                            print(f"🚨 Error validating scaffolding interaction response against Pydantic model: {e}")
                            print(f"Parsed data attempting to validate:\n{parsed_raw_output}")
                            traceback.print_exc()
                            buddy_response_message = "AI Educator could not process the response due to unexpected format."
                            print(f"💬 AI Educator: {buddy_response_message}")
                            conversation_history.append(("AI Educator", buddy_response_message))
                            continue

                        # Proceed if ai_educator_response_data is a valid AIResponse model
                        educator_response_section = ai_educator_response_data.educator_response
                        structured_data_section = educator_response_section.structured_data
                        action_taken = ai_educator_response_data.action_taken
                        buddy_response_message = educator_response_section.message

                        if action_taken == 'evaluate_response' and structured_data_section.evaluation_data:
                            recommendation = structured_data_section.evaluation_data.scaffolding_adjustment_recommendation
                            if structured_data_section.evaluation_data.response_assessment == 'Incorrect':
                                misconception_detail = structured_data_section.evaluation_data.process_analysis
                                student_learning_state['misconceptions_this_session'].append(misconception_detail)
                        elif action_taken == 'provide_hint' and structured_data_section.hint_data:
                            recommendation = "Generate_New_Hint"  # Explicitly set for loop control
                        elif action_taken == 'suggest_activity' and structured_data_section.activity_data:
                            recommendation = "Generate_New_Activity"  # Explicitly set for loop control

                        processed_successfully = True

                    else:  # If run_crew_with_retry returned None
                        print("❗️ run_crew_with_retry returned None. AI Educator could not provide a response.")
                        buddy_response_message = "AI Educator could not provide a structured response. Providing generic nudge."
                except Exception as e:  # This catches errors from run_crew_with_retry itself, or unexpected top-level issues
                    print(f"🚨 Top-level unexpected error during scaffolding interaction: {e}")
                    traceback.print_exc()
                    buddy_response_message = "I'm experiencing an unexpected issue. Could you please try rephrasing your thought?"

                print(f"💬 AI Educator: {buddy_response_message}")
                conversation_history.append(("AI Educator", buddy_response_message))

                # Logic to advance step based on recommendation
                if processed_successfully and recommendation == "Continue_Main_Problem":
                    current_step_index += 1
                elif processed_successfully and recommendation == "Confirm_Mastery":
                    current_step_index = len(learning_steps)  # End the loop

                continue

                # --- Task 3: Progress Monitoring & Mastery Validation (Final Call) ---
            if current_step_index >= len(learning_steps):
                final_message_1 = "\n🎉 Amazing! You've navigated through all the steps like a true math whiz!"
                print(final_message_1)
                conversation_history.append(("AI Educator", final_message_1))

                mastery_validation_task = Task(
                    description=(
                        f"As the AI Educator, your task is to confirm the student's mastery and provide a comprehensive reflection. "
                        f"The student has completed all the learning steps for problem: '{problem}'.\n"
                        f"Review the full conversation history: {format_history_for_llm(conversation_history)} "
                        f"and the student's final learning state: {json.dumps(student_learning_state)}.\n"
                        f"The initial problem breakdown was: {json.dumps(problem_breakdown_data)}\n\n"
                        "**Purpose:** To continuously track the student's journey, assess their understanding at critical junctures, and ultimately confirm the achievement of learning goals.\n"
                        "**Functions:**\n"
                        "  * Maintain an internal summary of the student's progress, including successful steps, persistent difficulties, and areas of growth throughout the session.\n"
                        "  * Verify that all defined learning objectives for the problem have been met (based on the steps being completed and interactions).\n"
                        "  * Provide a comprehensive summary of the student's understanding and progress.\n"
                        "  * Offer strong, personalized positive reinforcement and suggest logical next steps for continued learning.\n"
                        "**Constraint:** Do NOT provide the final numerical answer to the original problem. Focus entirely on the learning process, growth, and future learning.\n"
                        "Your output MUST be a JSON object conforming to the 'confirm_mastery' structure from your internal thought process.\n"
                        "**Specifically, populate these fields**: `overall_mastery_confirmation`, `goal_attainment_breakdown`, `summary_of_understanding`, `next_steps_suggestion`.\n"
                        "Example of expected output for mastery confirmation (full JSON object with 'confirm_mastery' structured_data):\n"
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
                    output_json=AIResponse  # Using the Pydantic model for output validation
                )

                reflection_crew = Crew(
                    agents=[ai_educator_agent],
                    tasks=[mastery_validation_task],
                    process=Process.sequential,
                    verbose=True
                )
                try:
                    summary_obj = run_crew_with_retry(reflection_crew, "final reflection and mastery validation")
                    if summary_obj:
                        # Robustly parse the LLM's raw output
                        parsed_raw_output_summary = parse_llm_output_robustly(
                            summary_obj.raw_output if hasattr(summary_obj, 'raw_output') else str(summary_obj))

                        if parsed_raw_output_summary is None:
                            print(
                                f"🚨 Error: Could not parse final reflection response as JSON or Python literal. Raw output:\n{summary_obj.raw_output if hasattr(summary_obj, 'raw_output') else str(summary_obj)}")
                            summary_message = "Great job finishing this problem! Keep up the amazing work!"
                            summary_details = {}
                            print(f"✨ AI Educator: {summary_message}")
                            conversation_history.append(("AI Educator", summary_message))
                            continue  # Continue to next problem

                        try:
                            summary_data = AIResponse.model_validate_json(json.dumps(parsed_raw_output_summary))
                        except Exception as e:  # Catch any Pydantic or JSON decode errors
                            print(f"🚨 JSON validation/parsing error from AI Educator's reflection response: {e}")
                            print(f"Raw output causing error:\n{parsed_raw_output_summary}")  # Changed here
                            traceback.print_exc()
                            summary_message = "Great job finishing this problem! Keep up the amazing work!"
                            summary_details = {}
                            print(f"✨ AI Educator: {summary_message}")
                            conversation_history.append(("AI Educator", summary_message))
                            continue  # Continue to next problem

                        # Access data from the validated Pydantic model
                        summary_message = summary_data.educator_response.message
                        # Ensure mastery_confirmation_data is accessed safely
                        summary_details = summary_data.educator_response.structured_data.mastery_confirmation_data

                        print(f"✨ AI Educator: {summary_message}")
                        if summary_details:  # Check if mastery_confirmation_data is not None
                            print(f"Overall Mastery: {summary_details.overall_mastery_confirmation}")
                            print(f"Summary: {summary_details.summary_of_understanding}")
                            print(f"Next Steps: {summary_details.next_steps_suggestion}")
                        conversation_history.append(("AI Educator", summary_message))
                    else:
                        print("❗️ Could not generate a reflection after retries. Providing a generic closing message.")
                        print("✨ AI Educator: Great job finishing this problem! Keep up the amazing work!")
                        conversation_history.append(
                            ("AI Educator", "Great job finishing this problem! Keep up the amazing work!"))
                except Exception as e:  # Catch any other unexpected errors during summary processing
                    print(f"🚨 Unexpected error during reflection flow after retries: {e}")
                    traceback.print_exc()
                    print("✨ AI Educator: Great job finishing this problem! Keep up the amazing work!")
                    conversation_history.append(
                        ("AI Educator", "Great job finishing this problem! Keep up the amazing work!"))

                final_message_2 = "Now, armed with your step-by-step understanding, take a moment to solve the full problem on your own. You've got all the tools you need!"
                print(final_message_2)
                conversation_history.append(("AI Educator", final_message_2))

                final_message_3 = "Ready for another challenge? Just type in your next problem!"
                print(final_message_3)
                conversation_history.append(("AI Educator", final_message_3))

        except Exception as e:
            print(f"🚨 An unhandled critical error occurred during the Math Buddy session: {e}")
            traceback.print_exc()
            print("\nIt seems we hit a major snag. Let's try starting fresh with a new problem.")
            continue

    print("👋 Goodbye for now, Math Explorer! Keep that brain sharp!")


if __name__ == "__main__":
    run_math_buddy()
