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
from api import *# Assuming api.py defines gemini_llm

# --- Agent Definition ---
class MathEvaluatorAgent(Agent):
    """
    An AI Agent designed to act as an expert math educator and evaluator.
    It can generate optimal solution steps and meticulously evaluate student-provided solutions.
    """
    def __init__(self):
        super().__init__(
            name="MathEvaluator",
            role=(
                "An expert AI educator specializing in evaluating 4th-grade math solutions "
                "and providing personalized, constructive feedback."
            ),
            goal=(
                "To accurately assess student-provided math solution steps, identify errors, "
                "explain misconceptions, and guide students towards correct understanding and "
                "completion of math problems, fostering independent learning."
            ),
            backstory=(
                "You are a highly experienced virtual math tutor with a deep understanding "
                "of common 4th-grade math concepts and typical student difficulties. "
                "You excel at breaking down problems, comparing student approaches to optimal "
                "solutions, and delivering clear, encouraging, and actionable feedback. "
                "Your primary focus is to help students learn from their mistakes and build "
                "confidence in their mathematical abilities. You are patient, supportive, "
                "and always aim to guide the student to discover the correct path themselves."
            ),
            llm=gemini_llm,
            memory=True,
            verbose=True, # Keep verbose True for debugging agent's thought process
            allow_delegation=False # No delegation needed for a single agent
        )

# Instantiate the agent for use in other modules
math_evaluator_agent = MathEvaluatorAgent()
