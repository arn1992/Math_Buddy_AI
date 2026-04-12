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

# --- Configuration ---
GEMINI_API_KEY = ""  # REMINDER: USE .ENV FOR SECURITY!

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.4
)