# Configuration settings for DecisionForge

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API and Model Configuration ---
API_KEY = os.getenv("XAI_API_KEY")
API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.x.ai/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "grok-3-beta")
APP_PASSWORD_HASH = os.getenv("APP_PASSWORD") # Store the plain password here for now, will be hashed in app.py if bcrypt is used

# --- Simulation Parameters ---
# Decision types (example list, can be customized)
DECISION_TYPES = [
    "Strategic Alliance",
    "Budget Allocation",
    "Product Launch",
    "Crisis Management",
    "Policy Change",
    "Resource Prioritization",
    "Ethical Dilemma",
    "Operational Improvement",
    "Other"
]

# Default number of debate rounds
DEBATE_ROUNDS = 5
# Default maximum simulation time in seconds (e.g., 3 minutes)
MAX_SIMULATION_TIME_SECONDS = 180
# Maximum tokens for LLM responses in simulation
MAX_TOKENS_PER_RESPONSE = 600
# Timeout for individual API calls in seconds
API_TIMEOUT_SECONDS = 30

# --- Persona Generation Parameters ---
MIN_STAKEHOLDERS = 3
MAX_STAKEHOLDERS = 7 # Adjusted based on README description

# Suggestions for generating persona traits (used if LLM needs guidance)
STAKEHOLDER_ANALYSIS = {
    "psychological_traits": [
        "risk-averse", "risk-tolerant", "collaborative", "competitive",
        "analytical", "decisive", "cautious", "impulsive", "innovative",
        "pragmatic", "detail-oriented", "big-picture thinker"
    ],
    "influences": [
        "regulatory bodies", "public opinion", "shareholders", "media",
        "competitors", "government policies", "industry trends",
        "community stakeholders", "environmental groups", "internal politics",
        "key advisors", "financial markets"
    ],
    "biases": [
        "confirmation bias", "optimism bias", "groupthink", "status quo bias",
        "cost-avoidance bias", "anchoring bias", "overconfidence bias",
        "availability bias", "recency bias", "hindsight bias"
    ],
    "historical_behavior": [
        "prioritizes short-term gains", "focuses on long-term strategy",
        "consensus-driven", "unilateral decision-maker", "data-driven",
        "resistant to change", "proactive in innovation", "reactive to crises",
        "relationship-focused", "task-focused"
    ],
    "tones": [
        "diplomatic", "assertive", "empathetic", "analytical", "cautious",
        "direct", "persuasive", "skeptical", "inspirational", "neutral"
    ]
}

# --- Simulation Methods ---
SIMULATION_METHODS = [
    f"{MODEL_NAME} Simulation", # Dynamically uses the configured model name
    "Simplified Monte Carlo Simulation",
    "Simplified Game Theory Simulation",
    # Add more methods here if developed
]

# --- Visualization Settings ---
# Example: Define color palettes or graph styles if needed
PLOTLY_TEMPLATE = "plotly_white" # Example: use a clean plotly theme

# --- Constants ---
PERSONAS_DIR = "personas"
DATABASE_FILE = "decisionforge.db"
