import json
import os
import time
import random
import numpy as np
from openai import OpenAI, OpenAIError, APITimeoutError
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import streamlit as st # Use streamlit for showing errors/warnings

# Import config variables
from config import (
    API_KEY, API_BASE_URL, MODEL_NAME, DEBATE_ROUNDS,
    MAX_TOKENS_PER_RESPONSE, API_TIMEOUT_SECONDS, MAX_SIMULATION_TIME_SECONDS
)

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# --- Agent State ---
# Simple dictionary to hold agent state across rounds (can be expanded)
agent_states = {}

def initialize_agent_states(personas: List[Dict]):
    """Initializes or resets the state for each agent."""
    global agent_states
    agent_states = {}
    for p in personas:
        agent_states[p['name']] = {
            "satisfaction": 0.5, # Example: 0=very unhappy, 1=very happy
            "negotiation_stance": "neutral", # Could be 'cooperative', 'competitive', 'compromising'
            "key_points_made": set(), # Track points they've emphasized
            "agreements_reached": set(), # Track agreements they are part of
            "round_commitments": {} # Commitments made in specific rounds
        }

# --- LLM Simulation ---

@retry(stop=stop_after_attempt(3), wait=fixed(2), reraise=True)
def _make_debate_api_call(prompt: str) -> Optional[str]:
    """Makes the API call to the LLM for a debate turn with retries."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are participating in a multi-stakeholder decision-making simulation. Act according to the persona provided. Your response should be a single JSON object containing your message."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.75, # Slightly higher temp for more human-like debate variance
            max_tokens=MAX_TOKENS_PER_RESPONSE,
            timeout=API_TIMEOUT_SECONDS,
            response_format={"type": "json_object"} # Expect JSON containing the message
        )
        response_content = completion.choices[0].message.content
        # Attempt to parse JSON to extract the message
        try:
            message_data = json.loads(response_content)
            if isinstance(message_data, dict) and 'message' in message_data:
                return message_data['message']
            else:
                 st.warning(f"LLM response JSON missing 'message' key: {response_content[:200]}...", icon="⚠️")
                 return f"Error: Invalid response format received. ({response_content[:100]}...)" # Fallback message
        except json.JSONDecodeError:
             st.warning(f"LLM response was not valid JSON: {response_content[:200]}...", icon="⚠️")
             # If not JSON, maybe the LLM just outputted text? Return it directly as a fallback.
             # Could indicate prompt issues or model limitations.
             return response_content # Return raw content as fallback message


    except APITimeoutError:
        st.warning(f"API call timed out after {API_TIMEOUT_SECONDS} seconds.", icon="⏳")
        return "My response timed out, but I am still considering the options." # Fallback message
    except OpenAIError as e:
        st.error(f"API Error during debate turn: {e}", icon="🚨")
        raise # Reraise to trigger tenacity retry
    except Exception as e:
        st.error(f"Unexpected error during debate API call: {e}", icon="🚨")
        raise # Reraise unexpected errors

def _run_llm_simulation_round(
    round_num: int,
    personas: List[Dict],
    current_step: str,
    objective: str,
    cumulative_context: str,
    start_time: float,
    max_time: int
    ) -> Tuple[List[Dict], str]:
    """Runs one round of the LLM-based debate simulation."""
    round_transcript = []
    round_context_addition = f"\n--- Round {round_num + 1}: {current_step} ---\nObjective: {objective}\n"

    # Randomize speaking order slightly? Or fixed order? Fixed for now.
    # random.shuffle(personas)

    for persona in personas:
         # --- Timeout Check ---
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            st.warning(f"Simulation interrupted: Exceeded maximum time of {max_time} seconds.", icon="⏳")
            round_transcript.append({
                "agent": "System", "round": round_num + 1, "step": current_step,
                "message": f"Timeout reached ({max_time}s). Ending round prematurely."
            })
            break # Exit loop for this round

        stakeholder_name = persona["name"]
        # Access current state (optional, for more complex logic)
        # current_state = agent_states.get(stakeholder_name, {})

        # --- Construct Prompt ---
        # Limit context size to avoid excessive token usage
        context_limit = 3000 # Characters
        truncated_context = cumulative_context[-(context_limit):]
        if len(cumulative_context) > context_limit:
            truncated_context = "... " + truncated_context # Indicate truncation

        prompt = (
             f"**Your Persona:**\n"
             f"- Name: {stakeholder_name}\n"
             f"- Role: {persona.get('role', 'N/A')}\n"
             f"- Goals: {', '.join(persona.get('goals', []))}\n"
             f"- Biases: {', '.join(persona.get('biases', []))}\n"
             f"- Tone: {persona.get('tone', 'neutral')}\n"
             f"- Motivators: {', '.join(persona.get('motivators', []))}\n"
             f"- Expected Behavior: {persona.get('expected_behavior', 'Participate constructively.')}\n\n"
             # f"- Current State: Satisfaction={current_state.get('satisfaction'):.2f}, Stance={current_state.get('negotiation_stance')}\n\n" # Optional state injection
             f"**Debate Context:**\n"
             f"Current Step: {current_step} (Round {round_num + 1})\n"
             f"Objective for this step: {objective}\n"
             f"Recent Discussion (last {context_limit} chars):\n```\n{truncated_context}\n```\n\n"
             f"**Your Task:**\nProvide your response based on your persona and the context. "
             f"Address the objective for this step. Consider previous points. "
             f"Keep your response concise (around 150-200 words). "
             f"Format your entire output as a single JSON object: {{\"message\": \"Your response here...\"}}"
         )

        # --- Make API Call ---
        message_content = "Error: Could not generate response." # Default message
        try:
            response = _make_debate_api_call(prompt)
            if response:
                message_content = response
            # else: message remains the error default set above

        except RetryError:
            message_content = "Error: API call failed after multiple retries."
            st.error(f"API call failed for {stakeholder_name} after retries.", icon="🚨")
        except Exception as e:
            message_content = f"Error: An unexpected error occurred ({type(e).__name__})."
            st.error(f"Unexpected error for {stakeholder_name}: {e}", icon="🚨")


        # --- Record Turn ---
        turn_entry = {
            "agent": stakeholder_name,
            "round": round_num + 1,
            "step": current_step,
            "message": message_content.strip()
        }
        round_transcript.append(turn_entry)
        round_context_addition += f" - {stakeholder_name}: {message_content[:150]}...\n" # Add summary to round context

        # --- Update Agent State (Example) ---
        # This is where you'd add logic to update agent_states based on the message
        # e.g., parse sentiment, detect agreements/disagreements, update satisfaction
        # update_agent_state(stakeholder_name, message_content)


    return round_transcript, round_context_addition


# --- Simplified Monte Carlo Simulation ---

def _run_monte_carlo_round(
    round_num: int,
    personas: List[Dict],
    current_step: str,
    objective: str,
    cumulative_context: str # Context not heavily used here, but passed for consistency
    ) -> Tuple[List[Dict], str]:
    """Runs one round of the simplified Monte Carlo simulation."""
    round_transcript = []
    round_context_addition = f"\n--- Round {round_num + 1}: {current_step} (Monte Carlo) ---\nObjective: {objective}\n"

    # Simple model: Each agent makes a probabilistic choice based on traits
    for persona in personas:
        stakeholder_name = persona["name"]
        traits = persona.get("psychological_traits", [])
        biases = persona.get("biases", [])

        # Define probabilities based on traits/biases
        prob_agree = 0.4
        prob_disagree = 0.3
        prob_propose = 0.2
        prob_question = 0.1

        if "collaborative" in traits: prob_agree += 0.2; prob_disagree -= 0.1
        if "competitive" in traits: prob_disagree += 0.2; prob_agree -= 0.1
        if "analytical" in traits or "cautious" in traits: prob_question += 0.15; prob_propose -= 0.05
        if "decisive" in traits or "innovative" in traits: prob_propose += 0.15; prob_question -= 0.05
        if "optimism bias" in biases: prob_agree += 0.1; prob_disagree -= 0.05
        if "status quo bias" in biases: prob_disagree += 0.1 # Disagree with changes

        # Normalize probabilities
        total_prob = prob_agree + prob_disagree + prob_propose + prob_question
        probabilities = [prob_agree/total_prob, prob_disagree/total_prob, prob_propose/total_prob, prob_question/total_prob]
        probabilities = [max(0, p) for p in probabilities] # Ensure non-negative
        probabilities = [p / sum(probabilities) for p in probabilities] # Re-normalize after clamping

        action = np.random.choice(
            ["support", "challenge", "propose", "question"],
            p=probabilities
        )

        # Generate a generic message based on the action
        message = f"As {stakeholder_name} ({persona.get('role', 'N/A')}), considering the objective '{objective}', my stance is to **{action}** the current direction."
        if action == "support":
            message += " I believe this aligns with our goals [" + ", ".join(persona.get('goals', ['general progress'])) + "]."
        elif action == "challenge":
            message += " I have concerns regarding potential risks or misalignment with priorities like [" + ", ".join(persona.get('influences', ['efficiency'])) + "]."
        elif action == "propose":
            message += f" Perhaps we could consider an alternative approach focusing on {random.choice(persona.get('goals', ['innovation']))}?"
        elif action == "question":
             message += f" Can we clarify how this addresses the concerns of {random.choice(persona.get('influences', ['stakeholders']))}?"


        turn_entry = {
            "agent": stakeholder_name,
            "round": round_num + 1,
            "step": current_step,
            "message": message
        }
        round_transcript.append(turn_entry)
        round_context_addition += f" - {stakeholder_name}: {action.upper()}\n"

    return round_transcript, round_context_addition

# --- Simplified Game Theory Simulation ---

def _run_game_theory_round(
    round_num: int,
    personas: List[Dict],
    current_step: str,
    objective: str,
    cumulative_context: str # Context not heavily used
    ) -> Tuple[List[Dict], str]:
    """Runs one round of the simplified game theory simulation (e.g., Prisoner's Dilemma variant)."""
    round_transcript = []
    round_context_addition = f"\n--- Round {round_num + 1}: {current_step} (Game Theory) ---\nObjective: {objective}\n"

    # Payoff matrix (Cooperate, Defect) - simplified interpretation
    # (My Payoff, Opponent Payoff)
    payoffs = {
        ('Cooperate', 'Cooperate'): (3, "Mutual Progress"),
        ('Cooperate', 'Defect'):    (0, "Exploited"),
        ('Defect', 'Cooperate'):    (5, "Individual Gain"),
        ('Defect', 'Defect'):       (1, "Stalemate/Conflict")
    }
    strategies = ['Cooperate', 'Defect']
    agent_strategies = {}

    # 1. Determine strategy for each agent based on traits/goals
    for persona in personas:
        name = persona['name']
        traits = persona.get('psychological_traits', [])
        goals = persona.get('goals', [])
        # Simple logic: Cooperate if collaborative/empathetic, defect if competitive/assertive, otherwise random
        if any(t in traits for t in ['collaborative', 'empathetic']):
            agent_strategies[name] = 'Cooperate'
        elif any(t in traits for t in ['competitive', 'assertive']):
             agent_strategies[name] = 'Defect'
        elif "long-term strategy" in persona.get("historical_behavior",""):
             # Long term players might cooperate more often
             agent_strategies[name] = random.choices(strategies, weights=[0.6, 0.4], k=1)[0]
        else:
            agent_strategies[name] = random.choice(strategies)

    # 2. Simulate interactions (simplified: each agent 'plays' against a generic 'group')
    for persona in personas:
        name = persona['name']
        my_strategy = agent_strategies[name]

        # Determine 'opponent' strategy (average or most common?) - simplified: assume 50/50 cooperation/defection in the 'group'
        opponent_strategy = random.choice(strategies)

        payoff_value, outcome_desc = payoffs.get((my_strategy, opponent_strategy), (1, "Default Outcome"))

        message = (f"As {name} ({persona.get('role', 'N/A')}), facing the objective '{objective}', my chosen strategy is **{my_strategy}**. "
                   f"Based on the assumed group dynamic, this leads to an outcome of '{outcome_desc}' with a payoff value of {payoff_value} for me. "
                   f"This aligns with my goal(s): {', '.join(persona.get('goals',[]))}."
                   )

        turn_entry = {
            "agent": name,
            "round": round_num + 1,
            "step": current_step,
            "message": message
        }
        round_transcript.append(turn_entry)
        round_context_addition += f" - {name}: Strategy={my_strategy}, Payoff={payoff_value}\n"


    return round_transcript, round_context_addition


# --- Main Simulation Function ---

def simulate_debate(
    personas: List[Dict],
    dilemma: str,
    extracted_structure: Dict,
    # scenarios: str = "", # Optional: Add logic to use scenarios if needed
    rounds: int = DEBATE_ROUNDS,
    max_simulation_time: int = MAX_SIMULATION_TIME_SECONDS,
    simulation_type: str = f"{MODEL_NAME} Simulation"
    ) -> List[Dict]:
    """
    Simulates a debate among stakeholder personas using the specified method.

    Args:
        personas (List[Dict]): Detailed personas with goals, biases, tone, etc.
        dilemma (str): The core decision dilemma.
        extracted_structure (Dict): Contains 'process_steps', 'key_issues', etc.
        rounds (int): Number of debate rounds to simulate.
        max_simulation_time (int): Max simulation duration in seconds.
        simulation_type (str): The simulation method to use.

    Returns:
        List[Dict]: The debate transcript.
    """
    transcript = []
    start_time = time.time()

    # --- Input Validation ---
    if not personas:
        st.error("Cannot start simulation: No personas provided.", icon="🚨")
        return [{"agent": "System", "round": 0, "step": "Setup", "message": "Error: No personas loaded."}]
    if not extracted_structure or 'process_steps' not in extracted_structure:
         st.error("Cannot start simulation: Missing decision process steps.", icon="🚨")
         return [{"agent": "System", "round": 0, "step": "Setup", "message": "Error: Missing process steps."}]

    process_steps = extracted_structure.get("process_steps", [])
    key_issues = extracted_structure.get("key_issues", [])

    # Ensure process steps cover the number of rounds
    if not process_steps: process_steps = ["Discussion"] # Fallback step
    num_steps = len(process_steps)
    if num_steps < rounds:
        process_steps.extend([f"Continuing {process_steps[-1]}"] * (rounds - num_steps))
    process_steps = process_steps[:rounds] # Limit to the specified number of rounds

    # --- Define Objectives (Simple Mapping) ---
    # More sophisticated mapping based on step keywords could be added
    objectives = {
        i: f"Discuss '{key_issues[i % len(key_issues)] if key_issues else 'the main topic'}' within the context of '{step}'"
        for i, step in enumerate(process_steps)
    }

    # --- Initial Context ---
    cumulative_context = f"**Decision Simulation Start**\nDilemma: {dilemma}\nKey Issues: {', '.join(key_issues)}\nProcess Overview: {' -> '.join(process_steps)}\nStakeholders: {', '.join([p['name'] for p in personas])}\n---\n"
    transcript.append({"agent": "System", "round": 0, "step": "Setup", "message": cumulative_context})

    # --- Initialize Agent States ---
    # initialize_agent_states(personas) # Optional: Initialize if using state updates

    # --- Simulation Loop ---
    st.info(f"Starting {simulation_type} for {rounds} rounds (max {max_simulation_time}s)...")
    progress_bar = st.progress(0)

    for round_num in range(rounds):
        current_step = process_steps[round_num]
        objective = objectives.get(round_num, "Continue the discussion.")

        # --- Timeout Check ---
        elapsed_time = time.time() - start_time
        if elapsed_time > max_simulation_time:
            st.warning(f"Simulation stopped after round {round_num}: Exceeded maximum time.", icon="⏳")
            transcript.append({
                "agent": "System", "round": round_num + 1, "step": current_step,
                "message": f"Simulation halted due to timeout ({max_simulation_time}s)."
            })
            break

        round_transcript_entries = []
        round_context_addition = ""

        # --- Select Simulation Method ---
        try:
            if simulation_type == f"{MODEL_NAME} Simulation":
                round_transcript_entries, round_context_addition = _run_llm_simulation_round(
                    round_num, personas, current_step, objective, cumulative_context, start_time, max_simulation_time
                )
            elif simulation_type == "Simplified Monte Carlo Simulation":
                round_transcript_entries, round_context_addition = _run_monte_carlo_round(
                    round_num, personas, current_step, objective, cumulative_context
                )
            elif simulation_type == "Simplified Game Theory Simulation":
                 round_transcript_entries, round_context_addition = _run_game_theory_round(
                    round_num, personas, current_step, objective, cumulative_context
                 )
            else:
                st.error(f"Unknown simulation type: {simulation_type}. Aborting.", icon="🚨")
                transcript.append({"agent": "System", "round": round_num + 1, "step": "Error", "message": f"Unknown simulation type '{simulation_type}' selected."})
                break

            # Add round results to main transcript and context
            transcript.extend(round_transcript_entries)
            cumulative_context += round_context_addition

            # Update progress bar
            progress_bar.progress((round_num + 1) / rounds)

             # Check if timeout happened during the round function itself
            if any(entry.get("agent") == "System" and "Timeout" in entry.get("message", "") for entry in round_transcript_entries):
                 break # Exit outer loop if timeout occurred within the round


        except Exception as e:
             st.error(f"Error during simulation round {round_num + 1} ({simulation_type}): {e}", icon="🚨")
             transcript.append({"agent": "System", "round": round_num + 1, "step": current_step, "message": f"Critical error in round: {e}"})
             # Decide whether to continue or break on critical errors
             break


    # --- Simulation End ---
    elapsed = time.time() - start_time
    final_message = f"Simulation completed {len(transcript)-1} turns in {elapsed:.2f} seconds."
    if elapsed > max_simulation_time:
         final_message = f"Simulation stopped after {elapsed:.2f} seconds (Timeout: {max_simulation_time}s)."

    transcript.append({"agent": "System", "round": rounds + 1, "step": "End", "message": final_message})
    st.success(final_message, icon="✅")

    return transcript
