import os
import random
from openai import OpenAI, OpenAIError
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import streamlit as st # Use streamlit for showing errors/warnings

# Import config variables
from config import API_KEY, API_BASE_URL, MODEL_NAME, STAKEHOLDER_ANALYSIS

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def _make_persona_api_call(prompt: str) -> Optional[Dict]:
    """Makes the API call to the LLM for persona details with retries."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI expert in organizational behavior and psychology. Your task is to generate detailed profiles for decision-making stakeholders based on provided context. Respond ONLY with a valid JSON object containing the specified fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Higher temperature for more creative/varied persona details
            max_tokens=800, # More tokens for detailed bio, goals, behavior
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = completion.choices[0].message.content
        # print(f"LLM Raw Persona Response:\n{response_content[:500]}...") # Debugging
        return json.loads(response_content)

    except json.JSONDecodeError as e:
        st.warning(f"LLM response for persona was not valid JSON: {e}. Raw response: {response_content[:500]}...", icon="⚠️")
        # print(f"JSONDecodeError: {e}. Response: {response_content}") # Debugging
        return None # Indicate failure
    except OpenAIError as e:
        st.error(f"API Error during persona generation: {e}", icon="🚨")
        # print(f"OpenAIError: {e}") # Debugging
        raise # Reraise to trigger tenacity retry
    except Exception as e:
        st.error(f"Unexpected error during persona API call: {e}", icon="🚨")
        # print(f"Unexpected API Error: {e}") # Debugging
        raise # Reraise unexpected errors


def generate_personas(extracted_structure: Dict) -> List[dict]:
    """
    Generates detailed personas for stakeholders identified in the extracted structure.

    Args:
        extracted_structure (Dict): The output from the extractor module, containing
                                    at least 'stakeholders', 'decision_type', 'key_issues'.

    Returns:
        List[dict]: List of generated personas, including details like goals, tone,
                    enhanced bio, and expected behavior. Returns an empty list on failure.
    """
    initial_stakeholders = extracted_structure.get("stakeholders", [])
    if not initial_stakeholders:
        st.error("No initial stakeholders provided to generate personas.", icon="🚨")
        return []

    decision_type = extracted_structure.get("decision_type", "Unknown")
    key_issues = extracted_structure.get("key_issues", ["Not specified"])

    generated_personas = []
    persona_count = len(initial_stakeholders)

    st.info(f"Generating detailed personas for {persona_count} stakeholders...")
    progress_bar = st.progress(0)

    for i, stakeholder in enumerate(initial_stakeholders):
        name = stakeholder.get("name", f"Stakeholder {i+1}")
        role = stakeholder.get("role", "Participant")
        initial_bio = stakeholder.get("bio", f"{name}, the {role}, has relevant experience.")
        # Include traits/influences/biases from extraction as context
        context_traits = stakeholder.get('psychological_traits', ['Not specified'])
        context_influences = stakeholder.get('influences', ['Not specified'])
        context_biases = stakeholder.get('biases', ['Not specified'])
        context_hist_behavior = stakeholder.get('historical_behavior', 'Not specified')


        # Construct the prompt for the LLM
        prompt = (
            f"Generate a detailed profile for the stakeholder: **{name} ({role})**.\n\n"
            f"**Decision Context:**\n"
            f"- Type: {decision_type}\n"
            f"- Key Issues: {', '.join(key_issues)}\n"
            f"- Initial Profile Hints:\n"
            f"  - Traits: {', '.join(context_traits)}\n"
            f"  - Influences: {', '.join(context_influences)}\n"
            f"  - Biases: {', '.join(context_biases)}\n"
            f"  - Typical Behavior: {context_hist_behavior}\n"
            f"  - Initial Bio: {initial_bio}\n\n"
            f"**Required JSON Output Structure:**\n"
            f"{{\n"
            f'  "name": "{name}",\n' # Keep original name
            f'  "role": "{role}",\n' # Keep original role
            f'  "goals": [List of 2-3 specific, action-oriented goals relevant to the decision context],\n'
            f'  "tone": (String) Dominant communication tone (e.g., {", ".join(STAKEHOLDER_ANALYSIS["tones"])}),\n'
            f'  "detailed_bio": (String) Expand the initial bio into 100-150 words, incorporating context and inferred background,\n'
            f'  "expected_behavior": (String) Describe likely negotiation style, arguments, and stance in this specific debate (100-150 words),\n'
            f'  "key_motivators": [List of 1-2 primary driving factors (e.g., "Career Advancement", "Department Success", "Risk Mitigation")]\n'
            f"}}\n\n"
            f"Generate only the JSON object as your response."
        )

        try:
            persona_details = _make_persona_api_call(prompt)

            if persona_details and isinstance(persona_details, dict):
                 # Combine initial data with LLM generated details
                full_persona = {
                    # From initial extraction
                    "name": name,
                    "role": role,
                    "psychological_traits": context_traits,
                    "influences": context_influences,
                    "biases": context_biases,
                    "historical_behavior": context_hist_behavior,
                    # From LLM generation (with defaults)
                    "goals": persona_details.get("goals", [f"Achieve successful outcome for {role}"]),
                    "tone": persona_details.get("tone", random.choice(STAKEHOLDER_ANALYSIS["tones"])),
                    "bio": persona_details.get("detailed_bio", initial_bio).strip(), # Use detailed bio
                    "expected_behavior": persona_details.get("expected_behavior", f"{name} will participate constructively based on their role.").strip(),
                    "motivators": persona_details.get("key_motivators", ["Achieve team goals"]) # Added field
                }
                # Basic validation on generated fields
                if not isinstance(full_persona["goals"], list): full_persona["goals"] = ["Goal generation failed"]
                if not isinstance(full_persona["tone"], str): full_persona["tone"] = "neutral"
                if not isinstance(full_persona["motivators"], list): full_persona["motivators"] = ["Motivator generation failed"]

                generated_personas.append(full_persona)
            else:
                # Fallback to using initial data if LLM fails
                st.warning(f"Failed to generate detailed profile for {name}. Using basic info.", icon="⚠️")
                generated_personas.append({
                    "name": name,
                    "role": role,
                    "psychological_traits": context_traits,
                    "influences": context_influences,
                    "biases": context_biases,
                    "historical_behavior": context_hist_behavior,
                    "goals": [f"Achieve successful outcome for {role} (Default)"],
                    "tone": random.choice(STAKEHOLDER_ANALYSIS["tones"]),
                    "bio": initial_bio,
                    "expected_behavior": f"{name} will participate based on their role. (Default)",
                    "motivators": ["Achieve team goals (Default)"]
                })

        except RetryError:
            st.error(f"API call failed for persona {name} after multiple retries. Using basic info.", icon="🚨")
            # Append basic info as fallback
            generated_personas.append({
                    "name": name, "role": role, "bio": initial_bio,
                     "psychological_traits": context_traits, "influences": context_influences,
                     "biases": context_biases, "historical_behavior": context_hist_behavior,
                     "goals": ["Default Goal"], "tone": "neutral",
                     "expected_behavior": "Default Behavior", "motivators": ["Default Motivator"]
            })
        except Exception as e:
             st.error(f"Unexpected error generating persona for {name}: {e}. Using basic info.", icon="🚨")
             # Append basic info as fallback
             generated_personas.append({
                    "name": name, "role": role, "bio": initial_bio,
                     "psychological_traits": context_traits, "influences": context_influences,
                     "biases": context_biases, "historical_behavior": context_hist_behavior,
                     "goals": ["Default Goal"], "tone": "neutral",
                     "expected_behavior": "Default Behavior", "motivators": ["Default Motivator"]
             })

        # Update progress bar
        progress_bar.progress((i + 1) / persona_count)

    if len(generated_personas) == persona_count:
        st.success(f"Successfully generated {persona_count} detailed stakeholder personas.", icon="✅")
    else:
         st.warning(f"Generated {len(generated_personas)} out of {persona_count} personas. Some generations may have failed.", icon="⚠️")

    return generated_personas
