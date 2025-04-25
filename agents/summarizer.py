import json
import os
from openai import OpenAI, OpenAIError
from typing import List, Dict, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import streamlit as st # Use streamlit for showing errors/warnings

# Import config variables
from config import API_KEY, API_BASE_URL, MODEL_NAME, API_TIMEOUT_SECONDS

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def _make_summary_api_call(prompt: str) -> Optional[Dict]:
    """Makes the API call to the LLM for summarization with retries."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing and summarizing decision-making discussions. Provide a concise yet comprehensive analysis based on the transcript. Respond ONLY with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, # Lower temperature for factual summary
            max_tokens=1200, # Allow more tokens for detailed analysis fields
            timeout=API_TIMEOUT_SECONDS * 2, # Allow more time for summarization
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = completion.choices[0].message.content
        # print(f"LLM Raw Summary Response:\n{response_content[:500]}...") # Debugging
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        st.warning(f"LLM summary response was not valid JSON: {e}. Raw response: {response_content[:500]}...", icon="⚠️")
        # print(f"JSONDecodeError: {e}. Response: {response_content}") # Debugging
        return None # Indicate failure
    except OpenAIError as e:
        st.error(f"API Error during summarization: {e}", icon="🚨")
        # print(f"OpenAIError: {e}") # Debugging
        raise # Reraise to trigger tenacity retry
    except Exception as e:
        st.error(f"Unexpected error during summary API call: {e}", icon="🚨")
        # print(f"Unexpected API Error: {e}") # Debugging
        raise # Reraise unexpected errors


def generate_summary_and_suggestion(transcript: List[Dict], dilemma: str, personas: List[Dict]) -> Tuple[str, str]:
    """
    Summarizes the debate transcript and provides optimization suggestions using an LLM.

    Args:
        transcript (List[Dict]): Debate transcript.
        dilemma (str): The original decision dilemma.
        personas (List[Dict]): List of stakeholder personas involved.

    Returns:
        Tuple[str, str]: Summary string and optimization suggestion string.
                         Returns default messages on failure.
    """
    if not transcript:
        return ("No transcript available to summarize.", "No suggestions possible without a transcript.")

    # Prepare input for LLM
    # Limit transcript size to avoid excessive tokens
    transcript_limit = 4000 # Characters
    transcript_str = json.dumps(transcript, indent=None) # Compact JSON
    if len(transcript_str) > transcript_limit:
        # Simple truncation - smarter summarization/chunking could be used for very long transcripts
        transcript_str = transcript_str[:transcript_limit] + "... (transcript truncated)"
        st.warning(f"Transcript too long, truncated to {transcript_limit} characters for summarization.", icon="⚠️")


    persona_summary = "\n".join([f"- {p['name']} ({p['role']}): Goals={p.get('goals',[])}, Biases={p.get('biases',[])}" for p in personas])

    # Construct prompt
    prompt = (
        f"**Analyze the following decision-making simulation transcript.**\n\n"
        f"**Original Dilemma:**\n{dilemma}\n\n"
        f"**Stakeholders Involved:**\n{persona_summary}\n\n"
        f"**Debate Transcript (JSON format, possibly truncated):**\n```json\n{transcript_str}\n```\n\n"
        f"**Analysis Task:**\nProvide your analysis as a JSON object with the following keys:\n"
        f"1. `executive_summary`: (String) A concise summary (150-200 words) covering key discussion points, arguments, shifts in stance, and overall outcome or status.\n"
        f"2. `key_agreements`: (List of Strings) Identify any explicit or strongly implied points of agreement reached.\n"
        f"3. `major_conflicts`: (List of Strings) Identify the main points of contention or disagreement between stakeholders.\n"
        f"4. `process_observations`: (List of Strings) Note any observations about the decision-making process itself (e.g., bottlenecks, effective collaboration moments, ignored perspectives).\n"
        f"5. `optimization_suggestions`: (List of Strings) Provide 2-4 actionable recommendations to improve the decision-making process or outcome based on the simulation (e.g., specific stakeholder engagement, data needs, process adjustments).\n\n"
        f"Respond ONLY with the valid JSON object."
    )

    try:
        with st.spinner("Generating summary and optimization suggestions..."):
            result = _make_summary_api_call(prompt)

        if result and isinstance(result, dict):
            summary = result.get("executive_summary", "Summary could not be generated.")
            # Combine other fields into the suggestion text
            agreements = result.get("key_agreements", ["N/A"])
            conflicts = result.get("major_conflicts", ["N/A"])
            observations = result.get("process_observations", ["N/A"])
            suggestions = result.get("optimization_suggestions", ["No specific suggestions generated."])

            suggestion_text = "**Analysis & Recommendations:**\n\n"
            suggestion_text += "**Key Agreements:**\n" + "\n".join([f"- {a}" for a in agreements]) + "\n\n"
            suggestion_text += "**Major Conflicts:**\n" + "\n".join([f"- {c}" for c in conflicts]) + "\n\n"
            suggestion_text += "**Process Observations:**\n" + "\n".join([f"- {o}" for o in observations]) + "\n\n"
            suggestion_text += "**Recommendations:**\n" + "\n".join([f"- {s}" for s in suggestions])

            return summary.strip(), suggestion_text.strip()
        else:
             st.error("Failed to generate summary and suggestions from LLM response.", icon="🚨")
             return ("Summary generation failed.", "Suggestion generation failed.")

    except RetryError:
        st.error("Summary generation API call failed after multiple retries.", icon="🚨")
        return ("Summary generation failed due to API errors.", "Suggestion generation failed due to API errors.")
    except Exception as e:
        st.error(f"An unexpected error occurred during summarization: {e}", icon="🚨")
        # print(f"Summarization Main Error: {e}") # Debugging
        return ("Summary generation failed due to an unexpected error.", "Suggestion generation failed due to an unexpected error.")
