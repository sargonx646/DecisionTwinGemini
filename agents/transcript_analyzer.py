import json
from typing import Dict, List, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import streamlit as st # For potential warnings

# --- NLTK Setup ---
# Ensure necessary NLTK data is available
NLTK_DATA_LOADED = False
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    NLTK_DATA_LOADED = True
except LookupError:
    st.info("Downloading necessary NLTK data (vader_lexicon, punkt, stopwords)...")
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        NLTK_DATA_LOADED = True
        st.info("NLTK data downloaded.")
    except Exception as download_e:
        st.error(f"Failed to download NLTK data: {download_e}. Sentiment analysis and keyword extraction might fail.", icon="🚨")
except Exception as e:
    st.error(f"An unexpected error occurred during NLTK setup: {e}", icon="🚨")


# --- Analysis Functions ---

def _get_sentiment(text: str, sid: SentimentIntensityAnalyzer) -> Dict[str, Any]:
    """Analyzes sentiment of a given text."""
    if not NLTK_DATA_LOADED or not text:
        return {"score": 0.0, "label": "neutral"}
    try:
        scores = sid.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"score": compound, "label": label}
    except Exception as e:
        # print(f"Sentiment analysis error: {e}") # Debugging
        st.warning(f"Sentiment analysis failed for text snippet: {e}", icon="⚠️")
        return {"score": 0.0, "label": "error"}


def _extract_keywords(text: str, stop_words: set, top_n: int = 10) -> List[Tuple[str, int]]:
    """Extracts top N keywords from text, excluding stop words."""
    if not NLTK_DATA_LOADED or not text:
        return []
    try:
        words = [
            word.lower()
            for word in word_tokenize(text)
            if word.lower() not in stop_words and word.isalnum() and len(word) > 2
        ]
        word_counts = Counter(words)
        return word_counts.most_common(top_n)
    except Exception as e:
        # print(f"Keyword extraction error: {e}") # Debugging
        st.warning(f"Keyword extraction failed: {e}", icon="⚠️")
        return []


def _find_arguments(text: str) -> List[Dict[str, str]]:
    """Identifies potential arguments (proposals, agreements, disagreements) using keywords."""
    arguments = []
    if not text: return arguments
    try:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            lower_sentence = sentence.lower()
            arg_type = None
            if re.search(r'\b(propose|suggest|recommend|offer|think we should|idea is)\b', lower_sentence):
                arg_type = "Proposal"
            elif re.search(r'\b(agree|support|concur|accept|positive|good point)\b', lower_sentence):
                arg_type = "Agreement"
            elif re.search(r'\b(disagree|oppose|challenge|concern|but|however|issue is|problem is|object)\b', lower_sentence):
                arg_type = "Disagreement/Concern"
            # Add more patterns as needed (e.g., questions, justifications)

            if arg_type:
                arguments.append({
                    "type": arg_type,
                    "content": sentence # Keep original casing
                })
    except Exception as e:
         st.warning(f"Argument mining failed: {e}", icon="⚠️")
         # print(f"Argument mining error: {e}") # Debugging
    return arguments


# --- Main Analyzer Function ---

def transcript_analyzer(input_data: str) -> str:
    """
    Analyzes the debate transcript for topics, sentiment, arguments, and other insights.

    Args:
        input_data (str): JSON string containing 'transcript', 'dilemma', 'personas'.

    Returns:
        str: JSON string with analysis results. Contains an 'error' key on failure.
    """
    if not NLTK_DATA_LOADED:
         return json.dumps({"error": "NLTK data required for analysis is not loaded."})

    try:
        data = json.loads(input_data)
        transcript = data.get("transcript", [])
        # dilemma = data.get("dilemma", "") # Use if needed for context
        # personas = data.get("personas", []) # Use if needed for context

        if not transcript:
            return json.dumps({"warning": "Transcript is empty, no analysis performed."})

        # Initialize tools
        try:
             sid = SentimentIntensityAnalyzer()
             stop_words = set(stopwords.words('english'))
             # Add custom stop words if needed
             stop_words.update(['-', '–', '...', "'s", "n't", 'like', 'would', 'could', 'also'])
        except Exception as tool_init_e:
             return json.dumps({"error": f"Failed to initialize NLTK tools: {tool_init_e}"})


        # --- Perform Analysis ---
        full_transcript_text = " ".join([entry.get('message', '') for entry in transcript])
        analysis_results = {
            "topics": [],
            "sentiment_analysis": [],
            "key_arguments": [],
            "agent_contributions": {}, # Track message count per agent
            "conflicts": [], # Track disagreements between agents
            "overall_sentiment": {},
            "analysis_summary": {} # Placeholder for overall insights
        }

        # 1. Keyword/Topic Extraction (Top 10 overall keywords)
        top_keywords = _extract_keywords(full_transcript_text, stop_words, top_n=10)
        total_words = len(full_transcript_text.split()) # Approximate total words
        analysis_results["topics"] = [
            {"label": word, "keywords": [word], "weight": count / total_words if total_words else 0}
            for word, count in top_keywords
        ]


        # 2. Sentiment & Argument Analysis per Turn, Contributions
        agent_sentiments = {entry['agent']: [] for entry in transcript if 'agent' in entry}
        conflict_pairs = []

        for i, entry in enumerate(transcript):
            agent = entry.get('agent', 'Unknown')
            message = entry.get('message', '')
            round_num = entry.get('round', 0)
            step = entry.get('step', 'N/A')

            if agent == "System": continue # Skip system messages for analysis

            # Contribution count
            analysis_results["agent_contributions"][agent] = analysis_results["agent_contributions"].get(agent, 0) + 1

            # Sentiment
            sentiment = _get_sentiment(message, sid)
            analysis_results["sentiment_analysis"].append({
                "agent": agent,
                "round": round_num,
                "step": step,
                "score": sentiment["score"],
                "label": sentiment["label"] # Use label (positive/negative/neutral)
            })
            if agent != 'Unknown': agent_sentiments.setdefault(agent, []).append(sentiment["score"])


            # Arguments & Conflicts
            arguments = _find_arguments(message)
            if arguments:
                 for arg in arguments:
                     analysis_results["key_arguments"].append({
                         "agent": agent,
                         "round": round_num,
                         "step": step,
                         "type": arg["type"],
                         "content": arg["content"][:200] + ('...' if len(arg["content"]) > 200 else '') # Limit length
                     })
                     # Identify conflicts (simple: disagreement follows another agent's turn)
                     if arg["type"] == "Disagreement/Concern" and i > 0:
                          prev_entry = transcript[i-1]
                          prev_agent = prev_entry.get('agent', 'Unknown')
                          if prev_agent != agent and prev_agent != "System":
                               conflict = {
                                   "round": round_num,
                                   "step": step,
                                   "issue": f"Disagreement from {agent} following {prev_agent}",
                                   "stakeholders": sorted(list(set([agent, prev_agent]))) # Unique pair
                               }
                               # Avoid adding duplicate conflict pairs for the same round/step
                               if conflict not in analysis_results["conflicts"]:
                                    analysis_results["conflicts"].append(conflict)


        # 3. Overall Sentiment Calculation
        overall_score = _get_sentiment(full_transcript_text, sid)["score"]
        analysis_results["overall_sentiment"] = {
             "score": overall_score,
             "label": "positive" if overall_score >= 0.05 else "negative" if overall_score <= -0.05 else "neutral"
        }
        analysis_results["agent_average_sentiment"] = {
             agent: sum(scores)/len(scores) if scores else 0
             for agent, scores in agent_sentiments.items()
        }

        # 4. Generate Analysis Summary / Insights
        num_conflicts = len(analysis_results["conflicts"])
        num_agents = len(analysis_results["agent_contributions"])
        insights = []
        insights.append(f"The debate involved {num_agents} active stakeholders across {transcript[-1].get('round', 'multiple')} rounds.")
        insights.append(f"Overall sentiment was {analysis_results['overall_sentiment']['label']} (Score: {analysis_results['overall_sentiment']['score']:.2f}).")
        insights.append(f"Key topics centered around: {', '.join([t['label'] for t in analysis_results['topics'][:3]])}...")
        insights.append(f"{len(analysis_results['key_arguments'])} potential arguments identified ({sum(1 for a in analysis_results['key_arguments'] if a['type']=='Proposal')} proposals, {sum(1 for a in analysis_results['key_arguments'] if a['type']=='Agreement')} agreements, {sum(1 for a in analysis_results['key_arguments'] if a['type']=='Disagreement/Concern')} disagreements/concerns).")
        insights.append(f"{num_conflicts} potential conflict points identified between stakeholders.")
        # Add contribution balance insight
        contributions = analysis_results["agent_contributions"].values()
        if contributions:
             if max(contributions) > 2 * (sum(contributions) / len(contributions)):
                  insights.append("Participation levels appear somewhat unbalanced.")
             else:
                  insights.append("Participation levels seem relatively balanced.")


        analysis_results["analysis_summary"] = {
             "insights": insights,
             "improvement_areas": [ # Basic suggestions based on analysis
                  "Review conflict points for potential resolution strategies." if num_conflicts > 3 else "Monitor potential conflict points.",
                  "Ensure all stakeholder perspectives are adequately considered, especially if participation is unbalanced." if "unbalanced" in insights[-1] else "Maintain balanced participation.",
                  "Focus discussion on key topics if analysis shows divergence." if len(analysis_results["topics"]) > 5 else "Discussion appears focused on key topics."
             ]
        }


        # Remove raw arguments list if too long? Or keep it? Keep for now.
        # Optional: Clean up results (e.g., limit list lengths) before returning

        return json.dumps(analysis_results, default=str) # Use default=str for potential non-serializable items

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Input data is not valid JSON: {e}"})
    except Exception as e:
        st.error(f"Transcript analysis failed: {e}", icon="🚨")
        # print(f"Transcript Analyzer Error: {e}\n{traceback.format_exc()}") # More detailed debug
        return json.dumps({"error": f"Analysis failed due to an unexpected error: {e}"})
