import streamlit as st
import json
import os
import random
import PyPDF2
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import pandas as pd
import bcrypt # For password hashing

# --- Configuration and Initialization ---
# Set page config first
st.set_page_config(page_title="DecisionForge", layout="wide", initial_sidebar_state="expanded")

# Load environment variables and config
from dotenv import load_dotenv
load_dotenv()
import config # Import config after dotenv load

# Import Agents and Utils
from agents.extractor import extract_decision_structure
from agents.persona_builder import generate_personas
from agents.debater import simulate_debate
from agents.summarizer import generate_summary_and_suggestion
from agents.transcript_analyzer import transcript_analyzer
from utils.visualizer import generate_visualizations, plot_interaction_network_agraph # Import the specific agraph plotter
from utils.db import init_db, save_persona, get_all_personas, get_persona_by_name, update_persona, delete_persona

# Import agraph for network viz
from streamlit_agraph import agraph, Node, Edge, Config

# Initialize Database
try:
    init_db()
except Exception as db_init_e:
    st.error(f"Fatal Error: Could not initialize database: {db_init_e}. Please check permissions and configuration.", icon="🚨")
    st.stop()

# Ensure personas directory exists (handled in db.py)
os.makedirs(config.PERSONAS_DIR, exist_ok=True)

# Check for API key
if not config.API_KEY:
    st.error("XAI_API_KEY environment variable is not set. Please configure it in your .env file.", icon="🚨")
    st.stop()

# --- Password Handling ---
def verify_password(stored_plain_password: Optional[str], provided_password: str) -> bool:
    """Verifies the provided password against the stored plain text password."""
    # In a real app, use bcrypt or another hashing library:
    # stored_hash = os.getenv("APP_PASSWORD_HASH") # Store hash in .env
    # if not stored_hash: return False # No password set
    # return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

    # Simple comparison for now using plain text from config/env
    if not stored_plain_password:
        st.warning("App password is not configured in the environment (.env file). Access is currently open.", icon="⚠️")
        return True # Allow access if no password is set in env
    return stored_plain_password == provided_password

# --- Hardcoded Personas (Examples/Fallback) ---
# Keep these as examples, but primary source should be DB/generated
HARDCODED_PERSONAS = [
    # Keep your list of hardcoded personas here...
     {
        "name": "John F. Kennedy",
        "role": "Former U.S. President",
        "bio": "John Fitzgerald Kennedy (1917–1963) was the 35th President...",
        "psychological_traits": ["charismatic", "decisive", "optimistic", "pragmatic"],
        "influences": ["public opinion", "international allies", "military advisors", "media"],
        "biases": ["optimism bias", "groupthink", "confirmation bias"],
        "historical_behavior": "Consensus-driven, proactive in crises, long-term strategist",
        "tone": "inspirational",
        "goals": ["promote global peace", "advance technology", "strengthen national unity"],
        "expected_behavior": "JFK negotiates with an inspirational and diplomatic tone...",
        "motivators": ["Legacy", "National Prestige"]
    },
    {
        "name": "Abraham Lincoln",
        "role": "Former U.S. President",
        "bio": "Abraham Lincoln (1809–1865) was the 16th President...",
        "psychological_traits": ["empathetic", "analytical", "resilient", "collaborative"],
        "influences": ["abolitionists", "military leaders", "public sentiment", "economic advisors"],
        "biases": ["status quo bias", "anchoring bias"],
        "historical_behavior": "Data-driven, consensus-driven, long-term strategist",
        "tone": "empathetic",
        "goals": ["preserve union", "advance equality", "stabilize economy"],
        "expected_behavior": "Lincoln negotiates with empathy and persuasion...",
         "motivators": ["Unity", "Justice"]
    },
    # Add your other hardcoded personas...
]

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "authenticated": False,
        "current_step": 0, # 0: Auth, 1: Define, 2: Review, 3: Simulate, 4: Analyze, 5: Results
        "decision_dilemma": "",
        "process_hints": "",
        "uploaded_context": "",
        "extracted_structure": None, # Stores output from extractor
        "personas": [], # List of persona dicts for the current simulation
        "simulation_transcript": [], # Stores output from debater
        "analysis_results": None, # Stores output from transcript_analyzer
        "visualization_objects": None, # Stores output from visualizer
        "error_message": None,
        "show_persona_library": False,
        "persona_replace_index": None, # Index of persona card being replaced
        "last_simulation_type": config.SIMULATION_METHODS[0], # Default simulation type
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Helper Functions ---
def read_pdf(file) -> str:
    """Extract text from uploaded PDF."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        st.success(f"Successfully extracted text from PDF '{file.name}'.", icon="📄")
        return text
    except Exception as e:
        st.error(f"Error reading PDF '{file.name}': {str(e)}", icon="🚨")
        return ""

def generate_mock_dilemma():
    """Generate a mock decision dilemma."""
    scenarios = [
        {
            "dilemma": "Allocate $5M marketing budget between launching Product A in a new region vs. scaling digital ads for existing Product B.",
            "hints": "Timeline: 6 months. Process: Market analysis (1mo), Strategy options (1mo), Budget committee review (2wks), Exec decision (1wk). Stakeholders: CMO (Lead), Product Mgr A, Product Mgr B, Finance Director, Sales VP."
        },
        {
            "dilemma": "Respond to a competitor's aggressive price cut for a similar service. Options: Match price, Improve service quality, Launch loyalty program, Do nothing.",
            "hints": "Process: Competitive analysis (2wks), Financial modeling (1wk), Customer survey (3wks), Management deliberation (1wk). Stakeholders: CEO, CFO, Head of Sales, Head of Marketing, Head of Customer Success."
        },
         {
            "dilemma": "Should the city invest in upgrading its aging public transit infrastructure (buses, stations) or expanding service to underserved areas?",
            "hints": "Budget: $20M available. Process: Needs assessment (2mo), Public consultation (1mo), Engineering feasibility (2mo), Council vote (1mo). Stakeholders: Mayor, Transit Authority Director, City Planner, Budget Committee Chair, Community Activist Group Lead."
        }
    ]
    choice = random.choice(scenarios)
    st.session_state.decision_dilemma = choice["dilemma"]
    st.session_state.process_hints = choice["hints"]
    st.info("Mock dilemma loaded. You can edit it below.")


def reset_simulation():
    """Resets session state for a new simulation."""
    # Keep authentication status
    auth_status = st.session_state.authenticated
    # Reset relevant keys
    keys_to_reset = [
        "current_step", "decision_dilemma", "process_hints", "uploaded_context",
        "extracted_structure", "personas", "simulation_transcript",
        "analysis_results", "visualization_objects", "error_message",
        "show_persona_library", "persona_replace_index"
    ]
    init_session_state() # Re-initialize all defaults
    st.session_state.authenticated = auth_status # Restore auth status
    st.session_state.current_step = 1 # Go back to step 1
    st.success("New simulation started.", icon="🔄")


# --- UI Styling ---
st.markdown("""
<style>
    /* Basic Styling */
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 5px; border: none; padding: 8px 15px; }
    .stButton>button:hover { background-color: #0056b3; color: white; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea { border-radius: 5px; border: 1px solid #ced4da; }
    .stProgress > div > div > div > div { background-color: #007bff; }

    /* Custom Component Styles */
    .main-title {
        font-size: 2.8em;
        color: #343a40; /* Darker title */
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
        letter-spacing: -1px;
    }
    .step-header {
        font-size: 1.8em;
        color: #007bff; /* Blue header */
        border-bottom: 2px solid #007bff;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .section-subheader {
        font-size: 1.3em;
        color: #495057; /* Gray subheader */
        margin-top: 15px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .info-box { background-color: #e7f1ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #004085; }
    .persona-card { background-color: #ffffff; border: 1px solid #e0e0e0; padding: 20px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .summary-box { background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #155724; }
    .suggestion-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; margin-bottom: 20px; color: #856404; }
    .analysis-section { background-color: #f0f2f6; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .transcript-entry { border-bottom: 1px dashed #ccc; padding-bottom: 10px; margin-bottom: 10px; }
    .transcript-agent { font-weight: bold; color: #0056b3; }
    .transcript-meta { font-size: 0.9em; color: #6c757d; }

    /* Fix button alignment in forms */
    .stForm [data-testid="stFormSubmitButton"] button { width: 100%; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #e9ecef; } /* Light gray sidebar */
    [data-testid="stSidebar"] h2 { color: #343a40; text-align: center; }
    [data-testid="stSidebar"] .stButton>button { background-color: #6c757d; } /* Gray buttons in sidebar */
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #5a6268; }

</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://github.com/sargonx646/DF_22AprilLate/raw/main/assets/decisionforge_logo.png.png", use_column_width=True)
    st.markdown("<h2>DecisionForge</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Simulation Progress**")
    # Adjust steps based on actual flow (e.g., 5 steps total excluding auth)
    total_steps = 5
    current_step_display = st.session_state.current_step if st.session_state.current_step > 0 else 0
    progress = (current_step_display -1) / total_steps if current_step_display > 0 else 0
    st.progress(progress)
    st.markdown(f"**Step {current_step_display} of {total_steps}**" if current_step_display > 0 else "**Authentication**")
    st.markdown("---")

    # Navigation buttons
    if st.session_state.current_step > 1: # Show Back button after Step 1
        if st.button("⬅️ Back", key="sidebar_back", use_container_width=True):
            st.session_state.current_step -= 1
            st.rerun()
    # Forward button logic depends on step completion / state validity
    # Disable forward if necessary inputs/actions for the current step are missing
    can_go_forward = False
    if st.session_state.current_step == 1 and st.session_state.extracted_structure:
         can_go_forward = True
    elif st.session_state.current_step == 2 and st.session_state.personas:
         can_go_forward = True
    elif st.session_state.current_step == 3 and st.session_state.simulation_transcript:
         can_go_forward = True
    elif st.session_state.current_step == 4 and st.session_state.analysis_results:
         can_go_forward = True

    # Only show Forward if not on the last step
    if st.session_state.current_step > 0 and st.session_state.current_step < total_steps + 1 :
         if st.button("Forward ➡️", key="sidebar_forward", use_container_width=True, disabled=not can_go_forward):
             st.session_state.current_step += 1
             st.rerun()
         elif not can_go_forward:
             # Provide hint if forward is disabled
             hint = ""
             if st.session_state.current_step == 1: hint = "Extract structure first."
             elif st.session_state.current_step == 2: hint = "Generate/Confirm personas."
             elif st.session_state.current_step == 3: hint = "Run the simulation."
             elif st.session_state.current_step == 4: hint = "Analyze the results."
             if hint: st.caption(f"*(Complete current step to proceed: {hint})*")


    st.markdown("---")
    if st.session_state.current_step > 0: # Show after authentication
        if st.button("🔄 Start New Simulation", key="sidebar_reset", use_container_width=True):
            reset_simulation()
            st.rerun()

# --- Main App Logic ---
def run_app():
    st.markdown("<h1 class='main-title'>DecisionForge Simulation</h1>", unsafe_allow_html=True)

    # Display global errors if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message, icon="🚨")
        st.session_state.error_message = None # Clear after showing

    # --- Step 0: Authentication ---
    if not st.session_state.authenticated:
        st.markdown("<div class='step-header'>Authentication Required</div>", unsafe_allow_html=True)
        st.warning("Please enter the password configured in your environment variables to proceed.")
        password_input = st.text_input("Password", type="password", key="password_input")
        if st.button("Login", key="login_button"):
            stored_password = config.APP_PASSWORD_HASH # Get password from config (loaded from env)
            if verify_password(stored_password, password_input):
                st.session_state.authenticated = True
                st.session_state.current_step = 1 # Move to first step
                st.success("Authentication successful!", icon="✅")
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.", icon="🔒")
        st.stop() # Stop execution until authenticated

    # --- Step 1: Define Decision ---
    if st.session_state.current_step == 1:
        st.markdown("<div class='step-header'>Step 1: Define Your Decision</div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box">Clearly describe the decision dilemma, the process involved (if known), key stakeholders, and optionally upload a PDF for more context. The AI will use this to structure the simulation.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1]) # Wider column for inputs

        with col1:
            with st.form(key="decision_definition_form"):
                dilemma = st.text_area(
                    "**Decision Dilemma:**",
                    height=150,
                    value=st.session_state.decision_dilemma,
                    placeholder="Example: Allocate a $10M budget across R&D, Marketing, and Sales departments for the next fiscal year.",
                    key="dilemma_input",
                    help="Describe the core problem or choice to be made."
                )
                hints = st.text_area(
                    "**Process & Stakeholder Hints:**",
                    height=100,
                    value=st.session_state.process_hints,
                    placeholder="Example: Process involves dept heads presenting proposals, finance review, CEO final decision. Stakeholders: CEO, CFO, R&D Head, Marketing Head, Sales Head.",
                    key="hints_input",
                    help="Provide details on how the decision is made and who is involved."
                )
                uploaded_file = st.file_uploader(
                    "**Upload Context PDF (Optional):**",
                    type="pdf",
                    key="pdf_upload",
                    help="Upload a PDF document with background information, reports, or data relevant to the dilemma."
                )

                submitted = st.form_submit_button("Analyze Context & Extract Structure", use_container_width=True)

                if submitted:
                    if not dilemma.strip():
                        st.error("Please provide a description of the decision dilemma.", icon="⚠️")
                    else:
                        st.session_state.decision_dilemma = dilemma
                        st.session_state.process_hints = hints
                        pdf_text = ""
                        if uploaded_file:
                            with st.spinner(f"Reading PDF '{uploaded_file.name}'..."):
                                pdf_text = read_pdf(uploaded_file)
                            st.session_state.uploaded_context = pdf_text

                        # Call the extractor agent
                        extracted = extract_decision_structure(
                            dilemma=st.session_state.decision_dilemma,
                            process_hint=st.session_state.process_hints,
                            context_text=st.session_state.uploaded_context
                        )

                        if extracted:
                            st.session_state.extracted_structure = extracted
                            # Clear previously generated personas/results if context changes
                            st.session_state.personas = []
                            st.session_state.simulation_transcript = []
                            st.session_state.analysis_results = None
                            st.session_state.visualization_objects = None
                            st.success("Decision structure extracted successfully!", icon="✅")
                            # Automatically move to next step? Or let user click? Let user click sidebar.
                            # st.session_state.current_step = 2
                            st.rerun() # Rerun to update UI state and enable sidebar forward button
                        else:
                            st.error("Failed to extract decision structure. Please check the input or try refining the description.", icon="🚨")
                            st.session_state.extracted_structure = None # Ensure it's reset on failure
        with col2:
             st.button("Load Mock Dilemma", key="load_mock", on_click=generate_mock_dilemma, use_container_width=True)
             st.markdown("---")
             # Display extracted structure preview if available
             if st.session_state.extracted_structure:
                  st.markdown("**Extracted Structure Preview:**")
                  exp = st.session_state.extracted_structure
                  st.caption(f"**Type:** {exp.get('decision_type', 'N/A')}")
                  st.caption(f"**Stakeholders:** {len(exp.get('stakeholders', []))} identified")
                  st.caption(f"**Issues:** {len(exp.get('key_issues', []))} identified")
                  st.caption(f"**Process Steps:** {len(exp.get('process_steps', []))} identified")
                  with st.expander("View Details"):
                       st.json(exp, expanded=False)


    # --- Step 2: Review Personas ---
    elif st.session_state.current_step == 2:
        st.markdown("<div class='step-header'>Step 2: Review & Refine Personas</div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box">Review the AI-generated stakeholder personas. You can edit their details, save them to your library, or replace them with personas from the library. Ensure the personas accurately reflect the expected participants.</div>', unsafe_allow_html=True)

        if not st.session_state.extracted_structure:
             st.warning("Please define the decision and extract the structure in Step 1 first.", icon="⚠️")
             st.stop()

        st.markdown("<div class='section-subheader'>Decision Context</div>", unsafe_allow_html=True)
        with st.expander("Show Dilemma & Extracted Structure", expanded=False):
            st.markdown(f"**Dilemma:** {st.session_state.decision_dilemma}")
            st.markdown(f"**Process Hints:** {st.session_state.process_hints}")
            if st.session_state.uploaded_context:
                 st.text_area("PDF Context (Preview)", st.session_state.uploaded_context[:1000]+"...", height=100, disabled=True)
            if st.session_state.extracted_structure:
                 st.json(st.session_state.extracted_structure, expanded=False)


        st.markdown("<div class='section-subheader'>Generated Personas</div>", unsafe_allow_html=True)

        if not st.session_state.personas:
            if st.button("Generate Personas from Extracted Structure", key="generate_personas_button", use_container_width=True):
                with st.spinner("Generating detailed stakeholder personas... This may take a moment."):
                    generated = generate_personas(st.session_state.extracted_structure)
                if generated:
                    st.session_state.personas = generated
                    # Attempt to save newly generated personas to DB and local JSON
                    for p in st.session_state.personas:
                        try:
                            db_id = save_persona(p) # Save/update in DB
                            p['id'] = db_id # Store db id if returned
                            # Save to local JSON file in PERSONAS_DIR
                            filename = os.path.join(config.PERSONAS_DIR, f"{p['name'].replace(' ', '_').lower()}.json")
                            with open(filename, "w") as f:
                                json.dump(p, f, indent=2)
                        except Exception as save_err:
                            st.warning(f"Could not save persona {p['name']} to DB/JSON: {save_err}", icon="⚠️")
                    st.success("Personas generated and saved!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to generate personas. Please check the extracted structure or try again.", icon="🚨")
        else:
            # --- Persona Card Display and Editing ---
            num_personas = len(st.session_state.personas)
            cols = st.columns(min(num_personas, 3)) # Max 3 columns

            for i, persona in enumerate(st.session_state.personas):
                 with cols[i % 3]:
                     st.markdown(f'<div class="persona-card">', unsafe_allow_html=True)
                     card_key_prefix = f"persona_{i}_{persona.get('id', i)}" # Unique key prefix

                     st.markdown(f"**{persona.get('name', f'Persona {i+1}')}** ({persona.get('role', 'Unknown Role')})")

                     with st.expander("View/Edit Details", expanded=False):
                          with st.form(key=f"edit_persona_{card_key_prefix}", clear_on_submit=False):
                               # Editable fields
                               name = st.text_input("Name", value=persona.get("name", ""), key=f"{card_key_prefix}_name")
                               role = st.text_input("Role/Title", value=persona.get("role", ""), key=f"{card_key_prefix}_role")
                               bio = st.text_area("Bio", value=persona.get("bio", ""), height=100, key=f"{card_key_prefix}_bio")
                               goals = st.text_area("Goals (comma-separated)", value=", ".join(persona.get("goals", [])), key=f"{card_key_prefix}_goals")
                               traits = st.text_area("Psychological Traits (comma-separated)", value=", ".join(persona.get("psychological_traits", [])), key=f"{card_key_prefix}_traits")
                               influences = st.text_area("Influences (comma-separated)", value=", ".join(persona.get("influences", [])), key=f"{card_key_prefix}_influences")
                               biases = st.text_area("Biases (comma-separated)", value=", ".join(persona.get("biases", [])), key=f"{card_key_prefix}_biases")
                               hist_behavior = st.text_input("Historical Behavior", value=persona.get("historical_behavior", ""), key=f"{card_key_prefix}_hist")
                               tone = st.selectbox("Tone", options=config.STAKEHOLDER_ANALYSIS["tones"], index=config.STAKEHOLDER_ANALYSIS["tones"].index(persona.get("tone", "neutral")) if persona.get("tone", "neutral") in config.STAKEHOLDER_ANALYSIS["tones"] else 0, key=f"{card_key_prefix}_tone")
                               exp_behavior = st.text_area("Expected Negotiation Behavior", value=persona.get("expected_behavior", ""), height=100, key=f"{card_key_prefix}_exp_beh")
                               motivators = st.text_area("Key Motivators (comma-separated)", value=", ".join(persona.get("motivators", [])), key=f"{card_key_prefix}_motivators")

                               if st.form_submit_button("Save Changes", use_container_width=True):
                                   # Update persona in session state
                                   updated_persona = {
                                       "id": persona.get("id"), # Keep existing ID if available
                                       "name": name, "role": role, "bio": bio,
                                       "goals": [g.strip() for g in goals.split(',') if g.strip()],
                                       "psychological_traits": [t.strip() for t in traits.split(',') if t.strip()],
                                       "influences": [inf.strip() for inf in influences.split(',') if inf.strip()],
                                       "biases": [b.strip() for b in biases.split(',') if b.strip()],
                                       "historical_behavior": hist_behavior,
                                       "tone": tone,
                                       "expected_behavior": exp_behavior,
                                       "motivators": [m.strip() for m in motivators.split(',') if m.strip()]
                                   }
                                   st.session_state.personas[i] = updated_persona
                                   # Save updated persona to DB and JSON
                                   try:
                                       db_id = save_persona(updated_persona) # Save/update in DB
                                       updated_persona['id'] = db_id # Update ID if changed/added
                                       st.session_state.personas[i]['id'] = db_id # Update ID in session state too
                                       # Save to local JSON
                                       filename = os.path.join(config.PERSONAS_DIR, f"{name.replace(' ', '_').lower()}.json")
                                       with open(filename, "w") as f:
                                           json.dump(updated_persona, f, indent=2)
                                       st.success(f"Persona '{name}' updated and saved.", icon="✅")
                                   except Exception as save_err:
                                       st.warning(f"Could not save updated persona {name} to DB/JSON: {save_err}", icon="⚠️")
                                   st.rerun()


                     # --- Action Buttons within Card ---
                     b_col1, b_col2 = st.columns(2)
                     with b_col1:
                          # Save to Library Button (saves current state of the card to DB)
                          if st.button("💾 Save to Library", key=f"save_lib_{card_key_prefix}", use_container_width=True, help="Save this persona's current details to the reusable library."):
                               try:
                                   save_persona(persona) # Save the current state
                                   st.success(f"Persona '{persona.get('name')}' saved to library!", icon="📚")
                               except Exception as save_err:
                                   st.error(f"Error saving '{persona.get('name')}' to library: {save_err}", icon="🚨")

                     with b_col2:
                           # Replace Button (initiates replacement process)
                          if st.button("🔄 Replace", key=f"replace_{card_key_prefix}", use_container_width=True, help="Replace this persona with one from the library."):
                               st.session_state.persona_replace_index = i # Store index to replace
                               st.session_state.show_persona_library = True # Show library modal/section
                               st.rerun()

                     st.markdown(f'</div>', unsafe_allow_html=True) # End persona-card div

            # --- Persona Replacement Logic ---
            if st.session_state.show_persona_library and st.session_state.persona_replace_index is not None:
                 idx_to_replace = st.session_state.persona_replace_index
                 st.markdown("---")
                 st.markdown("<div class='section-subheader'>Replace Persona</div>", unsafe_allow_html=True)
                 st.write(f"Select a persona from the library to replace **{st.session_state.personas[idx_to_replace].get('name')}**:")

                 library_personas = get_all_personas()
                 # Combine DB personas with hardcoded ones (ensure no duplicates by name)
                 library_options = {p['name']: p for p in library_personas}
                 for hp in HARDCODED_PERSONAS:
                      if hp['name'] not in library_options:
                           library_options[hp['name']] = hp # Add hardcoded if not in DB

                 if not library_options:
                      st.warning("Persona library is empty.", icon="⚠️")
                 else:
                      persona_names = list(library_options.keys())
                      selected_name = st.selectbox("Select from Library:", persona_names, key="library_select")

                      col_rep1, col_rep2 = st.columns(2)
                      with col_rep1:
                            if st.button("Confirm Replacement", key="confirm_replace", use_container_width=True):
                                 selected_persona_data = library_options[selected_name]
                                 # Ensure the selected persona is also saved/updated in DB/JSON before replacing
                                 try:
                                      save_persona(selected_persona_data)
                                      filename = os.path.join(config.PERSONAS_DIR, f"{selected_name.replace(' ', '_').lower()}.json")
                                      with open(filename, "w") as f: json.dump(selected_persona_data, f, indent=2)
                                 except Exception as save_err:
                                       st.warning(f"Could not save selected library persona '{selected_name}' before replacement: {save_err}", icon="⚠️")

                                 st.session_state.personas[idx_to_replace] = selected_persona_data
                                 st.session_state.show_persona_library = False
                                 st.session_state.persona_replace_index = None
                                 st.success(f"Replaced with '{selected_name}'.", icon="✅")
                                 st.rerun()
                      with col_rep2:
                           if st.button("Cancel Replacement", key="cancel_replace", use_container_width=True):
                                st.session_state.show_persona_library = False
                                st.session_state.persona_replace_index = None
                                st.rerun()

        # --- Manage Persona Library (Separate Section) ---
        st.markdown("---")
        with st.expander("Manage Persona Library (View/Edit/Delete Saved Personas)", expanded=False):
             saved_personas_db = get_all_personas()
             if not saved_personas_db:
                  st.write("No personas currently saved in the database library.")
             else:
                  st.write(f"Found {len(saved_personas_db)} personas in the database:")
                  for persona_db in saved_personas_db:
                       p_id = persona_db.get("id", None)
                       if p_id is None: continue # Skip if no ID somehow
                       p_name = persona_db.get("name", f"DB_Persona_{p_id}")
                       st.markdown(f"**{p_name}** ({persona_db.get('role', 'N/A')})")
                       with st.form(key=f"edit_db_persona_{p_id}"):
                           # Display non-editable ID for reference
                           # st.text_input("ID (Read-Only)", value=p_id, disabled=True)
                           # Editable fields matching the card editor
                           name_db = st.text_input("Name", value=persona_db.get("name", ""), key=f"db_{p_id}_name")
                           role_db = st.text_input("Role/Title", value=persona_db.get("role", ""), key=f"db_{p_id}_role")
                           bio_db = st.text_area("Bio", value=persona_db.get("bio", ""), height=100, key=f"db_{p_id}_bio")
                           goals_db = st.text_area("Goals (comma-separated)", value=", ".join(persona_db.get("goals", [])), key=f"db_{p_id}_goals")
                           # Add other fields if needed (traits, influences, etc.)

                           b_col_db1, b_col_db2 = st.columns(2)
                           with b_col_db1:
                                if st.form_submit_button("Update in Library", use_container_width=True):
                                     updated_db_persona = persona_db.copy() # Start with existing data
                                     updated_db_persona.update({
                                         "name": name_db, "role": role_db, "bio": bio_db,
                                         "goals": [g.strip() for g in goals_db.split(',') if g.strip()],
                                         # Update other fields here if they were made editable
                                     })
                                     if update_persona(updated_db_persona): # Update using ID
                                         st.success(f"Persona '{name_db}' (ID: {p_id}) updated in library.", icon="✅")
                                         # Update local JSON too
                                         try:
                                             filename = os.path.join(config.PERSONAS_DIR, f"{name_db.replace(' ', '_').lower()}.json")
                                             with open(filename, "w") as f: json.dump(updated_db_persona, f, indent=2)
                                         except Exception as json_err:
                                             st.warning(f"Could not update local JSON for {name_db}: {json_err}", icon="⚠️")
                                         st.rerun()
                                     else:
                                         st.error(f"Failed to update persona '{name_db}' in library.", icon="🚨")
                           with b_col_db2:
                               # Add a confirmation for delete
                               if f"confirm_delete_{p_id}" not in st.session_state:
                                    st.session_state[f"confirm_delete_{p_id}"] = False

                                if st.form_submit_button("⚠️ Delete from Library", use_container_width=True):
                                     st.session_state[f"confirm_delete_{p_id}"] = True # Show confirmation

                                if st.session_state[f"confirm_delete_{p_id}"]:
                                     st.warning(f"Are you sure you want to delete '{p_name}'? This cannot be undone.")
                                     c1, c2 = st.columns(2)
                                     with c1:
                                          if st.button("Confirm Delete", key=f"del_confirm_{p_id}", type="primary", use_container_width=True):
                                               if delete_persona(p_id):
                                                    st.success(f"Persona '{p_name}' deleted.", icon="🗑️")
                                                    # Delete local JSON file if it exists
                                                    try:
                                                         filename = os.path.join(config.PERSONAS_DIR, f"{p_name.replace(' ', '_').lower()}.json")
                                                         if os.path.exists(filename):
                                                              os.remove(filename)
                                                    except OSError as e:
                                                         st.warning(f"Could not delete local JSON file for {p_name}: {e}", icon="⚠️")
                                                    del st.session_state[f"confirm_delete_{p_id}"] # Reset confirmation state
                                                    st.rerun()
                                               else:
                                                    st.error(f"Failed to delete persona '{p_name}'.", icon="🚨")
                                     with c2:
                                          if st.button("Cancel Delete", key=f"del_cancel_{p_id}", use_container_width=True):
                                               st.session_state[f"confirm_delete_{p_id}"] = False
                                               st.rerun()


    # --- Step 3: Run Simulation ---
    elif st.session_state.current_step == 3:
        st.markdown("<div class='step-header'>Step 3: Configure & Run Simulation</div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box">Select the simulation method and parameters. The simulation will model the debate based on the defined personas and context.</div>', unsafe_allow_html=True)

        if not st.session_state.personas:
             st.warning("Please generate or confirm personas in Step 2 before running the simulation.", icon="⚠️")
             st.stop()
        if not st.session_state.extracted_structure:
            st.warning("Missing extracted decision structure. Please go back to Step 1.", icon="⚠️")
            st.stop()

        st.markdown("<div class='section-subheader'>Simulation Settings</div>", unsafe_allow_html=True)

        col_sim1, col_sim2 = st.columns(2)

        with col_sim1:
            simulation_type = st.selectbox(
                "**Simulation Method:**",
                options=config.SIMULATION_METHODS,
                index=config.SIMULATION_METHODS.index(st.session_state.last_simulation_type) if st.session_state.last_simulation_type in config.SIMULATION_METHODS else 0,
                key="simulation_type_select",
                help="Choose the engine for simulating the debate. LLM-based simulations are more nuanced but slower/costlier."
            )
        with col_sim2:
             simulation_rounds = st.number_input(
                "**Number of Rounds:**",
                min_value=1,
                max_value=15, # Set a reasonable max
                value=config.DEBATE_ROUNDS,
                step=1,
                key="simulation_rounds_input",
                help="How many turns or phases should the simulation run for?"
             )

        simulation_time_minutes = st.slider(
             "**Maximum Simulation Time (minutes):**",
             min_value=1,
             max_value=10, # Increased max time
             value=config.MAX_SIMULATION_TIME_SECONDS // 60,
             step=1,
             key="simulation_time_slider",
             help="Set a time limit for the simulation to prevent excessive runtimes, especially for LLM methods."
         )
        max_time_seconds = simulation_time_minutes * 60

        st.markdown("---")

        if st.button("🚀 Launch Simulation", key="launch_simulation_button", use_container_width=True):
             st.session_state.simulation_transcript = [] # Clear previous results
             st.session_state.analysis_results = None
             st.session_state.visualization_objects = None
             st.session_state.last_simulation_type = simulation_type # Remember choice

             # Call the debater agent
             transcript = simulate_debate(
                 personas=st.session_state.personas,
                 dilemma=st.session_state.decision_dilemma,
                 extracted_structure=st.session_state.extracted_structure,
                 rounds=simulation_rounds,
                 max_simulation_time=max_time_seconds,
                 simulation_type=simulation_type
             )

             st.session_state.simulation_transcript = transcript
             if transcript and not any(entry.get('message', '').startswith("Error:") for entry in transcript):
                 st.success("Simulation complete!", icon="✅")
                 st.session_state.current_step = 4 # Move to analysis
                 st.rerun()
             else:
                 st.error("Simulation encountered errors or did not complete successfully. Please review the transcript.", icon="🚨")
                 # Stay on step 3, but show transcript if generated
                 if transcript:
                     st.markdown("<div class='section-subheader'>Simulation Transcript (Partial/Error)</div>", unsafe_allow_html=True)
                     with st.expander("View Transcript", expanded=True):
                        for entry in transcript:
                            st.markdown(f"<div class='transcript-entry'>", unsafe_allow_html=True)
                            st.markdown(f"<span class='transcript-agent'>{entry.get('agent', 'System')}</span> <span class='transcript-meta'>(Round {entry.get('round', '?')}, Step: {entry.get('step', 'N/A')})</span>", unsafe_allow_html=True)
                            st.markdown(entry.get('message', 'No message content.'))
                            st.markdown(f"</div>", unsafe_allow_html=True)


    # --- Step 4: Analyze Results ---
    elif st.session_state.current_step == 4:
        st.markdown("<div class='step-header'>Step 4: Analyze Simulation Results</div>", unsafe_allow_html=True)
        st.markdown('<div class="info-box">The simulation is complete. Review the debate transcript below. Click "Analyze Debate" to generate insights, summaries, and visualizations.</div>', unsafe_allow_html=True)

        if not st.session_state.simulation_transcript:
            st.warning("No simulation transcript available. Please run a simulation in Step 3.", icon="⚠️")
            st.stop()

        st.markdown("<div class='section-subheader'>Simulation Transcript</div>", unsafe_allow_html=True)
        # Use columns for better layout potentially
        # transcript_col, action_col = st.columns([4, 1])
        # with transcript_col:

        # Display Transcript in an expandable container
        with st.expander("View Full Debate Transcript", expanded=True):
             # Add option to filter transcript?
             # search_term = st.text_input("Search transcript:", key="transcript_search")
             displayed_count = 0
             max_display = 50 # Limit initial display to avoid overwhelming UI
             for i, entry in enumerate(st.session_state.simulation_transcript):
                  # if search
