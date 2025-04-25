import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from typing import List, Dict, Optional
from collections import Counter
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config # Import agraph components

from config import PLOTLY_TEMPLATE # Use consistent template

# --- Word Cloud ---

def generate_word_cloud_image(text: str, width: int = 800, height: int = 400) -> Optional[BytesIO]:
    """Generates a word cloud image from text."""
    if not text:
        return None
    try:
        wordcloud = WordCloud(width=width, height=height, background_color='white', colormap='viridis').generate(text)
        buf = BytesIO()
        plt.figure(figsize=(width / 100, height / 100)) # Convert pixels to inches for figsize
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        st.warning(f"Failed to generate word cloud: {e}", icon="⚠️")
        # print(f"Word cloud generation error: {e}") # Keep for debugging if needed
        return None

# --- Stakeholder Interaction Network ---

def plot_interaction_network_agraph(transcript: List[Dict], personas: List[Dict]) -> Optional[go.Figure]:
    """Generates an interactive stakeholder interaction network using streamlit-agraph."""
    if not transcript or not personas:
        return None

    nodes = []
    edges = []
    persona_names = {p['name'] for p in personas}
    interactions = Counter()
    edge_data = {} # Store sentiment or other metrics

    # Ensure all personas are nodes, even if they didn't speak
    for persona in personas:
         nodes.append(Node(id=persona['name'],
                           label=persona['name'],
                           title=persona.get('role', 'Unknown Role'), # Hover title
                           size=15, # Base size
                           # Add more styling based on role, etc. if desired
                          )
                      )

    # Build edges and count interactions
    for i in range(len(transcript) - 1):
        source_agent = transcript[i]['agent']
        target_agent = transcript[i+1]['agent']

        # Only add edges between valid personas listed
        if source_agent in persona_names and target_agent in persona_names:
             edge_key = tuple(sorted((source_agent, target_agent))) # Undirected interaction count
             interactions[edge_key] += 1
             # Add directed edge for flow
             edge_id = f"{source_agent}->{target_agent}_{i}"
             if not any(e.id == edge_id for e in edges): # Avoid duplicate directed edges for now
                 edges.append(Edge(id=edge_id, source=source_agent, target=target_agent, type="CURVE_SMOOTH")) # Add arrows


    # Configure graph appearance (see streamlit-agraph documentation for more options)
    config = Config(width='100%',
                    height=400,
                    directed=True,
                    physics=True, # Enable physics simulation for layout
                    hierarchical=False,
                    # nodeHighlightBehavior=True, # Highlight node and neighbors on hover
                    # highlightColor="#F7A7A6",
                    # directed=True,
                    # collapsible=True, # If you have hierarchical data
                    node={'labelProperty': 'label', 'renderLabel': True},
                    # edge={'labelProperty': 'label', 'renderLabel': True}, # Can add labels to edges
                   )

    # Return the components needed by agraph function in app.py
    return nodes, edges, config


def plot_interaction_network_plotly(transcript: List[Dict], personas: List[Dict]) -> Optional[go.Figure]:
    """Generates a stakeholder interaction network using Plotly and NetworkX (alternative)."""
    if not transcript or not personas:
        return None

    try:
        G = nx.DiGraph()
        agents = list(set(entry['agent'] for entry in transcript if 'agent' in entry))
        persona_map = {p['name']: p for p in personas}

        # Add nodes with metadata
        for agent_name in agents:
             if agent_name in persona_map:
                 persona = persona_map[agent_name]
                 G.add_node(agent_name,
                            role=persona.get('role', 'N/A'),
                            goals=', '.join(persona.get('goals', [])),
                            hover_text=f"{agent_name}\nRole: {persona.get('role', 'N/A')}\nGoals: {', '.join(persona.get('goals', []))}"
                           )
             else: # Handle agents in transcript but not in personas list (e.g., 'System')
                 G.add_node(agent_name, role='System/Other', goals='', hover_text=agent_name)


        # Add edges based on sequential interaction
        edge_weights = Counter()
        for i in range(len(transcript) - 1):
            source = transcript[i].get('agent')
            target = transcript[i+1].get('agent')
            if source in G and target in G: # Ensure nodes exist
                edge_weights[(source, target)] += 1
                if G.has_edge(source, target):
                    G[source][target]['weight'] += 1
                else:
                    G.add_edge(source, target, weight=1)

        if not G.nodes:
             st.warning("No nodes found for interaction graph.", icon="⚠️")
             return None

        pos = nx.spring_layout(G, k=0.6, iterations=50) # NetworkX layout algorithm

        edge_x, edge_y = [], []
        edge_traces = []

        # Create edges for Plotly
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Customize edge appearance (e.g., width based on weight)
            # width = min(1 + edge[2]['weight'] * 0.5, 5) # Example: scale width
            # edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], line=dict(width=width, color='#888'), hoverinfo='none', mode='lines'))


        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x, node_y, node_text, node_hover_text = [], [], [], []
        node_sizes = []
        # Create nodes for Plotly
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node) # Text label next to marker
            node_hover_text.append(G.nodes[node].get('hover_text', node)) # Hover text
            # Example: Size based on out-degree (influence) or other metric
            node_sizes.append(15 + G.out_degree(node) * 3)


        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_hover_text,
            textposition='top center',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu', # Example colorscale
                reversescale=True,
                color=[], # Placeholder, can be set based on role/sentiment etc.
                size=node_sizes,
                # colorbar=dict(
                #     thickness=15,
                #     title='Node Connections', # Add colorbar title if using color scale
                #     xanchor='left',
                #     titleside='right'
                # ),
                line_width=1))

        # Set node colors based on role or other attribute if desired
        # Example: color = [role_to_color_map.get(G.nodes[node]['role'], '#ccc') for node in G.nodes()]
        # node_trace.marker.color = color

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Stakeholder Interaction Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template=PLOTLY_TEMPLATE # Apply consistent template
                        ))
        return fig

    except Exception as e:
        st.warning(f"Failed to generate Plotly interaction network: {e}", icon="⚠️")
        # print(f"Plotly network generation error: {e}") # Keep for debugging
        return None

# --- Topic Distribution ---

def plot_topic_distribution(analysis: Dict) -> Optional[go.Figure]:
    """Plots the distribution of identified topics/keywords."""
    if not analysis or 'topics' not in analysis or not analysis['topics']:
        st.info("No topic data available for visualization.")
        return None
    try:
        # Ensure data format is correct (list of dicts with 'label' and 'weight')
        topics_data = analysis['topics']
        if not isinstance(topics_data, list) or not all('label' in t and 'weight' in t for t in topics_data):
             st.warning("Topic data is not in the expected format (list of {'label': str, 'weight': float}).", icon="⚠️")
             return None

        df = pd.DataFrame(topics_data)
        # Ensure weight is numeric
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        df = df.dropna(subset=['weight']) # Remove rows where weight couldn't be converted

        if df.empty:
            st.warning("No valid topic data to plot after cleaning.", icon="⚠️")
            return None

        fig = px.bar(df, x="label", y="weight", title="Key Topic/Keyword Distribution in Debate",
                     labels={'label': 'Topic/Keyword', 'weight': 'Relative Frequency / Weight'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="", yaxis_title="Weight")
        return fig
    except Exception as e:
        st.warning(f"Failed to generate topic distribution plot: {e}", icon="⚠️")
        # print(f"Topic distribution plot error: {e}") # Keep for debugging
        return None

# --- Sentiment Trend ---

def plot_sentiment_trend(analysis: Dict) -> Optional[go.Figure]:
    """Plots the sentiment trend over rounds for each agent."""
    if not analysis or 'sentiment_analysis' not in analysis or not analysis['sentiment_analysis']:
        st.info("No sentiment data available for visualization.")
        return None
    try:
         # Ensure data format is correct
        sentiment_data = analysis['sentiment_analysis']
        if not isinstance(sentiment_data, list) or not all('agent' in s and 'round' in s and 'score' in s for s in sentiment_data):
             st.warning("Sentiment data is not in the expected format (list of {'agent': str, 'round': int, 'score': float}).", icon="⚠️")
             return None

        df = pd.DataFrame(sentiment_data)
        # Ensure score and round are numeric
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df['round'] = pd.to_numeric(df['round'], errors='coerce')
        df = df.dropna(subset=['score', 'round']) # Remove rows where conversion failed

        if df.empty:
            st.warning("No valid sentiment data to plot after cleaning.", icon="⚠️")
            return None


        fig = px.line(df, x="round", y="score", color="agent", title="Sentiment Trend Over Rounds",
                      markers=True, # Add markers to points
                      labels={'round': 'Debate Round', 'score': 'Sentiment Score (Compound)', 'agent': 'Stakeholder'},
                      template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_title="Debate Round", yaxis_title="Sentiment Score")
        # Add a horizontal line at y=0 for neutral sentiment reference
        fig.add_hline(y=0, line_dash="dot", line_color="grey", annotation_text="Neutral", annotation_position="bottom right")

        return fig
    except Exception as e:
        st.warning(f"Failed to generate sentiment trend plot: {e}", icon="⚠️")
        # print(f"Sentiment trend plot error: {e}") # Keep for debugging
        return None

# --- Combined Function ---

def generate_visualizations(analysis: Dict, transcript: List[Dict], personas: List[Dict]) -> Dict[str, object]:
    """
    Generates all standard visualizations.

    Args:
        analysis (Dict): The output from transcript_analyzer.
        transcript (List[Dict]): The debate transcript.
        personas (List[Dict]): List of personas involved.

    Returns:
        Dict[str, object]: A dictionary containing visualization objects
                           (e.g., {'word_cloud': BytesIO, 'network': go.Figure, ...})
                           or None for failed visualizations.
    """
    visuals = {}

    # 1. Word Cloud (from transcript text)
    all_text = " ".join([entry.get('message', '') for entry in transcript])
    visuals['word_cloud'] = generate_word_cloud_image(all_text)

    # 2. Interaction Network (using streamlit-agraph)
    # visuals['network_agraph'] = plot_interaction_network_agraph(transcript, personas) # Returns tuple (nodes, edges, config)
    # Using Plotly version as primary for now as agraph requires direct call in app.py
    visuals['network_plotly'] = plot_interaction_network_plotly(transcript, personas)


    # 3. Topic Distribution (from analysis dict)
    visuals['topic_distribution'] = plot_topic_distribution(analysis)

    # 4. Sentiment Trend (from analysis dict)
    visuals['sentiment_trend'] = plot_sentiment_trend(analysis)

    # Add more visualizations here as needed

    return visuals
