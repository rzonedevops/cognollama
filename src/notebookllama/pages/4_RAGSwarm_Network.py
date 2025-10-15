"""
RAGSwarm Network Interface - Streamlit Page

This page provides an interface to interact with the OpenCog-inspired
RAGSwarm distributed network for knowledge management and querying.
"""

import streamlit as st
import asyncio
import sys
import os
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarm_integration import (
    get_swarm_processor,
    SwarmConfig,
)
from opencog_atomspace import AtomType


def run_async(coro):
    """Helper to run async functions in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def format_swarm_response(result: Dict[str, Any]) -> str:
    """Format swarm query response for display."""
    output = "## 🌐 RAGSwarm Response\n\n"
    
    if result.get("synthesis", {}).get("synthesized_content"):
        output += f"**Answer:** {result['synthesis']['synthesized_content']}\n\n"
    
    if result.get("validation", {}).get("is_valid"):
        confidence = result["validation"].get("confidence", 0.0)
        status = "✅ Validated" if result["validation"]["is_valid"] else "❌ Invalid"
        output += f"**Validation:** {status} (confidence: {confidence:.2%})\n\n"
    
    if result.get("agents_used", 0) > 0:
        output += f"**Agents Collaborated:** {result['agents_used']}\n\n"
    
    if result.get("reasoning"):
        output += "**Reasoning Process:**\n"
        reasoning = result["reasoning"]
        if reasoning.get("confidence"):
            output += f"- Confidence: {reasoning['confidence']:.2%}\n"
    
    return output


# Streamlit UI Configuration
st.set_page_config(
    page_title="NotebookLlaMa - RAGSwarm Network",
    page_icon="🌐",
    layout="wide",
)

st.sidebar.header("RAGSwarm Network 🌐")
st.sidebar.info(
    "A distributed swarm intelligence system inspired by OpenCog. "
    "Multiple specialized agents collaborate to process queries."
)
st.markdown("---")
st.markdown("## NotebookLlaMa - RAGSwarm Network 🌐")

# Initialize session state
if "swarm_initialized" not in st.session_state:
    st.session_state.swarm_initialized = False
if "swarm_messages" not in st.session_state:
    st.session_state.swarm_messages = []

# Swarm Configuration Section
with st.expander("⚙️ Swarm Configuration", expanded=not st.session_state.swarm_initialized):
    st.markdown("### Configure Agent Swarm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_retrievers = st.number_input(
            "🔍 Number of Retriever Agents",
            min_value=1,
            max_value=10,
            value=3,
            help="Agents specialized in information retrieval",
        )
        
        num_reasoners = st.number_input(
            "🧠 Number of Reasoner Agents",
            min_value=1,
            max_value=10,
            value=2,
            help="Agents specialized in logical reasoning",
        )
    
    with col2:
        num_synthesizers = st.number_input(
            "🔄 Number of Synthesizer Agents",
            min_value=1,
            max_value=5,
            value=1,
            help="Agents specialized in combining information",
        )
        
        num_validators = st.number_input(
            "✅ Number of Validator Agents",
            min_value=1,
            max_value=5,
            value=1,
            help="Agents specialized in validation",
        )
    
    if st.button("🚀 Initialize Swarm", type="primary"):
        with st.spinner("Initializing RAGSwarm network..."):
            try:
                processor = get_swarm_processor()
                run_async(
                    processor.initialize_swarm(
                        num_retrievers=num_retrievers,
                        num_reasoners=num_reasoners,
                        num_synthesizers=num_synthesizers,
                        num_validators=num_validators,
                    )
                )
                st.session_state.swarm_initialized = True
                st.success(
                    f"✅ Swarm initialized with {num_retrievers + num_reasoners + num_synthesizers + num_validators} agents!"
                )
            except Exception as e:
                st.error(f"❌ Error initializing swarm: {str(e)}")

# Swarm Status Section
if st.session_state.swarm_initialized:
    with st.expander("📊 Swarm Status & Metrics", expanded=False):
        if st.button("🔄 Refresh Status"):
            with st.spinner("Fetching swarm status..."):
                try:
                    processor = get_swarm_processor()
                    metrics = run_async(processor.get_swarm_metrics())
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Agents", metrics["total_agents"])
                        st.metric("Knowledge Base Size", metrics["knowledge_base_size"])
                    
                    with col2:
                        st.metric("Average Load", f"{metrics['average_agent_load']:.2%}")
                        st.metric("Max Load", f"{metrics['max_agent_load']:.2%}")
                    
                    with col3:
                        st.metric("Min Load", f"{metrics['min_agent_load']:.2%}")
                    
                    # Agent distribution
                    st.markdown("### Agent Distribution by Role")
                    agents_by_role = metrics.get("agents_by_role", {})
                    
                    role_cols = st.columns(len(agents_by_role))
                    for i, (role, count) in enumerate(agents_by_role.items()):
                        with role_cols[i]:
                            emoji = {
                                "retriever": "🔍",
                                "reasoner": "🧠",
                                "synthesizer": "🔄",
                                "validator": "✅",
                                "coordinator": "🎯",
                            }.get(role, "🤖")
                            st.metric(f"{emoji} {role.title()}", count)
                
                except Exception as e:
                    st.error(f"Error fetching status: {str(e)}")

# Query Interface
st.markdown("---")
st.markdown("### 💬 Query the Swarm")

# Display conversation history
for message in st.session_state.swarm_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input
if query := st.chat_input("Ask the swarm a question..."):
    if not st.session_state.swarm_initialized:
        st.warning("⚠️ Please initialize the swarm first using the configuration section above.")
    else:
        # Add user message to history
        st.session_state.swarm_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query with swarm
        with st.chat_message("assistant"):
            with st.spinner("🌐 Swarm agents collaborating..."):
                try:
                    processor = get_swarm_processor()
                    result = run_async(processor.query_swarm(query))
                    
                    response = format_swarm_response(result)
                    st.markdown(response)
                    
                    # Show detailed results in expander
                    with st.expander("🔍 Detailed Swarm Results"):
                        st.json(result)
                    
                    # Add to history
                    st.session_state.swarm_messages.append(
                        {"role": "assistant", "content": response}
                    )
                
                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.swarm_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

# Knowledge Management Section
st.markdown("---")
st.markdown("### 📚 Knowledge Management")

with st.expander("➕ Add Knowledge to AtomSpace"):
    st.markdown("Add custom knowledge atoms to the distributed knowledge base.")
    
    atom_type = st.selectbox(
        "Atom Type",
        options=["CONCEPT", "DOCUMENT", "QUERY", "ANSWER", "NODE", "LINK"],
    )
    
    atom_name = st.text_input("Atom Name/Identifier")
    
    truth_value = st.slider(
        "Truth Value (Confidence)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )
    
    metadata_text = st.text_area(
        "Metadata (JSON format)",
        value='{"description": "Your metadata here"}',
    )
    
    if st.button("Add Atom"):
        if not atom_name:
            st.warning("Please provide an atom name.")
        else:
            try:
                import json
                metadata = json.loads(metadata_text)
                
                processor = get_swarm_processor()
                atom_id = run_async(
                    processor.add_knowledge_atom(
                        atom_type=AtomType[atom_type],
                        name=atom_name,
                        metadata=metadata,
                        truth_value=truth_value,
                    )
                )
                
                st.success(f"✅ Atom added successfully! ID: {atom_id}")
            
            except json.JSONDecodeError:
                st.error("Invalid JSON in metadata field.")
            except Exception as e:
                st.error(f"Error adding atom: {str(e)}")

# Information Section
with st.expander("ℹ️ About RAGSwarm"):
    st.markdown("""
    ### What is RAGSwarm?
    
    RAGSwarm is a distributed Retrieval-Augmented Generation system inspired by:
    - **OpenCog**: Uses a hypergraph-based knowledge representation (AtomSpace)
    - **Swarm Intelligence**: Multiple specialized agents collaborate on tasks
    - **Distributed Processing**: Knowledge and processing distributed across agents
    
    ### How it works:
    
    1. **Knowledge Storage**: Documents and information stored as atoms in a hypergraph
    2. **Agent Specialization**: Different agents specialize in:
       - 🔍 **Retrieval**: Finding relevant information
       - 🧠 **Reasoning**: Logical inference and pattern matching
       - 🔄 **Synthesis**: Combining information from multiple sources
       - ✅ **Validation**: Ensuring accuracy and consistency
    3. **Collaborative Processing**: Agents work together to answer queries
    4. **Attention Spreading**: Important concepts receive more attention in the network
    
    ### Benefits:
    
    - **Scalability**: Add more agents to handle more load
    - **Robustness**: System continues working if some agents fail
    - **Specialization**: Agents can be optimized for specific tasks
    - **Transparency**: See which agents contributed to each answer
    """)

# Footer
st.markdown("---")
st.markdown("*OpenCog-inspired RAGSwarm Network for NotebookLlaMa*")
