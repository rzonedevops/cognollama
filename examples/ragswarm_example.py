"""
Example script demonstrating RAGSwarm usage.

This script shows how to:
1. Initialize the RAGSwarm network
2. Add documents to the knowledge base
3. Query the swarm
4. Inspect swarm metrics
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from notebookllama.swarm_integration import (
    SwarmNotebookProcessor,
    SwarmConfig,
)
from notebookllama.opencog_atomspace import AtomType


async def main():
    """Main example function."""
    
    print("=" * 60)
    print("RAGSwarm Example - OpenCog-Inspired Distributed Network")
    print("=" * 60)
    print()
    
    # Step 1: Create and initialize the swarm
    print("📝 Step 1: Initializing RAGSwarm...")
    processor = SwarmNotebookProcessor(node_id="example_node")
    
    await processor.initialize_swarm(
        num_retrievers=3,
        num_reasoners=2,
        num_synthesizers=1,
        num_validators=1,
    )
    print("✅ Swarm initialized successfully!")
    print()
    
    # Step 2: Check swarm status
    print("📊 Step 2: Checking swarm status...")
    metrics = await processor.get_swarm_metrics()
    print(f"   Total Agents: {metrics['total_agents']}")
    print(f"   Knowledge Base Size: {metrics['knowledge_base_size']} atoms")
    print(f"   Average Agent Load: {metrics['average_agent_load']:.2%}")
    print()
    
    print("   Agents by role:")
    for role, count in metrics.get("agents_by_role", {}).items():
        print(f"     - {role.title()}: {count}")
    print()
    
    # Step 3: Add knowledge to the distributed knowledge base
    print("📚 Step 3: Adding knowledge to AtomSpace...")
    
    # Add concepts
    await processor.add_knowledge_atom(
        atom_type=AtomType.CONCEPT,
        name="artificial_intelligence",
        metadata={
            "description": "The simulation of human intelligence by machines",
            "field": "computer_science",
        },
        truth_value=0.95,
    )
    
    await processor.add_knowledge_atom(
        atom_type=AtomType.CONCEPT,
        name="machine_learning",
        metadata={
            "description": "A subset of AI focused on learning from data",
            "field": "computer_science",
        },
        truth_value=0.98,
    )
    
    # Add a document
    document_content = """
    Machine Learning: A Comprehensive Overview
    
    Machine learning is a method of data analysis that automates analytical model 
    building. It is a branch of artificial intelligence based on the idea that 
    systems can learn from data, identify patterns and make decisions with minimal 
    human intervention.
    
    Key concepts in machine learning include:
    - Supervised Learning: Learning from labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through trial and error
    - Deep Learning: Neural networks with multiple layers
    
    Applications of machine learning span across various domains including 
    computer vision, natural language processing, robotics, and predictive analytics.
    """
    
    await processor.orchestrator.add_document_to_knowledge(
        document_id="ml_overview_doc",
        content=document_content,
        metadata={
            "title": "Machine Learning Overview",
            "author": "Example Author",
            "topic": "machine_learning",
        },
    )
    
    print("✅ Knowledge added to AtomSpace!")
    print()
    
    # Step 4: Query the swarm
    print("🌐 Step 4: Querying the swarm...")
    print()
    
    queries = [
        "What is machine learning?",
        "What are the key concepts in machine learning?",
        "How does AI relate to machine learning?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"   Query {i}: {query}")
        result = await processor.query_swarm(query)
        
        print(f"   Agents used: {result.get('agents_used', 0)}")
        
        if result.get("synthesis", {}).get("synthesized_content"):
            print(f"   Answer: {result['synthesis']['synthesized_content']}")
        
        if result.get("validation", {}).get("is_valid"):
            confidence = result["validation"].get("confidence", 0.0)
            print(f"   Validation: ✅ (confidence: {confidence:.2%})")
        
        print()
    
    # Step 5: Process a document with the swarm
    print("📄 Step 5: Processing document with swarm...")
    
    doc_result = await processor.process_document_with_swarm(
        document_id="ai_basics_doc",
        content="""
        Artificial Intelligence (AI) represents one of the most transformative 
        technologies of our time. It encompasses various approaches to creating 
        intelligent machines that can perceive, reason, learn, and act.
        
        The field has evolved significantly since its inception in the 1950s, 
        with major breakthroughs in areas like natural language processing, 
        computer vision, and game playing.
        
        Today, AI systems are embedded in everyday applications, from 
        recommendation systems to autonomous vehicles, demonstrating both 
        the power and potential of intelligent machines.
        """,
        metadata={
            "title": "AI Basics",
            "category": "introduction",
        },
    )
    
    print(f"   Document ID: {doc_result['document_id']}")
    print(f"   Swarm status: {doc_result['swarm_status']['initialized']}")
    print("✅ Document processed successfully!")
    print()
    
    # Step 6: Demonstrate activation spreading
    print("🔗 Step 6: Demonstrating activation spreading...")
    
    # Get the AI concept atom
    atomspace = processor.orchestrator.distributed_atomspace.local_space
    ai_atoms = await atomspace.find_atoms_by_name("artificial_intelligence")
    
    if ai_atoms:
        ai_atom = ai_atoms[0]
        print(f"   Starting from atom: {ai_atom.name}")
        
        # Spread activation
        activated = await processor.spread_activation(ai_atom.atom_id, depth=2)
        
        print(f"   Activated {len(activated)} atoms:")
        for atom_dict in activated[:5]:  # Show first 5
            print(f"     - {atom_dict['name']} (attention: {atom_dict['attention_value']:.2f})")
        
        if len(activated) > 5:
            print(f"     ... and {len(activated) - 5} more")
    
    print()
    
    # Step 7: Final metrics
    print("📊 Step 7: Final swarm metrics...")
    final_metrics = await processor.get_swarm_metrics()
    print(f"   Total Agents: {final_metrics['total_agents']}")
    print(f"   Knowledge Base Size: {final_metrics['knowledge_base_size']} atoms")
    print(f"   Average Load: {final_metrics['average_agent_load']:.2%}")
    print()
    
    print("=" * 60)
    print("✨ RAGSwarm Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
