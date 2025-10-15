"""Tests for RAGSwarm distributed agent system."""

import pytest
import asyncio
from src.notebookllama.ragswarm import (
    SwarmAgent,
    AgentRole,
    AgentState,
    SwarmMessage,
    SwarmCoordinator,
    RAGSwarmOrchestrator,
)
from src.notebookllama.opencog_atomspace import AtomSpace, Atom, AtomType


@pytest.mark.asyncio
async def test_swarm_agent_creation():
    """Test creating a swarm agent."""
    atomspace = AtomSpace()
    agent = SwarmAgent(
        agent_id="test_agent",
        role=AgentRole.RETRIEVER,
        atomspace=atomspace,
        specialization="test_specialization",
    )
    
    assert agent.agent_id == "test_agent"
    assert agent.role == AgentRole.RETRIEVER
    assert agent.state == AgentState.IDLE
    assert agent.capability.specialization == "test_specialization"


@pytest.mark.asyncio
async def test_swarm_message_creation():
    """Test creating a swarm message."""
    message = SwarmMessage(
        sender_id="agent1",
        receiver_id="agent2",
        message_type="query",
        content={"query": "test query"},
    )
    
    assert message.sender_id == "agent1"
    assert message.receiver_id == "agent2"
    assert message.message_type == "query"
    assert message.content["query"] == "test query"


@pytest.mark.asyncio
async def test_agent_query_handling():
    """Test agent handling a query message."""
    atomspace = AtomSpace()
    
    # Add some test data
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="test",
        attention_value=0.8,
    )
    await atomspace.add_atom(atom)
    
    agent = SwarmAgent(
        agent_id="retriever_1",
        role=AgentRole.RETRIEVER,
        atomspace=atomspace,
    )
    
    message = SwarmMessage(
        sender_id="coordinator",
        receiver_id="retriever_1",
        message_type="query",
        content={"query": "test"},
    )
    
    response = await agent.process_message(message)
    assert response is not None
    assert response.message_type == "query_response"


@pytest.mark.asyncio
async def test_agent_retrieve_handling():
    """Test agent handling a retrieve message."""
    atomspace = AtomSpace()
    
    # Add test data
    doc_atom = Atom(
        atom_type=AtomType.DOCUMENT,
        name="test_doc",
    )
    await atomspace.add_atom(doc_atom)
    
    agent = SwarmAgent(
        agent_id="retriever_1",
        role=AgentRole.RETRIEVER,
        atomspace=atomspace,
    )
    
    message = SwarmMessage(
        sender_id="coordinator",
        receiver_id="retriever_1",
        message_type="retrieve",
        content={"query": "test_doc"},
    )
    
    response = await agent.process_message(message)
    assert response is not None
    assert response.message_type == "retrieve_response"


@pytest.mark.asyncio
async def test_coordinator_agent_registration():
    """Test registering agents with coordinator."""
    from src.notebookllama.opencog_atomspace import DistributedAtomSpace
    
    dist_atomspace = DistributedAtomSpace(node_id="test")
    coordinator = SwarmCoordinator(dist_atomspace)
    
    agent1 = SwarmAgent(
        agent_id="agent1",
        role=AgentRole.RETRIEVER,
        atomspace=dist_atomspace.local_space,
    )
    
    agent2 = SwarmAgent(
        agent_id="agent2",
        role=AgentRole.REASONER,
        atomspace=dist_atomspace.local_space,
    )
    
    await coordinator.register_agent(agent1)
    await coordinator.register_agent(agent2)
    
    assert len(coordinator.agents) == 2
    assert "agent1" in coordinator.agents
    assert "agent2" in coordinator.agents


@pytest.mark.asyncio
async def test_coordinator_get_agents_by_role():
    """Test getting agents by role."""
    from src.notebookllama.opencog_atomspace import DistributedAtomSpace
    
    dist_atomspace = DistributedAtomSpace(node_id="test")
    coordinator = SwarmCoordinator(dist_atomspace)
    
    # Register multiple agents with different roles
    for i in range(3):
        agent = SwarmAgent(
            agent_id=f"retriever_{i}",
            role=AgentRole.RETRIEVER,
            atomspace=dist_atomspace.local_space,
        )
        await coordinator.register_agent(agent)
    
    reasoner = SwarmAgent(
        agent_id="reasoner_1",
        role=AgentRole.REASONER,
        atomspace=dist_atomspace.local_space,
    )
    await coordinator.register_agent(reasoner)
    
    retrievers = coordinator.get_agents_by_role(AgentRole.RETRIEVER)
    assert len(retrievers) == 3
    
    reasoners = coordinator.get_agents_by_role(AgentRole.REASONER)
    assert len(reasoners) == 1


@pytest.mark.asyncio
async def test_coordinator_select_best_agent():
    """Test selecting best agent based on load."""
    from src.notebookllama.opencog_atomspace import DistributedAtomSpace
    
    dist_atomspace = DistributedAtomSpace(node_id="test")
    coordinator = SwarmCoordinator(dist_atomspace)
    
    # Register agents with different loads
    agent1 = SwarmAgent(
        agent_id="retriever_1",
        role=AgentRole.RETRIEVER,
        atomspace=dist_atomspace.local_space,
    )
    agent1.capability.load = 0.8
    
    agent2 = SwarmAgent(
        agent_id="retriever_2",
        role=AgentRole.RETRIEVER,
        atomspace=dist_atomspace.local_space,
    )
    agent2.capability.load = 0.2
    
    await coordinator.register_agent(agent1)
    await coordinator.register_agent(agent2)
    
    best = coordinator.select_best_agent(AgentRole.RETRIEVER)
    assert best is not None
    assert best.agent_id == "retriever_2"  # Lower load


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test RAGSwarm orchestrator initialization."""
    orchestrator = RAGSwarmOrchestrator(node_id="test_node")
    
    await orchestrator.initialize(
        num_retrievers=2,
        num_reasoners=1,
        num_synthesizers=1,
        num_validators=1,
    )
    
    status = await orchestrator.get_swarm_status()
    assert status["total_agents"] == 5
    assert status["agents_by_role"]["retriever"] == 2
    assert status["agents_by_role"]["reasoner"] == 1
    assert status["initialized"] is True


@pytest.mark.asyncio
async def test_orchestrator_add_document():
    """Test adding documents to orchestrator's knowledge base."""
    orchestrator = RAGSwarmOrchestrator(node_id="test_node")
    await orchestrator.initialize()
    
    await orchestrator.add_document_to_knowledge(
        document_id="test_doc",
        content="This is test content",
        metadata={"author": "test"},
    )
    
    # Verify document was added
    atomspace = orchestrator.distributed_atomspace.local_space
    docs = await atomspace.find_atoms_by_type(AtomType.DOCUMENT)
    assert len(docs) > 0


@pytest.mark.asyncio
async def test_orchestrator_query():
    """Test querying the orchestrator."""
    orchestrator = RAGSwarmOrchestrator(node_id="test_node")
    await orchestrator.initialize(
        num_retrievers=1,
        num_reasoners=1,
        num_synthesizers=1,
        num_validators=1,
    )
    
    # Add some test data
    await orchestrator.add_document_to_knowledge(
        document_id="test_doc",
        content="Test content about AI",
    )
    
    # Query the swarm
    result = await orchestrator.query("AI")
    
    assert result is not None
    assert "query" in result
    assert result["query"] == "AI"
    assert "agents_used" in result


@pytest.mark.asyncio
async def test_agent_load_update():
    """Test updating agent load."""
    atomspace = AtomSpace()
    agent = SwarmAgent(
        agent_id="test_agent",
        role=AgentRole.RETRIEVER,
        atomspace=atomspace,
    )
    
    initial_load = agent.capability.load
    await agent.update_load(0.3)
    assert agent.capability.load == initial_load + 0.3
    
    # Test boundary
    await agent.update_load(1.0)
    assert agent.capability.load == 1.0  # Should cap at 1.0


@pytest.mark.asyncio
async def test_coordinator_process_query_swarm():
    """Test processing a query through the swarm."""
    from src.notebookllama.opencog_atomspace import DistributedAtomSpace
    
    dist_atomspace = DistributedAtomSpace(node_id="test")
    coordinator = SwarmCoordinator(dist_atomspace)
    
    # Register agents
    retriever = SwarmAgent(
        agent_id="retriever_1",
        role=AgentRole.RETRIEVER,
        atomspace=dist_atomspace.local_space,
    )
    reasoner = SwarmAgent(
        agent_id="reasoner_1",
        role=AgentRole.REASONER,
        atomspace=dist_atomspace.local_space,
    )
    synthesizer = SwarmAgent(
        agent_id="synthesizer_1",
        role=AgentRole.SYNTHESIZER,
        atomspace=dist_atomspace.local_space,
    )
    
    await coordinator.register_agent(retriever)
    await coordinator.register_agent(reasoner)
    await coordinator.register_agent(synthesizer)
    await coordinator.start()
    
    # Process query
    result = await coordinator.process_query_swarm("test query")
    
    assert result is not None
    assert "query" in result
    assert result["query"] == "test query"
