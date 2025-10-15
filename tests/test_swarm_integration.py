"""Tests for RAGSwarm integration with NotebookLlaMa."""

import pytest
import asyncio
from src.notebookllama.swarm_integration import (
    SwarmNotebookProcessor,
    SwarmConfig,
    get_swarm_processor,
    process_with_swarm,
)
from src.notebookllama.opencog_atomspace import AtomType


@pytest.mark.asyncio
async def test_swarm_config_defaults():
    """Test SwarmConfig with default values."""
    config = SwarmConfig()
    
    assert config.num_retrievers >= 1
    assert config.num_reasoners >= 1
    assert config.node_id is not None


@pytest.mark.asyncio
async def test_swarm_config_to_dict():
    """Test SwarmConfig serialization."""
    config = SwarmConfig()
    config_dict = config.to_dict()
    
    assert "num_retrievers" in config_dict
    assert "num_reasoners" in config_dict
    assert "node_id" in config_dict


@pytest.mark.asyncio
async def test_swarm_processor_creation():
    """Test creating a SwarmNotebookProcessor."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    
    assert processor.node_id == "test_node"
    assert processor.orchestrator is not None
    assert processor._initialized is False


@pytest.mark.asyncio
async def test_swarm_processor_initialization():
    """Test initializing the swarm processor."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    
    await processor.initialize_swarm(
        num_retrievers=2,
        num_reasoners=1,
        num_synthesizers=1,
        num_validators=1,
    )
    
    assert processor._initialized is True
    status = await processor.orchestrator.get_swarm_status()
    assert status["total_agents"] == 5


@pytest.mark.asyncio
async def test_swarm_processor_add_knowledge_atom():
    """Test adding knowledge atoms through processor."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    await processor.initialize_swarm()
    
    atom_id = await processor.add_knowledge_atom(
        atom_type=AtomType.CONCEPT,
        name="test_concept",
        metadata={"description": "A test concept"},
        truth_value=0.95,
    )
    
    assert atom_id is not None
    
    # Verify atom was added
    atomspace = processor.orchestrator.distributed_atomspace.local_space
    atom = await atomspace.get_atom(atom_id)
    assert atom is not None
    assert atom.name == "test_concept"
    assert atom.truth_value == 0.95


@pytest.mark.asyncio
async def test_swarm_processor_query():
    """Test querying through the processor."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    await processor.initialize_swarm()
    
    result = await processor.query_swarm("test query")
    
    assert result is not None
    assert "query" in result
    assert result["query"] == "test query"


@pytest.mark.asyncio
async def test_swarm_processor_metrics():
    """Test getting swarm metrics."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    await processor.initialize_swarm(num_retrievers=2)
    
    metrics = await processor.get_swarm_metrics()
    
    assert "total_agents" in metrics
    assert "knowledge_base_size" in metrics
    assert "average_agent_load" in metrics
    assert "max_agent_load" in metrics
    assert "min_agent_load" in metrics


@pytest.mark.asyncio
async def test_spread_activation_through_processor():
    """Test spreading activation through the processor."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    await processor.initialize_swarm()
    
    # Add a concept
    atom_id = await processor.add_knowledge_atom(
        atom_type=AtomType.CONCEPT,
        name="central_concept",
    )
    
    # Spread activation
    activated = await processor.spread_activation(atom_id, depth=1)
    
    assert isinstance(activated, list)
    assert len(activated) > 0


@pytest.mark.asyncio
async def test_process_document_with_swarm():
    """Test processing a document with the swarm."""
    processor = SwarmNotebookProcessor(node_id="test_node")
    
    result = await processor.process_document_with_swarm(
        document_id="test_doc_123",
        content="This is a test document about artificial intelligence and machine learning.",
        metadata={"source": "test"},
    )
    
    assert result is not None
    assert "document_id" in result
    assert result["document_id"] == "test_doc_123"
    assert "summary" in result
    assert "highlights" in result
    assert "questions" in result
    assert "swarm_status" in result


@pytest.mark.asyncio
async def test_get_swarm_processor_singleton():
    """Test that get_swarm_processor returns singleton."""
    processor1 = get_swarm_processor()
    processor2 = get_swarm_processor()
    
    assert processor1 is processor2


@pytest.mark.asyncio
async def test_process_with_swarm_convenience_function():
    """Test the convenience function for processing with swarm."""
    result = await process_with_swarm(
        document_id="convenience_test",
        content="Test content for convenience function",
        metadata={"test": True},
    )
    
    assert result is not None
    assert "document_id" in result
    assert result["document_id"] == "convenience_test"
