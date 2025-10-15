"""
Integration module for RAGSwarm with NotebookLlaMa workflow.

This module provides integration points between the distributed RAGSwarm
and the existing NotebookLlaMa document processing workflow.
"""

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .ragswarm import RAGSwarmOrchestrator, AgentRole
from .opencog_atomspace import Atom, AtomType

try:
    from .models import Notebook
except ImportError:
    # models module may not be available in all contexts
    Notebook = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional
    pass


class SwarmNotebookProcessor:
    """
    Integrates RAGSwarm with NotebookLlaMa document processing.
    Enables distributed processing of documents using swarm intelligence.
    """
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or os.getenv("SWARM_NODE_ID", "primary")
        self.orchestrator = RAGSwarmOrchestrator(node_id=self.node_id)
        self._initialized = False
    
    async def initialize_swarm(
        self,
        num_retrievers: int = 3,
        num_reasoners: int = 2,
        num_synthesizers: int = 1,
        num_validators: int = 1,
    ):
        """Initialize the swarm with specified agent counts."""
        if not self._initialized:
            await self.orchestrator.initialize(
                num_retrievers=num_retrievers,
                num_reasoners=num_reasoners,
                num_synthesizers=num_synthesizers,
                num_validators=num_validators,
            )
            self._initialized = True
    
    async def process_document_with_swarm(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document using the distributed swarm.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content
            metadata: Optional metadata about the document
        
        Returns:
            Processing results including summary, highlights, questions, and answers
        """
        if not self._initialized:
            await self.initialize_swarm()
        
        # Add document to distributed knowledge base
        await self.orchestrator.add_document_to_knowledge(
            document_id=document_id,
            content=content,
            metadata=metadata,
        )
        
        # Extract key information using swarm
        summary_result = await self.orchestrator.query(
            f"Summarize the key points from document {document_id}"
        )
        
        highlights_result = await self.orchestrator.query(
            f"Extract 5-10 crucial highlights from document {document_id}"
        )
        
        questions_result = await self.orchestrator.query(
            f"Generate 5-15 questions about document {document_id}"
        )
        
        return {
            "summary": summary_result,
            "highlights": highlights_result,
            "questions": questions_result,
            "document_id": document_id,
            "swarm_status": await self.orchestrator.get_swarm_status(),
        }
    
    async def query_swarm(self, query: str) -> Dict[str, Any]:
        """
        Query the swarm with a question.
        
        Args:
            query: The query string
        
        Returns:
            Swarm response with answer and supporting information
        """
        if not self._initialized:
            await self.initialize_swarm()
        
        return await self.orchestrator.query(query)
    
    async def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the swarm."""
        status = await self.orchestrator.get_swarm_status()
        
        # Calculate additional metrics
        agents = self.orchestrator.coordinator.agents
        
        agent_loads = [
            agent.capability.load
            for agent in agents.values()
        ]
        
        avg_load = sum(agent_loads) / len(agent_loads) if agent_loads else 0.0
        
        return {
            **status,
            "average_agent_load": avg_load,
            "max_agent_load": max(agent_loads) if agent_loads else 0.0,
            "min_agent_load": min(agent_loads) if agent_loads else 0.0,
        }
    
    async def add_knowledge_atom(
        self,
        atom_type: AtomType,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        truth_value: float = 1.0,
    ) -> str:
        """
        Add a knowledge atom directly to the distributed knowledge base.
        
        Args:
            atom_type: Type of the atom
            name: Name/identifier for the atom
            metadata: Optional metadata
            truth_value: Confidence value (0-1)
        
        Returns:
            The atom ID
        """
        atom = Atom(
            atom_type=atom_type,
            name=name,
            truth_value=truth_value,
            metadata=metadata or {},
        )
        
        atom_id = await self.orchestrator.distributed_atomspace.local_space.add_atom(
            atom
        )
        await self.orchestrator.distributed_atomspace.broadcast_atom(atom)
        
        return atom_id
    
    async def spread_activation(self, atom_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Spread activation from a starting atom through the knowledge network.
        
        Args:
            atom_id: Starting atom ID
            depth: How many hops to spread
        
        Returns:
            List of activated atoms with their details
        """
        atomspace = self.orchestrator.distributed_atomspace.local_space
        activated = await atomspace.spread_activation(atom_id, depth)
        
        return [atom.to_dict() for atom in activated]


class SwarmConfig:
    """Configuration for RAGSwarm system."""
    
    def __init__(self):
        self.num_retrievers = int(os.getenv("SWARM_NUM_RETRIEVERS", "3"))
        self.num_reasoners = int(os.getenv("SWARM_NUM_REASONERS", "2"))
        self.num_synthesizers = int(os.getenv("SWARM_NUM_SYNTHESIZERS", "1"))
        self.num_validators = int(os.getenv("SWARM_NUM_VALIDATORS", "1"))
        self.node_id = os.getenv("SWARM_NODE_ID", "primary")
        self.enable_distributed = os.getenv("SWARM_ENABLE_DISTRIBUTED", "false").lower() == "true"
        self.sync_interval = int(os.getenv("SWARM_SYNC_INTERVAL", "60"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_retrievers": self.num_retrievers,
            "num_reasoners": self.num_reasoners,
            "num_synthesizers": self.num_synthesizers,
            "num_validators": self.num_validators,
            "node_id": self.node_id,
            "enable_distributed": self.enable_distributed,
            "sync_interval": self.sync_interval,
        }


# Global swarm processor instance
_swarm_processor: Optional[SwarmNotebookProcessor] = None


def get_swarm_processor() -> SwarmNotebookProcessor:
    """Get or create the global swarm processor instance."""
    global _swarm_processor
    if _swarm_processor is None:
        config = SwarmConfig()
        _swarm_processor = SwarmNotebookProcessor(node_id=config.node_id)
    return _swarm_processor


async def process_with_swarm(
    document_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to process a document with the swarm.
    
    Args:
        document_id: Document identifier
        content: Document content
        metadata: Optional metadata
    
    Returns:
        Processing results
    """
    processor = get_swarm_processor()
    return await processor.process_document_with_swarm(
        document_id=document_id,
        content=content,
        metadata=metadata,
    )


async def query_swarm_async(query: str) -> Dict[str, Any]:
    """
    Convenience function to query the swarm.
    
    Args:
        query: Query string
    
    Returns:
        Query results
    """
    processor = get_swarm_processor()
    return await processor.query_swarm(query)
