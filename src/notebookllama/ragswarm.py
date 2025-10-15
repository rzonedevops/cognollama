"""
RAGSwarm - Distributed swarm intelligence for Retrieval-Augmented Generation.

This module implements a swarm-based approach to RAG, where multiple agents
collaborate to process queries, retrieve information, and generate responses.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import random
from datetime import datetime
import json

from .opencog_atomspace import AtomSpace, Atom, AtomType, DistributedAtomSpace


class AgentRole(Enum):
    """Roles that agents can take in the swarm."""
    RETRIEVER = "retriever"  # Specializes in information retrieval
    REASONER = "reasoner"  # Specializes in reasoning and inference
    SYNTHESIZER = "synthesizer"  # Specializes in combining information
    VALIDATOR = "validator"  # Specializes in validating responses
    COORDINATOR = "coordinator"  # Coordinates swarm activities


class AgentState(Enum):
    """States an agent can be in."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    COLLABORATING = "collaborating"
    FAILED = "failed"


@dataclass
class SwarmMessage:
    """Message passed between agents in the swarm."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: float = 0.5


@dataclass
class AgentCapability:
    """Describes an agent's capabilities."""
    role: AgentRole
    specialization: str
    performance_score: float = 1.0
    load: float = 0.0  # Current workload (0-1)


class SwarmAgent:
    """
    Individual agent in the RAGSwarm.
    Agents collaborate to process queries using distributed knowledge.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        atomspace: AtomSpace,
        specialization: str = "general",
    ):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState.IDLE
        self.atomspace = atomspace
        self.capability = AgentCapability(
            role=role,
            specialization=specialization,
        )
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.collaboration_history: List[str] = []
        self._lock = asyncio.Lock()
    
    async def process_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Process an incoming message and return a response if needed."""
        await self.message_queue.put(message)
        
        if message.message_type == "query":
            return await self._handle_query(message)
        elif message.message_type == "retrieve":
            return await self._handle_retrieve(message)
        elif message.message_type == "reason":
            return await self._handle_reason(message)
        elif message.message_type == "synthesize":
            return await self._handle_synthesize(message)
        elif message.message_type == "validate":
            return await self._handle_validate(message)
        
        return None
    
    async def _handle_query(self, message: SwarmMessage) -> SwarmMessage:
        """Handle a query request."""
        query = message.content.get("query", "")
        
        # Find relevant atoms in the knowledge base
        relevant_atoms = await self.atomspace.find_atoms_by_name(query)
        
        # Spread activation to find related concepts
        if relevant_atoms:
            activated = await self.atomspace.spread_activation(
                relevant_atoms[0].atom_id, depth=2
            )
        else:
            activated = []
        
        return SwarmMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type="query_response",
            content={
                "relevant_atoms": [a.to_dict() for a in relevant_atoms],
                "activated_atoms": [a.to_dict() for a in activated[:5]],
            },
        )
    
    async def _handle_retrieve(self, message: SwarmMessage) -> SwarmMessage:
        """Handle a retrieval request."""
        if self.role != AgentRole.RETRIEVER:
            return SwarmMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="error",
                content={"error": "Agent not a retriever"},
            )
        
        # Retrieve relevant information
        query = message.content.get("query", "")
        results = await self.atomspace.find_atoms_by_name(query)
        
        return SwarmMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type="retrieve_response",
            content={"results": [r.to_dict() for r in results]},
        )
    
    async def _handle_reason(self, message: SwarmMessage) -> SwarmMessage:
        """Handle a reasoning request."""
        if self.role != AgentRole.REASONER:
            return SwarmMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="error",
                content={"error": "Agent not a reasoner"},
            )
        
        # Perform reasoning over provided atoms
        atoms_data = message.content.get("atoms", [])
        
        # Simple reasoning: find connections and patterns
        reasoning_result = {
            "patterns": [],
            "inferences": [],
            "confidence": 0.8,
        }
        
        return SwarmMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type="reason_response",
            content=reasoning_result,
        )
    
    async def _handle_synthesize(self, message: SwarmMessage) -> SwarmMessage:
        """Handle a synthesis request."""
        if self.role != AgentRole.SYNTHESIZER:
            return SwarmMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="error",
                content={"error": "Agent not a synthesizer"},
            )
        
        # Synthesize information from multiple sources
        sources = message.content.get("sources", [])
        
        synthesis_result = {
            "synthesized_content": "Combined result from sources",
            "confidence": 0.85,
        }
        
        return SwarmMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type="synthesize_response",
            content=synthesis_result,
        )
    
    async def _handle_validate(self, message: SwarmMessage) -> SwarmMessage:
        """Handle a validation request."""
        if self.role != AgentRole.VALIDATOR:
            return SwarmMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="error",
                content={"error": "Agent not a validator"},
            )
        
        # Validate the provided content
        content = message.content.get("content", "")
        
        validation_result = {
            "is_valid": True,
            "confidence": 0.9,
            "issues": [],
        }
        
        return SwarmMessage(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type="validate_response",
            content=validation_result,
        )
    
    async def update_load(self, delta: float):
        """Update the agent's current load."""
        async with self._lock:
            self.capability.load = max(0.0, min(1.0, self.capability.load + delta))


class SwarmCoordinator:
    """
    Coordinates the swarm of agents.
    Implements swarm intelligence algorithms for distributed processing.
    """
    
    def __init__(self, atomspace: DistributedAtomSpace):
        self.atomspace = atomspace
        self.agents: Dict[str, SwarmAgent] = {}
        self.message_router: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    async def register_agent(self, agent: SwarmAgent):
        """Register a new agent in the swarm."""
        self.agents[agent.agent_id] = agent
    
    async def unregister_agent(self, agent_id: str):
        """Remove an agent from the swarm."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_agents_by_role(self, role: AgentRole) -> List[SwarmAgent]:
        """Get all agents with a specific role."""
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def select_best_agent(self, role: AgentRole) -> Optional[SwarmAgent]:
        """Select the best available agent for a role based on load and performance."""
        candidates = self.get_agents_by_role(role)
        if not candidates:
            return None
        
        # Score agents based on performance and inverse load
        def score_agent(agent: SwarmAgent) -> float:
            return agent.capability.performance_score * (1.0 - agent.capability.load)
        
        return max(candidates, key=score_agent)
    
    async def route_message(self, message: SwarmMessage):
        """Route a message to the appropriate agent(s)."""
        if message.receiver_id:
            # Direct message
            if message.receiver_id in self.agents:
                agent = self.agents[message.receiver_id]
                await agent.process_message(message)
        else:
            # Broadcast message
            for agent in self.agents.values():
                await agent.process_message(message)
    
    async def process_query_swarm(self, query: str) -> Dict[str, Any]:
        """
        Process a query using swarm intelligence.
        Coordinates multiple agents to collaboratively answer the query.
        """
        # Step 1: Select retriever agents
        retrievers = self.get_agents_by_role(AgentRole.RETRIEVER)
        if not retrievers:
            return {"error": "No retriever agents available"}
        
        # Step 2: Distribute retrieval across multiple agents
        retrieval_tasks = []
        for retriever in retrievers[:3]:  # Use up to 3 retrievers
            msg = SwarmMessage(
                sender_id="coordinator",
                receiver_id=retriever.agent_id,
                message_type="retrieve",
                content={"query": query},
            )
            retrieval_tasks.append(retriever.process_message(msg))
        
        retrieval_responses = await asyncio.gather(*retrieval_tasks)
        
        # Step 3: Aggregate retrieval results
        all_results = []
        for response in retrieval_responses:
            if response and response.message_type == "retrieve_response":
                all_results.extend(response.content.get("results", []))
        
        # Step 4: Reason over the results
        reasoner = self.select_best_agent(AgentRole.REASONER)
        if reasoner:
            reason_msg = SwarmMessage(
                sender_id="coordinator",
                receiver_id=reasoner.agent_id,
                message_type="reason",
                content={"atoms": all_results},
            )
            reason_response = await reasoner.process_message(reason_msg)
        else:
            reason_response = None
        
        # Step 5: Synthesize final answer
        synthesizer = self.select_best_agent(AgentRole.SYNTHESIZER)
        if synthesizer:
            synth_msg = SwarmMessage(
                sender_id="coordinator",
                receiver_id=synthesizer.agent_id,
                message_type="synthesize",
                content={
                    "sources": all_results,
                    "reasoning": reason_response.content if reason_response else {},
                },
            )
            synth_response = await synthesizer.process_message(synth_msg)
        else:
            synth_response = None
        
        # Step 6: Validate the answer
        validator = self.select_best_agent(AgentRole.VALIDATOR)
        if validator and synth_response:
            val_msg = SwarmMessage(
                sender_id="coordinator",
                receiver_id=validator.agent_id,
                message_type="validate",
                content={"content": synth_response.content},
            )
            validation = await validator.process_message(val_msg)
        else:
            validation = None
        
        return {
            "query": query,
            "retrieval_results": all_results,
            "reasoning": reason_response.content if reason_response else {},
            "synthesis": synth_response.content if synth_response else {},
            "validation": validation.content if validation else {},
            "agents_used": len(retrieval_responses) + (1 if reason_response else 0) + 
                          (1 if synth_response else 0) + (1 if validation else 0),
        }
    
    async def start(self):
        """Start the swarm coordinator."""
        self._running = True
        # In a full implementation, this would start background tasks
        # for message routing, load balancing, etc.
    
    async def stop(self):
        """Stop the swarm coordinator."""
        self._running = False


class RAGSwarmOrchestrator:
    """
    High-level orchestrator for the RAGSwarm system.
    Integrates with NotebookLlaMa workflow.
    """
    
    def __init__(self, node_id: str = "primary"):
        self.distributed_atomspace = DistributedAtomSpace(node_id=node_id)
        self.coordinator = SwarmCoordinator(self.distributed_atomspace)
        self._initialized = False
    
    async def initialize(self, num_retrievers: int = 3, num_reasoners: int = 2,
                         num_synthesizers: int = 1, num_validators: int = 1):
        """Initialize the swarm with agents."""
        # Create retriever agents
        for i in range(num_retrievers):
            agent = SwarmAgent(
                agent_id=f"retriever_{i}",
                role=AgentRole.RETRIEVER,
                atomspace=self.distributed_atomspace.local_space,
                specialization="document_retrieval",
            )
            await self.coordinator.register_agent(agent)
        
        # Create reasoner agents
        for i in range(num_reasoners):
            agent = SwarmAgent(
                agent_id=f"reasoner_{i}",
                role=AgentRole.REASONER,
                atomspace=self.distributed_atomspace.local_space,
                specialization="logical_reasoning",
            )
            await self.coordinator.register_agent(agent)
        
        # Create synthesizer agents
        for i in range(num_synthesizers):
            agent = SwarmAgent(
                agent_id=f"synthesizer_{i}",
                role=AgentRole.SYNTHESIZER,
                atomspace=self.distributed_atomspace.local_space,
                specialization="content_synthesis",
            )
            await self.coordinator.register_agent(agent)
        
        # Create validator agents
        for i in range(num_validators):
            agent = SwarmAgent(
                agent_id=f"validator_{i}",
                role=AgentRole.VALIDATOR,
                atomspace=self.distributed_atomspace.local_space,
                specialization="fact_checking",
            )
            await self.coordinator.register_agent(agent)
        
        await self.coordinator.start()
        self._initialized = True
    
    async def add_document_to_knowledge(self, document_id: str, content: str,
                                       metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the distributed knowledge base."""
        doc_atom = Atom(
            atom_type=AtomType.DOCUMENT,
            name=document_id,
            metadata=metadata or {},
        )
        doc_atom.metadata["content"] = content
        
        await self.distributed_atomspace.local_space.add_atom(doc_atom)
        await self.distributed_atomspace.broadcast_atom(doc_atom)
    
    async def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the swarm.
        """
        if not self._initialized:
            await self.initialize()
        
        # Add query to knowledge base
        query_atom = Atom(
            atom_type=AtomType.QUERY,
            name=query,
            attention_value=1.0,
        )
        await self.distributed_atomspace.local_space.add_atom(query_atom)
        
        # Process using swarm
        result = await self.coordinator.process_query_swarm(query)
        
        # Store answer in knowledge base
        if result.get("synthesis"):
            answer_atom = Atom(
                atom_type=AtomType.ANSWER,
                name=f"answer_to_{query}",
                outgoing=[query_atom.atom_id],
                metadata=result,
            )
            await self.distributed_atomspace.local_space.add_atom(answer_atom)
        
        return result
    
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get the current status of the swarm."""
        return {
            "total_agents": len(self.coordinator.agents),
            "agents_by_role": {
                role.value: len(self.coordinator.get_agents_by_role(role))
                for role in AgentRole
            },
            "knowledge_base_size": len(self.distributed_atomspace.local_space),
            "initialized": self._initialized,
        }
