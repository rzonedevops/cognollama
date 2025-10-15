"""
OpenCog-inspired AtomSpace implementation for distributed knowledge representation.

This module provides a simplified AtomSpace-like structure for representing
knowledge as a hypergraph, enabling distributed reasoning and knowledge sharing.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import hashlib
import json
from datetime import datetime


class AtomType(Enum):
    """Types of atoms in the knowledge hypergraph."""
    NODE = "node"
    LINK = "link"
    CONCEPT = "concept"
    PREDICATE = "predicate"
    DOCUMENT = "document"
    QUERY = "query"
    ANSWER = "answer"


@dataclass
class Atom:
    """
    Base atom class representing a node or link in the knowledge hypergraph.
    Inspired by OpenCog's Atom structure.
    """
    atom_type: AtomType
    name: str
    truth_value: float = 1.0  # Confidence/truth value (0-1)
    attention_value: float = 0.5  # Attention/importance (0-1)
    outgoing: List[str] = field(default_factory=list)  # Links to other atoms
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def atom_id(self) -> str:
        """Generate unique ID for this atom."""
        content = f"{self.atom_type.value}:{self.name}:{','.join(self.outgoing)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize atom to dictionary."""
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type.value,
            "name": self.name,
            "truth_value": self.truth_value,
            "attention_value": self.attention_value,
            "outgoing": self.outgoing,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Atom":
        """Deserialize atom from dictionary."""
        return cls(
            atom_type=AtomType(data["atom_type"]),
            name=data["name"],
            truth_value=data.get("truth_value", 1.0),
            attention_value=data.get("attention_value", 0.5),
            outgoing=data.get("outgoing", []),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
        )


class AtomSpace:
    """
    Distributed knowledge hypergraph inspired by OpenCog's AtomSpace.
    Stores atoms and provides reasoning capabilities.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self._lock = asyncio.Lock()
        self._indices: Dict[str, Set[str]] = {
            "type": {},
            "name": {},
        }
    
    async def add_atom(self, atom: Atom) -> str:
        """Add an atom to the AtomSpace."""
        async with self._lock:
            atom_id = atom.atom_id
            self.atoms[atom_id] = atom
            
            # Update indices
            type_key = atom.atom_type.value
            if type_key not in self._indices["type"]:
                self._indices["type"][type_key] = set()
            self._indices["type"][type_key].add(atom_id)
            
            if atom.name not in self._indices["name"]:
                self._indices["name"][atom.name] = set()
            self._indices["name"][atom.name].add(atom_id)
            
            return atom_id
    
    async def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve an atom by ID."""
        return self.atoms.get(atom_id)
    
    async def find_atoms_by_type(self, atom_type: AtomType) -> List[Atom]:
        """Find all atoms of a specific type."""
        atom_ids = self._indices["type"].get(atom_type.value, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    async def find_atoms_by_name(self, name: str) -> List[Atom]:
        """Find all atoms with a specific name."""
        atom_ids = self._indices["name"].get(name, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    async def update_attention(self, atom_id: str, delta: float):
        """Update the attention value of an atom."""
        atom = await self.get_atom(atom_id)
        if atom:
            atom.attention_value = max(0.0, min(1.0, atom.attention_value + delta))
    
    async def spread_activation(self, start_atom_id: str, depth: int = 2) -> List[Atom]:
        """
        Spread activation from a starting atom through the network.
        Returns activated atoms sorted by attention value.
        """
        visited = set()
        to_visit = [(start_atom_id, depth)]
        activated = []
        
        while to_visit:
            current_id, remaining_depth = to_visit.pop(0)
            if current_id in visited or remaining_depth < 0:
                continue
            
            visited.add(current_id)
            atom = await self.get_atom(current_id)
            
            if atom:
                activated.append(atom)
                await self.update_attention(current_id, 0.1)
                
                if remaining_depth > 0:
                    for outgoing_id in atom.outgoing:
                        if outgoing_id not in visited:
                            to_visit.append((outgoing_id, remaining_depth - 1))
        
        return sorted(activated, key=lambda a: a.attention_value, reverse=True)
    
    async def export_state(self) -> Dict[str, Any]:
        """Export the entire AtomSpace state."""
        return {
            "atoms": {aid: atom.to_dict() for aid, atom in self.atoms.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def import_state(self, state: Dict[str, Any]):
        """Import AtomSpace state from another node."""
        async with self._lock:
            for atom_data in state.get("atoms", {}).values():
                atom = Atom.from_dict(atom_data)
                await self.add_atom(atom)
    
    def __len__(self) -> int:
        """Return the number of atoms in the AtomSpace."""
        return len(self.atoms)


class DistributedAtomSpace:
    """
    Manages multiple AtomSpace instances across distributed nodes.
    Provides synchronization and coordination.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_space = AtomSpace()
        self.peer_nodes: Dict[str, str] = {}  # node_id -> endpoint
        self._sync_interval = 60.0  # seconds
    
    async def sync_with_peer(self, peer_id: str):
        """Synchronize knowledge with a peer node."""
        # In a real implementation, this would use network protocols
        # For now, this is a placeholder for the synchronization logic
        pass
    
    async def broadcast_atom(self, atom: Atom):
        """Broadcast a new atom to all peer nodes."""
        atom_id = await self.local_space.add_atom(atom)
        # In real implementation, send to peers via network
        return atom_id
    
    async def query_distributed(self, query: str) -> List[Atom]:
        """Query across the distributed AtomSpace network."""
        local_results = await self.local_space.find_atoms_by_name(query)
        # In real implementation, query peers and merge results
        return local_results
