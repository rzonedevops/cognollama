"""Tests for OpenCog-inspired AtomSpace implementation."""

import pytest
import asyncio
from src.notebookllama.opencog_atomspace import (
    Atom,
    AtomType,
    AtomSpace,
    DistributedAtomSpace,
)


@pytest.mark.asyncio
async def test_atom_creation():
    """Test creating an atom."""
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="test_concept",
        truth_value=0.9,
        attention_value=0.7,
    )
    
    assert atom.atom_type == AtomType.CONCEPT
    assert atom.name == "test_concept"
    assert atom.truth_value == 0.9
    assert atom.attention_value == 0.7
    assert atom.atom_id is not None


@pytest.mark.asyncio
async def test_atom_serialization():
    """Test atom serialization and deserialization."""
    atom = Atom(
        atom_type=AtomType.NODE,
        name="test_node",
        truth_value=0.8,
    )
    
    # Serialize
    atom_dict = atom.to_dict()
    assert atom_dict["name"] == "test_node"
    assert atom_dict["atom_type"] == "node"
    
    # Deserialize
    restored_atom = Atom.from_dict(atom_dict)
    assert restored_atom.name == atom.name
    assert restored_atom.atom_type == atom.atom_type
    assert restored_atom.truth_value == atom.truth_value


@pytest.mark.asyncio
async def test_atomspace_add_atom():
    """Test adding atoms to AtomSpace."""
    atomspace = AtomSpace()
    
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="knowledge",
    )
    
    atom_id = await atomspace.add_atom(atom)
    assert atom_id is not None
    
    retrieved = await atomspace.get_atom(atom_id)
    assert retrieved is not None
    assert retrieved.name == "knowledge"


@pytest.mark.asyncio
async def test_atomspace_find_by_type():
    """Test finding atoms by type."""
    atomspace = AtomSpace()
    
    # Add different types of atoms
    concept1 = Atom(atom_type=AtomType.CONCEPT, name="concept1")
    concept2 = Atom(atom_type=AtomType.CONCEPT, name="concept2")
    document = Atom(atom_type=AtomType.DOCUMENT, name="document1")
    
    await atomspace.add_atom(concept1)
    await atomspace.add_atom(concept2)
    await atomspace.add_atom(document)
    
    # Find concepts
    concepts = await atomspace.find_atoms_by_type(AtomType.CONCEPT)
    assert len(concepts) == 2
    
    # Find documents
    documents = await atomspace.find_atoms_by_type(AtomType.DOCUMENT)
    assert len(documents) == 1


@pytest.mark.asyncio
async def test_atomspace_find_by_name():
    """Test finding atoms by name."""
    atomspace = AtomSpace()
    
    atom1 = Atom(atom_type=AtomType.CONCEPT, name="test")
    atom2 = Atom(atom_type=AtomType.NODE, name="test")
    
    await atomspace.add_atom(atom1)
    await atomspace.add_atom(atom2)
    
    results = await atomspace.find_atoms_by_name("test")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_attention_update():
    """Test updating attention values."""
    atomspace = AtomSpace()
    
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="important_concept",
        attention_value=0.5,
    )
    
    atom_id = await atomspace.add_atom(atom)
    
    # Increase attention
    await atomspace.update_attention(atom_id, 0.3)
    updated_atom = await atomspace.get_atom(atom_id)
    assert updated_atom.attention_value == 0.8
    
    # Decrease attention
    await atomspace.update_attention(atom_id, -0.2)
    updated_atom = await atomspace.get_atom(atom_id)
    assert updated_atom.attention_value == 0.6


@pytest.mark.asyncio
async def test_spread_activation():
    """Test spreading activation through the network."""
    atomspace = AtomSpace()
    
    # Create a small network
    atom1 = Atom(atom_type=AtomType.CONCEPT, name="root")
    atom1_id = await atomspace.add_atom(atom1)
    
    atom2 = Atom(
        atom_type=AtomType.CONCEPT,
        name="child1",
        outgoing=[atom1_id],
    )
    atom2_id = await atomspace.add_atom(atom2)
    
    atom3 = Atom(
        atom_type=AtomType.CONCEPT,
        name="child2",
        outgoing=[atom1_id],
    )
    await atomspace.add_atom(atom3)
    
    # Spread activation
    activated = await atomspace.spread_activation(atom1_id, depth=1)
    
    assert len(activated) > 0
    assert any(a.name == "root" for a in activated)


@pytest.mark.asyncio
async def test_atomspace_export_import():
    """Test exporting and importing AtomSpace state."""
    atomspace1 = AtomSpace()
    
    # Add some atoms
    atom1 = Atom(atom_type=AtomType.CONCEPT, name="concept1")
    atom2 = Atom(atom_type=AtomType.DOCUMENT, name="document1")
    
    await atomspace1.add_atom(atom1)
    await atomspace1.add_atom(atom2)
    
    # Export state
    state = await atomspace1.export_state()
    assert "atoms" in state
    assert len(state["atoms"]) == 2
    
    # Import into new atomspace
    atomspace2 = AtomSpace()
    await atomspace2.import_state(state)
    
    assert len(atomspace2) == 2
    concepts = await atomspace2.find_atoms_by_type(AtomType.CONCEPT)
    assert len(concepts) == 1


@pytest.mark.asyncio
async def test_distributed_atomspace_creation():
    """Test creating a distributed AtomSpace."""
    dist_atomspace = DistributedAtomSpace(node_id="test_node")
    
    assert dist_atomspace.node_id == "test_node"
    assert dist_atomspace.local_space is not None
    assert isinstance(dist_atomspace.local_space, AtomSpace)


@pytest.mark.asyncio
async def test_distributed_broadcast():
    """Test broadcasting atoms in distributed network."""
    dist_atomspace = DistributedAtomSpace(node_id="node1")
    
    atom = Atom(
        atom_type=AtomType.CONCEPT,
        name="shared_knowledge",
    )
    
    atom_id = await dist_atomspace.broadcast_atom(atom)
    assert atom_id is not None
    
    # Verify it's in local space
    local_atom = await dist_atomspace.local_space.get_atom(atom_id)
    assert local_atom is not None
    assert local_atom.name == "shared_knowledge"
