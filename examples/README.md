# RAGSwarm Examples

This directory contains example scripts demonstrating how to use the OpenCog-inspired RAGSwarm distributed network.

## Examples

### `ragswarm_example.py`

A comprehensive example showing:
- Initializing the RAGSwarm network
- Adding knowledge atoms to the distributed AtomSpace
- Processing documents with the swarm
- Querying the swarm with questions
- Spreading activation through the knowledge network
- Inspecting swarm metrics and status

**Run the example:**

```bash
cd /path/to/cognollama
python3 examples/ragswarm_example.py
```

**Note:** This example runs independently and doesn't require the full NotebookLlaMa stack to be running.

## Understanding the RAGSwarm

### Key Components

1. **AtomSpace**: OpenCog-inspired hypergraph for knowledge representation
   - Stores knowledge as interconnected atoms
   - Supports attention spreading and activation
   - Enables distributed knowledge sharing

2. **Swarm Agents**: Specialized agents that collaborate
   - **Retrievers** 🔍: Find relevant information
   - **Reasoners** 🧠: Perform logical reasoning
   - **Synthesizers** 🔄: Combine multiple sources
   - **Validators** ✅: Verify accuracy

3. **Coordinator**: Orchestrates agent collaboration
   - Routes messages between agents
   - Load balances across agents
   - Aggregates results

### Basic Usage Pattern

```python
from notebookllama.swarm_integration import SwarmNotebookProcessor

# Initialize
processor = SwarmNotebookProcessor(node_id="my_node")
await processor.initialize_swarm()

# Add knowledge
await processor.add_knowledge_atom(
    atom_type=AtomType.CONCEPT,
    name="example_concept",
    metadata={"info": "value"},
)

# Query
result = await processor.query_swarm("What is AI?")
print(result)

# Get metrics
metrics = await processor.get_swarm_metrics()
print(f"Total agents: {metrics['total_agents']}")
```

## Configuration

Configure the swarm behavior via environment variables in `.env`:

```bash
SWARM_NODE_ID="primary"              # Node identifier
SWARM_NUM_RETRIEVERS="3"             # Number of retriever agents
SWARM_NUM_REASONERS="2"              # Number of reasoner agents
SWARM_NUM_SYNTHESIZERS="1"           # Number of synthesizer agents
SWARM_NUM_VALIDATORS="1"             # Number of validator agents
SWARM_ENABLE_DISTRIBUTED="false"     # Enable distributed mode
SWARM_SYNC_INTERVAL="60"             # Sync interval (seconds)
```

## Further Reading

- [OpenCog AtomSpace](https://wiki.opencog.org/w/AtomSpace) - Original inspiration
- [Swarm Intelligence](https://en.wikipedia.org/wiki/Swarm_intelligence) - Theoretical background
- Main README.md - Full NotebookLlaMa documentation
