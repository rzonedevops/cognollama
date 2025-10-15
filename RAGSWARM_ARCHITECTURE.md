# RAGSwarm Architecture

## Overview

RAGSwarm is a distributed Retrieval-Augmented Generation system inspired by OpenCog's hypergraph knowledge representation and swarm intelligence principles. It enables collaborative, distributed processing of documents and queries using multiple specialized agents.

## Core Components

### 1. AtomSpace (`opencog_atomspace.py`)

The AtomSpace provides an OpenCog-inspired hypergraph database for knowledge representation.

#### Atom Structure

```python
@dataclass
class Atom:
    atom_type: AtomType        # Type of atom (NODE, LINK, CONCEPT, etc.)
    name: str                  # Unique identifier
    truth_value: float         # Confidence/truth value (0-1)
    attention_value: float     # Attention/importance (0-1)
    outgoing: List[str]        # Links to other atoms
    metadata: Dict[str, Any]   # Additional data
    timestamp: str             # Creation timestamp
```

#### Key Features

- **Hypergraph Storage**: Knowledge stored as interconnected atoms
- **Attention Mechanism**: Atoms have attention values that affect processing priority
- **Activation Spreading**: Propagates activation through the network to find related concepts
- **Truth Values**: Probabilistic logic for handling uncertainty
- **Distributed Sync**: State export/import for distributed synchronization

#### AtomTypes

- `NODE`: Basic node in the hypergraph
- `LINK`: Connection between nodes
- `CONCEPT`: Abstract concept or idea
- `PREDICATE`: Logical predicate
- `DOCUMENT`: Document content
- `QUERY`: User query
- `ANSWER`: Generated answer

### 2. RAGSwarm Agents (`ragswarm.py`)

Multiple specialized agents collaborate to process queries and documents.

#### Agent Roles

```python
class AgentRole(Enum):
    RETRIEVER = "retriever"        # Information retrieval
    REASONER = "reasoner"          # Logical reasoning
    SYNTHESIZER = "synthesizer"    # Information synthesis
    VALIDATOR = "validator"        # Response validation
    COORDINATOR = "coordinator"    # Activity coordination
```

#### Agent Lifecycle

1. **Idle**: Waiting for tasks
2. **Processing**: Actively working on a task
3. **Waiting**: Waiting for other agents
4. **Collaborating**: Working with other agents
5. **Failed**: Error state

#### Message Passing

Agents communicate via structured messages:

```python
@dataclass
class SwarmMessage:
    sender_id: str              # Message sender
    receiver_id: Optional[str]  # Target agent (None = broadcast)
    message_type: str           # Type of message
    content: Dict[str, Any]     # Message payload
    timestamp: str              # When sent
    priority: float             # Priority (0-1)
```

### 3. Swarm Coordination

#### SwarmCoordinator

Manages agent collaboration and message routing:

- **Agent Registry**: Tracks all active agents
- **Role-Based Selection**: Chooses best agent for each task
- **Load Balancing**: Distributes work based on agent load
- **Message Routing**: Delivers messages to appropriate agents

#### Query Processing Flow

```
User Query
    ↓
[Coordinator]
    ↓
[Multiple Retrievers] → Parallel retrieval
    ↓
[Reasoner] → Logical reasoning
    ↓
[Synthesizer] → Combine results
    ↓
[Validator] → Verify accuracy
    ↓
Return Response
```

### 4. Integration Layer (`swarm_integration.py`)

Bridges RAGSwarm with NotebookLlaMa workflow.

#### SwarmNotebookProcessor

Main interface for using RAGSwarm:

```python
processor = SwarmNotebookProcessor(node_id="primary")

# Initialize with custom agent counts
await processor.initialize_swarm(
    num_retrievers=3,
    num_reasoners=2,
    num_synthesizers=1,
    num_validators=1,
)

# Process document
result = await processor.process_document_with_swarm(
    document_id="doc123",
    content="Document content...",
    metadata={"source": "example"},
)

# Query the swarm
answer = await processor.query_swarm("What is machine learning?")
```

## Architecture Diagrams

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NotebookLlaMa Application                 │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │  Streamlit  │  │  Workflow   │  │  MCP Server      │   │
│  │     UI      │  │   Engine    │  │  (server.py)     │   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘   │
│         │                 │                   │              │
│         └─────────────────┴───────────────────┘              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAGSwarm Integration Layer                  │
│              (swarm_integration.py)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         SwarmNotebookProcessor                        │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAGSwarm Orchestrator                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Swarm Coordinator                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐            │  │
│  │  │Agent Pool│ │Load      │ │Message    │            │  │
│  │  │Registry  │ │Balancer  │ │Router     │            │  │
│  │  └──────────┘ └──────────┘ └───────────┘            │  │
│  └────────────────────┬─────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Retriever    │ │  Reasoner    │ │ Synthesizer  │
│  Agents      │ │   Agents     │ │   Agents     │
│              │ │              │ │              │
│  [Agent 1]   │ │  [Agent 1]   │ │  [Agent 1]   │
│  [Agent 2]   │ │  [Agent 2]   │ │              │
│  [Agent 3]   │ │              │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Distributed AtomSpace                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Local AtomSpace (Hypergraph)                  │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │  │
│  │  │ Atom 1 │──│ Atom 2 │──│ Atom 3 │──│ Atom N │    │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │  │
│  │                                                       │  │
│  │  Indices: [Type Index] [Name Index] [Attention]     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Peer Synchronization (for distributed mode)                │
│  [Node 1] ←→ [Node 2] ←→ [Node 3]                         │
└─────────────────────────────────────────────────────────────┘
```

### Query Processing Flow

```
User Query: "What is machine learning?"
         │
         ▼
┌────────────────────┐
│   Coordinator      │
│  - Receives query  │
│  - Plans execution │
└─────────┬──────────┘
          │
          ├─────────────────────────────┐
          │                             │
          ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│  Retriever 1     │          │  Retriever 2     │
│  - Search "ML"   │          │  - Search "ML"   │
│  - Find Atom A   │          │  - Find Atom B   │
└────────┬─────────┘          └────────┬─────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  Results Merger   │
              │  - Combine A + B  │
              └─────────┬─────────┘
                        ▼
              ┌──────────────────┐
              │    Reasoner      │
              │  - Analyze data  │
              │  - Draw insights │
              └─────────┬─────────┘
                        ▼
              ┌──────────────────┐
              │  Synthesizer     │
              │  - Generate text │
              │  - Format answer │
              └─────────┬─────────┘
                        ▼
              ┌──────────────────┐
              │   Validator      │
              │  - Check facts   │
              │  - Verify logic  │
              └─────────┬─────────┘
                        ▼
                   Response
```

### Activation Spreading

```
Initial Query Atom
      │
      ▼
  [Atom: "ML"]
   attention: 1.0
      │
      ├─────────┬─────────┐
      ▼         ▼         ▼
  [Concept]  [Doc]   [Related]
    +0.3      +0.3      +0.3
      │
      ▼
  [Connected Atoms]
   attention boost
      │
      ▼
  Return top N by attention
```

## Design Principles

### 1. Modularity

Each component is independent and can be extended:
- New agent roles can be added
- AtomSpace can use different backends
- Coordination strategies are pluggable

### 2. Scalability

Horizontal scaling through:
- Multiple agents per role
- Distributed AtomSpace nodes
- Load balancing across agents

### 3. Robustness

Fault tolerance via:
- Agent redundancy
- Graceful degradation
- State recovery from peers

### 4. Transparency

Observability through:
- Agent metrics
- Message tracing
- Attention tracking
- Performance monitoring

## Configuration

### Environment Variables

```bash
# Node Configuration
SWARM_NODE_ID="primary"              # Unique node identifier

# Agent Configuration
SWARM_NUM_RETRIEVERS="3"             # Retriever agents
SWARM_NUM_REASONERS="2"              # Reasoner agents
SWARM_NUM_SYNTHESIZERS="1"           # Synthesizer agents
SWARM_NUM_VALIDATORS="1"             # Validator agents

# Distributed Configuration
SWARM_ENABLE_DISTRIBUTED="false"     # Enable distributed mode
SWARM_SYNC_INTERVAL="60"             # Peer sync interval (seconds)
```

## Performance Considerations

### Optimization Strategies

1. **Agent Pool Sizing**: Balance between parallelism and resource usage
2. **Attention Tuning**: Optimize attention spreading parameters
3. **Caching**: Cache frequent queries and atom lookups
4. **Batch Processing**: Group similar operations
5. **Load Balancing**: Dynamic agent selection based on load

### Metrics to Monitor

- **Agent Load**: Current workload per agent
- **Query Latency**: Time to process queries
- **Knowledge Base Size**: Number of atoms
- **Attention Distribution**: Spread of attention values
- **Agent Utilization**: Percentage of time agents are busy

## Future Enhancements

### Short Term

1. Persistent AtomSpace storage (database backend)
2. Enhanced reasoning algorithms
3. Advanced attention spreading
4. Performance optimizations

### Long Term

1. True distributed operation across nodes
2. Learning from user feedback
3. Automatic agent scaling
4. Advanced inference capabilities
5. Integration with external knowledge bases

## References

- [OpenCog Framework](https://opencog.org/)
- [OpenCog AtomSpace](https://wiki.opencog.org/w/AtomSpace)
- [Swarm Intelligence](https://en.wikipedia.org/wiki/Swarm_intelligence)
- [Hypergraph Databases](https://en.wikipedia.org/wiki/Hypergraph)
- [RAG (Retrieval-Augmented Generation)](https://arxiv.org/abs/2005.11401)
