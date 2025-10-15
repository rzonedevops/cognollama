# RAGSwarm Quick Start Guide

## What is RAGSwarm?

RAGSwarm is an OpenCog-inspired distributed network for intelligent document processing and question answering. It uses multiple specialized AI agents working together like a swarm to provide better, more reliable answers.

## Quick Setup

### 1. Configure Environment

Add these lines to your `.env` file:

```bash
# RAGSwarm Configuration
SWARM_NODE_ID="primary"
SWARM_NUM_RETRIEVERS="3"
SWARM_NUM_REASONERS="2"
SWARM_NUM_SYNTHESIZERS="1"
SWARM_NUM_VALIDATORS="1"
```

### 2. Access RAGSwarm UI

After starting NotebookLlaMa:

```bash
streamlit run src/notebookllama/Home.py
```

Navigate to **"RAGSwarm Network"** in the sidebar.

### 3. Initialize the Swarm

In the RAGSwarm page:
1. Click "Initialize Swarm" 
2. Wait for confirmation
3. Start querying!

## Using RAGSwarm

### Via Streamlit UI

1. **Initialize**: Click "Initialize Swarm" button
2. **Query**: Type your question in the chat interface
3. **View Results**: See answer with validation confidence
4. **Check Status**: Expand "Swarm Status & Metrics" to see agent details

### Via Python Code

```python
from notebookllama.swarm_integration import get_swarm_processor

# Get processor
processor = get_swarm_processor()

# Initialize
await processor.initialize_swarm()

# Query
result = await processor.query_swarm("What is machine learning?")
print(result)
```

### Via MCP Tools

From the MCP server:
- `initialize_swarm_tool()` - Initialize the swarm
- `query_swarm_tool(query="...")` - Ask a question
- `get_swarm_status_tool()` - Get swarm status

## Key Concepts

### Agents

- **Retrievers** 🔍: Find relevant information
- **Reasoners** 🧠: Analyze and reason about data
- **Synthesizers** 🔄: Combine information into answers
- **Validators** ✅: Check answer accuracy

### Knowledge Base (AtomSpace)

A hypergraph database that stores:
- Documents
- Concepts
- Relationships
- Queries and answers

### How It Works

```
Your Question
    ↓
[Retrievers find relevant info]
    ↓
[Reasoner analyzes it]
    ↓
[Synthesizer creates answer]
    ↓
[Validator checks accuracy]
    ↓
Your Answer ✓
```

## Examples

### Example 1: Simple Query

```python
result = await processor.query_swarm("What is AI?")

# Result includes:
# - synthesis: The answer
# - validation: Confidence score
# - agents_used: Number of agents that collaborated
```

### Example 2: Add Knowledge

```python
from notebookllama.opencog_atomspace import AtomType

await processor.add_knowledge_atom(
    atom_type=AtomType.CONCEPT,
    name="machine_learning",
    metadata={"description": "A subset of AI"},
    truth_value=0.95,
)
```

### Example 3: Process Document

```python
result = await processor.process_document_with_swarm(
    document_id="my_doc",
    content="Your document text here...",
    metadata={"source": "example"},
)
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_NODE_ID` | "primary" | Unique identifier for this node |
| `SWARM_NUM_RETRIEVERS` | 3 | Number of retriever agents |
| `SWARM_NUM_REASONERS` | 2 | Number of reasoner agents |
| `SWARM_NUM_SYNTHESIZERS` | 1 | Number of synthesizer agents |
| `SWARM_NUM_VALIDATORS` | 1 | Number of validator agents |
| `SWARM_ENABLE_DISTRIBUTED` | false | Enable multi-node operation |
| `SWARM_SYNC_INTERVAL` | 60 | Sync interval in seconds |

## Performance Tips

### For Best Performance

1. **More Retrievers**: Faster information lookup (3-5 recommended)
2. **Balanced Reasoners**: Better analysis (2-3 recommended)
3. **Single Synthesizer**: Usually sufficient
4. **Single Validator**: Usually sufficient

### For Complex Queries

Increase reasoners to 3-4 for better analysis.

### For Simple Queries

Use 2 retrievers, 1 reasoner, 1 synthesizer, 1 validator for efficiency.

## Monitoring

### Check Swarm Status

```python
metrics = await processor.get_swarm_metrics()

print(f"Total Agents: {metrics['total_agents']}")
print(f"Knowledge Base: {metrics['knowledge_base_size']} atoms")
print(f"Avg Load: {metrics['average_agent_load']:.2%}")
```

### In Streamlit UI

Click "Refresh Status" in the "Swarm Status & Metrics" section.

## Troubleshooting

### Swarm won't initialize

- Check that all dependencies are installed
- Verify environment variables are set
- Look for error messages in console

### Queries are slow

- Increase number of retriever agents
- Check agent load in status metrics
- Reduce knowledge base size if very large

### Getting generic answers

- Add more specific knowledge atoms
- Increase truth values for important concepts
- Use more reasoner agents

## Next Steps

1. **Try the example**: Run `python examples/ragswarm_example.py`
2. **Read architecture**: See `RAGSWARM_ARCHITECTURE.md`
3. **Explore code**: Check `src/notebookllama/ragswarm.py`
4. **Add knowledge**: Build your custom knowledge base

## Help & Resources

- **Architecture**: `RAGSWARM_ARCHITECTURE.md`
- **Examples**: `examples/ragswarm_example.py`
- **Tests**: `tests/test_ragswarm.py`
- **Main README**: Project root `README.md`

## Common Use Cases

### Use Case 1: Document Q&A

```python
# Add document
await processor.orchestrator.add_document_to_knowledge(
    document_id="report_2024",
    content="Your report content...",
)

# Query it
result = await processor.query_swarm("What are the key findings?")
```

### Use Case 2: Knowledge Graph

```python
# Add interconnected concepts
concept1_id = await processor.add_knowledge_atom(
    atom_type=AtomType.CONCEPT,
    name="python",
)

concept2_id = await processor.add_knowledge_atom(
    atom_type=AtomType.CONCEPT,
    name="programming",
)

# Link them (in production, add relationship atoms)
```

### Use Case 3: Fact Verification

```python
# Query with validation
result = await processor.query_swarm("Is Python a programming language?")

# Check validation
if result['validation']['is_valid']:
    confidence = result['validation']['confidence']
    print(f"Answer validated with {confidence:.2%} confidence")
```

---

**Ready to get started?** Initialize your swarm and start asking questions! 🚀
