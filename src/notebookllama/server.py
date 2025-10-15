import os
import sys
from querying import query_index
from processing import process_file
from mindmap import get_mind_map
from fastmcp import FastMCP
from typing import List, Union, Literal, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from notebookllama.swarm_integration import (
        get_swarm_processor,
        SwarmConfig,
    )
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False


mcp: FastMCP = FastMCP(name="MCP For NotebookLM")


@mcp.tool(
    name="process_file_tool",
    description="This tool is useful to process files and produce summaries, question-answers and highlights.",
)
async def process_file_tool(
    filename: str,
) -> Union[str, Literal["Sorry, your file could not be processed."]]:
    notebook_model, text = await process_file(filename=filename)
    if notebook_model is None:
        return "Sorry, your file could not be processed."
    if text is None:
        text = ""
    return notebook_model + "\n%separator%\n" + text


@mcp.tool(name="get_mind_map_tool", description="This tool is useful to get a mind ")
async def get_mind_map_tool(
    summary: str, highlights: List[str]
) -> Union[str, Literal["Sorry, mind map creation failed."]]:
    mind_map_fl = await get_mind_map(summary=summary, highlights=highlights)
    if mind_map_fl is None:
        return "Sorry, mind map creation failed."
    return mind_map_fl


@mcp.tool(name="query_index_tool", description="Query a LlamaCloud index.")
async def query_index_tool(question: str) -> str:
    response = await query_index(question=question)
    if response is None:
        return "Sorry, I was unable to find an answer to your question."
    return response


if SWARM_AVAILABLE:
    @mcp.tool(
        name="query_swarm_tool",
        description="Query the distributed RAGSwarm network using swarm intelligence. "
        "Multiple specialized agents collaborate to retrieve, reason, synthesize, and validate answers.",
    )
    async def query_swarm_tool(query: str) -> str:
        """Query the RAGSwarm network for distributed knowledge retrieval and reasoning."""
        try:
            processor = get_swarm_processor()
            result = await processor.query_swarm(query)
            
            # Format the response
            response = f"## RAGSwarm Response\n\n**Query:** {query}\n\n"
            
            if result.get("synthesis", {}).get("synthesized_content"):
                response += f"**Answer:** {result['synthesis']['synthesized_content']}\n\n"
            
            if result.get("validation", {}).get("is_valid"):
                confidence = result["validation"].get("confidence", 0.0)
                response += f"**Validation:** ✓ Validated (confidence: {confidence:.2f})\n\n"
            
            response += f"**Agents Used:** {result.get('agents_used', 0)}\n"
            
            return response
        except Exception as e:
            return f"Error querying swarm: {str(e)}"


    @mcp.tool(
        name="get_swarm_status_tool",
        description="Get the current status of the RAGSwarm network including agent counts and metrics.",
    )
    async def get_swarm_status_tool() -> str:
        """Get status and metrics of the RAGSwarm network."""
        try:
            processor = get_swarm_processor()
            metrics = await processor.get_swarm_metrics()
            
            status = "## RAGSwarm Status\n\n"
            status += f"**Total Agents:** {metrics['total_agents']}\n"
            status += f"**Knowledge Base Size:** {metrics['knowledge_base_size']} atoms\n\n"
            
            status += "**Agents by Role:**\n"
            for role, count in metrics.get("agents_by_role", {}).items():
                status += f"- {role.capitalize()}: {count}\n"
            
            status += f"\n**Average Load:** {metrics.get('average_agent_load', 0.0):.2f}\n"
            status += f"**Max Load:** {metrics.get('max_agent_load', 0.0):.2f}\n"
            
            return status
        except Exception as e:
            return f"Error getting swarm status: {str(e)}"


    @mcp.tool(
        name="initialize_swarm_tool",
        description="Initialize the RAGSwarm network with specified agent counts. "
        "Use this before processing documents with the swarm.",
    )
    async def initialize_swarm_tool(
        num_retrievers: int = 3,
        num_reasoners: int = 2,
        num_synthesizers: int = 1,
        num_validators: int = 1,
    ) -> str:
        """Initialize the RAGSwarm with custom agent configuration."""
        try:
            processor = get_swarm_processor()
            await processor.initialize_swarm(
                num_retrievers=num_retrievers,
                num_reasoners=num_reasoners,
                num_synthesizers=num_synthesizers,
                num_validators=num_validators,
            )
            
            return (
                f"RAGSwarm initialized successfully with:\n"
                f"- {num_retrievers} retriever(s)\n"
                f"- {num_reasoners} reasoner(s)\n"
                f"- {num_synthesizers} synthesizer(s)\n"
                f"- {num_validators} validator(s)"
            )
        except Exception as e:
            return f"Error initializing swarm: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
