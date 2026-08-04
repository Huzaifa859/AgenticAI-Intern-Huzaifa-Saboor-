"""
Custom MCP server.

Exposes:
  - 1 resource : app://status                 -> app metadata / health info
  - 1 tool     : search_knowledge_base(query)  -> search a small in-memory KB
"""

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoApp")

KNOWLEDGE_BASE = [
    {"id": "kb1", "topic": "refund policy",   "text": "Refunds are available within 30 days of purchase with a valid receipt."},
    {"id": "kb2", "topic": "shipping",        "text": "Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days."},
    {"id": "kb3", "topic": "account password","text": "To reset your password, go to Settings > Security > Reset Password."},
    {"id": "kb4", "topic": "support hours",   "text": "Customer support is available Monday-Friday, 9am-6pm EST."},
    {"id": "kb5", "topic": "warranty",        "text": "All products include a 1-year limited warranty covering manufacturing defects."},
]


@mcp.resource("app://status")
def app_status() -> str:
    """App resource: current application status and metadata, as JSON."""
    return json.dumps(
        {
            "app_name": "DemoApp",
            "version": "1.0.0",
            "status": "healthy",
            "kb_entries": len(KNOWLEDGE_BASE),
        },
        indent=2,
    )


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for entries relevant to the query string."""
    words = query.lower().split()
    matches = [
        f"[{item['topic']}] {item['text']}"
        for item in KNOWLEDGE_BASE
        if any(w in item["topic"] or w in item["text"].lower() for w in words)
    ]
    return "\n".join(matches) if matches else "No matching knowledge base entries found."


if __name__ == "__main__":
    mcp.run()
