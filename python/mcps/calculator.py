from agent_framework import MCPStdioTool

calculator_server = MCPStdioTool(
    name="calculator",
    command="uvx",
    args=["mcp-server-calculator"]
)
