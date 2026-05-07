from agent_framework.ollama import OllamaChatClient

def create_agent(name: str, instructions: str, tools) -> OllamaChatClient:
    """Create an Ollama chat agent with the given name and instructions."""
    return OllamaChatClient(env_file_path = ".env").create_agent(
        name=name,
        instructions=instructions,
        tools=tools,
    )

def create_chat_client() -> OllamaChatClient:
    """Create a basic Ollama chat client."""
    return OllamaChatClient(env_file_path = ".env")
