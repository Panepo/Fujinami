import asyncio
from python.ollamaAgent import create_agent
from mcps.calculator import calculator_server

async def main() -> None:
    print("=== Interactive Ollama Chat Client ===")
    print("Type your questions or 'quit' to exit\n")

    try:
        async with (
            calculator_server,
            create_agent(
                name = "MathAgent",
                instructions = (
                    "You are a helpful math assistant. "
                    "Use the calculator tool for any math calculations."
                ),
                tools=[],
            ) as agent,
        ):
            while True:
                try:
                    user_input = input("You: ").strip()

                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break

                    if not user_input:
                        continue

                    result = await agent.run(user_input, tools=calculator_server)
                    print(f"Agent: {result}\n")

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}\n")
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
