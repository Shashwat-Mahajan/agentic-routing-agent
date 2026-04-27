from dotenv import load_dotenv
load_dotenv()
import os

from agno.agent import Agent
from agno.models.groq import Groq

# ------------------ Tool ------------------
def inventory_tool(product_name: str):
    data = {
        "iPhone 15": {"stock": 2},
        "AirPods Pro": {"stock": 0},
        "MacBook Air M3": {"stock": 5}
    }
    return data.get(product_name, None)

# ------------------ Agent ------------------
inventory_agent = Agent(
    name="Inventory Agent",
    model=Groq(
        id="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    ),
    tools=[inventory_tool],
    instructions=[
        """
        You are an Inventory Assistant.

        - Only answer inventory-related queries.
        - Otherwise say: "Sorry, I can’t assist with that."

        - Always call inventory_tool with exact product name.
        - Do not guess anything.

        Response format:
        - If found:
          <Product> is <In Stock / Out of Stock>. Available quantity: <number>.
        - If not found:
          The product is not available in our inventory.

        Keep response short.
        """
    ]
)

# ------------------ Run ------------------
inventory_agent.print_response(
    "Is AirPods Pro available?",
    stream=False
)