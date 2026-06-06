import os
import sys
import logging
import requests
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Suppress LangChain info messages
logging.getLogger("langchain").setLevel(logging.ERROR)

# Load the Groq API key from the environment instead of hardcoding it.
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def create_daily_thought() -> str:
    """Generate an inspirational daily thought using the LLM."""
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_KEY)
    response = llm.invoke("Generate a short, unique inspirational daily thought in one or two sentences.")
    return response.content

def get_post(post_id: int) -> str:
    """Fetch a post by its ID from JSONPlaceholder and return its details."""
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    result = (
        f"Post ID : {data['id']}\n"
        f"User ID : {data['userId']}\n"
        f"Title   : {data['title']}\n"
        f"Body    : {data['body']}"
    )
    print(result)
    return result

agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[get_weather, create_daily_thought, get_post],
    system_prompt="You are a helpful assistant. Make sure that you only respond with whatever is coming as input to the agent, and do not add any extra commentary or explanation.",
)

if len(sys.argv) < 2:
    print("Usage: python dynamic_agent_input.py \"<your message>\"")
    sys.exit(1)

user_input = " ".join(sys.argv[1:])
result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
print(result["messages"][-1].content)