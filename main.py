import os
import random
from typing import cast
from dotenv import load_dotenv
import chainlit as cl

# Agents SDK imports
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool
)
from agents.run import RunConfig

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# === Gemini-compatible client ===
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Tool Functions ===
@function_tool
def roll_dice(sides: int = 20) -> int:
    """Roll a dice with a given number of sides (default 20)."""
    return random.randint(1, sides)

@function_tool
def generate_event(context: str) -> str:
    """Generate a random event based on the location context."""
    events = {
        "forest": [
            "You hear rustling in the bushes. A goblin appears!",
            "You find an ancient tree with glowing runes.",
            "A traveling merchant offers you a mysterious potion."
        ],
        "dungeon": [
            "A trap triggers beneath your feet!",
            "A skeleton warrior blocks your path.",
            "You discover a chest filled with gold... or is it a mimic?"
        ],
        "village": [
            "A child runs up to you, asking for help.",
            "The blacksmith offers to upgrade your weapon.",
            "You overhear talk of a dragon nearby."
        ]
    }
    return random.choice(events.get(context.lower(), ["Nothing unusual happens..."]))

# === Specialist Agents ===
NarratorAgent = Agent(
    name="NarratorAgent",
    instructions="""
You are the main narrator of a fantasy adventure game. 
Guide the player through a story using rich descriptions. 
Always ask what they'd like to do next.
"""
)

MonsterAgent = Agent(
    name="MonsterAgent",
    instructions="""
You control monster encounters during combat. 
Ask the player to choose an action (attack, defend, run).
Roll a 20-sided dice to determine the outcome. 
Describe the result using the dice roll.
""",
    tools=[roll_dice]  # ✅ FIXED: list instead of dict
)

ItemAgent = Agent(
    name="ItemAgent",
    instructions="""
You manage item discovery and rewards. 
When the player explores a forest, dungeon, or village, 
use the event generator tool to describe what they find.
""",
    tools=[generate_event]  # ✅ FIXED: list instead of dict
)

# === Triage Agent ===
GameTriageAgent = Agent(
    name="GameTriageAgent",
    instructions="""
You are the adventure game master. 
Based on the player's message, decide which expert to hand off to:

- If the player mentions monsters, battle, attack, or combat → hand off to MonsterAgent
- If they mention chests, loot, items, or rewards → hand off to ItemAgent
- Otherwise, continue the main narration yourself

Always briefly explain any handoff before routing the message.
""",
    handoffs=[MonsterAgent, ItemAgent]
)

# === Chainlit Start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    await cl.Message(
        content=(
            "🧙 Welcome, adventurer! Your quest begins now...\n\n"
            "Tell me what you'd like to do — explore a forest, enter a dungeon, or visit a village?"
        )
    ).send()

# === Message Handling ===
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    try:
        # Use triage agent to handle + handoff
        result = Runner.run_streamed(
            GameTriageAgent, history, run_config=cast(RunConfig, config)
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, "delta"):
                await msg.stream_token(event.data.delta)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", history)

    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")
        print(f"Error: {e}")
