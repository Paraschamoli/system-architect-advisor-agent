"""System-Architect-Advisor-Agent - A Bindu Agent."""

import argparse
import asyncio
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, cast

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.mem0 import Mem0Tools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv
from openai import OpenAI

# Set up logging
_logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Error messages
_MISSING_API_KEY_ERROR = "OPENROUTER_API_KEY is required. Set environment variable or use --api-key argument."
_REASONING_CLIENT_NOT_INITIALIZED_ERROR = "Reasoning client not initialized"
_AGENT_NOT_INITIALIZED_ERROR = "Agent not initialized"

# Global instances
reasoning_client: OpenAI | None = None
drafting_agent: Agent | None = None

# Config variables
openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
mem0_api_key: str | None = os.getenv("MEM0_API_KEY")

# Model IDs (Defaults via OpenRouter)
REASONING_MODEL_ID = "deepseek/deepseek-r1"  # High reasoning capability
DRAFTING_MODEL_ID = "anthropic/claude-3.5-sonnet"  # High drafting capability
# Alternative Free/Cheap models you can swap in env vars:
# REASONING: "deepseek/deepseek-r1:free" (if available)
# DRAFTING: "google/gemini-2.0-flash-exp:free"

_initialized = False
_init_lock = asyncio.Lock()


def load_config() -> dict[str, Any]:
    """Load agent config from `agent_config.json` or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    return {
        "name": "system-architect-advisor-agent",
        "description": "AI System Architect Advisor for technical design",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


def _get_deepseek_system_prompt() -> str:
    """Return the strict JSON schema prompt for DeepSeek."""
    return """You are an expert software architect. Analyze the user's requirements and provide structured reasoning.

IMPORTANT: Your response must be a valid JSON object matching this schema exactly.
Do not include markdown formatting like ```json ... ``` inside the final JSON block.

Schema:
{
    "architecture_decision": {
        "pattern": "one of: microservices|monolithic|serverless|event_driven|layered",
        "rationale": "string",
        "trade_offs": {"pros": ["list"], "cons": ["list"]},
        "estimated_cost": {"implementation": float, "maintenance": float}
    },
    "infrastructure_resources": [{
        "resource_type": "string",
        "specifications": {"key": "value"},
        "scaling_policy": {"key": "value"},
        "estimated_cost": float
    }],
    "security_measures": [{
        "measure_type": "string",
        "priority": "integer 1-5",
        "compliance_standards": ["list"],
        "data_classification": "string"
    }],
    "database_choice": "string",
    "tech_stack": {
        "frontend": "string",
        "backend": "string",
        "devops": ["list"]
    },
    "risk_assessment": {"risk": "mitigation"},
    "implementation_roadmap": ["steps"]
}
"""


async def initialize_agent() -> None:
    """Initialize the OpenRouter client and Agno Agent."""
    global reasoning_client, drafting_agent, openrouter_api_key

    # Ensure API Key is set
    if not openrouter_api_key:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_api_key:
        raise ValueError(_MISSING_API_KEY_ERROR)

    # 1. Initialize Reasoning Client (OpenAI SDK pointing to OpenRouter)
    # We use this directly to handle R1's specific reasoning tokens better
    print(f"🔧 Initializing Reasoning Client ({REASONING_MODEL_ID})...")
    reasoning_client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")

    # 2. Initialize Drafting Agent (Agno + OpenRouter)
    print(f"🔧 Initializing Drafting Agent ({DRAFTING_MODEL_ID})...")

    # Optional tools
    tools = []
    if mem0_api_key:
        print("🧠 Mem0 memory enabled")
        tools.append(Mem0Tools(api_key=mem0_api_key))

    drafting_agent = Agent(
        name="System Architect Drafter",
        model=OpenRouter(
            id=DRAFTING_MODEL_ID,
            api_key=openrouter_api_key,
        ),
        tools=tools,
        description="You are a Technical Writer and Systems Architect.",
        instructions=[
            "Analyze the provided JSON technical specification and reasoning.",
            "Transform the raw JSON data into a beautiful, professional Markdown technical report.",
            "Ensure every architectural decision is explained using the provided reasoning context.",
            "Create Mermaid.js diagrams for the system architecture where appropriate.",
            "Do not output raw JSON, only the formatted report.",
        ],
        markdown=True,
    )
    print("✅ System Architect Agent initialized")


async def get_reasoning_analysis(user_input: str) -> tuple[str, str]:
    """Call OpenRouter (DeepSeek R1) to get reasoning and JSON spec."""
    global reasoning_client

    # Ensure reasoning client is initialized
    if not reasoning_client:
        raise RuntimeError(_REASONING_CLIENT_NOT_INITIALIZED_ERROR)

    try:
        # Run in executor to avoid blocking async loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: reasoning_client.chat.completions.create(  # type: ignore[union-attr]
                model=REASONING_MODEL_ID,
                messages=[
                    {"role": "system", "content": _get_deepseek_system_prompt()},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=4000,
                # OpenRouter often sends reasoning in 'reasoning' field or inside <think> tags
                extra_body={"include_reasoning": True},
            ),
        )

        message = response.choices[0].message
        content = message.content or ""

        # Extract Reasoning
        # 1. Check native reasoning field (some providers supported by OpenRouter use this)
        reasoning = getattr(message, "reasoning", "") or getattr(message, "reasoning_content", "")

        # 2. Check for <think> tags if native field is empty
        if not reasoning and "</think>" in content:
            match = re.search(r"</think>(.*?)</think>", content, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                # Remove the think block from content to get just the JSON
                content = re.sub(r"</think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # clean content of potential markdown code blocks to get raw JSON
        content = content.replace("```json", "").replace("```", "").strip()

    except Exception as e:
        print(f"❌ Reasoning API Error: {e}")
        raise
    else:
        return reasoning, content


async def run_architect_flow(messages: list[dict[str, str]]) -> Any:
    """Orchestrate the Architect flow."""
    global drafting_agent

    if not drafting_agent:
        raise RuntimeError(_AGENT_NOT_INITIALIZED_ERROR)

    # Extract the last user message
    user_input = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)

    if not user_input:
        return "Please provide project requirements."

    print("🤔 Analyzing requirements via OpenRouter...")

    # Step 1: Reasoning
    try:
        reasoning, json_spec = await get_reasoning_analysis(user_input)
    except Exception as e:
        return f"Error during reasoning phase: {e!s}"

    # Step 2: Drafting
    print("✍️  Drafting technical report...")
    prompt = f"""
    # Project Context
    User Query: {user_input}

    # Architect Reasoning
    {reasoning}

    # Technical Specification (JSON)
    {json_spec}

    # Task
    Generate a comprehensive technical design document based ONLY on the above information.
    Explain the 'Why' behind the choices using the reasoning provided.
    """

    # Run Agno Agent
    response = drafting_agent.run(prompt)

    # Combine Output
    final_output = f"{response.content}\n\n---\n\n<details><summary>🧠 View Architectural Reasoning</summary>\n\n{reasoning}\n\n</details>"

    return final_output


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            await initialize_agent()
            _initialized = True

    try:
        # Run the architect flow
        result = await run_architect_flow(messages)
    except Exception as e:
        traceback.print_exc()
        return f"An error occurred during architectural analysis: {e!s}"
    else:
        return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up System Architect Agent resources...")
    global reasoning_client, drafting_agent
    reasoning_client = None
    drafting_agent = None


def main():
    """Run the Agent."""
    global openrouter_api_key, mem0_api_key, REASONING_MODEL_ID, DRAFTING_MODEL_ID

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="System Architect Advisor Agent")

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    # Allow overriding models via CLI/Env
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default=os.getenv("REASONING_MODEL", "deepseek/deepseek-r1"),
        help="Model ID for reasoning phase",
    )
    parser.add_argument(
        "--drafting-model",
        type=str,
        default=os.getenv("DRAFTING_MODEL", "anthropic/claude-3.5-sonnet"),
        help="Model ID for drafting phase",
    )

    args = parser.parse_args()

    # Set globals
    openrouter_api_key = args.api_key
    mem0_api_key = args.mem0_key
    REASONING_MODEL_ID = args.reasoning_model
    DRAFTING_MODEL_ID = args.drafting_model

    if not openrouter_api_key:
        print("⚠️  Warning: OPENROUTER_API_KEY not found. Agent will fail if not provided in env.")

    print("🤖 System Architect Advisor")
    print(f"   Reasoning: {REASONING_MODEL_ID}")
    print(f"   Drafting:  {DRAFTING_MODEL_ID}")

    # Load configuration
    try:
        config = load_config()
    except Exception:
        config = {"deployment": {"url": "http://0.0.0.0:3773"}}

    try:
        # Bindufy and start the agent server
        print(f"🚀 Starting Bindu agent server on {config.get('deployment', {}).get('url', 'http://0.0.0.0:3773')}...")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
