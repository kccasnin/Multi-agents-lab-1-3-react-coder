"""
agent.py  ·  Lab 1.3 – ReACT Agent
────────────────────────────────────────────────────────────────────────────────
YOUR TASK
─────────
Implement the three functions marked with  TODO  below.  They wire together
the memory system (memory.py) and the tool layer (tools.py) into a working
ReACT loop.

The ReACT pattern structures LLM reasoning as a sequence of:
    Thought  → what the agent is planning to do next
    Action   → the tool to call (name + arguments)
    Observation → the tool's return value

The loop repeats until:
    • The LLM returns a  FINAL ANSWER:  marker (task complete), OR
    • The iteration counter reaches MAX_ITERATIONS (loop guard).

Files you will edit
───────────────────
    agent.py         ← THIS FILE.  Implement the three TODO functions.

Files provided (do not modify)
──────────────────────────────
    llm_client.py    – Helicone-routed OpenAI client + MODEL constant
    tools.py         – read_file, write_file, exec_python + dispatch_tool()
    memory.py        – SlidingWindowBuffer + SemanticMemory

Running the agent
─────────────────
    python agent.py

    The agent will attempt the default DEMO_TASK defined at the bottom of
    this file.  Swap it out with your own task to experiment.

Running the tests
─────────────────
    pytest tests/
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import client, MODEL
from memory import SemanticMemory, SlidingWindowBuffer
from tools import TOOL_SCHEMAS, dispatch_tool

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10   # Hard ceiling on agentic loop iterations.

SYSTEM_PROMPT = """You are a Coder Agent.  Your job is to complete coding tasks
by reasoning step-by-step and using the available tools.

## Response format

Every response MUST use EXACTLY one of the two formats below.

FORMAT A – when you need to call a tool:
    Thought: <your reasoning about what to do next>
    Action: <tool_name>
    Action Input: <valid JSON object with the tool's arguments>

FORMAT B – when the task is complete:
    Thought: <brief explanation of what you produced>
    FINAL ANSWER: <the deliverable: code, explanation, result, or a file path>

## Rules

- Always emit a Thought before every Action or FINAL ANSWER.
- Action Input must be a single-line valid JSON object.
- Never invent tool results; always wait for the Observation.
- If a tool returns an ERROR, try to fix the issue rather than repeating
  the same call.
- Stop and emit FINAL ANSWER as soon as the task is complete.
- You have a maximum of {max_iter} iterations.  Use them wisely.
""".format(max_iter=MAX_ITERATIONS)


# ── Output schema ─────────────────────────────────────────────────────────────

class AgentResult:
    """Structured result returned by run_agent()."""

    def __init__(
        self,
        answer: str,
        iterations: int,
        tool_calls: list[dict[str, Any]],
        stopped_early: bool,
    ) -> None:
        self.answer = answer          # The FINAL ANSWER text.
        self.iterations = iterations  # How many loop iterations ran.
        self.tool_calls = tool_calls  # List of {"tool": ..., "args": ..., "result": ...}.
        self.stopped_early = stopped_early  # True if MAX_ITERATIONS was reached.

    def __repr__(self) -> str:
        status = "MAX_ITER" if self.stopped_early else "DONE"
        return (
            f"AgentResult(status={status}, iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)})\n"
            f"ANSWER:\n{self.answer}"
        )


# ── Functions for students to implement ───────────────────────────────────────

def build_system_prompt(semantic_memory: SemanticMemory, task: str) -> str:
    """
    Construct the system prompt by injecting relevant long-term memories.
    This function retrieves up to three relevant memories associated with the
    specified task from the semantic memory. If relevant memories are found,
    they are formatted as a bulleted list and appended to the standard system
    prompt. If no memories are available, the original system prompt is returned unchanged.

    Parameters
    ----------
    semantic_memory : SemanticMemory
        The long-term vector store from which relevant memories are retrieved.
    task : str
        The user's task description, which is used as the retrieval query.

    Returns
    -------
    str
        The complete system prompt, optionally including any relevant memories
        formatted as a bulleted list.
    """
    # Retrieve up to three relevant memories
    relevant_memories = semantic_memory.retrieve(task, top_k=3)

    # If memories are found, format them as a bullet list
    if relevant_memories:
        memories_formatted = "\n## Relevant memories\n" + "\n- ".join(relevant_memories)
        return SYSTEM_PROMPT + memories_formatted

    # If no memories found, return the default SYSTEM_PROMPT
    return SYSTEM_PROMPT

def parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse a single LLM response into a structured dictionary.

    This function interprets the raw response from the language model, which
    can either indicate a tool call (FORMAT A) or a final answer (FORMAT B).
    It extracts the Thought, Action, and Action Input for FORMAT A, or
    the Thought and FINAL ANSWER for FORMAT B. Any parsing errors encountered
    during the extraction of Action Input are handled appropriately.

    Parameters
    ----------
    response_text : str
        The raw text response from the LLM's message content.

    Returns
    -------
    dict
        A structured dictionary containing:
        - 'thought': The reasoning behind the agent's next step.
        - Either 'final_answer' for FORMAT B or both 'action' and 'action_input' for FORMAT A.
        - If there's a parse error, an error message under the 'parse_error' key.
    """
    # Extract the Thought line
    thought = response_text.split("Thought:", 1)[1].strip()

    # Check for FINAL ANSWER
    if "FINAL ANSWER:" in response_text:
        final_answer = response_text.split("FINAL ANSWER:", 1)[1].strip()
        return {"thought": thought, "final_answer": final_answer}

    # Otherwise, extract Action and Action Input
    action_line = response_text.split("Action:", 1)[1].strip()
    action = action_line.split("\n", 1)[0].strip()
    action_input_raw = action_line.split("Action Input:", 1)[1].strip()
    action_input = action_input_raw.split("\n", 1)[0].strip()  # Handle newlines

    # Try to parse Action Input as JSON
    try:
        action_input_dict = json.loads(action_input)
    except json.JSONDecodeError as e:
        return {"thought": thought, "action": action, "parse_error": str(e)}

    return {"thought": thought, "action": action, "action_input": action_input_dict}

def run_agent(task: str) -> AgentResult:
    """
    Execute the full ReACT agentic loop for the given task.

    This function initializes necessary components (SlidingWindowBuffer and
    SemanticMemory), builds the system prompt, and adds the user's task. It
    enters a loop where it continuously calls the language model, processes
    the responses, and manages tool usage until a final answer is achieved
    or the maximum iteration limit is reached. Parsed results, tool calls,
    and observations are logged appropriately.

    Parameters
    ----------
    task : str
        A natural language description of the coding task that the agent is
        expected to complete.

    Returns
    -------
    AgentResult
        A structured result containing the final answer (if achieved), the
        total number of iterations, and a log of all tool calls made by
        the agent.
    """
    # Step 1: Initialize a SlidingWindowBuffer and a SemanticMemory.
    buffer = SlidingWindowBuffer()
    semantic_memory = SemanticMemory()
    # Step 2: Call build_system_prompt() and set the system prompt in the buffer.
    system_prompt = build_system_prompt(semantic_memory, task)
    buffer.set_system(system_prompt)
    # Step 3: Add the user task as the first message.
    buffer.add("user", task)

    tool_calls = []  # to record tool calls
    # Step 4: Enter the loop (up to MAX_ITERATIONS).
    for iteration in range(MAX_ITERATIONS):
        # Step 4a: Call the LLM.
        response = client.chat.completions.create(
            messages=buffer.messages(),
            tools=TOOL_SCHEMAS,
            tool_choice="none"
        )

        # Step 4b: Extract the assistant's text.
        assistant_text = response.choices[0].message.content

        # Step 4c: Add the assistant message to the buffer.
        buffer.add("assistant", assistant_text)

        # Step 4d: Parse the response.
        parsed_response = parse_llm_response(assistant_text)

        # Step 4e: Check if it contains a final answer.
        if "final_answer" in parsed_response:
            final_answer = parsed_response["final_answer"]

            # Store the answer in semantic memory.
            semantic_memory.store("Task: " + task + "\nAnswer: " + final_answer)
            return AgentResult(final_answer, iteration + 1, tool_calls, stopped_early=False)

        # Step 4f: Check for parse errors.
        elif "parse_error" in parsed_response:
            parse_error = parsed_response["parse_error"]
            observation = f"OBSERVATION: Parse error – {parse_error}"
            buffer.add("user", observation)
            continue

        # Step 4g: Handle normal tool call.
        action = parsed_response["action"]
        action_input = parsed_response["action_input"]
        tool_result = dispatch_tool(action, action_input)
        tool_calls.append({"tool": action, "args": action_input, "result": tool_result})

        # Build the observation.
        observation = f"OBSERVATION: {tool_result}"
        if not tool_result.startswith("ERROR"):
            semantic_memory.store(observation)
        buffer.add("user", observation)

    # Step 5: If the loop ends without a FINAL ANSWER, return the result indicating early termination.
    return AgentResult(answer="MAX_ITERATIONS reached", iterations=MAX_ITERATIONS, tool_calls=tool_calls, stopped_early=True)

# ── Entry point ───────────────────────────────────────────────────────────────

DEMO_TASK = (
    "Write a Python function called word_frequency(filepath: str) -> dict "
    "that reads a text file and returns a dictionary mapping each word to "
    "the number of times it appears.  The function should be case-insensitive "
    "and ignore punctuation.  Write the function to workspace/word_frequency.py, "
    "then write a small test script to workspace/test_word_frequency.py that "
    "creates a sample file, calls the function, and prints the result.  "
    "Finally, run the test script and confirm it produces output."
)

if __name__ == "__main__":
    result = run_agent(DEMO_TASK)
    print(result)

