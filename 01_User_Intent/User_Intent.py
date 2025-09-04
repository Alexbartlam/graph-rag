# Import necessary libraries
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For OpenAI support
from google.adk.tools import ToolContext

# Convenience libraries for working with Neo4j inside of Google ADK
from neo4j_for_adk import graphdb, tool_success, tool_error

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)

#print("Libraries imported.")

# --- Define Model Constants for easier use ---
MODEL_GPT_4O = "openai/gpt-4o"

llm = LiteLlm(model=MODEL_GPT_4O)

# Test LLM with a direct call
print(llm.llm_client.completion(model=llm.model, messages=[{"role": "user", "content": "Are you ready?"}], tools=[]))

#print("\nOpenAI is ready!")

# define the role and goal for the user intent agent
agent_role_and_goal = """
    You are an expert at knowledge graph use cases. 
    Your primary goal is to help the user come up with a knowledge graph use case.
"""

# give the agent some hints about what to say
agent_conversational_hints = """
    If the user is unsure what to do, make some suggestions based on classic use cases like:
    - social network involving friends, family, or professional relationships
    - logistics network with suppliers, customers, and partners
    - recommendation system with customers, products, and purchase patterns
    - fraud detection over multiple accounts with suspicious patterns of transactions
    - pop-culture graphs with movies, books, or music
"""

# describe what the output should look like
agent_output_definition = """
    A user goal has two components:
    - kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
    - description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."
"""

# specify the steps the agent should follow
agent_chain_of_thought_directions = """
    Think carefully and collaborate with the user:
    1. Understand the user's goal, which is a kind_of_graph with description
    2. Ask clarifying questions as needed
    3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record your perception
    4. Present the perceived user goal to the user for confirmation
    5. If the user agrees, use the 'approve_perceived_user_goal' tool to approve the user goal. This will save the goal in state under the 'approved_user_goal' key.
"""
# combine all the instruction components into one complete instruction...
complete_agent_instruction = f"""
{agent_role_and_goal}
{agent_conversational_hints}
{agent_output_definition}
{agent_chain_of_thought_directions}
"""

#print(complete_agent_instruction)

# Tool: Set Perceived User Goal
# to encourage collaboration with the user, the first tool only sets the perceived user goal

PERCEIVED_USER_GOAL = "perceived_user_goal"

def set_perceived_user_goal(kind_of_graph: str, graph_description:str, tool_context: ToolContext):
    """Sets the perceived user's goal, including the kind of graph and its description.
    
    Args:
        kind_of_graph: 2-3 word definition of the kind of graph, for example "recent US patents"
        graph_description: a single paragraph description of the graph, summarizing the user's intent
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    tool_context.state[PERCEIVED_USER_GOAL] = user_goal_data
    return tool_success(PERCEIVED_USER_GOAL, user_goal_data)

# Tool: Approve the perceived user goal
# approval from the user should trigger a call to this tool

APPROVED_USER_GOAL = "approved_user_goal"

def approve_perceived_user_goal(tool_context: ToolContext):
    """Upon approval from user, will record the perceived user goal as the approved user goal.
    
    Only call this tool if the user has explicitly approved the perceived user goal.
    """
    # Trust, but verify. 
    # Require that the perceived goal was set before approving it. 
    # Notice the tool error helps the agent take
    if PERCEIVED_USER_GOAL not in tool_context.state:
        return tool_error("perceived_user_goal not set. Set perceived user goal first, or ask clarifying questions if you are unsure.")
    
    tool_context.state[APPROVED_USER_GOAL] = tool_context.state[PERCEIVED_USER_GOAL]

    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])

# add the tools to a list
user_intent_agent_tools = [set_perceived_user_goal, approve_perceived_user_goal]

# Finally, construct the agent

user_intent_agent = Agent(
    name="user_intent_agent_v1", # a unique, versioned name
    model=llm, # defined earlier in a variable
    description="Helps the user ideate on a knowledge graph use case.", # used for delegation
    instruction=complete_agent_instruction, # the complete instructions you composed earlier
    tools=user_intent_agent_tools, # the list of tools
)

print(f"Agent '{user_intent_agent.name}' created.")

# =============================================================================
# DEMO CONFIGURATION - Modify these variables to change the script behavior
# =============================================================================

# Demo inputs - Change these to test different scenarios
DEMO_USER_GOAL = """I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, 
which can support root-cause analysis."""

DEMO_FOLLOW_UP = """I'm concerned about possible manufacturing or supplier issues."""

DEMO_APPROVAL = "Approve that goal."

# Demo settings
VERBOSE_MODE = True  # Set to False for less output
RUN_DEMO = True      # Set to False to skip the demo

# =============================================================================
# SYNCHRONOUS DEMO RUNNER - No async needed!
# =============================================================================

def simulate_tool_call(tool_function, *args, **kwargs):
    """Simulate calling a tool function directly without the agent."""
    print(f"🔧 Calling tool: {tool_function.__name__}")
    if args:
        print(f"   Args: {args}")
    if kwargs:
        print(f"   Kwargs: {kwargs}")
    
    # Create a mock tool context with state
    class MockToolContext:
        def __init__(self):
            self.state = {}
    
    tool_context = MockToolContext()
    result = tool_function(*args, tool_context, **kwargs)
    
    print(f"   Result: {result}")
    return result, tool_context.state

def run_user_intent_demo():
    """Run the user intent demo without async - just direct tool calls."""
    
    print("\n" + "="*60)
    print("🎯 USER INTENT DEMO - Synchronous Version")
    print("="*60)
    
    print(f"📝 Demo Goal: {DEMO_USER_GOAL}")
    print(f"📝 Follow-up: {DEMO_FOLLOW_UP}")
    print(f"📝 Approval: {DEMO_APPROVAL}")
    
    # Simulate the conversation flow using direct tool calls
    current_state = {}
    
    # Step 1: Parse the user goal and extract kind_of_graph and description
    print(f"\n{'='*50}")
    print("STEP 1: Processing User Goal")
    print(f"{'='*50}")
    
    # Simulate what the LLM would extract from the user input
    kind_of_graph = "bill of materials"
    graph_description = "A multi-level bill of materials for manufactured products, useful for root cause analysis."
    
    print(f"🧠 LLM extracted:")
    print(f"   Kind of graph: {kind_of_graph}")
    print(f"   Description: {graph_description}")
    
    # Step 2: Set perceived user goal
    print(f"\n{'='*50}")
    print("STEP 2: Setting Perceived User Goal")
    print(f"{'='*50}")
    
    result, state_update = simulate_tool_call(
        set_perceived_user_goal, 
        kind_of_graph, 
        graph_description
    )
    current_state.update(state_update)
    
    # Step 3: Approve the goal
    print(f"\n{'='*50}")
    print("STEP 3: Approving User Goal")
    print(f"{'='*50}")
    
    # Create a tool context with the current state
    class MockToolContext:
        def __init__(self, state):
            self.state = state
    
    tool_context = MockToolContext(current_state)
    approval_result = approve_perceived_user_goal(tool_context)
    
    print(f"🔧 Calling tool: approve_perceived_user_goal")
    print(f"   Result: {approval_result}")
    
    # Final results
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETED - FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"📊 Final State: {tool_context.state}")
    
    if APPROVED_USER_GOAL in tool_context.state:
        approved = tool_context.state[APPROVED_USER_GOAL]
        print(f"\n🎯 Approved Goal:")
        print(f"   Kind: {approved['kind_of_graph']}")
        print(f"   Description: {approved['graph_description']}")
    
    return tool_context.state

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if RUN_DEMO:
    final_state = run_user_intent_demo()
    
    print(f"\n{'='*60}")
    print("🔧 CUSTOMIZATION TIPS")
    print(f"{'='*60}")
    print("To test different scenarios, modify these variables at the top:")
    print("- DEMO_USER_GOAL: Change the user's request")
    print("- DEMO_FOLLOW_UP: Change the follow-up question")
    print("- VERBOSE_MODE: Toggle detailed output")
    
else:
    print("📴 Demo skipped (RUN_DEMO = False)")
    print("Set RUN_DEMO = True to run the demonstration")