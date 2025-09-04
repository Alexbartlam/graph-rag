# Import necessary libraries
import os
from pathlib import Path

from itertools import islice

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For OpenAI support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import ToolContext
from google.genai import types # For creating message Content/Parts

# For type hints
from typing import Dict, Any, List

# Convenience libraries for working with Neo4j inside of Google ADK
from neo4j_for_adk import graphdb, tool_success, tool_error

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.CRITICAL)

print("Libraries imported.")

# --- Define Model Constants for easier use ---
MODEL_GPT_4O = "openai/gpt-4o"

llm = LiteLlm(model=MODEL_GPT_4O)

# Test LLM with a direct call
print(llm.llm_client.completion(model=llm.model, messages=[{"role": "user", "content": "Are you ready?"}], tools=[]))

print("\nOpenAI is ready.")

file_suggestion_agent_instruction = """
You are a constructive critic AI reviewing a list of files. Your goal is to suggest relevant files
for constructing a knowledge graph.

**Task:**
Review the file list for relevance to the kind of graph and description specified in the approved user goal. 

For any file that you're not sure about, use the 'sample_file' tool to get 
a better understanding of the file contents. 

Only consider structured data files like CSV or JSON.

Prepare for the task:
- use the 'get_approved_user_goal' tool to get the approved user goal

Think carefully, repeating these steps until finished:
1. list available files using the 'list_available_files' tool
2. evaluate the relevance of each file, then record the list of suggested files using the 'set_suggested_files' tool
3. use the 'get_suggested_files' tool to get the list of suggested files
4. ask the user to approve the set of suggested files
5. If the user has feedback, go back to step 1 with that feedback in mind
6. If approved, use the 'approve_suggested_files' tool to record the approval
"""

# import tools defined in previous lesson
from helper import get_neo4j_import_dir

# Mock the get_approved_user_goal function since we're not using the tools module
def get_approved_user_goal(tool_context: ToolContext) -> dict:
    """Mock function that returns a predefined approved user goal."""
    if hasattr(tool_context, 'state') and 'approved_user_goal' in tool_context.state:
        return tool_success("approved_user_goal", tool_context.state['approved_user_goal'])
    else:
        # Return a default goal for demo
        default_goal = {
            "kind_of_graph": "supply chain analysis",
            "description": "A multi-level bill of materials for manufactured products, useful for root cause analysis."
        }
        return tool_success("approved_user_goal", default_goal)

# Tool: List Import Files

# this constant will be used as the key for storing the file list in the tool context state
ALL_AVAILABLE_FILES = "all_available_files"

def list_available_files(tool_context:ToolContext) -> dict:
    f"""Lists files available for knowledge graph construction.
    All files are relative to the import directory.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {ALL_AVAILABLE_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # get the import dir using the helper function
    import_dir = Path(get_neo4j_import_dir())

    # get a list of relative file names, so files must be rooted at the import dir
    file_names = [str(x.relative_to(import_dir)) 
                 for x in import_dir.rglob("*") 
                 if x.is_file()]

    # save the list to state so we can inspect it later
    tool_context.state[ALL_AVAILABLE_FILES] = file_names

    return tool_success(ALL_AVAILABLE_FILES, file_names)

    # Tool: Sample File
# This is a simple file reading tool that only works on files from the import directory
def sample_file(file_path: str, tool_context: ToolContext) -> dict:
    """Samples a file by reading its content as text.
    
    Treats any file as text and reads up to a maximum of 100 lines.
    
    Args:
      file_path: file to sample, relative to the import directory
      
    Returns:
        dict: A dictionary containing metadata about the content,
            along with a sampling of the file.
            Includes a 'status' key ('success' or 'error').
            If 'success', includes a 'content' key with textual file content.
            If 'error', includes an 'error_message' key.
            The 'error_message' may have instructions about how to handle the error.
    """
    # Trust, but verify. The agent may invent absolute file paths. 
    if Path(file_path).is_absolute():
        return tool_error("File path must be relative to the import directory. Make sure the file is from the list of available files.")
    
    import_dir = Path(get_neo4j_import_dir())

    # create the full path by extending from the import_dir
    full_path_to_file = import_dir / file_path
    
    # of course, _that_ may not exist
    if not full_path_to_file.exists():
        return tool_error(f"File does not exist in import directory. Make sure {file_path} is from the list of available files.")
    
    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)
    
    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")

        # Tool: Set/Get suggested files
SUGGESTED_FILES = "suggested_files"

def set_suggested_files(suggest_files:List[str], tool_context:ToolContext) -> Dict[str, Any]:
    """Set the suggested files to be used for data import.

    Args:
        suggest_files (List[str]): List of file paths to suggest

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)

# Helps encourage the LLM to first set the suggested files.
# This is an important strategy for maintaining consistency through defined values.
def get_suggested_files(tool_context:ToolContext) -> Dict[str, Any]:
    """Get the files to be used for data import.

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
    """
    return tool_success(SUGGESTED_FILES, tool_context.state[SUGGESTED_FILES])

# Tool: Approve Suggested Files
# Just like the previous lesson, you'll define a tool which
# accepts no arguments and can sanity check before approving.
APPROVED_FILES = "approved_files"

def approve_suggested_files(tool_context:ToolContext) -> Dict[str, Any]:
    """Approves the {SUGGESTED_FILES} in state for further processing as {APPROVED_FILES}.
    
    If {SUGGESTED_FILES} is not in state, return an error.
    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])

# List of tools for the file suggestion agent
file_suggestion_agent_tools = [get_approved_user_goal, list_available_files, sample_file, 
    set_suggested_files, get_suggested_files,
    approve_suggested_files
]

# Finally, construct the agent

file_suggestion_agent = Agent(
    name="file_suggestion_agent_v1",
    model=llm, # defined earlier in a variable
    description="Helps the user select files to import.",
    instruction=file_suggestion_agent_instruction,
    tools=file_suggestion_agent_tools,
)

print(f"Agent '{file_suggestion_agent.name}' created.")

# =============================================================================
# DEMO CONFIGURATION - Modify these variables to change the script behavior
# =============================================================================

# Demo inputs - Change these to test different scenarios
DEMO_USER_GOAL = {
    "kind_of_graph": "supply chain analysis",
    "description": "A multi-level bill of materials for manufactured products, useful for root cause analysis."
}

DEMO_QUESTION = "What files can we use for import?"
DEMO_APPROVAL = "Yes, let's do it"

# Demo settings
VERBOSE_MODE = True  # Set to False for less output
RUN_DEMO = True      # Set to False to skip the demo

# Mock data directory setup
import os
def setup_demo_data():
    """Create sample data files for demonstration."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample CSV files if they don't exist
    sample_files = {
        "products.csv": "product_id,product_name,product_type\nP001,Smartphone X1,Electronics\nP002,Laptop Pro,Electronics",
        "suppliers.csv": "supplier_id,supplier_name,country\nS001,TechCorp,USA\nS002,GlobalParts,China",
        "assemblies.csv": "assembly_id,assembly_name,product_id\nA001,Main Board,P001\nA002,Battery,P001",
        "parts.csv": "part_id,part_name,assembly_id\nPT001,CPU,A001\nPT002,RAM,A001",
        "readme.txt": "This is a readme file, not CSV data."
    }
    
    for filename, content in sample_files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
    
    return str(data_dir)

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
        def __init__(self, initial_state=None):
            self.state = initial_state or {}
    
    tool_context = MockToolContext()
    
    # Set up initial state for get_approved_user_goal
    if tool_function.__name__ == 'get_approved_user_goal':
        tool_context.state['approved_user_goal'] = DEMO_USER_GOAL
    
    result = tool_function(*args, tool_context, **kwargs)
    
    print(f"   Result: {result}")
    return result, tool_context.state

def run_file_suggestion_demo():
    """Run the file suggestion demo without async - just direct tool calls."""
    
    print("\n" + "="*60)
    print("📁 FILE SUGGESTION DEMO - Synchronous Version")
    print("="*60)
    
    print(f"📝 Demo Goal: {DEMO_USER_GOAL}")
    print(f"❓ Demo Question: {DEMO_QUESTION}")
    print(f"✅ Demo Approval: {DEMO_APPROVAL}")
    
    # Set up demo data
    data_dir = setup_demo_data()
    print(f"📂 Demo data directory: {data_dir}")
    
    # Simulate the conversation flow using direct tool calls
    current_state = {"approved_user_goal": DEMO_USER_GOAL}
    
    # Step 1: Get approved user goal
    print(f"\n{'='*50}")
    print("STEP 1: Getting Approved User Goal")
    print(f"{'='*50}")
    
    result, state_update = simulate_tool_call(get_approved_user_goal)
    current_state.update(state_update)
    
    # Step 2: List available files
    print(f"\n{'='*50}")
    print("STEP 2: Listing Available Files")
    print(f"{'='*50}")
    
    result, state_update = simulate_tool_call(list_available_files)
    current_state.update(state_update)
    available_files = result.get(ALL_AVAILABLE_FILES, [])
    
    # Step 3: Sample a file (if any exist)
    if available_files:
        print(f"\n{'='*50}")
        print("STEP 3: Sampling First File")
        print(f"{'='*50}")
        
        first_file = available_files[0]
        result, state_update = simulate_tool_call(sample_file, first_file)
    
    # Step 4: Set suggested files (simulate LLM decision)
    print(f"\n{'='*50}")
    print("STEP 4: Setting Suggested Files")
    print(f"{'='*50}")
    
    # Filter for CSV files (simulate what the LLM would do)
    csv_files = [f for f in available_files if f.endswith('.csv')]
    print(f"🧠 LLM filtered CSV files: {csv_files}")
    
    result, state_update = simulate_tool_call(set_suggested_files, csv_files)
    current_state.update(state_update)
    
    # Step 5: Get suggested files
    print(f"\n{'='*50}")
    print("STEP 5: Getting Suggested Files")
    print(f"{'='*50}")
    
    # Create tool context with current state
    class MockToolContext:
        def __init__(self, state):
            self.state = state
    
    tool_context = MockToolContext(current_state)
    result = get_suggested_files(tool_context)
    print(f"🔧 Calling tool: get_suggested_files")
    print(f"   Result: {result}")
    
    # Step 6: Approve suggested files
    print(f"\n{'='*50}")
    print("STEP 6: Approving Suggested Files")
    print(f"{'='*50}")
    
    approval_result = approve_suggested_files(tool_context)
    print(f"🔧 Calling tool: approve_suggested_files")
    print(f"   Result: {approval_result}")
    
    # Final results
    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETED - FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"📊 Final State: {tool_context.state}")
    
    if APPROVED_FILES in tool_context.state:
        approved = tool_context.state[APPROVED_FILES]
        print(f"\n📁 Approved Files:")
        for i, file in enumerate(approved, 1):
            print(f"   {i}. {file}")
    
    return tool_context.state

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if RUN_DEMO:
    final_state = run_file_suggestion_demo()
    
    print(f"\n{'='*60}")
    print("🔧 CUSTOMIZATION TIPS")
    print(f"{'='*60}")
    print("To test different scenarios, modify these variables at the top:")
    print("- DEMO_USER_GOAL: Change the user's goal")
    print("- DEMO_QUESTION: Change the initial question")
    print("- VERBOSE_MODE: Toggle detailed output")
    print("- Add more files to ./data/ directory")
    
else:
    print("📴 Demo skipped (RUN_DEMO = False)")
    print("Set RUN_DEMO = True to run the demonstration")