# Import necessary libraries
import os
from pathlib import Path

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models.lite_llm import LiteLlm # For OpenAI support
from google.adk.tools import ToolContext

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

print("\nOpenAI ready.")

# Notice the use of {feedback} in the string.
# Google ADK will replace this from session context if it exists.
# Because feedback could be long, XML-like delimiters are included to clarify the content.
proposal_agent_role_and_goal = """
    You are an expert at knowledge graph modeling with property graphs. Propose an appropriate
    schema by specifying construction rules which transform approved files into nodes or relationships.
    The resulting schema should describe a knowledge graph based on the user goal.
    
    Consider feedback if it is available: 
    <feedback>
    {feedback}
    </feedback> 
"""

proposal_agent_hints = """
    Every file in the approved files list will become either a node or a relationship.
    Determining whether a file likely represents a node or a relationship is based
    on a hint from the filename (is it a single thing or two things) and the
    identifiers found within the file.

    Because unique identifiers are so important for determining the structure of the graph,
    always verify the uniqueness of suspected unique identifiers using the 'search_file' tool.

    General guidance for identifying a node or a relationship:
    - If the file name is singular and has only 1 unique identifier it is likely a node
    - If the file name is a combination of two things, it is likely a full relationship
    - If the file name sounds like a node, but there are multiple unique identifiers, that is likely a node with reference relationships

    Design rules for nodes:
    - Nodes will have unique identifiers. 
    - Nodes _may_ have identifiers that are used as reference relationships.

    Design rules for relationships:
    - Relationships appear in two ways: full relationships and reference relationships.

    Full relationships:
    - Full relationships appear in dedicated relationship files, often having a filename that references two entities
    - Full relationships typically have references to a source and destination node.
    - Full relationships _do not have_ unique identifiers, but instead have references to the primary keys of the source and destination nodes.
    - The absence of a single, unique identifier is a strong indicator that a file is a full relationship.
    
    Reference relationships:
    - Reference relationships appear as foreign key references in node files
    - Reference relationship foreign key column names often hint at the destination node and relationship type
    - References may be hierarchical container relationships, with terminology revealing parent-child, "has", "contains", membership, or similar relationship
    - References may be peer relationships, that is often a self-reference to a similar class of nodes. For example, "knows" or "see also"

    The resulting schema should be a connected graph, with no isolated components.
"""

proposal_agent_chain_of_thought_directions = """
    Prepare for the task:
    - get the user goal using the 'get_approved_user_goal' tool
    - get the list of approved files using the 'get_approved_files' tool
    - get the current construction plan using the 'get_proposed_construction_plan' tool

    Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
    1. For each approved file, consider whether it represents a node or relationship. Check the content for potential unique identifiers using the 'sample_file' tool.
    2. For each identifier, verify that it is unique by using the 'search_file' tool.
    3. Use the node vs relationship guidance for deciding whether the file represents a node or a relationship.
    4. For a node file, propose a node construction using the 'propose_node_construction' tool. 
    5. If the node contains a reference relationship, use the 'propose_relationship_construction' tool to propose a relationship construction. 
    6. For a relationship file, propose a relationship construction using the 'propose_relationship_construction' tool
    7. If you need to remove a construction, use the 'remove_node_construction' or 'remove_relationship_construction' tool
    8. When you are done with construction proposals, use the 'get_proposed_construction_plan' tool to present the plan to the user
"""

# finally, combine all the prompt parts together
proposal_agent_instruction = f"""
{proposal_agent_role_and_goal}
{proposal_agent_hints}
{proposal_agent_chain_of_thought_directions}
"""

print(proposal_agent_instruction)

# import tools defined in previous notebook
from tools import get_approved_user_goal, get_approved_files, sample_file

# Import from helper with fallback for import directory
def get_neo4j_import_dir():
    """Get the Neo4j import directory with fallback to ./data"""
    import os
    from pathlib import Path
    from dotenv import load_dotenv, find_dotenv
    
    # Try to load environment
    try:
        load_dotenv(find_dotenv())
        import_dir = os.getenv("NEO4J_IMPORT_DIR", "./data")
    except:
        import_dir = "./data"
    
    # Ensure the directory exists
    Path(import_dir).mkdir(parents=True, exist_ok=True)
    return import_dir

SEARCH_RESULTS = "search_results"

# A simple grep-like tool for searching text files
def search_file(file_path: str, query: str) -> dict:
    """
    Searches any text file (markdown, csv, txt) for lines containing the given query string.
    Simple grep-like functionality that works with any text file.
    Search is always case insensitive.

    Args:
      file_path: Path to the file, relative to the Neo4j import directory.
      query: The string to search for.

    Returns:
        dict: A dictionary with 'status' ('success' or 'error').
              If 'success', includes 'search_results' containing 'matching_lines'
              (a list of dictionaries with 'line_number' and 'content' keys)
              and basic metadata about the search.
              If 'error', includes an 'error_message'.
    """
    import_dir = Path(get_neo4j_import_dir())
    p = import_dir / file_path

    if not p.exists():
        return tool_error(f"File does not exist: {file_path}")
    if not p.is_file():
        return tool_error(f"Path is not a file: {file_path}")

    # Handle empty query - return no results
    if not query:
        return tool_success(SEARCH_RESULTS, {
            "metadata": {
                "path": file_path,
                "query": query,
                "lines_found": 0
            },
            "matching_lines": []
        })

    matching_lines = []
    search_query = query.lower()
    
    try:
        with open(p, 'r', encoding='utf-8') as file:
            # Process the file line by line
            for i, line in enumerate(file, 1):
                line_to_check = line.lower()
                if search_query in line_to_check:
                    matching_lines.append({
                        "line_number": i,
                        "content": line.strip()  # Remove trailing newlines
                    })
                        
    except Exception as e:
        return tool_error(f"Error reading or searching file {file_path}: {e}")

    # Prepare basic metadata
    metadata = {
        "path": file_path,
        "query": query,
        "lines_found": len(matching_lines)
    }
    
    result_data = {
        "metadata": metadata,
        "matching_lines": matching_lines
    }
    return tool_success(SEARCH_RESULTS, result_data)

#  Tool: Propose Node Construction

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
NODE_CONSTRUCTION = "node_construction"

def propose_node_construction(approved_file: str, proposed_label: str, unique_column_name: str, proposed_properties: list[str], tool_context:ToolContext) -> dict:
    """Propose a node construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_label as the key.

    The construction entry will be a dictionary with the following keys:
    - construction_type: "node"
    - source_file: the approved file to propose a node construction for
    - label: the proposed label of the node
    - unique_column_name: the name of the column that will be used to uniquely identify constructed nodes
    - properties: A list of property names for the node, derived from column names in the approved file

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes (used as key in the construction plan)
        unique_column_name: The name of the column that will be used to uniquely identify constructed nodes
        proposed_properties: column names that should be imported as node properties

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "node_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # quick sanity check -- does the approved file have the unique column?
    search_results = search_file(approved_file, unique_column_name)
    if search_results["status"] == "error":
        return search_results # return the error
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the column {unique_column_name}. Check the file content and try again.")

    # get the current construction plan, or an empty one if none exists
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }   
    construction_plan[proposed_label] = node_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(NODE_CONSTRUCTION, node_construction_rule)

    RELATIONSHIP_CONSTRUCTION = "relationship_construction"

def propose_relationship_construction(approved_file: str, proposed_relationship_type: str, 
    from_node_label: str,from_node_column: str, to_node_label:str, to_node_column: str, 
    proposed_properties: list[str], 
    tool_context:ToolContext) -> dict:
    """Propose a relationship construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_relationship_type as the key.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_relationship_type: The proposed label for constructed relationships
        from_node_label: The label of the source node
        from_node_column: The name of the column within the approved file that will be used to uniquely identify source nodes
        to_node_label: The label of the target node
        to_node_column: The name of the column within the approved file that will be used to uniquely identify target nodes
        unique_column_name: The name of the column that will be used to uniquely identify target nodes

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # quick sanity check -- does the approved file have the from_node_column?
    search_results = search_file(approved_file, from_node_column)
    if search_results["status"] == "error": 
        return search_results  # return the error if there is one
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the from node column {from_node_column}. Check the content of the file and reconsider the relationship.")

    # quick sanity check -- does the approved file have the to_node_column?
    search_results = search_file(approved_file, to_node_column)
    if search_results["status"] == "error" or search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the to node column {to_node_column}. Check the content of the file and reconsider the relationship.")

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }   
    construction_plan[proposed_relationship_type] = relationship_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(RELATIONSHIP_CONSTRUCTION, relationship_construction_rule)

# Tool: Remove Node Construction
def remove_node_construction(node_label: str, tool_context:ToolContext) -> dict:
    """Remove a node construction from the proposed construction plan based on label.

    Args:
        node_label: The label of the node construction to remove
        tool_context: The tool context

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'node_construction_removed' key with the label of the removed node construction
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if node_label not in construction_plan:
        return tool_success("node construction rule not found. removal not needed.")

    del construction_plan[node_label]

    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("node_construction_removed", node_label)

    # Tool: Remove Relationship Construction
def remove_relationship_construction(relationship_type: str, tool_context:ToolContext) -> dict:
    """Remove a relationship construction from the proposed construction plan based on type.

    Args:
        relationship_type: The type of the relationship construction to remove
        tool_context: The tool context

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'relationship_construction_removed' key with the type of the removed relationship construction
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    if relationship_type not in construction_plan:
        return tool_success("relationship construction rule not found. removal not needed.")
    
    construction_plan.pop(relationship_type)
    
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("relationship_construction_removed", relationship_type) 

    # Tool: Get Proposed construction Plan
def get_proposed_construction_plan(tool_context:ToolContext) -> dict:
    """Get the proposed construction plan, a dictionary of construction rules."""
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    # Tool: Approve the proposed construction plan
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

def approve_proposed_construction_plan(tool_context:ToolContext) -> dict:
    """Approve the proposed construction plan, if there is one."""
    if not PROPOSED_CONSTRUCTION_PLAN in tool_context.state:
        return tool_error("No proposed construction plan found. Propose a plan first.")
    
    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    return tool_success(APPROVED_CONSTRUCTION_PLAN, tool_context.state[APPROVED_CONSTRUCTION_PLAN])
    
    # List of tools for the structured schema proposal agent
structured_schema_proposal_agent_tools = [
    get_approved_user_goal, get_approved_files, 
    get_proposed_construction_plan,
    sample_file, search_file,
    propose_node_construction, propose_relationship_construction, 
    remove_node_construction, remove_relationship_construction
]

from google.adk.agents.callback_context import CallbackContext

# a helper function to log the agent name during execution
def log_agent(callback_context: CallbackContext) -> None:
    print(f"\n### Entering Agent: {callback_context.agent_name}")

    SCHEMA_AGENT_NAME = "schema_proposal_agent_v1"
schema_proposal_agent = LlmAgent(
    name=SCHEMA_AGENT_NAME,
    description="Proposes a knowledge graph schema based on the user goal and approved file list",
    model=llm,
    instruction=proposal_agent_instruction,
    tools=structured_schema_proposal_agent_tools,
    before_agent_callback=log_agent
)

# =============================================================================
# DEMO CONFIGURATION - Modify these variables to change the script behavior
# =============================================================================

# Demo inputs - Change these to test different scenarios
DEMO_USER_GOAL = {
        "kind_of_graph": "supply chain analysis",
    "description": "A multi-level bill of materials for manufactured products, useful for root cause analysis."
}

DEMO_APPROVED_FILES = [
    'products.csv', 
        'assemblies.csv', 
        'parts.csv', 
        'suppliers.csv'
]

DEMO_QUESTION = "How can these files be imported to construct the knowledge graph?"

# Demo settings
VERBOSE_MODE = True  # Set to False for less output
RUN_DEMO = True      # Set to False to skip the demo

# =============================================================================
# MOCK CLASSES FOR TESTING
# =============================================================================

class MockToolContext:
    """Mock ToolContext for testing without the full ADK runner."""
    
    def __init__(self, initial_state=None):
        self.state = initial_state or {}

# =============================================================================
# DEMO DATA SETUP
# =============================================================================

def setup_demo_data():
    """Create sample data files for demonstration."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample CSV files if they don't exist
    sample_files = {
        "products.csv": "product_id,product_name,product_type\nP001,Smartphone X1,Electronics\nP002,Laptop Pro,Electronics",
        "suppliers.csv": "supplier_id,supplier_name,country\nS001,TechCorp,USA\nS002,GlobalParts,China",
        "assemblies.csv": "assembly_id,assembly_name,product_id\nA001,Main Board,P001\nA002,Battery,P001",
        "parts.csv": "part_id,part_name,assembly_id\nPT001,CPU,A001\nPT002,RAM,A001"
    }
    
    for filename, content in sample_files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
    
    return str(data_dir)

# =============================================================================
# SYNCHRONOUS DEMO FUNCTIONS
# =============================================================================

def simulate_tool_call(tool_function, *args, **kwargs):
    """Simulate calling a tool function directly without the agent."""
    print(f"🔧 Calling tool: {tool_function.__name__}")
    if args:
        print(f"   Args: {args}")
    if kwargs:
        print(f"   Kwargs: {kwargs}")
    
    # Create a mock tool context with state
    tool_context = MockToolContext()
    result = tool_function(*args, tool_context, **kwargs)
    
    print(f"   Result: {result}")
    return result, tool_context.state

def run_schema_proposal_demo():
    """Run the schema proposal demo without async - direct tool calls."""
    
    print("\n" + "="*60)
    print("🏗️ SCHEMA PROPOSAL DEMO - Synchronous Version")
    print("="*60)
    
    print(f"📝 Demo Goal: {DEMO_USER_GOAL}")
    print(f"📁 Demo Files: {DEMO_APPROVED_FILES}")
    print(f"❓ Demo Question: {DEMO_QUESTION}")
    
    # Set up demo data
    data_dir = setup_demo_data()
    print(f"📂 Demo data directory: {data_dir}")
    
    # Initialize state with demo data
    current_state = {
        "approved_user_goal": DEMO_USER_GOAL,
        "approved_files": DEMO_APPROVED_FILES,
        "feedback": "",
        "proposed_construction_plan": {}
    }
    
    # Step 1: Get approved user goal
    print(f"\n{'='*50}")
    print("STEP 1: Getting Approved User Goal")
    print(f"{'='*50}")
    
    tool_context = MockToolContext(current_state)
    result = get_approved_user_goal(tool_context)
    print(f"🔧 Result: {result}")
    
    # Step 2: Get approved files
    print(f"\n{'='*50}")
    print("STEP 2: Getting Approved Files")
    print(f"{'='*50}")
    
    result = get_approved_files(tool_context)
    print(f"🔧 Result: {result}")
    
    # Step 3: Sample each file to understand structure
    print(f"\n{'='*50}")
    print("STEP 3: Sampling Files to Understand Structure")
    print(f"{'='*50}")
    
    for file in DEMO_APPROVED_FILES:
        print(f"\n--- Sampling {file} ---")
        result = sample_file(file, tool_context)
        print(f"🔧 Sample result: {result}")
    
    # Step 4: Propose node constructions (simulate what the LLM would do)
    print(f"\n{'='*50}")
    print("STEP 4: Proposing Node Constructions")
    print(f"{'='*50}")
    
    # Simulate LLM analysis and propose nodes
    node_proposals = [
        ("products.csv", "Product", "product_id", ["product_name", "product_type"]),
        ("suppliers.csv", "Supplier", "supplier_id", ["supplier_name", "country"]),
        ("assemblies.csv", "Assembly", "assembly_id", ["assembly_name"]),
        ("parts.csv", "Part", "part_id", ["part_name"])
    ]
    
    for file, label, unique_col, properties in node_proposals:
        print(f"\n--- Proposing {label} from {file} ---")
        result = propose_node_construction(file, label, unique_col, properties, tool_context)
        print(f"🔧 Node construction result: {result}")
    
    # Step 5: Propose relationship constructions
    print(f"\n{'='*50}")
    print("STEP 5: Proposing Relationship Constructions")
    print(f"{'='*50}")
    
    # Simulate relationship proposals
    relationship_proposals = [
        ("assemblies.csv", "BELONGS_TO", "Assembly", "assembly_id", "Product", "product_id", []),
        ("parts.csv", "PART_OF", "Part", "part_id", "Assembly", "assembly_id", [])
    ]
    
    for file, rel_type, from_label, from_col, to_label, to_col, properties in relationship_proposals:
        print(f"\n--- Proposing {rel_type} from {file} ---")
        result = propose_relationship_construction(file, rel_type, from_label, from_col, to_label, to_col, properties, tool_context)
        print(f"🔧 Relationship construction result: {result}")
    
    # Step 6: Get final construction plan
    print(f"\n{'='*50}")
    print("STEP 6: Getting Final Construction Plan")
    print(f"{'='*50}")
    
    final_plan = get_proposed_construction_plan(tool_context)
    print(f"🔧 Final construction plan: {final_plan}")
    
    # Final results
    print(f"\n{'='*60}")
    print("✅ SCHEMA PROPOSAL DEMO COMPLETED")
    print(f"{'='*60}")
    
    print(f"📊 Final State: {tool_context.state}")
    
    if 'proposed_construction_plan' in tool_context.state:
        plan = tool_context.state['proposed_construction_plan']
        print(f"\n🏗️ Proposed Construction Plan:")
        for key, construction in plan.items():
            print(f"   {key}: {construction}")
    
    return tool_context.state

critic_agent_role_and_goal = """
    You are an expert at knowledge graph modeling with property graphs. 
    Criticize the proposed schema for relevance to the user goal and approved files.
"""

critic_agent_hints = """
    Criticize the proposed schema for relevance and correctness:
    - Are unique identifiers actually unique? Use the 'search_file' tool to validate. Composite identifier are not acceptable.
    - Could any nodes be relationships instead? Double-check that unique identifiers are unique and not references to other nodes. Use the 'search_file' tool to validate
    - Can you manually trace through the source data to find the necessary information for anwering a hypothetical question?
    - Is every node in the schema connected? What relationships could be missing? Every node should connect to at least one other node.
    - Are hierarchical container relationships missing? 
    - Are any relationships redundant? A relationship between two nodes is redundant if it is semantically equivalent to or the inverse of another relationship between those two nodes.
"""

critic_agent_chain_of_thought_directions = """
    Prepare for the task:
    - get the user goal using the 'get_approved_user_goal' tool
    - get the list of approved files using the 'get_approved_files' tool
    - get the construction plan using the 'get_proposed_construction_plan' tool
    - use the 'sample_file' and 'search_file' tools to validate the schema design

    Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
    1. Analyze each construction rule in the proposed construction plan.
    2. Use tools to validate the construction rules for relevance and correctness.
    3. If the schema looks good, respond with a one word reply: 'valid'.
    4. If the schema has problems, respond with 'retry' and provide feedback as a concise bullet list of problems.
"""

# combine all the prompt parts together
critic_agent_instruction = f"""
{critic_agent_role_and_goal}
{critic_agent_hints}
{critic_agent_chain_of_thought_directions}
"""

print(critic_agent_instruction)

schema_critic_agent_tools = [
    get_approved_user_goal, get_approved_files,
    get_proposed_construction_plan,
    sample_file, search_file
]

CRITIC_NAME = "schema_critic_agent_v1"
schema_critic_agent = LlmAgent(
    name=CRITIC_NAME,
    description="Criticizes the proposed schema for relevance to the user goal and approved files.",
    model=llm,
    instruction=critic_agent_instruction,
    tools=schema_critic_agent_tools, 
    output_key="feedback", # specify the context state key which will contain the result of calling the critic,
    before_agent_callback=log_agent
)

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        feedback = ctx.session.state.get("feedback", "valid")
        should_stop = (feedback == "valid")
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))

schema_refinement_loop = LoopAgent(
    name="schema_refinement_loop",
    description="Analyzes approved files to propose a schema based on user intent and feedback",
    max_iterations=2,
    sub_agents=[schema_proposal_agent, schema_critic_agent, CheckStatusAndEscalate(name="StopChecker")],
    before_agent_callback=log_agent
)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if RUN_DEMO:
    final_state = run_schema_proposal_demo()
    
    print(f"\n{'='*60}")
    print("🔧 CUSTOMIZATION TIPS")
    print(f"{'='*60}")
    print("To test different scenarios, modify these variables at the top:")
    print("- DEMO_USER_GOAL: Change the user's goal")
    print("- DEMO_APPROVED_FILES: Change the list of files")
    print("- VERBOSE_MODE: Toggle detailed output")
    print("- Add more files to ./data/ directory")
    
else:
    print("📴 Demo skipped (RUN_DEMO = False)")
    print("Set RUN_DEMO = True to run the demonstration")
