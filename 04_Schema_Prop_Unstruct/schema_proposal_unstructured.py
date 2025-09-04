# Import necessary libraries

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

print("Libraries imported.")

# --- Define Model Constants for easier use ---
MODEL_GPT_4O = "openai/gpt-4o"

llm = LiteLlm(model=MODEL_GPT_4O)

# Test LLM with a direct call
print(llm.llm_client.completion(model=llm.model, messages=[{"role": "user", "content": "Are you ready?"}], tools=[]))

print("\nOpenAI ready.")

# Check connection to Neo4j by sending a query

neo4j_is_ready = graphdb.send_query("RETURN 'Neo4j is Ready!' as message")

print(neo4j_is_ready)

## 7.3. Named Entity Recognition (NER) Sub-agent

ner_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the kind of named entities that could be extracted which would be relevant 
  for a user's goal.
  """

ner_agent_hints = """
  Entities are people, places, things and qualities, but not quantities. 
  Your goal is to propose a list of the type of entities, not the actual instances
  of entities.

  There are two general approaches to identifying types of entities:
  - well-known entities: these closely correlate with approved node labels in an existing graph schema
  - discovered entities: these may not exist in the graph schema, but appear consistently in the source text

  Design rules for well-known entities:
  - always use existing well-known entity types. For example, if there is a well-known type "Person", and people appear in the text, then propose "Person" as the type of entity.
  - prefer reusing existing entity types rather than creating new ones
  
  Design rules for discovered entities:
  - discovered entities are consistently mentioned in the text and are highly relevant to the user's goal
  - always look for entities that would provide more depth or breadth to the existing graph
  - for example, if the user goal is to represent social communities and the graph has "Person" nodes, look through the text to discover entities that are relevant like "Hobby" or "Event"
  - avoid quantitative types that may be better represented as a property on an existing entity or relationship.
  - for example, do not propose "Age" as a type of entity. That is better represented as an additional property "age" on a "Person".
"""

ner_agent_chain_of_thought_directions = """
  Prepare for the task:
  - use the 'get_user_goal' tool to get the user goal
  - use the 'get_approved_files' tool to get the list of approved files
  - use the 'get_well_known_types' tool to get the approved node labels

  Think step by step:
  1. Sample some of the files using the 'sample_file' tool to understand the content
  2. Consider what well-known entities are mentioned in the text
  3. Discover entities that are frequently mentioned in the text that support the user's goal
  4. Use the 'set_proposed_entities' tool to save the list of well-known and discovered entity types
  5. Use the 'get_proposed_entities' tool to retrieve the proposed entities and present them to the user for their approval
  6. If the user approves, use the 'approve_proposed_entities' tool to finalize the entity types
  7. If the user does not approve, consider their feedback and iterate on the proposal
"""

ner_agent_instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""
# tools to propose and approve entity types
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"

def set_proposed_entities(proposed_entity_types: list[str], tool_context:ToolContext) -> dict:
    """Sets the list proposed entity types to extract from unstructured text."""
    tool_context.state[PROPOSED_ENTITIES] = proposed_entity_types
    return tool_success(PROPOSED_ENTITIES, proposed_entity_types)

def get_proposed_entities(tool_context:ToolContext) -> dict:
    """Gets the list of proposed entity types to extract from unstructured text."""
    return tool_context.state.get(PROPOSED_ENTITIES, [])

def approve_proposed_entities(tool_context:ToolContext) -> dict:
    """Upon approval from user, records the proposed entity types as an approved list of entity types 

    Only call this tool if the user has explicitly approved the suggested files.
    """
    if PROPOSED_ENTITIES not in tool_context.state:
        return tool_error("No proposed entity types to approve. Please set proposed entities first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_ENTITIES] = tool_context.state.get(PROPOSED_ENTITIES)
    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])

def get_approved_entities(tool_context:ToolContext) -> dict:
    """Get the approved list of entity types to extract from unstructured text."""
    return tool_context.state.get(APPROVED_ENTITIES, [])

def get_well_known_types(tool_context:ToolContext) -> dict:
    """Gets the approved labels that represent well-known entity types in the graph schema."""
    construction_plan = tool_context.state.get("approved_construction_plan", {})
    # approved labels are the keys for each construction plan entry where `construction_type` is "node"
    approved_labels = {entry["label"] for entry in construction_plan.values() if entry["construction_type"] == "node"}
    return tool_success("approved_labels", approved_labels)

# =============================================================================
# DEMO CONFIGURATION - Modify these variables to change the script behavior
# =============================================================================

# Demo inputs - Change these to test different scenarios
DEMO_USER_GOAL = {
    "kind_of_graph": "supply chain analysis",
    "description": "A multi-level bill of materials for manufactured products, useful for root cause analysis. Add product reviews to start analysis from reported issues like quality, difficulty, or durability."
}

DEMO_APPROVED_FILES = [
    "product_reviews/gothenburg_table_reviews.md",
    "product_reviews/stockholm_chair_reviews.md",
    "product_reviews/malmo_desk_reviews.md"
]

DEMO_CONSTRUCTION_PLAN = {
    "Product": {"construction_type": "node", "label": "Product"},
    "Assembly": {"construction_type": "node", "label": "Assembly"}, 
    "Part": {"construction_type": "node", "label": "Part"},
    "Supplier": {"construction_type": "node", "label": "Supplier"}
}

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
    """Create sample product review files for demonstration."""
    from pathlib import Path
    
    data_dir = Path("./product_reviews")
    data_dir.mkdir(exist_ok=True)
    
    # Create sample review files if they don't exist
    sample_reviews = {
        "gothenburg_table_reviews.md": """# Gothenburg Table Reviews

## Customer Review 1
**Rating: 2/5**
The table assembly was difficult. The screws didn't fit properly and the wood seemed low quality. 
Customer reported durability issues after 3 months of use.

## Customer Review 2  
**Rating: 4/5**
Nice design but assembly instructions were unclear. Table wobbles slightly on uneven surfaces.
Overall satisfied with the purchase.

## Customer Review 3
**Rating: 1/5** 
Poor quality materials. Supplier clearly cut corners. The tabletop cracked within weeks.
Disappointed with this product.
""",
        
        "stockholm_chair_reviews.md": """# Stockholm Chair Reviews

## Customer Review 1
**Rating: 5/5**
Excellent chair! High quality materials and easy assembly. Very comfortable for long periods.
Would recommend this product to others.

## Customer Review 2
**Rating: 3/5**
Chair looks good but assembly was challenging. Some parts didn't align properly.
Supplier should improve the manufacturing process.

## Customer Review 3
**Rating: 4/5**
Comfortable and well-designed. Minor quality issues with the fabric but overall satisfied.
Good value for the price.
""",

        "malmo_desk_reviews.md": """# Malmo Desk Reviews

## Customer Review 1  
**Rating: 2/5**
Assembly was a nightmare. Instructions were confusing and hardware was missing.
Quality seems poor for the price point.

## Customer Review 2
**Rating: 4/5** 
Good desk overall. Sturdy construction and nice finish. Assembly took longer than expected
but final product is satisfactory.

## Customer Review 3
**Rating: 1/5**
Terrible quality! Desk wobbles constantly and drawers don't close properly.
Supplier needs to improve their manufacturing quality control.
"""
    }
    
    for filename, content in sample_reviews.items():
        file_path = data_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
    
    return str(data_dir)

# =============================================================================
# HELPER FUNCTIONS
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

# Import and setup tools
from tools import get_approved_user_goal, get_approved_files, sample_file

ner_agent_tools = [
    get_approved_user_goal, get_approved_files, sample_file,
    get_well_known_types,
    set_proposed_entities,
    approve_proposed_entities
]

# Setup demo data and sample a file
setup_demo_data()
print("📂 Demo product review files created.")

# Sample file for demonstration
try:
    file_result = sample_file("product_reviews/gothenburg_table_reviews.md")
    if file_result.get("status") == "success":
        print("📄 Sample file content:")
        print(file_result["content"][:500] + "..." if len(file_result["content"]) > 500 else file_result["content"])
except Exception as e:
    print(f"Note: Could not sample file yet: {e}")

NER_AGENT_NAME = "ner_schema_agent_v1"
ner_schema_agent = Agent(
    name=NER_AGENT_NAME,
    description="Proposes the kind of named entities that could be extracted from text files.",
    model=llm,
    instruction=ner_agent_instruction,
    tools=ner_agent_tools, 
)

# =============================================================================
# SYNCHRONOUS NER DEMO FUNCTIONS
# =============================================================================

def run_ner_demo():
    """Run the Named Entity Recognition demo without async - direct tool calls."""
    
    print("\n" + "="*60)
    print("🔍 NER AGENT DEMO - Synchronous Version")
    print("="*60)
    
    print(f"📝 Demo Goal: {DEMO_USER_GOAL}")
    print(f"📁 Demo Files: {DEMO_APPROVED_FILES}")
    
    # Initialize state with demo data
    current_state = {
        "approved_user_goal": DEMO_USER_GOAL,
        "approved_files": DEMO_APPROVED_FILES,
        "approved_construction_plan": DEMO_CONSTRUCTION_PLAN
    }
    
    # Step 1: Get user goal
    print(f"\n{'='*50}")
    print("STEP 1: Getting User Goal")
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
    
    # Step 3: Get well-known types
    print(f"\n{'='*50}")
    print("STEP 3: Getting Well-Known Entity Types")
    print(f"{'='*50}")
    
    result = get_well_known_types(tool_context)
    print(f"🔧 Result: {result}")
    
    # Step 4: Sample files to understand content
    print(f"\n{'='*50}")
    print("STEP 4: Sampling Files")
    print(f"{'='*50}")
    
    for file in DEMO_APPROVED_FILES:
        print(f"\n--- Sampling {file} ---")
        result = sample_file(file)
        print(f"🔧 Sample result: {result}")
    
    # Step 5: Propose entities (simulate what the LLM would extract)
    print(f"\n{'='*50}")
    print("STEP 5: Proposing Entity Types")
    print(f"{'='*50}")
    
    # Simulate LLM analysis and propose entities based on review content
    proposed_entities = [
        "Product",      # Well-known from existing schema
        "Customer",     # Discovered from reviews
        "Review",       # Discovered from reviews  
        "Quality",      # Discovered from quality mentions
        "Assembly"      # Well-known from existing schema
    ]
    
    print(f"🧠 LLM would extract these entities: {proposed_entities}")
    result = set_proposed_entities(proposed_entities, tool_context)
    print(f"🔧 Set entities result: {result}")
    
    # Step 6: Get proposed entities
    print(f"\n{'='*50}")
    print("STEP 6: Getting Proposed Entities")
    print(f"{'='*50}")
    
    result = get_proposed_entities(tool_context)
    print(f"🔧 Get entities result: {result}")
    
    # Step 7: Approve entities
    print(f"\n{'='*50}")
    print("STEP 7: Approving Proposed Entities")
    print(f"{'='*50}")
    
    result = approve_proposed_entities(tool_context)
    print(f"🔧 Approval result: {result}")
    
    # Final results
    print(f"\n{'='*60}")
    print("✅ NER DEMO COMPLETED")
    print(f"{'='*60}")
    
    print(f"📊 Final State: {tool_context.state}")
    
    if APPROVED_ENTITIES in tool_context.state:
        entities = tool_context.state[APPROVED_ENTITIES]
        print(f"\n🔍 Approved Entities:")
        for i, entity in enumerate(entities, 1):
            print(f"   {i}. {entity}")
    
    return tool_context.state

fact_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the type of facts that could be extracted from text that would be relevant 
  for a user's goal. 
"""

fact_agent_hints = """
  Do not propose specific individual facts, but instead propose the general type 
  of facts that would be relevant for the user's goal. 
  For example, do not propose "ABK likes coffee" but the general type of fact "Person likes Beverage".
  
  Facts are triplets of (subject, predicate, object) where the subject and object are
  approved entity types, and the proposed predicate provides information about
  how they are related. For example, a fact type could be (Person, likes, Beverage).

  Design rules for facts:
  - only use approved entity types as subjects or objects. Do not propose new types of entities
  - the proposed predicate should describe the relationship between the approved subject and object
  - the predicate should optimize for information that is relevant to the user's goal
  - the predicate must appear in the source text. Do not guess.
  - use the 'add_proposed_fact' tool to record each proposed fact type
"""

fact_agent_chain_of_thought_directions = """
    Prepare for the task:
    - use the 'get_approved_user_goal' tool to get the user goal
    - use the 'get_approved_files' tool to get the list of approved files
    - use the 'get_approved_entities' tool to get the list of approved entity types

    Think step by step:
    1. Use the 'get_approved_user_goal' tool to get the user goal
    2. Sample some of the approved files using the 'sample_file' tool to understand the content
    3. Consider how subjects and objects are related in the text
    4. Call the 'add_proposed_fact' tool for each type of fact you propose
    5. Use the 'get_proposed_facts' tool to retrieve all the proposed facts
    6. Present the proposed types of facts to the user, along with an explanation
"""

fact_agent_instruction = f"""
{fact_agent_role_and_goal}
{fact_agent_hints}
{fact_agent_chain_of_thought_directions}
"""

PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"

def add_proposed_fact(approved_subject_label:str,
                      proposed_predicate_label:str,
                      approved_object_label:str,
                      tool_context:ToolContext) -> dict:
    """Add a proposed type of fact that could be extracted from the files.

    A proposed fact type is a tuple of (subject, predicate, object) where
    the subject and object are approved entity types and the predicate 
    is a proposed relationship label.

    Args:
      approved_subject_label: approved label of the subject entity type
      proposed_predicate_label: label of the predicate
      approved_object_label: approved label of the object entity type
    """
    # Guard against invalid labels
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])
    
    if approved_subject_label not in approved_entities:
        return tool_error(f"Approved subject label {approved_subject_label} not found. Try again.")
    if approved_object_label not in approved_entities:
        return tool_error(f"Approved object label {approved_object_label} not found. Try again.")
    
    current_predicates = tool_context.state.get(PROPOSED_FACTS, {})
    current_predicates[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    tool_context.state[PROPOSED_FACTS] = current_predicates
    return tool_success(PROPOSED_FACTS, current_predicates)
    
def get_proposed_facts(tool_context:ToolContext) -> dict:
    """Get the proposed types of facts that could be extracted from the files."""
    return tool_context.state.get(PROPOSED_FACTS, {})


def approve_proposed_facts(tool_context:ToolContext) -> dict:
    """Upon user approval, records the proposed fact types as approved fact types

    Only call this tool if the user has explicitly approved the proposed fact types.
    """
    if PROPOSED_FACTS not in tool_context.state:
        return tool_error("No proposed fact types to approve. Please set proposed facts first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_FACTS] = tool_context.state.get(PROPOSED_FACTS)
    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])

fact_agent_tools = [
    get_approved_user_goal, get_approved_files, 
    get_approved_entities,
    sample_file,
    add_proposed_fact,
    get_proposed_facts,
    approve_proposed_facts
]

FACT_AGENT_NAME = "fact_type_extraction_agent_v1"
relevant_fact_agent = Agent(
    name=FACT_AGENT_NAME,
    description="Proposes the kind of relevant facts that could be extracted from text files.",
    model=llm,
    instruction=fact_agent_instruction,
    tools=fact_agent_tools, 
)

# =============================================================================
# SYNCHRONOUS FACT EXTRACTION DEMO FUNCTIONS
# =============================================================================

def run_fact_extraction_demo(ner_state):
    """Run the fact extraction demo without async - direct tool calls."""
    
    print("\n" + "="*60)
    print("🔗 FACT EXTRACTION DEMO - Synchronous Version")
    print("="*60)
    
    # Use the NER agent's state as starting point
    current_state = ner_state.copy()
    
    print(f"📝 Demo Goal: {current_state.get('approved_user_goal', {})}")
    print(f"🔍 Approved Entities: {current_state.get(APPROVED_ENTITIES, [])}")
    
    # Step 1: Get approved user goal
    print(f"\n{'='*50}")
    print("STEP 1: Getting User Goal")
    print(f"{'='*50}")
    
    tool_context = MockToolContext(current_state)
    result = get_approved_user_goal(tool_context)
    print(f"🔧 Result: {result}")
    
    # Step 2: Get approved entities
    print(f"\n{'='*50}")
    print("STEP 2: Getting Approved Entities")
    print(f"{'='*50}")
    
    result = get_approved_entities(tool_context)
    print(f"🔧 Result: {result}")
    
    # Step 3: Sample files for fact extraction
    print(f"\n{'='*50}")
    print("STEP 3: Sampling Files for Fact Analysis")
    print(f"{'='*50}")
    
    for file in current_state.get("approved_files", []):
        print(f"\n--- Analyzing {file} ---")
        result = sample_file(file)
        print(f"🔧 Sample result: {result}")
    
    # Step 4: Propose facts (simulate what the LLM would extract)
    print(f"\n{'='*50}")
    print("STEP 4: Proposing Fact Types")
    print(f"{'='*50}")
    
    # Simulate LLM analysis and propose facts based on review content
    fact_proposals = [
        ("Customer", "reviews", "Product"),      # Customer reviews Product
        ("Customer", "reports", "Quality"),      # Customer reports Quality issues
        ("Product", "has", "Assembly"),          # Product has Assembly issues
        ("Review", "mentions", "Quality"),       # Review mentions Quality
        ("Customer", "rates", "Product")         # Customer rates Product
    ]
    
    print(f"🧠 LLM would extract these fact types:")
    for subject, predicate, obj in fact_proposals:
        print(f"   ({subject}, {predicate}, {obj})")
        result = add_proposed_fact(subject, predicate, obj, tool_context)
        print(f"🔧 Add fact result: {result}")
    
    # Step 5: Get proposed facts
    print(f"\n{'='*50}")
    print("STEP 5: Getting Proposed Facts")
    print(f"{'='*50}")
    
    result = get_proposed_facts(tool_context)
    print(f"🔧 Get facts result: {result}")
    
    # Step 6: Approve facts
    print(f"\n{'='*50}")
    print("STEP 6: Approving Proposed Facts")
    print(f"{'='*50}")
    
    result = approve_proposed_facts(tool_context)
    print(f"🔧 Approval result: {result}")
    
    # Final results
    print(f"\n{'='*60}")
    print("✅ FACT EXTRACTION DEMO COMPLETED")
    print(f"{'='*60}")
    
    print(f"📊 Final State: {tool_context.state}")
    
    if APPROVED_FACTS in tool_context.state:
        facts = tool_context.state[APPROVED_FACTS]
        print(f"\n🔗 Approved Fact Types:")
        for predicate, fact_info in facts.items():
            subject = fact_info["subject_label"]
            obj = fact_info["object_label"]
            print(f"   ({subject}, {predicate}, {obj})")
    
    return tool_context.state

# =============================================================================
# COMPLETE UNSTRUCTURED SCHEMA DEMO
# =============================================================================

def run_complete_unstructured_demo():
    """Run the complete unstructured schema proposal demo."""
    
    print("\n" + "="*80)
    print("🏗️ COMPLETE UNSTRUCTURED SCHEMA PROPOSAL DEMO")
    print("="*80)
    
    # Step 1: Run NER demo
    ner_state = run_ner_demo()
    
    # Step 2: Run fact extraction demo
    fact_state = run_fact_extraction_demo(ner_state)
    
    print(f"\n{'='*80}")
    print("🎯 COMPLETE DEMO SUMMARY")
    print(f"{'='*80}")
    
    # Summary of results
    entities = fact_state.get(APPROVED_ENTITIES, [])
    facts = fact_state.get(APPROVED_FACTS, {})
    
    print(f"\n📋 Final Schema Proposal:")
    print(f"   Entities: {len(entities)} types")
    for entity in entities:
        print(f"      - {entity}")
    
    print(f"\n   Relationships: {len(facts)} types")
    for predicate, fact_info in facts.items():
        subject = fact_info["subject_label"]
        obj = fact_info["object_label"]
        print(f"      - {subject} → {predicate} → {obj}")
    
    return fact_state

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if RUN_DEMO:
    final_state = run_complete_unstructured_demo()
    
    print(f"\n{'='*80}")
    print("🔧 CUSTOMIZATION TIPS")
    print(f"{'='*80}")
    print("To test different scenarios, modify these variables at the top:")
    print("- DEMO_USER_GOAL: Change the user's goal")
    print("- DEMO_APPROVED_FILES: Change the list of review files")
    print("- DEMO_CONSTRUCTION_PLAN: Change existing schema")
    print("- VERBOSE_MODE: Toggle detailed output")
    print("- Edit the product review content in setup_demo_data()")
    
else:
    print("📴 Demo skipped (RUN_DEMO = False)")
    print("Set RUN_DEMO = True to run the demonstration")