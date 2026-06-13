# graph.py
"""
Graph Orchestration Module for PawCare+ LangGraph workflow.
Builds complete state machine with all nodes and edges.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# Import state
from state import PetCareState, get_initial_state

# Import node functions (all 17 nodes)
from nodes.input_validator_node import input_validator_node
from nodes.pet_profile_extractor_node import pet_profile_extractor_node
from nodes.pet_health_risk_scorer_node import pet_health_risk_scorer_node
from nodes.owner_care_capability_node import owner_care_capability_node
from nodes.health_risk_router_node import health_risk_router_node
from nodes.pet_health_risk_analysis_node import pet_health_risk_analysis_node
from nodes.emergency_preparedness_node import emergency_preparedness_node
from nodes.nutrition_critical_node import nutrition_critical_node
from nodes.behavioral_coaching_node import behavioral_coaching_node
from nodes.wellness_monitoring_node import wellness_monitoring_node
from nodes.health_assessment_preventive_node import health_assessment_preventive_node
from nodes.nutrition_preventive_node import nutrition_preventive_node
from nodes.wellness_tracking_preventive_node import wellness_tracking_preventive_node
from nodes.wellness_optimization_node import wellness_optimization_node
from nodes.nutrition_wellness_node import nutrition_wellness_node
from nodes.lifestyle_enrichment_node import lifestyle_enrichment_node
from nodes.output_aggregator_node import output_aggregator_node

# Import workflow constants
from workflow.workflow import (
    CRITICAL_CARE_PATH,
    PREVENTIVE_CARE_PATH,
    WELLNESS_PATH
)

# Import OpenAI client
from utils.openai_client import OpenAIClient, build_openai_client

# Configure logging
logger = logging.getLogger(__name__)

# Node name constants (for graph building)
class NodeNames:
    """Canonical node names for graph construction"""
    INPUT_VALIDATOR = "input_validator"
    PROFILE_EXTRACTOR = "profile_extractor"
    HEALTH_RISK_SCORER = "health_risk_scorer"
    CARE_CAPABILITY_SCORER = "care_capability_scorer"
    HEALTH_RISK_ROUTER = "health_risk_router"
    
    # Critical path nodes
    RISK_ANALYSIS = "risk_analysis"
    EMERGENCY_PREP = "emergency_prep"
    NUTRITION_CRITICAL = "nutrition_critical"
    BEHAVIORAL_COACHING = "behavioral_coaching"
    WELLNESS_MONITORING = "wellness_monitoring"
    
    # Preventive path nodes
    HEALTH_ASSESSMENT_PREVENTIVE = "health_assessment_preventive"
    NUTRITION_PREVENTIVE = "nutrition_preventive"
    WELLNESS_TRACKING_PREVENTIVE = "wellness_tracking_preventive"
    
    # Wellness path nodes
    WELLNESS_OPTIMIZATION = "wellness_optimization"
    NUTRITION_WELLNESS = "nutrition_wellness"
    LIFESTYLE_ENRICHMENT = "lifestyle_enrichment"
    
    # Final node
    OUTPUT_AGGREGATOR = "output_aggregator"


# ==========================================
# GRAPH BUILDING FUNCTION
# ==========================================

def build_petcare_graph(client: OpenAIClient) -> StateGraph:
    """
    Build complete LangGraph state machine with all nodes and edges.
    
    This function constructs the entire workflow graph with:
    - 17 nodes (validation, extraction, ML, routing, path-specific, aggregation)
    - Sequential edges for linear progression
    - Parallel edges for ML scoring
    - Conditional edges for path routing
    - Path-specific sequences (3-5 nodes per path)
    - Convergence to output aggregator
    
    Args:
        client: OpenAIClient instance for LLM nodes
        
    Returns:
        Compiled LangGraph.StateGraph ready for invocation
    """
    logger.info("=" * 60)
    logger.info("BUILDING PAWCARE+ LANGGRAPH WORKFLOW")
    logger.info("=" * 60)
    
    # ==========================================
    # CREATE GRAPH WITH STATE SCHEMA
    # ==========================================
    graph = StateGraph(PetCareState)
    logger.info("✓ Created StateGraph with PetCareState schema")
    
    # ==========================================
    # ADD ALL 17 NODES
    # ==========================================
    logger.info("\n📦 Adding 17 nodes to graph...")
    
    # Core processing nodes (direct function references)
    graph.add_node(NodeNames.INPUT_VALIDATOR, input_validator_node)
    logger.debug(f"  Added {NodeNames.INPUT_VALIDATOR}")
    
    # LLM nodes need client injection via lambda
    graph.add_node(
        NodeNames.PROFILE_EXTRACTOR,
        lambda state: pet_profile_extractor_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.PROFILE_EXTRACTOR} (with client)")
    
    # ML nodes (no client needed)
    graph.add_node(NodeNames.HEALTH_RISK_SCORER, pet_health_risk_scorer_node)
    logger.debug(f"  Added {NodeNames.HEALTH_RISK_SCORER}")
    
    graph.add_node(NodeNames.CARE_CAPABILITY_SCORER, owner_care_capability_node)
    logger.debug(f"  Added {NodeNames.CARE_CAPABILITY_SCORER}")
    
    # Router node
    graph.add_node(NodeNames.HEALTH_RISK_ROUTER, health_risk_router_node)
    logger.debug(f"  Added {NodeNames.HEALTH_RISK_ROUTER}")
    
    # CRITICAL PATH NODES (5 nodes - all need client)
    graph.add_node(
        NodeNames.RISK_ANALYSIS,
        lambda state: pet_health_risk_analysis_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.RISK_ANALYSIS} (with client)")
    
    graph.add_node(
        NodeNames.EMERGENCY_PREP,
        lambda state: emergency_preparedness_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.EMERGENCY_PREP} (with client)")
    
    graph.add_node(
        NodeNames.NUTRITION_CRITICAL,
        lambda state: nutrition_critical_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.NUTRITION_CRITICAL} (with client)")
    
    graph.add_node(
        NodeNames.BEHAVIORAL_COACHING,
        lambda state: behavioral_coaching_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.BEHAVIORAL_COACHING} (with client)")
    
    graph.add_node(
        NodeNames.WELLNESS_MONITORING,
        lambda state: wellness_monitoring_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.WELLNESS_MONITORING} (with client)")
    
    # PREVENTIVE PATH NODES (3 nodes - all need client)
    graph.add_node(
        NodeNames.HEALTH_ASSESSMENT_PREVENTIVE,
        lambda state: health_assessment_preventive_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.HEALTH_ASSESSMENT_PREVENTIVE} (with client)")
    
    graph.add_node(
        NodeNames.NUTRITION_PREVENTIVE,
        lambda state: nutrition_preventive_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.NUTRITION_PREVENTIVE} (with client)")
    
    graph.add_node(
        NodeNames.WELLNESS_TRACKING_PREVENTIVE,
        lambda state: wellness_tracking_preventive_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.WELLNESS_TRACKING_PREVENTIVE} (with client)")
    
    # WELLNESS PATH NODES (3 nodes - all need client)
    graph.add_node(
        NodeNames.WELLNESS_OPTIMIZATION,
        lambda state: wellness_optimization_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.WELLNESS_OPTIMIZATION} (with client)")
    
    graph.add_node(
        NodeNames.NUTRITION_WELLNESS,
        lambda state: nutrition_wellness_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.NUTRITION_WELLNESS} (with client)")
    
    graph.add_node(
        NodeNames.LIFESTYLE_ENRICHMENT,
        lambda state: lifestyle_enrichment_node(state, client)
    )
    logger.debug(f"  Added {NodeNames.LIFESTYLE_ENRICHMENT} (with client)")
    
    # Final aggregation node
    graph.add_node(NodeNames.OUTPUT_AGGREGATOR, output_aggregator_node)
    logger.debug(f"  Added {NodeNames.OUTPUT_AGGREGATOR}")
    
    logger.info(f"✓ Added all 17 nodes to graph")
    
    # ==========================================
    # DEFINE EDGES (CONNECTIONS)
    # ==========================================
    logger.info("\n🔗 Defining graph edges...")
    
    # Start -> Input Validation
    graph.set_entry_point(NodeNames.INPUT_VALIDATOR)
    logger.debug(f"  Entry point: {NodeNames.INPUT_VALIDATOR}")
    
    # Input Validation -> Profile Extractor (if valid) or END (if invalid)
    graph.add_edge(NodeNames.INPUT_VALIDATOR, NodeNames.PROFILE_EXTRACTOR)
    logger.debug(f"  Edge: {NodeNames.INPUT_VALIDATOR} → {NodeNames.PROFILE_EXTRACTOR}")
    
    # Profile Extractor -> Both ML scorers (parallel execution)
    graph.add_edge(NodeNames.PROFILE_EXTRACTOR, NodeNames.HEALTH_RISK_SCORER)
    graph.add_edge(NodeNames.PROFILE_EXTRACTOR, NodeNames.CARE_CAPABILITY_SCORER)
    logger.debug(f"  Parallel edges: {NodeNames.PROFILE_EXTRACTOR} → {NodeNames.HEALTH_RISK_SCORER} and {NodeNames.CARE_CAPABILITY_SCORER}")
    
    # Both ML scorers -> Router
    graph.add_edge(NodeNames.HEALTH_RISK_SCORER, NodeNames.HEALTH_RISK_ROUTER)
    graph.add_edge(NodeNames.CARE_CAPABILITY_SCORER, NodeNames.HEALTH_RISK_ROUTER)
    logger.debug(f"  Merge edges: ML scorers → {NodeNames.HEALTH_RISK_ROUTER}")
    
    # Router -> Conditional paths (handled by conditional edges)
    # This will be defined after the conditional routing function
    
    # ==========================================
    # CRITICAL PATH EDGES
    # ==========================================
    graph.add_edge(NodeNames.RISK_ANALYSIS, NodeNames.EMERGENCY_PREP)
    graph.add_edge(NodeNames.EMERGENCY_PREP, NodeNames.NUTRITION_CRITICAL)
    graph.add_edge(NodeNames.NUTRITION_CRITICAL, NodeNames.BEHAVIORAL_COACHING)
    graph.add_edge(NodeNames.BEHAVIORAL_COACHING, NodeNames.WELLNESS_MONITORING)
    graph.add_edge(NodeNames.WELLNESS_MONITORING, NodeNames.OUTPUT_AGGREGATOR)
    logger.debug(f"  Critical path edges: {NodeNames.RISK_ANALYSIS} → ... → {NodeNames.OUTPUT_AGGREGATOR}")
    
    # ==========================================
    # PREVENTIVE PATH EDGES
    # ==========================================
    graph.add_edge(NodeNames.HEALTH_ASSESSMENT_PREVENTIVE, NodeNames.NUTRITION_PREVENTIVE)
    graph.add_edge(NodeNames.NUTRITION_PREVENTIVE, NodeNames.WELLNESS_TRACKING_PREVENTIVE)
    graph.add_edge(NodeNames.WELLNESS_TRACKING_PREVENTIVE, NodeNames.OUTPUT_AGGREGATOR)
    logger.debug(f"  Preventive path edges: {NodeNames.HEALTH_ASSESSMENT_PREVENTIVE} → ... → {NodeNames.OUTPUT_AGGREGATOR}")
    
    # ==========================================
    # WELLNESS PATH EDGES
    # ==========================================
    graph.add_edge(NodeNames.WELLNESS_OPTIMIZATION, NodeNames.NUTRITION_WELLNESS)
    graph.add_edge(NodeNames.NUTRITION_WELLNESS, NodeNames.LIFESTYLE_ENRICHMENT)
    graph.add_edge(NodeNames.LIFESTYLE_ENRICHMENT, NodeNames.OUTPUT_AGGREGATOR)
    logger.debug(f"  Wellness path edges: {NodeNames.WELLNESS_OPTIMIZATION} → ... → {NodeNames.OUTPUT_AGGREGATOR}")
    
    # ==========================================
    # FINAL EDGE
    # ==========================================
    graph.add_edge(NodeNames.OUTPUT_AGGREGATOR, END)
    logger.debug(f"  Final edge: {NodeNames.OUTPUT_AGGREGATOR} → END")
    
    # ==========================================
    # CONDITIONAL ROUTING FUNCTION
    # ==========================================
    def route_after_router(state: PetCareState) -> Literal[
        NodeNames.RISK_ANALYSIS,
        NodeNames.HEALTH_ASSESSMENT_PREVENTIVE,
        NodeNames.WELLNESS_OPTIMIZATION
    ]:
        """
        Conditional routing function that directs flow based on path_taken.
        
        Args:
            state: Current state with path_taken field
            
        Returns:
            Name of the next node to execute
        """
        path = state.get("path_taken", PREVENTIVE_CARE_PATH)
        
        if path == CRITICAL_CARE_PATH:
            logger.info(f"🔴 Routing to critical path: {NodeNames.RISK_ANALYSIS}")
            return NodeNames.RISK_ANALYSIS
        elif path == PREVENTIVE_CARE_PATH:
            logger.info(f"🟡 Routing to preventive path: {NodeNames.HEALTH_ASSESSMENT_PREVENTIVE}")
            return NodeNames.HEALTH_ASSESSMENT_PREVENTIVE
        else:  # WELLNESS_PATH
            logger.info(f"🟢 Routing to wellness path: {NodeNames.WELLNESS_OPTIMIZATION}")
            return NodeNames.WELLNESS_OPTIMIZATION
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        NodeNames.HEALTH_RISK_ROUTER,
        route_after_router,
        {
            NodeNames.RISK_ANALYSIS: NodeNames.RISK_ANALYSIS,
            NodeNames.HEALTH_ASSESSMENT_PREVENTIVE: NodeNames.HEALTH_ASSESSMENT_PREVENTIVE,
            NodeNames.WELLNESS_OPTIMIZATION: NodeNames.WELLNESS_OPTIMIZATION
        }
    )
    logger.debug(f"  Conditional edges from {NodeNames.HEALTH_RISK_ROUTER} to all path starts")
    
    logger.info("✓ All edges defined successfully")
    
    # ==========================================
    # COMPILE GRAPH
    # ==========================================
    logger.info("\n⚙️ Compiling graph...")
    
    # Use MemorySaver for checkpointing
    memory_saver = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory_saver)
    
    logger.info("✓ Graph compiled successfully with MemorySaver")
    logger.info(f"\n✅ Graph built with {len(graph.nodes)} nodes and multiple edges")
    
    return compiled_graph


# ==========================================
# MAIN ENTRY POINT FUNCTION
# ==========================================

def assess_pet_health(form_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Main entry point for pet health assessment workflow.
    
    This function:
    1. Adds metadata to form data (request_id, timestamp)
    2. Initializes state with get_initial_state()
    3. Creates OpenAI client
    4. Builds graph via build_petcare_graph()
    5. Invokes graph with initial state
    6. Returns final state with all results
    
    Args:
        form_data: Dictionary with keys:
            - about_pet: Description of pet characteristics
            - daily_routine: Description of daily activities
            - health_concerns: Description of health issues
            
    Returns:
        Complete PetCareState dictionary with all results
    """
    logger.info("=" * 60)
    logger.info("PAWCARE+ PET HEALTH ASSESSMENT STARTED")
    logger.info("=" * 60)
    
    # ==========================================
    # ADD METADATA TO FORM DATA
    # ==========================================
    enriched_form_data = form_data.copy()
    enriched_form_data["request_id"] = str(uuid.uuid4())
    enriched_form_data["timestamp"] = datetime.now().isoformat()
    
    logger.info(f"📋 Request ID: {enriched_form_data['request_id']}")
    logger.debug(f"📅 Timestamp: {enriched_form_data['timestamp']}")
    
    # ==========================================
    # INITIALIZE STATE
    # ==========================================
    logger.info("\n📦 Initializing state...")
    initial_state = get_initial_state(enriched_form_data)
    logger.info(f"✓ State initialized with {len(initial_state)} fields")
    
    # ==========================================
    # CREATE OPENAI CLIENT
    # ==========================================
    logger.info("\n🔧 Creating OpenAI client...")
    try:
        client = build_openai_client()
        logger.info("✓ OpenAI client created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create OpenAI client: {str(e)}")
        initial_state["error_occurred"] = True
        initial_state["error_messages"].append(f"OpenAI client creation failed: {str(e)}")
        initial_state["processing_complete"] = False
        return initial_state
    
    # ==========================================
    # BUILD GRAPH
    # ==========================================
    logger.info("\n🔨 Building workflow graph...")
    try:
        graph = build_petcare_graph(client)
        logger.info("✓ Graph built successfully")
    except Exception as e:
        logger.error(f"❌ Failed to build graph: {str(e)}", exc_info=True)
        initial_state["error_occurred"] = True
        initial_state["error_messages"].append(f"Graph building failed: {str(e)}")
        initial_state["processing_complete"] = False
        return initial_state
    
    # ==========================================
    # INVOKE GRAPH
    # ==========================================
    logger.info("\n🚀 Invoking workflow graph...")
    
    try:
        # Execute the graph
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": enriched_form_data["request_id"]}}
        )
        
        # Mark as complete
        final_state["processing_complete"] = True
        
        logger.info(f"\n✅ Workflow completed successfully")
        logger.info(f"📊 Final path: {final_state.get('path_taken', 'unknown')}")
        
        # Log summary statistics
        if final_state.get("error_occurred"):
            logger.warning(f"⚠️ Workflow completed with {len(final_state.get('error_messages', []))} errors")
        else:
            logger.info(f"✨ Workflow completed with no errors")
        
        return final_state
        
    except Exception as e:
        # Handle graph execution errors
        error_msg = f"Graph execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        
        # Update state with error
        initial_state["error_occurred"] = True
        initial_state["error_messages"].append(error_msg)
        initial_state["processing_complete"] = False
        initial_state["processing_stage"] = "execution_failed"
        
        return initial_state


# ==========================================
# SUMMARY EXTRACTION FUNCTION
# ==========================================

def get_pet_health_summary(assessment_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and structure key findings from assessment result.
    
    This function builds a structured summary with:
    - Pet profile information
    - Health assessment scores
    - Path analysis and urgency
    - Path-specific outputs
    - System metadata
    
    Args:
        assessment_result: Complete state dictionary from assessment
        
    Returns:
        Dictionary with structured summary:
            - pet_profile: Extracted profile fields
            - health_assessment: ML predictions
            - path_analysis: Care path and urgency
            - outputs: Path-specific outputs
            - system: Metadata (request_id, timestamp, errors)
    """
    logger.debug("Building pet health summary from assessment result")
    
    # ==========================================
    # PET PROFILE SECTION
    # ==========================================
    pet_profile = {
        "name": assessment_result.get("pet_name", "Your Pet"),
        "species": assessment_result.get("pet_species", "unknown"),
        "breed": assessment_result.get("breed", "unknown"),
        "age_years": assessment_result.get("age_years", 0),
        "weight_status": assessment_result.get("weight_status", "unknown"),
        "sex": assessment_result.get("sex", "unknown"),
        "known_conditions": assessment_result.get("known_conditions", []),
        "medications_current": assessment_result.get("medications_current", []),
        "allergies_known": assessment_result.get("allergies_known", []),
        "behavioral_issues": assessment_result.get("behavioral_issues", [])
    }
    
    # ==========================================
    # HEALTH ASSESSMENT SECTION
    # ==========================================
    health_assessment = {
        "health_risk_score": assessment_result.get("health_risk_score"),
        "care_capability_score": assessment_result.get("care_capability_score"),
        "health_risk_factors": assessment_result.get("health_risk_factors", {}),
        "care_capability_factors": assessment_result.get("care_capability_factors", {}),
        "extraction_confidence": assessment_result.get("extraction_confidence", 0.0)
    }
    
    # ==========================================
    # PATH ANALYSIS SECTION
    # ==========================================
    path_taken = assessment_result.get("path_taken", "unknown")
    
    # Determine urgency level based on path
    urgency_map = {
        "CRITICAL_CARE_PATH": "IMMEDIATE - Veterinary attention required within 24 hours",
        "PREVENTIVE_CARE_PATH": "SCHEDULED - Schedule vet visit within 1-2 weeks",
        "WELLNESS_PATH": "ROUTINE - Regular wellness maintenance"
    }
    
    path_analysis = {
        "path_taken": path_taken,
        "urgency": urgency_map.get(path_taken, "Unknown"),
        "path_decision_rationale": assessment_result.get("path_decision_rationale", ""),
        "path_thresholds_used": assessment_result.get("path_thresholds_used", {})
    }
    
    # ==========================================
    # PATH-SPECIFIC OUTPUTS SECTION
    # ==========================================
    outputs = {}
    
    if path_taken == "CRITICAL_CARE_PATH":
        outputs = {
            "health_risk_analysis": assessment_result.get("health_risk_analysis_output", {}),
            "emergency_preparedness": assessment_result.get("emergency_prep_output", {}),
            "nutrition_plan": assessment_result.get("nutrition_critical_output", {}),
            "behavioral_coaching": assessment_result.get("behavioral_coaching_output", {}),
            "wellness_monitoring": assessment_result.get("wellness_monitoring_output", {})
        }
    elif path_taken == "PREVENTIVE_CARE_PATH":
        outputs = {
            "health_assessment": assessment_result.get("health_assessment_output", {}),
            "nutrition_guide": assessment_result.get("nutrition_preventive_output", {}),
            "wellness_tracking": assessment_result.get("wellness_tracking_output", {})
        }
    elif path_taken == "WELLNESS_PATH":
        outputs = {
            "wellness_optimization": assessment_result.get("wellness_optimization_output", {}),
            "nutrition_enhancement": assessment_result.get("nutrition_wellness_output", {}),
            "lifestyle_enrichment": assessment_result.get("lifestyle_enrichment_output", {})
        }
    else:
        # Unknown path - collect all available outputs
        outputs = {
            "health_risk_analysis": assessment_result.get("health_risk_analysis_output", {}),
            "emergency_preparedness": assessment_result.get("emergency_prep_output", {}),
            "critical_nutrition": assessment_result.get("nutrition_critical_output", {}),
            "behavioral_coaching": assessment_result.get("behavioral_coaching_output", {}),
            "wellness_monitoring": assessment_result.get("wellness_monitoring_output", {}),
            "preventive_assessment": assessment_result.get("health_assessment_output", {}),
            "preventive_nutrition": assessment_result.get("nutrition_preventive_output", {}),
            "wellness_tracking": assessment_result.get("wellness_tracking_output", {}),
            "wellness_optimization": assessment_result.get("wellness_optimization_output", {}),
            "wellness_nutrition": assessment_result.get("nutrition_wellness_output", {}),
            "lifestyle_enrichment": assessment_result.get("lifestyle_enrichment_output", {})
        }
    
    # ==========================================
    # SYSTEM METADATA SECTION
    # ==========================================
    system = {
        "request_id": assessment_result.get("request_id", "unknown"),
        "analysis_timestamp": assessment_result.get("analysis_timestamp", "unknown"),
        "processing_complete": assessment_result.get("processing_complete", False),
        "processing_stage": assessment_result.get("processing_stage", "unknown"),
        "error_occurred": assessment_result.get("error_occurred", False),
        "error_count": len(assessment_result.get("error_messages", [])),
        "errors": assessment_result.get("error_messages", []) if assessment_result.get("error_occurred") else []
    }
    
    # ==========================================
    # COMPLETE SUMMARY
    # ==========================================
    summary = {
        "pet_profile": pet_profile,
        "health_assessment": health_assessment,
        "path_analysis": path_analysis,
        "outputs": outputs,
        "system": system
    }
    
    logger.debug(f"Summary built with {len(outputs)} output sections")
    
    return summary


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_graph_info() -> Dict[str, Any]:
    """
    Get information about the graph structure.
    
    Returns:
        Dictionary with graph metadata
    """
    return {
        "total_nodes": 17,
        "node_names": list(NodeNames.__dict__.values() if hasattr(NodeNames, '__dict__') else []),
        "paths": {
            "critical": [NodeNames.RISK_ANALYSIS, NodeNames.EMERGENCY_PREP, 
                        NodeNames.NUTRITION_CRITICAL, NodeNames.BEHAVIORAL_COACHING, 
                        NodeNames.WELLNESS_MONITORING],
            "preventive": [NodeNames.HEALTH_ASSESSMENT_PREVENTIVE, NodeNames.NUTRITION_PREVENTIVE, 
                          NodeNames.WELLNESS_TRACKING_PREVENTIVE],
            "wellness": [NodeNames.WELLNESS_OPTIMIZATION, NodeNames.NUTRITION_WELLNESS, 
                        NodeNames.LIFESTYLE_ENRICHMENT]
        },
        "parallel_stages": [
            {
                "nodes": [NodeNames.HEALTH_RISK_SCORER, NodeNames.CARE_CAPABILITY_SCORER],
                "merge_point": NodeNames.HEALTH_RISK_ROUTER
            }
        ]
    }


def reset_graph_cache():
    """
    Reset any cached graph instances.
    Useful for testing or when client needs to be refreshed.
    """
    global _graph_cache
    _graph_cache = None
    logger.info("Graph cache reset")


# Graph cache for singleton pattern
_graph_cache = None


def get_cached_graph(client: Optional[OpenAIClient] = None) -> StateGraph:
    """
    Get or create cached graph instance (singleton pattern).
    
    Args:
        client: OpenAIClient instance (required for first build)
        
    Returns:
        Compiled graph
    """
    global _graph_cache
    
    if _graph_cache is None:
        if client is None:
            raise ValueError("Client required for initial graph build")
        _graph_cache = build_petcare_graph(client)
        logger.info("Graph built and cached")
    
    return _graph_cache


# ==========================================
# COMMAND LINE INTERFACE (for testing)
# ==========================================

if __name__ == "__main__":
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PAWCARE+ GRAPH ORCHESTRATION MODULE TEST")
    print("=" * 60)
    
    # Test graph building
    print("\n📊 Testing graph building...")
    try:
        client = build_openai_client()
        graph = build_petcare_graph(client)
        print(f"✅ Graph built successfully")
        print(f"   Nodes: {len(graph.nodes)}")
        
        # Test assess_pet_health with sample data
        print("\n📝 Testing assess_pet_health with sample data...")
        test_form = {
            "about_pet": "My dog is a 5-year-old Labrador Retriever named Max.",
            "daily_routine": "He eats twice daily, walks for 30 minutes, and sleeps indoors.",
            "health_concerns": "Recently noticed increased thirst and lethargy."
        }
        
        result = assess_pet_health(test_form)
        print(f"✅ Assessment completed")
        print(f"   Path taken: {result.get('path_taken', 'unknown')}")
        print(f"   Errors: {result.get('error_occurred', False)}")
        
        # Test summary extraction
        print("\n📋 Testing summary extraction...")
        summary = get_pet_health_summary(result)
        print(f"✅ Summary extracted")
        print(f"   Pet: {summary['pet_profile']['species']} {summary['pet_profile']['breed']}")
        print(f"   Risk score: {summary['health_assessment']['health_risk_score']}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")