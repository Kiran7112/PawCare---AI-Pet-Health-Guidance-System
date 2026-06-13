# workflow/workflow.py
"""
Workflow Metadata Module for PawCare+ LangGraph orchestration.
Provides workflow configuration, routing logic, parallel execution information,
token budgets, and execution strategy metadata.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS
# ==========================================

# Routing thresholds (must match those in health_risk_router_node)
CRITICAL_THRESHOLD = 0.6
WELLNESS_THRESHOLD = 0.3

# Path names
CRITICAL_CARE_PATH = "CRITICAL_CARE_PATH"
PREVENTIVE_CARE_PATH = "PREVENTIVE_CARE_PATH"
WELLNESS_PATH = "WELLNESS_PATH"


# ==========================================
# FUNCTION: get_routing_logic()
# ==========================================

def get_routing_logic() -> Dict[str, Any]:
    """
    Define routing thresholds and path conditions.
    
    Returns:
        Dictionary containing:
            - routing_field: The field used for routing decisions
            - routing_type: Type of routing (conditional_edges)
            - thresholds: The threshold values used
            - paths: Dictionary with 3 path definitions
            - default_path: Fallback path if routing fails
    """
    logger.debug("Building routing logic configuration")
    
    return {
        "routing_field": "health_risk_score",
        "routing_type": "conditional_edges",
        "thresholds": {
            "critical": CRITICAL_THRESHOLD,
            "wellness": WELLNESS_THRESHOLD
        },
        "paths": {
            CRITICAL_CARE_PATH: {
                "description": "High-risk pets requiring immediate veterinary attention and urgent care guidance",
                "condition": f"health_risk_score > {CRITICAL_THRESHOLD}",
                "execution_strategy": "sequential_critical",
                "token_budget": 12000,
                "nodes_count": 5,
                "agents": [
                    "pet_health_risk_analysis_llm",
                    "emergency_preparedness_llm",
                    "nutrition_critical_llm",
                    "behavioral_coaching_llm",
                    "wellness_monitoring_llm"
                ],
                "icon": "🔴",
                "urgency_level": "IMMEDIATE"
            },
            PREVENTIVE_CARE_PATH: {
                "description": "Moderate-risk pets needing proactive management and preventive measures",
                "condition": f"{WELLNESS_THRESHOLD} < health_risk_score ≤ {CRITICAL_THRESHOLD}",
                "execution_strategy": "sequential_preventive",
                "token_budget": 7500,
                "nodes_count": 3,
                "agents": [
                    "health_assessment_preventive_llm",
                    "nutrition_preventive_llm",
                    "wellness_tracking_preventive_llm"
                ],
                "icon": "🟡",
                "urgency_level": "SCHEDULED"
            },
            WELLNESS_PATH: {
                "description": "Low-risk healthy pets focusing on optimization and longevity",
                "condition": f"health_risk_score ≤ {WELLNESS_THRESHOLD}",
                "execution_strategy": "sequential_wellness",
                "token_budget": 6000,
                "nodes_count": 3,
                "agents": [
                    "wellness_optimization_llm",
                    "nutrition_wellness_llm",
                    "lifestyle_enrichment_llm"
                ],
                "icon": "🟢",
                "urgency_level": "ROUTINE"
            }
        },
        "default_path": PREVENTIVE_CARE_PATH,
        "routing_summary": f"Routes based on health_risk_score: >{CRITICAL_THRESHOLD}=Critical, ≤{WELLNESS_THRESHOLD}=Wellness, else Preventive"
    }


# ==========================================
# FUNCTION: get_parallel_execution_info()
# ==========================================

def get_parallel_execution_info() -> Dict[str, Any]:
    """
    Define which nodes can execute in parallel.
    
    Returns:
        Dictionary containing:
            - parallel_stages: List of stages that can run in parallel
            - sequential_stages: List of stages that must run sequentially
            - dependencies: Dictionary of node dependencies
            - execution_flow: Description of the overall flow
    """
    logger.debug("Building parallel execution configuration")
    
    return {
        "parallel_stages": [
            {
                "name": "ml_scoring",
                "description": "Both ML models can run simultaneously after profile extraction",
                "nodes": ["pet_health_risk_scorer", "owner_care_capability_scorer"],
                "merge_point": "health_risk_router",
                "max_concurrency": 2,
                "estimated_time_savings": "40% reduction in total execution time"
            }
        ],
        "sequential_stages": [
            {
                "name": "initialization",
                "order": 1,
                "nodes": ["input_validator"],
                "description": "First node - must run alone"
            },
            {
                "name": "extraction",
                "order": 2,
                "nodes": ["pet_profile_extractor"],
                "description": "LLM extraction - must complete before ML"
            },
            {
                "name": "ml_scoring",
                "order": 3,
                "nodes": ["pet_health_risk_scorer", "owner_care_capability_scorer"],
                "parallel": True,
                "description": "ML models run in parallel"
            },
            {
                "name": "routing",
                "order": 4,
                "nodes": ["health_risk_router"],
                "description": "Router - depends on both ML nodes"
            },
            {
                "name": "critical_path",
                "order": 5,
                "path": CRITICAL_CARE_PATH,
                "nodes": [
                    "pet_health_risk_analysis",
                    "emergency_preparedness",
                    "nutrition_critical",
                    "behavioral_coaching",
                    "wellness_monitoring"
                ],
                "description": "Critical path nodes - sequential due to dependencies"
            },
            {
                "name": "preventive_path",
                "order": 5,
                "path": PREVENTIVE_CARE_PATH,
                "nodes": [
                    "health_assessment_preventive",
                    "nutrition_preventive",
                    "wellness_tracking_preventive"
                ],
                "description": "Preventive path nodes - sequential due to dependencies"
            },
            {
                "name": "wellness_path",
                "order": 5,
                "path": WELLNESS_PATH,
                "nodes": [
                    "wellness_optimization",
                    "nutrition_wellness",
                    "lifestyle_enrichment"
                ],
                "description": "Wellness path nodes - sequential due to dependencies"
            },
            {
                "name": "aggregation",
                "order": 6,
                "nodes": ["output_aggregator"],
                "description": "Final aggregation - runs after path completion"
            }
        ],
        "dependencies": {
            "pet_health_risk_scorer": ["pet_profile_extractor"],
            "owner_care_capability_scorer": ["pet_profile_extractor"],
            "health_risk_router": ["pet_health_risk_scorer", "owner_care_capability_scorer"],
            "pet_health_risk_analysis": ["health_risk_router"],
            "emergency_preparedness": ["pet_health_risk_analysis"],
            "nutrition_critical": ["emergency_preparedness"],
            "behavioral_coaching": ["nutrition_critical"],
            "wellness_monitoring": ["behavioral_coaching"],
            "health_assessment_preventive": ["health_risk_router"],
            "nutrition_preventive": ["health_assessment_preventive"],
            "wellness_tracking_preventive": ["nutrition_preventive"],
            "wellness_optimization": ["health_risk_router"],
            "nutrition_wellness": ["wellness_optimization"],
            "lifestyle_enrichment": ["nutrition_wellness"],
            "output_aggregator": [
                "wellness_monitoring",
                "wellness_tracking_preventive",
                "lifestyle_enrichment"
            ]
        },
        "execution_flow": """
        1. Input Validation (sequential)
        2. Profile Extraction (sequential)
        3. ML Scoring (parallel) ← Parallel execution opportunity
           - Health Risk Scorer
           - Care Capability Scorer
        4. Health Risk Router (sequential - depends on both ML nodes)
        5. Path-Specific Execution (sequential within each path)
           - Critical Path: 5 nodes
           - Preventive Path: 3 nodes
           - Wellness Path: 3 nodes
        6. Output Aggregation (sequential)
        """
    }


# ==========================================
# FUNCTION: get_token_budget_info()
# ==========================================

def get_token_budget_info() -> Dict[str, Any]:
    """
    Define token budgets for different components.
    
    Returns:
        Dictionary containing:
            - total_tokens_per_path: Budget per care path
            - tokens_per_agent: Budget for each LLM agent
            - ml_tokens: Budget for ML operations (negligible)
            - estimated_costs: Approximate cost estimates
    """
    logger.debug("Building token budget configuration")
    
    # Token budgets per agent (in tokens)
    tokens_per_agent = {
        # Extraction agent
        "pet_profile_extractor_llm": {
            "max_input": 500,
            "max_output": 900,
            "total": 1400,
            "description": "Extracts 17 structured fields from natural language"
        },
        
        # Critical path agents (5)
        "pet_health_risk_analysis_llm": {
            "max_input": 1500,
            "max_output": 2000,
            "total": 3500,
            "description": "Detailed health risk analysis"
        },
        "emergency_preparedness_llm": {
            "max_input": 1200,
            "max_output": 1800,
            "total": 3000,
            "description": "Emergency preparedness planning"
        },
        "nutrition_critical_llm": {
            "max_input": 1000,
            "max_output": 1500,
            "total": 2500,
            "description": "Critical nutrition planning"
        },
        "behavioral_coaching_llm": {
            "max_input": 800,
            "max_output": 1200,
            "total": 2000,
            "description": "Behavioral coaching"
        },
        "wellness_monitoring_llm": {
            "max_input": 800,
            "max_output": 1200,
            "total": 2000,
            "description": "Wellness monitoring planning"
        },
        
        # Preventive path agents (3)
        "health_assessment_preventive_llm": {
            "max_input": 600,
            "max_output": 400,
            "total": 1000,
            "description": "Preventive health assessment"
        },
        "nutrition_preventive_llm": {
            "max_input": 500,
            "max_output": 350,
            "total": 850,
            "description": "Preventive nutrition guidance"
        },
        "wellness_tracking_preventive_llm": {
            "max_input": 500,
            "max_output": 300,
            "total": 800,
            "description": "Wellness tracking planning"
        },
        
        # Wellness path agents (3)
        "wellness_optimization_llm": {
            "max_input": 500,
            "max_output": 350,
            "total": 850,
            "description": "Wellness optimization"
        },
        "nutrition_wellness_llm": {
            "max_input": 500,
            "max_output": 350,
            "total": 850,
            "description": "Nutrition enhancement"
        },
        "lifestyle_enrichment_llm": {
            "max_input": 500,
            "max_output": 350,
            "total": 850,
            "description": "Lifestyle enrichment"
        }
    }
    
    # Calculate path totals
    total_tokens_per_path = {
        CRITICAL_CARE_PATH: {
            "total": sum([
                tokens_per_agent["pet_health_risk_analysis_llm"]["total"],
                tokens_per_agent["emergency_preparedness_llm"]["total"],
                tokens_per_agent["nutrition_critical_llm"]["total"],
                tokens_per_agent["behavioral_coaching_llm"]["total"],
                tokens_per_agent["wellness_monitoring_llm"]["total"]
            ]),
            "breakdown": {
                "analysis": tokens_per_agent["pet_health_risk_analysis_llm"]["total"],
                "emergency": tokens_per_agent["emergency_preparedness_llm"]["total"],
                "nutrition": tokens_per_agent["nutrition_critical_llm"]["total"],
                "behavioral": tokens_per_agent["behavioral_coaching_llm"]["total"],
                "monitoring": tokens_per_agent["wellness_monitoring_llm"]["total"]
            }
        },
        PREVENTIVE_CARE_PATH: {
            "total": sum([
                tokens_per_agent["health_assessment_preventive_llm"]["total"],
                tokens_per_agent["nutrition_preventive_llm"]["total"],
                tokens_per_agent["wellness_tracking_preventive_llm"]["total"]
            ]),
            "breakdown": {
                "assessment": tokens_per_agent["health_assessment_preventive_llm"]["total"],
                "nutrition": tokens_per_agent["nutrition_preventive_llm"]["total"],
                "tracking": tokens_per_agent["wellness_tracking_preventive_llm"]["total"]
            }
        },
        WELLNESS_PATH: {
            "total": sum([
                tokens_per_agent["wellness_optimization_llm"]["total"],
                tokens_per_agent["nutrition_wellness_llm"]["total"],
                tokens_per_agent["lifestyle_enrichment_llm"]["total"]
            ]),
            "breakdown": {
                "optimization": tokens_per_agent["wellness_optimization_llm"]["total"],
                "nutrition": tokens_per_agent["nutrition_wellness_llm"]["total"],
                "lifestyle": tokens_per_agent["lifestyle_enrichment_llm"]["total"]
            }
        }
    }
    
    # Add extraction tokens to all paths (common to all)
    extraction_tokens = tokens_per_agent["pet_profile_extractor_llm"]["total"]
    for path in total_tokens_per_path:
        total_tokens_per_path[path]["total_with_extraction"] = (
            total_tokens_per_path[path]["total"] + extraction_tokens
        )
        total_tokens_per_path[path]["extraction_tokens"] = extraction_tokens
    
    return {
        "total_tokens_per_path": total_tokens_per_path,
        "tokens_per_agent": tokens_per_agent,
        "ml_tokens": {
            "description": "ML operations use negligible tokens (local inference)",
            "estimated": "< 10 tokens"
        },
        "estimated_costs": {
            "critical_path": "$0.15 - $0.25 per assessment (at $0.01/1K tokens)",
            "preventive_path": "$0.08 - $0.12 per assessment",
            "wellness_path": "$0.06 - $0.10 per assessment",
            "currency": "USD",
            "assumptions": "Based on GPT-4 pricing, actual costs may vary"
        },
        "token_saving_recommendations": [
            "Cache identical profiles to avoid repeated extraction",
            "Implement prompt caching for repeated patterns",
            "Consider smaller models for less critical paths"
        ]
    }


# ==========================================
# FUNCTION: get_execution_strategy()
# ==========================================

def get_execution_strategy() -> Dict[str, Any]:
    """
    Describe overall workflow execution approach.
    
    Returns:
        Dictionary containing:
            - execution_model: Type of execution model
            - state_management: How state is handled
            - error_handling: Error handling strategy
            - client_pattern: Dependency injection pattern
            - compilation_strategy: Graph compilation approach
            - invocation_method: Entry point description
    """
    logger.debug("Building execution strategy configuration")
    
    return {
        "execution_model": {
            "type": "directed_acyclic_graph",
            "framework": "LangGraph",
            "description": "State machine with conditional routing"
        },
        "state_management": {
            "type": "TypedDict with reducer annotations",
            "fields": 44,
            "reducers": ["keep_first", "set_true", "add_lists", "merge_dicts", "latest_only"],
            "description": "Centralized state with custom reducers for field merging"
        },
        "error_handling": {
            "strategy": "graceful_degradation",
            "patterns": [
                "Try-except in all nodes with error accumulation in state",
                "Fallback values when predictions fail",
                "Graceful skip paths for validation/extraction failures",
                "Error flags that persist through workflow"
            ],
            "recovery": "Workflow continues with best available data"
        },
        "client_pattern": {
            "type": "dependency_injection",
            "implementation": "Lambda functions inject OpenAI client into LLM nodes",
            "benefits": [
                "Client only instantiated once",
                "Easy to swap clients (testing vs production)",
                "Thread-safe client sharing"
            ]
        },
        "compilation_strategy": {
            "type": "singleton_graph",
            "implementation": "Graph compiled once and reused",
            "checkpointing": "MemorySaver for state persistence",
            "benefits": [
                "Reduced overhead",
                "Consistent execution",
                "Debugging support"
            ]
        },
        "invocation_method": {
            "entry_point": "assess_pet_health(user_input)",
            "returns": "Complete PetCareState with aggregated_output",
            "example": """
                from graph import assess_pet_health
                result = assess_pet_health("My 5-year-old Labrador...")
                print(result["aggregated_output"])
            """
        },
        "workflow_summary": {
            "total_nodes": 17,
            "llm_agents": 11,
            "ml_agents": 2,
            "validation_nodes": 1,
            "routing_nodes": 1,
            "aggregation_nodes": 1,
            "parallel_opportunities": 1,
            "estimated_completion_time": "4-10 seconds depending on path"
        }
    }


# ==========================================
# FUNCTION: get_complete_workflow_metadata()
# ==========================================

def get_complete_workflow_metadata() -> Dict[str, Any]:
    """
    Get complete workflow metadata by combining all functions.
    
    Returns:
        Dictionary with all workflow metadata
    """
    return {
        "routing_logic": get_routing_logic(),
        "parallel_execution": get_parallel_execution_info(),
        "token_budgets": get_token_budget_info(),
        "execution_strategy": get_execution_strategy(),
        "version": "1.0.0",
        "last_updated": "2024-01-01"
    }


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_path_description(path_name: str) -> Optional[str]:
    """
    Get description for a specific path.
    
    Args:
        path_name: Name of the path (CRITICAL_CARE_PATH, etc.)
        
    Returns:
        Path description or None if not found
    """
    routing = get_routing_logic()
    paths = routing.get("paths", {})
    path_info = paths.get(path_name, {})
    return path_info.get("description")


def get_path_icon(path_name: str) -> str:
    """
    Get icon for a specific path.
    
    Args:
        path_name: Name of the path
        
    Returns:
        Icon string or empty string
    """
    routing = get_routing_logic()
    paths = routing.get("paths", {})
    path_info = paths.get(path_name, {})
    return path_info.get("icon", "")


def get_node_dependencies(node_name: str) -> List[str]:
    """
    Get dependencies for a specific node.
    
    Args:
        node_name: Name of the node
        
    Returns:
        List of node names that this node depends on
    """
    parallel_info = get_parallel_execution_info()
    dependencies = parallel_info.get("dependencies", {})
    return dependencies.get(node_name, [])


def format_workflow_summary() -> str:
    """
    Generate a human-readable workflow summary.
    
    Returns:
        Formatted summary string
    """
    strategy = get_execution_strategy()
    routing = get_routing_logic()
    parallel = get_parallel_execution_info()
    
    lines = []
    lines.append("=" * 60)
    lines.append("PAWCARE+ WORKFLOW SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\n📊 EXECUTION MODEL:")
    lines.append(f"  Framework: {strategy['execution_model']['framework']}")
    lines.append(f"  Total Nodes: {strategy['workflow_summary']['total_nodes']}")
    lines.append(f"  LLM Agents: {strategy['workflow_summary']['llm_agents']}")
    lines.append(f"  ML Agents: {strategy['workflow_summary']['ml_agents']}")
    
    lines.append(f"\n🔄 ROUTING LOGIC:")
    lines.append(f"  {routing['routing_summary']}")
    
    lines.append(f"\n⚡ PARALLEL EXECUTION:")
    for stage in parallel.get('parallel_stages', []):
        lines.append(f"  • {stage['name']}: {stage['description']}")
        lines.append(f"    Nodes: {', '.join(stage['nodes'])}")
    
    lines.append(f"\n💰 TOKEN BUDGETS:")
    token_info = get_token_budget_info()
    for path, budget in token_info.get('total_tokens_per_path', {}).items():
        lines.append(f"  • {path}: {budget['total']} tokens (with extraction: {budget['total_with_extraction']})")
    
    lines.append(f"\n⏱️  ESTIMATED COMPLETION TIME: {strategy['workflow_summary']['estimated_completion_time']}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

if __name__ == "__main__":
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print(format_workflow_summary())
    
    # Print detailed metadata if requested
    import sys
    if "--verbose" in sys.argv:
        print("\n" + "=" * 60)
        print("DETAILED WORKFLOW METADATA")
        print("=" * 60)
        metadata = get_complete_workflow_metadata()
        print(json.dumps(metadata, indent=2, default=str))