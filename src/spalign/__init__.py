"""
spalign: Conversation generation library with vLLM and OpenAI integration
=========================================================================

A library for generating multi-turn conversations using local vLLM models
and OpenAI GPT models for persona characters.
"""

# from .batcher import VLLMBatcher
# from .conversation import ConversationGenerator
from .database import (
    get_completed_results,
    get_failed_scenarios,
    get_pending_scenarios,
    get_progress_stats,
    get_scenario_hash,
    init_db,
    insert_pending_scenarios,
    mark_completed,
    mark_failed,
    reset_failed_to_pending,
)
from .models import PersonaParams, PersonaResponse
from .persona import PersonaGenerator
from .utils import history_to_msgs, parse_role, strip_tags

__version__ = "0.1.0"

__all__ = [
    # Core classes
    # "ConversationGenerator",
    # "VLLMBatcher",
    "PersonaGenerator",
    # Models
    "PersonaResponse",
    "PersonaParams",
    # Database functions
    "init_db",
    "get_scenario_hash",
    "insert_pending_scenarios",
    "get_pending_scenarios",
    "mark_completed",
    "mark_failed",
    "get_completed_results",
    "get_progress_stats",
    "get_failed_scenarios",
    "reset_failed_to_pending",
    # Utilities
    "strip_tags",
    "parse_role",
    "history_to_msgs",
]
