"""
Validator Module
Handles fact-checking and validation against web sources
"""
from .fact_checker import get_model_answer, trigger_update_pipeline, trigger_save_stable_fact, run_chatbot_check, run_validation_test
from .web_search import get_web_answer
from .llm_judge import get_clean_fact_from_web, is_answer_outdated_llm_judge

__all__ = ['get_model_answer', 'trigger_update_pipeline', 'trigger_save_stable_fact', 'run_chatbot_check', 'run_validation_test',
           'get_web_answer', 'get_clean_fact_from_web', 'is_answer_outdated_llm_judge', 'get_clean_fact_from_web']