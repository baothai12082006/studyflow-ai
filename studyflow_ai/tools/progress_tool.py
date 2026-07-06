"""
Progress Monitoring Tool for StudyFlow AI.
Updates and fetches study streaks using the repository layer.
"""
import logging
from google.adk import tool
from studyflow_ai.repositories.in_memory import InMemoryProgressRepository
from studyflow_ai.models.state import ProgressState

logger = logging.getLogger(__name__)

# Global mock DB persistence registry
progress_repo = InMemoryProgressRepository()

@tool
async def fetch_user_progress(user_id: str) -> ProgressState:
    """
    Fetches the current progress state (streaks, tasks completed) for a user.
    
    Args:
        user_id: Student unique ID
    """
    try:
        state = await progress_repo.get_progress(user_id)
        if not state:
            state = ProgressState(user_id=user_id)
            await progress_repo.save_progress(state)
        return state
    except Exception as e:
        logger.error(f"Error fetching progress: {e}")
        return ProgressState(user_id=user_id)

@tool
async def complete_study_task(user_id: str, task_topic: str) -> ProgressState:
    """
    Marks a study task as completed, increments progress, and recalculates streaks.
    
    Args:
        user_id: Student unique ID
        task_topic: Academic subject topic of the task
    """
    try:
        state = await progress_repo.get_progress(user_id)
        if not state:
            state = ProgressState(user_id=user_id)
        
        state.total_tasks_completed += 1
        state.current_streak_days += 1
        if state.current_streak_days > state.highest_streak_days:
            state.highest_streak_days = state.current_streak_days
            
        if task_topic not in state.struggle_topics:
            state.struggle_topics.append(task_topic)
            
        await progress_repo.save_progress(state)
        return state
    except Exception as e:
        logger.error(f"Error completing study task: {e}")
        raise RuntimeError(f"Failed to update progress: {e}")
