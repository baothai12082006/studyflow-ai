"""
In-memory mock implementations of the Repository contracts.
"""
from typing import Dict, List, Optional
from studyflow_ai.repositories.base import UserRepository, AcademicRepository, ProgressRepository
from studyflow_ai.models.state import UserState, AcademicState, ProgressState
from studyflow_ai.models.schemas import StudyTaskSchema

class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self._store: Dict[str, UserState] = {}

    async def get_by_id(self, user_id: str) -> Optional[UserState]:
        return self._store.get(user_id)

    async def save(self, user_state: UserState) -> None:
        self._store[user_state.user_id] = user_state

class InMemoryAcademicRepository(AcademicRepository):
    def __init__(self):
        self._courses: Dict[str, AcademicState] = {}
        self._tasks: Dict[str, List[StudyTaskSchema]] = {}

    async def get_course_state(self, course_id: str) -> Optional[AcademicState]:
        return self._courses.get(course_id)

    async def save_course_state(self, academic_state: AcademicState) -> None:
        self._courses[academic_state.course_id] = academic_state

    async def get_study_tasks(self, user_id: str) -> List[StudyTaskSchema]:
        return self._tasks.get(user_id, [])

    async def save_study_tasks(self, tasks: List[StudyTaskSchema]) -> None:
        if not tasks:
            return
        user_id = tasks[0].task_id.split("_")[0]  # Dummy extraction for user id
        self._tasks[user_id] = tasks

    async def update_task_status(self, task_id: str, is_completed: bool) -> None:
        for user_id, tasks in self._tasks.items():
            for task in tasks:
                if task.task_id == task_id:
                    task.is_completed = is_completed
                    return

class InMemoryProgressRepository(ProgressRepository):
    def __init__(self):
        self._store: Dict[str, ProgressState] = {}

    async def get_progress(self, user_id: str) -> Optional[ProgressState]:
        return self._store.get(user_id)

    async def save_progress(self, progress: ProgressState) -> None:
        self._store[progress.user_id] = progress
