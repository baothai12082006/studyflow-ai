"""
Placeholder persistent implementations for PostgreSQL database.
TODO: Replace database client mockup with real SQL queries/SQLAlchemy models in Phase 2.
"""
from typing import Optional, List
from studyflow_ai.repositories.base import UserRepository, AcademicRepository, ProgressRepository
from studyflow_ai.models.state import UserState, AcademicState, ProgressState
from studyflow_ai.models.schemas import StudyTaskSchema

class SQLUserRepository(UserRepository):
    async def get_by_id(self, user_id: str) -> Optional[UserState]:
        # TODO: Query database matching user_id
        return None

    async def save(self, user_state: UserState) -> None:
        # TODO: Upsert user details
        pass

class SQLAcademicRepository(AcademicRepository):
    async def get_course_state(self, course_id: str) -> Optional[AcademicState]:
        # TODO: Select query matching course_id
        return None

    async def save_course_state(self, academic_state: AcademicState) -> None:
        # TODO: Insert course deadlines
        pass

    async def get_study_tasks(self, user_id: str) -> List[StudyTaskSchema]:
        # TODO: Select tasks from database
        return []

    async def save_study_tasks(self, tasks: List[StudyTaskSchema]) -> None:
        # TODO: Bulk insert task schema
        pass

    async def update_task_status(self, task_id: str, is_completed: bool) -> None:
        # TODO: Execute update status SQL query
        pass

class SQLProgressRepository(ProgressRepository):
    async def get_progress(self, user_id: str) -> Optional[ProgressState]:
        # TODO: Select progress matching user_id
        return None

    async def save_progress(self, progress: ProgressState) -> None:
        # TODO: Upsert student streak metric
        pass
