"""
Abstract repository interfaces specifying data contract.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from studyflow_ai.models.state import UserState, AcademicState, ProgressState
from studyflow_ai.models.schemas import StudyTaskSchema

class UserRepository(ABC):
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[UserState]:
        pass

    @abstractmethod
    async def save(self, user_state: UserState) -> None:
        pass

class AcademicRepository(ABC):
    @abstractmethod
    async def get_course_state(self, course_id: str) -> Optional[AcademicState]:
        pass

    @abstractmethod
    async def save_course_state(self, academic_state: AcademicState) -> None:
        pass

    @abstractmethod
    async def get_study_tasks(self, user_id: str) -> List[StudyTaskSchema]:
        pass

    @abstractmethod
    async def save_study_tasks(self, tasks: List[StudyTaskSchema]) -> None:
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, is_completed: bool) -> None:
        pass

class ProgressRepository(ABC):
    @abstractmethod
    async def get_progress(self, user_id: str) -> Optional[ProgressState]:
        pass

    @abstractmethod
    async def save_progress(self, progress: ProgressState) -> None:
        pass
