# Repository managing AcademicState and tasks
class AcademicRepository:
    def __init__(self, db_client):
        self.db = db_client

    def save_course_timeline(self, course_id: str, timeline: dict):
        # TODO: Save parsed timeline
        pass

    def update_task_status(self, task_id: str, status: str):
        # TODO: Update task status
        pass
