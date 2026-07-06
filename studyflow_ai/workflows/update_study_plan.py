# Update Study Plan (Rescheduling) Workflow

from google.adk import Workflow

class UpdateStudyPlanWorkflow(Workflow):
    """Triggers a reschedule pass when dates shift or blocks are missed."""
    
    async def run(self, user_id: str, course_id: str):
        # TODO: Formulate path for recalculating tasks
        pass
