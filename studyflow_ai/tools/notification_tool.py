"""
Notification Dispatch Tool for StudyFlow AI.
Sends real-time messages and notifications to students using NotificationService.
"""
import logging
from google.adk import tool
from studyflow_ai.services.notification_service import NotificationService

logger = logging.getLogger(__name__)

# Mock config
notification_service = NotificationService(smtp_host="mock", smtp_port=587)

@tool
async def send_immediate_reminder(user_id: str, alert_message: str, priority: str = "NORMAL") -> bool:
    """
    Sends an immediate push reminder notification to the student.
    
    Args:
        user_id: Recipient user UUID
        alert_message: Body copy of the alert
        priority: Priority configuration (NORMAL or HIGH)
    """
    try:
        return await notification_service.send_push_notification(user_id, alert_message, priority)
    except Exception as e:
        logger.error(f"Failed to dispatch alert: {e}")
        return False
