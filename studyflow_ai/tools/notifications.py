# Notification Dispatch Tool Wrapper
from google.adk import tool
from studyflow_ai.services.notification_service import NotificationService

@tool
async def push_notification_tool(user_id: str, message: str, priority: str = "NORMAL") -> dict:
    """Sends web or push notification to student UI clients.
    
    Args:
        user_id: Recipient ID
        message: Notification body
        priority: NORMAL or HIGH
    """
    service = NotificationService()
    status = await service.send_push(user_id, message, priority)
    return {"delivery_status": status}
