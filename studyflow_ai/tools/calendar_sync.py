# Calendar Integration Tools Wrapper
from google.adk import tool
from studyflow_ai.services.calendar_service import CalendarService

@tool
async def read_calendar_tool(user_id: str, start_date: str, end_date: str) -> dict:
    """Queries user availability gaps from Google/Outlook Calendar.
    
    Args:
        user_id: Student unique ID
        start_date: ISO8601 start bound
        end_date: ISO8601 end bound
    """
    # Credentials should be fetched via user settings repository/context
    service = CalendarService(credentials={})
    busy_slots = await service.get_busy_slots(user_id, start_date, end_date)
    return {"busy_slots": busy_slots}

@tool
async def write_calendar_tool(user_id: str, event_title: str, start: str, end: str) -> dict:
    """Injects study block session event into external calendar.
    
    Args:
        user_id: Student unique ID
        event_title: Name of study block
        start: ISO8601 start timestamp
        end: ISO8601 end timestamp
    """
    service = CalendarService(credentials={})
    return await service.create_event(user_id, event_title, start, end)
