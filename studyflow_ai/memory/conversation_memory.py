# Conversation Memory Store mapping to ADK Session State

class ConversationMemory:
    """Abstraction managing short-term chat context inside the ADK Session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        
    def get_history(self) -> list:
        # TODO: Retrieve conversation history
        return []
        
    def append_message(self, role: str, content: str):
        # TODO: Append conversation turn
        pass
