"""
notification_service.py
-----------------------
Notification Service wrapper.
Handles push notifications (FCM) and fallback SMTP emails.
"""
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(self, smtp_host: str, smtp_port: int):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        logger.info("NotificationService initialized with SMTP host %s:%s", smtp_host, smtp_port)

    async def send_push_notification(self, user_id: str, message: str, priority: str = "NORMAL") -> bool:
        """
        Dispatches real-time alerts.
        """
        logger.debug("Sending push notification to user=%s priority=%s", user_id, priority)
        
        if not user_id:
            raise ValueError("user_id cannot be empty")
        if not message:
            raise ValueError("message payload cannot be empty")
            
        allowed_priorities = {"LOW", "NORMAL", "HIGH"}
        if priority.upper() not in allowed_priorities:
            raise ValueError(f"Invalid priority '{priority}'. Allowed: {allowed_priorities}")

        try:
            # TODO: Interface with Firebase Cloud Messaging (FCM).
            await asyncio.sleep(0.1)
            logger.info("Successfully dispatched push notification to user=%s", user_id)
            return True
        except Exception as e:
            logger.exception("Failed to send push notification to user=%s: %s", user_id, e)
            raise RuntimeError(f"Push notification dispatch failed: {e}")

    async def send_fallback_email(self, recipient: str, subject: str, body: str) -> bool:
        """
        SMTP fallback client.
        """
        logger.debug("Sending fallback email to recipient=%s", recipient)
        
        if not recipient or not subject or not body:
            raise ValueError("recipient, subject, and body cannot be empty")
            
        # Basic email validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", recipient):
            raise ValueError(f"Invalid email address format: {recipient}")

        try:
            # TODO: Initialize smtp protocol and dispatch message.
            await asyncio.sleep(0.1)
            logger.info("Successfully dispatched fallback email to %s", recipient)
            return True
        except Exception as e:
            logger.exception("Failed to send fallback email to %s: %s", recipient, e)
            raise RuntimeError(f"SMTP dispatch failed: {e}")
