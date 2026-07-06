"""
session.py
----------
Session and State Manager for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Map multi-turn conversation sessions to underlying storage and memory
    context wrappers.
  - Isolation-track individual student chat turns per session_id.
  - Bind three memory wrappers per session:
      ConversationMemory — short-term turn history.
      AcademicMemory    — course deadlines and streak metrics.
      UserMemory        — user preferences and persona profile.
  - Load baseline state from injected repositories asynchronously on session
    creation.
  - Produce context frames ready for LLM prompt grounding.
  - Flush and persist modified session state back through repositories.

Future Compatibility (Documentation Only):
  - TODO(ArtifactService): Cache serialised session snapshots in the
    ArtifactService store so that warm-start resumes skip the repository
    fetch entirely.
  - TODO(DistributedLock): Wrap ``create_session()`` and ``close_session()``
    with a Redis-backed distributed lock (e.g., ``aioredlock``) so that
    concurrent requests for the same session_id never race on state writes.
  - TODO(Observability): Emit structured OpenTelemetry span events at
    session open, context extraction, state save, and session close to
    support end-to-end latency attribution.

References:
  docs/design/multi_agent_architecture.md
  docs/design/state_model.md
  docs/design/interface_contracts.md
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Memory wrappers (bound per session)
# ---------------------------------------------------------------------------
from studyflow_ai.memory.conversation_memory import ConversationMemory
from studyflow_ai.memory.academic_memory import AcademicMemory
from studyflow_ai.memory.user_memory import UserMemory

# ---------------------------------------------------------------------------
# Repository interfaces (injected, never instantiated here)
# ---------------------------------------------------------------------------
from studyflow_ai.repositories.base import (
    UserRepository,
    AcademicRepository,
    ProgressRepository,
)

# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------
from studyflow_ai.models.state import (
    ConversationState,
    UserState,
    ProgressState,
)

# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------
from studyflow_ai.config.constants import MAX_HISTORY_MESSAGES_BUFFER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session status registry
# ---------------------------------------------------------------------------

class SessionStatus(str, Enum):
    """
    Lifecycle states for a managed session.

    Transitions:
      PENDING  → ACTIVE  (after ``create_session()`` completes successfully)
      ACTIVE   → CLOSED  (after ``close_session()`` flushes state)
      ACTIVE   → ERROR   (if an unrecoverable failure occurs mid-session)
    """

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# SessionContext — lightweight result container
# ---------------------------------------------------------------------------

class SessionContext:
    """
    Snapshot of all session-scoped state produced by ``get_session_context()``.

    Passed directly into agent ``handle()`` calls and LLM prompt builders so
    that every turn is grounded in the most recent user, academic, and
    conversation state without repeating repository fetches.

    Attributes:
        session_id:         Unique identifier for this session.
        user_id:            The authenticated student's UUID.
        status:             Current ``SessionStatus`` of the session.
        conversation_state: Rolling chat history and active intent.
        user_profile:       Persona, timezone, and preference dict (may be
                            ``None`` if the user record has not been created yet).
        progress_snapshot:  Streak metrics and struggle topics (may be ``None``
                            if no progress has been recorded yet).
        message_window:     The most recent ``MAX_HISTORY_MESSAGES_BUFFER``
                            messages, pre-trimmed for LLM context injection.
        opened_at:          UTC timestamp when the session was opened.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        status: SessionStatus,
        conversation_state: ConversationState,
        user_profile: Optional[Dict[str, Any]],
        progress_snapshot: Optional[ProgressState],
        message_window: List[Dict[str, str]],
        opened_at: datetime,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.status = status
        self.conversation_state = conversation_state
        self.user_profile = user_profile
        self.progress_snapshot = progress_snapshot
        self.message_window = message_window
        self.opened_at = opened_at

    def __repr__(self) -> str:
        return (
            f"SessionContext(session_id={self.session_id!r}, "
            f"user_id={self.user_id!r}, "
            f"status={self.status!r}, "
            f"messages={len(self.message_window)})"
        )


# ---------------------------------------------------------------------------
# _SessionRecord — internal envelope (not exposed publicly)
# ---------------------------------------------------------------------------

class _SessionRecord:
    """
    Internal envelope holding all live memory wrappers and mutable state for a
    single active session.

    Created by ``SessionManager._open_session_record()`` and stored in the
    ``_sessions`` registry keyed by ``session_id``.  Never returned to callers
    directly; ``SessionContext`` is the public-facing projection.

    Attributes:
        session_id:          Unique session identifier.
        user_id:             The authenticated student's UUID.
        status:              Current ``SessionStatus``.
        conversation_memory: Short-term chat history wrapper.
        academic_memory:     Course deadlines and streak wrapper.
        user_memory:         Persona and preference wrapper.
        conversation_state:  Mutable Pydantic model reflecting current turn.
        user_state:          Loaded ``UserState`` from repository (optional).
        progress_state:      Loaded ``ProgressState`` from repository (optional).
        opened_at:           UTC timestamp of session open.
        closed_at:           UTC timestamp of session close (None until closed).
    """

    __slots__ = (
        "session_id",
        "user_id",
        "status",
        "conversation_memory",
        "academic_memory",
        "user_memory",
        "conversation_state",
        "user_state",
        "progress_state",
        "opened_at",
        "closed_at",
    )

    def __init__(
        self,
        session_id: str,
        user_id: str,
        conversation_memory: ConversationMemory,
        academic_memory: AcademicMemory,
        user_memory: UserMemory,
        conversation_state: ConversationState,
        user_state: Optional[UserState],
        progress_state: Optional[ProgressState],
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.status: SessionStatus = SessionStatus.PENDING
        self.conversation_memory = conversation_memory
        self.academic_memory = academic_memory
        self.user_memory = user_memory
        self.conversation_state = conversation_state
        self.user_state = user_state
        self.progress_state = progress_state
        self.opened_at: datetime = datetime.now(tz=timezone.utc)
        self.closed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Async session and state manager for StudyFlow AI.

    Design contract:
      - One ``_SessionRecord`` is maintained per active ``session_id`` in
        the in-process registry (``_sessions``).
      - Session creation eagerly loads ``UserState`` and ``ProgressState``
        from the injected repositories to pre-warm the context window.
      - All repository calls are ``await``-ed; this class must be consumed
        from async call-sites.
      - No session state leaks across session boundaries; ``close_session()``
        removes the record from the registry after flushing.

    Args:
        user_repository:     Persistence layer for ``UserState`` records.
        academic_repository: Persistence layer for ``AcademicState`` records.
        progress_repository: Persistence layer for ``ProgressState`` records.
    """

    def __init__(
        self,
        user_repository: UserRepository,
        academic_repository: AcademicRepository,
        progress_repository: ProgressRepository,
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Store injected repository dependencies
        # ------------------------------------------------------------------
        self._user_repository = user_repository
        self._academic_repository = academic_repository
        self._progress_repository = progress_repository

        # ------------------------------------------------------------------
        # 2. Active session registry — keyed by session_id
        # ------------------------------------------------------------------
        self._sessions: Dict[str, _SessionRecord] = {}

        logger.info(
            "SessionManager initialised | repositories=(%s, %s, %s)",
            type(self._user_repository).__name__,
            type(self._academic_repository).__name__,
            type(self._progress_repository).__name__,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Open a new session and pre-warm its memory context from repositories.

        Steps:
          1. Generate (or accept) a unique ``session_id``.
          2. Instantiate all three memory wrappers for this session.
          3. Load ``UserState`` from ``user_repository`` (async).
          4. Load ``ProgressState`` from ``progress_repository`` (async).
          5. Construct the initial ``ConversationState``.
          6. Register the ``_SessionRecord`` in the active registry.
          7. Mark the session ``ACTIVE`` and log.

        Args:
            user_id:    The authenticated student's UUID. Must be non-empty.
            session_id: Optional caller-supplied identifier.  A new UUID is
                        generated if omitted.

        Returns:
            The ``session_id`` string for this session.

        Raises:
            ValueError:  If ``user_id`` is empty.
            RuntimeError: If the session cannot be initialised due to a
                          repository failure.

        # TODO(ArtifactService): Before loading from repositories, attempt a
        # warm-start from a cached ArtifactService snapshot keyed by
        # ``(user_id, session_id)`` to skip the repository round-trip.
        # TODO(DistributedLock): Acquire a Redis lock on ``session_id`` here
        # and release it only after the record is registered, so concurrent
        # create calls for the same session_id are serialised.
        """
        self._validate_user_id(user_id)

        resolved_session_id = session_id or self._generate_session_id()

        if resolved_session_id in self._sessions:
            logger.warning(
                "Session '%s' already exists; returning existing id.",
                resolved_session_id,
            )
            return resolved_session_id

        logger.info(
            "Creating session | session_id=%s user_id=%s",
            resolved_session_id,
            user_id,
        )

        # ------------------------------------------------------------------
        # Step 1 – instantiate memory wrappers
        # ------------------------------------------------------------------
        conversation_memory = ConversationMemory(session_id=resolved_session_id)
        academic_memory = AcademicMemory(user_id=user_id)
        user_memory = UserMemory(user_id=user_id)

        # ------------------------------------------------------------------
        # Step 2 – load baseline state from repositories (async)
        # ------------------------------------------------------------------
        user_state = await self._load_user_state(user_id)
        progress_state = await self._load_progress_state(user_id)

        # ------------------------------------------------------------------
        # Step 3 – construct initial ConversationState
        # ------------------------------------------------------------------
        conversation_state = ConversationState(
            session_id=resolved_session_id,
            user_id=user_id,
            active_intent="NONE",
            messages=[],
            awaiting_callback=False,
        )

        # ------------------------------------------------------------------
        # Step 4 – register session record
        # ------------------------------------------------------------------
        record = _SessionRecord(
            session_id=resolved_session_id,
            user_id=user_id,
            conversation_memory=conversation_memory,
            academic_memory=academic_memory,
            user_memory=user_memory,
            conversation_state=conversation_state,
            user_state=user_state,
            progress_state=progress_state,
        )
        record.status = SessionStatus.ACTIVE
        self._sessions[resolved_session_id] = record

        logger.info(
            "Session opened | session_id=%s user_id=%s user_found=%s progress_found=%s",
            resolved_session_id,
            user_id,
            user_state is not None,
            progress_state is not None,
        )
        return resolved_session_id

    async def get_session_context(self, session_id: str) -> SessionContext:
        """
        Extract a real-time ``SessionContext`` frame for prompt grounding.

        Combines the live in-memory conversation state with the pre-warmed
        user profile and progress snapshot so that every agent turn receives
        a fully-populated context without additional repository calls.

        Args:
            session_id: The active session identifier returned by
                        ``create_session()``.

        Returns:
            A ``SessionContext`` snapshot ready for injection into agent
            ``handle()`` calls and LLM prompt builders.

        Raises:
            KeyError: If ``session_id`` is not found in the active registry.
            RuntimeError: If the session has already been closed.

        # TODO(ArtifactService): Merge any warm-start ArtifactService data
        # into the returned ``SessionContext`` so downstream agents can
        # access cached course artefacts without separate lookups.
        # TODO(Observability): Emit a span event here tagged with
        # ``session_id`` and ``message_count`` for latency attribution.
        """
        record = self._get_active_record(session_id)

        message_window = self._trim_message_window(
            record.conversation_state.messages
        )

        user_profile = record.user_state.model_dump() if record.user_state else None

        context = SessionContext(
            session_id=session_id,
            user_id=record.user_id,
            status=record.status,
            conversation_state=record.conversation_state,
            user_profile=user_profile,
            progress_snapshot=record.progress_state,
            message_window=message_window,
            opened_at=record.opened_at,
        )

        logger.debug(
            "Context extracted | session_id=%s messages=%d",
            session_id,
            len(message_window),
        )
        return context

    async def save_session_state(
        self,
        session_id: str,
        role: str,
        content: str,
        active_intent: Optional[str] = None,
    ) -> None:
        """
        Append a new conversation turn and persist the updated progress state.

        Steps:
          1. Resolve the active ``_SessionRecord``.
          2. Append the message to both the ``ConversationMemory`` wrapper and
             the ``ConversationState.messages`` list.
          3. Optionally update ``active_intent`` on the ``ConversationState``.
          4. Flush the current ``ProgressState`` back to the repository.

        Args:
            session_id:    The active session identifier.
            role:          Message author — ``"user"`` or ``"assistant"``.
            content:       Plain text of the message turn.
            active_intent: Optional new intent label to stamp on the session
                           (e.g. ``"QNA"``, ``"PLAN"``).

        Raises:
            KeyError:    If ``session_id`` is not found in the active registry.
            RuntimeError: If the session has already been closed.
            ValueError:  If ``role`` is not ``"user"`` or ``"assistant"``.

        # TODO(DistributedLock): Acquire a per-session async lock before
        # mutating ``conversation_state.messages`` to prevent concurrent
        # write races when multiple requests share the same session_id.
        """
        self._validate_role(role)
        record = self._get_active_record(session_id)

        # ------------------------------------------------------------------
        # Step 1 – append to memory wrapper
        # ------------------------------------------------------------------
        record.conversation_memory.append_message(role=role, content=content)

        # ------------------------------------------------------------------
        # Step 2 – append to ConversationState
        # ------------------------------------------------------------------
        record.conversation_state.messages.append(
            {"role": role, "content": content}
        )

        # ------------------------------------------------------------------
        # Step 3 – update active intent if supplied
        # ------------------------------------------------------------------
        if active_intent is not None:
            record.conversation_state.active_intent = active_intent
            logger.debug(
                "Intent updated | session_id=%s intent=%s",
                session_id,
                active_intent,
            )

        # ------------------------------------------------------------------
        # Step 4 – flush progress state to repository
        # ------------------------------------------------------------------
        if record.progress_state is not None:
            await self._progress_repository.save_progress(record.progress_state)

        logger.debug(
            "Session state saved | session_id=%s role=%s total_messages=%d",
            session_id,
            role,
            len(record.conversation_state.messages),
        )

    async def close_session(self, session_id: str) -> None:
        """
        Flush all modified state and remove the session from the active registry.

        Steps:
          1. Resolve the active ``_SessionRecord``.
          2. Persist ``UserState`` to ``user_repository`` if it was loaded.
          3. Persist ``ProgressState`` to ``progress_repository`` if it was loaded.
          4. Mark the session ``CLOSED`` and stamp ``closed_at``.
          5. Remove the record from the active registry.

        Args:
            session_id: The active session identifier.

        Raises:
            KeyError:    If ``session_id`` is not found in the active registry.
            RuntimeError: If the session has already been closed.

        # TODO(ArtifactService): Serialise the full ``_SessionRecord`` to the
        # ArtifactService store before removing it from the registry, enabling
        # warm-start resumes on the next ``create_session()`` call.
        # TODO(DistributedLock): Release the Redis session lock acquired in
        # ``create_session()`` only after the record has been removed.
        # TODO(Observability): Emit a session-close span event here, including
        # total turn count, open duration, and final intent label.
        """
        record = self._get_active_record(session_id)

        logger.info(
            "Closing session | session_id=%s user_id=%s total_messages=%d",
            session_id,
            record.user_id,
            len(record.conversation_state.messages),
        )

        # ------------------------------------------------------------------
        # Step 1 – flush UserState
        # ------------------------------------------------------------------
        if record.user_state is not None:
            await self._user_repository.save(record.user_state)
            logger.debug(
                "UserState flushed | session_id=%s user_id=%s",
                session_id,
                record.user_id,
            )

        # ------------------------------------------------------------------
        # Step 2 – flush ProgressState
        # ------------------------------------------------------------------
        if record.progress_state is not None:
            await self._progress_repository.save_progress(record.progress_state)
            logger.debug(
                "ProgressState flushed | session_id=%s user_id=%s",
                session_id,
                record.user_id,
            )

        # ------------------------------------------------------------------
        # Step 3 – mark closed and remove from registry
        # ------------------------------------------------------------------
        record.status = SessionStatus.CLOSED
        record.closed_at = datetime.now(tz=timezone.utc)
        del self._sessions[session_id]

        logger.info("Session closed | session_id=%s", session_id)

    # ------------------------------------------------------------------
    # Helpers — session record resolution
    # ------------------------------------------------------------------

    def _get_active_record(self, session_id: str) -> _SessionRecord:
        """
        Look up and validate the active ``_SessionRecord`` for ``session_id``.

        Args:
            session_id: The session identifier to resolve.

        Returns:
            The live ``_SessionRecord`` for this session.

        Raises:
            KeyError:    If ``session_id`` is not in the active registry.
            RuntimeError: If the record's status is not ``ACTIVE``.
        """
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(
                f"Session '{session_id}' not found in the active registry. "
                "It may have already been closed or was never created."
            )
        if record.status != SessionStatus.ACTIVE:
            raise RuntimeError(
                f"Session '{session_id}' has status '{record.status.value}' "
                "and cannot be operated on. Only ACTIVE sessions are mutable."
            )
        return record

    # ------------------------------------------------------------------
    # Helpers — repository fetch wrappers
    # ------------------------------------------------------------------

    async def _load_user_state(self, user_id: str) -> Optional[UserState]:
        """
        Fetch ``UserState`` from the user repository; returns ``None`` on miss.

        Args:
            user_id: The student's UUID to load.

        Returns:
            The persisted ``UserState`` or ``None`` if not yet created.
        """
        try:
            user_state = await self._user_repository.get_by_id(user_id)
            if user_state is None:
                logger.debug(
                    "No UserState found for user_id=%s; session will start cold.",
                    user_id,
                )
            return user_state
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "UserState load failed for user_id=%s | error=%s",
                user_id,
                exc,
            )
            return None

    async def _load_progress_state(self, user_id: str) -> Optional[ProgressState]:
        """
        Fetch ``ProgressState`` from the progress repository; returns ``None``
        on miss.

        Args:
            user_id: The student's UUID to load.

        Returns:
            The persisted ``ProgressState`` or ``None`` if no activity recorded.
        """
        try:
            progress_state = await self._progress_repository.get_progress(user_id)
            if progress_state is None:
                logger.debug(
                    "No ProgressState found for user_id=%s; streak starts at zero.",
                    user_id,
                )
            return progress_state
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ProgressState load failed for user_id=%s | error=%s",
                user_id,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Helpers — validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """
        Guard against empty or whitespace-only ``user_id`` values.

        Args:
            user_id: The value to validate.

        Raises:
            ValueError: If ``user_id`` is falsy or blank.
        """
        if not user_id or not user_id.strip():
            raise ValueError(
                "user_id must be a non-empty string. "
                "Provide the authenticated student's UUID."
            )

    @staticmethod
    def _validate_role(role: str) -> None:
        """
        Guard against invalid message-author role labels.

        Args:
            role: The role string to validate.

        Raises:
            ValueError: If ``role`` is not ``"user"`` or ``"assistant"``.
        """
        allowed_roles = {"user", "assistant"}
        if role not in allowed_roles:
            raise ValueError(
                f"Invalid role '{role}'. "
                f"Allowed values are: {sorted(allowed_roles)}"
            )

    # ------------------------------------------------------------------
    # Helpers — message window trimming
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_message_window(
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Return the most recent ``MAX_HISTORY_MESSAGES_BUFFER`` messages.

        Keeps the LLM context window within the configured token budget by
        dropping the oldest turns first.

        Args:
            messages: The full ``ConversationState.messages`` list.

        Returns:
            A trimmed list containing at most ``MAX_HISTORY_MESSAGES_BUFFER``
            entries, ordered oldest-to-newest.
        """
        if len(messages) <= MAX_HISTORY_MESSAGES_BUFFER:
            return list(messages)
        return list(messages[-MAX_HISTORY_MESSAGES_BUFFER:])

    # ------------------------------------------------------------------
    # Helpers — token generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate a cryptographically random session token.

        Returns:
            A lowercase UUID4 string (e.g. ``"3f2504e0-4f89-11d3-9a0c-..."``)
            suitable for use as a session identifier.
        """
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Properties (read-only access to injected dependencies)
    # ------------------------------------------------------------------

    @property
    def user_repository(self) -> UserRepository:
        """Injected ``UserRepository`` for user-state persistence."""
        return self._user_repository

    @property
    def academic_repository(self) -> AcademicRepository:
        """Injected ``AcademicRepository`` for course-state persistence."""
        return self._academic_repository

    @property
    def progress_repository(self) -> ProgressRepository:
        """Injected ``ProgressRepository`` for progress-state persistence."""
        return self._progress_repository

    @property
    def active_session_ids(self) -> List[str]:
        """
        Snapshot of all currently active session identifiers.

        Returns:
            A list of ``session_id`` strings for sessions with
            ``SessionStatus.ACTIVE``.
        """
        return [
            sid
            for sid, record in self._sessions.items()
            if record.status == SessionStatus.ACTIVE
        ]
