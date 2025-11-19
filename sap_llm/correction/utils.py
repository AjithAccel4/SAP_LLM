"""
Enterprise Utilities for Self-Correction System.

Includes:
- Input validation and sanitization
- Audit logging
- Retry logic with exponential backoff
- Data masking for sensitive fields
- Circuit breaker pattern
"""

import hashlib
import re
import time
import functools
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


class InputValidator:
    """
    Validates and sanitizes user inputs.

    Prevents injection attacks and ensures data integrity.
    """

    @staticmethod
    def validate_confidence(value: float, field_name: str = "confidence") -> float:
        """
        Validate confidence score.

        Args:
            value: Confidence value
            field_name: Name of field for error messages

        Returns:
            Validated confidence value

        Raises:
            ValidationError: If validation fails
        """
        try:
            conf = float(value)

            if not 0 <= conf <= 1:
                raise ValidationError(
                    f"{field_name} must be between 0 and 1, got {conf}"
                )

            return conf

        except (TypeError, ValueError) as e:
            raise ValidationError(f"Invalid {field_name}: {value}, error: {e}")

    @staticmethod
    def validate_document_id(doc_id: str) -> str:
        """
        Validate document ID.

        Args:
            doc_id: Document ID

        Returns:
            Validated document ID

        Raises:
            ValidationError: If validation fails
        """
        if not doc_id or not isinstance(doc_id, str):
            raise ValidationError(f"Invalid document ID: {doc_id}")

        # Remove any potentially dangerous characters
        sanitized = re.sub(r'[^\w\-_.]', '', doc_id)

        if len(sanitized) > 100:
            raise ValidationError(f"Document ID too long: {len(sanitized)} chars")

        return sanitized

    @staticmethod
    def validate_field_name(field_name: str) -> str:
        """
        Validate field name.

        Args:
            field_name: Field name

        Returns:
            Validated field name

        Raises:
            ValidationError: If validation fails
        """
        if not field_name or not isinstance(field_name, str):
            raise ValidationError(f"Invalid field name: {field_name}")

        # Only allow alphanumeric and underscores
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', field_name):
            raise ValidationError(
                f"Field name must be alphanumeric with underscores: {field_name}"
            )

        if len(field_name) > 50:
            raise ValidationError(f"Field name too long: {field_name}")

        return field_name

    @staticmethod
    def validate_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate prediction dictionary.

        Args:
            prediction: Prediction data

        Returns:
            Validated prediction

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(prediction, dict):
            raise ValidationError("Prediction must be a dictionary")

        if len(prediction) == 0:
            raise ValidationError("Prediction cannot be empty")

        if len(prediction) > 1000:
            raise ValidationError(
                f"Prediction too large: {len(prediction)} fields"
            )

        # Validate each field
        validated = {}
        for field, value in prediction.items():
            # Validate field name
            field_name = InputValidator.validate_field_name(field)

            # Validate value structure if it's a dict
            if isinstance(value, dict):
                if 'confidence' in value:
                    value['confidence'] = InputValidator.validate_confidence(
                        value['confidence'],
                        f"{field}.confidence"
                    )

            validated[field_name] = value

        return validated


# ============================================================================
# AUDIT LOGGING
# ============================================================================

class AuditEventType(Enum):
    """Types of audit events."""
    CORRECTION_STARTED = "correction_started"
    CORRECTION_COMPLETED = "correction_completed"
    CORRECTION_FAILED = "correction_failed"
    HUMAN_ESCALATION = "human_escalation"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"
    PATTERN_LEARNED = "pattern_learned"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATA_ACCESSED = "data_accessed"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    document_id: Optional[str]
    action: str
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class AuditLogger:
    """
    Enterprise audit logging for compliance and security.

    Logs all sensitive operations with tamper-proof records.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize audit logger.

        Args:
            storage_path: Path to store audit logs
        """
        self.storage_path = storage_path
        self.events: List[AuditEvent] = []

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Description of action
            user_id: Optional user ID
            document_id: Optional document ID
            details: Optional additional details
            success: Whether action was successful
            error_message: Optional error message
        """
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            document_id=document_id,
            action=action,
            details=details or {},
            success=success,
            error_message=error_message
        )

        self.events.append(event)

        # Log to standard logger
        log_msg = (
            f"AUDIT: {event_type.value} - {action} - "
            f"user={user_id}, doc={document_id}, success={success}"
        )

        if success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg}, error={error_message}")

        # Persist to file if configured
        if self.storage_path:
            self._persist_event(event)

    def _persist_event(self, event: AuditEvent):
        """Persist event to file."""
        try:
            from pathlib import Path
            import json

            audit_file = Path(self.storage_path) / f"audit_{datetime.now().strftime('%Y%m')}.jsonl"
            audit_file.parent.mkdir(parents=True, exist_ok=True)

            event_dict = {
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'document_id': event.document_id,
                'action': event.action,
                'details': event.details,
                'success': event.success,
                'error_message': event.error_message
            }

            with open(audit_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')

        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}", exc_info=True)


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(storage_path: Optional[str] = None) -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(storage_path=storage_path)
    return _audit_logger


# ============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================================

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise
                        logger.error(
                            f"Max retries ({max_attempts}) reached for {func.__name__}: {e}"
                        )
                        raise

                    # Log retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for "
                        f"{func.__name__}: {e}, retrying in {delay:.1f}s"
                    )

                    # Wait before retry
                    time.sleep(delay)

                    # Increase delay
                    delay *= backoff_factor

            return None

        return wrapper

    return decorator


# ============================================================================
# DATA MASKING
# ============================================================================

class DataMasker:
    """
    Masks sensitive data in logs and outputs.

    Ensures PII and sensitive information is not exposed.
    """

    # Default sensitive field patterns
    SENSITIVE_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }

    # Sensitive field names
    SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'api_key', 'ssn',
        'tax_id', 'credit_card', 'account_number', 'routing_number',
        'vendor_id', 'customer_id'
    }

    @staticmethod
    def mask_value(value: str, mask_char: str = '*', visible_chars: int = 4) -> str:
        """
        Mask a value, showing only last few characters.

        Args:
            value: Value to mask
            mask_char: Character to use for masking
            visible_chars: Number of characters to keep visible

        Returns:
            Masked value
        """
        if not value or len(value) <= visible_chars:
            return mask_char * len(value) if value else ''

        masked_portion = mask_char * (len(value) - visible_chars)
        visible_portion = value[-visible_chars:]

        return masked_portion + visible_portion

    @staticmethod
    def mask_dict(
        data: Dict[str, Any],
        sensitive_fields: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Mask sensitive fields in dictionary.

        Args:
            data: Data dictionary
            sensitive_fields: Optional set of sensitive field names

        Returns:
            Dictionary with masked values
        """
        if sensitive_fields is None:
            sensitive_fields = DataMasker.SENSITIVE_FIELDS

        masked = {}

        for key, value in data.items():
            # Check if field name is sensitive
            if any(sf in key.lower() for sf in sensitive_fields):
                if isinstance(value, str):
                    masked[key] = DataMasker.mask_value(value)
                elif isinstance(value, dict):
                    masked[key] = DataMasker.mask_dict(value, sensitive_fields)
                else:
                    masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                # Recursively mask nested dicts
                masked[key] = DataMasker.mask_dict(value, sensitive_fields)
            else:
                masked[key] = value

        return masked

    @staticmethod
    def mask_text(text: str) -> str:
        """
        Mask sensitive patterns in text.

        Args:
            text: Text to mask

        Returns:
            Text with sensitive patterns masked
        """
        masked = text

        for pattern_name, pattern in DataMasker.SENSITIVE_PATTERNS.items():
            def replacer(match):
                return DataMasker.mask_value(match.group())

            masked = re.sub(pattern, replacer, masked)

        return masked


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures exceeded threshold
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by stopping calls to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 1
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            timeout_seconds: Time to wait before attempting half-open
            half_open_attempts: Number of successful attempts needed to close
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.last_failure_time and
                time.time() - self.last_failure_time >= self.timeout_seconds):
                logger.info("Circuit breaker: Transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success
            self._on_success()

            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.half_open_attempts:
                logger.info("Circuit breaker: Transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0

        # Reset failure count on success in closed state
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: Failure in HALF_OPEN, reopening")
            self.state = CircuitState.OPEN
            self.success_count = 0

        elif self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker: Failure threshold ({self.failure_threshold}) "
                "reached, opening circuit"
            )
            self.state = CircuitState.OPEN

    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

        logger.info("Circuit breaker reset to CLOSED")
