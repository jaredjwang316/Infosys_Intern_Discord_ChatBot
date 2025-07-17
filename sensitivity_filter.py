"""
sensitivity_filter.py

Purpose:
--------
This module configures logging to automatically redact sensitive information 
preventing accidental exposure of secrrets such as API keys, tokens, passwords, 
and other private data into log files.

Key Features:
-------------
- Redacts sensitive data including API keys, tokens, passwords, database credentials,
  internal URLs, IP addresses, email addresses, UUIDs, and more.
- Dynamically redacts environment variable values related to secrets (e.g., DB_PASSWORD,
  DISCORD_BOT_TOKEN, GOOGLE_API_KEY) wherever they appear in log messages.
- Uses a RedactionFilter to apply redaction to both log messages and exception tracebacks.
- Provides a convenient `configure_logging_with_redaction()` function to setup
  logging handlers (console and optional file) with the redaction filter automatically applied.

This module is intended for use in applications handling sensitive credentials or data,
ensuring logs are safe to share or store without risking leakage of confidential information.
"""

import os
import re
import logging

# --- Redaction Function ---
def redact_error_message(error_message: str) -> str:
    """
    Redacts sensitive or non-essential information from an error message string.
    This version loads the API key from environment variables internally.

    Args:
        error_message (str): The raw error message string.

    Returns:
        str: The redacted error message string.
    """
    redacted_message = error_message

    # --- API Key Redaction (Loaded Internally) ---
    # Load the API key from environment variables within the function's scope.
    # This ensures the function only takes 'error_message' as an explicit argument.
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        # Use re.escape to handle special characters in the API key itself
        redacted_message = re.sub(
            re.escape(api_key),
            '[REDACTED_API_KEY]',
            redacted_message
        )

    # --- Add-on: Redact sensitive env values like DB credentials and tokens ---
    for key, value in os.environ.items():
        if not value:
            continue

        # Check if the environment variable name indicates it might contain sensitive data
        if re.search(r'(password|token|secret|api|key|host|user|db_name)', key, re.IGNORECASE):
            # If the value appears in the error message, replace it with a redacted placeholder
            # The placeholder uses the environment variable's name in uppercase, e.g., [REDACTED_DB_PASSWORD]
            redacted_message = redacted_message.replace(value, f"[REDACTED_{key.upper()}]")

    # --- Specific Keyword Redaction ---
    # Redact the specific phrase "SUPER DUPER SECRET"
    redacted_message = re.sub(
        r'SUPER DUPER SECRET',
        '[REDACTED]',
        redacted_message
    )

    # --- General Redaction Rules ---
    # Redact specific quota_metric, quota_id, quota_value (common in Google API errors)
    # Example: quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    redacted_message = re.sub(
        r'quota_metric: "[^"]+"',
        'quota_metric: "[REDACTED_QUOTA_METRIC]"',
        redacted_message
    )
    # Example: quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    redacted_message = re.sub(
        r'quota_id: "[^"]+"',
        'quota_id: "[REDACTED_QUOTA_ID]"',
        redacted_message
    )
    # Example: quota_value: 10
    redacted_message = re.sub(
        r'quota_value: \d+',
        'quota_value: [REDACTED_QUOTA_VALUE]',
        redacted_message
    )

    # Redact specific billing or project IDs if they appear (general pattern)
    # Example: projects/my-project-123/billingAccounts/abc-def-456
    redacted_message = re.sub(
        r'projects/[a-zA-Z0-9-]+/billingAccounts/[a-zA-Z0-9-]+',
        'projects/[REDACTED_PROJECT_ID]/billingAccounts/[REDACTED_BILLING_ACCOUNT_ID]',
        redacted_message
    )
    # Example: billingAccounts/abc-def-456
    redacted_message = re.sub(
        r'billingAccounts/[a-zA-Z0-9-]+',
        'billingAccounts/[REDACTED_BILLING_ACCOUNT_ID]',
        redacted_message
    )
    # Example: projectId: "my-project-id"
    redacted_message = re.sub(
        r'projectId: "[^"]+"',
        'projectId: "[REDACTED_PROJECT_ID]"',
        redacted_message
    )

    # Redact specific resource names if they reveal internal structure beyond model/location
    # Example: resource: "my-internal-database-name"
    redacted_message = re.sub(
        r'resource: "[^"]+"',
        'resource: "[REDACTED_RESOURCE_NAME]"',
        redacted_message
    )
    
    # Redact common UUID/GUID patterns (e.g., "123e4567-e89b-12d3-a456-426614174000")
    redacted_message = re.sub(
        r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
        '[REDACTED_UUID]',
        redacted_message
    )
    # Redact email addresses (e.g., "user@example.com")
    redacted_message = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[REDACTED_EMAIL]',
        redacted_message
    )
    # Redact IP addresses (IPv4: e.g., "192.168.1.1", IPv6: e.g., "2001:0db8::8a2e:0370:7334")
    redacted_message = re.sub(
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', # IPv4
        '[REDACTED_IP_ADDRESS]',
        redacted_message
    )
    redacted_message = re.sub(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', # Full IPv6
        '[REDACTED_IP_ADDRESS]',
        redacted_message
    )
    redacted_message = re.sub(
        r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:[0-9a-fA-F]{1,4}\b', # Compressed IPv6
        '[REDACTED_IP_ADDRESS]',
        redacted_message
    )

    # Redact URLs that are not public documentation (be careful with this one)
    # This regex is a general attempt to catch URLs that might contain sensitive paths/details.
    # It attempts to avoid common public documentation URLs (like ai.google.dev)
    # You might need to refine this based on your specific needs.
    redacted_message = re.sub(
        r'https?://(?!ai\.google\.dev)[^\s/$.?#].\S*\.[^:]+:\d{4}/[^\s]+', # URLs with port and path
        '[REDACTED_INTERNAL_URL]',
        redacted_message
    )
    redacted_message = re.sub(
        r'https?://(?!ai\.google\.dev)[^\s/$.?#].\S+', # General URLs
        '[REDACTED_URL]',
        redacted_message
    )


    # Add a general indicator that information was redacted
    if redacted_message != error_message:
        redacted_message += " [INFO_REDACTED]"

    return redacted_message

# --- Custom Redaction Filter Class ---
class RedactionFilter(logging.Filter):
    """
    A logging filter that redacts sensitive information from log messages
    and exception details. It now relies on redact_error_message loading API key internally.
    """
    def __init__(self, name: str = ''): # Removed api_key from __init__
        super().__init__(name)

    def filter(self, record):
        # Redact the main log message
        if isinstance(record.msg, str):
            record.msg = redact_error_message(record.msg) # Removed api_key argument
        
        # Redact exception traceback details (if present)
        if record.exc_text:
            record.exc_text = redact_error_message(record.exc_text) # Removed api_key argument
        
        return True # Always return True to allow the record to be processed

# --- Logging Configuration Function ---
def configure_logging_with_redaction(log_level=logging.INFO, log_filename=None): # Removed api_key from signature
    """
    Configures the root logger with a file handler, stream handler,
    and the RedactionFilter.

    Args:
        log_level (int): The minimum logging level (e.g., logging.INFO, logging.DEBUG).
        log_filename (str, optional): Path to the log file. If None, only stream logging.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate logs if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add the RedactionFilter to the root logger
    root_logger.addFilter(RedactionFilter()) # No api_key argument needed here

    # Configure handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Stream handler (for console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler (optional)
    if log_filename:
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.info("Logging configured with redaction filter.")
