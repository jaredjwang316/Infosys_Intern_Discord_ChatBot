import os
import re
import logging

# --- Helper Redaction Functions ---

def _redact_phone_number(text: str, partial: bool = False) -> str:
    """
    Redacts phone numbers from text.
    If partial is True, only the last 5 digits are exposed.
    Supports common formats: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX, XXXXXXXXXX.
    """
    # Regex to find common phone number patterns
    # This regex is designed to be broad to catch various formats
    phone_regex = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'

    def replacer(match):
        phone_num = match.group(0)
        if partial and len(phone_num) > 5:
            return '[REDACTED]' + phone_num[-5:]
        return '[REDACTED_PHONE_NUMBER]'

    return re.sub(phone_regex, replacer, text)

def _redact_email_address(text: str, partial: bool = False) -> str:
    """
    Redacts email addresses from text.
    If partial is True, only the domain is exposed.
    """
    # Regex for email addresses
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    def replacer(match):
        email = match.group(0)
        if partial:
            # Keep only the domain part
            parts = email.split('@')
            if len(parts) == 2:
                return '[REDACTED]@' + parts[1]
        return '[REDACTED_EMAIL_ADDRESS]'

    return re.sub(email_regex, replacer, text)

def _redact_physical_address(text: str, partial: bool = False) -> str:
    """
    Redacts physical addresses from text.
    If partial is True, only the last 5 characters are exposed.
    This is a more complex regex and might need refinement based on specific address formats.
    It tries to capture street numbers, street names, city, state/province, and zip/postal codes.
    """
    # This regex attempts to capture common address components.
    # It's a simplified approach and might not catch all variations or might over-redact.
    # A more robust solution might involve a dedicated NLP library or a more extensive regex.
    address_regex = r'\b\d{1,5}\s(?:[A-Za-z0-9]+\s){1,5}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Square|Sq|Terrace|Ter|Way|Wy|Circle|Cir|Parkway|Pkwy|Route|Rt|Highway|Hwy)\b(?:,?\s(?:[A-Za-z]+\s){1,3}(?:[A-Za-z]{2}|[A-Z]{2})\s\d{5}(?:-\d{4})?)?\b'
    
    def replacer(match):
        address = match.group(0)
        if partial and len(address) > 5:
            return '[REDACTED]' + address[-5:]
        return '[REDACTED_PHYSICAL_ADDRESS]'

    return re.sub(address_regex, replacer, text, flags=re.IGNORECASE)

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

    # Apply full redaction for phone numbers, emails, and physical addresses in general error messages
    redacted_message = _redact_phone_number(redacted_message, partial=False)
    redacted_message = _redact_email_address(redacted_message, partial=False)
    redacted_message = _redact_physical_address(redacted_message, partial=False)


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
            original_msg = record.msg
            
            # Apply specific redaction rules based on log message prefix
            if original_msg.startswith("💬 Content:"):
                # Redact phone numbers, emails, and addresses except for the last 5 chars/domain
                redacted_msg = _redact_phone_number(original_msg, partial=True)
                redacted_msg = _redact_email_address(redacted_msg, partial=True)
                redacted_msg = _redact_physical_address(redacted_msg, partial=True)
                record.msg = redacted_msg
            elif original_msg.startswith("LLM INPUT") or original_msg.startswith("LLM OUTPUT"):
                # Completely redact phone numbers, emails, and addresses
                redacted_msg = _redact_phone_number(original_msg, partial=False)
                redacted_msg = _redact_email_address(redacted_msg, partial=False)
                redacted_msg = _redact_physical_address(redacted_msg, partial=False)
                record.msg = redacted_msg
            else:
                # For all other messages, use the general error message redaction
                record.msg = redact_error_message(record.msg) # Removed api_key argument
        
        # Redact exception traceback details (if present)
        if record.exc_text:
            record.exc_text = redact_error_message(record.exc_text) # Removed api_key argument
        
        return True # Always return True to allow the record to be processed

# --- Logging Configuration Function ---
def configure_logging_with_redaction(log_level=logging.DEBUG, log_filename=None): # Removed api_key from signature
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