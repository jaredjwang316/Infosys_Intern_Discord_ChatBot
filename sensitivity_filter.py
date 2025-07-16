import logging
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file immediately
# This ensures that sensitive values are available for the filter
# when the module is imported.
load_dotenv()

class SensitiveDataFilter(logging.Filter):
    """
    A logging filter that redacts sensitive information from log records.

    It loads sensitive values from environment variables (typically from a .env file)
    and replaces them with a [REDACTED] placeholder in log messages.
    """
    def __init__(self, name="", sensitive_keys=None):
        super().__init__(name)
        # Define a list of environment variable names that might hold sensitive data.
        # These names should correspond to the keys in your .env file.
        self.sensitive_keys = sensitive_keys if sensitive_keys is not None else [
            "DATABASE_USER",
            "DATABASE_PASSWORD",
            "DATABASE_NAME",
            "GEMINI_API_KEY",
            "DISCORD_BOT_TOKEN",
            "MODEL_NAME", # While not strictly a secret, could be sensitive info in some contexts
            "GOOGLE_API_KEY", # Often the same as GEMINI_API_KEY but good to include
            # Add any other keys from your .env file that should be redacted
        ]
        self.sensitive_values = self._load_sensitive_values()
        # Create regex patterns for efficient redaction.
        # Escape special characters in sensitive values to use them in regex.
        # Use re.escape to handle values like 'my.password?$'
        self.redaction_patterns = self._compile_redaction_patterns()
        logging.info("SensitiveDataFilter initialized. Sensitive values loaded.")

    def _load_sensitive_values(self):
        """
        Loads sensitive values from environment variables.
        """
        values = []
        for key in self.sensitive_keys:
            value = os.getenv(key)
            if value:
                values.append(value)
        return values

    def _compile_redaction_patterns(self):
        """
        Compiles regex patterns for each sensitive value to be redacted.
        Patterns are sorted by length (longest first) to prevent partial redactions.
        """
        patterns = []
        # Sort values by length in descending order to ensure longer matches are found first
        # E.g., redact "my_long_password" before "password"
        sorted_values = sorted(self.sensitive_values, key=len, reverse=True)
        for value in sorted_values:
            # Escape special regex characters in the value
            # Use word boundaries (\b) to match whole words if appropriate,
            # but be careful if sensitive values can be part of other strings.
            # For general redaction, a simple non-greedy match might be better.
            # Using re.escape and then a direct match is usually safest.
            patterns.append(re.compile(re.escape(value)))
        return patterns

    def filter(self, record):
        """
        Filters and redacts sensitive information from a log record's message.
        """
        # Convert message to string if it's not already (e.g., if it's an exception object)
        message = str(record.msg)

        # Iterate through compiled patterns and redact sensitive values
        for pattern in self.redaction_patterns:
            message = pattern.sub("[REDACTED]", message)

        # Update the record's message with the redacted version
        record.msg = message

        # If there are arguments (e.g., %s, %d), also try to redact them if they are strings
        if isinstance(record.args, tuple):
            redacted_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    redacted_arg = arg
                    for pattern in self.redaction_patterns:
                        redacted_arg = pattern.sub("[REDACTED]", redacted_arg)
                    redacted_args.append(redacted_arg)
                else:
                    redacted_args.append(arg)
            record.args = tuple(redacted_args)

        return True # Return True to allow the record to be processed by handlers
