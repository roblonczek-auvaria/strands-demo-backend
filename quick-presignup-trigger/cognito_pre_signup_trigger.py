"""
Cognito Pre-Sign Up Trigger Lambda Function

This Lambda function serves as a Cognito pre-sign up trigger that only allows
users with @auvaria.com email addresses to sign up.

Cognito Event Structure:
- The event contains user attributes including email
- We check the email domain and either allow or deny the sign-up
- If denied, we raise an exception that Cognito will handle gracefully
"""

import json
import logging
from typing import Dict, Any

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Lambda handler for Cognito pre-sign up trigger.
    
    Args:
        event: Cognito pre-sign up trigger event
        context: Lambda context object
        
    Returns:
        The original event if user is allowed to sign up
        
    Raises:
        Exception: If user email domain is not @auvaria.com
    """
    
    try:
        # Log the incoming event for debugging
        logger.info(f"Received pre-sign up event: {json.dumps(event, indent=2)}")
        
        # Extract user attributes from the event
        user_attributes = event.get('request', {}).get('userAttributes', {})
        
        # Get the email from user attributes
        email = user_attributes.get('email', '').lower().strip()
        
        if not email:
            logger.error("No email found in user attributes")
            raise Exception("Email is required for sign up")
        
        logger.info(f"Processing sign-up request for email: {email}")
        
        # Check if email domain is @auvaria.com
        allowed_domain = "@auvaria.com"
        
        if not email.endswith(allowed_domain):
            logger.warning(f"Sign-up denied for email: {email} - Domain not allowed")
            raise Exception(f"Sign up is restricted to {allowed_domain} email addresses only")
        
        logger.info(f"Sign-up approved for email: {email}")
        
        # If we reach here, the email domain is allowed
        # Return the event unchanged to allow the sign-up to continue
        return event
        
    except Exception as e:
        # Log the error for monitoring
        logger.error(f"Pre-sign up trigger failed: {str(e)}")
        
        # Re-raise the exception to deny the sign-up
        # Cognito will handle this gracefully and show an appropriate error message
        raise e

def validate_email_format(email: str) -> bool:
    """
    Basic email format validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid, False otherwise
    """
    import re
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

# Alternative implementation with more detailed validation
def lambda_handler_with_validation(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Alternative implementation with more comprehensive email validation.
    """
    
    try:
        logger.info(f"Received pre-sign up event: {json.dumps(event, indent=2)}")
        
        user_attributes = event.get('request', {}).get('userAttributes', {})
        email = user_attributes.get('email', '').lower().strip()
        
        # Validate email presence
        if not email:
            logger.error("No email found in user attributes")
            raise Exception("Email is required for sign up")
        
        # Validate email format
        if not validate_email_format(email):
            logger.error(f"Invalid email format: {email}")
            raise Exception("Please provide a valid email address")
        
        # Check domain restriction
        allowed_domain = "@auvaria.com"
        
        if not email.endswith(allowed_domain):
            logger.warning(f"Sign-up denied for email: {email} - Domain not allowed")
            raise Exception(f"Access is restricted to Auvaria employees only. Please use your {allowed_domain} email address.")
        
        logger.info(f"Sign-up approved for email: {email}")
        return event
        
    except Exception as e:
        logger.error(f"Pre-sign up trigger failed: {str(e)}")
        raise e