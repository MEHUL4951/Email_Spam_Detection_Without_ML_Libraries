def clean_email(text):
    # Lowercase
    text = text.lower()
    
    # Keep only alphanumeric characters and spaces (faster version)
    cleaned = ''.join([char for char in text if char.isalnum() or char.isspace()])
    
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def preprocess_emails(email_list):
    return [clean_email(email) for email in email_list]
