"""
Text Preprocessing Module for SMS Spam Detection.
Provides beginner-friendly functions for cleaning and tokenizing raw SMS text messages.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing URLs, phone numbers, and non-alphanumeric special characters.
    3. Normalizing whitespace.

    Args:
        text (str): Raw input SMS message.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove digits and phone-like patterns
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Strip excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str) -> list:
    """
    Splits cleaned text into individual word tokens.
    Filters out empty strings and single-character words.

    Args:
        text (str): Cleaned or raw text string.

    Returns:
        list of str: List of tokenized words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = [word for word in cleaned.split() if len(word) > 1]
    return tokens

