def clean_context(self, text: str) -> str:
    """
    Clean and normalize context text for better evaluation.
    
    Args:
        text (str): The raw context text.
        
    Returns:
        str: Cleaned context text.
    """
    # Check if the text is a string representation of a list or other data structure
    if isinstance(text, str) and (text.startswith('[') or text.startswith('{')):
        try:
            # Try to safely evaluate it as a literal
            import ast
            evaluated_text = ast.literal_eval(text)
            if isinstance(evaluated_text, list):
                # If it's a list, join the elements
                text = " ".join(evaluated_text)
        except (SyntaxError, ValueError):
            # If it can't be evaluated, keep as is
            pass
    
    # Replace multiple newlines with a single space
    text = ' '.join(line.strip() for line in text.splitlines() if line.strip())
    
    # Remove duplicate sentences (simple approach)
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    cleaned_text = '. '.join(unique_sentences)
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'
        
    return cleaned_text