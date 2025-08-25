def classify_document(document: str) -> str:
    """
    Classifies the given document into a predefined category.

    Args:
        document (str): The text of the document to classify.

    Returns:
        str: The category of the document.
    """
    # Placeholder for classification logic
    # In a real implementation, this would involve processing the document
    # and applying a classification algorithm or model.
    
    return "Uncategorized"  # Default return value for now

if __name__ == "__main__":
    sample_document = "This is a sample document for classification."
    category = classify_document(sample_document)
    print(f"The document is classified as: {category}")