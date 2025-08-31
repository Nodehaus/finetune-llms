# Base context for trademark classification
CONTEXT = """A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word."""

# Sample evaluation questions (you can expand this to 100 questions)
EVALUATION_QUESTIONS = [
    {
        "question": 'The mark "Ivory" for a product made of elephant tusks. What is the type of mark?',
        "answer": "generic",
    },
    {
        "question": 'The mark "Tasty" for bread. What is the type of mark?',
        "answer": "descriptive",
    },
    {
        "question": 'The mark "Caress" for body soap. What is the type of mark?',
        "answer": "suggestive",
    },
    {
        "question": 'The mark "Virgin" for wireless communications. What is the type of mark?',
        "answer": "arbitrary",
    },
    {
        "question": 'The mark "Aswelly" for a taxi service. What is the type of mark?',
        "answer": "fanciful",
    },
    {
        "question": 'The mark "Mask" for cloth that you wear on your face to filter air. What is the type of mark?',
        "answer": "generic",
    },
    # Add more questions here to reach 100 total
    {
        "question": 'The mark "FastCar" for automobiles. What is the type of mark?',
        "answer": "descriptive",
    },
    {
        "question": 'The mark "Zephyr" for air conditioning units. What is the type of mark?',
        "answer": "suggestive",
    },
    {
        "question": 'The mark "Apple" for computers. What is the type of mark?',
        "answer": "arbitrary",
    },
    {
        "question": 'The mark "Xerox" for copying machines. What is the type of mark?',
        "answer": "fanciful",
    },
]


def format_question(question_dict, include_examples=True):
    """Format a single question with context and examples."""
    if include_examples:
        examples = """Q: The mark "Ivory" for a product made of elephant tusks. What is the type of mark?
A: generic

Q: The mark "Tasty" for bread. What is the type of mark?
A: descriptive

Q: The mark "Caress" for body soap. What is the type of mark?
A: suggestive

Q: The mark "Virgin" for wireless communications. What is the type of mark?
A: arbitrary

Q: The mark "Aswelly" for a taxi service. What is the type of mark?
A: fanciful

"""
        return f"{CONTEXT}\n\n{examples}Q: {question_dict['question']}\nA:"
    else:
        return f"{CONTEXT}\n\nQ: {question_dict['question']}\nA:"


def get_evaluation_dataset():
    """Get the formatted evaluation dataset."""
    return EVALUATION_QUESTIONS
