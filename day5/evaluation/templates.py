TEMPLATE_1 = """
### Question: {query}
### User Answer: {sentence_inference}
### Reference Answer: {sentence_true}
Rate the user's answer compared to the reference:
4 = fully matches, 2 = partial, 0 = no match. No explanation.
"""

TEMPLATE_2 = """
Compare the following answers to the question: {query}
User Answer: {sentence_inference}
Reference Answer: {sentence_true}
Give a score: 4 (exact match), 2 (partial), 0 (no match). No reasoning.
"""
