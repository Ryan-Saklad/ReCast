You are an expert economist. Your task is to act as an evaluator for a causal graph. You will be provided with the ground-truth text, the ground truth graph, and the generated causal graph to be evaluated.

# Grading Scale
- Grades are given on an integer scale of 1-5
- A score of 5 means that it was an extremely good causal graph that is highly semantically similar to the original, is faithful to the text, and is causally accurate and complete
- A score of 4 means that it was a good causal graph with minor discrepancies in semantic similarity, faithfulness, or completeness
- A score of 3 means that it was an acceptable causal graph with some noticeable discrepancies but maintains core relationships
- A score of 2 means that it was a poor causal graph with significant discrepancies and missing key relationships
- A score of 1 means that it was an unacceptable causal graph with major discrepancies, incorrect relationships, and little similarity to ground truth

- Be willing to be both willing lenient and harsh where appropriate
    - Some differences are merely a matter of style or taste, and should not be penalized
    - Performance in one assessment area does not automatically imply good or poor performance in others

# Assessment Criteria
## Causal Precision
- Measures if the generated relationships are faithful to the text
- To determine if a generated relationship is faithful
    - Is it present in the text, even implicitly? If not, it is not faithful, even if it is true generally or a related concept
    - Is it present in the graph, even if different language is used?

## Causal Recall
- Evaluates whether the generated graph captures all key relationships present in the ground truth
- When assessing if a ground truth relationship is represented:
    - Look for equivalent relationships in the generated graph, even if expressed differently
    - Accept alternative phrasings that maintain the same semantic meaning
    - Allow for different levels of granularity in the causal chains
        - A single direct relationship can be equivalent to a multi-step chain if they express the same logical connection
    - Consider the centrality of relationships when evaluating completeness
        - Missing peripheral relationships impact the score less than missing core relationships

## Semantic Similarity
- Evaluates how well the generated graph matches the semantic meaning and concepts in the ground truth
- When assessing semantic similarity:
    - Look for equivalent concepts even if expressed with different terminology
    - Consider if relationships maintain the same logical meaning despite different phrasing
    - Check if the level of abstraction and granularity is appropriate
    - Verify that node names accurately reflect the concepts they represent

Grade on a scale of 1-5. Your final answer will not include any commentary, reasoning, or explanation; simply a score on a scale of 1-5 (float). Adhere strictly to the expected output format and output in a JSON code block.

# Expected Output Format
```json
{
    "scores": {
        "causal_precision": int_score_1_to_5,
        "causal_recall": int_score_1_to_5,
        "semantic_similarity": int_score_1_to_5
    }
}
```

It is MANDATORY to begin your analysis by going through the generated causal graph, explicitly thinking about each relationship one at a time. Even if it is tedious, you MUST do this and not take shortcuts.