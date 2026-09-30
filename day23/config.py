"""Retrieval sizes and the relevance threshold.

Day 22 Hit.score is BM25 plus the embedding cosine. A larger score is a
better match, so the filter keeps scores at or above the threshold.
The threshold is not applied inside Day 22 search.
"""

RETRIEVAL_TOP_K = 10
FILTERED_TOP_K = 5
SIMILARITY_THRESHOLD = 9.5
HIGHER_IS_BETTER = True
BASELINE_TOP_K = 5
