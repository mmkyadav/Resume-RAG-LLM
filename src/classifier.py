import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QueryClassifier")

class QueryClassifier:
    """
    QueryClassifier performs rule-based intent classification locally without LLM calls.
    It blocks multi-candidate queries (comparisons, vs, both), aggregations, rankings,
    and lists to ensure routing safety.
    """
    def __init__(self, api_key: str = None):
        # API key is accepted for backward compatibility, but not used since classification runs locally.
        logger.info("Initializing local QueryClassifier...")

    def classify(self, query: str) -> dict:
        """
        Classifies the user query based on local keyword heuristics.
        Returns: {"is_single_candidate": bool, "reason": str}
        """
        if not query or not query.strip():
            return {
                "is_single_candidate": True,
                "reason": "Empty query defaults to single candidate search context."
            }
            
        lower_query = query.lower()

        # Keywords that signify multi-candidate comparison, ranking, recommendation, or listing
        comparison_triggers = {
            "compare", "comparison", "list", "rank", "ranking", "shortlist",
            "best", "who has", "which candidate", "difference", "differences",
            "similarities", "match", "between", "vs", "versus", "candidates", 
            "resumes", "both", "all", "recommend", "recommendation", "who is the best"
        }

        # Check if any trigger word exists in the query as a standalone token or substring
        matched_triggers = []
        for trigger in comparison_triggers:
            # Match word boundary or exact phrase
            if trigger in lower_query:
                matched_triggers.append(trigger)

        if matched_triggers:
            reason = f"Flagged as non-single candidate query due to matching terms: {matched_triggers}"
            logger.info(f"Query classification: BLOCKED. Reason: {reason}")
            return {
                "is_single_candidate": False,
                "reason": reason
            }
            
        logger.info("Query classification: PASSED (local heuristic).")
        return {
            "is_single_candidate": True,
            "reason": "Local heuristic: no comparison or aggregation keywords detected."
        }

if __name__ == "__main__":
    classifier = QueryClassifier()
    test_queries = [
        "What is Ashok's experience in Python?",
        "Compare Ashok and Pawan's python experience",
        "Who is the best candidate for Java?",
        "List all candidates with react experience",
        "Recommend the best AI Engineer",
        "Summarize Yasasvi's achievements"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", classifier.classify(q))
