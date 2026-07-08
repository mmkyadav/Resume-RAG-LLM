import json
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from config import (
    GEMINI_API_KEY,
    LLM_MODEL
)


class QueryClassifier:
    """
    QueryClassifier uses the configured Gemini LLM (via google-genai SDK) to classify
    if a query is about a single candidate or involves comparisons/aggregations/rankings.
    Falls back to local keyword heuristics if the API key is missing or the LLM call fails.
    """
    def __init__(self, api_key: str = None):
        self.llm = None
        key_to_use = api_key or GEMINI_API_KEY

        if key_to_use and key_to_use not in ("your_gemini_api_key_here", ""):
            try:
                self.llm = GoogleGenAI(
                    model=LLM_MODEL,
                    api_key=key_to_use,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini LLM client: {e}. Using fallback classifier.")

    def classify(self, query: str) -> dict:
        """
        Classifies the user query.
        Returns: {"is_single_candidate": bool, "reason": str}
        """
        if not self.llm:
            return self._fallback_classify(query, "LLM not initialized (missing API key)")

        prompt = f"""You are a query classifier for a Resume Retrieval-Augmented Generation (RAG) system.
The system answers questions about exactly one candidate at a time. It cannot handle questions that compare, list, rank, aggregate, or filter multiple candidates.

Classify the following query:
Query: "{query}"

Respond ONLY in valid JSON — no extra text or markdown.
Format:
{{
  "is_single_candidate": true or false,
  "reason": "Brief explanation"
}}"""

        try:
            response = self.llm.complete(prompt)
            content = response.text.strip()

            # Clean up markdown code fences if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return {
                "is_single_candidate": bool(result.get("is_single_candidate", True)),
                "reason": str(result.get("reason", ""))
            }
        except Exception as e:
            return self._fallback_classify(query, f"LLM error: {e}")

    def _fallback_classify(self, query: str, trigger_reason: str) -> dict:
        """Fallback local keyword-based heuristic classifier."""
        lower_query = query.lower()

        comparison_triggers = {
            "compare", "comparison", "list", "rank", "ranking", "shortlist",
            "best", "who has", "which candidate", "difference", "differences",
            "similarities", "match", "between", "vs", "versus", "candidates", "resumes"
        }

        is_single = not any(trigger in lower_query for trigger in comparison_triggers)
        reason = f"[{trigger_reason}] Fallback heuristic"
        if not is_single:
            reason += " — comparison/aggregation keywords detected."
        else:
            reason += "."

        return {
            "is_single_candidate": is_single,
            "reason": reason
        }


if __name__ == "__main__":
    classifier = QueryClassifier()
    test_queries = [
        "What is Ashok's experience in Python?",
        "Compare Ashok and Pawan's python experience",
        "Who is the best candidate for Java?",
        "List all candidates with react experience",
        "Summarize Yasasvi's achievements"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Result:", classifier.classify(q))
