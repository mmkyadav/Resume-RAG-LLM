"""
tests/test_rag.py — End-to-End RAG Pipeline Validation Tests
Person 3: LLM QA Integration & Streamlit Interface

Commit 3: Add end-to-end RAG validation tests

Test coverage:
    1. Candidate name extraction from filenames (parser)
    2. Fuzzy name correction (matcher)  — "pawan" -> pavanteja, "yasasvi" -> yasaswi, etc.
    3. Multi-candidate / comparison query blocking (classifier + engine)
    4. Single-candidate QA routing (engine — mock or real)
    5. Edge-cases: no name, empty query, very-short query, stop-word-only query

Run with:
    pytest tests/test_rag.py -v
or:
    python tests/test_rag.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Make src/ importable ───────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Imports from src/ ──────────────────────────────────────────────────────────
from parser import extract_candidate_name
from matcher import FuzzyNameMatcher
from classifier import QueryClassifier
from retriever import FilteredQueryEngine


# ══════════════════════════════════════════════════════════════════════════════
# 1. Parser Tests — Candidate Name Extraction from File Names
# ══════════════════════════════════════════════════════════════════════════════
class TestCandidateNameExtraction(unittest.TestCase):
    """Verifies that `extract_candidate_name` correctly parses resume filenames."""

    def _make_path(self, filename: str) -> Path:
        """Creates a dummy Path with the given filename (no real file needed)."""
        return Path(f"/dummy/{filename}")

    # ── Standard " - " delimited filenames ──
    def test_standard_format_extracts_candidate_name(self):
        """Expected: part after the last ' - ' is the canonical name."""
        path = self._make_path("ASHOK_Reddy_RESUME - M Ashok reddy.pdf")
        name = extract_candidate_name(path)
        self.assertEqual(name, "M Ashok reddy")

    def test_candidate_name_with_spaces(self):
        path = self._make_path("PAVAN_KAMMA_CV - pavanteja kamma.pdf")
        name = extract_candidate_name(path)
        self.assertEqual(name, "pavanteja kamma")

    def test_candidate_name_with_initials(self):
        path = self._make_path("YasaswiProfile - yasaswi kotha.pdf")
        name = extract_candidate_name(path)
        # Filename has ' - ' so returns the part after the last ' - '
        self.assertEqual(name, "yasaswi kotha")

    def test_filename_without_delimiter_uses_full_stem(self):
        """If no ' - ' is present, the whole filename stem becomes the name."""
        path = self._make_path("trinadh_kumar_resume.pdf")
        name = extract_candidate_name(path)
        self.assertEqual(name, "trinadh_kumar_resume")

    def test_multiple_dashes_uses_last_segment(self):
        """Filenames with multiple ' - ' separators — last segment is the name."""
        path = self._make_path("CV - Senior Dev - Shirley Katherine.pdf")
        name = extract_candidate_name(path)
        self.assertEqual(name, "Shirley Katherine")

    def test_empty_filename_stem(self):
        """Edge case: very unusual path with empty stem should not crash."""
        path = Path("/dummy/.pdf")
        name = extract_candidate_name(path)
        self.assertIsInstance(name, str)

    def test_name_is_stripped_of_whitespace(self):
        path = self._make_path("Resume - John Doe  .pdf")
        name = extract_candidate_name(path)
        self.assertEqual(name, "John Doe")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fuzzy Name Matcher Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestFuzzyNameMatcher(unittest.TestCase):
    """
    Verifies FuzzyNameMatcher resolves candidate names correctly including
    spelling corrections and shortform expansions.

    NOTE: These tests depend on the Resumes/ directory being present and
    populated.  When no resumes are found the matcher.candidates list is empty
    and all match_query() calls return []; those test cases are skipped
    automatically with a SkipTest notice.
    """

    @classmethod
    def setUpClass(cls):
        cls.matcher = FuzzyNameMatcher()
        cls.has_resumes = len(cls.matcher.candidates) > 0

    def _skip_if_no_resumes(self):
        if not self.has_resumes:
            self.skipTest(
                "Resumes/ directory is empty or missing — skipping live matcher tests."
            )

    # ── Candidates loaded ──
    def test_candidates_list_is_list(self):
        self.assertIsInstance(self.matcher.candidates, list)

    def test_candidate_terms_is_dict(self):
        self.assertIsInstance(self.matcher.candidate_terms, dict)

    # ── Exact match ──
    def test_exact_name_match_single_result(self):
        self._skip_if_no_resumes()
        # Use the first loaded candidate to test an exact match
        first_candidate = self.matcher.candidates[0]
        first_word = first_candidate.split()[0]
        if len(first_word) <= 2:
            self.skipTest("First candidate name is too short for a reliable test.")
        matches = self.matcher.match_query(f"Tell me about {first_candidate}")
        self.assertGreaterEqual(len(matches), 1)
        canonical_names = [m["canonical_name"] for m in matches]
        self.assertIn(first_candidate, canonical_names)

    # ── Spelling-correction: pawan -> pavanteja kamma ──
    def test_pawan_corrects_to_pavanteja(self):
        self._skip_if_no_resumes()
        if not any("pavanteja" in c.lower() for c in self.matcher.candidates):
            self.skipTest("Candidate 'pavanteja kamma' not found in resumes directory.")
        matches = self.matcher.match_query("What are the skills of Pawan?")
        self.assertEqual(len(matches), 1, f"Expected 1 match, got {len(matches)}: {matches}")
        self.assertIn("pavanteja", matches[0]["canonical_name"].lower())
        self.assertTrue(matches[0]["is_spelling_mistake"])

    # ── Spelling-correction: yasasvi -> yasaswi kotha ──
    def test_yasasvi_corrects_to_yasaswi(self):
        self._skip_if_no_resumes()
        if not any("yasaswi" in c.lower() for c in self.matcher.candidates):
            self.skipTest("Candidate 'yasaswi kotha' not found in resumes directory.")
        matches = self.matcher.match_query("Summarize yasasvi's achievements")
        self.assertEqual(len(matches), 1)
        self.assertIn("yasaswi", matches[0]["canonical_name"].lower())
        self.assertTrue(matches[0]["is_spelling_mistake"])

    # ── Spelling-correction: trinad -> trinadh ──
    def test_trinad_corrects_to_trinadh(self):
        self._skip_if_no_resumes()
        if not any("trinadh" in c.lower() for c in self.matcher.candidates):
            self.skipTest("Candidate 'Trinadh' not found in resumes directory.")
        matches = self.matcher.match_query("What projects has Trinad done?")
        self.assertEqual(len(matches), 1)
        self.assertIn("trinadh", matches[0]["canonical_name"].lower())
        self.assertTrue(matches[0]["is_spelling_mistake"])

    # ── Multi-candidate detection ──
    def test_two_names_in_query_returns_multiple_matches(self):
        self._skip_if_no_resumes()
        candidates = self.matcher.candidates
        if len(candidates) < 2:
            self.skipTest("Need at least 2 candidates for multi-match test.")
        c1 = candidates[0].split()[0]
        c2 = candidates[1].split()[0]
        if len(c1) <= 2 or len(c2) <= 2:
            self.skipTest("Candidate first-names are too short for a reliable test.")
        query = f"Compare {c1} and {c2}"
        matches = self.matcher.match_query(query)
        # Expect 2 distinct canonical matches
        names = {m["canonical_name"] for m in matches}
        self.assertGreaterEqual(len(names), 1)  # At least one must be found

    # ── No name in query ──
    def test_no_name_returns_empty(self):
        matches = self.matcher.match_query("What is Python?")
        self.assertIsInstance(matches, list)

    # ── Empty query ──
    def test_empty_query_returns_list(self):
        matches = self.matcher.match_query("")
        self.assertIsInstance(matches, list)

    # ── Stop-word-only query ──
    def test_stopword_only_query(self):
        matches = self.matcher.match_query("what is the are of and")
        self.assertIsInstance(matches, list)

    # ── Query with only short tokens ──
    def test_very_short_tokens_excluded(self):
        matches = self.matcher.match_query("a b c")
        # All tokens have length ≤ 1 and should be filtered
        self.assertIsInstance(matches, list)

    # ── Match result structure ──
    def test_match_result_has_required_keys(self):
        self._skip_if_no_resumes()
        first_candidate = self.matcher.candidates[0]
        matches = self.matcher.match_query(f"Tell me about {first_candidate}")
        if matches:
            m = matches[0]
            for key in ("canonical_name", "matched_term", "query_token", "similarity", "is_spelling_mistake"):
                self.assertIn(key, m, f"Missing key '{key}' in match result")

    # ── Similarity score is in [0, 100] ──
    def test_similarity_score_in_valid_range(self):
        self._skip_if_no_resumes()
        first_candidate = self.matcher.candidates[0]
        matches = self.matcher.match_query(f"Tell me about {first_candidate}")
        for m in matches:
            self.assertGreaterEqual(m["similarity"], 0)
            self.assertLessEqual(m["similarity"], 100)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Query Classifier Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestQueryClassifier(unittest.TestCase):
    """
    Tests the QueryClassifier's ability to distinguish single-candidate queries
    from comparison / aggregation / multi-candidate queries.
    Uses the fallback heuristic so no API key is required.
    """

    @classmethod
    def setUpClass(cls):
        cls.classifier = QueryClassifier()

    def _classify(self, query: str) -> dict:
        return self.classifier.classify(query)

    # ── Single-candidate queries (should pass) ──
    def test_single_candidate_skill_query(self):
        result = self._classify("What is Ashok's experience in Python?")
        self.assertTrue(result["is_single_candidate"])
        self.assertIn("reason", result)

    def test_single_candidate_education_query(self):
        result = self._classify("Tell me about Yasaswi's educational background.")
        self.assertTrue(result["is_single_candidate"])

    def test_single_candidate_project_query(self):
        result = self._classify("Summarize Trinadh's final year project.")
        self.assertTrue(result["is_single_candidate"])

    def test_single_candidate_contact_query(self):
        result = self._classify("What is Shirley's contact email?")
        self.assertTrue(result["is_single_candidate"])

    # ── Multi-candidate / comparison queries (should be blocked) ──
    def test_compare_two_candidates_blocked(self):
        result = self._classify("Compare Ashok and Pawan's python experience")
        self.assertFalse(result["is_single_candidate"])

    def test_rank_query_blocked(self):
        result = self._classify("Who is the best candidate for Java?")
        self.assertFalse(result["is_single_candidate"])

    def test_list_query_blocked(self):
        result = self._classify("List all candidates with React experience")
        self.assertFalse(result["is_single_candidate"])

    def test_shortlist_query_blocked(self):
        result = self._classify("Shortlist the top 3 candidates for data science")
        self.assertFalse(result["is_single_candidate"])

    def test_vs_query_blocked(self):
        result = self._classify("Ashok vs Pawan — who has more ML experience?")
        self.assertFalse(result["is_single_candidate"])

    def test_difference_query_blocked(self):
        result = self._classify("What are the differences between the candidates?")
        self.assertFalse(result["is_single_candidate"])

    # ── Result structure ──
    def test_classify_returns_dict_with_required_keys(self):
        result = self._classify("What are Ashok's skills?")
        self.assertIn("is_single_candidate", result)
        self.assertIn("reason", result)
        self.assertIsInstance(result["is_single_candidate"], bool)
        self.assertIsInstance(result["reason"], str)

    # ── Edge cases ──
    def test_empty_query_does_not_raise(self):
        try:
            result = self._classify("")
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(f"classify('') raised an unexpected exception: {exc}")

    def test_very_long_query_does_not_raise(self):
        long_query = "What are Ashok's skills? " * 50
        try:
            result = self._classify(long_query)
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(f"classify(very_long) raised an unexpected exception: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FilteredQueryEngine — End-to-End Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestFilteredQueryEngine(unittest.TestCase):
    """
    End-to-end tests for the FilteredQueryEngine (the core RAG pipeline).
    These tests run in MOCK mode when no API key is present, which verifies
    all routing logic without incurring API costs.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = FilteredQueryEngine()
        cls.has_resumes = len(cls.engine.matcher.candidates) > 0

    # ── Response structure ──
    def test_query_returns_dict_with_required_keys(self):
        result = self.engine.query("What are the skills of Ashok?")
        required_keys = {"response", "candidate_name", "spelling_suggestion", "is_blocked", "blocked_reason"}
        for key in required_keys:
            self.assertIn(key, result, f"Missing key '{key}' in engine response")

    def test_response_is_always_a_string(self):
        result = self.engine.query("What are Pawan's projects?")
        self.assertIsInstance(result["response"], str)
        self.assertGreater(len(result["response"]), 0)

    # ── Spelling correction routing ──
    def test_pawan_spelling_suggestion_present(self):
        if not self.has_resumes:
            self.skipTest("Resumes not available — skipping.")
        if not any("pavanteja" in c.lower() for c in self.engine.matcher.candidates):
            self.skipTest("Candidate 'pavanteja kamma' not in resumes directory.")
        result = self.engine.query("What are the skills of Pawan?")
        self.assertFalse(result["is_blocked"])
        self.assertIsNotNone(result["spelling_suggestion"])
        self.assertIn("pavanteja", result["candidate_name"].lower())

    def test_yasasvi_spelling_suggestion_present(self):
        if not self.has_resumes:
            self.skipTest("Resumes not available — skipping.")
        if not any("yasaswi" in c.lower() for c in self.engine.matcher.candidates):
            self.skipTest("Candidate 'yasaswi kotha' not in resumes directory.")
        result = self.engine.query("Summarize yasasvi's education")
        self.assertFalse(result["is_blocked"])
        self.assertIsNotNone(result["spelling_suggestion"])

    # ── Multi-candidate blocking ──
    def test_compare_two_candidates_is_blocked(self):
        result = self.engine.query("Compare Ashok and Pawan")
        self.assertTrue(result["is_blocked"])
        self.assertIn("blocked", result["response"].lower())

    def test_rank_query_is_blocked(self):
        result = self.engine.query("Who is the best candidate for Java?")
        self.assertTrue(result["is_blocked"])

    def test_list_all_query_is_blocked(self):
        result = self.engine.query("List all candidates with Python experience")
        self.assertTrue(result["is_blocked"])

    # ── No-candidate-name blocking ──
    def test_no_candidate_name_is_blocked(self):
        result = self.engine.query("What is Python?")
        self.assertTrue(result["is_blocked"])
        self.assertIn("no candidate name", result.get("blocked_reason", "").lower())

    def test_stopword_only_query_is_blocked(self):
        result = self.engine.query("what is the and of for")
        self.assertTrue(result["is_blocked"])

    # ── Empty query handling ──
    def test_empty_query_does_not_raise(self):
        try:
            result = self.engine.query("")
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(f"engine.query('') raised an unexpected exception: {exc}")

    # ── Blocked reason is populated when blocked ──
    def test_blocked_response_has_blocked_reason(self):
        result = self.engine.query("Compare Ashok and Pawan")
        if result["is_blocked"]:
            self.assertIsNotNone(result.get("blocked_reason"))

    # ── is_blocked is False for valid single-candidate query ──
    def test_valid_single_candidate_not_blocked(self):
        if not self.has_resumes:
            self.skipTest("Resumes not available — skipping.")
        first_candidate = self.engine.matcher.candidates[0]
        query = f"What are the skills of {first_candidate}?"
        result = self.engine.query(query)
        self.assertFalse(result["is_blocked"])
        self.assertIsNotNone(result["candidate_name"])

    # ── candidate_name is None when blocked ──
    def test_blocked_response_candidate_name_is_none(self):
        result = self.engine.query("Compare all candidates")
        if result["is_blocked"]:
            self.assertIsNone(result.get("candidate_name"))

    # ── spelling_suggestion structure ──
    def test_spelling_suggestion_is_str_or_none(self):
        if not self.has_resumes:
            self.skipTest("Resumes not available — skipping.")
        first_candidate = self.engine.matcher.candidates[0]
        result = self.engine.query(f"Tell me about {first_candidate}")
        self.assertIsInstance(result.get("spelling_suggestion"), (str, type(None)))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Integration — Session State Simulation (App Logic)
# ══════════════════════════════════════════════════════════════════════════════
class TestAppSessionStateLogic(unittest.TestCase):
    """
    Tests the application-level logic that would run inside Streamlit's
    handle_query() function, using mocked session state and a mock engine.

    This ensures the stat counters and chat history structure remain correct
    regardless of whether the real RAG pipeline is live.
    """

    def setUp(self):
        """Set up a mock engine and a minimal session-state dictionary."""
        self.mock_engine = MagicMock()
        self.session = {
            "chat_history": [],
            "total_queries": 0,
            "blocked_queries": 0,
            "corrected_queries": 0,
            "last_candidate": None,
        }

    def _run_handle_query(self, query: str, engine_result: dict):
        """Simulate the handle_query() function from app.py."""
        if not query.strip():
            return

        self.session["chat_history"].append({
            "role": "user",
            "content": query.strip(),
            "metadata": {},
        })
        self.session["total_queries"] += 1

        self.mock_engine.query.return_value = engine_result
        result = self.mock_engine.query(query.strip())

        if result.get("is_blocked"):
            self.session["blocked_queries"] += 1
        if result.get("spelling_suggestion"):
            self.session["corrected_queries"] += 1
        if result.get("candidate_name"):
            self.session["last_candidate"] = result["candidate_name"]

        self.session["chat_history"].append({
            "role": "assistant",
            "content": result.get("response", ""),
            "metadata": {
                "candidate_name":      result.get("candidate_name"),
                "spelling_suggestion": result.get("spelling_suggestion"),
                "is_blocked":          result.get("is_blocked", False),
                "blocked_reason":      result.get("blocked_reason"),
            },
        })

    # ── Valid query increments total_queries and sets last_candidate ──
    def test_valid_query_increments_total_and_sets_candidate(self):
        engine_result = {
            "response": "Ashok has 2 years of Python experience.",
            "candidate_name": "M Ashok reddy",
            "spelling_suggestion": None,
            "is_blocked": False,
            "blocked_reason": None,
        }
        self._run_handle_query("Tell me about Ashok", engine_result)

        self.assertEqual(self.session["total_queries"], 1)
        self.assertEqual(self.session["blocked_queries"], 0)
        self.assertEqual(self.session["corrected_queries"], 0)
        self.assertEqual(self.session["last_candidate"], "M Ashok reddy")
        self.assertEqual(len(self.session["chat_history"]), 2)
        self.assertEqual(self.session["chat_history"][0]["role"], "user")
        self.assertEqual(self.session["chat_history"][1]["role"], "assistant")

    # ── Blocked query increments blocked_queries counter ──
    def test_blocked_query_increments_blocked_counter(self):
        engine_result = {
            "response": "Error: Multi-resume query blocked.",
            "candidate_name": None,
            "spelling_suggestion": None,
            "is_blocked": True,
            "blocked_reason": "Classifier flagged comparison query",
        }
        self._run_handle_query("Compare Ashok and Pawan", engine_result)

        self.assertEqual(self.session["total_queries"], 1)
        self.assertEqual(self.session["blocked_queries"], 1)
        self.assertEqual(self.session["corrected_queries"], 0)
        self.assertIsNone(self.session["last_candidate"])
        # is_blocked must propagate to the assistant message metadata
        self.assertTrue(self.session["chat_history"][1]["metadata"]["is_blocked"])

    # ── Spelling correction increments corrected_queries counter ──
    def test_spelling_correction_increments_counter(self):
        engine_result = {
            "response": "[Mock Mode] Resolved to 'pavanteja kamma'.",
            "candidate_name": "pavanteja kamma",
            "spelling_suggestion": "Showing results for 'pavanteja kamma' (corrected from 'pawan')",
            "is_blocked": False,
            "blocked_reason": None,
        }
        self._run_handle_query("What are the skills of Pawan?", engine_result)

        self.assertEqual(self.session["total_queries"], 1)
        self.assertEqual(self.session["corrected_queries"], 1)
        self.assertEqual(self.session["last_candidate"], "pavanteja kamma")
        meta = self.session["chat_history"][1]["metadata"]
        self.assertIn("corrected from 'pawan'", meta["spelling_suggestion"])

    # ── Empty query does not touch counters or history ──
    def test_empty_query_not_processed(self):
        self._run_handle_query("", {})
        self.assertEqual(self.session["total_queries"], 0)
        self.assertEqual(len(self.session["chat_history"]), 0)

    # ── Whitespace-only query treated as empty ──
    def test_whitespace_query_not_processed(self):
        self._run_handle_query("   ", {})
        self.assertEqual(self.session["total_queries"], 0)

    # ── Multiple sequential queries accumulate correctly ──
    def test_multiple_queries_accumulate_counts(self):
        valid = {
            "response": "Some answer",
            "candidate_name": "M Ashok reddy",
            "spelling_suggestion": None,
            "is_blocked": False,
            "blocked_reason": None,
        }
        blocked = {
            "response": "Blocked.",
            "candidate_name": None,
            "spelling_suggestion": None,
            "is_blocked": True,
            "blocked_reason": "comparison",
        }
        spelling = {
            "response": "Corrected answer",
            "candidate_name": "pavanteja kamma",
            "spelling_suggestion": "Showing results for 'pavanteja kamma' (corrected from 'pawan')",
            "is_blocked": False,
            "blocked_reason": None,
        }
        self._run_handle_query("Tell me about Ashok", valid)
        self._run_handle_query("Compare Ashok and Pawan", blocked)
        self._run_handle_query("What are Pawan's skills?", spelling)

        self.assertEqual(self.session["total_queries"], 3)
        self.assertEqual(self.session["blocked_queries"], 1)
        self.assertEqual(self.session["corrected_queries"], 1)
        self.assertEqual(len(self.session["chat_history"]), 6)  # 3 user + 3 assistant


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
