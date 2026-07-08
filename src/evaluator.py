import time
import json
import logging
from typing import List, Dict, Any
from retriever import FilteredQueryEngine
# Add app/ directory to sys.path to avoid name collision with app.py file
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "app") not in sys.path:
    sys.path.append(str(ROOT_DIR / "app"))

from llm.openrouter_service import OpenRouterClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluator")

# Defined evaluation cases based on the actual resumes present in Resumes/ directory
EVAL_CASES = [
    {
        "query": "Summarize Trinadh's resume",
        "expected_candidate": "Trinadh Kumar Reddi",
        "ground_truth": "Trinadh Kumar Reddi is a Software Engineer with experience in MuleSoft development, Java, Salesforce, and Cloud platforms (AWS, Azure). He holds multiple certifications like AWS Cloud Practitioner, Java SE 8 Programmer, Azure Fundamentals, and MuleSoft Developer."
    },
    {
        "query": "What are Trinadh's certifications?",
        "expected_candidate": "Trinadh Kumar Reddi",
        "ground_truth": "AWS Certified Cloud Practitioner, Microsoft Certified: Azure Fundamentals (AZ-900), Oracle Certified Associate: Java SE 8 Programmer, MuleSoft Certified Developer (Level 1)."
    },
    {
        "query": "What is Yasaswi's educational background?",
        "expected_candidate": "yasaswi kotha",
        "ground_truth": "B.Tech in Computer Science and Engineering from Vardhaman College of Engineering with a CGPA of 8.16."
    },
    {
        "query": "Tell me about Ashok Reddy's experience",
        "expected_candidate": "M Ashok reddy",
        "ground_truth": "Worked at Virtusa as Software Development Engineer (April 2024 - Present), Senior Software Engineer (Feb 2022 - March 2024), and at Capgemini as Software Engineer (Nov 2020 - Jan 2022)."
    }
]

class RAGEvaluator:
    """
    RAGEvaluator assesses the quality of the RAG pipeline using LLM-as-a-judge scorers.
    It runs queries against FilteredQueryEngine, compares responses to ground truth,
    and calculates RAG metrics (Faithfulness, Relevancy, Precision, Recall, Correctness).
    """
    def __init__(self):
        self.engine = FilteredQueryEngine()
        self.eval_client = OpenRouterClient()

    def _judge_metric(self, prompt: str) -> float:
        """Helper to invoke judge LLM and parse a float score from [0.0, 1.0]."""
        if not self.eval_client or not self.eval_client.api_key:
            return 0.5  # default baseline if key is missing
            
        try:
            messages = [
                {"role": "system", "content": "You are a precise, objective academic evaluator. Output ONLY a JSON object containing a 'score' float between 0.0 and 1.0 and a short 'reason' string. Do not output markdown other than JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self.eval_client.get_completion(messages=messages)
            
            # Clean response to parse JSON
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            score = float(data.get("score", 0.5))
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Failed to calculate metric: {e}. Defaulting to 0.5.")
            return 0.5

    def evaluate_faithfulness(self, context: str, response: str) -> float:
        """Measures if the response is fully supported by the retrieved context."""
        prompt = f"""Evaluate if the generated answer is faithful to and supported ONLY by the provided context.
Do not consider outside knowledge.

Retrieved Context:
{context}

Generated Answer:
{response}

JSON output format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means fully faithful and 0.0 means complete hallucination or unsupported statements>,
  "reason": "explanation of score"
}}"""
        return self._judge_metric(prompt)

    def evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """Measures if the generated response directly and fully addresses the user query."""
        prompt = f"""Evaluate if the generated answer directly answers the question.
Assess relevancy and completeness.

User Question:
{query}

Generated Answer:
{response}

JSON output format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means highly relevant and completely answers the question, and 0.0 means totally irrelevant>,
  "reason": "explanation of score"
}}"""
        return self._judge_metric(prompt)

    def evaluate_context_precision(self, query: str, context: str) -> float:
        """Measures how relevant the retrieved context chunks are to the user query."""
        prompt = f"""Evaluate the relevance of the retrieved context chunks to the user question.
Does the context contain details necessary to construct the answer?

User Question:
{query}

Retrieved Context:
{context}

JSON output format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means every piece of context is highly relevant, and 0.0 means completely useless context>,
  "reason": "explanation of score"
}}"""
        return self._judge_metric(prompt)

    def evaluate_context_recall(self, ground_truth: str, context: str) -> float:
        """Measures if the retrieved context covers all facts in the ground truth answer."""
        prompt = f"""Evaluate if the retrieved context covers all the facts mentioned in the Ground Truth answer.
Is there missing context that was needed?

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

JSON output format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means the context covers 100% of facts in the ground truth, and 0.0 means none of the facts are present in the context>,
  "reason": "explanation of score"
}}"""
        return self._judge_metric(prompt)

    def evaluate_answer_correctness(self, ground_truth: str, response: str) -> float:
        """Measures semantic similarity and factual accuracy between generated response and ground truth."""
        prompt = f"""Evaluate the factual correctness of the generated answer compared to the Ground Truth answer.

Ground Truth Answer:
{ground_truth}

Generated Answer:
{response}

JSON output format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means it is factually identical in meaning, and 0.0 means factually incorrect or opposite>,
  "reason": "explanation of score"
}}"""
        return self._judge_metric(prompt)

    def run_evaluation(self) -> Dict[str, Any]:
        """Runs the entire evaluation suite over defined test cases."""
        logger.info("Starting RAG evaluation suite...")
        results = []
        
        total_faithfulness = 0.0
        total_relevancy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_correctness = 0.0
        
        for case in EVAL_CASES:
            query = case["query"]
            ground_truth = case["ground_truth"]
            
            logger.info(f"Running evaluation query: '{query}'...")
            t0 = time.time()
            res = self.engine.query(query)
            latency_ms = (time.time() - t0) * 1000.0
            
            response = res["response"]
            # Reconstruct retrieved context for judges
            debug = res.get("debug_info", {})
            nodes = debug.get("retrieved_nodes", [])
            context = "\n\n".join([n["text"] for n in nodes if n.get("score", 0.0) >= 0.4])
            
            if not context:
                context = "No relevant context found above similarity threshold."

            # Calculate metrics
            faithfulness = self.evaluate_faithfulness(context, response)
            relevancy = self.evaluate_answer_relevancy(query, response)
            precision = self.evaluate_context_precision(query, context)
            recall = self.evaluate_context_recall(ground_truth, context)
            correctness = self.evaluate_answer_correctness(ground_truth, response)
            
            total_faithfulness += faithfulness
            total_relevancy += relevancy
            total_precision += precision
            total_recall += recall
            total_correctness += correctness
            
            results.append({
                "query": query,
                "candidate": res["candidate_name"],
                "response": response,
                "ground_truth": ground_truth,
                "latency_ms": latency_ms,
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevancy": relevancy,
                    "context_precision": precision,
                    "context_recall": recall,
                    "answer_correctness": correctness
                }
            })
            
        num_cases = len(EVAL_CASES)
        summary = {
            "average_faithfulness": total_faithfulness / num_cases,
            "average_answer_relevancy": total_relevancy / num_cases,
            "average_context_precision": total_precision / num_cases,
            "average_context_recall": total_recall / num_cases,
            "average_answer_correctness": total_correctness / num_cases
        }
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "results": results
        }
        
        # Save report
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        self.generate_markdown_report(report)
        logger.info("Evaluation complete. Reports generated: evaluation_report.json, evaluation_report.md")
        return report

    def generate_markdown_report(self, report: Dict[str, Any]):
        """Generates a formatted markdown evaluation report."""
        summary = report["summary"]
        md = f"""# RAG Evaluation Report
Generated on: {report["timestamp"]}

## Performance Summary

| Metric | Average Score | Target | Status |
|---|---|---|---|
| **Faithfulness** | {summary["average_faithfulness"]:.2f} | ≥ 0.90 | {"✅ Passed" if summary["average_faithfulness"] >= 0.90 else "⚠️ Investigate"} |
| **Answer Relevancy** | {summary["average_answer_relevancy"]:.2f} | ≥ 0.90 | {"✅ Passed" if summary["average_answer_relevancy"] >= 0.90 else "⚠️ Investigate"} |
| **Context Precision** | {summary["average_context_precision"]:.2f} | ≥ 0.85 | {"✅ Passed" if summary["average_context_precision"] >= 0.85 else "⚠️ Investigate"} |
| **Context Recall** | {summary["average_context_recall"]:.2f} | ≥ 0.85 | {"✅ Passed" if summary["average_context_recall"] >= 0.85 else "⚠️ Investigate"} |
| **Answer Correctness** | {summary["average_answer_correctness"]:.2f} | ≥ 0.80 | {"✅ Passed" if summary["average_answer_correctness"] >= 0.80 else "⚠️ Investigate"} |

---

## Detailed Test Case Results
"""
        for i, res in enumerate(report["results"], 1):
            metrics = res["metrics"]
            md += f"""
### Test Case {i}: {res["query"]}
- **Resolved Candidate**: {res["candidate"]}
- **Latency**: {res["latency_ms"]:.2f} ms

#### Generated Response:
> {res["response"]}

#### Ground Truth Answer:
> {res["ground_truth"]}

#### Scores:
- **Faithfulness**: {metrics["faithfulness"]:.2f}
- **Answer Relevancy**: {metrics["answer_relevancy"]:.2f}
- **Context Precision**: {metrics["context_precision"]:.2f}
- **Context Recall**: {metrics["context_recall"]:.2f}
- **Answer Correctness**: {metrics["answer_correctness"]:.2f}

---
"""
        with open("evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(md)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()
