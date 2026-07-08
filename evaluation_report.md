# RAG Evaluation Report
Generated on: 2026-07-08 17:20:45

## Performance Summary

| Metric | Average Score | Target | Status |
|---|---|---|---|
| **Faithfulness** | 0.93 | ≥ 0.90 | ✅ Passed |
| **Answer Relevancy** | 0.72 | ≥ 0.90 | ⚠️ Investigate |
| **Context Precision** | 0.69 | ≥ 0.85 | ⚠️ Investigate |
| **Context Recall** | 0.06 | ≥ 0.85 | ⚠️ Investigate |
| **Answer Correctness** | 0.00 | ≥ 0.80 | ⚠️ Investigate |

---

## Detailed Test Case Results

### Test Case 1: Summarize Trinadh's resume
- **Resolved Candidate**: Trinadh Kumar Reddi
- **Latency**: 15629.10 ms

#### Generated Response:
> Here is a summary of Trinadh Kumar Reddi's resume:

**Education:**
- B.Tech in Artificial Intelligence and Machine Learning (CGPA: 7.86) from Bonam Venkata Chalamayya Engineering College (2021-2025)
- Intermediate (MPC) from Sri Chaitanya Jr College (2019-2021)
- Class X from G R Mariappan Sir C V Raman (EM) P S (2018-2019)

**Skills:**
- Programming Languages: Python, SQL
- Data Analysis & Visualization: Pandas, NumPy, Matplotlib, Plotly, Power BI
- Web Scraping & Data Collection: BeautifulSoup, Requests, Regular Expressions (Regex)
- Machine Learning: Scikit-learn, Deep Learning Basics, NLP, RAG, LangGraph
- Databases: MySQL, RDBMS, SQL Queries, Joins
- Tools: Jupyter Notebook, MySQL, VS Code, Git & GitHub, Streamlit, Power BI, MS Excel
- Soft Skills: Teamwork, Critical Thinking, Communication, Problem Solving

**Experience:**
- Internship project at Innomatics Research Labs (built a production-style Retrieval-Augmented Generation system)

**Projects:**
- RAG Customer Support Assistant (LangGraph, ChromaDB & Streamlit)
- Risk-Return Evaluation of Indian Mutual Funds (Web Scraping & Data Analysis)
- Brain Stroke Prediction (Exploratory Data Analysis)
- Bank Account Simulation System
- Handwritten Character Recognition

**Certifications:**
- Microsoft Certified: Azure AI Fundamentals (June 19, 2024)
- Machine Learning with Python (ML0101EN) from Cognitive Class (IBM) (February 12, 2024)
- Python from GUVI (August 6, 2023)

**Technologies:**
- LangGraph
- ChromaDB
- Streamlit
- Python
- SQL
- MySQL
- RDBMS
- Jupyter Notebook
- VS Code
- Git & GitHub
- Power BI
- MS Excel

#### Ground Truth Answer:
> Trinadh Kumar Reddi is a Software Engineer with experience in MuleSoft development, Java, Salesforce, and Cloud platforms (AWS, Azure). He holds multiple certifications like AWS Cloud Practitioner, Java SE 8 Programmer, Azure Fundamentals, and MuleSoft Developer.

#### Scores:
- **Faithfulness**: 0.90
- **Answer Relevancy**: 1.00
- **Context Precision**: 0.95
- **Context Recall**: 0.00
- **Answer Correctness**: 0.00

---

### Test Case 2: What are Trinadh's certifications?
- **Resolved Candidate**: Trinadh Kumar Reddi
- **Latency**: 7304.25 ms

#### Generated Response:
> Trinadh Kumar Reddi's certifications are:

1. Microsoft Certified: Azure AI Fundamentals — Microsoft, June 19, 2024
2. Machine Learning with Python (ML0101EN) — Cognitive Class (IBM), February 12, 2024
3. Python — GUVI, August 6, 2023

#### Ground Truth Answer:
> AWS Certified Cloud Practitioner, Microsoft Certified: Azure Fundamentals (AZ-900), Oracle Certified Associate: Java SE 8 Programmer, MuleSoft Certified Developer (Level 1).

#### Scores:
- **Faithfulness**: 1.00
- **Answer Relevancy**: 1.00
- **Context Precision**: 0.00
- **Context Recall**: 0.25
- **Answer Correctness**: 0.00

---

### Test Case 3: What is Yasaswi's educational background?
- **Resolved Candidate**: yasaswi kotha
- **Latency**: 4646.65 ms

#### Generated Response:
> According to the retrieved resume context, Yasaswi's educational background is:

* Intermediate, Sri Gayatri Junior College (2020-2022), Board: TSBIE, Percentage: 89.2%
* 10th Grade, Pragathi Central School (2019-2020), Board: CBSE, Percentage: 78%

Note: Yasaswi is also mentioned as a "Computer Science undergraduate" in the profile section, but the resume does not provide further details about the undergraduate degree.

#### Ground Truth Answer:
> B.Tech in Computer Science and Engineering from Vardhaman College of Engineering with a CGPA of 8.16.

#### Scores:
- **Faithfulness**: 0.80
- **Answer Relevancy**: 0.90
- **Context Precision**: 0.90
- **Context Recall**: 0.00
- **Answer Correctness**: 0.00

---

### Test Case 4: Tell me about Ashok Reddy's experience
- **Resolved Candidate**: M Ashok reddy
- **Latency**: 1687.32 ms

#### Generated Response:
> The requested information was not found in the retrieved resume.

#### Ground Truth Answer:
> Worked at Virtusa as Software Development Engineer (April 2024 - Present), Senior Software Engineer (Feb 2022 - March 2024), and at Capgemini as Software Engineer (Nov 2020 - Jan 2022).

#### Scores:
- **Faithfulness**: 1.00
- **Answer Relevancy**: 0.00
- **Context Precision**: 0.90
- **Context Recall**: 0.00
- **Answer Correctness**: 0.00

---
