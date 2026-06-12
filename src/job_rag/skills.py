"""Skill extraction and overlap analysis using LLM."""

import json
import re

EXTRACT_PROMPT = """Extract a flat JSON array of distinct technical skills, tools,
frameworks, and domain competencies from the text below. Return ONLY a valid
JSON array of lowercase strings, nothing else. Limit to the 25 most important.

Example output: ["python", "fastapi", "docker", "machine learning", "postgresql"]

Text:
{text}
"""


def extract_skills_with_llm(text: str, llm) -> list[str]:
    """Use an LLM to extract a skill list from free-form text."""
    if not text.strip():
        return []
    # Truncate to ~3000 chars to stay within context limits
    prompt = EXTRACT_PROMPT.format(text=text[:3000])
    response = llm.invoke(prompt)
    content = response.content.strip()
    # Parse JSON array from the response (handle markdown fences)
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            skills = json.loads(match.group())
            return [s.strip().lower() for s in skills if isinstance(s, str)]
        except json.JSONDecodeError:
            pass
    return []


def extract_skills_heuristic(text: str) -> list[str]:
    """Fast keyword-based fallback when no LLM is available."""
    keywords = {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "sql", "nosql", "html",
        "css", "react", "angular", "vue", "svelte", "next.js", "node.js",
        "express", "django", "flask", "fastapi", "spring", "rails",
        "docker", "kubernetes", "terraform", "ansible", "aws", "azure", "gcp",
        "ci/cd", "git", "github", "gitlab", "jenkins", "circleci",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "rabbitmq", "graphql", "rest", "grpc", "microservices",
        "machine learning", "deep learning", "nlp", "computer vision",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
        "spark", "hadoop", "airflow", "mlops", "data engineering",
        "devops", "sre", "linux", "networking", "security",
        "agile", "scrum", "jira", "figma", "product management",
        "rag", "langchain", "llm", "vector database", "chroma", "pinecone",
    }
    text_lower = text.lower()
    return sorted(skill for skill in keywords if skill in text_lower)


def compute_skill_overlap(
    resume_skills: list[str], job_skills: list[str],
) -> dict:
    """Compute matched, missing, and extra skills between resume and job."""
    resume_set = {s.lower() for s in resume_skills}
    job_set = {s.lower() for s in job_skills}
    matched = sorted(resume_set & job_set)
    missing = sorted(job_set - resume_set)
    extra = sorted(resume_set - job_set)
    total = len(job_set)
    pct = round(len(matched) / total * 100) if total > 0 else 0
    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_percent": pct,
        "matched_count": len(matched),
        "required_count": total,
    }
