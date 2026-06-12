"""Search for real job postings from free public job board APIs."""

import logging
from dataclasses import dataclass
from urllib.parse import quote_plus, urlencode

import httpx

logger = logging.getLogger(__name__)

TIMEOUT = 12.0  # seconds


@dataclass
class WebJob:
    title: str
    company: str
    location: str
    description: str
    url: str = ""
    source: str = ""


def _search_remotive(query: str, limit: int = 20) -> list[WebJob]:
    """Search Remotive API — free, no key needed, remote-only jobs."""
    try:
        url = f"https://remotive.com/api/remote-jobs?search={quote_plus(query)}&limit={limit}"
        resp = httpx.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for j in data.get("jobs", [])[:limit]:
            desc = j.get("description", "")
            # Strip HTML tags from description
            import re

            desc = re.sub(r"<[^>]+>", " ", desc)
            desc = re.sub(r"\s+", " ", desc).strip()
            if not desc:
                continue
            jobs.append(WebJob(
                title=j.get("title", "Unknown"),
                company=j.get("company_name", "Unknown"),
                location=j.get("candidate_required_location", "Remote"),
                description=desc[:2000],
                url=j.get("url", ""),
                source="Remotive",
            ))
        return jobs
    except Exception as e:
        logger.warning("Remotive search failed: %s", e)
        return []


def _search_arbeitnow(query: str, limit: int = 20) -> list[WebJob]:
    """Search Arbeitnow API — free, no key needed."""
    try:
        url = "https://www.arbeitnow.com/api/job-board-api?" + urlencode({
            "search": query, "page": 1,
        })
        resp = httpx.get(url, timeout=TIMEOUT, headers={
            "User-Agent": "JobScope/1.0",
        })
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for j in data.get("data", [])[:limit]:
            desc = j.get("description", "")
            import re

            desc = re.sub(r"<[^>]+>", " ", desc)
            desc = re.sub(r"\s+", " ", desc).strip()
            if not desc:
                continue
            tags = ", ".join(j.get("tags", []))
            if tags:
                desc = f"Skills/Tags: {tags}\n\n{desc}"
            jobs.append(WebJob(
                title=j.get("title", "Unknown"),
                company=j.get("company_name", "Unknown"),
                location=j.get("location", "Unknown"),
                description=desc[:2000],
                url=j.get("url", ""),
                source="Arbeitnow",
            ))
        return jobs
    except Exception as e:
        logger.warning("Arbeitnow search failed: %s", e)
        return []


def _search_himalayas(query: str, limit: int = 20) -> list[WebJob]:
    """Search Himalayas API — free, no key needed, remote jobs."""
    try:
        url = "https://himalayas.app/jobs/api?" + urlencode({
            "limit": limit, "q": query,
        })
        resp = httpx.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for j in data.get("jobs", [])[:limit]:
            desc = j.get("description", "") or j.get("excerpt", "")
            if not desc:
                continue
            jobs.append(WebJob(
                title=j.get("title", "Unknown"),
                company=j.get("companyName", "Unknown"),
                location=j.get("location", "Remote"),
                description=desc[:2000],
                url=j.get("applicationUrl", "") or j.get("url", ""),
                source="Himalayas",
            ))
        return jobs
    except Exception as e:
        logger.warning("Himalayas search failed: %s", e)
        return []


def search_jobs(query: str, limit: int = 15) -> list[WebJob]:
    """Search multiple free job boards and merge results.

    No API keys required — uses Remotive, Arbeitnow, and Himalayas APIs.
    """
    per_source = max(limit // 3, 5)
    results: list[WebJob] = []

    for searcher in [_search_remotive, _search_arbeitnow, _search_himalayas]:
        results.extend(searcher(query, per_source))

    # Deduplicate by (title, company)
    seen: set[tuple[str, str]] = set()
    unique: list[WebJob] = []
    for job in results:
        key = (job.title.lower(), job.company.lower())
        if key not in seen:
            seen.add(key)
            unique.append(job)

    return unique[:limit]
