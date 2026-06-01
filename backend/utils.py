import os
import re
import pdfplumber
import docx
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()


# ---------- Text extraction from PDF/DOCX ----------
def extract_text_from_file(file) -> str:
    """Extract text from PDF or DOCX file."""
    filename = file.filename.lower()
    text = ""
    if filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
    return text.strip()


# ---------- Mistral skill extraction ----------
def extract_skills_with_mistral(text: str) -> dict:
    """Use Mistral API to extract skills and experience level."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY missing in .env")
    client = MistralClient(api_key=api_key)
    prompt = f"""
    Analyze the following resume and return a JSON object with:
    - "skills": list of technical and soft skills (max 15)
    - "experience_level": one of ["Junior", "Mid", "Senior"] based on years of experience and role seniority.

    Resume:
    {text[:3000]}
    """
    messages = [ChatMessage(role="user", content=prompt)]
    try:
        response = client.chat(model="mistral-small-latest", messages=messages)
        result = response.choices[0].message.content
        # Parse JSON from response (simple regex, assume it's valid)
        import json

        # Find JSON block
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "skills": data.get("skills", [])[:15],
                "experience_level": data.get("experience_level", "Mid"),
            }
    except Exception as e:
        print("Mistral error:", e)
    # Fallback: use keyword extraction
    return extract_skills_fallback(text)


def extract_skills_fallback(text: str) -> dict:
    """Simple keyword-based fallback when Mistral fails."""
    skills_db = [
        "python",
        "java",
        "javascript",
        "react",
        "angular",
        "vue",
        "node.js",
        "sql",
        "mongodb",
        "aws",
        "docker",
        "kubernetes",
        "machine learning",
        "data science",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "git",
    ]
    found = [s for s in skills_db if s in text.lower()]
    exp_level = "Mid"
    if re.search(r"\b(junior|entry|0-2 years|fresher)\b", text, re.I):
        exp_level = "Junior"
    elif re.search(r"\b(senior|lead|principal|5\+ years)\b", text, re.I):
        exp_level = "Senior"
    return {"skills": found[:15], "experience_level": exp_level}


# ---------- JSearch API ----------
def fetch_jsearch(query: str, country: str = "IN") -> list:
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        return []
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    params = {
        "query": f"{query} in {country}",
        "page": "1",
        "num_pages": "2",
        "date_posted": "week",
    }
    jobs = []
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for job in data.get("data", []):
                if not job.get("job_apply_link"):
                    continue
                salary = "Not specified"
                min_sal = job.get("job_min_salary")
                max_sal = job.get("job_max_salary")
                if min_sal and max_sal:
                    curr = job.get("job_salary_currency", "USD")
                    salary = (
                        f"₹{min_sal:,} - ₹{max_sal:,}"
                        if curr == "INR"
                        else f"${min_sal:,} - ${max_sal:,}"
                    )
                jobs.append(
                    {
                        "title": job.get("job_title", query),
                        "company": job.get("employer_name", "Unknown"),
                        "city": job.get("job_city", ""),
                        "country": job.get(
                            "job_country", "India" if country == "IN" else "USA"
                        ),
                        "remote": job.get("job_is_remote", False),
                        "description": job.get("job_description", "")[:1000],
                        "apply_link": job.get("job_apply_link"),
                        "salary": salary,
                        "posted_date": job.get("job_posted_at_datetime_utc", ""),
                        "source": "JSearch",
                    }
                )
    except Exception as e:
        print("JSearch error:", e)
    return jobs


# ---------- Adzuna API ----------
def fetch_adzuna(query: str, country: str = "IN") -> list:
    app_id = os.getenv("ADZUNA_APP_ID")
    api_key = os.getenv("ADZUNA_API_KEY")
    if not app_id or not api_key:
        return []
    country_code = "in" if country == "IN" else "us"
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {
        "app_id": app_id,
        "app_key": api_key,
        "what": query,
        "results_per_page": 50,
        "max_days_old": 30,
    }
    if country == "IN":
        params["where"] = "india"
    jobs = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for job in data.get("results", []):
                company = job.get("company", {})
                if isinstance(company, dict):
                    company_name = company.get("display_name", "Unknown")
                else:
                    company_name = str(company)
                location = job.get("location", {})
                if isinstance(location, dict):
                    city = location.get("area", [""])[-1]
                else:
                    city = str(location)
                salary = "Not specified"
                min_sal = job.get("salary_min")
                max_sal = job.get("salary_max")
                if min_sal and max_sal:
                    salary = (
                        f"₹{min_sal:,} - ₹{max_sal:,}"
                        if country == "IN"
                        else f"${min_sal:,} - ${max_sal:,}"
                    )
                jobs.append(
                    {
                        "title": job.get("title", query),
                        "company": company_name,
                        "city": city,
                        "country": "India" if country == "IN" else country,
                        "remote": "remote" in str(job.get("contract_type", "")).lower(),
                        "description": job.get("description", "")[:1000],
                        "apply_link": job.get("redirect_url", "#"),
                        "salary": salary,
                        "posted_date": job.get("created", ""),
                        "source": "Adzuna",
                    }
                )
    except Exception as e:
        print("Adzuna error:", e)
    return jobs


# ---------- Unified job fetch (both APIs) ----------
def fetch_jobs_by_keywords(keywords: list, country: str = "IN") -> list:
    """Fetch jobs for a list of keywords (skills or user preferences)."""
    all_jobs = []
    for kw in keywords[:3]:  # limit to 3 keywords to avoid rate limits
        jobs1 = fetch_jsearch(kw, country)
        jobs2 = fetch_adzuna(kw, country)
        all_jobs.extend(jobs1)
        all_jobs.extend(jobs2)
    # Deduplicate by (title, company, city)
    unique = []
    seen = set()
    for job in all_jobs:
        key = f"{job['title']}_{job['company']}_{job['city']}"
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


# ---------- Match score (cosine similarity) ----------
def compute_match_score(resume_text: str, job_description: str) -> float:
    if not resume_text or not job_description:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        vectors = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
        return round(score, 2)
    except:
        return 0.0


# ---------- CSV export ----------
def export_to_csv(jobs: list, filename: str = "opportunities") -> str:
    df = pd.DataFrame(jobs)
    cols = [
        "title",
        "company",
        "city",
        "country",
        "salary",
        "posted_date",
        "description",
        "apply_link",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = [
        "Title",
        "Company",
        "City",
        "Country",
        "Salary",
        "Posted Date",
        "Description",
        "Apply Link",
    ]
    filepath = f"{filename}.csv"
    df.to_csv(filepath, index=False, encoding="utf-8")
    return filepath
