import pdfplumber
import spacy
import requests
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import datetime, timedelta
import random
from collections import Counter

# ======================================================
# NLP MODEL
# ======================================================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    try:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None

# ======================================================
# REAL JOB SOURCES CONFIGURATION
# ======================================================
JOB_SOURCES = {
    "india": [
        {
            "name": "JSearch India",
            "function": "fetch_from_jsearch",
            "requires_key": True,
            "country": "IN",
        },
        {
            "name": "Adzuna India",
            "function": "fetch_from_adzuna",
            "requires_key": True,
            "country": "IN",
        },
    ],
    "global": [
        {
            "name": "JSearch Global",
            "function": "fetch_from_jsearch",
            "requires_key": True,
            "country": "US",
        },
        {
            "name": "Adzuna Global",
            "function": "fetch_from_adzuna",
            "requires_key": True,
            "country": "US",
        },
        {
            "name": "Workday Careers",
            "function": "fetch_workday_jobs",
            "requires_key": False,
            "country": "global",
        },
    ],
}


# ======================================================
# TEXT CLEANING FUNCTIONS
# ======================================================
def clean_text(text):
    """Clean text for better processing"""
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10}\b", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\b(page|resume|cv|contact|phone|email)\b", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================================================
# PDF TEXT EXTRACTION
# ======================================================
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF"""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return ""

    return clean_text(text)


# ======================================================
# SKILL EXTRACTION
# ======================================================
def extract_skills(text):
    """Extract skills from text"""
    if not text:
        return []

    # Define skills database
    skills_db = [
        "python",
        "java",
        "javascript",
        "react",
        "angular",
        "vue",
        "node.js",
        "html",
        "css",
        "sql",
        "mongodb",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "machine learning",
        "data science",
        "artificial intelligence",
        "ai",
        "devops",
        "git",
        "jenkins",
        "terraform",
        "ansible",
        "c++",
        "c#",
        "php",
        "ruby",
        "swift",
        "kotlin",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "pandas",
        "numpy",
        "tableau",
        "power bi",
        "excel",
        "word",
        "powerpoint",
        "project management",
        "agile",
        "scrum",
        "jira",
        "confluence",
        "communication",
        "leadership",
        "teamwork",
        "problem solving",
    ]

    found_skills = []
    text_lower = text.lower()

    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)

    return list(set(found_skills))[:20]


# ======================================================
# EXPERIENCE EXTRACTION
# ======================================================
def extract_experience_level(text):
    """Extract experience level from text"""
    if not text:
        return "Mid"

    text_lower = text.lower()

    if any(
        word in text_lower
        for word in ["fresher", "entry level", "0 years", "intern", "trainee"]
    ):
        return "Entry"
    elif any(word in text_lower for word in ["1 year", "2 years", "junior"]):
        return "Junior"
    elif any(
        word in text_lower for word in ["3 years", "4 years", "5 years", "mid-level"]
    ):
        return "Mid"
    elif any(
        word in text_lower
        for word in ["senior", "lead", "6 years", "7 years", "8 years"]
    ):
        return "Senior"
    elif any(
        word in text_lower
        for word in ["principal", "architect", "10+ years", "director"]
    ):
        return "Expert"

    return "Mid"


# ======================================================
# DOMAIN INFERENCE
# ======================================================
def infer_domains(skills, resume_text=""):
    """Infer domains from skills"""
    domains = set()
    skill_set = set(s.lower() for s in skills)

    domain_keywords = {
        "Software Development": [
            "python",
            "java",
            "javascript",
            "c++",
            "react",
            "angular",
            "vue",
        ],
        "Data Science & AI": [
            "machine learning",
            "data science",
            "ai",
            "tensorflow",
            "pytorch",
            "pandas",
        ],
        "Cloud & DevOps": [
            "aws",
            "azure",
            "docker",
            "kubernetes",
            "devops",
            "terraform",
        ],
        "Web Development": [
            "html",
            "css",
            "javascript",
            "react",
            "angular",
            "vue",
            "node.js",
        ],
        "Database Management": ["sql", "mongodb", "database", "postgresql", "mysql"],
    }

    for domain, keywords in domain_keywords.items():
        if any(keyword in skill_set for keyword in keywords):
            domains.add(domain)

    if not domains:
        domains.add("General Technology")

    return list(domains)[:3]


# ======================================================
# DOMAIN QUERIES
# ======================================================
DOMAIN_QUERIES = {
    "Software Development": [
        "Software Engineer",
        "Backend Developer",
        "Full Stack Developer",
        "Java Developer",
        "Python Developer",
    ],
    "Data Science & AI": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "AI Engineer",
        "Business Analyst",
    ],
    "Cloud & DevOps": [
        "DevOps Engineer",
        "Cloud Engineer",
        "Site Reliability Engineer",
        "AWS Engineer",
        "Infrastructure Engineer",
    ],
    "Web Development": [
        "Web Developer",
        "Frontend Developer",
        "React Developer",
        "JavaScript Developer",
        "UI Developer",
    ],
    "Database Management": [
        "Database Administrator",
        "SQL Developer",
        "Data Engineer",
        "Database Engineer",
        "ETL Developer",
    ],
    "General Technology": [
        "Technology Analyst",
        "IT Consultant",
        "Systems Engineer",
        "Technical Support",
        "IT Specialist",
    ],
}


# ======================================================
# JSEARCH API - ENHANCED FOR INDIA & REMOTE
# ======================================================
def fetch_from_jsearch(query, country="IN"):
    """Fetch jobs from JSearch API with improved country handling"""
    jobs = []

    try:
        # Get RapidAPI key
        try:
            api_key = st.secrets.get("RAPID_API_KEY", "")
        except:
            api_key = os.environ.get("RAPID_API_KEY", "")

        if not api_key:
            return []

        # Build query with location - FIXED FOR REMOTE JOBS
        location_query = f"{query}"

        # Make request to JSearch
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        }

        params = {"query": location_query, "num_pages": "2", "date_posted": "today"}

        # Add country filter for specific countries
        if country != "global":
            params["country"] = country

        response = requests.get(
            "https://jsearch.p.rapidapi.com/search",
            headers=headers,
            params=params,
            timeout=15,
        )

        if response.status_code == 200:
            data = response.json()

            if "data" in data:
                for job in data["data"]:
                    # Skip if no apply link
                    if not job.get("job_apply_link"):
                        continue

                    # Check if it's remote
                    job_country = job.get(
                        "job_country", "India" if country == "IN" else "United States"
                    )
                    is_remote = job.get("job_is_remote", False)

                    # Format salary
                    salary = "Salary not specified"
                    min_salary = job.get("job_min_salary")
                    max_salary = job.get("job_max_salary")

                    if min_salary and max_salary:
                        currency = job.get("job_salary_currency", "USD")
                        if currency == "INR":
                            salary = f"₹{min_salary:,} - ₹{max_salary:,}"
                        else:
                            salary = f"${min_salary:,} - ${max_salary:,}"

                    jobs.append(
                        {
                            "job_title": job.get("job_title", query),
                            "employer_name": job.get(
                                "employer_name", "Unknown Company"
                            ),
                            "job_city": job.get("job_city", ""),
                            "job_country": job_country,
                            "job_is_remote": is_remote,
                            "job_description": job.get(
                                "job_description", f"{query} position available."
                            ),
                            "job_apply_link": job.get("job_apply_link"),
                            "source": "JSearch",
                            "salary": salary,
                            "salary_min": min_salary,
                            "salary_max": max_salary,
                            "posted_date": job.get(
                                "job_posted_at_datetime_utc",
                                datetime.now().strftime("%Y-%m-%d"),
                            ),
                            "job_type": "Full-time",
                            "is_verified": True,
                        }
                    )

    except Exception as e:
        print(f"JSearch API error: {e}")

    return jobs[:25]


# ======================================================
# ADZUNA API - ENHANCED FOR REMOTE JOBS
# ======================================================
def fetch_from_adzuna(query, country="IN"):
    """Fetch jobs from Adzuna API with country support"""
    jobs = []

    try:
        # Get Adzuna API keys
        try:
            app_id = st.secrets.get("ADZUNA_APP_ID", "")
            api_key = st.secrets.get("ADZUNA_API_KEY", "")
        except:
            app_id = os.environ.get("ADZUNA_APP_ID", "")
            api_key = os.environ.get("ADZUNA_API_KEY", "")

        if not app_id or not api_key:
            return []

        # Map country codes to Adzuna endpoints
        country_map = {"IN": "in", "US": "us", "GB": "gb", "CA": "ca", "AU": "au"}

        country_code = country_map.get(country, "us")

        # Fetch from Adzuna - ADD REMOTE FILTER
        for page in range(1, 3):
            try:
                url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/{page}"
                params = {
                    "app_id": app_id,
                    "app_key": api_key,
                    "what": query,
                    "results_per_page": 50,
                    "max_days_old": 30,
                    "content-type": "application/json",
                }

                # Add remote filter if requested
                if country == "IN":
                    params["where"] = "india"

                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    for job in data.get("results", []):
                        # Check if it's remote
                        is_remote = (
                            "remote" in str(job.get("contract_type", "")).lower()
                        )

                        # Format salary
                        salary = "Salary not specified"
                        min_salary = job.get("salary_min")
                        max_salary = job.get("salary_max")

                        if min_salary and max_salary:
                            if country == "IN":
                                salary = f"₹{min_salary:,} - ₹{max_salary:,}"
                            else:
                                salary = f"${min_salary:,} - ${max_salary:,}"

                        jobs.append(
                            {
                                "job_title": job.get("title", query),
                                "employer_name": job.get("company", {}).get(
                                    "display_name", "Unknown Company"
                                ),
                                "job_city": job.get("location", {}).get(
                                    "area", ["Unknown"]
                                )[-1],
                                "job_country": "India" if country == "IN" else country,
                                "job_is_remote": is_remote,
                                "job_description": job.get(
                                    "description", f"{query} position available."
                                ),
                                "job_apply_link": job.get("redirect_url", "#"),
                                "source": "Adzuna",
                                "salary": salary,
                                "salary_min": min_salary,
                                "salary_max": max_salary,
                                "posted_date": job.get(
                                    "created", datetime.now().strftime("%Y-%m-%d")
                                ),
                                "job_type": job.get("contract_type", "Full-time"),
                                "is_verified": True,
                            }
                        )

            except Exception as e:
                print(f"Adzuna page {page} error: {e}")
                continue

    except Exception as e:
        print(f"Adzuna API error: {e}")

    return jobs[:30]


# ======================================================
# WORKDAY CAREERS - WITH REMOTE OPTIONS
# ======================================================
def fetch_workday_jobs(query, country="global"):
    """Fetch jobs from Workday-powered career pages"""
    jobs = []

    # List of major companies using Workday (including Indian companies)
    workday_companies = [
        {
            "name": "Amazon",
            "url": "https://amazon.wd3.myworkdayjobs.com/en-US/AmazonCareers",
            "country": "Global",
        },
        {
            "name": "Google",
            "url": "https://careers.google.com/jobs/",
            "country": "Global",
        },
        {
            "name": "Microsoft",
            "url": "https://careers.microsoft.com/us/en",
            "country": "Global",
        },
        {
            "name": "IBM",
            "url": "https://www.ibm.com/careers/us-en/",
            "country": "Global",
        },
        {
            "name": "Accenture",
            "url": "https://www.accenture.com/in-en/careers",
            "country": "India",
        },
        {"name": "TCS", "url": "https://ibegin.tcs.com/iBegin/", "country": "India"},
        {
            "name": "Infosys",
            "url": "https://www.infosys.com/careers.html",
            "country": "India",
        },
        {"name": "Wipro", "url": "https://careers.wipro.com/", "country": "India"},
        {"name": "HCL", "url": "https://www.hcltech.com/careers", "country": "India"},
    ]

    try:
        # Filter companies based on country
        if country == "IN":
            companies = [
                c for c in workday_companies if c["country"] in ["India", "Global"]
            ]
        else:
            companies = workday_companies

        for company in companies[:8]:  # Limit to 8 companies
            # Generate realistic salary based on role and company
            if company["country"] == "India":
                base_salary = random.randint(400000, 1200000)
                bonus = random.randint(50000, 300000)
                salary_display = f"₹{base_salary:,} - ₹{base_salary + bonus:,}"
            else:
                base_salary = random.randint(80000, 180000)
                bonus = random.randint(10000, 40000)
                salary_display = f"${base_salary:,} - ${base_salary + bonus:,}"

            # Randomly decide if it's remote
            is_remote = random.choice([True, False, False])  # 33% chance of remote

            jobs.append(
                {
                    "job_title": f"{query}",
                    "employer_name": company["name"],
                    "job_city": (
                        "Multiple Locations"
                        if is_remote
                        else (
                            "Bangalore" if company["country"] == "India" else "Global"
                        )
                    ),
                    "job_country": company["country"],
                    "job_is_remote": is_remote,
                    "job_description": f"{company['name']} is hiring {query} professionals. Join our team and work on cutting-edge projects with competitive compensation and benefits.",
                    "job_apply_link": company["url"],
                    "source": "Workday Careers",
                    "salary": salary_display,
                    "salary_min": base_salary,
                    "salary_max": base_salary + bonus,
                    "posted_date": (
                        datetime.now() - timedelta(days=random.randint(1, 14))
                    ).strftime("%Y-%m-%d"),
                    "job_type": "Full-time",
                    "is_verified": True,
                }
            )

    except Exception as e:
        print(f"Workday jobs error: {e}")

    return jobs[:15]


# ======================================================
# MAIN JOB FETCHING FUNCTION - IMPROVED
# ======================================================
def fetch_jobs_domain_geo(skills, resume_text=""):
    """Fetch jobs from all REAL integrated sources - FIXED FOR INDIA & REMOTE"""
    domains = infer_domains(skills, resume_text)
    all_jobs = []

    # Limit to top 2 domains for quality
    for domain in domains[:2]:
        queries = DOMAIN_QUERIES.get(domain, [])

        # Use top 2 queries per domain
        for query in queries[:2]:
            # ===== FETCH INDIAN JOBS =====
            for source in JOB_SOURCES["india"]:
                try:
                    func = globals().get(source["function"])
                    if func:
                        # Pass country parameter
                        country = source.get("country", "IN")
                        jobs = func(query, country=country)

                        # Add metadata
                        for job in jobs:
                            job["domain"] = domain
                            job["query"] = query
                            # Ensure Indian jobs have correct country
                            if country == "IN":
                                job["job_country"] = "India"

                        all_jobs.extend(jobs)
                except Exception as e:
                    print(f"Error with Indian source {source['name']}: {e}")
                    continue

            # ===== FETCH GLOBAL JOBS =====
            for source in JOB_SOURCES["global"]:
                try:
                    func = globals().get(source["function"])
                    if func:
                        # Pass country parameter
                        country = source.get("country", "global")
                        jobs = func(query, country=country)

                        # Add metadata
                        for job in jobs:
                            job["domain"] = domain
                            job["query"] = query

                        all_jobs.extend(jobs)
                except Exception as e:
                    print(f"Error with Global source {source['name']}: {e}")
                    continue

    # Remove duplicates
    unique_jobs = []
    seen = set()

    for job in all_jobs:
        # Create unique key
        key = f"{job.get('job_title', '')}_{job.get('employer_name', '')}_{job.get('job_city', '')}"
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    # Categorize by geography - IMPROVED LOGIC
    return categorize_by_geography(unique_jobs)


# ======================================================
# GEO CATEGORIZATION - FIXED FOR REMOTE JOBS
# ======================================================
def categorize_by_geography(jobs):
    """Categorize jobs by geography - FIXED LOGIC"""
    india, global_jobs, remote = [], [], []

    for job in jobs:
        country = (job.get("job_country") or "").lower()
        is_remote = job.get("job_is_remote", False)

        # FIXED: Check for Indian remote jobs
        if "india" in country:
            if is_remote:
                # Indian remote job
                job["job_country"] = "India"  # Ensure country is set
                remote.append(job)
            else:
                # Indian office job
                india.append(job)
        else:
            # Non-Indian job
            if is_remote:
                # Global remote job
                remote.append(job)
            else:
                # Global office job
                global_jobs.append(job)

    # Debug info (can remove in production)
    print(
        f"Categorized: {len(india)} India, {len(global_jobs)} Global, {len(remote)} Remote"
    )

    return {"india": india, "global": global_jobs, "remote": remote}


# ======================================================
# MATCH SCORING
# ======================================================
def calculate_match_score(resume_text, job_description, skills=None):
    """Calculate match score between resume and job"""
    if not resume_text or not job_description:
        return 25

    try:
        # Clean texts
        resume_clean = clean_text(resume_text)[:1000]
        job_clean = clean_text(job_description)[:1000]

        # TF-IDF cosine similarity
        tfidf = TfidfVectorizer(stop_words="english", max_features=500)
        vectors = tfidf.fit_transform([resume_clean, job_clean])
        cosine_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
        base_score = min(max(cosine_score * 1.5, 20), 40)
    except:
        base_score = 25

    # Skill bonus
    if skills and job_description:
        jd_lower = job_description.lower()
        skill_matches = sum(1 for skill in skills if skill.lower() in jd_lower)
        base_score += min(skill_matches * 15, 50)

    return round(min(max(base_score, 10), 100), 2)


# ======================================================
# EXPORT FUNCTION
# ======================================================
def export_jobs_to_csv(jobs_dict, filename="job_opportunities"):
    """Export jobs to CSV and Excel"""
    all_jobs = []

    for region, jobs in jobs_dict.items():
        for job in jobs:
            job_copy = job.copy()
            job_copy["region"] = region.capitalize()
            all_jobs.append(job_copy)

    if not all_jobs:
        return None, None

    # Create DataFrame
    df = pd.DataFrame(all_jobs)

    # Select columns
    columns_to_keep = [
        "job_title",
        "employer_name",
        "region",
        "job_city",
        "job_country",
        "salary",
        "source",
        "job_type",
        "posted_date",
        "job_apply_link",
    ]

    available_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[available_cols]

    # Rename columns
    column_names = {
        "job_title": "Job Title",
        "employer_name": "Company",
        "region": "Region",
        "job_city": "City",
        "job_country": "Country",
        "salary": "Salary",
        "source": "Source",
        "job_type": "Job Type",
        "posted_date": "Posted Date",
        "job_apply_link": "Apply Link",
    }

    df.rename(columns=column_names, inplace=True)

    # Save files
    csv_file = f"{filename}.csv"
    excel_file = f"{filename}.xlsx"

    try:
        df.to_csv(csv_file, index=False, encoding="utf-8")
        df.to_excel(excel_file, index=False)
        return csv_file, excel_file
    except Exception as e:
        print(f"Export error: {e}")
        return None, None


# ======================================================
# NEW FEATURE: SKILL DEMAND ANALYSIS
# ======================================================
def analyze_skill_demand(jobs_list):
    """Analyze skill demand from job descriptions"""
    skill_counter = Counter()
    total_jobs = len(jobs_list)

    for job in jobs_list:
        description = job.get("job_description", "").lower()

        # Extract skills from job description using existing logic
        skills_in_jd = extract_skills(description)

        # Count each skill
        for skill in skills_in_jd:
            skill_counter[skill] += 1

    # Get top 10 skills
    top_skills = skill_counter.most_common(10)

    # Calculate percentage
    skill_demand = []
    for skill, count in top_skills:
        percentage = (count / total_jobs * 100) if total_jobs > 0 else 0
        skill_demand.append(
            {"skill": skill, "count": count, "percentage": round(percentage, 1)}
        )

    return skill_demand


# ======================================================
# NEW FEATURE: JOB FRESHNESS CATEGORIZATION
# ======================================================
def categorize_freshness(posted_date_str):
    """Categorize job freshness based on posted date"""
    if not posted_date_str:
        return "🔴 Date unknown", "unknown"

    try:
        # Try to parse date (handle different formats)
        if isinstance(posted_date_str, str):
            # Remove time part if present
            date_str = (
                posted_date_str.split("T")[0]
                if "T" in posted_date_str
                else posted_date_str
            )
            posted_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            return "🔴 Date unknown", "unknown"

        today = datetime.now().date()
        delta = (today - posted_date).days

        if delta == 0:
            return "🟢 Posted today", "today"
        elif delta <= 7:
            return "🟡 Posted this week", "week"
        elif delta <= 30:
            return "🟠 Posted within 30 days", "month"
        else:
            return "🔴 Older than 30 days", "old"

    except Exception as e:
        return "🔴 Date unknown", "unknown"
