import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import List, Optional
from pydantic import BaseModel

# Updated imports – no 'backend.' prefix
from utils import (
    extract_text_from_file,
    extract_skills_with_mistral,
    fetch_jobs_by_keywords,
    compute_match_score,
    export_to_csv,
)

app = FastAPI(title="RAY - Resume-based Application Yield")


# -------------------- FORCED CORS MIDDLEWARE --------------------
class ForceCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Set CORS headers for every response – use your exact frontend URL
        response.headers["Access-Control-Allow-Origin"] = (
            "https://job-recommender-sigma.vercel.app"
        )
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, Accept, Origin"
        )
        return response


app.add_middleware(ForceCORSMiddleware)


# Handle OPTIONS preflight requests for all paths
@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://job-recommender-sigma.vercel.app",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept, Origin",
        },
    )


# -------------------- END CORS FIX --------------------


# ---------- Request/Response Models ----------
class JobResponse(BaseModel):
    title: str
    company: str
    city: str
    country: str
    remote: bool
    description: str
    apply_link: str
    salary: str
    posted_date: str
    source: str
    match_score: Optional[float] = None


class PreferenceRequest(BaseModel):
    keywords: List[str]  # user-provided roles/interests
    min_score: float = 0  # optional filter


# ---------- Endpoint 1: Resume Upload ----------
@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    # 1. Extract text
    try:
        text = extract_text_from_file(file)
        if not text:
            raise HTTPException(400, "Could not extract text. Upload a clean PDF/DOCX.")
    except Exception as e:
        raise HTTPException(400, str(e))

    # 2. Extract skills & experience level using Mistral
    try:
        extracted = extract_skills_with_mistral(text)
    except Exception as e:
        raise HTTPException(500, f"Skill extraction failed: {str(e)}")

    skills = extracted["skills"]
    exp_level = extracted["experience_level"]

    # 3. Fetch jobs using those skills (India + Global)
    india_jobs = fetch_jobs_by_keywords(skills, country="IN")
    global_jobs = fetch_jobs_by_keywords(skills, country="US")
    all_jobs = india_jobs + global_jobs

    # 4. Compute match scores
    for job in all_jobs:
        job["match_score"] = compute_match_score(text, job["description"])

    # 5. Return
    return {"skills": skills, "experience_level": exp_level, "jobs": all_jobs}


# ---------- Endpoint 2: Preference-Based ----------
@app.post("/api/preference-jobs")
async def preference_jobs(req: PreferenceRequest):
    if not req.keywords:
        raise HTTPException(400, "Provide at least one keyword.")
    # Fetch jobs for each keyword (India + Global)
    all_jobs = []
    for kw in req.keywords[:5]:
        india = fetch_jobs_by_keywords([kw], "IN")
        global_jobs = fetch_jobs_by_keywords([kw], "US")
        all_jobs.extend(india)
        all_jobs.extend(global_jobs)
    # Deduplicate
    seen = set()
    unique = []
    for job in all_jobs:
        key = f"{job['title']}_{job['company']}_{job['city']}"
        if key not in seen:
            seen.add(key)
            unique.append(job)
    # Optional match score based on keyword presence (simple)
    for job in unique:
        title_desc = (job["title"] + " " + job["description"]).lower()
        matches = sum(1 for kw in req.keywords if kw.lower() in title_desc)
        job["match_score"] = (
            round((matches / len(req.keywords)) * 100, 2) if req.keywords else 0
        )
    # Filter by min_score if provided
    if req.min_score > 0:
        unique = [j for j in unique if j.get("match_score", 0) >= req.min_score]
    return {"jobs": unique}


# ---------- Endpoint 3: CSV Export ----------
@app.post("/api/export-csv")
async def export_csv(jobs: List[JobResponse]):
    if not jobs:
        raise HTTPException(400, "No jobs to export")
    # Convert to dict list
    jobs_dict = [j.dict() for j in jobs]
    filepath = export_to_csv(jobs_dict)
    return FileResponse(filepath, media_type="text/csv", filename="opportunities.csv")


@app.get("/")
def root():
    return {"message": "RAY Backend is running"}
