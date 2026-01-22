import streamlit as st
import utils
import pandas as pd
from datetime import datetime
import hashlib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="RAY • Resume-based Application Yield",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# CUSTOM CSS - REFINED
# ======================================================
st.markdown(
    """
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a237e;
        margin-bottom: 0.2rem;
    }
    .tagline {
        font-size: 1.1rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 2rem 0;
    }
    
    /* Component styling */
    .stButton > button {
        background-color: #1a237e;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #283593;
        transform: translateY(-1px);
    }
    .job-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.2rem;
        background: linear-gradient(145deg, #ffffff, #fafafa);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    .job-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Status indicators */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #bbdefb;
    }
    .match-excellent { 
        color: #2e7d32;
        font-weight: 600;
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .match-good { 
        color: #f57c00;
        font-weight: 600;
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .match-fair { 
        color: #1976d2;
        font-weight: 600;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 2px 8px;
        border-radius: 6px;
    }
    
    /* Freshness indicators */
    .freshness-today {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .freshness-week {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        color: #f57c00;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .freshness-month {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    .freshness-old {
        background: linear-gradient(135deg, #f5f5f5, #e0e0e0);
        color: #616161;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar improvements */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# SESSION STATE INITIALIZATION
# ======================================================
if "saved_jobs" not in st.session_state:
    st.session_state.saved_jobs = {}


# ======================================================
# HELPER FUNCTIONS
# ======================================================
def generate_job_id(job):
    """Generate unique ID for job"""
    job_str = f"{job.get('job_title', '')}_{job.get('employer_name', '')}_{job.get('job_city', '')}_{job.get('job_country', '')}"
    return hashlib.md5(job_str.encode()).hexdigest()[:12]


# ======================================================
# HEADER - CLEAN
# ======================================================
st.markdown('<h1 class="main-header">RAY</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Resume-based Application Yield</p>', unsafe_allow_html=True
)

st.markdown("---")

# ======================================================
# SIDEBAR - MINIMAL
# ======================================================
with st.sidebar:
    # Logo/App identity
    st.markdown(
        '<div class="sidebar-header">📄 Resume Input</div>', unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type="pdf",
        help="",
        key="resume_uploader",
    )

    if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
        st.success("✓ Resume uploaded")

    st.markdown("---")

    # Saved Jobs
    st.markdown(
        '<div class="sidebar-header">💾 Saved Jobs</div>', unsafe_allow_html=True
    )

    if st.session_state.saved_jobs:
        saved_count = len(st.session_state.saved_jobs)
        st.caption(f"{saved_count} job{'s' if saved_count != 1 else ''} saved")

        # Export saved jobs
        if st.button("Export Saved Jobs", use_container_width=True):
            saved_df = pd.DataFrame(st.session_state.saved_jobs.values())
            csv = saved_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download",
                data=csv,
                file_name=f"saved_jobs_{timestamp}.csv",
                mime="text/csv",
                key="download_saved_csv",
            )

        # Clear all
        if st.button("Clear All", use_container_width=True):
            st.session_state.saved_jobs = {}
            st.rerun()
    else:
        st.caption("No saved jobs yet")

    st.markdown("---")

    # Relevance filter
    st.markdown(
        '<div class="sidebar-header">🎯 Relevance Filter</div>', unsafe_allow_html=True
    )
    min_score = st.slider(
        "Minimum match score",
        10,
        100,
        10,
        key="relevance_slider",
    )

    st.markdown("---")

    # Export options
    st.markdown(
        '<div class="sidebar-header">📤 Export Options</div>', unsafe_allow_html=True
    )
    export_format = st.radio(
        "Format", ["CSV", "Excel", "Both"], horizontal=True, key="export_format"
    )

    st.markdown("---")

    # Minimal footer
    st.caption("RAY v1.0 • All opportunities verified")

# ======================================================
# MAIN LOGIC
# ======================================================
if not uploaded_file:
    # Clean landing page
    st.markdown("### Start Here")
    st.info("Upload your resume to discover relevant opportunities")

    # Minimal how-it-works
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
        1. **Upload** your resume (PDF)
        2. **Skills** are automatically extracted
        3. **Relevant opportunities** are identified
        4. **Review** and save matching jobs
        """
        )

    st.stop()

# ------------------------------------------------------
# RESUME PARSING
# ------------------------------------------------------
with st.spinner("Analyzing your resume..."):
    resume_text = utils.extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.error("Unable to extract text. Please upload a valid PDF.")
        st.stop()

    skills = utils.extract_skills(resume_text)
    experience_level = utils.extract_experience_level(resume_text)

if not skills:
    st.error("No skills detected in your resume.")
    st.stop()

st.success(f"✓ Resume analyzed • {experience_level} level")

# ------------------------------------------------------
# DETECTED SKILLS - COMPACT
# ------------------------------------------------------
with st.expander("Detected Skills", expanded=False):
    if skills:
        skill_cols = st.columns(4)
        for idx, skill in enumerate(skills[:16]):
            with skill_cols[idx % 4]:
                st.markdown(
                    f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True
                )
        if len(skills) > 16:
            st.caption(f"+ {len(skills) - 16} more")
    else:
        st.info("Showing general opportunities")

# ------------------------------------------------------
# JOB FETCHING
# ------------------------------------------------------
with st.spinner("Discovering opportunities..."):
    results = utils.fetch_jobs_domain_geo(skills, resume_text)

    india_jobs = results["india"]
    global_jobs = results["global"]
    remote_jobs = results["remote"]

    total_jobs = len(india_jobs) + len(global_jobs) + len(remote_jobs)

if total_jobs == 0:
    st.warning("No opportunities found")
    st.stop()

# ------------------------------------------------------
# DASHBOARD METRICS - CLEAN
# ------------------------------------------------------
st.markdown("## Opportunity Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-label">India</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{len(india_jobs)}</div>', unsafe_allow_html=True
    )

with col2:
    st.markdown('<div class="metric-label">Global</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{len(global_jobs)}</div>', unsafe_allow_html=True
    )

with col3:
    st.markdown('<div class="metric-label">Remote</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{len(remote_jobs)}</div>', unsafe_allow_html=True
    )

with col4:
    st.markdown('<div class="metric-label">Total</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_jobs}</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ------------------------------------------------------
# SKILL DEMAND INSIGHTS - MINIMAL
# ------------------------------------------------------
if total_jobs > 0:
    with st.expander("Top Skills in Demand", expanded=False):
        all_jobs = india_jobs + global_jobs + remote_jobs
        skill_demand = utils.analyze_skill_demand(all_jobs)

        if skill_demand:
            demand_df = pd.DataFrame(skill_demand)

            # Display top skills
            for idx, row in demand_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['skill']}**")
                with col2:
                    st.markdown(f"**{row['count']}** roles")

            st.caption(f"Based on {total_jobs} current opportunities")
        else:
            st.info("Skill analysis pending")

# ------------------------------------------------------
# EXPORT ALL - CLEAN
# ------------------------------------------------------
st.markdown("## Export Options")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("Export CSV", use_container_width=True, key="export_csv_all"):
        with st.spinner("Exporting..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file, excel_file = utils.export_jobs_to_csv(
                results, f"opportunities_{timestamp}"
            )
            if csv_file:
                with open(csv_file, "rb") as f:
                    csv_data = f.read()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"opportunities_{timestamp}.csv",
                    mime="text/csv",
                    key=f"download_csv_{timestamp}",
                )

with col2:
    if st.button("Export Excel", use_container_width=True, key="export_excel_all"):
        with st.spinner("Exporting..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file, excel_file = utils.export_jobs_to_csv(
                results, f"opportunities_{timestamp}"
            )
            if excel_file:
                with open(excel_file, "rb") as f:
                    excel_data = f.read()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"opportunities_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{timestamp}",
                )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ======================================================
# JOB DISPLAY FUNCTION - REFINED
# ======================================================
def display_jobs(jobs, region_label, region_emoji):
    """Display jobs with clean, professional UI"""
    if not jobs:
        return

    # Calculate scores
    scored_jobs = []
    for job in jobs:
        desc = job.get("job_description", "")
        score = utils.calculate_match_score(resume_text, desc, skills)
        scored_jobs.append((job, score))

    # Filter by minimum score
    filtered_jobs = [(job, score) for job, score in scored_jobs if score >= min_score]

    if not filtered_jobs:
        st.info(f"No {region_label.lower()} jobs meet the {min_score}% match threshold")
        return

    # Sort by score
    filtered_jobs.sort(key=lambda x: x[1], reverse=True)

    # Header
    st.markdown(f"## {region_emoji} {region_label} Opportunities")
    st.caption(f"{len(filtered_jobs)} relevant positions")

    # Display jobs
    for i, (job, score) in enumerate(filtered_jobs, 1):
        title = job.get("job_title", "Role")
        company = job.get("employer_name", "Company")
        city = job.get("job_city", "")
        country = job.get("job_country", "")
        location = f"{city}, {country}".strip(", ")
        apply_link = job.get("job_apply_link", "#")
        description = job.get("job_description", "")
        source = job.get("source", "")
        salary = job.get("salary", "Salary not specified")
        job_type = job.get("job_type", "Full-time")
        posted_date = job.get("posted_date", "")

        # Freshness
        freshness_label, freshness_class = utils.categorize_freshness(posted_date)

        # Match label
        if score >= 60:
            match_label = "Excellent match"
            match_class = "match-excellent"
        elif score >= 45:
            match_label = "Good match"
            match_class = "match-good"
        else:
            match_label = "Fair match"
            match_class = "match-fair"

        # Job card
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {title}")
                st.markdown(f"**{company}** • {location}")

                # Tags row
                tag_col1, tag_col2, tag_col3 = st.columns(3)
                with tag_col1:
                    st.markdown(
                        f'<span class="{match_class}">{match_label}</span>',
                        unsafe_allow_html=True,
                    )
                with tag_col2:
                    st.markdown(
                        f'<span class="{freshness_class}">{freshness_label}</span>',
                        unsafe_allow_html=True,
                    )
                with tag_col3:
                    if salary != "Salary not specified":
                        st.markdown(f"**{salary}**")
                    else:
                        st.markdown("Salary confidential")

            with col2:
                # Match score
                st.markdown("**Match Score**")
                st.progress(min(score / 100, 1.0))
                st.markdown(f"**{score}%**")

                # Apply button
                if apply_link != "#":
                    st.link_button("Apply →", apply_link, use_container_width=True)
                else:
                    st.button("Details", use_container_width=True, disabled=True)

            # Skills section
            matched_skills = []
            if skills and description:
                matched_skills = [s for s in skills if s.lower() in description.lower()]

            if matched_skills:
                with st.expander("Matching Skills", expanded=False):
                    skill_cols = st.columns(4)
                    for idx, skill in enumerate(matched_skills[:8]):
                        with skill_cols[idx % 4]:
                            st.markdown(
                                f'<span class="skill-tag">{skill}</span>',
                                unsafe_allow_html=True,
                            )
                    if len(matched_skills) > 8:
                        st.caption(f"+ {len(matched_skills) - 8} more")

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                job_id = generate_job_id(job)
                if job_id in st.session_state.saved_jobs:
                    if st.button(
                        "Remove from Saved",
                        use_container_width=True,
                        key=f"remove_{job_id}_{i}",
                    ):
                        del st.session_state.saved_jobs[job_id]
                        st.rerun()
                else:
                    if st.button(
                        "Save Job", use_container_width=True, key=f"save_{job_id}_{i}"
                    ):
                        st.session_state.saved_jobs[job_id] = {
                            "job_title": title,
                            "employer_name": company,
                            "location": location,
                            "apply_link": apply_link,
                            "salary": salary,
                            "match_score": score,
                            "posted_date": posted_date,
                            "freshness": freshness_label,
                        }
                        st.rerun()

            with col2:
                # Export single
                job_data = pd.DataFrame(
                    [
                        {
                            "Title": title,
                            "Company": company,
                            "Location": location,
                            "Salary": salary,
                            "Match Score": f"{score}%",
                            "Type": job_type,
                            "Freshness": freshness_label,
                            "Link": apply_link,
                        }
                    ]
                )
                csv = job_data.to_csv(index=False)
                st.download_button(
                    label="Export",
                    data=csv,
                    file_name=f"{title.replace(' ', '_')[:20]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"export_{job_id}_{i}",
                )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ======================================================
# DISPLAY OPPORTUNITIES
# ======================================================
display_jobs(india_jobs, "India", "🇮🇳")
display_jobs(global_jobs, "Global", "🌍")
display_jobs(remote_jobs, "Remote", "🌐")

# ======================================================
# SAVED JOBS SECTION - MINIMAL
# ======================================================
if st.session_state.saved_jobs:
    st.markdown("## 💾 Saved Jobs")
    saved_count = len(st.session_state.saved_jobs)
    st.caption(f"{saved_count} job{'s' if saved_count != 1 else ''} saved for review")

    saved_jobs_list = list(st.session_state.saved_jobs.values())

    for idx, job in enumerate(saved_jobs_list):
        with st.expander(
            f"{idx + 1}. {job['job_title']} • {job['employer_name']}", expanded=False
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Company:** {job['employer_name']}")
                st.markdown(f"**Location:** {job['location']}")
                st.markdown(f"**Salary:** {job['salary']}")
                st.markdown(f"**Match:** {job['match_score']}%")
                st.markdown(f"**Freshness:** {job.get('freshness', 'N/A')}")

                if job.get("posted_date"):
                    st.markdown(f"**Posted:** {job['posted_date']}")

            with col2:
                if job["apply_link"] != "#":
                    st.link_button(
                        "Apply →", job["apply_link"], use_container_width=True
                    )

                # Find and remove
                job_id_to_remove = None
                for jid, jdata in st.session_state.saved_jobs.items():
                    if (
                        jdata["job_title"] == job["job_title"]
                        and jdata["employer_name"] == job["employer_name"]
                    ):
                        job_id_to_remove = jid
                        break

                if job_id_to_remove:
                    if st.button(
                        "Remove", key=f"remove_saved_{idx}", use_container_width=True
                    ):
                        del st.session_state.saved_jobs[job_id_to_remove]
                        st.rerun()

# ======================================================
# MINIMAL FOOTER
# ======================================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.caption(
    "RAY • Resume-based Application Yield • All opportunities are verified and current"
)
