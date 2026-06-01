# RAY – Resume-based Application Yield

A modern AI-powered job recommendation platform that analyzes resumes, extracts skills and experience, and matches users with relevant job opportunities using real-time job APIs.

## Live Demo

🔗 https://job-recommender-sigma.vercel.app/

---

## Overview

RAY helps users discover job opportunities that align with their skills, experience, and interests.

Users can:

- Upload a resume (PDF or DOCX)
- Automatically extract technical skills and experience level
- Search opportunities using job preferences or keywords
- Receive curated job recommendations from multiple sources
- Filter opportunities based on match score
- Export results for future reference

The platform combines AI-powered resume analysis with real-time job aggregation to provide personalized recommendations through a clean and intuitive interface.

---

## Features

### Resume Analysis

- PDF and DOCX resume upload
- AI-powered skill extraction using Mistral AI
- Experience level detection
- Technical skill identification
- Resume-based job matching

### Job Discovery

- Real-time job fetching from:
  - JSearch (RapidAPI)
  - Adzuna

- India and Global job opportunities

- Preference-based job search

- Multiple domain coverage

- Fresh job listings

### Smart Matching

- TF-IDF cosine similarity matching
- Resume-to-job relevance scoring
- Adjustable minimum match threshold
- Skill-based ranking
- Personalized recommendations

### Export & Management

- Export opportunities as CSV
- Save interesting jobs
- Filter and organize results
- Easy sharing and review

### User Experience

- Modern responsive interface
- Fast interactions
- Premium user experience
- Mobile-friendly design
- Smooth animations

---

## Tech Stack

### Backend

- FastAPI
- Python
- Mistral AI
- JSearch API
- Adzuna API
- Scikit-learn
- Pandas

### Frontend

- Next.js 14
- TypeScript
- Tailwind CSS
- Framer Motion
- Lucide React
- React Dropzone

---

## Project Structure

```text
ray/
├── backend/
│   ├── main.py
│   ├── utils.py
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   │
│   ├── components/
│   │   ├── UploadSection.tsx
│   │   ├── PreferenceSection.tsx
│   │   ├── JobCard.tsx
│   │   └── Footer.tsx
│   │
│   ├── services/
│   │   └── api.ts
│   │
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js
│   └── tsconfig.json
│
└── README.md
```

---

## Future Enhancements

- Advanced skill extraction
- Improved recommendation engine
- Job explainability
- Resume quality insights
- Enhanced filtering and ranking
- Additional job sources
- Better analytics and insights

---

## License

MIT License

---

## Author

**Sathvik**

RAY — Resume-based Application Yield
