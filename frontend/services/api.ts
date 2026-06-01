import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export const uploadResume = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await axios.post(`${API_BASE}/upload-resume`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export const fetchPreferenceJobs = async (keywords: string[], minScore: number) => {
  const res = await axios.post(`${API_BASE}/preference-jobs`, {
    keywords,
    min_score: minScore,
  })
  return res.data
}

export const exportToCSV = async (jobs: any[]) => {
  const res = await axios.post(`${API_BASE}/export-csv`, jobs, {
    responseType: 'blob',
  })
  const url = window.URL.createObjectURL(new Blob([res.data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', 'opportunities.csv')
  document.body.appendChild(link)
  link.click()
  link.remove()
}