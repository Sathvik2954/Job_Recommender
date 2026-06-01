'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Search, Loader2, AlertCircle, Download, Globe, MapPin } from 'lucide-react'
import { fetchPreferenceJobs, exportToCSV } from '@/services/api'
import JobCard from './JobCard'

type Region = 'india' | 'global' | 'both'

export default function PreferenceSection({ onJobsFetched, jobs }: any) {
  const [keywords, setKeywords] = useState('')
  const [region, setRegion] = useState<Region>('both')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSearch = async () => {
    if (!keywords.trim()) return
    setLoading(true)
    setError('')
    try {
      const kwArray = keywords.split(',').map(k => k.trim())
      const res = await fetchPreferenceJobs(kwArray, 0) // minScore ignored in backend, we'll filter region on frontend
      let filtered = res.jobs
      if (region === 'india') {
        filtered = filtered.filter((job: any) => job.country?.toLowerCase().includes('india'))
      } else if (region === 'global') {
        filtered = filtered.filter((job: any) => !job.country?.toLowerCase().includes('india'))
      }
      onJobsFetched(filtered)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleExport = async () => {
    if (jobs.length === 0) return
    await exportToCSV(jobs)
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row gap-4">
        <input
          type="text"
          value={keywords}
          onChange={(e) => setKeywords(e.target.value)}
          placeholder="e.g., backend developer, data scientist, remote"
          className="flex-1 input-warm"
        />
        <div className="flex gap-2">
          <button
            onClick={() => setRegion('india')}
            className={`px-4 py-2 rounded-xl text-sm flex items-center gap-2 transition-all ${
              region === 'india' ? 'bg-accent/10 text-accent border border-accent/30' : 'bg-surfaceSecondary/30 text-textSecondary'
            }`}
          >
            <MapPin className="w-4 h-4" /> India
          </button>
          <button
            onClick={() => setRegion('global')}
            className={`px-4 py-2 rounded-xl text-sm flex items-center gap-2 transition-all ${
              region === 'global' ? 'bg-accent/10 text-accent border border-accent/30' : 'bg-surfaceSecondary/30 text-textSecondary'
            }`}
          >
            <Globe className="w-4 h-4" /> Global
          </button>
          <button
            onClick={() => setRegion('both')}
            className={`px-4 py-2 rounded-xl text-sm flex items-center gap-2 transition-all ${
              region === 'both' ? 'bg-accent/10 text-accent border border-accent/30' : 'bg-surfaceSecondary/30 text-textSecondary'
            }`}
          >
            Both
          </button>
        </div>
        <button onClick={handleSearch} disabled={loading} className="btn-primary flex items-center gap-2 min-w-[120px]">
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
          {loading ? 'Searching...' : 'Find Jobs'}
        </button>
      </div>

      {error && (
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="bg-highlight/10 border border-highlight rounded-xl p-4">
          <p className="text-highlight text-sm">{error}</p>
        </motion.div>
      )}

      {jobs.length > 0 && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-textSecondary">{jobs.length} opportunities found</p>
            <button onClick={handleExport} className="btn-secondary flex items-center gap-2 text-sm">
              <Download className="w-4 h-4" /> Export CSV
            </button>
          </div>
          <div className="space-y-4">
            {jobs.map((job: any, idx: number) => (
              <JobCard key={idx} job={job} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}