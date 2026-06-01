'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, Loader2, AlertCircle, Download, Sliders, Briefcase, Globe } from 'lucide-react';
import { uploadResume, exportToCSV } from '../services/api';
import JobCard from './JobCard';

export default function UploadSection({ onJobsFetched, jobs }: { onJobsFetched: (jobs: any[]) => void; jobs: any[] }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matchThreshold, setMatchThreshold] = useState(0);
  const [skills, setSkills] = useState<string[]>([]);
  const [expLevel, setExpLevel] = useState('');
  const [activeRegion, setActiveRegion] = useState<'india' | 'global'>('india');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const response = await uploadResume(file);
      setSkills(response.skills);
      setExpLevel(response.experience_level);
      // Sort jobs by match score descending
      const sortedJobs = [...response.jobs].sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
      onJobsFetched(sortedJobs);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process resume. Please try again.');
    } finally {
      setUploading(false);
    }
  }, [onJobsFetched]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    maxFiles: 1,
    disabled: uploading
  });

  const indiaJobs = jobs.filter(job => job.country?.toLowerCase().includes('india'));
  const globalJobs = jobs.filter(job => !job.country?.toLowerCase().includes('india'));

  const filterByScore = (jobList: any[]) => jobList.filter(job => (job.match_score || 0) >= matchThreshold);
  const filteredIndia = filterByScore(indiaJobs);
  const filteredGlobal = filterByScore(globalJobs);
  const currentJobs = activeRegion === 'india' ? filteredIndia : filteredGlobal;

  const handleExport = async () => {
    const allFiltered = [...filteredIndia, ...filteredGlobal];
    if (allFiltered.length === 0) return;
    await exportToCSV(allFiltered);
  };

  return (
    <div className="space-y-8">
      <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-accent bg-accent/5' : 'border-border bg-surface/30'} ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}>
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          {uploading ? <Loader2 className="w-10 h-10 text-accent animate-spin" /> : <Upload className="w-10 h-10 text-muted" />}
          <div>
            <p className="text-textPrimary font-medium">{uploading ? 'Processing...' : isDragActive ? 'Drop resume here' : 'Drag & drop resume'}</p>
            <p className="text-textSecondary text-sm mt-1">PDF or DOCX</p>
          </div>
        </div>
      </div>

      {error && (
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="bg-highlight/10 border border-highlight rounded-xl p-4">
          <p className="text-highlight text-sm">{error}</p>
        </motion.div>
      )}

      {skills.length > 0 && (
        <div className="bento-card">
          <h3 className="font-medium text-textPrimary">Extracted Skills</h3>
          <div className="flex flex-wrap gap-2 mt-2">
            {skills.map(s => <span key={s} className="px-3 py-1 bg-accentSage/10 text-accentSage rounded-full text-sm">{s}</span>)}
          </div>
          <p className="text-textSecondary text-sm mt-3">Experience: <span className="font-medium text-textPrimary">{expLevel}</span></p>
        </div>
      )}

      {(indiaJobs.length > 0 || globalJobs.length > 0) && (
        <>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex gap-2">
              <button
                onClick={() => setActiveRegion('india')}
                className={`px-5 py-2 rounded-xl text-sm font-medium flex items-center gap-2 transition-all ${activeRegion === 'india' ? 'bg-accent text-white shadow-md' : 'bg-surfaceSecondary/50 text-textSecondary hover:bg-surfaceSecondary'}`}
              >
                <Briefcase className="w-4 h-4" /> India ({filteredIndia.length})
              </button>
              <button
                onClick={() => setActiveRegion('global')}
                className={`px-5 py-2 rounded-xl text-sm font-medium flex items-center gap-2 transition-all ${activeRegion === 'global' ? 'bg-accent text-white shadow-md' : 'bg-surfaceSecondary/50 text-textSecondary hover:bg-surfaceSecondary'}`}
              >
                <Globe className="w-4 h-4" /> Global ({filteredGlobal.length})
              </button>
            </div>
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex items-center gap-3">
                <Sliders className="w-4 h-4 text-muted" />
                <span className="text-sm text-textSecondary">Match ≥</span>
                <input type="range" min="0" max="100" value={matchThreshold} onChange={(e) => setMatchThreshold(Number(e.target.value))} className="w-32 accent-accent" />
                <span className="text-sm font-medium text-textPrimary w-8">{matchThreshold}%</span>
              </div>
              <button onClick={handleExport} className="btn-secondary flex items-center gap-2 text-sm"><Download className="w-4 h-4" /> Export CSV</button>
            </div>
          </div>

          {currentJobs.length === 0 ? (
            <p className="text-center text-textSecondary py-12">No jobs meet the threshold. Try lowering the slider.</p>
          ) : (
            <div className="space-y-4">
              {currentJobs.map((job, idx) => <JobCard key={`${activeRegion}-${idx}`} job={job} />)}
            </div>
          )}
        </>
      )}
    </div>
  );
}