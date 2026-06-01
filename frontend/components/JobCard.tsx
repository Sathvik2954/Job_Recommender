'use client';

import { motion } from 'framer-motion';
import { Briefcase, MapPin, Calendar, DollarSign, ExternalLink, Building2, Bookmark, BookmarkCheck } from 'lucide-react';

export default function JobCard({ job, onSave, onRemove, isSaved }: any) {
  const postedDate = job.posted_date ? new Date(job.posted_date).toLocaleDateString() : 'Date unknown';

  const getSalaryDisplay = () => {
    if (!job.salary || job.salary === 'Not specified') return 'Salary not specified';
    const isIndia = job.country?.toLowerCase().includes('india');
    let raw = String(job.salary);
    if (isIndia) {
      let cleaned = raw.replace(/[$]/g, '').replace(/USD/gi, '').trim();
      if (!cleaned.startsWith('₹')) return `₹${cleaned}`;
      return cleaned;
    } else {
      let cleaned = raw.replace(/[₹]/g, '').replace(/INR/gi, '').trim();
      if (!cleaned.startsWith('$')) return `$${cleaned}`;
      return cleaned;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="bento-card group">
      <div className="flex flex-col md:flex-row justify-between gap-4">
        <div className="space-y-2 flex-1">
          <h3 className="text-xl font-heading font-semibold text-textPrimary">{job.title}</h3>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-textSecondary">
            <span className="flex items-center gap-1"><Building2 className="w-4 h-4" /> {job.company}</span>
            <span className="flex items-center gap-1"><MapPin className="w-4 h-4" /> {job.city}, {job.country}</span>
            <span className="flex items-center gap-1"><DollarSign className="w-4 h-4" /> {getSalaryDisplay()}</span>
            <span className="flex items-center gap-1"><Calendar className="w-4 h-4" /> {postedDate}</span>
          </div>
          <p className="text-textSecondary text-sm line-clamp-2">{job.description || 'No description available.'}</p>
          {job.match_score !== undefined && (
            <div className="mt-2">
              <span className="text-xs font-medium text-accent">Match {Math.round(job.match_score)}%</span>
              <div className="w-full bg-surfaceSecondary rounded-full h-1.5 mt-1">
                <div className="bg-accent h-1.5 rounded-full" style={{ width: `${job.match_score}%` }} />
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-col sm:flex-row items-center gap-2">
          <a href={job.apply_link} target="_blank" rel="noopener noreferrer" className="btn-primary flex items-center gap-2 text-sm whitespace-nowrap">
            Apply <ExternalLink className="w-4 h-4" />
          </a>
          {isSaved ? (
            <button onClick={() => onRemove(job)} className="btn-secondary flex items-center gap-2 text-sm">
              <BookmarkCheck className="w-4 h-4" /> Saved
            </button>
          ) : (
            <button onClick={() => onSave(job)} className="btn-ghost flex items-center gap-2 text-sm">
              <Bookmark className="w-4 h-4" /> Save
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}