'use client';

import { useState, useEffect } from 'react';
import JobCard from './JobCard';
import { useSavedJobs } from '@/utils/session';

export default function SavedJobsSection() {
  const { savedJobs, removeJob } = useSavedJobs();
  const [jobsList, setJobsList] = useState<any[]>([]);

  useEffect(() => {
    setJobsList(Object.values(savedJobs));
  }, [savedJobs]);

  if (jobsList.length === 0) return null;

  return (
    <div className="mt-16">
      <h2 className="text-2xl font-heading font-semibold mb-6">Saved Jobs</h2>
      <div className="space-y-4">
        {jobsList.map((job, idx) => (
          <JobCard
            key={`saved-${idx}`}
            job={job}
            isSaved={true}
            onRemove={() => removeJob(`${job.title}-${job.company}`)}
          />
        ))}
      </div>
    </div>
  );
}