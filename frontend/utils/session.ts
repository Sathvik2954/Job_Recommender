import { useEffect, useState } from 'react';

export const useSavedJobs = () => {
  const [savedJobs, setSavedJobs] = useState<Record<string, any>>({});

  useEffect(() => {
    const stored = localStorage.getItem('ray_saved_jobs');
    if (stored) setSavedJobs(JSON.parse(stored));
  }, []);

  const saveJob = (job: any) => {
    const id = `${job.title}-${job.company}`;
    const updated = { ...savedJobs, [id]: job };
    setSavedJobs(updated);
    localStorage.setItem('ray_saved_jobs', JSON.stringify(updated));
  };

  const removeJob = (id: string) => {
    const updated = { ...savedJobs };
    delete updated[id];
    setSavedJobs(updated);
    localStorage.setItem('ray_saved_jobs', JSON.stringify(updated));
  };

  return { savedJobs, saveJob, removeJob };
};