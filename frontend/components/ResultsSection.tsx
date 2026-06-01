const renderJobs = (jobList: any[]) =>
  [...jobList]
    .sort((a, b) => (b.match_score || 0) - (a.match_score || 0))
    .map((job: any) => {
      const jobId = `${job.title}-${job.company}`;
      const isSaved = !!savedJobs[jobId];
      return (
        <JobCard
          key={jobId}
          job={job}
          onSave={() => saveJob(job)}
          onRemove={() => removeJob(jobId)}
          isSaved={isSaved}
          onExportSingle={(j: any) => {
            const singleJobDict = { india: [j], global_jobs: [], remote: [] };
            exportJobs(singleJobDict, 'csv');
          }}
        />
      );
    });