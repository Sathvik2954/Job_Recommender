'use client';

import { useState } from 'react';
import UploadSection from '@/components/UploadSection';
import PreferenceSection from '@/components/PreferenceSection';
import SavedJobsSection from '@/components/SavedJobsSection';
import Footer from '@/components/Footer';
import { motion } from 'framer-motion';

export default function Home() {
  const [resumeJobs, setResumeJobs] = useState<any[]>([]);
  const [prefJobs, setPrefJobs] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'resume' | 'preference'>('resume');

  return (
    <main className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-6 py-12 md:py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-16 text-center md:text-left"
        >
          <h1 className="text-5xl md:text-6xl font-heading font-bold tracking-tight text-textPrimary">
            RAY
            <span className="block text-2xl font-normal text-muted mt-2">Resume-based Application Yield</span>
          </h1>
          <p className="text-textSecondary text-lg mt-4 max-w-2xl mx-auto md:mx-0">
            Upload your resume or describe your dream job – get personalized, verified opportunities.
          </p>
        </motion.div>

        <div className="flex justify-center gap-3 mb-12">
          <button
            onClick={() => setActiveTab('resume')}
            className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
              activeTab === 'resume' ? 'bg-accent text-textPrimary shadow-md' : 'bg-surfaceSecondary/50 text-textSecondary hover:bg-surfaceSecondary'
            }`}
          >
             Upload Resume
          </button>
          <button
            onClick={() => setActiveTab('preference')}
            className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${
              activeTab === 'preference' ? 'bg-accent text-textPrimary shadow-md' : 'bg-surfaceSecondary/50 text-textSecondary hover:bg-surfaceSecondary'
            }`}
          >
             Preference Search
          </button>
        </div>

        <div className="mt-8">
          {activeTab === 'resume' && <UploadSection onJobsFetched={setResumeJobs} jobs={resumeJobs} />}
          {activeTab === 'preference' && <PreferenceSection onJobsFetched={setPrefJobs} jobs={prefJobs} />}
        </div>

        <SavedJobsSection />
      </div>
      <Footer />
    </main>
  );
}