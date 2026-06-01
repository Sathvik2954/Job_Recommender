/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#F4F0F8',   // new – soft lavender-gray
        surface: '#E9E2D7',
        surfaceSecondary: '#D8D0C4',
        accent: '#7C5CFC',
        accentSage: '#6F8D7A',
        highlight: '#D88F5D',
        success: '#6C9B7B',
        border: '#CFC6BA',
        textPrimary: '#2E2A27',
        textSecondary: '#6A625B',
        muted: '#958A80',
        darkBg: '#26211D',
        darkSurface: '#342E2A',
        darkAccent: '#8E78FF',
        darkText: '#EEE8E1',
        darkMuted: '#A89F96',
      },
      fontFamily: {
        heading: ['General Sans', 'Satoshi', 'Inter', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        'xl': '20px',
        '2xl': '22px',
      },
      boxShadow: {
        'card': '0 2px 4px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.03)',
        'hover': '0 8px 20px rgba(0,0,0,0.05)',
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
      },
    },
  },
  plugins: [],
}