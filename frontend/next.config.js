/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,   // bypass TypeScript errors during build
  },
  eslint: {
    ignoreDuringBuilds: true,  // also ignore ESLint errors
  },
}

module.exports = nextConfig