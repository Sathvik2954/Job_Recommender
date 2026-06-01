export default function Footer() {
  return (
    <footer className="border-t border-border mt-16 py-6 text-center text-textSecondary text-sm">
      <p>© {new Date().getFullYear()} RAY. All rights reserved.</p>
    </footer>
  );
}