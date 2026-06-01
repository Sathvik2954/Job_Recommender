'use client';

export default function Slider({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return (
    <div className="flex items-center gap-4">
      <span className="text-sm text-textSecondary">Match ≥</span>
      <input
        type="range"
        min="0"
        max="100"
        step="1"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-48 accent-accent"
      />
      <span className="text-sm font-medium text-textPrimary w-8">{value}%</span>
    </div>
  );
}