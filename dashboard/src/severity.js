// Shared severity helpers, so the colour used by the score ring, the badges and
// the chart all come from one place.

export const SEVERITIES = ["Low", "Medium", "High", "Critical"];

export const SEVERITY_COLORS = {
  Low: "#3fb950",
  Medium: "#d29922",
  High: "#f0883e",
  Critical: "#f85149",
};

export function colorFor(severity) {
  return SEVERITY_COLORS[severity] || "#8b93a7";
}

// Mirrors app/scoring.py: sub-scores are banded on the same boundaries the
// backend uses, so a detector row never disagrees with the composite badge.
export function bandForSubScore(subScore) {
  if (subScore <= 0.2) return "Low";
  if (subScore <= 0.5) return "Medium";
  if (subScore <= 0.8) return "High";
  return "Critical";
}
