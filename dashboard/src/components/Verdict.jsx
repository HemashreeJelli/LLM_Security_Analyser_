import { bandForSubScore, colorFor } from "../severity.js";

function DetectorRow({ detector }) {
  const band = bandForSubScore(detector.sub_score);
  const color = detector.error ? "#8b93a7" : colorFor(band);

  return (
    <details className="detector">
      <summary>
        <span className="bar">
          <i
            style={{
              width: `${Math.round(detector.sub_score * 100)}%`,
              background: color,
            }}
          />
        </span>
        <span className="name">{detector.detector_name}</span>
        {detector.error ? (
          <span className="nums" style={{ color: "var(--medium)" }}>
            unavailable
          </span>
        ) : (
          <span className="nums">
            s={detector.sub_score.toFixed(2)} &middot; c=
            {detector.confidence.toFixed(2)} &middot; {detector.latency_ms}ms
          </span>
        )}
        <span className={`badge sev-${detector.error ? "Low" : band}`}>
          {detector.error ? "n/a" : detector.is_flagged ? "flagged" : "clean"}
        </span>
      </summary>
      <pre>
        {detector.error
          ? detector.error
          : JSON.stringify(detector.evidence, null, 2)}
      </pre>
    </details>
  );
}

export default function Verdict({ result }) {
  const color = colorFor(result.severity);

  return (
    <div>
      <div className="verdict">
        <div
          className="score-ring"
          style={{
            border: `4px solid ${color}`,
            color,
          }}
        >
          {result.risk_score}
        </div>
        <div>
          <div className={`badge sev-${result.severity}`}>{result.severity}</div>
          <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
            {result.latency_ms} ms end to end &middot; {result.detectors.length}{" "}
            detector{result.detectors.length === 1 ? "" : "s"} ran
          </div>
          <div
            className="muted"
            style={{ marginTop: 2, fontSize: 11, fontFamily: "var(--mono)" }}
          >
            {result.request_id}
          </div>
        </div>
      </div>

      <div className="recommendation">{result.recommendation}</div>

      <h2 style={{ marginTop: 18 }}>Per-detector evidence</h2>
      {result.detectors.map((d) => (
        <DetectorRow key={d.detector_name} detector={d} />
      ))}

      {result.degraded.length > 0 && (
        <p className="warn">
          Not contributing to this score: {result.degraded.join(", ")}. The
          verdict is a partial assessment.
        </p>
      )}
    </div>
  );
}
