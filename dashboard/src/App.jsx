import { useCallback, useEffect, useState } from "react";
import { getHealth, getStats, listAnalyses } from "./api.js";
import Analyzer from "./components/Analyzer.jsx";
import {
  DetectorTable,
  SeverityChart,
  StatTiles,
  SystemHealth,
} from "./components/Overview.jsx";
import { colorFor } from "./severity.js";

function RecentTable({ rows }) {
  if (!rows.length) {
    return <p className="muted">Nothing analyzed yet.</p>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Prompt</th>
          <th>Flagged by</th>
          <th className="num">Latency</th>
          <th className="num">Score</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.request_id}>
            <td className="muted" style={{ whiteSpace: "nowrap", fontSize: 12 }}>
              {new Date(row.created_at).toLocaleTimeString()}
            </td>
            <td className="prompt" title={row.prompt}>
              {row.prompt}
            </td>
            <td style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
              {row.flagged_detectors.join(", ") || (
                <span className="muted">—</span>
              )}
            </td>
            <td className="num muted">{row.latency_ms} ms</td>
            <td className="num">
              <span
                style={{ color: colorFor(row.severity), fontWeight: 600 }}
              >
                {row.risk_score}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      // Fetched together so a slow endpoint cannot leave the tiles and the
      // table disagreeing about how many analyses exist.
      const [h, s, a] = await Promise.all([
        getHealth(),
        getStats(),
        listAnalyses(25),
      ]);
      setHealth(h);
      setStats(s);
      setRecent(a.items);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => clearInterval(timer);
  }, [refresh]);

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>LLM Security Analyzer</h1>
          <div className="sub">
            Prompt injection, jailbreak, data leakage, hallucination and unsafe
            output — scored, evidenced, and logged.
          </div>
        </div>
        <button className="ghost" onClick={refresh}>
          Refresh
        </button>
      </header>

      {error && (
        <div className="error">
          Cannot reach the API: {error}. Is it running on port 8000?
        </div>
      )}

      <div className="panel">
        <h2>System</h2>
        <SystemHealth health={health} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <StatTiles stats={stats} />
      </div>

      <Analyzer onAnalyzed={refresh} />

      <div className="grid two">
        <div className="panel">
          <h2>Severity distribution</h2>
          <SeverityChart stats={stats} />
        </div>
        <div className="panel">
          <h2>Detector activity</h2>
          <DetectorTable stats={stats} />
        </div>
      </div>

      <div className="panel">
        <h2>Recent analyses</h2>
        <RecentTable rows={recent} />
      </div>
    </div>
  );
}
