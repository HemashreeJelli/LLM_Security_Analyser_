import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SEVERITIES, colorFor } from "../severity.js";

function Tile({ label, value }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

export function StatTiles({ stats }) {
  if (!stats) return null;

  const critical =
    (stats.by_severity.Critical || 0) + (stats.by_severity.High || 0);

  return (
    <div className="tiles">
      <Tile label="Analyses" value={stats.total_analyses} />
      <Tile label="High + Critical" value={critical} />
      <Tile label="Avg latency" value={`${stats.avg_latency_ms} ms`} />
      <Tile label="Detectors reporting" value={stats.by_detector.length} />
    </div>
  );
}

export function SeverityChart({ stats }) {
  if (!stats || !stats.total_analyses) {
    return <p className="muted">No analyses recorded yet.</p>;
  }

  const data = SEVERITIES.map((name) => ({
    name,
    count: stats.by_severity[name] || 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={190}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
        <XAxis
          dataKey="name"
          tick={{ fill: "#8b93a7", fontSize: 12 }}
          axisLine={{ stroke: "#2a2f3d" }}
          tickLine={false}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fill: "#8b93a7", fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "#1e222d" }}
          contentStyle={{
            background: "#171a23",
            border: "1px solid #2a2f3d",
            borderRadius: 6,
            fontSize: 12,
          }}
        />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {data.map((entry) => (
            <Cell key={entry.name} fill={colorFor(entry.name)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DetectorTable({ stats }) {
  if (!stats || !stats.by_detector.length) {
    return <p className="muted">No detector results recorded yet.</p>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Detector</th>
          <th className="num">Runs</th>
          <th className="num">Flagged</th>
          <th className="num">Flag rate</th>
          <th className="num">Avg latency</th>
        </tr>
      </thead>
      <tbody>
        {stats.by_detector.map((d) => (
          <tr key={d.detector_name}>
            <td style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
              {d.detector_name}
            </td>
            <td className="num">{d.runs}</td>
            <td className="num">{d.flagged}</td>
            <td className="num">{(d.flag_rate * 100).toFixed(1)}%</td>
            <td className="num">{d.avg_latency_ms} ms</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SystemHealth({ health }) {
  if (!health) return null;

  const unavailable = Object.entries(health.detectors_unavailable);

  return (
    <div>
      <div className="pill-row" style={{ marginBottom: 10 }}>
        <span className={`pill ${health.database === "up" ? "ok" : "off"}`}>
          db: {health.database}
        </span>
        <span className={`pill ${health.auth_enabled ? "ok" : "off"}`}>
          auth: {health.auth_enabled ? "on" : "off"}
        </span>
        <span className="pill">env: {health.environment}</span>
      </div>

      <div className="pill-row">
        {health.detectors_loaded.map((name) => (
          <span key={name} className="pill ok">
            {name}
          </span>
        ))}
        {unavailable.map(([name, reason]) => (
          <span key={name} className="pill off" title={reason}>
            {name} — off
          </span>
        ))}
      </div>

      {unavailable.length > 0 && (
        <p className="warn">
          {unavailable.length} of {unavailable.length + health.detectors_loaded.length}{" "}
          detectors are not running in this deployment. Hover a pill for the
          reason. Scores below are computed from the detectors that did run.
        </p>
      )}
    </div>
  );
}
