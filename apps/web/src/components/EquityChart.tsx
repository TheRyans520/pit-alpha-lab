import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { decimal } from "../utils";

interface ChartPoint {
  date: string;
  strategy: number;
  benchmark: number | undefined;
}

function ChartTooltip({ active, payload, label }: Record<string, unknown>) {
  const entries = payload as Array<{ name: string; value: number; color: string }> | undefined;
  if (!active || !entries?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-date">{String(label)}</div>
      {entries.map((entry) => (
        <div className="tooltip-row" key={entry.name}>
          <span className="tooltip-swatch" style={{ background: entry.color }} />
          <span>{entry.name}</span>
          <strong>{decimal(entry.value, 3)}×</strong>
        </div>
      ))}
    </div>
  );
}

export default function EquityChart({
  data,
  color,
  label,
}: {
  data: ChartPoint[];
  color: string;
  label: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 16, right: 10, left: -14, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="#dcd7cd" strokeDasharray="3 5" />
        <XAxis
          dataKey="date"
          tickFormatter={(value: string) => value.slice(0, 4)}
          minTickGap={58}
          axisLine={false}
          tickLine={false}
          tick={{ fill: "#77756f", fontSize: 11 }}
        />
        <YAxis
          domain={["auto", "auto"]}
          tickFormatter={(value: number) => `${value.toFixed(1)}×`}
          axisLine={false}
          tickLine={false}
          tick={{ fill: "#77756f", fontSize: 11 }}
        />
        <Tooltip content={<ChartTooltip />} />
        <ReferenceLine y={1} stroke="#9c978d" strokeDasharray="3 4" />
        <Line
          type="monotone"
          dataKey="benchmark"
          name="Equal weight"
          dot={false}
          stroke="#a8a49c"
          strokeWidth={1.5}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="strategy"
          name={label}
          dot={false}
          stroke={color}
          strokeWidth={2.25}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
