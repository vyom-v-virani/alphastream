"use client";

import { useState } from "react";
import type React from "react";

interface Signal {
  id: number;
  ticker: string;
  source: string;
  score: number;
  direction: string;
  timestamp: string;
}

const DIRECTION_STYLES: Record<string, string> = {
  bullish: "bg-green-100 text-green-700",
  bearish: "bg-red-100 text-red-700",
  neutral: "bg-gray-100 text-gray-600",
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [signal, setSignal] = useState<Signal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSearch(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    const ticker = query.trim().toUpperCase();
    if (!ticker) return;

    setLoading(true);
    setError(null);
    setSignal(null);

    try {
      const res = await fetch(`http://localhost:8000/signals/${ticker}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.detail ?? `Error ${res.status}`);
        return;
      }
      setSignal(await res.json());
    } catch {
      setError("Could not reach the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  const directionClass =
    signal ? (DIRECTION_STYLES[signal.direction] ?? DIRECTION_STYLES.neutral) : "";

  return (
    <main className="flex flex-col items-center px-4 py-20">
      <h1 className="text-3xl font-semibold tracking-tight mb-2">AlphaStream</h1>
      <p className="text-gray-500 mb-10 text-sm">Alternative data signals for equities</p>

      <form onSubmit={handleSearch} className="flex gap-2 w-full max-w-md mb-8">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter ticker (e.g. AAPL)"
          className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}

      {signal && (
        <div className="w-full max-w-md rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <span className="text-2xl font-bold">{signal.ticker}</span>
            <span className={`rounded-full px-3 py-1 text-xs font-medium capitalize ${directionClass}`}>
              {signal.direction}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-y-3 text-sm">
            <span className="text-gray-500">Score</span>
            <span className="font-medium">{signal.score.toFixed(4)}</span>

            <span className="text-gray-500">Source</span>
            <span className="font-medium capitalize">{signal.source.replace(/_/g, " ")}</span>

            <span className="text-gray-500">Timestamp</span>
            <span className="font-medium">
              {new Date(signal.timestamp).toLocaleString()}
            </span>

            <span className="text-gray-500">Signal ID</span>
            <span className="font-medium text-gray-400">#{signal.id}</span>
          </div>
        </div>
      )}
    </main>
  );
}
