"use client"

import { Activity, Crosshair, Repeat, Zap, Timer } from "lucide-react"
import type { LucideIcon } from "lucide-react"

type MetricsSidebarProps = {
  active: boolean
  reps: number
  /** "reps" for squats/lunges, "strikes" for punching modes — see app/page.tsx for why these are labeled differently. */
  repLabel: string
  accuracy: number
  /** True once the pose model has actually detected a body this session. */
  tracking: boolean
  mode: string
  modes: readonly string[]
  onModeChange: (mode: string) => void
  /** True for Plank — a held position has no rep count or form score, so it gets its own card instead of the Reps/Accuracy ones. */
  isHoldMode: boolean
  holdSeconds: number
  bestHoldSeconds: number
}

function MetricCard({
  icon: Icon,
  label,
  children,
  accent = "primary",
}: {
  icon: LucideIcon
  label: string
  children: React.ReactNode
  accent?: "primary" | "success" | "warning"
}) {
  const accentColor =
    accent === "success" ? "text-success" : accent === "warning" ? "text-warning" : "text-primary"
  return (
    <div className="rounded-xl border border-border bg-card/60 p-4 backdrop-blur transition hover:bg-card">
      <div className="mb-2 flex items-center gap-2">
        <Icon className={`size-4 ${accentColor}`} />
        <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
      </div>
      {children}
    </div>
  )
}

export function MetricsSidebar({
  active,
  reps,
  repLabel,
  accuracy,
  tracking,
  mode,
  modes,
  onModeChange,
  isHoldMode,
  holdSeconds,
  bestHoldSeconds,
}: MetricsSidebarProps) {
  return (
    <aside className="flex h-full flex-col gap-4 rounded-2xl border border-border bg-sidebar/80 p-5 neon-border backdrop-blur">
      <header className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-widest text-foreground">
          <Activity className="size-4 text-primary text-glow" />
          Live Metrics
        </h2>
        <span
          className={`size-2 rounded-full ${active ? "bg-success animate-neon-pulse" : "bg-muted-foreground/40"}`}
          aria-label={active ? "Live" : "Idle"}
        />
      </header>

      {isHoldMode ? (
        <MetricCard icon={Timer} label="Plank Hold" accent="success">
          <div className="flex items-end gap-2">
            <span className="font-mono text-5xl font-bold leading-none text-success text-glow">
              {holdSeconds.toFixed(1)}
            </span>
            <span className="pb-1 text-xs text-muted-foreground">seconds</span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {active
              ? tracking
                ? bestHoldSeconds > 0
                  ? `Best this session: ${bestHoldSeconds.toFixed(1)}s`
                  : "Hold a straight line from shoulders to ankles."
                : "Step into frame so the camera can see your full body."
              : "Start the camera to begin your hold."}
          </p>
        </MetricCard>
      ) : (
        <>
          {/* Reps */}
          <MetricCard icon={Repeat} label={repLabel === "strikes" ? "Strikes Detected" : "Reps Counted"} accent="success">
            <div className="flex items-end gap-2">
              <span className="font-mono text-5xl font-bold leading-none text-success text-glow">
                {String(reps).padStart(2, "0")}
              </span>
              <span className="pb-1 text-xs text-muted-foreground">total {repLabel}</span>
            </div>
            {active && (
              <p className="mt-2 text-xs text-muted-foreground">
                {tracking ? "Tracking your movement…" : "Step into frame so the camera can see your full body."}
              </p>
            )}
          </MetricCard>

          {/* Stance accuracy */}
          <MetricCard icon={Crosshair} label="Stance Accuracy" accent="primary">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="font-mono text-3xl font-bold text-primary text-glow">{accuracy}%</span>
              <span className="text-xs text-muted-foreground">
                {accuracy >= 85 ? "Excellent" : accuracy >= 70 ? "Good" : "Adjust form"}
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary/70 to-primary transition-all duration-700"
                style={{ width: `${accuracy}%` }}
              />
            </div>
          </MetricCard>
        </>
      )}

      {/* Movement mode */}
      <MetricCard icon={Zap} label="Current Movement Mode" accent="warning">
        <div className="flex flex-wrap gap-2">
          {modes.map((m) => {
            const isActive = m === mode
            return (
              <button
                key={m}
                type="button"
                onClick={() => onModeChange(m)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                  isActive
                    ? "border-warning/60 bg-warning/15 text-warning"
                    : "border-border bg-secondary/50 text-muted-foreground hover:text-foreground"
                }`}
              >
                {m}
              </button>
            )
          })}
        </div>
      </MetricCard>
    </aside>
  )
}
