"use client"

import { AlertTriangle, CheckCircle2, Sparkles } from "lucide-react"

export type Feedback = {
  id: number
  text: string
  tone: "good" | "warn" | "info"
}

function ToneIcon({ tone }: { tone: Feedback["tone"] }) {
  if (tone === "good") return <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-success" />
  if (tone === "warn") return <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" />
  return <Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />
}

export function AiFeedback({ active, items }: { active: boolean; items: Feedback[] }) {
  return (
    <section className="flex min-h-0 flex-1 flex-col gap-3 rounded-2xl border border-border bg-sidebar/80 p-5 neon-border backdrop-blur">
      <header className="flex items-center gap-2">
        <span className="flex size-7 items-center justify-center rounded-lg bg-primary/15">
          <Sparkles className="size-4 text-primary text-glow" />
        </span>
        <h2 className="text-sm font-semibold uppercase tracking-widest text-foreground">
          AI Coach Feedback
        </h2>
      </header>

      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {!active && items.length === 0 ? (
          <p className="py-6 text-center text-sm leading-relaxed text-muted-foreground">
            Start the camera to receive real-time coaching cues on your form, guard, and depth.
          </p>
        ) : (
          items.map((f) => (
            <div
              key={f.id}
              className={`flex items-start gap-2.5 rounded-xl border bg-card/60 p-3 text-sm leading-relaxed ${
                f.tone === "good"
                  ? "border-success/30"
                  : f.tone === "warn"
                    ? "border-warning/30"
                    : "border-primary/30"
              }`}
            >
              <ToneIcon tone={f.tone} />
              <span className="text-foreground/90">{f.text}</span>
            </div>
          ))
        )}
      </div>
    </section>
  )
}
