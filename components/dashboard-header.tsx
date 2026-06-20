import { Cpu, Radio } from "lucide-react"

export function DashboardHeader({ active }: { active: boolean }) {
  return (
    <header className="flex flex-col gap-3 border-b border-border/60 pb-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3">
        <div className="flex size-11 items-center justify-center rounded-xl border border-primary/40 bg-primary/10 neon-border">
          <Cpu className="size-6 text-primary text-glow" />
        </div>
        <div>
          <h1 className="text-balance text-xl font-bold leading-tight tracking-tight text-foreground sm:text-2xl">
            Kinematix <span className="text-primary text-glow">AI</span>
          </h1>
          <p className="text-xs text-muted-foreground">Neural Combat &amp; Fitness Tracker</p>
        </div>
      </div>

      <div className="flex items-center gap-2 self-start rounded-full border border-border bg-card/60 px-3.5 py-1.5 backdrop-blur sm:self-auto">
        <Radio
          className={`size-4 ${active ? "text-success animate-neon-pulse" : "text-muted-foreground"}`}
        />
        <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
          {active ? "System Online" : "Awaiting Input"}
        </span>
      </div>
    </header>
  )
}
