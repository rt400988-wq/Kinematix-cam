"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, CameraOff, Loader2, ScanLine, Video, TriangleAlert } from "lucide-react"
import { usePoseLandmarker } from "@/lib/use-pose-landmarker"
import type { NormalizedLandmark } from "@mediapipe/tasks-vision"
import type { FormAlert } from "@/lib/exercise-analyzer"

type CameraFeedProps = {
  active: boolean
  mode: string
  reps: number
  /** Seconds into the current plank hold, or null for non-hold exercises. When non-null, the HUD shows a timer instead of a rep count. */
  holdSeconds: number | null
  onToggle: (active: boolean) => void
  /** Forwarded straight through to usePoseLandmarker — see that hook for details. */
  onLandmarks?: (landmarks: NormalizedLandmark[] | undefined, timestampMs: number) => void
  /** True while the pre-exercise "get into position" phase is running — see CalibrationController in lib/exercise-analyzer.ts. */
  calibrating: boolean
  calibrationMessage: string
  countdownValue: number | null
  /** Real-time form correction to flash over the feed (e.g. "Push your knees outward."). Distinct from the post-rep feedback list. */
  formAlert: FormAlert | null
}

export function CameraFeed({
  active,
  mode,
  reps,
  holdSeconds,
  onToggle,
  onLandmarks,
  calibrating,
  calibrationMessage,
  countdownValue,
  formAlert,
}: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [status, setStatus] = useState<"idle" | "loading" | "live" | "error">("idle")
  const [error, setError] = useState<string | null>(null)

  // Pose detection only starts once the camera stream is actually live —
  // starting it earlier would mean detectForVideo() runs against a video
  // element with no frames yet.
  const { status: poseStatus, error: poseError } = usePoseLandmarker({
    enabled: status === "live",
    videoRef,
    canvasRef,
    modelComplexity: "full",
    onLandmarks,
  })

  useEffect(() => {
    async function start() {
      setStatus("loading")
      setError(null)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => {})
        }
        setStatus("live")
      } catch (err) {
        console.log("[v0] camera error:", err)
        setError("Camera access was denied or is unavailable. Check your browser permissions.")
        setStatus("error")
        onToggle(false)
      }
    }

    function stop() {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
      setStatus("idle")
    }

    if (active) start()
    else stop()

    return () => stop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])

  return (
    <section className="relative flex h-full flex-col overflow-hidden rounded-2xl border border-border bg-card neon-border">
      {/* top status bar */}
      <div className="flex items-center justify-between gap-3 border-b border-border/60 bg-background/40 px-4 py-3 backdrop-blur">
        <div className="flex items-center gap-2 text-sm">
          <Video className="size-4 text-primary" />
          <span className="font-medium tracking-wide text-foreground">LIVE FEED</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`size-2 rounded-full ${
              status === "live" ? "bg-destructive animate-neon-pulse" : "bg-muted-foreground/50"
            }`}
            aria-hidden
          />
          <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            {status === "live" ? "Recording" : status === "loading" ? "Connecting" : "Standby"}
          </span>
        </div>
      </div>

      {/* video stage */}
      <div className="relative flex-1 grid-bg">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`absolute inset-0 size-full object-cover transition-opacity duration-500 ${
            status === "live" ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* Skeleton overlay — sized/cropped identically to the video above via
            the same object-cover treatment, so landmark points never drift
            from the body they're tracking. */}
        <canvas
          ref={canvasRef}
          className={`pointer-events-none absolute inset-0 size-full object-cover transition-opacity duration-500 ${
            status === "live" ? "opacity-100" : "opacity-0"
          }`}
        />

        {/* scanning overlay when live */}
        {status === "live" && (
          <>
            {!calibrating && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className="h-12 w-full bg-gradient-to-b from-primary/40 to-transparent animate-scan" />
              </div>
            )}
            {/* corner brackets */}
            <div className="pointer-events-none absolute inset-6">
              {[
                "left-0 top-0 border-l-2 border-t-2",
                "right-0 top-0 border-r-2 border-t-2",
                "left-0 bottom-0 border-l-2 border-b-2",
                "right-0 bottom-0 border-r-2 border-b-2",
              ].map((pos) => (
                <span key={pos} className={`absolute size-10 rounded-sm border-primary/70 ${pos}`} aria-hidden />
              ))}
            </div>

            {!calibrating && (
              <>
                {/* live HUD chips */}
                <div className="pointer-events-none absolute bottom-4 left-4 flex flex-wrap gap-2 font-mono text-xs">
                  <span className="rounded-md border border-primary/40 bg-background/70 px-2.5 py-1 text-primary text-glow backdrop-blur">
                    MODE: {mode.toUpperCase()}
                  </span>
                  {holdSeconds !== null ? (
                    <span className="rounded-md border border-success/40 bg-background/70 px-2.5 py-1 text-success backdrop-blur">
                      HOLD: {holdSeconds.toFixed(1)}s
                    </span>
                  ) : (
                    <span className="rounded-md border border-success/40 bg-background/70 px-2.5 py-1 text-success backdrop-blur">
                      {mode === "Shadowboxing" || mode === "Jab-Cross" ? "STRIKES" : "REPS"}: {reps}
                    </span>
                  )}
                  <span
                    className={`rounded-md border px-2.5 py-1 backdrop-blur ${
                      poseStatus === "ready"
                        ? "border-success/40 text-success"
                        : poseStatus === "error"
                          ? "border-destructive/40 text-destructive"
                          : "border-border text-muted-foreground"
                    } bg-background/70`}
                  >
                    <ScanLine className={`mr-1 inline size-3 ${poseStatus === "loading" ? "animate-pulse" : ""}`} />
                    {poseStatus === "ready"
                      ? "POSE LOCK"
                      : poseStatus === "error"
                        ? "TRACKER ERROR"
                        : "CALIBRATING…"}
                  </span>
                </div>

                {/* pose-model error banner — distinct from camera errors below,
                    since the webcam can be perfectly fine while only the
                    skeleton model fails to load (offline, blocked CDN, no WebGL). */}
                {poseStatus === "error" && poseError && (
                  <div className="absolute right-4 top-4 max-w-xs rounded-lg border border-destructive/40 bg-background/90 px-3 py-2 text-xs leading-relaxed text-destructive backdrop-blur">
                    {poseError}
                  </div>
                )}

                {/* Real-time form-correction flash — distinct from the
                    pose-error banner above (that's a tracking problem; this
                    is a movement-quality warning) and from the post-rep
                    feedback list in the sidebar (that's retrospective; this
                    interrupts mid-movement). */}
                {formAlert && (
                  <div
                    key={formAlert.text}
                    className="absolute inset-x-0 top-4 mx-auto flex w-fit max-w-[90%] items-center gap-2 rounded-xl border border-warning/50 bg-background/90 px-4 py-2.5 text-sm font-semibold text-warning shadow-lg shadow-warning/20 backdrop-blur animate-flash-in"
                  >
                    <TriangleAlert className="size-4 shrink-0" />
                    {formAlert.text}
                  </div>
                )}
              </>
            )}

            {/* Calibration overlay — pre-exercise "get into position" phase.
                Shown instead of the scoring HUD above, since nothing is
                being counted yet. See CalibrationController. */}
            {calibrating && (
              <div className="absolute inset-x-0 bottom-4 mx-auto flex w-fit max-w-[92%] flex-col items-center gap-2 rounded-xl border border-primary/40 bg-background/85 px-5 py-3 text-center backdrop-blur">
                {countdownValue !== null ? (
                  <span className="font-mono text-4xl font-bold text-primary text-glow">{countdownValue}</span>
                ) : (
                  <Loader2 className="size-5 animate-spin text-primary" />
                )}
                <span className="font-mono text-xs uppercase tracking-wide text-muted-foreground">
                  {calibrationMessage || "Calibrating…"}
                </span>
              </div>
            )}
          </>
        )}

        {/* idle / loading / error overlay */}
        {status !== "live" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 px-6 text-center">
            <div className="flex size-20 items-center justify-center rounded-2xl border border-border bg-card/60 neon-border">
              {status === "loading" ? (
                <Loader2 className="size-9 animate-spin text-primary" />
              ) : status === "error" ? (
                <CameraOff className="size-9 text-destructive" />
              ) : (
                <Camera className="size-9 text-primary" />
              )}
            </div>

            <div className="max-w-sm space-y-1.5">
              <h2 className="text-balance text-lg font-semibold text-foreground">
                {status === "loading"
                  ? "Initializing neural tracker…"
                  : status === "error"
                    ? "Camera unavailable"
                    : "Camera feed offline"}
              </h2>
              <p className="text-pretty text-sm leading-relaxed text-muted-foreground">
                {error ??
                  "Start your camera to begin real-time pose estimation, rep counting, and AI coaching."}
              </p>
            </div>

            <button
              type="button"
              onClick={() => onToggle(!active)}
              disabled={status === "loading"}
              className="group inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 font-semibold text-primary-foreground shadow-lg shadow-primary/30 transition hover:shadow-primary/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:opacity-60"
            >
              <Camera className="size-5 transition-transform group-hover:scale-110" />
              Start Camera
            </button>
          </div>
        )}
      </div>

      {/* bottom control bar (when live) */}
      {status === "live" && (
        <div className="flex items-center justify-between gap-3 border-t border-border/60 bg-background/40 px-4 py-3 backdrop-blur">
          <span className="font-mono text-xs text-muted-foreground">
            Tracking active · 30 FPS
          </span>
          <button
            type="button"
            onClick={() => onToggle(false)}
            className="inline-flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-2 text-sm font-medium text-destructive transition hover:bg-destructive/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-destructive"
          >
            <CameraOff className="size-4" />
            Stop Camera
          </button>
        </div>
      )}
    </section>
  )
}
