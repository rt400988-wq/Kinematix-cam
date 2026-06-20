"use client"

import { useEffect, useRef, useState } from "react"
import type { PoseLandmarker as PoseLandmarkerType, NormalizedLandmark } from "@mediapipe/tasks-vision"

// MediaPipe's 33-point pose model. Each pair is a bone we draw between two
// landmark indices. Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const POSE_CONNECTIONS: [number, number][] = [
  // face
  [0, 1], [0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
]

// Landmarks 0-10 are the face mesh points (nose, eyes, ears, mouth corners).
// We skip drawing dots for these so the overlay reads as a body skeleton,
// not a cluttered face cage.
const FACE_LANDMARK_COUNT = 11

export type PoseLandmarkerStatus = "idle" | "loading" | "ready" | "error"

type UsePoseLandmarkerOptions = {
  /** Only runs detection while true. Pass the same flag that drives the camera. */
  enabled: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  modelComplexity?: "lite" | "full" | "heavy"
  /**
   * Fired on every successful detection frame (~30x/sec) with the raw
   * landmarks for the most prominent detected pose, or undefined if no pose
   * was found in that frame. Intentionally NOT a dependency of the effect
   * below — callers should do their own setState batching/threshold logic
   * inside this callback rather than re-rendering on every call.
   */
  onLandmarks?: (landmarks: NormalizedLandmark[] | undefined, timestampMs: number) => void
}

const MODEL_URLS: Record<"lite" | "full" | "heavy", string> = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"

/**
 * Owns a PoseLandmarker instance and a requestAnimationFrame loop that:
 *   1. Reads the current video frame
 *   2. Runs detectForVideo()
 *   3. Draws the resulting skeleton onto the overlay canvas
 *
 * Drawing happens directly on canvas (not via React state) so per-frame
 * updates don't trigger React re-renders ~30x/sec.
 */
export function usePoseLandmarker({
  enabled,
  videoRef,
  canvasRef,
  modelComplexity = "full",
  onLandmarks,
}: UsePoseLandmarkerOptions) {
  const [status, setStatus] = useState<PoseLandmarkerStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const landmarkerRef = useRef<PoseLandmarkerType | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef(-1)
  const cancelledRef = useRef(false)
  const onLandmarksRef = useRef(onLandmarks)
  onLandmarksRef.current = onLandmarks

  useEffect(() => {
    if (!enabled) {
      setStatus("idle")
      return
    }

    cancelledRef.current = false
    setStatus("loading")
    setError(null)

    async function init() {
      try {
        const { FilesetResolver, PoseLandmarker } = await import("@mediapipe/tasks-vision")
        const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE)

        if (cancelledRef.current) return

        let landmarker: PoseLandmarkerType
        try {
          landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: MODEL_URLS[modelComplexity],
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numPoses: 1,
          })
        } catch (gpuErr) {
          // Some devices/browsers fail to compile the GPU shader pipeline
          // (older integrated graphics, certain Linux configs). Retry once
          // on CPU rather than surfacing an error the person can't act on.
          console.warn("[pose-landmarker] GPU delegate failed, retrying on CPU:", gpuErr)
          landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: MODEL_URLS[modelComplexity],
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numPoses: 1,
          })
        }

        if (cancelledRef.current) {
          landmarker.close()
          return
        }

        landmarkerRef.current = landmarker
        setStatus("ready")
        rafRef.current = requestAnimationFrame(detectFrame)
      } catch (err) {
        console.error("[pose-landmarker] init failed:", err)
        if (!cancelledRef.current) {
          setError(
            "Couldn't load the pose detection model. Check your connection and that your browser supports WebGL.",
          )
          setStatus("error")
        }
      }
    }

    function detectFrame() {
      const video = videoRef.current
      const canvas = canvasRef.current
      const landmarker = landmarkerRef.current

      if (!video || !canvas || !landmarker || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(detectFrame)
        return
      }

      // detectForVideo requires a strictly increasing timestamp per call.
      // Skip frames the video hasn't actually advanced past (e.g. while paused).
      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime
        const startTimeMs = performance.now()
        // For runningMode: "VIDEO", detectForVideo is synchronous and returns
        // the result directly — unlike LIVE_STREAM mode, which instead takes
        // a result_callback. No callback argument here.
        const result = landmarker.detectForVideo(video, startTimeMs)
        const firstPose = result.landmarks?.[0]
        drawSkeleton(canvas, video, firstPose)
        onLandmarksRef.current?.(firstPose, startTimeMs)
      }

      rafRef.current = requestAnimationFrame(detectFrame)
    }

    init()

    return () => {
      cancelledRef.current = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      landmarkerRef.current?.close()
      landmarkerRef.current = null
      lastVideoTimeRef.current = -1

      const canvas = canvasRef.current
      const ctx = canvas?.getContext("2d")
      if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, modelComplexity])

  return { status, error }
}

function drawSkeleton(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  landmarks: NormalizedLandmark[] | undefined,
) {
  // Keep the canvas's pixel buffer in lockstep with the video's native
  // resolution (not its CSS size) so points line up exactly with the feed.
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
  }

  const ctx = canvas.getContext("2d")
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!landmarks || landmarks.length === 0) return

  // Video and canvas share the same object-cover box sizing and neither is
  // mirrored, so MediaPipe's normalized [0,1] coordinates map directly onto
  // canvas pixels with no extra transform needed.
  const w = canvas.width
  const h = canvas.height

  const rootStyles = getComputedStyle(document.documentElement)
  const colorPrimary = rootStyles.getPropertyValue("--primary").trim() || "oklch(0.8 0.16 195)"
  const colorSuccess = rootStyles.getPropertyValue("--success").trim() || "oklch(0.78 0.2 150)"

  ctx.lineWidth = Math.max(2, w * 0.004)
  ctx.lineCap = "round"
  ctx.strokeStyle = colorPrimary
  ctx.shadowColor = colorPrimary
  ctx.shadowBlur = 8

  for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
    const a = landmarks[startIdx]
    const b = landmarks[endIdx]
    if (!a || !b) continue
    // visibility close to 0 means MediaPipe couldn't confidently locate the
    // point (e.g. occluded limb) — skip drawing that bone to avoid noisy snapping.
    if ((a.visibility ?? 1) < 0.4 || (b.visibility ?? 1) < 0.4) continue

    ctx.beginPath()
    ctx.moveTo(a.x * w, a.y * h)
    ctx.lineTo(b.x * w, b.y * h)
    ctx.stroke()
  }

  ctx.shadowBlur = 0
  const dotRadius = Math.max(2.5, w * 0.005)
  for (let i = FACE_LANDMARK_COUNT; i < landmarks.length; i++) {
    const point = landmarks[i]
    if ((point.visibility ?? 1) < 0.4) continue

    ctx.beginPath()
    ctx.fillStyle = colorSuccess
    ctx.arc(point.x * w, point.y * h, dotRadius, 0, Math.PI * 2)
    ctx.fill()
  }
}
