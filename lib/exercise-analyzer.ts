// Pure exercise-analysis logic, decoupled from React so it's easy to
// unit-test and easy to retune thresholds later. Takes MediaPipe's own
// NormalizedLandmark type directly at the public boundary (rather than a
// parallel custom shape) so there's no structural-compatibility guesswork
// at the call site in app/page.tsx.
//
// PHASE 2 — what changed and why, read this before touching thresholds:
//
//   1. Multi-joint form correction (squats, shadowboxing guard) is layered
//      ON TOP of the existing single-angle rep counters, not instead of
//      them. Rep counting still happens off one clean angle/velocity
//      signal; the new checks only ever produce supplementary `formAlert`
//      warnings — they can't block a rep from counting. This matters
//      because the published accuracy of MediaPipe-derived knee-valgus
//      angle has a measured error band of roughly ±19° and only "modest"
//      inter-rater reliability (Heliyon, 2024, knee valgus at drop
//      landing). A multi-joint heuristic this noisy should never be load-
//      bearing for whether a rep counts — only for an advisory cue.
//
//   2. Spinal-lean / back-sag checks compare against THIS PERSON'S OWN
//      calibration-phase baseline, not a fixed "ideal" angle. An earlier
//      version of this file scored squat torso lean against a global
//      constant and that was wrong — proper forward lean varies
//      legitimately by femur length, stance, and style. Comparing against
//      a personal baseline avoids repeating that mistake for the new
//      back-sag checks (push-ups, plank).
//
//   3. Push-up depth (shoulder-elbow-wrist angle) and plank/push-up back
//      alignment (shoulder-hip-ankle line) are textbook patterns, same
//      family as the hip-knee-ankle squat angle. The straight-line check
//      is most reliable filmed from the side — frontal framing can make a
//      perfectly straight back look like it's sagging or piking, since the
//      camera is foreshortening the line being measured. There's no way to
//      detect camera angle from 2D landmarks alone with any real
//      confidence, so this is surfaced as a comment/constant, not silently
//      papered over.
//
//   4. Plank has no rep cycle — it's a held static position. Forcing it
//      into the repCompleted/formScore shape built for cyclic exercises
//      would be dishonest about what's actually being measured, so it gets
//      its own fields on FrameResult (holdSeconds, bestHoldSeconds,
//      holdBroken) instead.
//
//   5. Shadowboxing/Jab-Cross strike counting is still a velocity-spike
//      heuristic, NOT an established technique like the angle-based
//      exercises — see the original module note preserved below. The new
//      guard-check (non-punching hand height vs. chin) is a much more
//      reliable signal by comparison: it's a simple, well-defined Y-axis
//      comparison, no angle-estimation noise involved.

import type { NormalizedLandmark } from "@mediapipe/tasks-vision"

export type Landmark = NormalizedLandmark
export type Tone = "good" | "warn" | "info"
export type ExerciseMode = "Shadowboxing" | "Squats" | "Lunges" | "Jab-Cross" | "Push-ups" | "Plank"

/**
 * Real-time, mid-movement correction — fires the instant a form fault is
 * detected, independent of whether/when a rep completes. Distinct from
 * `cue`, which only fires once a rep finishes. A formAlert is meant to
 * interrupt ("you are leaning too far forward — right now"), so the UI
 * should treat it with more urgency / different styling than a post-rep cue.
 */
export type FormAlert = { text: string; tone: "warn"; code: string }

export type FrameResult = {
  repCompleted: boolean
  formScore: number | null // 0-100, this rep's form quality, null if no rep judged yet
  cue: { text: string; tone: Tone } | null
  formAlert: FormAlert | null
  trackingOk: boolean // false if required landmarks aren't visible enough to analyze

  // Plank-only fields. Always present (zeroed) for non-Plank modes so
  // callers don't need mode-specific destructuring.
  holdSeconds: number
  bestHoldSeconds: number
  holdBroken: boolean // true the instant a held position breaks — fire-once edge, not a level
}

// --- MediaPipe BlazePose landmark indices --------------------------------
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const NOSE = 0
const MOUTH_L = 9, MOUTH_R = 10
const L_SHOULDER = 11, R_SHOULDER = 12
const L_ELBOW = 13, R_ELBOW = 14
const L_WRIST = 15, R_WRIST = 16
const L_HIP = 23, R_HIP = 24
const L_KNEE = 25, R_KNEE = 26
const L_ANKLE = 27, R_ANKLE = 28

const VISIBILITY_MIN = 0.5 // "intermediate" tolerance — strict would be ~0.7+

function angleAt(a: Landmark, vertex: Landmark, b: Landmark): number {
  // Angle at `vertex`, formed by rays to `a` and `b`, in degrees.
  const v1x = a.x - vertex.x, v1y = a.y - vertex.y
  const v2x = b.x - vertex.x, v2y = b.y - vertex.y
  const dot = v1x * v2x + v1y * v2y
  const mag1 = Math.hypot(v1x, v1y)
  const mag2 = Math.hypot(v2x, v2y)
  if (mag1 === 0 || mag2 === 0) return 180
  const cos = Math.min(1, Math.max(-1, dot / (mag1 * mag2)))
  return (Math.acos(cos) * 180) / Math.PI
}

// Angle of the line a→b from horizontal, in degrees, 0-180. Used for
// straight-line checks (shoulder-hip-ankle for plank/push-ups) where what
// matters is "is this line straight" rather than a joint bend angle.
function lineAngleFromHorizontal(a: Landmark, b: Landmark): number {
  const dx = b.x - a.x
  const dy = b.y - a.y
  return (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180) / Math.PI
}

function visible(lm: Landmark | undefined): lm is Landmark {
  return !!lm && (lm.visibility ?? 1) >= VISIBILITY_MIN
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v))
}

function mapRange(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  const t = (v - inMin) / (inMax - inMin)
  return outMin + clamp(t, 0, 1) * (outMax - outMin)
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

// Small moving average so a single noisy frame can't flip a state machine
// and produce a phantom rep, or a phantom form-fault flash. Window of 5 at
// ~30fps is ~165ms — enough to smooth jitter without meaningfully lagging
// real movement.
class RollingAverage {
  private buf: number[] = []
  constructor(private size = 5) {}
  push(v: number): number {
    this.buf.push(v)
    if (this.buf.length > this.size) this.buf.shift()
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length
  }
  get sampleCount() {
    return this.buf.length
  }
  reset() {
    this.buf = []
  }
}

// Fires a boolean condition only after it has been continuously true for
// `persistFrames` consecutive checks, and clears the instant the condition
// goes false. This is what keeps the new multi-joint form alerts (knee
// valgus, back sag, dropped guard) from flickering on a single noisy frame
// — given MediaPipe's published ~19° error band on derived angles like
// knee valgus, a single-frame trigger would fire constantly on noise alone.
class PersistentFlag {
  private count = 0
  constructor(private persistFrames = 6) {}
  update(condition: boolean): boolean {
    if (condition) {
      this.count++
    } else {
      this.count = 0
    }
    return this.count === this.persistFrames // fires exactly once, on the frame it crosses the threshold
  }
  reset() {
    this.count = 0
  }
}

// Time-based persistence flag — same idea as PersistentFlag but driven by
// elapsed video timestamp rather than frame count, used by
// CalibrationController where steps are measured in "hold this for ~1
// second" rather than "N consecutive frames" (frame rate can vary with
// device load, so a frame-count threshold would be inconsistent across
// devices for these slower, coarser checks).
class PersistentFlagSeconds {
  private trueSinceMs: number | null = null
  constructor(private requiredSeconds: number) {}
  update(condition: boolean, timestampMs: number): boolean {
    if (!condition) {
      this.trueSinceMs = null
      return false
    }
    if (this.trueSinceMs === null) this.trueSinceMs = timestampMs
    return (timestampMs - this.trueSinceMs) / 1000 >= this.requiredSeconds
  }
  reset() {
    this.trueSinceMs = null
  }
}

// ===========================================================================
// Calibration — the pre-exercise "get into position" phase.
// ===========================================================================

export type CalibrationStepId = "frame" | "position" | "countdown" | "done"

export type CalibrationStatus = {
  step: CalibrationStepId
  message: string
  /** Seconds remaining, only meaningful during the "countdown" step. */
  countdownValue?: number
}

// How long a person must continuously satisfy a calibration step before
// advancing — same noise-rejection reasoning as PersistentFlag above, just
// expressed as seconds since these checks are coarser/slower than per-rep
// form alerts.
const FRAME_CHECK_SECONDS = 1.0
const POSITION_CHECK_SECONDS = 1.0
const COUNTDOWN_SECONDS = 3

type ReadyPositionCheck = (lm: Landmark[]) => boolean

// Mode-specific "are you in the starting position" checks. Deliberately
// loose (intermediate, not strict) — these only gate the countdown, not
// rep scoring, so erring toward permissive here just means calibration
// finishes a bit faster, not that bad reps get scored as good ones.
const READY_CHECKS: Record<ExerciseMode, ReadyPositionCheck> = {
  Squats: (lm) => isStandingUpright(lm),
  Lunges: (lm) => isStandingUpright(lm),
  "Push-ups": (lm) => isInPlankLikePosition(lm),
  Plank: (lm) => isInPlankLikePosition(lm),
  Shadowboxing: (lm) => isGuardUpBothHands(lm),
  "Jab-Cross": (lm) => isGuardUpBothHands(lm),
}

function isStandingUpright(lm: Landmark[]): boolean {
  const lk = lm[L_KNEE], rk = lm[R_KNEE], lh = lm[L_HIP], rh = lm[R_HIP], la = lm[L_ANKLE], ra = lm[R_ANKLE]
  if (![lk, rk, lh, rh, la, ra].every(visible)) return false
  // "Standing" ~= knees are close to straight (large hip-knee-ankle angle)
  // on at least one visible side.
  const leftAngle = angleAt(lh, lk, la)
  const rightAngle = angleAt(rh, rk, ra)
  return Math.max(leftAngle, rightAngle) > 150
}

function isInPlankLikePosition(lm: Landmark[]): boolean {
  const ls = lm[L_SHOULDER], rs = lm[R_SHOULDER], lh = lm[L_HIP], rh = lm[R_HIP]
  if (![ls, rs, lh, rh].every(visible)) return false
  // Roughly horizontal torso (shoulder-midpoint to hip-midpoint line close
  // to flat) is a cheap proxy for "in a plank/push-up start position"
  // without requiring ankle visibility, since feet are often out of frame
  // on tighter shots.
  const shoulderMid = midpoint(ls, rs)
  const hipMid = midpoint(lh, rh)
  const torsoFlatness = lineAngleFromHorizontal(shoulderMid, hipMid)
  return torsoFlatness < 35
}

function isGuardUpBothHands(lm: Landmark[]): boolean {
  const lw = lm[L_WRIST], rw = lm[R_WRIST], nose = lm[NOSE]
  if (![lw, rw, nose].every(visible)) return false
  // Both wrists at or above chin height (nose.y is a close, always-visible proxy for chin height).
  return lw.y <= nose.y + 0.1 && rw.y <= nose.y + 0.1
}

function isFullBodyInFrame(lm: Landmark[], mode: ExerciseMode): boolean {
  // What "full body" means depends on the exercise — match the limb chain
  // each mode's tracking logic actually depends on, rather than requiring
  // every landmark regardless of relevance:
  //   - Squats/Lunges need legs (that's what's measured).
  //   - Plank/Push-ups need one full arm chain, shot more tightly/from the side.
  //   - Shadowboxing/Jab-Cross need wrists — requiring ankle visibility here
  //     would fail calibration for a perfectly normal waist-up framing.
  const core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP].map((i) => lm[i])
  if (!core.every(visible)) return false

  if (mode === "Plank" || mode === "Push-ups") {
    return [lm[L_ELBOW], lm[L_WRIST]].every(visible) || [lm[R_ELBOW], lm[R_WRIST]].every(visible)
  }
  if (mode === "Shadowboxing" || mode === "Jab-Cross") {
    return visible(lm[L_WRIST]) && visible(lm[R_WRIST])
  }
  const oneLegChain = [lm[L_KNEE], lm[L_ANKLE]].every(visible) || [lm[R_KNEE], lm[R_ANKLE]].every(visible)
  return oneLegChain
}

/**
 * Drives the "Calibration" pre-exercise phase described in the product
 * request: frame check -> ready-position check -> 3-2-1 countdown -> done.
 * Owned separately from ExerciseAnalyzer (which only runs once calibration
 * is done) so the two state machines don't tangle.
 */
export class CalibrationController {
  private step: CalibrationStepId = "frame"
  private frameOkFlag = new PersistentFlagSeconds(FRAME_CHECK_SECONDS)
  private positionOkFlag = new PersistentFlagSeconds(POSITION_CHECK_SECONDS)
  private countdownStartedAt: number | null = null

  reset() {
    this.step = "frame"
    this.frameOkFlag.reset()
    this.positionOkFlag.reset()
    this.countdownStartedAt = null
  }

  get currentStep() {
    return this.step
  }

  /**
   * Advances the calibration state machine by one frame and returns the
   * current step + message to show the user. When this returns
   * `step: "done"` for the first time, the caller should immediately call
   * `ExerciseAnalyzer.captureBaseline(landmarks, mode)` with that same
   * frame's landmarks, then start feeding frames to `ExerciseAnalyzer`
   * instead of this controller.
   */
  process(landmarks: Landmark[] | undefined, mode: ExerciseMode, timestampMs: number): CalibrationStatus {
    if (this.step === "done") return { step: "done", message: "" }

    if (!landmarks || landmarks.length < 33) {
      // Lost tracking entirely — fall back to the frame-check message but
      // don't reset progress on a single dropped frame.
      return { step: this.step === "frame" ? "frame" : this.step, message: "Make sure your camera can see you." }
    }

    if (this.step === "frame") {
      const ok = this.frameOkFlag.update(isFullBodyInFrame(landmarks, mode), timestampMs)
      if (ok) {
        this.step = "position"
      } else {
        return { step: "frame", message: "Step back so your full body is in frame." }
      }
    }

    if (this.step === "position") {
      const check = READY_CHECKS[mode]
      const ok = this.positionOkFlag.update(check(landmarks), timestampMs)
      if (ok) {
        this.step = "countdown"
        this.countdownStartedAt = timestampMs
      } else {
        return { step: "position", message: positionMessage(mode) }
      }
    }

    if (this.step === "countdown") {
      const elapsed = (timestampMs - (this.countdownStartedAt ?? timestampMs)) / 1000
      const remaining = Math.max(0, Math.ceil(COUNTDOWN_SECONDS - elapsed))
      if (remaining <= 0) {
        this.step = "done"
        return { step: "done", message: "Go!" }
      }
      return { step: "countdown", message: `Perfect, starting in ${remaining}...`, countdownValue: remaining }
    }

    return { step: "done", message: "" }
  }
}

function positionMessage(mode: ExerciseMode): string {
  switch (mode) {
    case "Squats":
    case "Lunges":
      return "Stand upright with your feet shoulder-width apart."
    case "Push-ups":
    case "Plank":
      return "Get into a straight-arm plank position, filmed from the side if possible."
    case "Shadowboxing":
    case "Jab-Cross":
      return "Bring both hands up to guard your chin."
    default: {
      const _exhaustive: never = mode
      return _exhaustive
    }
  }
}

// ===========================================================================
// Per-exercise state
// ===========================================================================

type AngleCycleState = {
  phase: "up" | "down"
  angleAvg: RollingAverage
  minAngleThisRep: number
}

type WristState = {
  prevPos: { x: number; y: number } | null
  prevT: number | null
  speedAvg: RollingAverage
  armed: boolean
  peakSpeed: number
}

type PlankState = {
  holdStartMs: number | null
  bestHoldSeconds: number
  alignmentAvg: RollingAverage
}

// --- Squat / Lunge depth thresholds (intermediate / balanced tolerance) ---
const SQUAT_DOWN_ANGLE = 110
const SQUAT_UP_ANGLE = 160
const LUNGE_DOWN_ANGLE = 110
const LUNGE_UP_ANGLE = 160

// --- Push-up depth thresholds — same convention as squats: interior elbow
// angle, smaller = deeper. 90° elbow bend is the common "good depth" cue. ---
const PUSHUP_DOWN_ANGLE = 95
const PUSHUP_UP_ANGLE = 155

// --- Strike velocity thresholds (normalized units/sec) — heuristic, not
// calibrated against real footage. See module note above. ---
const STRIKE_EXTEND_SPEED = 1.4
const STRIKE_RETRACT_SPEED = 0.5

// --- Multi-joint form-correction thresholds ---
// Knee valgus: how far the knee drifts medially, expressed as a fraction of
// hip width, relative to the straight hip-ankle line. Given MediaPipe's
// published ~19° error band on this kind of derived angle, this is set
// deliberately loose and requires PersistentFlag confirmation before firing.
const KNEE_VALGUS_RATIO = 0.18
// Spinal lean: degrees of additional forward lean beyond this person's own
// calibration-phase standing baseline before flagging it. Comparing to a
// personal baseline (not a fixed constant) is the fix for the torso-lean
// mistake from the previous version of this file.
const SPINAL_LEAN_DELTA_DEG = 25
// Guard drop: how far below chin height (in normalized units) the
// non-punching wrist must fall before it counts as "guard down."
const GUARD_DROP_MARGIN = 0.05
// Back sag/pike for push-ups & plank: degrees the shoulder-hip-ankle line
// may deviate from straight before flagging. Loosened from a textbook ~10°
// because this metric is sensitive to camera angle — see module note above
// about side-view framing.
const BACK_ALIGNMENT_TOLERANCE_DEG = 18

const SQUAT_FEEDBACK: Record<Tone, string[]> = {
  good: ["Solid depth on that rep.", "Nice — full range of motion that time."],
  warn: ["Try to squat a touch deeper next rep.", "That one was a bit shallow — aim for thighs closer to parallel."],
  info: ["Keep your chest up and core braced as you descend."],
}
const LUNGE_FEEDBACK: Record<Tone, string[]> = {
  good: ["Good depth on that lunge.", "Nice full range on that rep."],
  warn: ["Lengthen your stride a little for more depth.", "Try to get that front thigh closer to parallel."],
  info: ["Push through the front heel as you rise."],
}
const STRIKE_FEEDBACK: Record<Tone, string[]> = {
  good: ["Quick extension on that one.", "Good snap — nice retraction speed too."],
  warn: ["Reset your guard faster after extending.", "Keep your other hand up while you strike."],
  info: ["Stay light on your feet between strikes."],
}
const PUSHUP_FEEDBACK: Record<Tone, string[]> = {
  good: ["Full depth on that rep — nice work.", "Great range of motion."],
  warn: ["Lower a bit further next rep for full depth.", "Try to get your chest closer to the floor."],
  info: ["Keep your core tight through the rep."],
}

export class ExerciseAnalyzer {
  private leftLeg = freshAngleCycle()
  private rightLeg = freshAngleCycle()
  private leftArm = freshAngleCycle()
  private rightArm = freshAngleCycle()
  private leftWrist = freshWristState()
  private rightWrist = freshWristState()
  private plank: PlankState = { holdStartMs: null, bestHoldSeconds: 0, alignmentAvg: new RollingAverage(6) }

  // Personal baseline captured once, right when calibration completes —
  // see CalibrationController. Used so spinal-lean checks compare against
  // THIS person's own standing posture rather than a fixed global angle.
  private baselineSpinalAngle: number | null = null

  // Persistence gates for the new multi-joint form alerts. Separate
  // instances per check so e.g. a knee-valgus flag and a spinal-lean flag
  // can each independently confirm over a few frames without interfering.
  private kneeValgusFlag = new PersistentFlag(6)
  private spinalLeanFlag = new PersistentFlag(6)
  private guardDropFlag = new PersistentFlag(5)
  private backSagFlag = new PersistentFlag(6)

  reset() {
    this.leftLeg = freshAngleCycle()
    this.rightLeg = freshAngleCycle()
    this.leftArm = freshAngleCycle()
    this.rightArm = freshAngleCycle()
    this.leftWrist = freshWristState()
    this.rightWrist = freshWristState()
    this.plank = { holdStartMs: null, bestHoldSeconds: 0, alignmentAvg: new RollingAverage(6) }
    this.baselineSpinalAngle = null
    this.kneeValgusFlag.reset()
    this.spinalLeanFlag.reset()
    this.guardDropFlag.reset()
    this.backSagFlag.reset()
  }

  /**
   * Call once, the moment CalibrationController reports "done", with that
   * same frame's landmarks. Captures this person's standing posture as the
   * reference point for the squat/lunge spinal-lean check, instead of a
   * fixed constant — see module note #2 at the top of this file for why.
   *
   * Deliberately does nothing for Push-ups/Plank: their back-alignment
   * check targets a true geometric straight line (BACK_ALIGNMENT_TOLERANCE_DEG),
   * which is an objective target unlike "ideal torso lean," so it doesn't
   * need or use a personal baseline.
   */
  captureBaseline(landmarks: Landmark[], mode: ExerciseMode) {
    if (mode === "Squats" || mode === "Lunges") {
      const ls = landmarks[L_SHOULDER], rs = landmarks[R_SHOULDER], lh = landmarks[L_HIP], rh = landmarks[R_HIP]
      if ([ls, rs, lh, rh].every(visible)) {
        const shoulderMid = midpoint(ls, rs)
        const hipMid = midpoint(lh, rh)
        this.baselineSpinalAngle = lineAngleFromHorizontal(shoulderMid, hipMid)
      }
    }
  }

  private emptyResult(trackingOk: boolean): FrameResult {
    return {
      repCompleted: false,
      formScore: null,
      cue: null,
      formAlert: null,
      trackingOk,
      holdSeconds: 0,
      bestHoldSeconds: this.plank.bestHoldSeconds,
      holdBroken: false,
    }
  }

  processFrame(landmarks: Landmark[] | undefined, mode: ExerciseMode, timestampMs: number): FrameResult {
    if (!landmarks || landmarks.length < 33) {
      return this.emptyResult(false)
    }

    switch (mode) {
      case "Squats":
        return this.processSquat(landmarks)
      case "Lunges":
        return this.processLunge(landmarks)
      case "Push-ups":
        return this.processPushup(landmarks)
      case "Plank":
        return this.processPlank(landmarks, timestampMs)
      case "Shadowboxing":
      case "Jab-Cross":
        return this.processStrike(landmarks, timestampMs)
      default: {
        const _exhaustive: never = mode
        return _exhaustive
      }
    }
  }

  // --- Squats -------------------------------------------------------------
  // Rep counting: single hip-knee-ankle angle (more visible side), unchanged
  // from the original implementation — that part is the well-established
  // technique and isn't where the new complexity belongs.
  // Form correction (new): knee valgus (knee drifting inward of the
  // hip-ankle line) and spinal lean vs. this person's own baseline. Both
  // are advisory only — neither can prevent a rep from counting.
  private processSquat(lm: Landmark[]): FrameResult {
    const side = pickVisibleLegSide(lm)
    if (!side) return this.emptyResult(false)

    const { hip, knee, ankle, isLeft } = side
    const leg = isLeft ? this.leftLeg : this.rightLeg
    const angle = leg.angleAvg.push(angleAt(hip, knee, ankle))
    leg.minAngleThisRep = Math.min(leg.minAngleThisRep, angle)

    const result = this.emptyResult(true)

    // -- knee valgus check (advisory) --
    // Reference point: where the knee WOULD be, in x, if it sat exactly on
    // the straight hip-ankle line at the knee's height — i.e. linear
    // interpolation along that line at knee.y. This is the correct
    // reference (matches the published "knee position relative to the
    // hip-ankle line" approach) rather than assuming the hip sits directly
    // above the ankle, which isn't true once the hips shift back at depth.
    const hipWidth = Math.abs(lm[L_HIP].x - lm[R_HIP].x) || 0.0001
    const hipAnkleSpanY = ankle.y - hip.y
    const t = hipAnkleSpanY !== 0 ? clamp((knee.y - hip.y) / hipAnkleSpanY, 0, 1) : 0.5
    const expectedKneeX = hip.x + t * (ankle.x - hip.x)
    const medialDrift = isLeft ? expectedKneeX - knee.x : knee.x - expectedKneeX // positive = drifting inward
    const valgusNow = leg.phase === "down" && medialDrift / hipWidth > KNEE_VALGUS_RATIO
    if (this.kneeValgusFlag.update(valgusNow)) {
      result.formAlert = { text: "Push your knees outward.", tone: "warn", code: "knee_valgus" }
    }

    // -- spinal lean check (advisory, vs. personal baseline) --
    if (this.baselineSpinalAngle !== null) {
      const shoulderMid = midpoint(lm[L_SHOULDER], lm[R_SHOULDER])
      const hipMid = midpoint(lm[L_HIP], lm[R_HIP])
      if ([lm[L_SHOULDER], lm[R_SHOULDER], lm[L_HIP], lm[R_HIP]].every(visible)) {
        const currentSpinalAngle = lineAngleFromHorizontal(shoulderMid, hipMid)
        const leanNow = leg.phase === "down" && Math.abs(currentSpinalAngle - this.baselineSpinalAngle) > SPINAL_LEAN_DELTA_DEG
        if (this.spinalLeanFlag.update(leanNow) && !result.formAlert) {
          result.formAlert = {
            text: "Keep your chest up, you are leaning too far forward.",
            tone: "warn",
            code: "spinal_lean",
          }
        }
      }
    }

    if (leg.phase === "up" && angle < SQUAT_DOWN_ANGLE) {
      leg.phase = "down"
    } else if (leg.phase === "down" && angle > SQUAT_UP_ANGLE) {
      leg.phase = "up"
      result.repCompleted = true
      result.formScore = Math.round(clamp(mapRange(leg.minAngleThisRep, 70, SQUAT_DOWN_ANGLE, 100, 60), 0, 100))
      const tone: Tone = result.formScore >= 85 ? "good" : result.formScore >= 65 ? "info" : "warn"
      result.cue = { text: pick(SQUAT_FEEDBACK[tone]), tone }
      leg.minAngleThisRep = 180
    }

    return result
  }

  // --- Lunges --------------------------------------------------------------
  private processLunge(lm: Landmark[]): FrameResult {
    const leftOk = [lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]].every(visible)
    const rightOk = [lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]].every(visible)
    if (!leftOk && !rightOk) return this.emptyResult(false)

    const result = this.emptyResult(true)

    for (const [ok, hipIdx, kneeIdx, ankleIdx, leg] of [
      [leftOk, L_HIP, L_KNEE, L_ANKLE, this.leftLeg],
      [rightOk, R_HIP, R_KNEE, R_ANKLE, this.rightLeg],
    ] as const) {
      if (!ok) continue
      const hip = lm[hipIdx], knee = lm[kneeIdx], ankle = lm[ankleIdx]
      const angle = leg.angleAvg.push(angleAt(hip, knee, ankle))
      leg.minAngleThisRep = Math.min(leg.minAngleThisRep, angle)

      if (leg.phase === "up" && angle < LUNGE_DOWN_ANGLE) {
        leg.phase = "down"
      } else if (leg.phase === "down" && angle > LUNGE_UP_ANGLE) {
        leg.phase = "up"
        result.repCompleted = true
        const depthScore = clamp(mapRange(leg.minAngleThisRep, 70, LUNGE_DOWN_ANGLE, 100, 60), 0, 100)
        result.formScore = Math.round(depthScore)
        const tone: Tone = result.formScore >= 85 ? "good" : result.formScore >= 65 ? "info" : "warn"
        result.cue = { text: pick(LUNGE_FEEDBACK[tone]), tone }
        leg.minAngleThisRep = 180
      }
    }

    return result
  }

  // --- Push-ups -------------------------------------------------------------
  // Depth: shoulder-elbow-wrist angle (more visible side), same convention
  // as the squat knee angle. Back alignment: shoulder-hip-ankle line should
  // stay close to straight — most reliable filmed from the side (see
  // module note). Advisory only, like the squat checks.
  private processPushup(lm: Landmark[]): FrameResult {
    const side = pickVisibleArmSide(lm)
    if (!side) return this.emptyResult(false)

    const { shoulder, elbow, wrist, hip, ankle, isLeft } = side
    const arm = isLeft ? this.leftArm : this.rightArm
    const angle = arm.angleAvg.push(angleAt(shoulder, elbow, wrist))
    arm.minAngleThisRep = Math.min(arm.minAngleThisRep, angle)

    const result = this.emptyResult(true)

    if (hip && ankle) {
      const deviationDeg = backAlignmentDeviation(shoulder, hip, ankle)
      if (this.backSagFlag.update(deviationDeg > BACK_ALIGNMENT_TOLERANCE_DEG)) {
        const sagging = hipSagDirection(shoulder, hip, ankle) > 0
        result.formAlert = sagging
          ? { text: "Keep your back straight — don't let your hips sag.", tone: "warn", code: "back_sag" }
          : { text: "Lower your hips — you're piking up.", tone: "warn", code: "back_pike" }
      }
    }

    if (arm.phase === "up" && angle < PUSHUP_DOWN_ANGLE) {
      arm.phase = "down"
    } else if (arm.phase === "down" && angle > PUSHUP_UP_ANGLE) {
      arm.phase = "up"
      result.repCompleted = true
      result.formScore = Math.round(clamp(mapRange(arm.minAngleThisRep, 70, PUSHUP_DOWN_ANGLE, 100, 60), 0, 100))
      const tone: Tone = result.formScore >= 85 ? "good" : result.formScore >= 65 ? "info" : "warn"
      result.cue = { text: pick(PUSHUP_FEEDBACK[tone]), tone }
      arm.minAngleThisRep = 180
    }

    return result
  }

  // --- Plank -----------------------------------------------------------------
  // Not a rep counter — a held-state timer. Alignment uses the same
  // shoulder-hip-ankle straight-line idea as push-up back-sag, but here
  // it's the primary signal (whether the hold continues) rather than an
  // advisory extra, since "is the line straight" IS the entire exercise.
  private processPlank(lm: Landmark[], timestampMs: number): FrameResult {
    const ls = lm[L_SHOULDER], rs = lm[R_SHOULDER], lh = lm[L_HIP], rh = lm[R_HIP]
    const la = lm[L_ANKLE], ra = lm[R_ANKLE]
    const coreVisible = [ls, rs, lh, rh].every(visible)
    if (!coreVisible) {
      return this.breakPlankHold(timestampMs, false)
    }

    const shoulder = midpoint(ls, rs)
    const hip = midpoint(lh, rh)
    let deviation: number
    let sagging: boolean | null = null // null = direction unknown (ankles not visible)
    if (visible(la) || visible(ra)) {
      const ankle = visible(la) && visible(ra) ? midpoint(la, ra) : (visible(la) ? la : ra)
      deviation = backAlignmentDeviation(shoulder, hip, ankle)
      sagging = hipSagDirection(shoulder, hip, ankle) > 0
    } else {
      // Ankles out of frame on a tight shot — fall back to shoulder-hip
      // flatness alone. Less reliable, and direction (sag vs pike) can't be
      // determined without the ankle, so sagging stays null here.
      deviation = lineAngleFromHorizontal(shoulder, hip)
    }
    const sampleCountBeforePush = this.plank.alignmentAvg.sampleCount
    const smoothedDeviation = this.plank.alignmentAvg.push(deviation)
    // Don't trust the rolling average until it actually has a few samples —
    // otherwise the very first frame of a plank attempt could read as
    // "aligned" off a single lucky reading.
    const smoothDeviation = sampleCountBeforePush >= 2 ? smoothedDeviation : deviation

    const aligned = smoothDeviation <= BACK_ALIGNMENT_TOLERANCE_DEG

    if (aligned) {
      if (this.plank.holdStartMs === null) this.plank.holdStartMs = timestampMs
      const holdSeconds = (timestampMs - this.plank.holdStartMs) / 1000
      return {
        ...this.emptyResult(true),
        holdSeconds,
        bestHoldSeconds: Math.max(this.plank.bestHoldSeconds, holdSeconds),
      }
    }

    return this.breakPlankHold(timestampMs, true, sagging)
  }

  private breakPlankHold(timestampMs: number, trackingOk: boolean, sagging: boolean | null = null): FrameResult {
    const wasHolding = this.plank.holdStartMs !== null
    const finishedHoldSeconds = wasHolding ? (timestampMs - this.plank.holdStartMs!) / 1000 : 0
    if (wasHolding) {
      this.plank.bestHoldSeconds = Math.max(this.plank.bestHoldSeconds, finishedHoldSeconds)
    }
    this.plank.holdStartMs = null

    let breakMessage: string | null = null
    if (wasHolding) {
      if (sagging === true) breakMessage = "Hips dropped — reset and try to hold a straight line."
      else if (sagging === false) breakMessage = "Hips piked up — reset and try to hold a straight line."
      else breakMessage = "Form broke — reset and try to hold a straight line."
    }

    return {
      ...this.emptyResult(trackingOk),
      holdSeconds: 0,
      bestHoldSeconds: this.plank.bestHoldSeconds,
      holdBroken: wasHolding, // only true on the exact frame the hold ends
      formAlert: breakMessage ? { text: breakMessage, tone: "warn", code: "plank_broken" } : null,
    }
  }

  // --- Shadowboxing / Jab-Cross --------------------------------------------
  // Strike counting: velocity-spike heuristic, unchanged from the original
  // implementation — see module note, this is NOT an established technique
  // the way the angle-based exercises are.
  // Guard check (new): is the non-punching wrist at or above chin height.
  // This one IS a simple, reliable signal — no angle-estimation noise.
  private processStrike(lm: Landmark[], timestampMs: number): FrameResult {
    const result = this.emptyResult(false)
    let anyTracked = false

    const nose = lm[NOSE]
    const chinY = visible(nose) ? nose.y + GUARD_DROP_MARGIN : null

    for (const [wristIdx, wrist] of [
      [L_WRIST, this.leftWrist],
      [R_WRIST, this.rightWrist],
    ] as const) {
      const point = lm[wristIdx]
      if (!visible(point)) {
        wrist.prevPos = null
        wrist.prevT = null
        continue
      }
      anyTracked = true

      if (wrist.prevPos !== null && wrist.prevT !== null) {
        const dt = (timestampMs - wrist.prevT) / 1000
        if (dt > 0) {
          const dist = Math.hypot(point.x - wrist.prevPos.x, point.y - wrist.prevPos.y)
          const speed = wrist.speedAvg.push(dist / dt)

          if (!wrist.armed && speed > STRIKE_EXTEND_SPEED) {
            wrist.armed = true
            wrist.peakSpeed = speed
          } else if (wrist.armed) {
            wrist.peakSpeed = Math.max(wrist.peakSpeed, speed)
            if (speed < STRIKE_RETRACT_SPEED) {
              wrist.armed = false
              result.repCompleted = true

              const otherWristIdx = wristIdx === L_WRIST ? R_WRIST : L_WRIST
              const otherShoulderIdx = wristIdx === L_WRIST ? R_SHOULDER : L_SHOULDER
              const guardLm = lm[otherWristIdx]
              const guardShoulderLm = lm[otherShoulderIdx]
              const guardUp = visible(guardLm) && visible(guardShoulderLm) && guardLm.y <= guardShoulderLm.y + 0.08
              const speedScore = clamp(mapRange(wrist.peakSpeed, STRIKE_EXTEND_SPEED, 3.5, 60, 100), 0, 100)
              const guardScore = guardUp ? 100 : 55
              result.formScore = Math.round(speedScore * 0.7 + guardScore * 0.3)

              const tone: Tone = result.formScore >= 85 ? "good" : result.formScore >= 65 ? "info" : "warn"
              result.cue = { text: pick(STRIKE_FEEDBACK[tone]), tone }
              wrist.peakSpeed = 0
            }
          }
        }
      }

      wrist.prevPos = { x: point.x, y: point.y }
      wrist.prevT = timestampMs
    }

    // Standalone guard check — independent of whether a strike is mid-flight,
    // since dropping your guard between strikes is exactly when it matters.
    if (chinY !== null) {
      const leftWristLm = lm[L_WRIST], rightWristLm = lm[R_WRIST]
      const leftDown = visible(leftWristLm) && leftWristLm.y > chinY
      const rightDown = visible(rightWristLm) && rightWristLm.y > chinY
      // Flag if BOTH hands are down (a single extended punching hand is
      // expected to drop momentarily — that's not a guard fault by itself).
      const guardDown = leftDown && rightDown
      if (this.guardDropFlag.update(guardDown) && !result.formAlert) {
        result.formAlert = { text: "Keep your guard up!", tone: "warn", code: "guard_down" }
      }
    }

    result.trackingOk = anyTracked
    return result
  }
}

// ===========================================================================
// Shared geometry helpers
// ===========================================================================

function freshAngleCycle(): AngleCycleState {
  return { phase: "up", angleAvg: new RollingAverage(5), minAngleThisRep: 180 }
}

function freshWristState(): WristState {
  return { prevPos: null, prevT: null, speedAvg: new RollingAverage(4), armed: false, peakSpeed: 0 }
}

function midpoint(a: Landmark, b: Landmark): Landmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2, visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1) }
}

// Degrees the shoulder-hip-ankle path deviates from a straight line — 0 means
// perfectly straight. Used for plank holds and push-up back-sag/pike checks.
function backAlignmentDeviation(shoulder: Landmark, hip: Landmark, ankle: Landmark): number {
  const angleAtHip = angleAt(shoulder, hip, ankle)
  return Math.abs(180 - angleAtHip)
}

// Signed version: positive means the hip sits BELOW where a straight
// shoulder-ankle line would put it (sagging), negative means above (piking).
// Magnitude is in the same normalized units as landmark y-coordinates, not
// degrees — only the sign matters here, the angle itself still comes from
// backAlignmentDeviation.
function hipSagDirection(shoulder: Landmark, hip: Landmark, ankle: Landmark): number {
  const t = ankle.x !== shoulder.x ? clamp((hip.x - shoulder.x) / (ankle.x - shoulder.x), 0, 1) : 0.5
  const expectedHipY = shoulder.y + t * (ankle.y - shoulder.y)
  return hip.y - expectedHipY // image y increases downward, so positive = hip is lower = sagging
}

function pickVisibleLegSide(lm: Landmark[]) {
  const leftOk = [lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]].every(visible)
  const rightOk = [lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]].every(visible)
  if (leftOk) return { hip: lm[L_HIP], knee: lm[L_KNEE], ankle: lm[L_ANKLE], isLeft: true as const }
  if (rightOk) return { hip: lm[R_HIP], knee: lm[R_KNEE], ankle: lm[R_ANKLE], isLeft: false as const }
  return null
}

function pickVisibleArmSide(lm: Landmark[]) {
  const leftOk = [lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST]].every(visible)
  const rightOk = [lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST]].every(visible)
  const hipAnkle = (hipIdx: number, ankleIdx: number) =>
    visible(lm[hipIdx]) && visible(lm[ankleIdx]) ? { hip: lm[hipIdx], ankle: lm[ankleIdx] } : { hip: null, ankle: null }

  if (leftOk) {
    const { hip, ankle } = hipAnkle(L_HIP, L_ANKLE)
    return { shoulder: lm[L_SHOULDER], elbow: lm[L_ELBOW], wrist: lm[L_WRIST], hip, ankle, isLeft: true as const }
  }
  if (rightOk) {
    const { hip, ankle } = hipAnkle(R_HIP, R_ANKLE)
    return { shoulder: lm[R_SHOULDER], elbow: lm[R_ELBOW], wrist: lm[R_WRIST], hip, ankle, isLeft: false as const }
  }
  return null
}
