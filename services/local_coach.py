"""Offline Stage D: a deterministic coach for when Gemini is unreachable.

Gemini has failed three distinct ways in a single day of testing -- a withdrawn
model (404), per-model overload (503), and a dropped connection
(RemoteProtocolError). Each time, Stages A-C had already produced everything
they produce on a good day: the exercise, the rep boundaries, the worst moment,
and a segmented region around it. Returning nothing because the narration leg
was down throws away work that succeeded.

So this module writes the report from the local measurements alone. It is
rule-based and fully deterministic: the same clip yields the same words every
time, which is the right trade when the alternative is no report.

WHAT IT HONESTLY KNOWS, AND WHAT IT DOES NOT
    It knows the exercise ONLY when the footage is gym work -- the classifier
    has 27 classes and about 44% validation accuracy, so the label is a guess
    even then, and on a tennis serve it is meaningless. It knows WHEN the
    movement broke from its own trend, because that is measured. It does NOT
    know what the athlete's knee did, because nothing local looks at knees.

    Every cue here is therefore a MOVEMENT-STANDARD reminder keyed to the
    detected exercise -- the thing a coach says to everyone doing that lift --
    never a claim about this particular athlete. "Keep the bar over midfoot"
    is safe and useful without having seen the bar. "Your bar drifted forward"
    would be a fabrication, and this module never makes one.

    The region label is positional, derived from where the SAM box sits in the
    frame. It is reported as "upper-frame region" rather than "shoulder"
    precisely because SAM has no vocabulary and neither does this.
"""

# Movement families. Grouping beats 27 hand-written entries: the members of a
# family share the same coaching standards, and a family stays correct when a
# new class is added to it.
FAMILIES = {
    "squat": {
        "members": ("squats", "barbell_squat", "lunges", "box_jumps"),
        "noun": "squat pattern",
        "clean": "Keep driving through midfoot and let the knees track over "
                 "the toes.",
        "flagged": "Control the descent and keep the knees tracking over the "
                   "toes -- most squat faults appear at the bottom, not the top.",
    },
    "hinge": {
        "members": ("deadlift", "kettlebell_swing", "barbell_row"),
        "noun": "hip hinge",
        "clean": "Keep the spine neutral and finish the lift with the hips, "
                 "not the lower back.",
        "flagged": "Brace the trunk before the pull and drive the hips through "
                   "-- a hinge that changes speed abruptly is usually the back "
                   "taking over from the hips.",
    },
    "horizontal_press": {
        "members": ("push_ups", "diamond_pushups", "bench_press", "dips"),
        "noun": "pressing movement",
        "clean": "Keep the elbows tucked to about 45 degrees and the ribs down.",
        "flagged": "Lower under control and keep the elbows at roughly 45 "
                   "degrees -- flaring them is the usual cause of an uneven "
                   "press.",
    },
    "vertical_press": {
        "members": ("shoulder_press", "lat_raise", "tricep_extension"),
        "noun": "overhead work",
        "clean": "Keep the ribs down and press in a straight line overhead.",
        "flagged": "Squeeze the glutes and keep the ribs down -- an arched "
                   "lower back is how overhead pressing hides a lack of "
                   "shoulder range.",
    },
    "vertical_pull": {
        "members": ("pull_ups", "chin_ups", "lat_pulldown", "bicep_curl"),
        "noun": "pulling movement",
        "clean": "Start each rep from a full hang and lead with the elbows.",
        "flagged": "Control the lowering phase -- pulling movements lose "
                   "tension on the way down, and that is where the tempo "
                   "usually breaks.",
    },
    "core": {
        "members": ("plank", "sit_ups", "leg_raises", "russian_twist"),
        "noun": "core movement",
        "clean": "Keep the ribs stacked over the pelvis and breathe steadily.",
        "flagged": "Slow the movement and keep the lower back flat -- core "
                   "work degrades into hip flexor work as soon as the tempo "
                   "runs away.",
    },
    "conditioning": {
        "members": ("burpees", "jumping_jacks", "jump_rope",
                    "mountain_climbers", "high_knees"),
        "noun": "conditioning drill",
        "clean": "Keep the contacts light and the rhythm even.",
        "flagged": "Hold a steady rhythm and land softly -- an uneven cadence "
                   "under fatigue is what turns conditioning into sloppy "
                   "landings.",
    },
}

# Reverse index, built once.
_FAMILY_OF = {m: name for name, f in FAMILIES.items() for m in f["members"]}

GENERIC = {
    "noun": "movement",
    "clean": "Keep the tempo even and the trunk braced throughout.",
    "flagged": "Slow the movement down and keep the trunk braced -- an abrupt "
               "change in speed is usually where control is lost.",
}

# Kinetic-error thresholds, matching the severity bands in latent_dynamics so
# the two never disagree about what "critical" means.
CRITICAL, WARNING = 3.0, 1.5


def _region(bbox):
    """A SAM box -> a positional description of where it sits in frame.

    Positional, NOT anatomical. SAM produced this box from a point prompt and
    has no idea what it enclosed; calling the top third "the shoulder" would
    invent an identification that nothing performed.
    """
    if not bbox or len(bbox) != 4:
        return None
    ymin, xmin, ymax, xmax = bbox
    cy = (ymin + ymax) / 2.0
    band = ("upper" if cy < 0.34 else "mid" if cy < 0.67 else "lower")
    width = xmax - xmin
    size = "wide" if width > 0.55 else "narrow" if width < 0.2 else ""
    return f"{band}-frame region{(' (' + size + ')') if size else ''}"


# Bounds on the BOX, expressed as a fraction of the frame. Above the max it
# points at everything; below the min it is a speck the eye cannot find.
BOX_AREA_MAX = 0.35
BOX_AREA_MIN = 0.01


def _box_area(bbox):
    ymin, xmin, ymax, xmax = bbox
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


def _pick_box(candidates):
    """The most useful SAM candidate to show, not simply the best-scoring one.

    SAM ranks by mask quality, and its cleanest mask is almost always the whole
    subject -- a box around the entire frame is a confident answer to a question
    nobody asked. So prefer a tight box.

    Filter on the BOX area, not the mask area. They diverge badly and it is not
    an edge case: a standing lifter with arms out measured mask_area 0.19 while
    its bounding box covered 0.97 of the frame. The mask is thin and sprawling;
    the box that encloses it is nearly everything. Since the box is what gets
    drawn on the video, the box is what has to be judged.

    With Gemini present this choice does not arise -- it sees the keyframe and
    picks the candidate matching the fault it observed. Offline there is no such
    judgement available, so the geometry stands in for it.
    """
    if not candidates:
        return None
    usable = [c for c in candidates
              if c.get("normalized_bbox")
              and BOX_AREA_MIN <= _box_area(c["normalized_bbox"]) <= BOX_AREA_MAX]
    # Nothing in range means the frame offered only whole-body masks and
    # specks. Fall back to the tightest available rather than the largest.
    pool = usable or sorted(candidates,
                            key=lambda c: _box_area(c["normalized_bbox"]))[:1]
    return max(pool, key=lambda c: c.get("sam_score", 0.0))["normalized_bbox"]


def _score(dynamics, reps):
    """A deterministic 0-100 from measured trajectory consistency.

    Deliberately blunt and deliberately generous. It measures how steady the
    movement was relative to itself, which is a real signal but NOT technique
    -- a beautifully smooth rep with terrible knee position scores well here.
    The response labels this `local-fallback` so nothing downstream mistakes
    it for a coach's judgement.
    """
    peak = float(dynamics.get("peak_kinetic_error") or 0.0)
    mean = float(dynamics.get("mean_kinetic_error") or 0.0)
    # Peak dominates: one hard break matters more than a slightly noisy mean.
    penalty = min(40.0, peak * 7.0) + min(15.0, abs(mean) * 5.0)
    score = int(round(max(45.0, 100.0 - penalty)))
    # A clip with no detected reps gives the score nothing to stand on.
    if reps.get("confidence") in ("none", "low"):
        score = min(score, 80)
    return score


def _phases(reps, dynamics, duration):
    """Phases from measured rep boundaries, or one span when there are none."""
    out = []
    worst = dynamics.get("worst_moment") or {}
    wt = worst.get("timestamp")
    for r in (reps.get("reps") or [])[:8]:
        flagged = wt is not None and r["t_start"] <= wt <= r["t_end"]
        err = r.get("kinetic_error", 0.0)
        out.append({
            "phase_name": f"Repetition {r['rep_number']}",
            "timestamp_start": r["t_start"],
            "timestamp_end": r["t_end"],
            "status": ("critical" if err >= CRITICAL else
                       "warning" if flagged or err >= WARNING else "optimal"),
            "observation": (
                f"Rep {r['rep_number']} ran {r['duration_s']}s with the bottom "
                f"position at {r['t_inflection']}s. Mean kinetic error "
                f"{err}, measured from the latent trajectory."
                + (" This rep contains the clip's single worst moment."
                   if flagged else "")),
        })
    if not out:
        out.append({
            "phase_name": "Full clip",
            "timestamp_start": 0.0,
            "timestamp_end": duration,
            "status": "warning" if wt is not None else "optimal",
            "observation": (
                "No repeating structure was detected, so the clip is reported "
                "as one continuous movement rather than split into reps."),
        })
    return out


OFFLINE_NOTE = (
    "Generated locally from V-JEPA classification, latent-trajectory "
    "measurements and SAM segmentation. No language model was reachable. Cues "
    "are movement standards for the detected exercise, not observations of "
    "this athlete.")


# --------------------------------------------------------------------------
# Public helpers, used per chapter by the timeline builder. These exist so the
# multi-exercise path can fill one field at a time -- Gemini may name three
# chapters and omit the fourth, and that fourth still needs words.
# --------------------------------------------------------------------------

def pick_box(candidates):
    """Public form of the box heuristic. See _pick_box."""
    return _pick_box(candidates)


def region_of(bbox):
    """Public form of the positional region label. See _region."""
    return _region(bbox)


def score_of(dynamics, reps):
    """Public form of the deterministic score. See _score."""
    return _score(dynamics or {}, reps or {})


def issue_text(worst_moment, bbox):
    """One sentence describing a flagged moment, with no fault asserted."""
    if not worst_moment:
        return ("No kinetic inflection stood out; the trajectory stayed "
                "consistent across this block.")
    region = _region(bbox)
    return (
        f"Kinetic inflection at {worst_moment['timestamp']}s"
        + (f" in the {region}" if region else "")
        + f", severity {worst_moment.get('severity')} "
          f"(error {worst_moment.get('kinetic_error')}). This marks where the "
          "movement departed from its own recent trend. No model reviewed the "
          "footage, so it is a place to look, not a diagnosed fault.")


def chapter_cue(label, worst_moment):
    """-> {"cue", "observation"} for one chapter, from its label alone."""
    fam = FAMILIES.get(_FAMILY_OF.get(label), GENERIC)
    pretty = (label or "movement").replace("_", " ")
    if worst_moment:
        return {
            "cue": f"{fam['flagged']}",
            "observation": (
                f"Local classifier read this block as {pretty}. Its worst "
                f"moment is {worst_moment['timestamp']}s "
                f"({worst_moment.get('severity')}, error "
                f"{worst_moment.get('kinetic_error')}), measured from the "
                "latent trajectory rather than observed."),
        }
    return {
        "cue": fam["clean"],
        "observation": (
            f"Local classifier read this block as {pretty}. The trajectory "
            "stayed consistent throughout, with no moment standing out."),
    }


def chapter_breakdown(label, worst_moment, reps, region=None):
    """The measured half of the coaching card, for when Gemini has not landed.

    WHAT THIS DELIBERATELY DOES NOT DO
        It does not write a corrective drill. Prescribing "banded lateral
        walks" requires knowing that the knee actually caved, and nothing here
        looked at the picture -- a spike in latent kinetic error says the
        motion CHANGED, not that it was wrong. A drill invented from that
        would be a confident correction for a fault that may not exist, which
        is the worst output this system can produce. The field stays null and
        the card says the coach has not answered yet.

        What it can honestly do is state the measurement, in this session's
        own numbers, which is why nothing here is a fixed sentence.
    """
    if not worst_moment:
        n = (reps or {}).get("total_reps") or 0
        return {
            "mechanical_breakdown": (
                f"No single moment stood out across {n} rep(s): the latent "
                "trajectory stayed within its own trend for the whole block. "
                "That is consistency, measured -- not a verdict on technique, "
                "which needs the video."),
            "corrective_drill": None,
        }

    sev = worst_moment.get("severity")
    err = worst_moment.get("kinetic_error")
    ts = worst_moment.get("timestamp")
    n = (reps or {}).get("total_reps") or 0
    cad = (reps or {}).get("cadence_per_min")
    where = f" around the {region}" if region else ""
    pace = f", at a cadence of {cad}/min" if cad else ""
    return {
        "mechanical_breakdown": (
            f"At {ts}s the movement pattern departed furthest from its own "
            f"trend for this block{where} -- severity {sev}, kinetic error "
            f"{err}, across {n} rep(s){pace}. That is a measured change in "
            "trajectory, located by V-JEPA and framed by SAM. Whether the "
            "change is a fault is a judgement about the picture, and the AI "
            "coach has not returned it yet."),
        "corrective_drill": None,
    }


def synthesize(gym_read, gym_conf, dynamics, reps, segments, duration):
    """Local measurements -> a complete report in the contract's shape.

    Returns the same keys Stage D returns, so callers cannot tell the two
    apart structurally -- only by `analysis_source`, which says plainly that
    no model wrote this.
    """
    fam = FAMILIES.get(_FAMILY_OF.get(gym_read), GENERIC)
    pretty = gym_read.replace("_", " ")
    worst = dynamics.get("worst_moment") or {}
    ts = worst.get("timestamp")
    err = worst.get("kinetic_error")
    sev = worst.get("severity")

    cands = (segments[0].get("candidates") if segments else None) or []
    bbox = _pick_box(cands)
    region = _region(bbox)

    # The classifier is only worth naming when it was reasonably sure. Below
    # that, saying "unclear" is more honest than asserting a coin-flip label.
    confident = gym_conf is not None and gym_conf >= 0.35
    activity = pretty if confident else "unclear (low-confidence local read)"

    if ts is not None:
        issue = (
            f"Kinetic inflection detected at {ts}s"
            + (f" in the {region}" if region else "")
            + f", severity {sev} (error {err}). This marks where the movement "
              "departed from its own recent trend. No model reviewed the "
              "footage, so this is a location to check, not a diagnosed fault.")
        cue = (f"Detected {pretty} with a kinetic inflection at {ts}s"
               + (f" on the {region}" if region else "") + ". "
               + fam["flagged"])
    else:
        issue = ("No kinetic inflection stood out: the trajectory stayed "
                 "consistent across the clip.")
        cue = f"Detected {pretty} with an even trajectory. {fam['clean']}"

    return {
        "activity_detected": activity,
        "session_type": ("gym_reps" if reps.get("total_reps") else
                         "sports_drill"),
        "form_score": _score(dynamics, reps),
        "total_reps": reps.get("total_reps", 0),
        "best_rep": reps.get("best_rep"),
        "worst_rep": reps.get("worst_rep"),
        "phases": _phases(reps, dynamics, duration),
        "visual_proof": ({
            "timestamp": None,          # the caller stamps the measured value
            "target_object": region,
            "normalized_bbox": bbox,
            "issue_description": issue,
        } if ts is not None else None),
        "actionable_cue": cue,
        # Stated in the payload as well as in analysis_source, so a consumer
        # reading only the body still learns no model was involved.
        "synthesis_note": (
            "Generated locally from V-JEPA classification, latent-trajectory "
            "measurements and SAM segmentation. No language model was "
            "reachable. Cues are movement standards for the detected exercise, "
            "not observations of this athlete."),
    }
