// SPDX-FileCopyrightText: 2026 Gundu Labs
// SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

#[derive(Default)]
struct VerifyFrameBudget {
    rgb: u32,
    ir: u32,
}

impl VerifyFrameBudget {
    fn frames(&mut self, spectrum: Spectrum) -> &mut u32 {
        match spectrum {
            Spectrum::Rgb => &mut self.rgb,
            Spectrum::Ir => &mut self.ir,
        }
    }

    /// Reset this camera's budget; return whether shared deadlines should reset.
    fn phase_started(&mut self, spectrum: Spectrum, serial_capture: bool) -> bool {
        *self.frames(spectrum) = 0;
        serial_capture
    }

    fn frame_spent(&mut self, spectrum: Spectrum, max_frames: u32) -> bool {
        let frames = self.frames(spectrum);
        *frames = frames.saturating_add(1);
        *frames > max_frames
    }
}

pub(super) enum VerifyMsg {
    PhaseStarted(Spectrum),
    Diagnostic(String),
    Status(Spectrum, CaptureStatus, Option<ndarray::Array1<f32>>, f64),
    Success(Spectrum, ndarray::Array1<f32>),
    Duress(Spectrum),
    /// A matched face cleared the smile requirement, so a later failure is not the smile's.
    Smiled,
    Unavailable(Spectrum, String),
    Error(String),
}

pub(super) fn should_yield_rgb_to_ir(policy: &str, run_ir: bool, status: CaptureStatus) -> bool {
    run_ir && !matches!(policy, "or" | "and") && matches!(status, CaptureStatus::TooDark)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyGiveUp {
    NoFace,
    NoUsableFrame,
}

impl VerifyGiveUp {
    fn reason(self) -> String {
        match self {
            Self::NoFace => format!(
                "giving up after {}s without a detected face",
                VERIFY_NO_FACE_TIMEOUT.as_secs()
            ),
            Self::NoUsableFrame => format!(
                "giving up after {}s without a usable frame",
                VERIFY_NO_USABLE_TIMEOUT.as_secs()
            ),
        }
    }
}

/// Bound both missing faces and unusable frames: `Clipped` and `Ready` refresh only
/// the face deadline, so framing problems cannot keep verification alive forever.
pub(super) fn verify_give_up(since_face: Duration, since_usable: Duration) -> Option<VerifyGiveUp> {
    if since_face >= VERIFY_NO_FACE_TIMEOUT {
        return Some(VerifyGiveUp::NoFace);
    }
    if since_usable >= VERIFY_NO_USABLE_TIMEOUT {
        return Some(VerifyGiveUp::NoUsableFrame);
    }
    None
}

pub(super) fn ir_waits_for_rgb(run_rgb: bool, serial_capture: bool) -> bool {
    run_rgb && serial_capture
}

pub(super) fn rgb_yields_camera_on_budget(run_ir: bool, serial_capture: bool) -> bool {
    run_ir && serial_capture
}

pub(super) fn hybrid_auth_passed(
    policy: &str,
    run_rgb: bool,
    run_ir: bool,
    rgb_attempted: bool,
    rgb_status: CaptureStatus,
    rgb_success: bool,
    ir_success: bool,
) -> bool {
    match (run_rgb, run_ir) {
        (true, true) => match policy {
            "or" => rgb_success || ir_success,
            "and" => rgb_success && ir_success,
            // Fallback policy: both spectra must pass unless RGB ran and was too dark to judge.
            _ => {
                if !rgb_attempted {
                    rgb_success && ir_success
                } else if matches!(rgb_status, CaptureStatus::TooDark) {
                    ir_success
                } else {
                    rgb_success && ir_success
                }
            }
        },
        (true, false) => rgb_success,
        (false, true) => ir_success,
        (false, false) => false,
    }
}

// Losing RGB is no weaker than covering its lens, which already hands fallback_on_dark to IR.
pub(super) fn ir_continues_without_rgb(policy: &str, run_ir: bool) -> bool {
    run_ir && policy != "and"
}

pub(super) fn rgb_continues_without_ir(policy: &str, run_rgb: bool) -> bool {
    run_rgb && policy != "and"
}

pub(super) fn auth_streams(
    rgb_device: &str,
    ir_device: &str,
    has_rgb_templates: bool,
    has_ir_templates: bool,
) -> (bool, bool) {
    (
        !rgb_device.is_empty() && has_rgb_templates,
        !ir_device.is_empty() && has_ir_templates,
    )
}

pub(super) fn and_policy_unsatisfiable(
    policy: &str,
    rgb_device: &str,
    ir_device: &str,
    run_rgb: bool,
    run_ir: bool,
) -> bool {
    if policy != "and" || rgb_device.is_empty() || ir_device.is_empty() {
        return false;
    }
    !(run_rgb && run_ir)
}

pub(super) fn process_frame_sync(
    checker: &mut FaceChecker,
    recognizer: &mut FaceRecognizer,
    frame: &Mat,
    keep_liveness_frame: bool,
) -> anyhow::Result<(CaptureStatus, Option<FaceData>)> {
    let (status, result_opt) = checker.capture_status(frame)?;

    if status != CaptureStatus::Usable {
        return Ok((status, None));
    }

    if let Some(res) = result_opt {
        let Some(kpss) = res.kpss else {
            return Ok((status, None));
        };
        let Some(mat_rgb) = res.mat_rgb else {
            return Ok((status, None));
        };

        let aligned = align_face(&mat_rgb, &kpss, 0)?;
        let embedding = recognizer.get_embedding(&aligned)?;

        let Some((x1, y1, x2, y2)) = res.bbox else {
            return Ok((status, None));
        };
        let liveness_frame = if keep_liveness_frame {
            Some(mat_rgb)
        } else {
            None
        };
        Ok((
            status,
            Some(FaceData {
                embedding,
                liveness_frame,
                frame_size: (res.width, res.height),
                bbox: [x1, y1, x2, y2],
                kpss,
                yaw: res.yaw,
                pitch: res.pitch,
            }),
        ))
    } else {
        Ok((status, None))
    }
}

// Strip the square padding first, since its black bars read as a replay bezel
// to the anti-spoof model.
/// Whether one camera's smile gate is held right now. Both cameras watch the same face, so a
/// smile either one holds counts for both: RGB landmarks read smiles weaker than IR. Dropping
/// the flag clears it, so a camera that stopped can no longer vouch for a smile.
pub(super) struct SharedSmile(Arc<std::sync::atomic::AtomicBool>);

impl SharedSmile {
    pub(super) fn pair() -> (
        Self,
        Self,
        Arc<std::sync::atomic::AtomicBool>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let rgb = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ir = Arc::new(std::sync::atomic::AtomicBool::new(false));
        (Self(rgb.clone()), Self(ir.clone()), rgb, ir)
    }

    fn set(&self, held: bool) {
        self.0.store(held, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for SharedSmile {
    fn drop(&mut self) {
        self.set(false);
    }
}

/// A frame may pass the smile requirement when its own camera, or the other one, holds a smile.
pub(super) fn smile_counts(
    own: &SharedSmile,
    held: bool,
    other: &std::sync::atomic::AtomicBool,
) -> bool {
    own.set(held);
    held || other.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SmileState {
    Waiting,
    Arming(Instant),
    Held,
}

/// Holds a matched face back until it smiles, so a glance at the camera cannot unlock. This
/// guards against accidents, not attackers, so it uses hysteresis rather than a hard cut: the
/// ratio must stay at `start_ratio` for `HOLD`, after which it only has to stay above
/// `keep_ratio`. Up to `GRACE_FRAMES` consecutive frames below either threshold are forgiven,
/// since the landmarks jitter. While not held it reports the ratio a few times a second, which
/// `gaze auth --verbose` shows live for calibration.
pub(super) struct SmileGate {
    start_ratio: f32,
    keep_ratio: f32,
    state: SmileState,
    misses: u32,
    best: Option<f32>,
    last_report: Option<Instant>,
}

impl SmileGate {
    const HOLD: Duration = Duration::from_millis(500);
    const GRACE_FRAMES: u32 = 2;
    const REPORT_INTERVAL: Duration = Duration::from_millis(300);

    pub(super) fn new(start_ratio: f32, keep_ratio: f32) -> Self {
        Self {
            start_ratio,
            keep_ratio,
            state: SmileState::Waiting,
            misses: 0,
            best: None,
            last_report: None,
        }
    }

    /// Advances the state by one matched frame and returns whether the smile is held now.
    pub(super) fn check(
        &mut self,
        kpss: &ndarray::Array3<f32>,
        spectrum: Spectrum,
        now: Instant,
    ) -> (bool, Option<String>) {
        let ratio = gaze_vision::face::estimate_smile_ratio(kpss);
        if let Some(ratio) = ratio {
            self.best = Some(self.best.map_or(ratio, |best| best.max(ratio)));
        }
        let previous = self.state;
        self.state = self.next_state(ratio, now);
        let held = self.state == SmileState::Held;
        tracing::debug!(?spectrum, ?ratio, state = ?self.state, misses = self.misses, "Smile check");

        let changed = std::mem::discriminant(&previous) != std::mem::discriminant(&self.state);
        let due = !held
            && self
                .last_report
                .is_none_or(|last| now.duration_since(last) >= Self::REPORT_INTERVAL);
        if !changed && !due {
            return (held, None);
        }
        self.last_report = Some(now);
        let state = match self.state {
            SmileState::Held => "held".to_string(),
            SmileState::Arming(since) => format!(
                "holding {}/{} ms",
                now.duration_since(since).as_millis(),
                Self::HOLD.as_millis()
            ),
            SmileState::Waiting if previous == SmileState::Held => "dropped".to_string(),
            SmileState::Waiting => "waiting".to_string(),
        };
        let format_ratio =
            |ratio: Option<f32>| ratio.map_or_else(|| "unknown".to_string(), |r| format!("{r:.2}"));
        let message = format!(
            "{spectrum:?} face matched; smile {state}: smile_ratio={}, best={}, start={:.2}, keep={:.2}",
            format_ratio(ratio),
            format_ratio(self.best),
            self.start_ratio,
            self.keep_ratio
        );
        (held, Some(message))
    }

    fn next_state(&mut self, ratio: Option<f32>, now: Instant) -> SmileState {
        let reaches = |threshold: f32| ratio.is_some_and(|ratio| ratio >= threshold);
        let (threshold, on_miss) = match self.state {
            SmileState::Waiting => {
                return if reaches(self.start_ratio) {
                    self.misses = 0;
                    SmileState::Arming(now)
                } else {
                    SmileState::Waiting
                };
            }
            SmileState::Arming(_) => (self.start_ratio, self.state),
            SmileState::Held => (self.keep_ratio, self.state),
        };
        if reaches(threshold) {
            self.misses = 0;
            return match self.state {
                SmileState::Arming(since) if now.duration_since(since) >= Self::HOLD => {
                    SmileState::Held
                }
                state => state,
            };
        }
        self.misses += 1;
        if self.misses > Self::GRACE_FRAMES {
            self.misses = 0;
            SmileState::Waiting
        } else {
            on_miss
        }
    }
}

/// Names the gate that held a matched face back until the frame budget ran out.
pub(super) fn gate_diagnostic(smile_required: bool, smiled: bool) -> &'static str {
    if smile_required && !smiled {
        SMILE_GATE_DIAGNOSTIC
    } else {
        LIVENESS_GATE_DIAGNOSTIC
    }
}

pub(super) fn crop_liveness_face(data: &FaceData) -> anyhow::Result<image::RgbImage> {
    let mat_rgb = data
        .liveness_frame
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("liveness frame was not retained"))?;
    let rgb = mat_to_rgb(mat_rgb)?;
    let (frame_w, frame_h) = data.frame_size;
    let frame_w = frame_w.min(rgb.width()).max(1);
    let frame_h = frame_h.min(rgb.height()).max(1);
    let pad_x = (rgb.width() - frame_w) / 2;
    let pad_y = (rgb.height() - frame_h) / 2;
    let content = image::imageops::crop_imm(&rgb, pad_x, pad_y, frame_w, frame_h);
    let bbox = [
        data.bbox[0] - pad_x as f32,
        data.bbox[1] - pad_y as f32,
        data.bbox[2] - pad_x as f32,
        data.bbox[3] - pad_y as f32,
    ];
    crate::liveness::crop_face(&*content, bbox)
}

pub fn load_eye_state(
    inference: &gaze_core::config::InferenceConfig,
) -> anyhow::Result<EyeStateClassifier> {
    let path = crate::models::ensure_eye_state_model(gaze_core::config::MODELS_DIR)?;
    EyeStateClassifier::new_with_inference(&path, inference)
}

pub(super) fn eyes_closed_in_frame(
    eye_state: &Mutex<Option<EyeStateClassifier>>,
    data: &FaceData,
    threshold: f32,
) -> anyhow::Result<bool> {
    let frame = data
        .liveness_frame
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("frame was not retained for the eye-state check"))?;
    let eyes =
        eyes_from_kpss(&data.kpss).ok_or_else(|| anyhow::anyhow!("face landmarks are missing"))?;
    let rgb = mat_to_rgb(frame)?;
    let mut guard = eye_state.blocking_lock();
    let classifier = guard.as_mut().ok_or_else(|| {
        anyhow::anyhow!("duress detection is enabled but the eye-state model is unavailable")
    })?;
    let probabilities = classifier.closed_probabilities(&rgb, [eyes[0], eyes[1]])?;
    tracing::debug!(?probabilities, threshold, "Duress eye state");
    Ok(crate::duress::any_eye_closed(probabilities, threshold))
}

/// One row per enrolled face, holding (name, rgb_sim, rgb_pct, rgb_passed, ir_sim, ir_pct,
/// ir_passed) and sorted best match first.
pub(super) fn build_hybrid_scores(
    db: &UserDatabase,
    username: &str,
    rgb_threshold: f32,
    ir_threshold: f32,
    rgb_embed: Option<&ndarray::Array1<f32>>,
    ir_embed: Option<&ndarray::Array1<f32>>,
) -> Vec<(String, f64, f64, bool, f64, f64, bool)> {
    let rgb_scores = rgb_embed.and_then(|embed| {
        db.match_faces(username, embed, rgb_threshold, Spectrum::Rgb)
            .ok()
    });
    let ir_scores = ir_embed.and_then(|embed| {
        db.match_faces(username, embed, ir_threshold, Spectrum::Ir)
            .ok()
    });

    let mut final_scores = Vec::new();
    if let Ok(faces) = db.list_faces(username) {
        for (name, _, _, _) in faces {
            let (rgb_sim, rgb_pct, rgb_passed) = if let Some(ref scores) = rgb_scores {
                if let Some(score) = scores.iter().find(|s| s.0 == name) {
                    (score.1 as f64, score.2 as f64, score.3)
                } else {
                    (0.0, 0.0, false)
                }
            } else {
                (0.0, 0.0, false)
            };

            let (ir_sim, ir_pct, ir_passed) = if let Some(ref scores) = ir_scores {
                if let Some(score) = scores.iter().find(|s| s.0 == name) {
                    (score.1 as f64, score.2 as f64, score.3)
                } else {
                    (0.0, 0.0, false)
                }
            } else {
                (0.0, 0.0, false)
            };

            final_scores.push((
                name, rgb_sim, rgb_pct, rgb_passed, ir_sim, ir_pct, ir_passed,
            ));
        }
    }

    final_scores.sort_by(|a, b| {
        let a_max = a.1.max(a.4);
        let b_max = b.1.max(b.4);
        b_max
            .partial_cmp(&a_max)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    final_scores
}

impl AuthDaemon {
    pub(super) async fn start_verification(
        &self,
        ctxt: SignalEmitter<'_>,
        header: Header<'_>,
        pam_service: Option<String>,
        require_keyring: bool,
    ) -> fdo::Result<()> {
        if Self::face_barred_for_admin_prompt(pam_service.as_deref()) {
            return Err(fdo::Error::AccessDenied(
                "changing Gaze's own settings or faces needs a password, not a face match".into(),
            ));
        }
        let claim = self.check_claim(&header).await?;
        let (caller_uid, target_uid) = Self::ensure_claim_camera_access(&header, &claim).await?;
        let mut rx =
            replace_claim_task(&self.claim_state, &self.active_cancel, claim.epoch).await?;
        let config = self.settings.for_run(&mut rx).await?;
        let lid_wait = tokio::select! {
            biased;
            _ = &mut rx => return Err(fdo::Error::Failed("capture cancelled".into())),
            result = async {
                self.ensure_auth_not_aborted(&header, &config.auth).await?;
                self.lid_closed_wait(config.auth.abort_if_lid_closed).await
            } => result?,
        };
        let mut sleeping = self.sleeping.clone();
        // Ignore sleep notifications predating this request.
        sleeping.borrow_and_update();

        let resumed = self.resume_pending.load(Ordering::SeqCst);
        let resume_pending = self.resume_pending.clone();

        let username = claim.username.clone();
        let signal_destination = Self::signal_destination(&claim.sender)?;

        let sessions = self.capture_models(&config)?;
        let detector_arc = sessions.rgb.detector;
        let recognizer_rgb_arc = sessions.rgb.recognizer;
        let ir_models = sessions.ir;
        let liveness_arc = self.liveness.clone();
        let db_arc = self.db.clone();
        let rgb_threshold = config.security.rgb_threshold();
        let ir_threshold = config.security.ir_threshold();
        let active_session = active_session().await;
        let surface = Self::classify_surface(pam_service.as_deref(), active_session.as_ref());
        let lock_elapsed_ms = self.lock_elapsed_ms(active_session.as_ref()).await;
        let delay = Duration::from_millis(config.auth.start_delay_after_lock_ms(
            resumed,
            surface,
            lock_elapsed_ms,
        ));
        info!(
            service = pam_service.as_deref().unwrap_or("<unknown>"),
            ?surface,
            lock_elapsed_ms,
            "Face auth requested"
        );
        let abort_if_lid_closed = config.auth.abort_if_lid_closed;
        let sources = resolve_configured_sources(&config.cameras);
        let rgb_device = sources.rgb;
        let ir_device = sources.ir;
        let ir_node = sources.ir_node;
        let emitter_enabled = config.cameras.emitter_enabled;
        let serial_capture = sources.serial_capture;
        let liveness_cfg = config.liveness.clone();
        validate_keyring_verification(
            require_keyring,
            &liveness_cfg,
            self.db.lock().await.is_encrypted(),
        )?;
        let hybrid_policy = config.security.hybrid_policy().to_string();
        let duress_cfg = config.duress.clone();
        let smile_cfg = config.smile.clone();
        if duress_cfg.enabled && self.eye_state.lock().await.is_none() {
            let inference = config.inference.clone();
            let loading = self
                .rt_handle
                .spawn_blocking(move || load_eye_state(&inference));
            let loaded = tokio::select! {
                biased;
                _ = &mut rx => return Err(fdo::Error::Failed("capture cancelled".into())),
                loaded = loading => loaded,
            };
            match loaded {
                Ok(Ok(classifier)) => *self.eye_state.lock().await = Some(classifier),
                Ok(Err(e)) => {
                    warn!("Duress detection is enabled but the eye-state model failed to load: {e}")
                }
                Err(e) => warn!("Loading the eye-state model panicked: {e}"),
            }
        }
        let eye_state_arc = self.eye_state.clone();
        let duress_lockout = self.duress_lockout.clone();
        let duress_locked = duress_lockout.is_locked(&username);
        let camera_lease = self.camera_lease.clone();
        let conn = ctxt.connection().clone();
        let path = ctxt.path().to_owned();

        self.rt_handle.spawn(async move {
            let ctxt = match SignalEmitter::new(&conn, path) {
                Ok(emitter) => emitter.set_destination(signal_destination),
                Err(e) => {
                    error!("Failed to create signal emitter: {e}");
                    return;
                }
            };

            if duress_locked {
                info!(
                    "Face authentication for {} is locked after a duress signal until a password login",
                    username
                );
                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                return;
            }

            let db = db_arc.lock().await;
            let faces_list = db.list_faces(&username).unwrap_or_default();
            let mut has_rgb_templates = false;
            let mut has_ir_templates = false;
            for (_, _, has_rgb, has_ir) in &faces_list {
                if *has_rgb {
                    has_rgb_templates = true;
                }
                if *has_ir {
                    has_ir_templates = true;
                }
            }
            drop(db);

            let (run_rgb, run_ir) = auth_streams(
                &rgb_device,
                &ir_device,
                has_rgb_templates,
                has_ir_templates,
            );

            if !run_rgb && !run_ir {
                error!("No matching templates or cameras configured for auth");
                // Idle rather than no-face: nothing a retry could change.
                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                return;
            }

            if and_policy_unsatisfiable(&hybrid_policy, &rgb_device, &ir_device, run_rgb, run_ir) {
                error!(
                    run_rgb,
                    run_ir,
                    has_rgb_templates,
                    has_ir_templates,
                    "Hybrid policy \"and\" requires both spectra but {} has no {} templates; \
                     refusing to authenticate on one spectrum. Re-enroll to cover both.",
                    username,
                    if has_rgb_templates { "IR" } else { "RGB" }
                );
                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                return;
            }

            let mut delay = delay;
            let mut lid_wait = lid_wait;
            let mut held = false;
            loop {
                match hold_for_sleep(&mut sleeping, &mut rx, lid_wait).await {
                    SleepHold::Awake => {}
                    SleepHold::Resumed => {
                        held = true;
                        delay = Duration::from_millis(config.auth.resume_grace_ms);
                    }
                    SleepHold::Cancelled => {
                        info!("VerifyStart: cancelled while the system was suspending");
                        let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                        return;
                    }
                    SleepHold::LidClosed => {
                        warn!("Laptop lid is closed, aborting face auth");
                        let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                        return;
                    }
                }
                lid_wait = false;

                if !delay.is_zero() {
                    info!(?delay, resumed = resumed || held, ?surface, "Delaying face auth before capture");
                    tokio::select! {
                        biased;
                        _ = &mut rx => {
                            info!("VerifyStart: cancelled during start delay");
                            let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::NoFace, CaptureStatus::NoFace).await;
                            return;
                        }
                        changed = sleeping.changed() => {
                            if changed.is_err() {
                                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                                return;
                            }
                            held = true;
                            delay = Duration::from_millis(config.auth.resume_grace_ms);
                            continue;
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
                let _camera_lease_guard = match acquire_camera_lease(
                    &camera_lease,
                    &mut rx,
                    Some(&mut sleeping),
                    async {
                        if abort_if_lid_closed && Self::is_lid_closed().await {
                            warn!("Laptop lid is closed, aborting face auth");
                            return false;
                        }
                        // Check the seat last, after all other waits.
                        let allowed = Self::seat_camera_available(caller_uid, target_uid).await;
                        if !allowed {
                            warn!("VerifyStart: seat camera access changed before capture");
                        }
                        allowed
                    },
                ).await {
                    Ok(guard) => guard,
                    Err(CameraLeaseError::SleepChanged) => {
                        held = true;
                        delay = Duration::from_millis(config.auth.resume_grace_ms);
                        continue;
                    }
                    Err(reason) => {
                        info!(?reason, "VerifyStart: camera lease refused");
                        let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                        return;
                    }
                };
                if duress_lockout.is_locked(&username)
                    || rx.try_recv() != Err(oneshot::error::TryRecvError::Empty)
                {
                    let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                    return;
                }
                if !acknowledge_resume(&mut sleeping, &resume_pending) {
                    held = true;
                    delay = Duration::from_millis(config.auth.resume_grace_ms);
                    continue;
                }
                let mut run_rgb = run_rgb;
                let mut run_ir = run_ir;
                if run_ir && rgb_continues_without_ir(&hybrid_policy, run_rgb) {
                    let source = ir_device.clone();
                    let disconnected = tokio::task::spawn_blocking(move || {
                        gaze_vision::camera::camera_disconnected(&source, false)
                    })
                    .await
                    .unwrap_or(false);
                    if disconnected {
                        warn!("VerifyStart: IR camera {ir_device:?} is not connected; continuing with RGB only");
                        let _ = Self::verify_diagnostic(&ctxt, "IR camera is not connected; continuing with the RGB camera").await;
                        run_ir = false;
                    }
                }

                info!(
                    liveness_enabled = liveness_cfg.enabled,
                    liveness_threshold = liveness_cfg.effective_threshold(),
                    run_rgb = run_rgb,
                    run_ir = run_ir,
                    serial_capture = serial_capture,
                    "VerifyStart: sensing faces for user {}",
                    username
                );

                let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<VerifyMsg>(10);
                let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
                // Signals that the RGB phase released its camera so the IR thread can take it,
                // letting single-function UVC devices (e.g. Logitech Brio) run hybrid verify.
                let rgb_phase_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let (rgb_smile, ir_smile, rgb_smile_seen, ir_smile_seen) = SharedSmile::pair();
                let mut rgb_smile = Some(rgb_smile);
                let mut ir_smile = Some(ir_smile);

                let mut rgb_thread = None;
                if run_rgb {
                    let stop_clone = stop_flag.clone();
                    let tx = result_tx.clone();
                    let detector_arc = detector_arc.clone();
                    let config_clone = config.clone();
                    let recognizer_rgb_arc = recognizer_rgb_arc.clone();
                    let liveness_arc = liveness_arc.clone();
                    let db_arc = db_arc.clone();
                    let username_clone = username.clone();
                    let liveness_enabled = liveness_cfg.enabled;
                    let liveness_threshold = liveness_cfg.effective_threshold();
                    let rgb_device_clone = rgb_device.clone();
                    let rgb_phase_done_clone = rgb_phase_done.clone();
                    let hybrid_policy_clone = hybrid_policy.clone();
                    let eye_state_arc = eye_state_arc.clone();
                    let duress_enabled = duress_cfg.enabled;
                    let duress_threshold = duress_cfg.effective_closed_threshold();
                    let duress_hold = duress_cfg.effective_hold();
                    let smile_enabled = smile_cfg.enabled;
                    let (smile_start_ratio, smile_keep_ratio) = smile_cfg.effective_ratios();
                    let own_smile = rgb_smile.take().expect("RGB smile flag is taken once");
                    let other_smile = ir_smile_seen.clone();

                    rgb_thread = Some(std::thread::spawn(move || {
                        // Set on every exit path (incl. panic) once the RGB camera is released.
                        // Declared before `cam` so `cam` drops first and release precedes the signal.
                        struct RgbPhaseGuard(Arc<std::sync::atomic::AtomicBool>);
                        impl Drop for RgbPhaseGuard {
                            fn drop(&mut self) {
                                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        let _rgb_phase_guard = RgbPhaseGuard(rgb_phase_done_clone);
                        // In serial mode (IR also runs) yield the camera after a budget even
                        // without a match, so the IR spectrum can still be captured.
                        let rgb_deadline = rgb_yields_camera_on_budget(run_ir, serial_capture)
                            .then(|| Instant::now() + VERIFY_SERIAL_RGB_BUDGET);
                        let mut yielded_to_ir = false;

                        let mut cam = match Camera::open_privileged(&rgb_device_clone) {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = tx.blocking_send(VerifyMsg::Unavailable(Spectrum::Rgb, format!("RGB Camera open error: {e}")));
                                return;
                            }
                        };
                        tracing::debug!("RGB camera opened successfully at: {}", rgb_device_clone);
                        let _ = tx.blocking_send(VerifyMsg::PhaseStarted(Spectrum::Rgb));

                        let mut checker = FaceChecker::new(detector_arc, &config_clone, Spectrum::Rgb, false);
                        let mut warmup = RgbWarmupGate::new(config_clone.cameras.dark_luma_threshold);
                        // A bright background can settle the whole-frame gate while the face
                        // is still underexposed. Give face exposure its own bounded warmup.
                        let mut face_warmup = RgbWarmupGate::new(config_clone.cameras.dark_luma_threshold);
                        let mut logged_dark_stream = false;
                        let mut logged_rgb_luma_statuses = Vec::new();
                        let mut live_scores: Vec<f32> = Vec::new();
                        let mut landmark_seq: Vec<[(f32, f32); 5]> = Vec::new();
                        let mut duress = DuressTracker::new(duress_hold);
                        let mut smile = SmileGate::new(smile_start_ratio, smile_keep_ratio);
                        let mut announced_smile = false;

                        while let Some(frame) = cam.next_interruptible(&stop_clone) {
                            if stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                                break;
                            }
                            if let Some(deadline) = rgb_deadline
                                && Instant::now() >= deadline
                            {
                                // Serial mode hands the camera to the IR phase even without a
                                // match, so hybrid auth can still capture the IR spectrum.
                                yielded_to_ir = true;
                                break;
                            }

                            match warmup.classify_with_luma(&frame) {
                                (RgbFrameKind::Lit, _) => {}
                                (RgbFrameKind::WarmupDark, _) => continue,
                                (RgbFrameKind::SettledDark, luma) => {
                                    if !logged_dark_stream {
                                        let message = format!(
                                            "RGB stream remains dark after warmup: mean_luma={luma}"
                                        );
                                        info!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        logged_dark_stream = true;
                                    }
                                }
                                (RgbFrameKind::SteadyDark, luma) => {
                                    if !logged_dark_stream {
                                        let message = format!(
                                            "RGB stream never brightened: mean_luma={luma}"
                                        );
                                        info!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        logged_dark_stream = true;
                                    }
                                }
                            }

                            let (status, embed_opt) = {
                                let mut recognizer = recognizer_rgb_arc.blocking_lock();
                                match process_frame_sync(&mut checker, &mut recognizer, &frame, liveness_enabled || duress_enabled) {
                                    Ok(res) => res,
                                    Err(error) => {
                                        let message = format!("RGB inference failed: {error:#}");
                                        error!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Error(message));
                                        return;
                                    }
                                }
                            };
                            tracing::debug!("Processed RGB frame: status={:?}, embedding_extracted={}", status, embed_opt.is_some());

                            if let Some(luma) = checker.rgb_face_luma() {
                                let frame_kind = face_warmup.classify_mean_luma(luma.mean);
                                tracing::debug!(
                                    mean_luma = luma.mean,
                                    ?frame_kind,
                                    "RGB face exposure"
                                );
                                if frame_kind == RgbFrameKind::WarmupDark {
                                    continue;
                                }
                            }

                            if let Some(luma) = checker.rgb_face_luma()
                                && !logged_rgb_luma_statuses.contains(&status)
                            {
                                let message = format!(
                                    "RGB face region: mean_luma={}, rolling_mean_luma={:.1}, threshold={}, status={status:?}",
                                    luma.mean, luma.rolling_mean, luma.threshold
                                );
                                info!("{message}");
                                let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                logged_rgb_luma_statuses.push(status);
                            }

                            let latest_embed = embed_opt.as_ref().map(|d| d.embedding.clone());
                            let _ = tx.try_send(VerifyMsg::Status(Spectrum::Rgb, status, latest_embed, cam.fps()));

                            if should_yield_rgb_to_ir(&hybrid_policy_clone, run_ir, status) {
                                yielded_to_ir = true;
                                break;
                            }

                            if status == CaptureStatus::Usable && let Some(data) = embed_opt {
                                let threshold = rgb_threshold;
                                let db = db_arc.blocking_lock();
                                let scores = match db.match_faces(&username_clone, &data.embedding, threshold, Spectrum::Rgb) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        let _ = tx.blocking_send(VerifyMsg::Error(format!("DB error: {e}")));
                                        return;
                                    }
                                };
                                drop(db);

                                tracing::debug!("RGB match scores: {:?}", scores);

                                let matched = scores.iter().any(|(_, _, _, passed, _)| *passed);
                                if matched {
                                    if duress_enabled {
                                        let closed = match eyes_closed_in_frame(&eye_state_arc, &data, duress_threshold) {
                                            Ok(closed) => closed,
                                            Err(e) => {
                                                let _ = tx.blocking_send(VerifyMsg::Error(format!("Duress check failed: {e}")));
                                                return;
                                            }
                                        };
                                        if duress.observe(closed, Instant::now()) {
                                            let _ = tx.blocking_send(VerifyMsg::Duress(Spectrum::Rgb));
                                            return;
                                        }
                                        if closed {
                                            continue;
                                        }
                                    }

                                    let mut liveness_passed = true;
                                    if liveness_enabled {
                                        if let Some(eyes) = eyes_from_kpss(&data.kpss) {
                                            landmark_seq.push(eyes);
                                        }
                                        let liveness_face = match crop_liveness_face(&data) {
                                            Ok(face) => face,
                                            Err(e) => {
                                                error!("Liveness crop failed: {e}");
                                                continue;
                                            }
                                        };
                                        let mut live_guard = liveness_arc.blocking_lock();
                                        let Some(detector) = live_guard.as_mut() else {
                                            let _ = tx.blocking_send(VerifyMsg::Error(
                                                "Liveness is enabled but the anti-spoof model is unavailable".to_string(),
                                            ));
                                            return;
                                        };
                                        let live_score = match detector.live_score(&liveness_face) {
                                            Ok(score) => score,
                                            Err(e) => {
                                                let _ = tx.blocking_send(VerifyMsg::Error(format!(
                                                    "Liveness inference failed: {e}"
                                                )));
                                                return;
                                            }
                                        };
                                        drop(live_guard);
                                        live_scores.push(live_score);

                                        let model_pass = crate::liveness::liveness_passes(&live_scores, liveness_threshold as f32);
                                        let motion = crate::liveness::eye_motion_is_live(&landmark_seq, None);
                                        let confirmed_static = crate::liveness::confirmed_static(&motion);
                                        liveness_passed = model_pass && !confirmed_static;

                                        tracing::debug!(
                                            "Liveness checked: score={:?}, pass={}, motion={:?}, confirmed_static={}, overall={}",
                                            live_scores,
                                            model_pass,
                                            motion,
                                            confirmed_static,
                                            liveness_passed
                                        );
                                    }

                                    // Checked after liveness so liveness keeps learning from every matched frame.
                                    if smile_enabled {
                                        let (held, message) = smile.check(&data.kpss, Spectrum::Rgb, Instant::now());
                                        if let Some(message) = message {
                                            info!("{message}");
                                            let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        }
                                        let held = smile_counts(&own_smile, held, &other_smile);
                                        if held && !announced_smile {
                                            announced_smile = true;
                                            let _ = tx.blocking_send(VerifyMsg::Smiled);
                                        }
                                        if !held {
                                            continue;
                                        }
                                    }

                                    if liveness_passed {
                                        let _ = tx.blocking_send(VerifyMsg::Success(Spectrum::Rgb, data.embedding));
                                        return;
                                    }
                                }
                            }
                        }

                        if !yielded_to_ir && !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            // A device taken by another program only fails once it tries to stream,
                            // so this is where "already in use" surfaces.
                            let reason = cam.take_stream_error().unwrap_or_else(|| {
                                "RGB camera stream stopped unexpectedly".to_string()
                            });
                            let _ = tx.blocking_send(VerifyMsg::Unavailable(Spectrum::Rgb, reason));
                        }
                    }));
                }

                let mut ir_thread = None;
                if let Some(ir_models) = ir_models.clone().filter(|_| run_ir) {
                    let stop_clone = stop_flag.clone();
                    let tx = result_tx.clone();
                    let detector_arc = ir_models.detector;
                    let config_clone = config.clone();
                    let recognizer_ir_arc = ir_models.recognizer;
                    let db_arc = db_arc.clone();
                    let username_clone = username.clone();
                    let liveness_enabled = liveness_cfg.enabled;
                    let ir_device_clone = ir_device.clone();
                    let ir_node_clone = ir_node.clone();
                    let emitter_enabled = emitter_enabled;
                    let serial_capture = serial_capture;
                    let rgb_phase_done_clone = rgb_phase_done.clone();
                    let eye_state_arc = eye_state_arc.clone();
                    let duress_enabled = duress_cfg.enabled;
                    let duress_threshold = duress_cfg.effective_closed_threshold();
                    let duress_hold = duress_cfg.effective_hold();
                    let smile_enabled = smile_cfg.enabled;
                    let (smile_start_ratio, smile_keep_ratio) = smile_cfg.effective_ratios();
                    let own_smile = ir_smile.take().expect("IR smile flag is taken once");
                    let other_smile = rgb_smile_seen.clone();

                    ir_thread = Some(std::thread::spawn(move || {
                        // Wait for RGB to release its camera before opening IR and firing the emitter,
                        // so single-function devices keep one live stream. Bail if verify passed.
                        if ir_waits_for_rgb(run_rgb, serial_capture) {
                            while !rgb_phase_done_clone.load(std::sync::atomic::Ordering::Relaxed)
                                && !stop_clone.load(std::sync::atomic::Ordering::Relaxed)
                            {
                                std::thread::sleep(std::time::Duration::from_millis(20));
                            }
                            if stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                                return;
                            }
                        }

                        let mut emitter = EmitterGuard::engage(
                            &CameraKind::Ir { source: ir_device_clone.clone(), node: ir_node_clone.clone() },
                            emitter_enabled
                        );
                        if let Some(message) = emitter.activation_message() {
                            let _ = tx.blocking_send(VerifyMsg::Diagnostic(message.to_owned()));
                        }

                        let mut cam = match Camera::open_ir_privileged(&ir_device_clone, config_clone.cameras.ir_frame_size()) {
                            Ok(c) => c,
                            Err(e) => {
                                let _ = tx.blocking_send(VerifyMsg::Error(format!("IR Camera open error: {e}")));
                                return;
                            }
                        };
                        tracing::debug!("IR camera opened successfully at: {}", ir_device_clone);

                        let _ = tx.blocking_send(VerifyMsg::PhaseStarted(Spectrum::Ir));

                        let mut checker = FaceChecker::new(detector_arc, &config_clone, Spectrum::Ir, false);
                        let mut dark_gate = IrDarkFrameGate::new(config_clone.cameras.dark_luma_threshold);
                        let mut logged_lit_luma = false;
                        let mut logged_dark_luma = false;
                        let mut landmark_seq: Vec<[(f32, f32); 5]> = Vec::new();
                        let mut duress = DuressTracker::new(duress_hold);
                        let mut smile = SmileGate::new(smile_start_ratio, smile_keep_ratio);
                        let mut announced_smile = false;

                        while let Some(frame) = cam.next_interruptible(&stop_clone) {
                            if stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                                break;
                            }
                            emitter.stream_started();

                            let (frame_kind, luma) = dark_gate.classify_with_luma(&frame);
                            tracing::debug!(
                                mean_luma = luma,
                                threshold = config_clone.cameras.dark_luma_threshold,
                                ?frame_kind,
                                "IR frame exposure"
                            );
                            match frame_kind {
                                IrFrameKind::Lit => {
                                    if !logged_lit_luma {
                                        let message = format!(
                                            "IR stream produced a lit frame: mean_luma={luma}"
                                        );
                                        info!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        logged_lit_luma = true;
                                    }
                                }
                                // A gap between emitter strobes, not a fault, so drop it silently.
                                IrFrameKind::StrobeDark => continue,
                                IrFrameKind::EmitterDark => {
                                    if !logged_dark_luma {
                                        let message = format!(
                                            "IR stream remains dark after emitter warmup: mean_luma={luma}"
                                        );
                                        info!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        logged_dark_luma = true;
                                    }
                                    let _ = tx.try_send(VerifyMsg::Status(Spectrum::Ir, CaptureStatus::TooDark, None, cam.fps()));
                                    continue;
                                }
                            }

                            let (status, embed_opt) = {
                                let mut recognizer = recognizer_ir_arc.blocking_lock();
                                match process_frame_sync(&mut checker, &mut recognizer, &frame, duress_enabled) {
                                    Ok(res) => res,
                                    Err(error) => {
                                        let message = format!("IR inference failed: {error:#}");
                                        error!("{message}");
                                        let _ = tx.blocking_send(VerifyMsg::Error(message));
                                        return;
                                    }
                                }
                            };
                            tracing::debug!("Processed IR frame: status={:?}, embedding_extracted={}", status, embed_opt.is_some());

                            let latest_embed = embed_opt.as_ref().map(|d| d.embedding.clone());
                            let _ = tx.try_send(VerifyMsg::Status(Spectrum::Ir, status, latest_embed, cam.fps()));

                            if status == CaptureStatus::Usable && let Some(data) = embed_opt {
                                let threshold = ir_threshold;
                                let db = db_arc.blocking_lock();
                                let scores = match db.match_faces(&username_clone, &data.embedding, threshold, Spectrum::Ir) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        let _ = tx.blocking_send(VerifyMsg::Error(format!("DB error: {e}")));
                                        return;
                                    }
                                };
                                drop(db);

                                tracing::debug!("IR match scores: {:?}", scores);

                                let matched = scores.iter().any(|(_, _, _, passed, _)| *passed);
                                if matched {
                                    if duress_enabled {
                                        let closed = match eyes_closed_in_frame(&eye_state_arc, &data, duress_threshold) {
                                            Ok(closed) => closed,
                                            Err(e) => {
                                                let _ = tx.blocking_send(VerifyMsg::Error(format!("Duress check failed: {e}")));
                                                return;
                                            }
                                        };
                                        if duress.observe(closed, Instant::now()) {
                                            let _ = tx.blocking_send(VerifyMsg::Duress(Spectrum::Ir));
                                            return;
                                        }
                                        if closed {
                                            continue;
                                        }
                                    }

                                    let mut liveness_passed = true;
                                    if liveness_enabled {
                                        if let Some(eyes) = eyes_from_kpss(&data.kpss) {
                                            landmark_seq.push(eyes);
                                        }
                                        let motion = crate::liveness::eye_motion_is_live(&landmark_seq, None);
                                        liveness_passed = crate::liveness::motion_confirms_live(
                                            &motion,
                                            crate::liveness::MIN_MOTION_PAIRS,
                                        );

                                        tracing::debug!(
                                            "Liveness checked (IR): motion={:?}, overall={}",
                                            motion,
                                            liveness_passed
                                        );
                                    }

                                    // Checked after liveness so liveness keeps learning from every matched frame.
                                    if smile_enabled {
                                        let (held, message) = smile.check(&data.kpss, Spectrum::Ir, Instant::now());
                                        if let Some(message) = message {
                                            info!("{message}");
                                            let _ = tx.blocking_send(VerifyMsg::Diagnostic(message));
                                        }
                                        let held = smile_counts(&own_smile, held, &other_smile);
                                        if held && !announced_smile {
                                            announced_smile = true;
                                            let _ = tx.blocking_send(VerifyMsg::Smiled);
                                        }
                                        if !held {
                                            continue;
                                        }
                                    }

                                    if liveness_passed {
                                        let _ = tx.blocking_send(VerifyMsg::Success(Spectrum::Ir, data.embedding));
                                        return;
                                    }
                                }
                            }
                        }

                        if !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            let reason = cam.take_stream_error().unwrap_or_else(|| {
                                "IR camera stream stopped unexpectedly".to_string()
                            });
                            let _ = tx.blocking_send(VerifyMsg::Error(reason));
                        }
                    }));
                }

                drop(result_tx);

                let mut last_emitted_status: Option<CaptureStatus> = None;
                let mut suspended = false;
                let mut rgb_status = CaptureStatus::Unused;
                let mut ir_status = CaptureStatus::Unused;
                let mut rgb_attempted = false;
                let mut dark_since: Option<Instant> = None;
                let mut last_face_at = Instant::now();
                let mut last_usable_at = Instant::now();
                let mut frame_budget = VerifyFrameBudget::default();
                let mut smiled = false;

                let mut rgb_success_embed = None;
                let mut ir_success_embed = None;
                let mut rgb_latest_embed = None;
                let mut ir_latest_embed = None;

                macro_rules! hybrid_scores {
                    () => {{
                        let db = db_arc.lock().await;
                        let final_scores = build_hybrid_scores(
                            &db,
                            &username,
                            rgb_threshold,
                            ir_threshold,
                            rgb_success_embed.as_ref().or(rgb_latest_embed.as_ref()),
                            ir_success_embed.as_ref().or(ir_latest_embed.as_ref()),
                        );
                        drop(db);
                        final_scores
                    }};
                }

                macro_rules! emit_verify_with_scores {
                    ($result:expr) => {{
                        let final_scores = hybrid_scores!();
                        let _ = Self::verify_status(&ctxt, $result, final_scores, rgb_status, ir_status).await;
                    }};
                }

                macro_rules! finish_if_auth_passed {
                    () => {{
                        if hybrid_auth_passed(
                            &hybrid_policy,
                            run_rgb,
                            run_ir,
                            rgb_attempted,
                            rgb_status,
                            rgb_success_embed.is_some(),
                            ir_success_embed.is_some(),
                        ) {
                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            emit_verify_with_scores!(VerifyResult::VerifyMatch);
                            true
                        } else {
                            false
                        }
                    }};
                }

                loop {
                    tokio::select! {
                        biased;
                        _ = &mut rx => {
                            info!("VerifyStart: cancelled");
                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            // Report the camera as idle, not as a rejection: a cancelled attempt
                            // never decided anything, and a rejection counts toward lockout.
                            let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                            break;
                        }
                        true = sleep_reaches(&mut sleeping, true) => {
                            info!("VerifyStart: system is suspending, pausing until resume");
                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            suspended = true;
                            break;
                        }
                        _ = tokio::time::sleep(VERIFY_WATCHDOG_POLL) => {
                            if let Some(give_up) = verify_give_up(last_face_at.elapsed(), last_usable_at.elapsed()) {
                                info!("VerifyStart: {}", give_up.reason());
                                stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), rgb_status, ir_status).await;
                                break;
                            }
                        }
                        msg_opt = result_rx.recv() => {
                            let Some(msg) = msg_opt else {
                                warn!("VerifyStart: all capture threads exited without a result");
                                stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), rgb_status, ir_status).await;
                                break;
                            };
                            match msg {
                                VerifyMsg::PhaseStarted(spectrum) => {
                                    if frame_budget.phase_started(spectrum, serial_capture) {
                                        last_face_at = Instant::now();
                                        last_usable_at = Instant::now();
                                        dark_since = None;
                                    }
                                }
                                VerifyMsg::Diagnostic(message) => {
                                    let _ = Self::verify_diagnostic(&ctxt, &message).await;
                                }
                                VerifyMsg::Smiled => smiled = true,
                                VerifyMsg::Status(spectrum, status, embed_opt, fps) => {
                                    let has_face = embed_opt.is_some();
                                    match spectrum {
                                        Spectrum::Rgb => {
                                            rgb_status = status;
                                            rgb_attempted = true;
                                            if let Some(embed) = embed_opt {
                                                rgb_latest_embed = Some(embed);
                                            }
                                        }
                                        Spectrum::Ir => {
                                            ir_status = status;
                                            if let Some(embed) = embed_opt {
                                                ir_latest_embed = Some(embed);
                                            }
                                        }
                                    }

                                    if status.indicates_face() {
                                        last_face_at = Instant::now();
                                    }

                                    if has_face {
                                        last_usable_at = Instant::now();
                                        if frame_budget.frame_spent(spectrum, liveness_cfg.effective_max_frames(fps)) {
                                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                            let final_scores = hybrid_scores!();
                                            let matched = final_scores
                                                .iter()
                                                .any(|face| face.3 || face.6);
                                            info!(matched, "VerifyStart: frame budget spent");
                                            if matched {
                                                let _ = Self::verify_diagnostic(&ctxt, gate_diagnostic(smile_cfg.enabled, smiled)).await;
                                            }
                                            let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, final_scores, rgb_status, ir_status).await;
                                            break;
                                        }
                                    }

                                    Self::emit_effective_face_status(
                                        &ctxt,
                                        &mut last_emitted_status,
                                        rgb_status,
                                        ir_status,
                                    ).await;

                                    let both_dark = match (run_rgb, run_ir) {
                                        (true, true) => rgb_status == CaptureStatus::TooDark && ir_status == CaptureStatus::TooDark,
                                        (true, false) => rgb_status == CaptureStatus::TooDark,
                                        (false, true) => ir_status == CaptureStatus::TooDark,
                                        (false, false) => false,
                                    };

                                    if both_dark {
                                        let started = *dark_since.get_or_insert_with(Instant::now);
                                        if started.elapsed() >= VERIFY_TOO_DARK_TIMEOUT {
                                            info!(
                                                "VerifyStart: giving up after {}ms of dark frames",
                                                VERIFY_TOO_DARK_TIMEOUT.as_millis()
                                            );
                                            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                            let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), rgb_status, ir_status).await;
                                            break;
                                        }
                                    } else {
                                        dark_since = None;
                                    }

                                    if let Some(give_up) = verify_give_up(last_face_at.elapsed(), last_usable_at.elapsed()) {
                                        info!("VerifyStart: {}", give_up.reason());
                                        stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                        let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), rgb_status, ir_status).await;
                                        break;
                                    }

                                    if finish_if_auth_passed!() {
                                        break;
                                    }
                                }
                                VerifyMsg::Success(spectrum, embedding) => {
                                    match spectrum {
                                        Spectrum::Rgb => {
                                            rgb_success_embed = Some(embedding);
                                            rgb_attempted = true;
                                        }
                                        Spectrum::Ir => ir_success_embed = Some(embedding),
                                    }

                                    if finish_if_auth_passed!() {
                                        break;
                                    }
                                }
                                VerifyMsg::Duress(spectrum) => {
                                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    info!(?spectrum, "Duress signal held; locking face authentication for {} until a password login", username);
                                    if let Err(e) = duress_lockout.lock(&username) {
                                        error!("Failed to persist the duress lockout, holding it in memory: {e}");
                                    }
                                    let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), rgb_status, ir_status).await;
                                    break;
                                }
                                VerifyMsg::Unavailable(Spectrum::Rgb, reason)
                                    if run_rgb && ir_continues_without_rgb(&hybrid_policy, run_ir) =>
                                {
                                    warn!("VerifyStart: {reason}; continuing with IR only");
                                    let _ = Self::verify_diagnostic(&ctxt, &format!("{reason}; continuing with the IR camera")).await;
                                    run_rgb = false;
                                    rgb_status = CaptureStatus::Unused;
                                    if finish_if_auth_passed!() {
                                        break;
                                    }
                                }
                                VerifyMsg::Error(e) | VerifyMsg::Unavailable(_, e) => {
                                    error!("VerifyStart loop error: {e}");
                                    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    // The verdict alone reads as a face that was not found.
                                    let _ = Self::verify_diagnostic(&ctxt, &e).await;
                                    // Idle, not rejected: the run broke off instead of deciding, and
                                    // a hardware failure must not count against the lockout budget.
                                    let _ = Self::verify_status(&ctxt, VerifyResult::VerifyNoMatch, Vec::new(), CaptureStatus::Unused, CaptureStatus::Unused).await;
                                    break;
                                }
                            }
                        }
                    }
                }

                // A producer may be blocked in blocking_send when cancellation wins.
                // Closing the receiver releases it so joining cannot deadlock.
                drop(result_rx);
                let _ = tokio::task::spawn_blocking(move || {
                    if let Some(t) = rgb_thread {
                        let _ = t.join();
                    }
                    if let Some(t) = ir_thread {
                        let _ = t.join();
                    }
                })
                .await;

                if !suspended {
                    break;
                }
                held = true;
                delay = Duration::from_millis(config.auth.resume_grace_ms);
            }
            drop(config);
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaze_core::dbus::CaptureStatus;

    fn mouth_landmarks(mouth_half_width: f32) -> ndarray::Array3<f32> {
        let points = [
            (40.0, 50.0),
            (80.0, 50.0),
            (60.0, 70.0),
            (60.0 - mouth_half_width, 90.0),
            (60.0 + mouth_half_width, 90.0),
        ];
        ndarray::Array3::from_shape_fn((1, 5, 2), |(_, point, axis)| {
            if axis == 0 {
                points[point].0
            } else {
                points[point].1
            }
        })
    }

    fn ms(start: Instant, millis: u64) -> Instant {
        start + Duration::from_millis(millis)
    }

    #[test]
    fn smile_must_hold_the_start_ratio_before_it_counts() {
        let mut gate = SmileGate::new(0.90, 0.80);
        let t = Instant::now();
        let neutral = mouth_landmarks(15.0); // 0.75
        let smile = mouth_landmarks(18.5); // 0.925

        let (held, message) = gate.check(&neutral, Spectrum::Rgb, t);
        assert!(!held);
        assert!(message.unwrap().contains("smile waiting: smile_ratio=0.75"));

        let (held, message) = gate.check(&smile, Spectrum::Rgb, ms(t, 100));
        assert!(!held);
        assert!(message.unwrap().contains("smile holding 0/500 ms"));
        let (held, _) = gate.check(&smile, Spectrum::Rgb, ms(t, 400));
        assert!(!held, "arming has not lasted 500 ms yet");
        let (held, message) = gate.check(&smile, Spectrum::Rgb, ms(t, 600));
        assert!(held);
        assert!(message.unwrap().contains("smile held"));
    }

    #[test]
    fn a_held_smile_only_needs_the_keep_ratio_and_forgives_short_dips() {
        let mut gate = SmileGate::new(0.90, 0.80);
        let t = Instant::now();
        let smile = mouth_landmarks(18.5); // 0.925
        let relaxed = mouth_landmarks(17.0); // 0.85
        let neutral = mouth_landmarks(15.0); // 0.75
        gate.check(&smile, Spectrum::Ir, t);
        assert!(gate.check(&smile, Spectrum::Ir, ms(t, 500)).0);

        assert!(
            gate.check(&relaxed, Spectrum::Ir, ms(t, 600)).0,
            "above keep stays held"
        );
        assert!(
            gate.check(&neutral, Spectrum::Ir, ms(t, 633)).0,
            "first dip is forgiven"
        );
        assert!(
            gate.check(&neutral, Spectrum::Ir, ms(t, 666)).0,
            "second dip is forgiven"
        );
        assert!(
            gate.check(&relaxed, Spectrum::Ir, ms(t, 700)).0,
            "recovery resets the grace"
        );
        gate.check(&neutral, Spectrum::Ir, ms(t, 733));
        gate.check(&neutral, Spectrum::Ir, ms(t, 766));
        let (held, message) = gate.check(&neutral, Spectrum::Ir, ms(t, 800));
        assert!(!held, "a third consecutive dip drops the smile");
        assert!(message.unwrap().contains("smile dropped"));

        let (held, _) = gate.check(&smile, Spectrum::Ir, ms(t, 900));
        assert!(!held, "a dropped smile has to arm again");
    }

    #[test]
    fn arming_forgives_short_dips_but_restarts_after_a_real_one() {
        let mut gate = SmileGate::new(0.90, 0.80);
        let t = Instant::now();
        let smile = mouth_landmarks(18.5);
        let relaxed = mouth_landmarks(17.0);
        gate.check(&smile, Spectrum::Rgb, t);
        gate.check(&relaxed, Spectrum::Rgb, ms(t, 100));
        gate.check(&relaxed, Spectrum::Rgb, ms(t, 200));
        assert!(
            gate.check(&smile, Spectrum::Rgb, ms(t, 500)).0,
            "two dips keep the arming time"
        );

        let mut gate = SmileGate::new(0.90, 0.80);
        gate.check(&smile, Spectrum::Rgb, t);
        for step in 1..=3 {
            gate.check(&relaxed, Spectrum::Rgb, ms(t, step * 50));
        }
        assert!(
            !gate.check(&smile, Spectrum::Rgb, ms(t, 500)).0,
            "three dips restart arming"
        );
    }

    #[test]
    fn smile_gate_reports_progress_at_a_steady_pace() {
        let mut gate = SmileGate::new(0.90, 0.80);
        let t = Instant::now();
        let neutral = mouth_landmarks(15.0);
        assert!(gate.check(&neutral, Spectrum::Rgb, t).1.is_some());
        assert!(
            gate.check(&mouth_landmarks(16.0), Spectrum::Rgb, ms(t, 100))
                .1
                .is_none()
        );
        let message = gate.check(&neutral, Spectrum::Rgb, ms(t, 300)).1.unwrap();
        assert!(message.contains("smile_ratio=0.75, best=0.80, start=0.90, keep=0.80"));
    }

    #[test]
    fn either_camera_holding_a_smile_counts_for_both_until_it_stops() {
        let (rgb, ir, rgb_seen, ir_seen) = SharedSmile::pair();

        assert!(!smile_counts(&rgb, false, &ir_seen));
        assert!(smile_counts(&ir, true, &rgb_seen), "IR holds its own smile");
        assert!(
            smile_counts(&rgb, false, &ir_seen),
            "RGB borrows the smile IR holds"
        );

        assert!(!smile_counts(&ir, false, &rgb_seen));
        assert!(
            !smile_counts(&rgb, false, &ir_seen),
            "a dropped smile stops counting"
        );

        smile_counts(&ir, true, &rgb_seen);
        drop(ir);
        assert!(
            !smile_counts(&rgb, false, &ir_seen),
            "a stopped camera cannot vouch"
        );
    }

    #[test]
    fn a_budget_spent_waiting_for_a_smile_blames_the_smile_not_liveness() {
        assert_eq!(gate_diagnostic(true, false), SMILE_GATE_DIAGNOSTIC);
        assert_eq!(gate_diagnostic(true, true), LIVENESS_GATE_DIAGNOSTIC);
        assert_eq!(gate_diagnostic(false, false), LIVENESS_GATE_DIAGNOSTIC);
    }

    #[test]
    fn smile_gate_refuses_unreadable_landmarks() {
        let mut gate = SmileGate::new(0.6, 0.6);
        let (smiling, _) = gate.check(
            &ndarray::Array3::zeros((1, 5, 2)),
            Spectrum::Ir,
            Instant::now(),
        );
        assert!(!smiling);
    }

    #[test]
    fn parallel_cameras_have_independent_frame_budgets() {
        let mut budget = VerifyFrameBudget::default();
        for _ in 0..60 {
            assert!(!budget.frame_spent(Spectrum::Rgb, 60));
            assert!(!budget.frame_spent(Spectrum::Ir, 60));
        }
        assert!(budget.frame_spent(Spectrum::Rgb, 60));
        assert!(budget.frame_spent(Spectrum::Ir, 60));
    }

    #[test]
    fn late_rgb_start_preserves_ir_frames_and_shared_deadlines() {
        let mut budget = VerifyFrameBudget::default();
        for _ in 0..60 {
            assert!(!budget.frame_spent(Spectrum::Ir, 60));
        }
        assert!(!budget.phase_started(Spectrum::Rgb, false));
        assert!(budget.frame_spent(Spectrum::Ir, 60));
        assert!(!budget.frame_spent(Spectrum::Rgb, 60));
    }

    #[test]
    fn late_ir_start_preserves_rgb_frames_and_shared_deadlines() {
        let mut budget = VerifyFrameBudget::default();
        assert!(!budget.frame_spent(Spectrum::Rgb, 1));
        assert!(!budget.phase_started(Spectrum::Ir, false));
        assert!(budget.frame_spent(Spectrum::Rgb, 1));
    }

    #[test]
    fn serial_handoff_refreshes_deadlines_and_only_the_new_phase_budget() {
        let mut budget = VerifyFrameBudget::default();
        assert!(!budget.frame_spent(Spectrum::Rgb, 1));
        assert!(!budget.frame_spent(Spectrum::Ir, 1));
        assert!(budget.phase_started(Spectrum::Ir, true));
        assert!(!budget.frame_spent(Spectrum::Ir, 1));
        assert!(budget.frame_spent(Spectrum::Rgb, 1));
    }

    #[test]
    fn each_camera_uses_its_own_fps_adjusted_limit() {
        let mut budget = VerifyFrameBudget::default();
        for _ in 0..30 {
            assert!(!budget.frame_spent(Spectrum::Rgb, 30));
            assert!(!budget.frame_spent(Spectrum::Ir, 60));
            assert!(!budget.frame_spent(Spectrum::Ir, 60));
        }
        assert!(budget.frame_spent(Spectrum::Rgb, 30));
        assert!(budget.frame_spent(Spectrum::Ir, 60));
    }

    #[test]
    fn watchdog_polls_faster_than_the_timeouts_it_guards() {
        use super::{
            VERIFY_NO_FACE_TIMEOUT, VERIFY_NO_USABLE_TIMEOUT, VERIFY_TOO_DARK_TIMEOUT,
            VERIFY_WATCHDOG_POLL,
        };

        assert!(VERIFY_WATCHDOG_POLL < VERIFY_NO_FACE_TIMEOUT);
        assert!(VERIFY_WATCHDOG_POLL < VERIFY_TOO_DARK_TIMEOUT);
        assert!(VERIFY_WATCHDOG_POLL < VERIFY_NO_USABLE_TIMEOUT);
        assert!(!VERIFY_WATCHDOG_POLL.is_zero());
    }

    #[test]
    fn a_warming_camera_still_reports_dark_before_the_no_face_deadline() {
        use super::{RgbWarmupGate, VERIFY_NO_FACE_TIMEOUT, VERIFY_TOO_DARK_TIMEOUT};

        assert!(RgbWarmupGate::WARMUP + VERIFY_TOO_DARK_TIMEOUT < VERIFY_NO_FACE_TIMEOUT);
    }

    // A backstop that fired first would report a timeout for a run the daemon had already decided.
    #[test]
    fn every_daemon_deadline_lands_inside_the_client_backstop() {
        use super::{VERIFY_NO_FACE_TIMEOUT, VERIFY_NO_USABLE_TIMEOUT, VERIFY_TOO_DARK_TIMEOUT};

        let backstop = gaze_core::dbus::VERIFY_CLIENT_TIMEOUT;
        for deadline in [
            VERIFY_NO_FACE_TIMEOUT,
            VERIFY_NO_USABLE_TIMEOUT,
            VERIFY_TOO_DARK_TIMEOUT,
        ] {
            assert!(
                deadline < backstop,
                "{deadline:?} must fire before {backstop:?}"
            );
        }
    }

    // Before the usable deadline existed this case ran forever.
    #[test]
    fn a_face_that_never_becomes_usable_still_hits_a_deadline() {
        use super::{VERIFY_NO_USABLE_TIMEOUT, VerifyGiveUp, verify_give_up};

        let never_stale = std::time::Duration::ZERO;
        assert_eq!(verify_give_up(never_stale, never_stale), None);
        assert_eq!(
            verify_give_up(never_stale, VERIFY_NO_USABLE_TIMEOUT),
            Some(VerifyGiveUp::NoUsableFrame)
        );
    }

    #[test]
    fn a_vanished_face_still_reports_the_no_face_deadline_first() {
        use super::{
            VERIFY_NO_FACE_TIMEOUT, VERIFY_NO_USABLE_TIMEOUT, VerifyGiveUp, verify_give_up,
        };

        assert_eq!(
            verify_give_up(VERIFY_NO_FACE_TIMEOUT, VERIFY_NO_USABLE_TIMEOUT),
            Some(VerifyGiveUp::NoFace)
        );
        assert_eq!(
            verify_give_up(VERIFY_NO_FACE_TIMEOUT, std::time::Duration::ZERO),
            Some(VerifyGiveUp::NoFace)
        );
    }

    // Every status that counts as a face but carries no embedding relies on the usable deadline.
    #[test]
    fn framing_hints_and_ready_count_as_a_face_without_being_usable() {
        for status in [
            CaptureStatus::Clipped,
            CaptureStatus::NotCentered,
            CaptureStatus::TooFar,
            CaptureStatus::TooClose,
            CaptureStatus::Ready,
        ] {
            assert!(
                status.indicates_face(),
                "{status:?} refreshes the no-face deadline"
            );
            assert_ne!(
                status,
                CaptureStatus::Usable,
                "{status:?} never reaches the embedding path in process_frame_sync"
            );
        }
    }

    #[test]
    fn a_stalled_capture_stream_still_reaches_the_no_face_deadline() {
        use super::VERIFY_WATCHDOG_POLL;

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (_retained_tx, mut rx) = tokio::sync::mpsc::channel::<u32>(10);
            let deadline = std::time::Duration::from_millis(500);
            let started = std::time::Instant::now();
            let mut gave_up = false;

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(VERIFY_WATCHDOG_POLL) => {
                        if started.elapsed() >= deadline {
                            gave_up = true;
                            break;
                        }
                    }
                    msg = rx.recv() => {
                        if msg.is_none() {
                            break;
                        }
                    }
                }
            }

            assert!(gave_up);
            assert!(started.elapsed() < deadline * 4);
        });
    }

    #[test]
    fn liveness_crop_excludes_square_padding_bars() {
        use super::{FaceData, crop_liveness_face};
        use opencv::core::{CV_8UC3, Mat, Scalar};

        let frame = Mat::new_rows_cols_with_default(480, 640, CV_8UC3, Scalar::all(255.0)).unwrap();
        let padded = gaze_vision::detect::FaceDetector::pad_to_square(&frame).unwrap();

        let data = FaceData {
            embedding: ndarray::Array1::zeros(512),
            liveness_frame: Some(padded),
            frame_size: (640, 480),
            // The 2.7x crop margin around this bbox reaches both padding bars.
            bbox: [220.0, 200.0, 420.0, 440.0],
            kpss: ndarray::Array3::zeros((1, 5, 2)),
            yaw: 0.0,
            pitch: 0.0,
        };

        let crop = crop_liveness_face(&data).unwrap();
        assert!(
            crop.pixels().all(|p| p.0 == [255, 255, 255]),
            "liveness crop must not contain padding pixels"
        );
    }

    #[test]
    fn authentication_starts_only_streams_with_a_camera_and_matching_templates() {
        assert_eq!(
            auth_streams("primary", "/dev/video2", true, true),
            (true, true)
        );
        assert_eq!(
            auth_streams("primary", "/dev/video2", true, false),
            (true, false)
        );
        assert_eq!(
            auth_streams("primary", "/dev/video2", false, true),
            (false, true)
        );
        assert_eq!(auth_streams("", "/dev/video2", true, true), (false, true));
        assert_eq!(auth_streams("primary", "", true, true), (true, false));
        assert_eq!(auth_streams("", "", true, true), (false, false));
    }

    #[test]
    fn independent_camera_functions_capture_both_spectra_at_once() {
        assert!(!ir_waits_for_rgb(true, false));
        assert!(!rgb_yields_camera_on_budget(true, false));
    }

    #[test]
    fn a_shared_camera_function_still_serializes_the_two_phases() {
        assert!(ir_waits_for_rgb(true, true));
        assert!(rgb_yields_camera_on_budget(true, true));
    }

    #[test]
    fn a_lone_spectrum_never_waits_on_the_other() {
        for serial_capture in [true, false] {
            assert!(!ir_waits_for_rgb(false, serial_capture), "{serial_capture}");
            assert!(
                !rgb_yields_camera_on_budget(false, serial_capture),
                "{serial_capture}"
            );
        }
    }

    #[test]
    fn parallel_capture_does_not_weaken_the_and_policy() {
        for (rgb_success, ir_success) in [(true, false), (false, true), (false, false)] {
            assert!(
                !hybrid_auth_passed(
                    "and",
                    true,
                    true,
                    true,
                    CaptureStatus::Usable,
                    rgb_success,
                    ir_success
                ),
                "{rgb_success} {ir_success}"
            );
        }
        assert!(hybrid_auth_passed(
            "and",
            true,
            true,
            true,
            CaptureStatus::Usable,
            true,
            true
        ));
    }

    #[test]
    fn parallel_capture_keeps_rgb_latching_on_dark_for_the_fallback_policy() {
        assert!(should_yield_rgb_to_ir(
            "fallback_on_dark",
            true,
            CaptureStatus::TooDark
        ));
        assert!(hybrid_auth_passed(
            "fallback_on_dark",
            true,
            true,
            true,
            CaptureStatus::TooDark,
            false,
            true
        ));
    }

    #[test]
    fn an_unavailable_rgb_camera_hands_auth_to_ir_unless_both_are_required() {
        for policy in ["or", "fallback_on_dark", "default"] {
            assert!(ir_continues_without_rgb(policy, true), "{policy}");
            assert!(hybrid_auth_passed(
                policy,
                false,
                true,
                false,
                CaptureStatus::Unused,
                false,
                true
            ));
        }
        assert!(!ir_continues_without_rgb("and", true));
        assert!(!ir_continues_without_rgb("fallback_on_dark", false));
    }

    #[test]
    fn a_disconnected_ir_camera_hands_auth_to_rgb_unless_both_are_required() {
        for policy in ["or", "fallback_on_dark", "default"] {
            assert!(rgb_continues_without_ir(policy, true), "{policy}");
            assert!(hybrid_auth_passed(
                policy,
                true,
                false,
                true,
                CaptureStatus::Ready,
                true,
                false
            ));
        }
        assert!(!rgb_continues_without_ir("and", true));
        assert!(!rgb_continues_without_ir("fallback_on_dark", false));
    }

    #[test]
    fn and_policy_refuses_to_degrade_to_one_spectrum() {
        let (run_rgb, run_ir) = auth_streams("primary", "/dev/video2", true, false);
        assert!(and_policy_unsatisfiable(
            "and",
            "primary",
            "/dev/video2",
            run_rgb,
            run_ir
        ));

        let (run_rgb, run_ir) = auth_streams("primary", "/dev/video2", false, true);
        assert!(and_policy_unsatisfiable(
            "and",
            "primary",
            "/dev/video2",
            run_rgb,
            run_ir
        ));
    }

    #[test]
    fn and_policy_is_satisfiable_with_both_spectra_enrolled() {
        let (run_rgb, run_ir) = auth_streams("primary", "/dev/video2", true, true);
        assert!(!and_policy_unsatisfiable(
            "and",
            "primary",
            "/dev/video2",
            run_rgb,
            run_ir
        ));
    }

    #[test]
    fn single_camera_hosts_are_not_blocked_by_the_and_policy() {
        let (run_rgb, run_ir) = auth_streams("primary", "", true, false);
        assert!(!and_policy_unsatisfiable(
            "and", "primary", "", run_rgb, run_ir
        ));

        let (run_rgb, run_ir) = auth_streams("", "/dev/video2", false, true);
        assert!(!and_policy_unsatisfiable(
            "and",
            "",
            "/dev/video2",
            run_rgb,
            run_ir
        ));
    }

    #[test]
    fn other_policies_still_allow_a_single_spectrum() {
        for policy in ["or", "fallback_on_dark", "default", ""] {
            let (run_rgb, run_ir) = auth_streams("primary", "/dev/video2", true, false);
            assert!(
                !and_policy_unsatisfiable(policy, "primary", "/dev/video2", run_rgb, run_ir),
                "{policy}"
            );
        }
    }

    #[test]
    fn hybrid_or_and_policies_require_the_configured_successes() {
        for rgb_status in [CaptureStatus::Usable, CaptureStatus::TooDark] {
            assert!(hybrid_auth_passed(
                "or", true, true, true, rgb_status, true, false
            ));
            assert!(hybrid_auth_passed(
                "or", true, true, true, rgb_status, false, true
            ));
            assert!(!hybrid_auth_passed(
                "and", true, true, true, rgb_status, true, false
            ));
            assert!(hybrid_auth_passed(
                "and", true, true, true, rgb_status, true, true
            ));
        }
    }

    #[test]
    fn hybrid_fallback_uses_ir_only_after_rgb_is_unavailable() {
        assert!(!hybrid_auth_passed(
            "fallback",
            true,
            true,
            false,
            CaptureStatus::Unused,
            false,
            true
        ));
        assert!(hybrid_auth_passed(
            "fallback",
            true,
            true,
            true,
            CaptureStatus::TooDark,
            false,
            true
        ));
        assert!(!hybrid_auth_passed(
            "fallback",
            true,
            true,
            true,
            CaptureStatus::NoFace,
            false,
            true
        ));
        assert!(!hybrid_auth_passed(
            "fallback",
            true,
            true,
            true,
            CaptureStatus::Usable,
            false,
            true
        ));
    }

    #[test]
    fn hybrid_fallback_yields_settled_dark_rgb_to_ir() {
        for policy in ["fallback_on_dark", "default", ""] {
            assert!(should_yield_rgb_to_ir(policy, true, CaptureStatus::TooDark));
        }
        assert!(!should_yield_rgb_to_ir(
            "fallback_on_dark",
            false,
            CaptureStatus::TooDark
        ));
        assert!(!should_yield_rgb_to_ir(
            "fallback_on_dark",
            true,
            CaptureStatus::NoFace
        ));
        assert!(!should_yield_rgb_to_ir("or", true, CaptureStatus::TooDark));
        assert!(!should_yield_rgb_to_ir("and", true, CaptureStatus::TooDark));
    }

    #[test]
    fn single_spectrum_authentication_ignores_the_other_result() {
        assert!(hybrid_auth_passed(
            "and",
            true,
            false,
            true,
            CaptureStatus::Usable,
            true,
            false
        ));
        assert!(hybrid_auth_passed(
            "and",
            false,
            true,
            false,
            CaptureStatus::Unused,
            false,
            true
        ));
        assert!(!hybrid_auth_passed(
            "or",
            false,
            false,
            false,
            CaptureStatus::Unused,
            true,
            true
        ));
    }
}
