// SPDX-FileCopyrightText: 2026 Gundu Labs
// SPDX-License-Identifier: GPL-3.0-or-later

//! Keeps face detection working when a convertible is held in portrait or upside down. The
//! camera sits in the screen bezel and turns with it, so its frames arrive rotated and the
//! detector, which only finds upright faces, sees nothing.

use opencv::core::Mat;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// The turn that brings a camera frame upright, in quarter turns clockwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    Upright = 0,
    Clockwise = 1,
    UpsideDown = 2,
    CounterClockwise = 3,
}

impl Rotation {
    const ALL: [Self; 4] = [
        Self::Upright,
        Self::Clockwise,
        Self::UpsideDown,
        Self::CounterClockwise,
    ];

    /// Maps an iio-sensor-proxy `AccelerometerOrientation`, which names the screen edge
    /// pointing up, to the turn that undoes it. With the left edge up the device has turned a
    /// quarter clockwise, so the camera sees the scene a quarter counter-clockwise.
    pub fn from_sensor(orientation: &str) -> Option<Self> {
        match orientation {
            "normal" => Some(Self::Upright),
            "left-up" => Some(Self::Clockwise),
            "bottom-up" => Some(Self::UpsideDown),
            "right-up" => Some(Self::CounterClockwise),
            _ => None,
        }
    }

    /// Returns the frame turned upright, or `None` when it already is, so the common case
    /// copies nothing.
    fn apply(self, frame: &Mat) -> opencv::Result<Option<Mat>> {
        let code = match self {
            Self::Upright => return Ok(None),
            Self::Clockwise => opencv::core::ROTATE_90_CLOCKWISE,
            Self::UpsideDown => opencv::core::ROTATE_180,
            Self::CounterClockwise => opencv::core::ROTATE_90_COUNTERCLOCKWISE,
        };
        let mut rotated = Mat::default();
        opencv::core::rotate(frame, &mut rotated, code)?;
        Ok(Some(rotated))
    }
}

/// The sensor's latest reading, shared between the task reading it and the capture threads.
/// It stays empty until the sensor answers, and on devices without one.
#[derive(Clone, Default)]
pub struct OrientationHint(Arc<AtomicU8>);

impl OrientationHint {
    pub fn set(&self, rotation: Rotation) {
        self.0.store(rotation as u8 + 1, Ordering::Relaxed);
    }

    pub(crate) fn get(&self) -> Option<Rotation> {
        let value = self.0.load(Ordering::Relaxed);
        Rotation::ALL
            .get(usize::from(value).checked_sub(1)?)
            .copied()
    }
}

/// Picks the rotations to run the detector on, per frame.
#[derive(Debug, Default)]
pub(crate) struct RotationSearch {
    found: Option<Rotation>,
    probes: usize,
}

impl RotationSearch {
    /// The rotation to try first, being the one that last found a face, else the sensor's, else
    /// upright. Then, on devices with a sensor, one other rotation, round-robin, for when the
    /// first finds nothing: that covers a sensor that is slow, wrong, or read while lying flat.
    pub(crate) fn candidates(&mut self, sensor: Option<Rotation>) -> (Rotation, Option<Rotation>) {
        let first = self.found.or(sensor).unwrap_or(Rotation::Upright);
        let probe = sensor.map(|_| {
            self.probes += 1;
            Rotation::ALL[(first as usize + 1 + self.probes % 3) % 4]
        });
        (first, probe)
    }

    /// Records the rotation that found a face; returns true when it differs from before.
    pub(crate) fn found(&mut self, rotation: Rotation) -> bool {
        self.found.replace(rotation) != Some(rotation)
    }

    /// Detects on `frame` turned by each candidate in turn, returning the turned frame (`None`
    /// when upright) with the first detection that finds a face.
    pub(crate) fn detect<T, E>(
        &mut self,
        sensor: Option<Rotation>,
        frame: &Mat,
        mut detect: impl FnMut(&Mat) -> Result<Option<T>, E>,
    ) -> Result<Option<(Rotation, Option<Mat>, T)>, E>
    where
        E: From<opencv::Error>,
    {
        let (first, probe) = self.candidates(sensor);
        for rotation in std::iter::once(first).chain(probe) {
            let rotated = rotation.apply(frame)?;
            if let Some(found) = detect(rotated.as_ref().unwrap_or(frame))? {
                return Ok(Some((rotation, rotated, found)));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::prelude::*;

    fn frame(rows: i32, cols: i32) -> Mat {
        Mat::new_rows_cols_with_default(rows, cols, opencv::core::CV_8UC3, 0.0.into()).unwrap()
    }

    #[test]
    fn sensor_orientations_map_to_the_turn_that_undoes_them() {
        assert_eq!(Rotation::from_sensor("normal"), Some(Rotation::Upright));
        assert_eq!(Rotation::from_sensor("left-up"), Some(Rotation::Clockwise));
        assert_eq!(
            Rotation::from_sensor("right-up"),
            Some(Rotation::CounterClockwise)
        );
        assert_eq!(
            Rotation::from_sensor("bottom-up"),
            Some(Rotation::UpsideDown)
        );
        assert_eq!(Rotation::from_sensor("undefined"), None);
    }

    #[test]
    fn the_hint_is_empty_until_set_and_round_trips_every_rotation() {
        let hint = OrientationHint::default();
        assert_eq!(hint.get(), None);
        for rotation in Rotation::ALL {
            hint.set(rotation);
            assert_eq!(hint.get(), Some(rotation));
        }
    }

    #[test]
    fn quarter_turns_swap_the_frame_dimensions() {
        let frame = frame(480, 640);
        assert!(Rotation::Upright.apply(&frame).unwrap().is_none());
        let turned = Rotation::Clockwise.apply(&frame).unwrap().unwrap();
        assert_eq!((turned.cols(), turned.rows()), (480, 640));
        let flipped = Rotation::UpsideDown.apply(&frame).unwrap().unwrap();
        assert_eq!((flipped.cols(), flipped.rows()), (640, 480));
    }

    #[test]
    fn without_a_sensor_only_the_upright_frame_is_tried() {
        let mut search = RotationSearch::default();
        assert_eq!(search.candidates(None), (Rotation::Upright, None));
    }

    #[test]
    fn the_sensor_leads_and_probes_cover_every_other_rotation() {
        let mut search = RotationSearch::default();
        let sensor = Some(Rotation::Clockwise);
        let probes: Vec<_> = (0..3)
            .map(|_| {
                let (first, probe) = search.candidates(sensor);
                assert_eq!(first, Rotation::Clockwise);
                probe.unwrap()
            })
            .collect();
        for rotation in [
            Rotation::Upright,
            Rotation::UpsideDown,
            Rotation::CounterClockwise,
        ] {
            assert!(probes.contains(&rotation), "{rotation:?} was never probed");
        }
    }

    #[test]
    fn a_rotation_that_found_a_face_wins_over_the_sensor() {
        let mut search = RotationSearch::default();
        assert!(search.found(Rotation::CounterClockwise));
        assert!(!search.found(Rotation::CounterClockwise));
        assert_eq!(
            search.candidates(Some(Rotation::Clockwise)).0,
            Rotation::CounterClockwise
        );
    }

    #[test]
    fn detect_returns_the_first_rotation_that_finds_a_face() {
        let mut search = RotationSearch::default();
        let landscape = frame(480, 640);
        // Only a portrait frame "has a face", so the sensor's upright guess must miss and a
        // quarter-turn probe must hit within three frames.
        let found = (0..3).find_map(|_| {
            search
                .detect(Some(Rotation::Upright), &landscape, |m: &Mat| {
                    Ok::<_, opencv::Error>((m.rows() > m.cols()).then_some(()))
                })
                .unwrap()
        });
        let (rotation, rotated, ()) = found.expect("a quarter turn finds the face");
        assert!(matches!(
            rotation,
            Rotation::Clockwise | Rotation::CounterClockwise
        ));
        assert_eq!(rotated.unwrap().rows(), 640);
    }
}
