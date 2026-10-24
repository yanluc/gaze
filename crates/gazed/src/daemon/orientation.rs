// SPDX-FileCopyrightText: 2026 Gundu Labs
// SPDX-License-Identifier: GPL-3.0-or-later

//! Reads the screen orientation of convertibles from iio-sensor-proxy, so capture can turn the
//! bezel camera's frames upright. Any failure leaves frames as they are.

use super::*;
use gaze_vision::orientation::{OrientationHint, Rotation};

const SENSOR_SERVICE: &str = "net.hadess.SensorProxy";
const SENSOR_PATH: &str = "/net/hadess/SensorProxy";
// The proxy reports "undefined" until a claimed accelerometer delivers its first sample.
const SENSOR_SETTLE: Duration = Duration::from_millis(1500);
const SENSOR_POLL: Duration = Duration::from_millis(50);

impl AuthDaemon {
    /// Starts reading the orientation in the background and returns the hint capture follows.
    /// Capture starts at once and treats frames as upright until the reading arrives.
    pub(super) fn orientation_hint(&self, conn: &zbus::Connection) -> OrientationHint {
        let hint = OrientationHint::default();
        let reader = hint.clone();
        let conn = conn.clone();
        // Spawned on Tokio: the polling sleeps, and zbus runs method handlers elsewhere.
        self.rt_handle.spawn(async move {
            match sensor_rotation(&conn).await {
                Ok(Some(rotation)) => reader.set(rotation),
                Ok(None) => tracing::debug!("No accelerometer; camera frames stay as they come"),
                Err(error) => tracing::debug!("Orientation sensor unavailable: {error}"),
            }
        });
        hint
    }
}

/// The turn that brings camera frames upright, or `None` on a device without an accelerometer.
async fn sensor_rotation(conn: &zbus::Connection) -> zbus::Result<Option<Rotation>> {
    let proxy: zbus::Proxy<'_> = zbus::proxy::Builder::new(conn)
        .destination(SENSOR_SERVICE)
        .path(SENSOR_PATH)
        .interface(SENSOR_SERVICE)
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await?;
    if !proxy.get_property::<bool>("HasAccelerometer").await? {
        return Ok(None);
    }

    proxy.call_method("ClaimAccelerometer", &()).await?;
    let rotation = settled_rotation(&proxy).await;
    // Release whatever the reading did: a claim left behind keeps the sensor powered until
    // gazed disconnects from the bus.
    if let Err(error) = proxy.call_method("ReleaseAccelerometer", &()).await {
        tracing::debug!("Failed to release the accelerometer: {error}");
    }
    let rotation = rotation?;
    tracing::debug!(?rotation, "Orientation sensor reading");
    Ok(Some(rotation))
}

/// Polls a claimed accelerometer until it reports an orientation.
async fn settled_rotation(proxy: &zbus::Proxy<'_>) -> zbus::Result<Rotation> {
    let deadline = Instant::now() + SENSOR_SETTLE;
    loop {
        let orientation: String = proxy.get_property("AccelerometerOrientation").await?;
        if let Some(rotation) = Rotation::from_sensor(&orientation) {
            return Ok(rotation);
        }
        if Instant::now() >= deadline {
            // A sensor that will not settle (lying flat, say) still marks a convertible worth
            // probing.
            return Ok(Rotation::Upright);
        }
        tokio::time::sleep(SENSOR_POLL).await;
    }
}
