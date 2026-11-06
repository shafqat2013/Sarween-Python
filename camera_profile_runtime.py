"""Resolve one explicitly selected camera calibration for a CV session."""

from __future__ import annotations

import copy
import numpy as np


_ACTIVE_PROFILE = object()


class CameraCalibrationChanged(ValueError):
    """A running CV session needs new registration before it can continue."""


def legacy_offline_camera_params():
    """Read unchanged historical parameters only for raw legacy file replay."""
    from app_paths import data_path
    try:
        matrix = np.load(data_path("camera_matrix.npy"), allow_pickle=False)
        distortion = np.load(data_path("dist_coeffs.npy"), allow_pickle=False)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all()):
            return None, None
        return matrix, distortion
    except (OSError, TypeError, ValueError):
        return None, None


class CameraCorrection:
    """Freeze the chosen profile; resolve against actual frame dimensions.

    Camera indices and OBS device labels cannot identify an iPhone lens or OBS
    crop. The setup dialog explicitly confirms that pairing. Once a session
    starts, changing the selected profile affects the next session only.
    """

    def __init__(self, *, profiles=None, profile_id=_ACTIVE_PROFILE):
        if profiles is None:
            from hardware_profiles import HardwareProfiles
            profiles = HardwareProfiles()
        self.profiles = profiles
        if profile_id is _ACTIVE_PROFILE:
            selected = profiles.active_profile("camera")
            profile_id = selected.get("id") if selected else None
        else:
            selected = profiles.get_profile("camera", profile_id) if profile_id else None
        self.profile_id = profile_id
        self.profile_name = selected.get("name") if selected else None
        self._size = None
        self._revision = None
        self.matrix = None
        self.dist_coeffs = None
        self.applied = False
        self.reason = "No camera profile selected; lens correction is off."

    def resolve(self, image_size):
        size = tuple(int(value) for value in image_size)
        try:
            revision = self.profiles.revision() if self.profile_id else None
        except (TypeError, ValueError, OSError) as exc:
            self._size, self._revision = size, None
            self.matrix, self.dist_coeffs, self.applied = None, None, False
            self.reason = f"Camera calibration unavailable: {exc}"
            return None, None
        if size == self._size and revision == self._revision:
            return self.matrix, self.dist_coeffs
        self._size, self._revision = size, revision
        self.matrix, self.dist_coeffs, self.applied = None, None, False
        if not self.profile_id:
            self.reason = "No camera profile selected; lens correction is off."
            return None, None
        try:
            calibration, self.reason = self.profiles.resolve_camera_calibration(size, profile_id=self.profile_id)
            selected = self.profiles.get_profile("camera", self.profile_id)
            self.profile_name = selected.get("name") if selected else self.profile_name
            if calibration is not None:
                self.matrix = np.asarray(calibration["cameraMatrix"], dtype=np.float64)
                self.dist_coeffs = np.asarray(calibration["distCoeffs"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self.reason = f"Camera calibration unavailable: {exc}"
        return self.matrix, self.dist_coeffs

    def snapshot(self):
        return copy.deepcopy({"profileId": self.profile_id, "profileName": self.profile_name,
                              "applied": self.applied, "reason": self.reason,
                              "frameSize": list(self._size) if self._size else None})


def camera_params(image_size=None, *, profiles=None, profile_id=_ACTIVE_PROFILE):
    """Compatibility helper for one-shot mini captures; never load legacy NPYs."""
    if image_size is None:
        return None, None
    return CameraCorrection(profiles=profiles, profile_id=profile_id).resolve(image_size)
