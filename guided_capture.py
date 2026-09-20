"""Validated placement labels, independent of tracking predictions."""

import copy


class GuidedCapture:
    def __init__(self, payload, scene_id):
        self.session_id = str(payload.get("sessionId") or "")
        self.scene_id = str(payload.get("sceneId") or "")
        if not self.session_id or self.scene_id != scene_id:
            raise ValueError("Capture must belong to the active Foundry scene")
        self.participants = copy.deepcopy(payload.get("participants") or [])
        self.targets = copy.deepcopy(payload.get("targets") or [])
        if not 1 <= len(self.participants) <= 5:
            raise ValueError("Choose between one and five minis")
        ids = [str(item.get("miniId") or "") for item in self.participants]
        tokens = [str(item.get("tokenId") or "") for item in self.participants]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("Every mini needs a unique identity")
        if not all(tokens) or len(set(tokens)) != len(tokens):
            raise ValueError("Choose a different Foundry token for each mini")
        if len(self.targets) != len(ids) * 6:
            raise ValueError("Each mini needs an initial placement and five moves")
        cells = set()
        for index, target in enumerate(self.targets):
            if target.get("miniId") not in ids:
                raise ValueError("Unknown mini in capture route")
            row, column = int(target["row"]), int(target["column"])
            if min(row, column) < 0 or (row, column) in cells:
                raise ValueError("Capture targets must be distinct grid cells")
            cells.add((row, column))
            target["index"] = index
        self.scenario = str(payload.get("scenario") or "fog-on")
        self.notes = str(payload.get("notes") or "")[:2000]
        self.module_version = str(payload.get("moduleVersion") or "")
        self.scene_vision = copy.deepcopy(payload.get("sceneVision") or {})
        self.marker_settings = copy.deepcopy(payload.get("markerSettings") or {})
        self.next_index = 0
        self.prompt_frame = None

    def metadata(self):
        return {
            "sessionId": self.session_id,
            "sceneId": self.scene_id,
            "scenario": self.scenario,
            "notes": self.notes,
            "moduleVersion": self.module_version,
            "sceneVision": self.scene_vision,
            "markerSettings": self.marker_settings,
            "participants": self.participants,
            "targets": self.targets,
        }

    def prompt(self, index, frame):
        if int(index) != self.next_index or self.next_index >= len(self.targets):
            raise ValueError("Capture prompt is out of sequence")
        # Reconnection/retries must not shorten the original movement window.
        if self.prompt_frame is None:
            self.prompt_frame = int(frame)

    def confirm(self, index, frame):
        if int(index) != self.next_index or self.prompt_frame is None:
            raise ValueError("Placement is duplicate or has not been prompted")
        target = self.targets[self.next_index]
        event = {
            **target,
            "mini": target["miniId"],
            "frame": int(frame),
            "promptFrame": self.prompt_frame,
            "source": "userConfirmation",
        }
        self.next_index += 1
        self.prompt_frame = None
        return event


def regression_case(data, video_name):
    """Use prompted/confirmed frame windows, never detector emissions, as truth."""
    fps = float(data["fps"])
    expectations = []
    previous = {}
    for event in data.get("groundTruth") or []:
        mini = event["mini"]
        expectation = {
            "mini": mini,
            "to": event["cell"],
            "between": [event["promptFrame"] / fps, event["frame"] / fps + 3.0],
        }
        if mini in previous:
            expectation["from"] = previous[mini]
        expectations.append(expectation)
        previous[mini] = event["cell"]
    return {
        "cases": [{
            "name": video_name.removesuffix(".mp4"),
            "video": video_name,
            "timeline": video_name.removesuffix(".mp4") + ".tracking.json",
            "profiles": "combo_profiles.json",
            "allow_unexpected": False,
            "expectations": expectations,
        }]
    }
