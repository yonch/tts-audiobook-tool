from __future__ import annotations

from tts_audiobook_tool.tts_model.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
else:
    Project = object


class KokoroBaseModel(TtsBaseModel):

    INFO = TtsModelInfos.KOKORO.value

    DEFAULT_SPEED = 1.0
    DEFAULT_LANG = "en-us"

    @classmethod
    def get_prereq_errors(
            cls, project: Project, instance: TtsBaseModel | None, is_short: bool
    ) -> list[str]:
        # Kokoro uses named voice presets, not voice clone files.
        # The only prerequisite is that a voice name is set (and it always has a default).
        if not project.kokoro_voice:
            err = "requires voice name" if is_short else "Kokoro voice name must be set (e.g. af_bella)"
            return [err]
        return []

    @classmethod
    def get_voice_tag(cls, project: Project) -> str:
        return project.kokoro_voice or "default"

    @classmethod
    def get_voice_display_info(
            cls, project: Project, instance: TtsBaseModel | None = None
    ) -> tuple[str, str]:
        voice = project.kokoro_voice or "(not set)"
        prefix = COL_DIM + "current voice"
        value = COL_ACCENT + voice
        return prefix, value
