import os

from kokoro_onnx import Kokoro  # type: ignore

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.kokoro_base_model import KokoroBaseModel
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import make_error_string


class KokoroModel(KokoroBaseModel):
    """
    Kokoro-82M inference via kokoro-onnx (ONNX runtime, CPU or GPU).
    """

    def __init__(self, model_path: str, voices_path: str):
        self._kokoro = Kokoro(model_path, voices_path)
        self._voices = self._kokoro.get_voices()

    def kill(self) -> None:
        self._kokoro = None  # type: ignore
        self._voices = []

    def get_voices(self) -> list[str]:
        return list(self._voices)

    def generate_using_project(
            self,
            project: Project,
            prompts: list[str],
            force_random_seed: bool = False
    ) -> list[Sound] | str:

        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        voice = project.kokoro_voice or "af_bella"
        speed = project.kokoro_speed if project.kokoro_speed > 0 else KokoroBaseModel.DEFAULT_SPEED
        lang = project.kokoro_lang or KokoroBaseModel.DEFAULT_LANG

        result = self.generate(text=prompt, voice=voice, speed=speed, lang=lang)

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
        self,
        text: str,
        voice: str = "af_bella",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> Sound | str:

        if self._kokoro is None:
            return "Logic error: Model is not initialized"

        try:
            samples, sr = self._kokoro.create(text, voice=voice, speed=speed, lang=lang)
            return Sound(samples, sr)
        except Exception as e:
            return make_error_string(e)
