import os
import random
import numpy as np
import torch
import chatterbox.mtl_tts # type: ignore
from chatterbox.mtl_tts import ChatterboxMultilingualTTS # type: ignore
from chatterbox.tts_turbo import ChatterboxTurboTTS # type: ignore

import logging

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxBaseModel, ChatterboxType
logging.getLogger("transformers").setLevel(logging.ERROR)

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import make_error_string


class ChatterboxModel(ChatterboxBaseModel):
    """
    Chatterbox inference logic
    """

    def __init__(self, model_type: ChatterboxType, device: str):
        self._device = device
        self._model_type = model_type

        # Work around upstream bugs in chatterbox-tts:
        # 1. from_local() compares `device in ["cpu", "mps"]` which fails with a
        #    torch.device object — pass device as a string.
        # 2. tts_turbo uses `token=os.getenv("HF_TOKEN") or True` which forces
        #    HF authentication even for public repos — set a dummy value so the
        #    `or True` branch is never reached.
        had_token = "HF_TOKEN" in os.environ
        old_token = os.environ.get("HF_TOKEN")
        if not old_token:
            os.environ["HF_TOKEN"] = "none"
        try:
            match self._model_type:
                case ChatterboxType.MULTILINGUAL:
                    self._chatterbox = ChatterboxMultilingualTTS.from_pretrained(device=device)
                case ChatterboxType.TURBO:
                    self._chatterbox = ChatterboxTurboTTS.from_pretrained(device=device)
        finally:
            if not had_token:
                os.environ.pop("HF_TOKEN", None)
            elif old_token is not None:
                os.environ["HF_TOKEN"] = old_token

    def supported_languages_multi(self) -> list[str]:
        return list(chatterbox.mtl_tts.SUPPORTED_LANGUAGES)

    def kill(self) -> None:
        self._chatterbox = None # type: ignore

    def generate_using_project(
            self, 
            project: Project, 
            prompts: list[str], 
            force_random_seed: bool=False
        ) -> list[Sound] | str:
        
        if len(prompts) != 1:
            raise ValueError("Implementation does not support batching")
        prompt = prompts[0]

        if project.chatterbox_voice_file_name:
            voice_path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
        else:
            voice_path = ""

        if project.chatterbox_type == ChatterboxType.MULTILINGUAL:
            language_id = project.language_code 
        else:
            language_id = ""

        seed = -1 if force_random_seed else project.chatterbox_seed

        result = self.generate(
            text=prompt,
            voice_path=voice_path,
            exaggeration=project.chatterbox_exaggeration,
            cfg=project.chatterbox_cfg,
            temperature=project.chatterbox_temperature,
            seed=seed,
            language_id=language_id
        )

        if isinstance(result, Sound):
            return [result]
        else:
            return result

    def generate(
        self,
        text: str,
        voice_path: str = "",
        exaggeration: float = -1,
        cfg: float = -1,
        temperature: float = -1,
        seed: int = -1,
        language_id: str = ""
    ) -> Sound | str:

        if self._chatterbox is None:
            return "Logic error: Model is not initialized"
        if language_id and self._model_type == ChatterboxType.TURBO:
            return "Logic error: language_id is not supported for Chatterbox Turbo"
        
        if seed <= -1:
            seed = random.randrange(0, 2**32 - 1)
        self._set_seed(seed)

        dic = {}
        if language_id:
            dic["language_id"] = language_id
        if voice_path:
            dic["audio_prompt_path"] = voice_path
        if exaggeration != -1:
            dic["exaggeration"] = exaggeration
        if cfg != -1:
            dic["cfg_weight"] = cfg
        if temperature != -1:
            dic["temperature"] = temperature

        try:
            data = self._chatterbox.generate(text, **dic)
            data = data.cpu().numpy().squeeze()
            return Sound(data, TtsModelInfos.CHATTERBOX.value.sample_rate)
        except Exception as e:
            return make_error_string(e)
        
    def _set_seed(self, seed: int):
        """Sets the random seed for reproducibility across torch, numpy, and random."""
        torch.manual_seed(seed)
        if self._device.startswith("cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
