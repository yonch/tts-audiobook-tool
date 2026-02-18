import unittest

from tts_audiobook_tool.api import AudiobookConfig, _config_to_project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos


class TestConfigToProject(unittest.TestCase):
    """Tests for _config_to_project multi-model parameter mapping."""

    def setUp(self):
        self._saved_type = getattr(Tts, "_type", TtsModelInfos.NONE)

    def tearDown(self):
        Tts._type = self._saved_type

    def _make_project(self, **kwargs):
        config = AudiobookConfig(project_dir="", **kwargs)
        return _config_to_project(config)

    # -- Chatterbox-specific fields --

    def test_chatterbox_type_multilingual(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(chatterbox_type="multilingual")
        self.assertEqual(project.chatterbox_type, ChatterboxType.MULTILINGUAL)

    def test_chatterbox_type_turbo(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(chatterbox_type="turbo")
        self.assertEqual(project.chatterbox_type, ChatterboxType.TURBO)

    def test_chatterbox_exaggeration(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(chatterbox_exaggeration=0.7)
        self.assertEqual(project.chatterbox_exaggeration, 0.7)

    def test_chatterbox_cfg(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(chatterbox_cfg=0.3)
        self.assertEqual(project.chatterbox_cfg, 0.3)

    def test_chatterbox_defaults_leave_project_defaults(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project()
        self.assertEqual(project.chatterbox_exaggeration, -1)
        self.assertEqual(project.chatterbox_cfg, -1)
        self.assertEqual(project.chatterbox_temperature, -1)
        self.assertEqual(project.chatterbox_seed, -1)

    # -- Generic temperature routing --

    def test_temperature_routes_to_chatterbox(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.chatterbox_temperature, 0.9)

    def test_temperature_routes_to_fish(self):
        Tts._type = TtsModelInfos.FISH
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.fish_temperature, 0.9)

    def test_temperature_routes_to_higgs(self):
        Tts._type = TtsModelInfos.HIGGS
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.higgs_temperature, 0.9)

    def test_temperature_routes_to_oute(self):
        Tts._type = TtsModelInfos.OUTE
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.oute_temperature, 0.9)

    def test_temperature_routes_to_mira(self):
        Tts._type = TtsModelInfos.MIRA
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.mira_temperature, 0.9)

    def test_temperature_routes_to_qwen3(self):
        Tts._type = TtsModelInfos.QWEN3TTS
        project = self._make_project(temperature=0.9)
        self.assertEqual(project.qwen3_temperature, 0.9)

    def test_temperature_ignored_for_kokoro(self):
        Tts._type = TtsModelInfos.KOKORO
        project = self._make_project(temperature=0.9)
        # Kokoro has no temperature — verify chatterbox temp wasn't set as side effect
        self.assertEqual(project.chatterbox_temperature, -1)

    def test_temperature_default_leaves_project_default(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project()  # temperature=-1 (default)
        self.assertEqual(project.chatterbox_temperature, -1)

    # -- Generic seed routing --

    def test_seed_routes_to_chatterbox(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(seed=42)
        self.assertEqual(project.chatterbox_seed, 42)

    def test_seed_routes_to_fish(self):
        Tts._type = TtsModelInfos.FISH
        project = self._make_project(seed=42)
        self.assertEqual(project.fish_seed, 42)

    def test_seed_routes_to_vibevoice(self):
        Tts._type = TtsModelInfos.VIBEVOICE
        project = self._make_project(seed=42)
        self.assertEqual(project.vibevoice_seed, 42)

    def test_seed_routes_to_glm(self):
        Tts._type = TtsModelInfos.GLM
        project = self._make_project(seed=42)
        self.assertEqual(project.glm_seed, 42)

    def test_seed_routes_to_qwen3(self):
        Tts._type = TtsModelInfos.QWEN3TTS
        project = self._make_project(seed=42)
        self.assertEqual(project.qwen3_seed, 42)

    def test_seed_ignored_for_kokoro(self):
        Tts._type = TtsModelInfos.KOKORO
        project = self._make_project(seed=42)
        self.assertEqual(project.chatterbox_seed, -1)

    def test_seed_default_leaves_project_default(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project()  # seed=-1 (default)
        self.assertEqual(project.chatterbox_seed, -1)

    # -- Kokoro fields still work --

    def test_kokoro_voice(self):
        Tts._type = TtsModelInfos.KOKORO
        project = self._make_project(kokoro_voice="af_heart")
        self.assertEqual(project.kokoro_voice, "af_heart")

    def test_kokoro_speed(self):
        Tts._type = TtsModelInfos.KOKORO
        project = self._make_project(kokoro_speed=1.5)
        self.assertEqual(project.kokoro_speed, 1.5)

    # -- model_overrides precedence --

    def test_model_overrides_beats_temperature(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(
            temperature=0.5,
            model_overrides={"chatterbox_temperature": 0.9},
        )
        self.assertEqual(project.chatterbox_temperature, 0.9)

    def test_model_overrides_beats_seed(self):
        Tts._type = TtsModelInfos.CHATTERBOX
        project = self._make_project(
            seed=42,
            model_overrides={"chatterbox_seed": 99},
        )
        self.assertEqual(project.chatterbox_seed, 99)

    # -- Cross-model isolation --

    def test_chatterbox_fields_set_even_when_kokoro_active(self):
        Tts._type = TtsModelInfos.KOKORO
        project = self._make_project(
            chatterbox_type="turbo",
            chatterbox_exaggeration=0.8,
            chatterbox_cfg=0.4,
        )
        # Chatterbox project fields are set regardless of active model
        self.assertEqual(project.chatterbox_type, ChatterboxType.TURBO)
        self.assertEqual(project.chatterbox_exaggeration, 0.8)
        self.assertEqual(project.chatterbox_cfg, 0.4)


if __name__ == "__main__":
    unittest.main()
