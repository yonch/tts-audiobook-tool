"""
Programmatic Python API for tts-audiobook-tool.

Provides importable functions for text-to-speech audiobook generation
without the interactive terminal UI.

Usage:
    from tts_audiobook_tool.api import (
        init, segment_text, generate_segment, generate_all,
        concatenate, create_audiobook, AudiobookConfig
    )

    model_name = init()
    config = AudiobookConfig(
        project_dir="/path/to/my-audiobook",
        kokoro_voice="af_bella",
    )
    result = create_audiobook(open("book.txt").read(), config)
    print(result.output_path)

Note: Not thread-safe. The TTS and STT subsystems use shared static state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tts_audiobook_tool.app_types import (
    ChapterMode, ExportType, NormalizationType,
    SegmentationStrategy, Sound, SttConfig, SttVariant, Strictness,
)
from tts_audiobook_tool.phrase import PhraseGroup


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AudiobookConfig:
    """Configuration for audiobook generation."""

    # Required — directory for project files (segments/, combined/, etc.)
    project_dir: str = ""

    # Text processing
    language_code: str = "en"
    max_words_per_segment: int = 40
    segmentation_strategy: str = "normal"   # "normal" | "multi" | "max_len"
    word_substitutions: dict[str, str] = field(default_factory=dict)

    # Kokoro-specific voice settings
    kokoro_voice: str = "af_bella"
    kokoro_speed: float = 1.0
    kokoro_lang: str = "en-us"
    kokoro_model_path: str = ""
    kokoro_voices_path: str = ""

    # Voice-clone models (Chatterbox, Fish, Higgs, etc.)
    voice_clone_path: str = ""
    voice_clone_transcript: str = ""

    # Generation
    max_retries: int = 1

    # Validation (STT)
    stt_enabled: bool = True
    stt_variant: str = "large-v3"           # "large-v3" | "large-v3-turbo" | "disabled"
    stt_device: str = ""                    # "" = auto-detect, "cpu", "cuda"
    strictness: str = "moderate"            # "low" | "moderate" | "high"

    # Output
    export_type: str = "m4a"                # "m4a" | "flac"
    normalization: str = "default"          # "default" | "stronger" | "none"
    use_section_sound_effect: bool = False
    chapter_mode: str = "files"             # "files" | "metadata"

    # Model
    force_cpu: bool = False

    # Set arbitrary Project fields (advanced)
    model_overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of generating audio for a single segment."""
    index: int
    sound: Sound | None
    error: str
    validation_passed: bool
    validation_message: str
    saved_path: str


@dataclass
class AudiobookResult:
    """Result of the full audiobook creation pipeline."""
    output_path: str
    error: str
    generation_results: list[GenerationResult]
    num_segments: int
    num_succeeded: int
    num_failed: int


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def init(force_cpu: bool = False) -> str:
    """
    Initialize the TTS subsystem. Must be called before other API functions.

    Args:
        force_cpu: Force CPU-only inference.

    Returns:
        Proper name of the detected TTS model (e.g. "Kokoro TTS").

    Raises:
        RuntimeError: If no supported TTS model is installed, or if multiple
            model libraries are detected simultaneously.
    """
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
    from huggingface_hub import constants  # noqa: F401  (side-effect import)

    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

    model_type, num_matches = Tts.init_model_type()

    if model_type == TtsModelInfos.NONE:
        if num_matches > 1:
            raise RuntimeError(
                "Multiple TTS model libraries detected. "
                "Use a virtual environment with only one installed."
            )
        raise RuntimeError(
            "No supported TTS model library found. "
            "Install one following the project README."
        )

    if force_cpu:
        Tts.set_force_cpu(True)

    return model_type.value.ui["proper_name"]


def segment_text(
    text: str,
    language_code: str = "en",
    max_words: int = 40,
    strategy: str = "normal",
) -> list[PhraseGroup]:
    """
    Segment text into PhraseGroups for TTS generation.

    Each PhraseGroup becomes one TTS prompt / audio segment.

    Args:
        text: Raw input text.
        language_code: pysbd language code (e.g. "en", "de", "fr").
        max_words: Maximum words per segment.
        strategy: "normal", "multi", or "max_len".

    Returns:
        List of PhraseGroup objects.
    """
    strat = SegmentationStrategy.from_id(strategy)
    if strat is None:
        valid = [s.id for s in SegmentationStrategy]
        raise ValueError(
            f"Invalid segmentation strategy {strategy!r}. "
            f"Valid options: {valid}"
        )

    from tts_audiobook_tool.phrase_grouper import PhraseGrouper

    return PhraseGrouper.text_to_groups(
        text, max_words=max_words, strategy=strat, pysbd_lang=language_code
    )


def generate_segment(
    index: int,
    phrase_groups: list[PhraseGroup],
    config: AudiobookConfig,
    validate: bool = True,
    force_random_seed: bool = False,
) -> GenerationResult:
    """
    Generate audio for a single text segment.

    Args:
        index: 0-based index into phrase_groups.
        phrase_groups: Full list (as returned by segment_text).
        config: Generation settings.
        validate: Run STT validation on the generated audio.
        force_random_seed: Use a random seed (useful for retries).

    Returns:
        GenerationResult with the generated Sound and validation info.
    """
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.tts import Tts

    project = _config_to_project(config)
    project.phrase_groups = phrase_groups
    Tts.set_model_params_using_project(project)

    # Ensure TTS model is loaded
    _ = Tts.get_instance()

    gen_results = GenerateUtil.generate(
        project=project,
        indices=[index],
        phrase_groups=phrase_groups,
        force_random_seed=force_random_seed,
        is_realtime=False,
        save_debug_files=False,
    )

    gen_result = gen_results[0]
    if isinstance(gen_result, str):
        return GenerationResult(
            index=index, sound=None, error=gen_result,
            validation_passed=False, validation_message="", saved_path="",
        )

    sound = gen_result
    validation_passed = True
    validation_message = "Validation skipped"

    stt_disabled = (
        not validate
        or not config.stt_enabled
        or config.stt_variant == "disabled"
    )

    if not stt_disabled:
        val_passed, val_msg, val_sound = _validate_sound(
            sound, phrase_groups[index], config
        )
        validation_passed = val_passed
        validation_message = val_msg
        if val_sound is not None:
            sound = val_sound

    return GenerationResult(
        index=index, sound=sound, error="",
        validation_passed=validation_passed,
        validation_message=validation_message,
        saved_path="",
    )


def generate_all(
    phrase_groups: list[PhraseGroup],
    config: AudiobookConfig,
    indices: set[int] | None = None,
    save_segments: bool = True,
    on_progress: Callable[[int, int, GenerationResult], None] | None = None,
) -> list[GenerationResult]:
    """
    Generate audio for all (or specified) segments with retry logic.

    Args:
        phrase_groups: Full list of PhraseGroups.
        config: Generation settings.
        indices: Specific 0-based indices to generate (None = all).
        save_segments: Save .flac segment files to config.project_dir.
        on_progress: Optional callback(completed, total, latest_result).

    Returns:
        List of GenerationResult in index order.
    """
    from tts_audiobook_tool.generate_util import GenerateUtil
    from tts_audiobook_tool.tts import Tts

    project = _config_to_project(config)
    project.phrase_groups = phrase_groups
    Tts.set_model_params_using_project(project)

    # Ensure TTS model is loaded
    _ = Tts.get_instance()

    if indices is None:
        indices = set(range(len(phrase_groups)))

    sorted_indices = sorted(indices)
    results_map: dict[int, GenerationResult] = {}

    # Work queue: (index, retry_count)
    items: list[tuple[int, int]] = [(i, 0) for i in sorted_indices]
    completed = 0
    total = len(sorted_indices)

    while items:
        index, retry_count = items.pop(0)

        gen_results = GenerateUtil.generate(
            project=project,
            indices=[index],
            phrase_groups=phrase_groups,
            force_random_seed=(retry_count > 0),
            is_realtime=False,
            save_debug_files=False,
        )

        gen_result = gen_results[0]

        if isinstance(gen_result, str):
            result = GenerationResult(
                index=index, sound=None, error=gen_result,
                validation_passed=False, validation_message="",
                saved_path="",
            )
        else:
            sound = gen_result
            validation_passed = True
            validation_message = "Validation skipped"

            stt_disabled = (
                not config.stt_enabled
                or config.stt_variant == "disabled"
            )

            if not stt_disabled:
                val_passed, val_msg, val_sound = _validate_sound(
                    sound, phrase_groups[index], config
                )
                validation_passed = val_passed
                validation_message = val_msg
                if val_sound is not None:
                    sound = val_sound

            result = GenerationResult(
                index=index, sound=sound, error="",
                validation_passed=validation_passed,
                validation_message=validation_message,
                saved_path="",
            )

        # Retry?
        needs_retry = (result.error or not result.validation_passed)
        if needs_retry and retry_count < config.max_retries:
            items.append((index, retry_count + 1))
            continue

        # Save segment file
        if save_segments and result.sound is not None:
            saved_path = _save_segment(project, index, phrase_groups[index], result)
            result = GenerationResult(
                index=result.index, sound=result.sound, error=result.error,
                validation_passed=result.validation_passed,
                validation_message=result.validation_message,
                saved_path=saved_path,
            )

        results_map[index] = result
        completed += 1

        if on_progress is not None:
            on_progress(completed, total, result)

    return [results_map[i] for i in sorted_indices]


def concatenate(
    config: AudiobookConfig,
    phrase_groups: list[PhraseGroup],
    raw_text: str = "",
    output_path: str = "",
) -> str:
    """
    Concatenate generated audio segments into a final audiobook file.

    Segments must have been previously generated and saved to
    config.project_dir/segments/.

    Args:
        config: Must match the config used during generation.
        phrase_groups: The PhraseGroups used during generation.
        raw_text: The original source text (embedded as player metadata).
                  If empty, attempts to load from the project directory.
        output_path: Override output file path. If empty, a default path
                     is created under project_dir/combined/.

    Returns:
        Path to the created output file.

    Raises:
        RuntimeError: On concatenation, normalization, or metadata failure.
    """
    from tts_audiobook_tool.app_metadata import AppMetadata
    from tts_audiobook_tool.chapter_metadata import ChapterMetadata
    from tts_audiobook_tool.concat_util import ConcatUtil
    from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
    from tts_audiobook_tool.timed_phrase import TimedPhrase
    from tts_audiobook_tool.util import delete_silently, timestamp_string

    project = _config_to_project(config)
    project.phrase_groups = phrase_groups

    # Ensure raw text is available
    if raw_text:
        project.save_raw_text(raw_text)
    else:
        raw_text = project.load_raw_text()
    if not raw_text:
        raise RuntimeError("No raw text available for player metadata")

    # Resolve output directory and stem
    if output_path:
        dest_dir = str(Path(output_path).parent)
        stem_path = str(Path(output_path).with_suffix(""))
    else:
        dest_dir = os.path.join(project.concat_path, timestamp_string())
        project_name = Path(project.dir_path).name[:20]
        stem_path = os.path.join(dest_dir, project_name)

    os.makedirs(dest_dir, exist_ok=True)

    is_aac = (project.export_type == ExportType.AAC)
    should_normalize = (project.normalization_type != NormalizationType.DISABLED)

    # Build intermediate file paths
    concat_suffix = ".m4b" if is_aac and not should_normalize else ".flac"
    concat_path = stem_path + " [concat]" + concat_suffix

    suffix = ".m4b" if is_aac else ".flac"
    norm_path = (stem_path + " [norm]" + suffix) if should_normalize else ""

    bookmark_indices: list[int] = []
    chapter_meta_path = ""
    if is_aac and bookmark_indices:
        chapter_meta_path = stem_path + " [chaptermeta]" + suffix
    final_path = stem_path + ".abr" + suffix

    intermediate_paths = [concat_path, norm_path, chapter_meta_path]

    def cleanup(keep_final: bool = False) -> None:
        for p in intermediate_paths:
            if p:
                delete_silently(p)
        if not keep_final and os.path.exists(final_path):
            delete_silently(final_path)

    # [0] Build phrases and paths
    phrases_and_paths = ConcatUtil.make_phrases_and_paths(
        project, 0, len(phrase_groups) - 1
    )

    # [1] Concatenate segments
    result = ConcatUtil.concatenate_sound_segments(
        concat_path, phrases_and_paths,
        print_progress=False,
        use_section_sound_effect=project.use_section_sound_effect,
    )
    if isinstance(result, str):
        cleanup()
        raise RuntimeError(f"Concatenation failed: {result}")
    durations = result
    last_path = concat_path

    # [2] Loudness normalization
    if norm_path:
        err = LoudnessNormalizationUtil.normalize_file(
            source_flac=last_path,
            specs=project.normalization_type.value,
            dest_path=norm_path,
        )
        if err:
            cleanup()
            raise RuntimeError(f"Loudness normalization failed: {err}")
        last_path = norm_path

    # [3] Chapter metadata (M4B)
    if chapter_meta_path:
        chapter_metadata = ChapterMetadata.make_metadata(
            project, durations, file_title=Path(stem_path).name
        )
        err = ChapterMetadata.make_copy_with_metadata(
            source_path=last_path, dest_path=chapter_meta_path,
            metadata=chapter_metadata,
        )
        if err:
            cleanup()
            raise RuntimeError(f"Chapter metadata failed: {err}")
        last_path = chapter_meta_path

    # [4] App metadata (final file)
    phrases = [item[0] for item in phrases_and_paths]
    timed_phrases = TimedPhrase.make_list_using(phrases, durations)
    app_meta = AppMetadata(
        timed_phrases=timed_phrases,
        raw_text=raw_text,
        bookmark_indices=bookmark_indices,
        has_section_break_audio=project.use_section_sound_effect,
    )
    if is_aac:
        err = AppMetadata.save_to_mp4(app_meta, last_path, final_path)
    else:
        err = AppMetadata.save_to_flac(app_meta, last_path, final_path)

    if err:
        cleanup()
        raise RuntimeError(f"Metadata embedding failed: {err}")

    cleanup(keep_final=True)
    return final_path


def create_audiobook(
    text: str,
    config: AudiobookConfig,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> AudiobookResult:
    """
    High-level: segment text, generate all audio, and concatenate.

    Args:
        text: Raw input text for the audiobook.
        config: Full generation settings.
        on_progress: Optional callback(stage, completed, total).
                     stage is "segment", "generate", or "concatenate".

    Returns:
        AudiobookResult with output path and per-segment details.
    """
    if not config.project_dir:
        raise ValueError("config.project_dir is required")

    os.makedirs(config.project_dir, exist_ok=True)

    # 1. Segment
    if on_progress:
        on_progress("segment", 0, 1)

    phrase_groups = segment_text(
        text,
        language_code=config.language_code,
        max_words=config.max_words_per_segment,
        strategy=config.segmentation_strategy,
    )

    if on_progress:
        on_progress("segment", 1, 1)

    if not phrase_groups:
        return AudiobookResult(
            output_path="", error="Text produced no segments",
            generation_results=[], num_segments=0,
            num_succeeded=0, num_failed=0,
        )

    # Save phrase groups and raw text to project
    project = _config_to_project(config)
    project.set_phrase_groups_and_save(
        phrase_groups=phrase_groups,
        strategy=SegmentationStrategy.from_id(config.segmentation_strategy)
            or SegmentationStrategy.NORMAL,
        max_words=config.max_words_per_segment,
        language_code=config.language_code,
        raw_text=text,
    )

    # 2. Generate
    def gen_progress(completed: int, total: int, result: GenerationResult) -> None:
        if on_progress:
            on_progress("generate", completed, total)

    gen_results = generate_all(
        phrase_groups, config, save_segments=True, on_progress=gen_progress,
    )

    num_succeeded = sum(1 for r in gen_results if not r.error and r.validation_passed)
    num_failed = len(gen_results) - num_succeeded

    # 3. Concatenate
    if on_progress:
        on_progress("concatenate", 0, 1)

    try:
        output_path = concatenate(config, phrase_groups, raw_text=text)
    except RuntimeError as e:
        return AudiobookResult(
            output_path="", error=str(e),
            generation_results=gen_results,
            num_segments=len(phrase_groups),
            num_succeeded=num_succeeded, num_failed=num_failed,
        )

    if on_progress:
        on_progress("concatenate", 1, 1)

    return AudiobookResult(
        output_path=output_path, error="",
        generation_results=gen_results,
        num_segments=len(phrase_groups),
        num_succeeded=num_succeeded, num_failed=num_failed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_to_project(config: AudiobookConfig) -> "Project":
    """Build a Project from AudiobookConfig without going through State."""
    from tts_audiobook_tool.project import Project

    project = Project(config.project_dir)

    # Text processing
    project.language_code = config.language_code
    project.max_words = config.max_words_per_segment
    project.segmentation_strategy = (
        SegmentationStrategy.from_id(config.segmentation_strategy)
        or SegmentationStrategy.NORMAL
    )
    project.word_substitutions = dict(config.word_substitutions)

    # Kokoro
    project.kokoro_voice = config.kokoro_voice
    project.kokoro_speed = config.kokoro_speed
    project.kokoro_lang = config.kokoro_lang
    project.kokoro_model_path = config.kokoro_model_path
    project.kokoro_voices_path = config.kokoro_voices_path

    # Generation
    project.max_retries = config.max_retries

    # Validation
    strictness = Strictness.get_by_id(config.strictness)
    if strictness is not None:
        project.strictness = strictness

    # Output
    export_type = ExportType.get_by_id(config.export_type)
    if export_type is not None:
        project.export_type = export_type

    normalization = NormalizationType.from_id(config.normalization)
    if normalization is not None:
        project.normalization_type = normalization

    project.use_section_sound_effect = config.use_section_sound_effect

    chapter_mode = ChapterMode.get_by_id(config.chapter_mode)
    if chapter_mode is not None:
        project.chapter_mode = chapter_mode

    # Voice clone (for non-Kokoro models)
    if config.voice_clone_path:
        _apply_voice_clone(project, config)

    # Advanced overrides
    for key, value in config.model_overrides.items():
        if hasattr(project, key):
            setattr(project, key, value)

    return project


def _apply_voice_clone(project: "Project", config: AudiobookConfig) -> None:
    """Copy voice clone file into project and set the appropriate field."""
    from tts_audiobook_tool.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

    tts_type = Tts.get_type()

    # Skip models that don't use audio voice cloning
    # (Kokoro uses named presets, Oute uses a JSON voice file)
    if tts_type in (TtsModelInfos.KOKORO, TtsModelInfos.OUTE, TtsModelInfos.NONE):
        return

    voice_attr = tts_type.value.voice_file_name_attr
    if not voice_attr:
        return

    result = SoundFileUtil.load(config.voice_clone_path)
    if isinstance(result, str):
        return  # silently skip if voice file can't be loaded

    stem = Path(config.voice_clone_path).stem
    project.set_voice_and_save(
        source_sound=result,
        voice_file_stem=stem,
        transcript=config.voice_clone_transcript,
        tts_type=tts_type,
    )


def _resolve_stt_config(config: AudiobookConfig) -> SttConfig:
    """Resolve STT device/compute settings."""
    if config.stt_device == "cuda":
        return SttConfig.CUDA_FLOAT16
    elif config.stt_device == "cpu":
        return SttConfig.CPU_INT8FLOAT32
    else:
        return SttConfig.get_default()


def _validate_sound(
    sound: Sound,
    phrase_group: PhraseGroup,
    config: AudiobookConfig,
) -> tuple[bool, str, Sound | None]:
    """
    Run STT validation on generated audio.

    Returns (passed, message, possibly_trimmed_sound).
    """
    from tts_audiobook_tool.validate_util import ValidateUtil
    from tts_audiobook_tool.whisper_util import WhisperUtil
    from tts_audiobook_tool.util import strip_ansi_codes

    # Check language support
    if ValidateUtil.is_unsupported_language_code(config.language_code):
        return True, "Unsupported language for validation", None

    stt_variant = SttVariant.get_by_id(config.stt_variant) or SttVariant.LARGE_V3
    stt_config = _resolve_stt_config(config)

    words_result = WhisperUtil.transcribe_to_words(
        sound, config.language_code, stt_variant, stt_config,
    )
    if isinstance(words_result, str):
        return False, f"Transcription error: {words_result}", None

    text = phrase_group.as_flattened_phrase().text
    strictness = Strictness.get_by_id(config.strictness) or Strictness.MODERATE

    val_result = ValidateUtil.validate(
        sound, text, words_result, config.language_code, strictness,
    )

    passed = not val_result.is_fail
    message = strip_ansi_codes(val_result.get_ui_message())
    return passed, message, val_result.sound


def _save_segment(
    project: "Project",
    index: int,
    phrase_group: PhraseGroup,
    result: GenerationResult,
) -> str:
    """Save a generated sound segment to the project directory. Returns path."""
    from tts_audiobook_tool.sound_file_util import SoundFileUtil
    from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
    from tts_audiobook_tool.tts import Tts
    from tts_audiobook_tool.validation_result import SkippedResult

    if result.sound is None:
        return ""

    # Build a minimal ValidationResult for file naming
    validation_result = SkippedResult(sound=result.sound, message="api")

    file_name = SoundSegmentUtil.make_file_name(
        index=index,
        phrase_group=phrase_group,
        project=project,
        tts_model_info=Tts.get_type().value,
        validation_result=validation_result,
        is_real_time=False,
    )

    dir_path = project.sound_segments_path
    os.makedirs(dir_path, exist_ok=True)

    sound_path = os.path.join(dir_path, file_name)
    err = SoundFileUtil.save_flac(result.sound, sound_path)
    if err:
        return ""

    return sound_path
