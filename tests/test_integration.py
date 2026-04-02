"""End-to-end integration test using a 30-second audio clip.

This test runs actual ASR models on real audio, so it requires:
- mlx-whisper (for distil-large-v3)
- A machine with enough RAM for the model

Run with: pytest tests/test_integration.py -m integration
Skip in normal runs: pytest tests/ -m "not integration"
"""

import shutil
from pathlib import Path

import pytest

from transcribe_critic.shared import SpeechConfig, SpeechData, ASR_MERGED_TXT

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CLIP_PATH = FIXTURE_DIR / "tao_30s.mp3"


def _has_mlx_whisper():
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pyannote():
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ollama():
    """Check if Ollama is running and responsive."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/v1/models")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.integration
class TestEndToEndPipeline:
    """Run real ASR on a 30s clip and verify outputs."""

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    def test_single_model_transcribe(self, tmp_path):
        """Transcribe with a single Whisper model and verify output files."""
        from transcribe_critic.transcription import transcribe_audio
        from transcribe_critic.shared import check_dependencies

        # Set up output dir with audio
        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3"],
            skip_existing=False,
            no_llm=True,
            diarize=False,
            podcast=True,
        )
        data = SpeechData(audio_path=audio)

        transcribe_audio(config, data)

        # Verify transcript file was created with unified naming
        txt = tmp_path / "asr_distil-large-v3.txt"
        json_file = tmp_path / "asr_distil-large-v3.json"
        assert txt.exists(), f"Expected {txt.name} to exist"
        assert json_file.exists(), f"Expected {json_file.name} to exist"

        # Verify transcript has real content (not empty or hallucination)
        text = txt.read_text()
        words = text.split()
        assert len(words) > 10, f"Transcript too short: {len(words)} words"
        assert len(words) < 500, f"Transcript suspiciously long: {len(words)} words"

        # Verify JSON has segments with timestamps
        import json
        meta = json.loads(json_file.read_text())
        assert "segments" in meta
        assert len(meta["segments"]) > 0
        assert "start" in meta["segments"][0]
        assert "end" in meta["segments"][0]

        # Verify it was registered in data
        assert "distil-large-v3" in data.asr_transcripts
        assert data.transcript_path == txt

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    def test_multimodel_ensemble_no_llm(self, tmp_path):
        """Transcribe with two models and ensemble (--no-llm uses base directly)."""
        from transcribe_critic.transcription import transcribe_audio, _ensemble_asr_transcripts

        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3", "small"],
            skip_existing=False,
            no_llm=True,
            diarize=False,
            podcast=True,
        )
        data = SpeechData(audio_path=audio)

        # Transcribe both models
        transcribe_audio(config, data)

        assert "distil-large-v3" in data.asr_transcripts
        assert "small" in data.asr_transcripts

        # Ensemble (no-llm mode just picks the base model)
        _ensemble_asr_transcripts(config, data)

        merged = tmp_path / ASR_MERGED_TXT
        assert merged.exists(), f"Expected {ASR_MERGED_TXT} to exist"

        merged_text = merged.read_text()
        assert len(merged_text.split()) > 10

        # In no-llm mode, merged should equal the higher-ranked model (distil-large-v3)
        distil_text = (tmp_path / "asr_distil-large-v3.txt").read_text()
        assert merged_text == distil_text

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    def test_hydrate_finds_transcribed_outputs(self, tmp_path):
        """After transcribe, _hydrate_data discovers the output files."""
        from transcribe_critic.transcription import transcribe_audio
        from transcribe_critic.transcriber import _hydrate_data

        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3"],
            skip_existing=False,
            no_llm=True,
            diarize=False,
            podcast=True,
        )
        data = SpeechData(audio_path=audio)
        transcribe_audio(config, data)

        # Now hydrate a fresh SpeechData from disk (simulates --steps ensemble)
        fresh_data = SpeechData()
        _hydrate_data(config, fresh_data)

        assert "distil-large-v3" in fresh_data.asr_transcripts
        assert fresh_data.transcript_path is not None
        assert fresh_data.transcript_path.exists()
        assert fresh_data.transcript_json_path is not None
        assert len(fresh_data.transcript_segments) > 0

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    @pytest.mark.skipif(not _has_pyannote(), reason="pyannote not installed")
    def test_transcribe_then_diarize(self, tmp_path):
        """Full transcribe → diarize pipeline on real audio."""
        from transcribe_critic.transcription import transcribe_audio
        from transcribe_critic.diarization import diarize_audio

        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3"],
            skip_existing=False,
            no_llm=True,
            diarize=True,
            podcast=True,
        )
        data = SpeechData(audio_path=audio)

        # Stage 1: Transcribe
        transcribe_audio(config, data)
        assert len(data.transcript_segments) > 0, "Segments should be loaded after transcribe"

        # Stage 2: Diarize
        diarize_audio(config, data)

        diarized = tmp_path / "diarized.txt"
        assert diarized.exists(), "Diarized transcript should be written"
        diarized_text = diarized.read_text()
        assert len(diarized_text) > 0
        # Should have speaker labels and timestamps
        assert "SPEAKER_" in diarized_text or "[" in diarized_text

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    def test_transcribe_then_markdown(self, tmp_path):
        """Transcribe → markdown output (no LLM merge needed)."""
        from transcribe_critic.transcription import transcribe_audio
        from transcribe_critic.output import generate_markdown

        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3"],
            skip_existing=False,
            no_llm=True,
            diarize=False,
            podcast=True,
            title="Integration Test",
        )
        data = SpeechData(audio_path=audio, title="Integration Test")

        transcribe_audio(config, data)
        generate_markdown(config, data)

        md = tmp_path / "transcript.md"
        assert md.exists()
        md_text = md.read_text()
        assert "Integration Test" in md_text
        assert len(md_text.split()) > 20

    @pytest.mark.skipif(not _has_mlx_whisper(), reason="mlx-whisper not installed")
    @pytest.mark.skipif(not _has_ollama(), reason="Ollama not running")
    def test_multimodel_ensemble_with_local_llm(self, tmp_path):
        """Transcribe two models and ensemble with local LLM adjudication."""
        from transcribe_critic.transcription import transcribe_audio, _ensemble_asr_transcripts

        audio = tmp_path / "audio.mp3"
        shutil.copy(CLIP_PATH, audio)

        config = SpeechConfig(
            url="local",
            output_dir=tmp_path,
            models=["distil-large-v3", "small"],
            skip_existing=False,
            no_llm=False,
            local=True,
            diarize=False,
            podcast=True,
        )
        data = SpeechData(audio_path=audio)

        # Transcribe both models
        transcribe_audio(config, data)
        assert "distil-large-v3" in data.asr_transcripts
        assert "small" in data.asr_transcripts

        # Ensemble with local LLM
        _ensemble_asr_transcripts(config, data)

        merged = tmp_path / ASR_MERGED_TXT
        assert merged.exists(), f"Expected {ASR_MERGED_TXT}"
        merged_text = merged.read_text()
        assert len(merged_text.split()) > 10

        # Checkpoints should have been saved
        ensemble_dir = tmp_path / "ensemble_chunks"
        assert ensemble_dir.exists()
        checkpoints = list(ensemble_dir.glob("cluster_*.json"))
        # May be 0 if transcripts are identical, but dir should exist
        assert ensemble_dir.exists()
