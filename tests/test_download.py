"""Tests for download.py — VTT caption cleaning and media download logic."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from transcribe_critic.shared import SpeechConfig, SpeechData

from transcribe_critic.download import clean_vtt_captions, download_media


# ---------------------------------------------------------------------------
# clean_vtt_captions
# ---------------------------------------------------------------------------

class TestCleanVttCaptions:
    def test_basic_vtt(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n"
            "\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world.\n"
            "\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "This is a test.\n"
        )
        result = clean_vtt_captions(vtt)
        assert "Hello world." in result
        assert "This is a test." in result
        assert "WEBVTT" not in result
        assert "-->" not in result

    def test_deduplicates_consecutive_captions(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world.\n\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "Hello world.\n"
        )
        result = clean_vtt_captions(vtt)
        assert result.count("Hello world.") == 1

    def test_preserves_non_consecutive_repeated_captions(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Please remember this.\n\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "A different thought.\n\n"
            "00:00:05.000 --> 00:00:07.000\n"
            "Please remember this.\n"
        )
        result = clean_vtt_captions(vtt)
        assert result == (
            "Please remember this. A different thought. Please remember this."
        )

    def test_short_form_cue_timestamps(self, tmp_path):
        """WebVTT allows MM:SS.mmm cues with no hour field. These miss the
        HH:MM:SS timestamp pattern and must still be dropped as cue lines."""
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:01.000 --> 00:03.000\n"
            "Hello world.\n\n"
            "00:03.000 --> 00:05.000\n"
            "Second line.\n"
        )
        result = clean_vtt_captions(vtt)
        assert result == "Hello world. Second line."
        assert "-->" not in result

    def test_strips_html_tags(self, tmp_path):
        vtt = tmp_path / "captions.vtt"
        vtt.write_text(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "<c>Hello</c> <i>world</i>.\n"
        )
        result = clean_vtt_captions(vtt)
        assert "<c>" not in result
        assert "<i>" not in result
        assert "Hello world." in result


# ---------------------------------------------------------------------------
# download_media
# ---------------------------------------------------------------------------

def _mock_run_command_for_download(cmd, desc, verbose=False):
    """Mock run_command that returns fake yt-dlp JSON for --dump-json."""
    if "--dump-json" in cmd:
        return MagicMock(stdout=json.dumps({
            "title": "Test Video",
            "id": "abc123",
            "channel": "Test Channel",
            "upload_date": "20240101",
            "duration": 300,
            "description": "A test video",
        }))
    return MagicMock(stdout="", stderr="")


class TestDownloadMedia:
    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_downloads_audio_video_captions(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        assert data.title == "Test Video"
        assert data.audio_path == tmp_path / "audio.mp3"
        assert data.video_path == tmp_path / "video.mp4"
        # Should have called run_command for: info, audio, video, captions
        assert mock_run.call_count == 4

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_reuses_existing_audio(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=True)
        # Pre-create audio file
        (tmp_path / "audio.mp3").write_text("fake audio")
        data = SpeechData()
        download_media(config, data)
        # Should NOT have called yt-dlp for audio download (only info, video, captions)
        audio_calls = [c for c in mock_run.call_args_list if "-x" in c[0][0]]
        assert len(audio_calls) == 0

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_no_slides_skips_video(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "Skipping video download" in out
        assert data.video_path is None

    @patch("transcribe_critic.download.run_command")
    def test_captions_failure_is_graceful(self, mock_run, tmp_path):
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "--write-auto-sub" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False)
        data = SpeechData()
        # Should not raise
        download_media(config, data)
        assert data.captions_path is None

    @patch("transcribe_critic.download.run_command")
    def test_audio_failure_still_fetches_captions(self, mock_run, tmp_path):
        """A failed media fetch must not prevent captions from being retrieved.

        Extraction can succeed while the media download fails (stale signatures →
        HTTP 403, unavailable format, ffmpeg postprocessing), and captions come
        from a different endpoint that is still reachable.
        """
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "-x" in cmd:  # audio extraction
                raise subprocess.CalledProcessError(1, cmd)
            if "--write-sub" in cmd:
                (tmp_path / "captions.en.vtt").write_text("WEBVTT\n")
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=False)
        data = SpeechData()

        # The stage still fails, so downstream steps do not run on missing audio...
        with pytest.raises(subprocess.CalledProcessError):
            download_media(config, data)

        # ...but the captions were fetched and recorded before it did.
        assert (tmp_path / "captions.en.vtt").exists()
        assert data.captions_path == tmp_path / "captions.en.vtt"

    @patch("transcribe_critic.download.run_command")
    def test_video_failure_still_fetches_captions(self, mock_run, tmp_path):
        """Video download has the same independence from captions as audio."""
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "-f" in cmd and "mp4" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            if "--write-sub" in cmd:
                (tmp_path / "captions.en.vtt").write_text("WEBVTT\n")
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False)
        data = SpeechData()

        with pytest.raises(subprocess.CalledProcessError):
            download_media(config, data)

        assert data.captions_path == tmp_path / "captions.en.vtt"

    @patch("transcribe_critic.download.run_command")
    def test_both_media_failures_still_fetch_captions(self, mock_run, tmp_path):
        """Audio and video both failing must still leave captions on disk."""
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "-x" in cmd or ("-f" in cmd and "mp4" in cmd):
                raise subprocess.CalledProcessError(1, cmd)
            if "--write-sub" in cmd:
                (tmp_path / "captions.en.vtt").write_text("WEBVTT\n")
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False)
        data = SpeechData()

        with pytest.raises(subprocess.CalledProcessError):
            download_media(config, data)

        assert data.captions_path == tmp_path / "captions.en.vtt"

    @patch("transcribe_critic.download.run_command")
    def test_captions_attempted_even_when_audio_fails_and_captions_absent(
            self, mock_run, tmp_path, capsys):
        """A failing audio fetch must not stop the captions call from being made,
        even when the video turns out to have no captions either."""
        def side_effect(cmd, desc, verbose=False):
            if "--dump-json" in cmd:
                return MagicMock(stdout=json.dumps({"title": "T", "id": "x"}))
            if "-x" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            if "--write-sub" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock(stdout="", stderr="")
        mock_run.side_effect = side_effect
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=False)
        data = SpeechData()

        with pytest.raises(subprocess.CalledProcessError):
            download_media(config, data)

        assert any("--write-sub" in c[0][0] for c in mock_run.call_args_list)
        assert data.captions_path is None
        assert "No captions available" in capsys.readouterr().out

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_reuses_cached_metadata(self, mock_run, tmp_path, capsys):
        """With skip_existing, metadata.json is read from disk instead of re-fetched."""
        cached = {"url": "https://example.com/v", "title": "Cached Title",
                  "video_id": "abc123", "channel": "C"}
        (tmp_path / "metadata.json").write_text(json.dumps(cached))
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=True)
        data = SpeechData()
        download_media(config, data)

        assert data.title == "Cached Title"
        assert data.metadata["video_id"] == "abc123"
        assert "Reusing cached metadata" in capsys.readouterr().out
        assert not any("--dump-json" in c[0][0] for c in mock_run.call_args_list)

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_reuses_existing_video(self, mock_run, tmp_path):
        """A present video.mp4 is not re-downloaded under skip_existing."""
        (tmp_path / "video.mp4").write_text("fake video")
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=True)
        data = SpeechData()
        download_media(config, data)

        video_calls = [c for c in mock_run.call_args_list if "mp4" in c[0][0]]
        assert len(video_calls) == 0
        assert data.video_path == tmp_path / "video.mp4"

    @patch("transcribe_critic.download.run_command")
    def test_prefers_uploaded_caption_track(self, mock_run, tmp_path):
        """--write-sub must precede --write-auto-sub so yt-dlp takes the
        creator-uploaded track when one exists."""
        mock_run.side_effect = _mock_run_command_for_download
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=True, skip_existing=False)
        download_media(config, SpeechData())
        caption_cmd = next(c[0][0] for c in mock_run.call_args_list
                           if "--skip-download" in c[0][0])
        assert caption_cmd.index("--write-sub") < caption_cmd.index("--write-auto-sub")

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_dry_run_skips_downloads(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              dry_run=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        # Only the info fetch should run (not dry-run gated)
        assert mock_run.call_count == 1
        # Dry-run should not create any files
        assert not (tmp_path / "metadata.json").exists()

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_saves_metadata(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["title"] == "Test Video"
        assert metadata["url"] == "https://example.com/v"
        assert metadata["video_id"] == "abc123"

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_external_transcript_in_metadata(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              skip_existing=False,
                              external_transcript="https://example.com/transcript.txt")
        data = SpeechData()
        download_media(config, data)
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert metadata["external_transcript"] == "https://example.com/transcript.txt"

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_passes_cookies_from_browser_to_all_calls(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False,
                              cookies_from_browser="safari")
        data = SpeechData()
        download_media(config, data)
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            assert "--cookies-from-browser" in cmd, f"missing cookies in: {cmd}"
            assert cmd[cmd.index("--cookies-from-browser") + 1] == "safari"

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_omits_cookies_when_unset(self, mock_run, tmp_path):
        config = SpeechConfig(url="https://example.com/v", output_dir=tmp_path,
                              no_slides=False, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        for call in mock_run.call_args_list:
            assert "--cookies-from-browser" not in call[0][0]

    @patch("transcribe_critic.download.run_command", side_effect=_mock_run_command_for_download)
    def test_podcast_skips_video_and_captions(self, mock_run, tmp_path, capsys):
        config = SpeechConfig(url="https://example.com/podcast/ep1", output_dir=tmp_path,
                              podcast=True, no_slides=True, skip_existing=False)
        data = SpeechData()
        download_media(config, data)
        out = capsys.readouterr().out
        assert "Skipping video download (--podcast)" in out
        assert "Skipping captions download (--podcast)" in out
        assert data.audio_path == tmp_path / "audio.mp3"
        assert data.video_path is None
        # info + audio only (no video, no captions)
        assert mock_run.call_count == 2
