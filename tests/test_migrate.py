"""Tests for the migration utility (whisper_* → asr_* renaming)."""

from pathlib import Path

from transcribe_critic.migrate import migrate_directory


class TestMigrateDirectory:
    def test_renames_whisper_files(self, tmp_path):
        (tmp_path / "whisper_small.txt").write_text("small")
        (tmp_path / "whisper_small.json").write_text("{}")
        (tmp_path / "whisper_medium.txt").write_text("medium")

        renames = migrate_directory(tmp_path)
        assert len(renames) == 3
        assert (tmp_path / "asr_small.txt").read_text() == "small"
        assert (tmp_path / "asr_small.json").read_text() == "{}"
        assert (tmp_path / "asr_medium.txt").read_text() == "medium"
        assert not (tmp_path / "whisper_small.txt").exists()

    def test_renames_whisper_merged(self, tmp_path):
        (tmp_path / "whisper_merged.txt").write_text("merged")
        renames = migrate_directory(tmp_path)
        assert len(renames) == 1
        assert (tmp_path / "asr_merged.txt").read_text() == "merged"
        assert not (tmp_path / "whisper_merged.txt").exists()

    def test_skips_when_target_exists(self, tmp_path, capsys):
        """When asr_* already exists, whisper_* is not migrated."""
        (tmp_path / "whisper_small.txt").write_text("old")
        (tmp_path / "asr_small.txt").write_text("new")
        renames = migrate_directory(tmp_path)
        assert len(renames) == 0
        # Original should be untouched
        assert (tmp_path / "whisper_small.txt").read_text() == "old"
        assert (tmp_path / "asr_small.txt").read_text() == "new"

    def test_dry_run_no_changes(self, tmp_path, capsys):
        (tmp_path / "whisper_small.txt").write_text("small")
        renames = migrate_directory(tmp_path, dry_run=True)
        assert len(renames) == 1
        # File should NOT have been renamed
        assert (tmp_path / "whisper_small.txt").exists()
        assert not (tmp_path / "asr_small.txt").exists()
        out = capsys.readouterr().out
        assert "[dry-run]" in out

    def test_leaves_asr_files_alone(self, tmp_path):
        """Already-migrated asr_* files should not be touched."""
        (tmp_path / "asr_parakeet.txt").write_text("parakeet")
        renames = migrate_directory(tmp_path)
        assert len(renames) == 0
        assert (tmp_path / "asr_parakeet.txt").read_text() == "parakeet"

    def test_empty_directory(self, tmp_path):
        renames = migrate_directory(tmp_path)
        assert len(renames) == 0

    def test_ignores_whisper_merged_variants(self, tmp_path):
        """whisper_merged_7b.txt etc. should be skipped (they contain 'merged')."""
        (tmp_path / "whisper_merged_7b.txt").write_text("variant")
        renames = migrate_directory(tmp_path)
        assert len(renames) == 0
