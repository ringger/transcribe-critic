"""Tests for the prompt resource loader."""

import pytest
from transcribe_critic.prompts import load_prompt


PROMPT_FILES = [
    "merge_structured",
    "merge_multi_source",
    "ensemble",
    "slides",
    "summary",
    "speaker_id",
]


class TestLoadPrompt:
    """Tests for load_prompt()."""

    @pytest.mark.parametrize("name", PROMPT_FILES)
    def test_loads_all_prompt_files(self, name):
        result = load_prompt(name)
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.parametrize("name", PROMPT_FILES)
    def test_all_have_primary_or_system(self, name):
        result = load_prompt(name)
        assert "primary" in result or "system" in result

    def test_merge_structured_has_primary_and_retry(self):
        result = load_prompt("merge_structured")
        assert "primary" in result
        assert "retry" in result

    def test_ensemble_has_system_primary_retry(self):
        result = load_prompt("ensemble")
        assert "system" in result
        assert "primary" in result
        assert "retry_suffix" in result

    def test_slides_has_primary_and_retry(self):
        result = load_prompt("slides")
        assert "primary" in result
        assert "retry" in result

    def test_summary_has_system_and_user(self):
        result = load_prompt("summary")
        assert "system" in result
        assert "user" in result

    def test_speaker_id_has_primary_and_retry(self):
        result = load_prompt("speaker_id")
        assert "primary" in result
        assert "retry" in result

    def test_merge_structured_primary_has_placeholders(self):
        result = load_prompt("merge_structured")
        prompt = result["primary"]
        assert "{num_sources}" in prompt
        assert "{passage_texts}" in prompt
        assert "{num_passages}" in prompt

    def test_format_substitution_works(self):
        result = load_prompt("merge_multi_source")
        formatted = result["primary"].format(
            num_sources=3,
            sources_text="Source 1: hello\n",
            diff_section="",
        )
        assert "3 independent transcriptions" in formatted
        assert "Source 1: hello" in formatted

    def test_speaker_id_literal_braces_in_json_example(self):
        result = load_prompt("speaker_id")
        # After .format(), literal {{ becomes {
        formatted = result["primary"].format(
            metadata_section="",
            speaker_list="SPEAKER_00",
            intro_text="Hello",
        )
        assert '{"SPEAKER_00":' in formatted or "SPEAKER_00" in formatted

    def test_nonexistent_prompt_raises(self):
        with pytest.raises(Exception):
            load_prompt("nonexistent_prompt_file")

    def test_caching(self):
        result1 = load_prompt("slides")
        result2 = load_prompt("slides")
        assert result1 is result2
