"""Student-visible tests for compact KV-cache behavior."""

from . import test_causal_self_attention as shared_tests


def check_cached_and_full_attention_match() -> None:
    shared_tests.test_cached_outputs_match_full_sequence_for_all_attention_types()


def check_cached_chunk_mask_uses_absolute_positions() -> None:
    shared_tests.test_cached_chunk_uses_offset_causal_mask()


def check_cache_keeps_compact_kv_heads() -> None:
    shared_tests.test_cache_stores_compact_kv_heads()


def check_cache_storage_tracks_kv_head_count() -> None:
    shared_tests.test_cache_storage_scales_linearly_with_kv_heads()


def check_cache_is_detached_from_autograd() -> None:
    shared_tests.test_cache_does_not_retain_autograd_history()


def check_cache_validation() -> None:
    shared_tests.test_attention_validation_and_cache_errors()


if __name__ == "__main__":
    check_cached_and_full_attention_match()
    check_cached_chunk_mask_uses_absolute_positions()
    check_cache_keeps_compact_kv_heads()
    check_cache_storage_tracks_kv_head_count()
    check_cache_is_detached_from_autograd()
    check_cache_validation()
    print("All KV-cache tests passed.")
