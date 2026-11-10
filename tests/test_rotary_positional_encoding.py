"""Student-visible tests for rotary positional encoding."""

from . import test_causal_self_attention as shared_tests


def check_rotary_values_and_offset() -> None:
    shared_tests.test_rotary_positional_encoding_values_and_offset()


def check_rotary_validation() -> None:
    shared_tests.test_rotary_positional_encoding_validation()


if __name__ == "__main__":
    check_rotary_values_and_offset()
    check_rotary_validation()
    print("All rotary positional encoding tests passed.")
