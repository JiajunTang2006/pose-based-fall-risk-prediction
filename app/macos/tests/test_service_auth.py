"""Tests for the Bearer-token authentication module."""

import unittest

from fall_prediction_service.auth import validate_token


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.token = "my-secret-token-123"

    def test_valid_token_accepted(self):
        self.assertTrue(validate_token("Bearer my-secret-token-123", self.token))

    def test_missing_header_rejected(self):
        self.assertFalse(validate_token(None, self.token))

    def test_empty_header_rejected(self):
        self.assertFalse(validate_token("", self.token))

    def test_missing_prefix_rejected(self):
        self.assertFalse(validate_token("my-secret-token-123", self.token))

    def test_wrong_token_rejected(self):
        self.assertFalse(validate_token("Bearer wrong-token", self.token))

    def test_partial_match_rejected(self):
        self.assertFalse(validate_token("Bearer my-secret-token-12", self.token))

    def test_extra_chars_rejected(self):
        self.assertFalse(
            validate_token("Bearer my-secret-token-123-extra", self.token)
        )

    def test_lowercase_bearer_rejected(self):
        # The prefix is case-sensitive: "Bearer " not "bearer "
        self.assertFalse(validate_token("bearer my-secret-token-123", self.token))

    def test_token_with_special_chars(self):
        tok = "abc/def+gHi="
        self.assertTrue(validate_token(f"Bearer {tok}", tok))

    def test_long_token(self):
        tok = "x" * 256
        self.assertTrue(validate_token(f"Bearer {tok}", tok))
        self.assertFalse(validate_token(f"Bearer {tok[:-1]}", tok))


if __name__ == "__main__":
    unittest.main()
