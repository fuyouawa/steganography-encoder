"""
Unit tests for encoding utilities.
"""

import unittest
import numpy as np
import torch

# Import the encoding utilities
from utils.encoding import (
    encode_steganography,
    decode_steganography,
    encode_bytes,
    decode_bytes
)


class TestEncodingUtilities(unittest.TestCase):
    """Test encoding utilities."""

    def test_encode_decode_steganography_basic(self):
        """Test basic steganography encoding and decoding."""
        # Test data
        test_data = b"Hello, World! This is a test message."

        # Encode data
        encoded_image = encode_steganography(test_data)

        # Verify image properties
        self.assertIsInstance(encoded_image, torch.Tensor)
        self.assertEqual(encoded_image.dtype, torch.float32)
        self.assertEqual(encoded_image.shape[0], 1)  # batch dimension
        self.assertEqual(encoded_image.shape[3], 4)  # RGBA channels by default

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_decode_steganography_rgb(self):
        """Test steganography encoding and decoding with RGB format."""
        # Test data
        test_data = b"RGB format test"

        # Encode data with RGB format
        encoded_image = encode_steganography(test_data, use_alpha=False)

        # Verify image properties
        self.assertIsInstance(encoded_image, torch.Tensor)
        self.assertEqual(encoded_image.dtype, torch.float32)
        self.assertEqual(encoded_image.shape[0], 1)  # batch dimension
        self.assertEqual(encoded_image.shape[3], 3)  # RGB channels

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_decode_steganography_custom_dimensions(self):
        """Test steganography encoding and decoding with custom dimensions."""
        # Test data
        test_data = b"Custom dimensions test"

        # Encode data with custom dimensions
        encoded_image = encode_steganography(test_data, width=32, height=32)

        # Verify image properties
        self.assertIsInstance(encoded_image, torch.Tensor)
        self.assertEqual(encoded_image.shape[1], 32)  # height
        self.assertEqual(encoded_image.shape[2], 32)  # width

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_decode_steganography_empty_data(self):
        """Test steganography encoding and decoding with empty data."""
        # Test empty data
        test_data = b""

        # Encode data
        encoded_image = encode_steganography(test_data)

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_decode_steganography_large_data(self):
        """Test steganography encoding and decoding with large data."""
        # Test large data
        test_data = b"x" * 1000

        # Encode data
        encoded_image = encode_steganography(test_data)

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_decode_steganography_custom_margins(self):
        """Test steganography encoding and decoding with custom margins."""
        # Test data
        test_data = b"Custom margins test"

        # Encode data with custom margins
        encoded_image = encode_steganography(
            test_data,
            top_margin_ratio=0.1,
            bottom_margin_ratio=0.15
        )

        # Decode data with same margins
        decoded_data = decode_steganography(
            encoded_image,
            top_margin_ratio=0.1,
            bottom_margin_ratio=0.15
        )

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_encode_steganography_data_too_large(self):
        """Test steganography encoding with data that's too large."""
        # Test data that's too large for the specified dimensions
        test_data = b"x" * 1000

        # Attempt to encode with small dimensions
        with self.assertRaises(ValueError):
            encode_steganography(test_data, width=10, height=10)

    def test_decode_steganography_invalid_image(self):
        """Test steganography decoding with invalid image."""
        # Test with grayscale image (should fail)
        grayscale_image = torch.rand((1, 32, 32), dtype=torch.float32)

        with self.assertRaises(ValueError):
            decode_steganography(grayscale_image)

    def test_encode_bytes_binary(self):
        """Test bytes encoding in binary format."""
        test_data = b"hello"
        encoded = encode_bytes(test_data, "binary")

        # Verify format
        self.assertIsInstance(encoded, str)
        self.assertEqual(len(encoded), len(test_data) * 8)

        # Verify round-trip
        decoded = decode_bytes(encoded, "binary")
        self.assertEqual(decoded, test_data)

    def test_encode_bytes_octal(self):
        """Test bytes encoding in octal format."""
        test_data = b"test"
        encoded = encode_bytes(test_data, "octal")

        # Verify format
        self.assertIsInstance(encoded, str)
        self.assertEqual(len(encoded), len(test_data) * 3)

        # Verify round-trip
        decoded = decode_bytes(encoded, "octal")
        self.assertEqual(decoded, test_data)

    def test_encode_bytes_decimal(self):
        """Test bytes encoding in decimal format."""
        test_data = b"123"
        encoded = encode_bytes(test_data, "decimal")

        # Verify format
        self.assertIsInstance(encoded, str)

        # Verify round-trip
        decoded = decode_bytes(encoded, "decimal")
        self.assertEqual(decoded, test_data)

    def test_encode_bytes_hexadecimal(self):
        """Test bytes encoding in hexadecimal format."""
        test_data = b"hello world"
        encoded = encode_bytes(test_data, "hexadecimal")

        # Verify format
        self.assertIsInstance(encoded, str)
        self.assertEqual(len(encoded), len(test_data) * 2)

        # Verify round-trip
        decoded = decode_bytes(encoded, "hexadecimal")
        self.assertEqual(decoded, test_data)

    def test_encode_bytes_base64(self):
        """Test bytes encoding in base64 format."""
        test_data = b"test data for base64"
        encoded = encode_bytes(test_data, "base64")

        # Verify format
        self.assertIsInstance(encoded, str)

        # Verify round-trip
        decoded = decode_bytes(encoded, "base64")
        self.assertEqual(decoded, test_data)

    def test_encode_bytes_unsupported_base(self):
        """Test bytes encoding with unsupported base."""
        test_data = b"test"

        with self.assertRaises(ValueError):
            encode_bytes(test_data, "unsupported")

    def test_decode_bytes_invalid_binary(self):
        """Test bytes decoding with invalid binary string."""
        with self.assertRaises(ValueError):
            decode_bytes("0101010", "binary")  # Length not multiple of 8

    def test_decode_bytes_invalid_octal(self):
        """Test bytes decoding with invalid octal string."""
        with self.assertRaises(ValueError):
            decode_bytes("12345", "octal")  # Length not multiple of 3

    def test_decode_bytes_invalid_hexadecimal(self):
        """Test bytes decoding with invalid hexadecimal string."""
        with self.assertRaises(ValueError):
            decode_bytes("abc", "hexadecimal")  # Length not even

    def test_decode_bytes_unsupported_base(self):
        """Test bytes decoding with unsupported base."""
        with self.assertRaises(ValueError):
            decode_bytes("test", "unsupported")

    def test_decode_bytes_empty_string(self):
        """Test bytes decoding with empty string."""
        # Test all formats with empty string
        for base in ["binary", "octal", "decimal", "hexadecimal", "base64"]:
            decoded = decode_bytes("", base)
            self.assertEqual(decoded, b"")

    def test_steganography_round_trip_numpy_array(self):
        """Test steganography encoding and decoding with numpy array input."""
        test_data = b"Numpy array test"

        # Encode data
        encoded_image = encode_steganography(test_data)

        # Convert to numpy array
        numpy_image = encoded_image[0].numpy()

        # Decode from numpy array
        decoded_data = decode_steganography(numpy_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)

    def test_steganography_round_trip_with_whitespace(self):
        """Test steganography encoding and decoding with data containing whitespace."""
        test_data = b"Data with\nnewline\ttab and spaces"

        # Encode data
        encoded_image = encode_steganography(test_data)

        # Decode data
        decoded_data = decode_steganography(encoded_image)

        # Verify data integrity
        self.assertEqual(decoded_data, test_data)


if __name__ == "__main__":
    unittest.main()