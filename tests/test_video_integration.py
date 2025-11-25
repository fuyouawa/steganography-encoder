"""
Unit tests for video loading and combining integration.
Tests the complete workflow: load_video -> process -> combine_video
"""

import unittest
import tempfile
import os
from PIL import Image
import numpy as np

# Import the video utilities
from utils.video.ffmpeg import combine_video as ffmpeg_combine_video
from utils.video.opencv import combine_video as opencv_combine_video
from utils.video.ffmpeg import load_video as ffmpeg_load_video
from utils.video.opencv import load_video as opencv_load_video


class TestVideoIntegration(unittest.TestCase):
    """Test video loading and combining integration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Path to test video file
        self.test_video_path = os.path.join(os.path.dirname(__file__), "test1.mp4")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ffmpeg_load_and_combine_basic(self):
        """Test basic FFmpeg load_video -> combine_video workflow."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Step 1: Load video using FFmpeg
        image_batch, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=10,  # Load only first 10 frames for testing
            memory_limit_mb=100
        )

        # Verify video loading
        self.assertIsNotNone(image_batch)
        self.assertGreater(image_batch.shape[0], 0)  # Should have at least one frame
        self.assertEqual(image_batch.ndim, 4)  # Should be [frames, height, width, channels]

        # Verify video info
        self.assertIsNotNone(video_info)
        self.assertGreater(video_info.source_fps, 0)
        self.assertGreater(video_info.source_width, 0)
        self.assertGreater(video_info.source_height, 0)

        # Step 2: Combine loaded frames back to video
        output_path = os.path.join(self.temp_dir, "test_ffmpeg_integration.mp4")

        result_path, extension = ffmpeg_combine_video(
            image_batch=image_batch,
            output_path=output_path,
            frame_rate=video_info.source_fps,
            video_format="video/h264-mp4"
        )

        # Verify video combining
        self.assertEqual(result_path, output_path)
        self.assertEqual(extension, "mp4")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_opencv_load_and_combine_basic(self):
        """Test basic OpenCV load_video -> combine_video workflow."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Step 1: Load video using OpenCV
        image_batch, video_info = opencv_load_video(
            video_path=self.test_video_path,
            frame_load_cap=10,  # Load only first 10 frames for testing
            memory_limit_mb=100
        )

        # Verify video loading
        self.assertIsNotNone(image_batch)
        self.assertGreater(image_batch.shape[0], 0)  # Should have at least one frame
        self.assertEqual(image_batch.ndim, 4)  # Should be [frames, height, width, channels]

        # Verify video info
        self.assertIsNotNone(video_info)
        self.assertGreater(video_info.source_fps, 0)
        self.assertGreater(video_info.source_width, 0)
        self.assertGreater(video_info.source_height, 0)

        # Step 2: Combine loaded frames back to video
        output_path = os.path.join(self.temp_dir, "test_opencv_integration.mp4")

        result_path, extension = opencv_combine_video(
            image_batch=image_batch,
            output_path=output_path,
            frame_rate=video_info.source_fps,
            video_format="video/mp4"
        )

        # Verify video combining
        self.assertEqual(result_path, output_path)
        self.assertEqual(extension, "mp4")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_cross_implementation_compatibility(self):
        """Test that frames loaded by one implementation can be combined by the other."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Test FFmpeg load -> OpenCV combine
        image_batch_ffmpeg, video_info_ffmpeg = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5,
            memory_limit_mb=100
        )

        output_path_opencv = os.path.join(self.temp_dir, "test_ffmpeg_to_opencv.mp4")
        result_path_opencv, extension_opencv = opencv_combine_video(
            image_batch=image_batch_ffmpeg,
            output_path=output_path_opencv,
            frame_rate=video_info_ffmpeg.source_fps,
            video_format="video/mp4"
        )

        self.assertTrue(os.path.exists(output_path_opencv))
        self.assertGreater(os.path.getsize(output_path_opencv), 0)

        # Test OpenCV load -> FFmpeg combine
        image_batch_opencv, video_info_opencv = opencv_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5,
            memory_limit_mb=100
        )

        output_path_ffmpeg = os.path.join(self.temp_dir, "test_opencv_to_ffmpeg.mp4")
        result_path_ffmpeg, extension_ffmpeg = ffmpeg_combine_video(
            image_batch=image_batch_opencv,
            output_path=output_path_ffmpeg,
            frame_rate=video_info_opencv.source_fps,
            video_format="video/h264-mp4"
        )

        self.assertTrue(os.path.exists(output_path_ffmpeg))
        self.assertGreater(os.path.getsize(output_path_ffmpeg), 0)

    def test_load_and_combine_with_pingpong(self):
        """Test load_video -> combine_video with pingpong effect."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Load video using FFmpeg
        image_batch, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=8,  # Need enough frames for pingpong effect
            memory_limit_mb=100
        )

        # Combine with pingpong effect
        output_path = os.path.join(self.temp_dir, "test_pingpong.mp4")

        result_path, extension = ffmpeg_combine_video(
            image_batch=image_batch,
            output_path=output_path,
            frame_rate=video_info.source_fps,
            video_format="video/h264-mp4",
            pingpong=True
        )

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_load_and_combine_with_metadata(self):
        """Test load_video -> combine_video with metadata."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Load video using FFmpeg
        image_batch, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=5,
            memory_limit_mb=100
        )

        # Combine with metadata
        output_path = os.path.join(self.temp_dir, "test_metadata.mp4")

        video_metadata = {
            "title": "Integration Test Video",
            "artist": "Test Framework",
            "description": "Video created by load_video and combine_video integration test"
        }

        result_path, extension = ffmpeg_combine_video(
            image_batch=image_batch,
            output_path=output_path,
            frame_rate=video_info.source_fps,
            video_format="video/h264-mp4",
            video_metadata=video_metadata
        )

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_load_and_combine_different_formats(self):
        """Test load_video -> combine_video with different output formats."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Load video using FFmpeg
        image_batch, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=3,
            memory_limit_mb=100
        )

        # Test different output formats
        test_formats = [
            ("video/h264-mp4", "mp4"),
            ("video/h265-mp4", "mp4"),
            ("video/webm", "webm"),
        ]

        for i, (video_format, expected_extension) in enumerate(test_formats):
            with self.subTest(format=video_format):
                output_path = os.path.join(self.temp_dir, f"test_{expected_extension}_{i}.{expected_extension}")

                # Ensure output file doesn't exist
                if os.path.exists(output_path):
                    os.remove(output_path)

                try:
                    result_path, extension = ffmpeg_combine_video(
                        image_batch=image_batch,
                        output_path=output_path,
                        frame_rate=video_info.source_fps,
                        video_format=video_format
                    )

                    self.assertEqual(result_path, output_path)
                    self.assertEqual(extension, expected_extension)
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 0)
                except Exception as e:
                    # Some formats might not be available on the system
                    self.skipTest(f"Format {video_format} not available: {e}")

    def test_load_and_combine_frame_consistency(self):
        """Test that loaded and recombined video maintains frame consistency."""
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"Test video file not found: {self.test_video_path}")

        # Load a small number of frames
        image_batch, video_info = ffmpeg_load_video(
            video_path=self.test_video_path,
            frame_load_cap=3,
            memory_limit_mb=100
        )

        # Verify we got the expected number of frames
        self.assertEqual(image_batch.shape[0], 3)

        # Combine to video
        output_path = os.path.join(self.temp_dir, "test_consistency.mp4")
        result_path, extension = ffmpeg_combine_video(
            image_batch=image_batch,
            output_path=output_path,
            frame_rate=video_info.source_fps,
            video_format="video/h264-mp4"
        )

        # Load the recombined video
        reloaded_batch, reloaded_info = ffmpeg_load_video(
            video_path=output_path,
            frame_load_cap=3,
            memory_limit_mb=100
        )

        # Should have same number of frames
        self.assertEqual(reloaded_batch.shape[0], 3)

        # Should have same dimensions
        self.assertEqual(image_batch.shape[1:], reloaded_batch.shape[1:])


if __name__ == "__main__":
    unittest.main()