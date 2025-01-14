from pathlib import Path
import subprocess
from typing import Optional, Union


def video_to_frames_ffmpeg(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_pattern: str = "frame_%07d.jpg",
    quality: int = 2,
    fps: Optional[float] = None
) -> None:
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory where frames will be saved
        output_pattern: Naming pattern for output files (default: frame_%07d.jpg)
        quality: JPEG quality (2-31, lower is better quality)
        fps: Optional specific fps to extract (None = extract all frames)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_pattern
    
    # command
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-qscale:v', str(quality)
    ]
    if fps is not None:
        cmd.extend(['-vf', f'fps={fps}'])
    cmd.append(str(output_path))
    
    # run ffmpeg
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")


def frames_to_video_ffmpeg(
    input_pattern: Union[str, Path],
    output_path: Union[str, Path],
    fps: int = 30,
    crf: int = 23,
    preset: str = "medium"
) -> None:
    """
    Create video from frame sequence using ffmpeg.
    
    Args:
        input_pattern: Input frame pattern (e.g., "frame_%07d.jpg")
        output_path: Output video path
        fps: Frames per second
        crf: Quality (17-28, lower is better)
        preset: Encoding preset (ultrafast to veryslow)
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(input_pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', str(crf),
        '-preset', preset,
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
