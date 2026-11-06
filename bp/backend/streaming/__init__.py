"""BP v2 Streaming Pipeline: STT → MT → TTS with sub-second first-audio latency.

Exports:
  StreamingPipeline  — per-speaker→language pipeline instance
  StreamingResult    — emitted translation + audio chunk
"""

from .pipeline import StreamingPipeline, StreamingResult

__all__ = ["StreamingPipeline", "StreamingResult"]