"""BP v2 Streaming Pipeline: STT → MT → TTS with sub-second first-audio latency.

Exports:
  ContinuousStreamingPipeline  — per-speaker→language pipeline instance
  StreamingResult              — emitted translation + audio chunk
"""

from .pipeline import ContinuousStreamingPipeline, StreamingResult

__all__ = ["ContinuousStreamingPipeline", "StreamingResult"]