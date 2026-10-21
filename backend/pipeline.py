"""Public Pipeline facade — the single import for external consumers.

This is the SDK entry point. It wraps the full STT→MT→TTS cascade behind a
simple async interface, handles model pooling, and enforces the licence gate.

    from backend.pipeline import Pipeline

    pipe = Pipeline(source_lang="en", target_lang="sk", tts_engine="piper")
    await pipe.initialize()
    audio = await pipe.translate("Hello world")
    # audio is (np.ndarray, sample_rate) — ready to play or save
    await pipe.close()

The cascade remains local-first, no cloud dependencies.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.registry import instantiate, select, engines as list_engines, license_ok
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Immutable configuration for a Pipeline instance."""
    source_lang: str
    target_lang: str
    stt_engine: str = "faster_whisper"
    stt_model_size: str = "base"
    mt_engine: str = "opus-mt"
    tts_engine: str = "piper"
    beam_size: int = 5
    allow_noncommercial: bool = False
    allow_unknown_license: bool = False
    speaker_wav_path: Optional[str] = None
    speaker_text: Optional[str] = None
    speaker_lang: Optional[str] = None
    # Pooling config (per-process, shared across Pipeline instances)
    pool_stt: int = 3
    pool_mt: int = 5
    pool_tts: int = 5


@dataclass
class PipelineResult:
    """Result of a single translate() call."""
    transcript: str
    translation: str
    audio: np.ndarray
    sample_rate: int
    latency: Dict[str, float]  # {"stt": s, "mt": s, "tts": s, "total": s}
    detected_lang: Optional[str] = None


class Pipeline:
    """
    High-level async pipeline: STT → MT → TTS.
    
    One instance = one language direction + engine choices.
    Models are pooled at the module level (shared across instances).
    """
    
    _model_pool: Dict[Tuple, Dict[str, Any]] = {}
    _pool_locks: Dict[Tuple, asyncio.Lock] = {}
    _pool_config: Dict[str, int] = {"stt": 3, "mt": 5, "tts": 5}
    
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._stt: Optional[FasterWhisperSTT] = None
        self._mt: Optional[CTranslate2MT] = None
        self._tts: Optional[Any] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Apply pool config from first instance
        if config.pool_stt:
            Pipeline._pool_config["stt"] = config.pool_stt
        if config.pool_mt:
            Pipeline._pool_config["mt"] = config.pool_mt
        if config.pool_tts:
            Pipeline._pool_config["tts"] = config.pool_tts
    
    @classmethod
    def _pool_key(cls, stage: str, engine: str, **config) -> Tuple:
        import hashlib, json
        cfg_str = json.dumps(config, sort_keys=True)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
        return (stage, engine, cfg_hash)
    
    @classmethod
    async def _get_pooled(cls, stage: str, engine: str, factory: Callable, **config) -> Tuple[Any, bool]:
        key = cls._pool_key(stage, engine, **config)
        
        if key not in cls._pool_locks:
            cls._pool_locks[key] = asyncio.Lock()
        
        async with cls._pool_locks[key]:
            if key in cls._model_pool:
                cls._model_pool[key]["ref_count"] += 1
                cls._model_pool[key]["last_used"] = time.time()
                return cls._model_pool[key]["instance"], False
            
            # Check pool limit
            stage_limit = cls._pool_config.get(stage, 10)
            current = sum(1 for k in cls._model_pool if k[0] == stage)
            if current >= stage_limit:
                candidates = [(k, v) for k, v in cls._model_pool.items() if k[0] == stage and v["ref_count"] == 0]
                if candidates:
                    oldest = min(candidates, key=lambda x: x[1]["last_used"])
                    log.info(f"Pipeline: Evicting LRU {oldest[0]} (stage={stage})")
                    del cls._model_pool[oldest[0]]
            
            log.info(f"Pipeline: Creating pooled {stage}/{engine} ({key})")
            instance = factory()
            cls._model_pool[key] = {
                "instance": instance,
                "ref_count": 1,
                "last_used": time.time(),
                "created_at": time.time(),
                "stage": stage,
                "engine": engine,
                "config": config,
            }
            return instance, True
    
    @classmethod
    async def _release_pooled(cls, stage: str, engine: str, **config) -> bool:
        key = cls._pool_key(stage, engine, **config)
        if key not in cls._model_pool:
            return False
        
        cls._model_pool[key]["ref_count"] -= 1
        cls._model_pool[key]["last_used"] = time.time()
        
        if cls._model_pool[key]["ref_count"] <= 0:
            idle_ttl = 300  # 5 minutes
            if time.time() - cls._model_pool[key]["last_used"] >= idle_ttl:
                del cls._model_pool[key]
                if key in cls._pool_locks:
                    del cls._pool_locks[key]
                return True
        return False
    
    async def initialize(self) -> None:
        """Load/pool all models. Idempotent — safe to call multiple times."""
        async with self._init_lock:
            if self._initialized:
                return
            
            # Licence gate check (fail closed)
            for stage, engine in [
                ("stt", self.cfg.stt_engine),
                ("mt", self.cfg.mt_engine),
                ("tts", self.cfg.tts_engine),
            ]:
                try:
                    select(stage, engine, 
                           allow_noncommercial=self.cfg.allow_noncommercial,
                           allow_unknown=self.cfg.allow_unknown_license)
                except PermissionError as e:
                    raise RuntimeError(f"Licence gate blocked {stage}/{engine}: {e}")
                except KeyError:
                    raise RuntimeError(f"Unknown engine {stage}/{engine}")
            
            # STT
            self._stt, _ = await self._get_pooled(
                "stt", self.cfg.stt_engine,
                lambda: self._make_stt(),
                model_size=self.cfg.stt_model_size
            )
            if hasattr(self._stt, "beam_size"):
                self._stt.beam_size = self.cfg.beam_size
            
            # MT
            mt_path = self._mt_model_path()
            self._mt, _ = await self._get_pooled(
                "mt", self.cfg.mt_engine,
                lambda: instantiate("mt", self.cfg.mt_engine, 
                                    model_path=mt_path, device="cpu",
                                    allow_noncommercial=self.cfg.allow_noncommercial,
                                    allow_unknown=self.cfg.allow_unknown_license),
                source=self.cfg.source_lang, target=self.cfg.target_lang
            )
            
            # TTS
            self._tts, _ = await self._get_pooled(
                "tts", self.cfg.tts_engine,
                lambda: instantiate("tts", self.cfg.tts_engine,
                                    allow_noncommercial=self.cfg.allow_noncommercial,
                                    allow_unknown=self.cfg.allow_unknown_license),
            )
            
            self._initialized = True
            log.info(f"Pipeline initialized: {self.cfg.source_lang}→{self.cfg.target_lang} "
                     f"({self.cfg.stt_engine}/{self.cfg.mt_engine}/{self.cfg.tts_engine})")
    
    def _make_stt(self) -> FasterWhisperSTT:
        return FasterWhisperSTT(model_size=self.cfg.stt_model_size, compute_type="int8")
    
    def _mt_model_path(self) -> str:
        name = f"Helsinki-NLP/opus-mt-{self.cfg.source_lang}-{self.cfg.target_lang}"
        return os.path.join("ct2_models", name.replace('/', '--'))
    
    async def translate(self, text_or_audio: str | np.ndarray, 
                        sample_rate: int = 16000) -> PipelineResult:
        """
        Translate text or audio.
        
        Args:
            text_or_audio: Either a text string (MT→TTS only) or audio array (full STT→MT→TTS)
            sample_rate: Sample rate if audio is provided (default 16kHz)
        
        Returns:
            PipelineResult with transcript, translation, audio, and latency breakdown.
        """
        if not self._initialized:
            await self.initialize()
        
        latencies = {}
        total_start = time.perf_counter()
        
        # STT (if audio provided)
        if isinstance(text_or_audio, np.ndarray):
            stt_start = time.perf_counter()
            segments, stt_time, detected = self._stt.transcribe_audio(
                text_or_audio.astype(np.float32), sample_rate,
                language=self.cfg.source_lang
            )
            transcript = " ".join(s.text if hasattr(s, "text") else s["text"] for s in segments).strip()
            latencies["stt"] = time.perf_counter() - stt_start
        else:
            transcript = text_or_audio
            latencies["stt"] = 0.0
            detected = self.cfg.source_lang
        
        # MT
        mt_start = time.perf_counter()
        chunks = self._mt.split_chunks(transcript) or [transcript]
        translated_parts, _ = self._mt.translate_segments(
            chunks, self.cfg.source_lang, self.cfg.target_lang
        )
        translation = " ".join(translated_parts).strip()
        latencies["mt"] = time.perf_counter() - mt_start
        
        # TTS
        tts_start = time.perf_counter()
        audio, sr, syn_s = self._tts.synthesize(translation, language=self.cfg.target_lang)
        latencies["tts"] = time.perf_counter() - tts_start
        
        latencies["total"] = time.perf_counter() - total_start
        
        return PipelineResult(
            transcript=transcript,
            translation=translation,
            audio=audio,
            sample_rate=sr,
            latency=latencies,
            detected_lang=detected,
        )
    
    async def translate_streaming(self, audio_chunks: List[np.ndarray],
                                  sample_rate: int = 16000) -> List[PipelineResult]:
        """
        Translate a list of audio chunks (for real-time streaming).
        Each chunk goes through the full pipeline independently.
        """
        results = []
        for chunk in audio_chunks:
            results.append(await self.translate(chunk, sample_rate))
        return results
    
    async def close(self) -> None:
        """Release model references. Call when done with this pipeline."""
        if self._stt:
            await self._release_pooled("stt", self.cfg.stt_engine, model_size=self.cfg.stt_model_size)
        if self._mt:
            await self._release_pooled("mt", self.cfg.mt_engine, 
                                       source=self.cfg.source_lang, target=self.cfg.target_lang)
        if self._tts:
            await self._release_pooled("tts", self.cfg.tts_engine)
        self._stt = self._mt = self._tts = None
        self._initialized = False
    
    def __repr__(self) -> str:
        return (f"Pipeline({self.cfg.source_lang}→{self.cfg.target_lang}, "
                f"stt={self.cfg.stt_engine}/{self.cfg.stt_model_size}, "
                f"mt={self.cfg.mt_engine}, tts={self.cfg.tts_engine})")


# Convenience factory for common pairs
async def create_pipeline(source_lang: str, target_lang: str,
                          tts_engine: str = "piper",
                          stt_model_size: str = "base",
                          beam_size: int = 5) -> Pipeline:
    """Create and initialize a pipeline for a language pair."""
    pipe = Pipeline(PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        tts_engine=tts_engine,
        stt_model_size=stt_model_size,
        beam_size=beam_size,
    ))
    await pipe.initialize()
    return pipe


# Backwards compatibility with session-style API
class SessionPipeline:
    """
    Drop-in replacement for the WebSocket session flow.
    Maintains a session config dict compatible with backend/main.py.
    """
    def __init__(self, session_config: Dict[str, Any]):
        self.pipe = Pipeline(PipelineConfig(
            source_lang=session_config["source_lang"],
            target_lang=session_config["target_lang"],
            stt_engine=session_config.get("stt_engine", "faster_whisper"),
            stt_model_size=session_config.get("stt_model_size", "base"),
            mt_engine=session_config.get("mt_engine", "opus-mt"),
            tts_engine=session_config.get("tts_model_choice", "piper"),
            beam_size=session_config.get("beam_size", 5),
            allow_noncommercial=session_config.get("allow_noncommercial", False),
            allow_unknown_license=session_config.get("allow_unknown_license", False),
            speaker_wav_path=session_config.get("speaker_wav_path"),
            speaker_text=session_config.get("speaker_text"),
            speaker_lang=session_config.get("speaker_lang"),
        ))
    
    async def initialize(self):
        await self.pipe.initialize()
    
    async def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> PipelineResult:
        return await self.pipe.translate(audio, sample_rate)
    
    async def process_text(self, text: str) -> PipelineResult:
        return await self.pipe.translate(text)
    
    async def close(self):
        await self.pipe.close()


if __name__ == "__main__":
    # Quick self-test
    async def test():
        pipe = await create_pipeline("en", "sk", tts_engine="piper", stt_model_size="base")
        # Text-only (MT→TTS)
        r = await pipe.translate("Good morning, this is a test.")
        print(f"MT→TTS: {r.translation[:50]}... audio {len(r.audio)} samples @ {r.sample_rate}Hz")
        print(f"Latency: {r.latency}")
        await pipe.close()
    
    asyncio.run(test())