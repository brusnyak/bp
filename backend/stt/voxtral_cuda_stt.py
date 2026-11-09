"""Voxtral Mini 4B Realtime (Apache 2.0) through Hugging Face Transformers (>= 5.22) on an NVIDIA GPU.

Same streaming interface as backend/stt/voxtral_stt.py (open_stream / stream_feed / transcribe_audio), so the incremental route in
backend/main.py does not care which one runs. Measured 2026-11-09 on a Kaggle T4 (fp16, 8.25 GB VRAM, benchmark script
scripts/gpu_bench/kaggle_voxtral_t4/job.py): audio released at real time, text arrives 0.04-0.08 s after the audio chunk that holds it plus the
model's ~0.5 s own delay; word error 1-4% on clean generated English/German speech.

Transformers' streaming API is experimental: model.generate() runs on a thread and pulls input features from a generator. Here the generator
blocks on the samples pushed by stream_feed(); text pieces come back through a TextIteratorStreamer queue that stream_feed() drains."""
from __future__ import annotations

import os
import queue
import threading
import time
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

from backend.stt.voxtral_stt import LANGUAGES, _Stream  # noqa: F401  (LANGUAGES re-exported)

DEFAULT_REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"


class _HFStream(_Stream):
    def __init__(self, eng):
        super().__init__(None)
        self.eng, self.pcm, self.closed, self.thread, self.err = eng, np.zeros(0, np.float32), False, None, None
        self.cv = threading.Condition()
        self.streamer = None

    def push(self, audio: np.ndarray, final: bool):
        p = self.eng.processor
        with self.cv:
            if len(audio):
                self.pcm = np.concatenate([self.pcm, audio])
            if final:
                self.pcm = np.concatenate([self.pcm, np.zeros(p.num_right_pad_tokens * p.raw_audio_length_per_tok, np.float32)])
                self.closed = True
            self.cv.notify_all()
        if self.thread is None and len(self.pcm) >= p.num_samples_first_audio_chunk:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        from transformers import TextIteratorStreamer
        eng = self.eng
        p, m = eng.processor, eng.model
        try:
            self.streamer = TextIteratorStreamer(p.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            first = p(self.pcm[:p.num_samples_first_audio_chunk], is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
            first.to(m.device, dtype=m.dtype)

            def features():
                yield first.input_features
                idx = p.num_mel_frames_first_audio_chunk
                hop, win = p.feature_extractor.hop_length, p.feature_extractor.win_length
                start = idx * hop - win // 2
                while True:
                    end = start + p.num_samples_per_audio_chunk
                    with self.cv:
                        while len(self.pcm) <= end and not self.closed:
                            self.cv.wait()
                        if len(self.pcm) <= end:
                            return
                        chunk = self.pcm[start:end].copy()
                    x = p(chunk, is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
                    x.to(m.device, dtype=m.dtype)
                    yield x.input_features
                    idx += p.audio_length_per_tok
                    start = idx * hop - win // 2

            m.generate(input_ids=first.input_ids, input_features=features(), num_delay_tokens=first.num_delay_tokens, streamer=self.streamer)
        except Exception as e:   # surface in the server log, never hang the route
            self.err = e
            print(f"VoxtralCudaSTT: generate failed: {type(e).__name__}: {e}")
            if self.streamer is not None:
                self.streamer.text_queue.put(self.streamer.stop_signal)

    def drain(self) -> list:
        out = []
        while self.streamer is not None:
            try:
                t = self.streamer.text_queue.get_nowait()
            except queue.Empty:
                break
            if t is self.streamer.stop_signal:
                break
            out.append(t)
        return out


class VoxtralCudaSTT:
    incr_step_s = float(os.environ.get("BP_VOXTRAL_STEP_S", "0.2"))   # the GPU keeps up with small pushes (T4 2026-11-09); the Mac needs 0.64 s
    incr_min_audio_s = 0.0

    def __init__(self, model_size: str = "", device: str = "auto", compute_type: str = "", beam_size: int = 1):
        import torch
        from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("VoxtralCudaSTT needs a CUDA GPU (Apple Silicon uses backend/stt/voxtral_stt.py)")
        self.repo = os.environ.get("BP_VOXTRAL_MODEL", DEFAULT_REPO)
        self.model_size = self.repo
        self.processor = VoxtralRealtimeProcessor.from_pretrained(self.repo)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.repo, device_map="cuda", dtype=torch.float16)
        print(f"VoxtralCudaSTT initialized with {self.repo}")
        self.transcribe_audio(np.zeros(16000, dtype=np.float32), 16000)   # warm-up

    # --- streaming ---
    def open_stream(self, language=None) -> _HFStream:
        return _HFStream(self)

    def stream_feed(self, st: _HFStream, audio: np.ndarray, final: bool = False) -> list:
        st.push(np.ascontiguousarray(audio, dtype=np.float32), final)
        if final and st.thread is not None:
            st.thread.join(timeout=20)
        return st._words(st.drain(), final)

    # --- whole clip (phrase/clause/chunks strategies) ---
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True,
                         initial_prompt: Optional[str] = None, word_timestamps: bool = False) -> Tuple[list, float, Optional[str]]:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) / 16000 < 0.3:
            return [], 0.0, language
        t0 = time.time()
        inputs = self.processor(audio, return_tensors="pt").to(self.model.device)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        segs = [SimpleNamespace(start=0.0, end=len(audio) / 16000, text=text, words=None, no_speech_prob=0.0, avg_logprob=0.0)] if text else []
        return segs, time.time() - t0, language
