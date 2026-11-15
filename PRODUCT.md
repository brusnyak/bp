# Real-Time Speech Translation

A local-first research prototype for people who need translated speech and live captions during meetings, presentations and calls. It connects speech recognition, machine translation and text-to-speech in one browser-controlled pipeline.

## Routes and setup

English↔Czech is the primary route. English output to Slovak and German is also available; language and hardware coverage is recorded in `documentation/LANGUAGES.md`. Slovak speech input remains experimental, especially for accented speech. Installing a language pack does not establish full-route quality.

This is a local, single-user tool that you clone and run. The browser starts a short-lived anonymous session automatically. Run the backend on your own machine and use the browser to select languages, prepare installed models, check sound and start translation. CPU, Apple Silicon and NVIDIA profiles have different model and performance requirements; follow `README.md` for setup.

## Privacy and operation

Recognition, translation and speech generation run on the processing machine with locally installed assets. Uploaded recordings stay on that host. A shared host needs access and retention policies; Model installation requires downloads before offline use.

Prepare the route before capture, check input/output and use headphones or isolated audio channels. Stop ends capture, synthesis and queued playback. Virtual audio routing can feed a meeting application; room-specific integration and echo isolation require testing.

## Limits and acceptance

Generic Piper speech is the baseline. Personal voice cloning remains experimental: a recording or successful model load does not prove voice identity, intelligibility or live speed. OmniVoice batch generation is not evidence of streaming translation.

Room mode has model-free tests; full real-model, multi-listener and cloned-voice acceptance remains pending. Performance depends on hardware, language, model and workload. Saved benchmarks do not establish sustained meeting quality or commercial readiness.

Promote a route only with current asset/configuration fingerprints, reviewed recognition and translation, intelligible target speech, voice-identity review where required, licence review and a continuous rehearsal with measured delay, queue stability, interruption and recovery. See `documentation/STATUS.md` and `documentation/BENCHMARKS.md` for evidence and limits.
