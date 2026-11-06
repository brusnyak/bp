/* One owner for translated audio: sequential output, explicit routing, terminal Stop.

Chunks play one at a time, advanced by the media 'ended' event. Playback cannot
run faster than the audio is long, so delay is bounded only by (a) shorter audio,
(b) playbackRate > 1 (pitch preserved) and (c) the queue budget below, which stops
the session visibly instead of letting delay grow. The budget counts queued audio
at the effective rate.

stop() is terminal: it bumps generation, revokes blob URLs and ignores stale events.
*/
(function (root) {
    class SessionPlayback {
        constructor({ makeAudio = url => new Audio(url), urls = URL, onError = () => {}, maxQueuedDurationS = 10, playbackRate = 1, adaptive = false } = {}) {
            Object.assign(this, { makeAudio, urls, onError, maxQueuedDurationS, playbackRate, adaptive });
            this.generation = 0;
            this.active = false;
            this.queue = [];   // { blob, url, audio, duration, sink, startTime }
            this.current = null;
        }
        start() { this.stop(); this.active = true; }
        stop() {
            this.active = false;
            this.generation++;
            this.queue = [];
            if (this.current) {
                const { audio, url } = this.current;
                audio.onended = null;
                audio.onerror = null;
                audio.pause();
                audio.removeAttribute('src');
                this.urls.revokeObjectURL(url);
                this.current = null;
            }
        }
        async enqueue(blob, sink = '') {
            if (!this.active) return;
            let duration = 0;
            try {
                const ab = await blob.arrayBuffer();
                duration = SessionPlayback._wavDuration(ab);
            } catch { /* duration unknown; onended will drive advancement */ }
            const queuedDuration = this.queue.reduce((sum, it) => sum + it.duration, 0) / this.playbackRate + duration / this.playbackRate;
            if (queuedDuration >= this.maxQueuedDurationS) {
                this.stop();
                this.onError(new Error('Translated audio is falling behind. Session stopped; use a faster profile.'));
                return;
            }
            this.queue.push({ blob, duration, sink });
            if (!this.current) this.playNext();
        }
        // Speed up only while behind; same table as scripts/live/paced_replay.py (pitch preserved).
        static adaptiveRate(backlogS) { return backlogS < 1 ? 1 : backlogS < 2.5 ? 1.15 : backlogS < 4 ? 1.3 : 1.45; }
        // Parse WAV header to get duration without decoding: RIFF + fmt + data chunks.
        // duration_s = data_chunk_bytes / (sample_rate * channels * bits_per_sample / 8)
        static _wavDuration(arrayBuffer) {
            try {
                const dv = new DataView(arrayBuffer);
                if (dv.getUint32(0, false) !== 0x46464952) return 0; // "RIFF"
                if (dv.getUint32(8, false) !== 0x45564157) return 0; // "WAVE"
                let sampleRate = 0, channels = 0, bitsPerSample = 0, dataSize = 0, audioFormat = 0;
                let pos = 12;
                while (pos + 8 <= dv.byteLength) {
                    const chunkId = dv.getUint32(pos, false);
                    const chunkSize = dv.getUint32(pos + 4, true);
                    const dataStart = pos + 8;
                    if (chunkId === 0x20746d66) { // "fmt "
                        if (chunkSize >= 16) {
                            audioFormat = dv.getUint16(dataStart, true);
                            channels = dv.getUint16(dataStart + 2, true);
                            sampleRate = dv.getUint32(dataStart + 4, true);
                            bitsPerSample = dv.getUint16(dataStart + 14, true);
                        }
                    } else if (chunkId === 0x61746164) { // "data"
                        dataSize = chunkSize;
                    }
                    if (sampleRate && channels && bitsPerSample && dataSize) break;
                    pos = (dataStart + chunkSize + 1) & ~1; // word-align to next chunk
                }
                if (!sampleRate || !channels || !bitsPerSample || !dataSize) return 0;
                const blockAlign = (channels * bitsPerSample) / 8;
                return dataSize / (sampleRate * blockAlign);
            } catch { return 0; }
        }
        async playNext() {
            if (!this.active || this.current || !this.queue.length) return;
            const generation = this.generation;
            const { blob, duration, sink } = this.queue.shift();
            const url = this.urls.createObjectURL(blob);
            const audio = this.makeAudio(url);
            audio.playbackRate = this.adaptive ? SessionPlayback.adaptiveRate(this.queue.reduce((t, it) => t + it.duration, 0) + duration) : this.playbackRate;
            audio.preservesPitch = true;
            this.current = { audio, url, duration, sink, startTime: performance.now() };

            if (sink) {
                if (typeof audio.setSinkId !== 'function') {
                    this.stop();
                    this.onError(new Error('This browser cannot select an output device. Choose default headphones explicitly or use a supported browser.'));
                    return;
                }
                try {
                    await audio.setSinkId(sink);
                } catch (e) {
                    this.stop();
                    this.onError(e);
                    return;
                }
                if (!this.active || generation !== this.generation) return;
                if (audio.sinkId !== sink) {
                    this.stop();
                    this.onError(new Error('Selected output was not applied. No audio was played.'));
                    return;
                }
            }

            const onEnd = () => {
                if (generation !== this.generation) return;
                this.current = null;
                this.urls.revokeObjectURL(url);
                if (this.active && this.queue.length) this.playNext();
            };
            audio.onended = onEnd;
            audio.onerror = () => {
                if (generation !== this.generation) return;
                this.stop();
                this.onError(new Error('Audio playback failed. Check the selected output, then prepare again.'));
            };

            try {
                await audio.play();
            } catch (error) {
                if (generation !== this.generation) return;
                this.stop();
                this.onError(error);
                return;
            }
        }
    }
    if (typeof module !== 'undefined') module.exports = SessionPlayback;
    else root.SessionPlayback = SessionPlayback;
})(typeof window === 'undefined' ? globalThis : window);
