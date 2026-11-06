const test = require('node:test');
const assert = require('node:assert/strict');

// Build a minimal valid WAV buffer (RIFF/WAVE, fmt + data chunks) so the
// SessionPlayback WAV-duration parser can extract a playback duration.
// 8 kHz, mono, 16-bit PCM, `dataBytes` bytes of payload -> dataBytes / (8000*2) seconds.
function wavBuffer(dataBytes) {
    const header = new ArrayBuffer(44);
    const dv = new DataView(header);
    const write = (off, val, little = true) => dv['setUint' + (val > 0xffff ? 32 : 16)](off, val, little);
    // RIFF
    write(0, 0x46464952, false);            // "RIFF"
    write(4, 36 + dataBytes, true);         // file size
    write(8, 0x45564157, false);            // "WAVE"
    // fmt  (pos 12)
    write(12, 0x20746d66, false);           // "fmt "
    write(16, 16, true);                    // fmt chunk size
    write(20, 1, true);                     // audio format = PCM
    write(22, 1, true);                     // channels = 1
    write(24, 8000, true);                  // sample rate = 8000 Hz
    write(28, 16000, true);                 // byte rate
    write(32, 2, true);                     // block align
    write(34, 16, true);                    // bits per sample
    // data (pos 36)
    write(36, 0x61746164, false);           // "data"
    write(40, dataBytes, true);             // data chunk size
    const payload = new Uint8Array(dataBytes);
    return new Uint8Array([...new Uint8Array(header), ...payload]).buffer;
}

function wavBlob(dataBytes) {
    const ab = wavBuffer(dataBytes);
    return {
        async arrayBuffer() { return ab; }
    };
}

const tick = () => new Promise(resolve => setImmediate(resolve));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function fixture(route = async () => {}) {
    const events = [], audios = [], errors = [];
    const player = new (require('../ui/live-speech/session-playback.js'))({
        urls: { createObjectURL: () => 'blob:' + audios.length, revokeObjectURL: u => events.push('revoke:' + u) },
        makeAudio: () => {
            const audio = {
                sinkId: '',
                async setSinkId(id) { await route(); this.sinkId = id; events.push('route'); },
                async play() { events.push('play'); },
                pause() { events.push('pause'); },
                removeAttribute() {}
            };
            audios.push(audio); return audio;
        },
        onError: error => errors.push(error.message),
        maxQueuedDurationS: 10
    });
    player.start(); return { player, events, audios, errors };
}

test('sequential playback advances on ended, never overlaps chunks', async () => {
    const f = fixture();
    f.player.enqueue(wavBlob(160), 'cable');
    f.player.enqueue(wavBlob(160), 'cable');
    await tick();
    assert.deepEqual(f.events, ['route', 'play']);         // only first chunk starts
    f.audios[0].onended();
    await tick();
    assert.deepEqual(f.events, ['route', 'play', 'revoke:blob:0', 'route', 'play']);
    assert.equal(f.audios.length, 2);
});

test('playbackRate is applied with pitch preserved and shrinks the queue budget', async () => {
    const f = fixture();
    f.player.playbackRate = 2;
    f.player.maxQueuedDurationS = 0.015;             // 10ms chunks -> 5ms each at 2x
    await f.player.enqueue(wavBlob(160));
    await f.player.enqueue(wavBlob(160));
    await f.player.enqueue(wavBlob(160));            // first becomes current; 1 queued = 5ms < 15ms
    await tick();
    assert.equal(f.audios[0].playbackRate, 2);
    assert.equal(f.audios[0].preservesPitch, true);
    assert.equal(f.player.active, true);
});

test('Stop during pending routing prevents late playback', async () => {
    let release;
    const f = fixture(() => new Promise(resolve => release = resolve));
    await f.player.enqueue(wavBlob(160), 'cable');   // await so route() assigns `release`
    f.player.stop();
    release();
    await tick();
    assert.ok(!f.events.includes('play'));
    assert.equal(f.player.current, null);
});

test('routing rejection stops without speaker fallback', async () => {
    const f = fixture(async () => { throw new Error('unplugged'); });
    await f.player.enqueue(wavBlob(160), 'missing');  // await so setSinkId runs
    await tick();
    assert.ok(!f.events.includes('play'));
    assert.deepEqual(f.errors, ['unplugged']);
    assert.equal(f.player.active, false);
});

test('Stop cancels playback and queue; stale ended event cannot restart it', async () => {
    const f = fixture();
    f.player.enqueue(wavBlob(160));
    f.player.enqueue(wavBlob(160));
    await tick();
    const stale = f.audios[0].onended;
    f.player.stop();
    if (stale) stale();                              // stale event after Stop must not restart
    // Enqueue after stop must be ignored
    f.player.enqueue(wavBlob(160));
    await tick();
    assert.equal(f.audios.length, 1);
    assert.equal(f.player.queue.length, 0);
});

test('queue overload stops visibly instead of growing without limit', async () => {
    const f = fixture();
    f.player.maxQueuedDurationS = 0.015; // 15ms budget
    // Each blob is 160 bytes = 10ms at 8kHz mono 16-bit.
    await f.player.enqueue(wavBlob(160)); // 10ms total, accepted; item becomes current
    await f.player.enqueue(wavBlob(160)); // 10ms queued, accepted (queue=10ms)
    await f.player.enqueue(wavBlob(160)); // 20ms queued >= 15ms -> stop
    await tick();
    assert.equal(f.player.active, false);
    assert.match(f.errors[0], /falling behind/);
});

test('adaptive rate speeds up only while behind and is applied per chunk', async () => {
    const SessionPlayback = require('../ui/live-speech/session-playback.js');
    assert.deepEqual([0, 0.99, 1, 2.4, 2.5, 3.9, 4, 9].map(SessionPlayback.adaptiveRate), [1, 1, 1.15, 1.15, 1.3, 1.3, 1.45, 1.45]);
    const f = fixture();
    f.player.adaptive = true;
    await f.player.enqueue(wavBlob(160));            // 10 ms backlog -> normal speed
    assert.equal(f.audios[0].playbackRate, 1);
});
