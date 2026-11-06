"""No-model scheduling and WAV normalization regressions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('replay', Path(__file__).with_name('job.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_slow_service_accumulates_gpu_queue(self):
        rows = replay.schedule([0, 1, 2], [3, 3, 3], [1, 1, 1])
        self.assertEqual([r['generation_queue_s'] for r in rows], [0, 2, 4])
        self.assertEqual([r['first_playable_delay_s'] for r in rows], [3, 5, 7])

    def test_long_speech_accumulates_playback_even_with_fast_tts(self):
        rows = replay.schedule([0, 1, 2], [.2, .2, .2], [3, 3, 3])
        self.assertAlmostEqual(rows[-1]['playback_queue_s'], 4)
        self.assertEqual([r['generation_queue_s'] for r in rows], [0, 0, 0])

    def test_rate_shortens_backlog_and_drop_bounds_delay(self):
        base = replay.schedule([0, 3, 6], [.2] * 3, [5, 5, 5])
        fast = replay.schedule([0, 3, 6], [.2] * 3, [5, 5, 5], rate=1.25)
        self.assertGreater(base[-1]['first_playable_delay_s'], fast[-1]['first_playable_delay_s'])
        dropped = replay.schedule([0, 3, 6], [.2] * 3, [5, 5, 5], drop_late_s=2)
        self.assertEqual([r['dropped'] for r in dropped], [False, True, False])  # dropping chunk 2 lets chunk 3 recover
        self.assertTrue(all(r['first_playable_delay_s'] <= 2 or r['dropped'] for r in dropped))

    def test_stop_drops_queued_and_cuts_playing(self):
        rows = replay.schedule([0, 1, 2], [.2, .2, .2], [3, 3, 3])
        self.assertEqual([r['action'] for r in replay.simulated_stop(rows, 5)],
                         ['already_finished', 'cut', 'drop'])

    def test_invalid_or_mismatched_measurements_fail(self):
        for args in [([0], [], []), ([0], [-1], [1]), ([float('nan')], [1], [1])]:
            with self.assertRaises(ValueError):
                replay.schedule(*args)

    def test_float_stereo_24khz_reference_becomes_pcm16_mono_16khz(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d)/'in.wav', Path(d)/'out.wav'
            tone = .1*np.sin(np.arange(48000)*.03)
            sf.write(source, np.column_stack([tone, tone]), 24000, subtype='FLOAT')
            receipt = replay.normalize_reference(source, output)
            info = sf.info(output)
            self.assertEqual((info.samplerate, info.channels, info.subtype),
                             (16000, 1, 'PCM_16'))
            self.assertAlmostEqual(receipt['duration_s'], 2)
            self.assertEqual(receipt['original']['subtype'], 'FLOAT')


if __name__ == '__main__':
    unittest.main()
