# Concurrent Performance Test Report

**Test Date:** 2025-11-28 09:30:13

---

## Test Summary

### 5 Concurrent Users

- **Total Sessions:** 5
- **Successful:** 5 (100.0%)
- **Failed:** 0

**Latency Statistics:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| E2E Latency | 10.20s | 4.62s | 26.97s | 8.43s |
| STT Time | 0.90s | 0.48s | 2.07s | 0.49s |
| MT Time | 0.14s | 0.03s | 0.37s | 0.09s |
| TTS Time | 0.21s | 0.05s | 0.34s | 0.09s |

**Resource Usage:**

- **CPU:** Avg 4.2%, Max 19.3%
- **Memory:** Avg 112MB, Max 207MB


### 8 Concurrent Users

- **Total Sessions:** 8
- **Successful:** 8 (100.0%)
- **Failed:** 0

**Latency Statistics:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| E2E Latency | 16.75s | 7.04s | 26.22s | 9.45s |
| STT Time | 1.73s | 0.54s | 5.52s | 1.59s |
| MT Time | 0.39s | 0.03s | 1.86s | 0.49s |
| TTS Time | 0.34s | 0.08s | 0.82s | 0.22s |

**Resource Usage:**

- **CPU:** Avg 4.9%, Max 16.6%
- **Memory:** Avg 133MB, Max 226MB


### 10 Concurrent Users

- **Total Sessions:** 10
- **Successful:** 10 (100.0%)
- **Failed:** 0

**Latency Statistics:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| E2E Latency | 22.15s | 7.68s | 28.09s | 8.38s |
| STT Time | 1.97s | 0.54s | 5.53s | 1.66s |
| MT Time | 0.55s | 0.12s | 1.69s | 0.47s |
| TTS Time | 0.56s | 0.16s | 1.15s | 0.32s |

**Resource Usage:**

- **CPU:** Avg 5.3%, Max 23.8%
- **Memory:** Avg 138MB, Max 266MB


## Recommendations

✅ System successfully handled 10 concurrent users with 100.0% success rate.

💡 Consider testing with more users to find the upper limit.

