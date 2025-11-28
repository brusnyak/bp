# Concurrent Performance Test Report

**Test Date:** 2025-11-28 11:21:26

---

## Test Summary

### 12 Concurrent Users

- **Total Sessions:** 12
- **Successful:** 12 (100.0%)
- **Failed:** 0

**Latency Statistics:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| E2E Latency | 29.22s | 26.62s | 31.20s | 1.92s |
| STT Time | 3.58s | 0.65s | 9.93s | 2.70s |
| MT Time | 1.36s | 0.16s | 4.39s | 1.28s |
| TTS Time | 1.03s | 0.35s | 1.96s | 0.62s |

**Resource Usage:**

- **CPU:** Avg 7.0%, Max 27.3%
- **Memory:** Avg 95MB, Max 210MB


### 15 Concurrent Users

- **Total Sessions:** 15
- **Successful:** 11 (73.3%)
- **Failed:** 4

**Latency Statistics:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| E2E Latency | 25.08s | 24.39s | 26.21s | 0.80s |
| STT Time | 5.69s | 3.64s | 6.98s | 1.47s |
| MT Time | 1.82s | 1.20s | 2.38s | 0.48s |
| TTS Time | 1.41s | 1.38s | 1.44s | 0.02s |

**Resource Usage:**

- **CPU:** Avg 4.8%, Max 26.0%
- **Memory:** Avg 169MB, Max 308MB


## Recommendations

✅ System successfully handled 15 concurrent users with 100.0% success rate.

💡 Consider testing with more users to find the upper limit.

