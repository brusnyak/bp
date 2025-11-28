# Concurrent Performance Test

## Overview

This test validates the system's ability to handle multiple simultaneous users performing speech translation. It generates detailed performance reports with precise timeline tracking.

## Features

- **Timeline Tracking**: Records exact timestamps for every event (audio sent, transcription, translation, TTS)
- **Resource Monitoring**: Tracks CPU and memory usage during tests
- **Visualizations**:
  - Gantt chart showing when each session processed
  - Box plots of latency distributions
  - Resource usage over time
- **Reports**:
  - Markdown summary with statistics
  - JSON export of all metrics
  - CSV export of timeline events

## Requirements

Install optional dependencies for full functionality:

```bash
pip install psutil matplotlib seaborn tqdm scipy
```

## Usage

### Quick Start

```bash
# Ensure the server is running
make run

# In a new terminal, run the test
python test/concurrent_performance_test.py
```

### What it Tests

The test simulates 5, 8, and 10 concurrent users:

- Each user sends a WAV file
- Tracks all processing events with precise timestamps
- Monitors system resources
- Generates comprehensive reports

### Output

Results are saved to `test_output/performance_tests/run_<timestamp>/`:

```
run_20251128_070000/
├── summary_report.md              # Human-readable summary
├── timeline_gantt_5_users.png     # Visual timeline
├── timeline_gantt_8_users.png
├── timeline_gantt_10_users.png
├── latency_distributions_5_users.png  # Box plots
├── latency_distributions_8_users.png
├── latency_distributions_10_users.png
├── timeline_5_users.csv           # Detailed timeline data
├── timeline_8_users.csv
├── timeline_10_users.csv
├── metrics_5_users.json           # Session metrics
├── metrics_8_users.json
└── metrics_10_users.json
```

## Interpreting Results

### Success Criteria

- **Success Rate**: ≥95% sessions should complete successfully
- **E2E Latency**: Should be <5s for good user experience
- **CPU Usage**: Should stay below 80% for stability
- **Memory**: Should not exceed available RAM

### Timeline Gantt Chart

Shows when each session ran:

- Green bars = successful sessions
- Red bars = failed sessions
- Overlapping bars = true concurrent processing

### Latency Distributions

Box plots show:

- **Median** (line in box)
- **25th-75th percentile** (box)
- **Min/Max** (whiskers)
- **Outliers** (dots)

## Customization

Edit the test configuration in `concurrent_performance_test.py`:

```python
# Test different user counts
user_counts = [5, 8, 10, 15, 20]

# Use different audio files
TEST_AUDIO_FILES = [
    "test/your_file1.wav",
    "test/your_file2.wav",
]

# Test with XTTS instead of Piper
tts_model = "xtts"
```

## Troubleshooting

### ModuleNotFoundError

If you get import errors:

```bash
pip install psutil matplotlib seaborn tqdm scipy
```

### Connection Refused

Ensure the server is running:

```bash
make run
```

### SSL Certificate Errors

The test uses the self-signed certificate at `certs/cert.pem`. If you've changed the certificate location, update `CERT_PATH` in the script.

## Presentation Tips

For presenting results:

1. Open `summary_report.md` for high-level statistics
2. Show the Gantt chart to visualize concurrency
3. Use latency distributions to show consistency
4. Reference specific JSON data for detailed analysis

## Performance Benchmarks

**M1 Pro 16GB Expected Performance:**

- 5 users: Should handle easily, <50% CPU
- 10 users: May reach 60-70% CPU, still stable
- 20 users: Approaching limits, monitor memory

Piper TTS is much lighter than XTTS, so you can scale further with Piper.
