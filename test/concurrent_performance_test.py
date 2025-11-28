#!/usr/bin/env python3
"""
Concurrent Performance Test for Live Speech Translation System

This test simulates multiple concurrent users to validate system performance
and generate detailed timeline and performance reports.

Features:
- Precise timestamp tracking for all events
- Resource monitoring (CPU, Memory)
- Timeline visualization (Gantt chart)
- Comprehensive performance reports (MD, JSON, CSV)
"""

import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import os
import time
import logging
import random
import statistics
import argparse
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import ssl
from dataclasses import dataclass, asdict
import csv

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Resource monitoring will be disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available. Visualizations will be disabled.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
WS_URL = "wss://localhost:8000/ws"
CERT_PATH = "certs/cert.pem"
AUDIO_SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 20  # ms
AUDIO_CHUNK_LENGTH_SECONDS = VAD_FRAME_DURATION / 1000

# Test configuration
TEST_AUDIO_FILES = [
    "test/Hello.wav",
    "test/Can you hear me_.wav",
    "test/My test speech_xtts_speaker_clean.wav",
]

# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"test_output/performance_tests/run_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class TimelineEvent:
    """Represents a single event in the processing timeline"""
    session_id: int
    event_type: str  # 'audio_sent', 'transcription', 'translation', 'tts_audio', 'session_start', 'session_end'
    timestamp: float
    data: Optional[Dict[str, Any]] = None


@dataclass
class SessionMetrics:
    """Comprehensive metrics for a single session"""
    session_id: int
    audio_file: str
    status: str
    start_time: float
    end_time: Optional[float] = None
    first_audio_sent: Optional[float] = None
    last_tts_received: Optional[float] = None
    transcriptions: List[str] = None
    translations: List[str] = None
    stt_latencies: List[float] = None
    mt_latencies: List[float] = None
    tts_latencies: List[float] = None
    e2e_latency: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.transcriptions is None:
            self.transcriptions = []
        if self.translations is None:
            self.translations = []
        if self.stt_latencies is None:
            self.stt_latencies = []
        if self.mt_latencies is None:
            self.mt_latencies = []
        if self.tts_latencies is None:
            self.tts_latencies = []


class ResourceMonitor:
    """Monitors system resource usage during tests"""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    async def start(self):
        """Start monitoring resources"""
        if not PSUTIL_AVAILABLE:
            return
        
        self.monitoring = True
        while self.monitoring:
            self.timestamps.append(time.perf_counter())
            self.cpu_samples.append(self.process.cpu_percent())
            self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB
            await asyncio.sleep(self.interval)
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        if not self.cpu_samples:
            return {}
        
        return {
            "cpu_avg": np.mean(self.cpu_samples),
            "cpu_max": np.max(self.cpu_samples),
            "cpu_min": np.min(self.cpu_samples),
            "memory_avg_mb": np.mean(self.memory_samples),
            "memory_max_mb": np.max(self.memory_samples),
            "memory_min_mb": np.min(self.memory_samples),
        }


async def simulate_user_session(
    session_id: int,
    audio_file_path: str,
    source_lang: str,
    target_lang: str,
    tts_model_choice: str,
    timeline_events: List[TimelineEvent],
    progress_bar: Optional[tqdm] = None,
    start_delay: float = 0.0
) -> SessionMetrics:
    """
    Simulates a single user session: Connect -> Send Audio -> Receive Results
    """
    if start_delay > 0:
        await asyncio.sleep(start_delay)
        
    metrics = SessionMetrics(
        session_id=session_id,
        audio_file=audio_file_path,
        status="failed",
        start_time=time.perf_counter()
    )
    
    timeline_events.append(TimelineEvent(
        session_id=session_id,
        event_type="session_start",
        timestamp=metrics.start_time
    ))
    
    logging.info(f"Session {session_id}: Starting with {audio_file_path}")
    
    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(CERT_PATH)
    ssl_context.check_hostname = False
    
    try:
        async with websockets.connect(WS_URL, ssl=ssl_context) as websocket:
            # Send config
            config_message = {
                "type": "config_update",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "tts_model_choice": tts_model_choice,
            }
            await websocket.send(json.dumps(config_message))
            
            # Wait for config response
            response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
            if response.get("status") == "error":
                metrics.error = f"Config failed: {response.get('message')}"
                logging.error(f"Session {session_id}: {metrics.error}")
                return metrics
            
            # Send start command
            await websocket.send(json.dumps({"type": "start"}))
            
            # Load and prepare audio
            audio_data, sample_rate = sf.read(audio_file_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample if needed
            if sample_rate != AUDIO_SAMPLE_RATE:
                from scipy.signal import resample
                num_samples = int(len(audio_data) * AUDIO_SAMPLE_RATE / sample_rate)
                audio_data = resample(audio_data, num_samples)
            
            # Chunk audio
            chunk_size = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_LENGTH_SECONDS)
            audio_chunks = [audio_data[i:i + chunk_size] 
                          for i in range(0, len(audio_data), chunk_size)]
            
            # Message receiver task
            async def receive_messages():
                while True:
                    try:
                        raw_response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        recv_time = time.perf_counter()
                        
                        if isinstance(raw_response, str):
                            response = json.loads(raw_response)
                            
                            if response.get("type") == "transcription_result":
                                text = response.get("transcribed")
                                metrics.transcriptions.append(text)
                                timeline_events.append(TimelineEvent(
                                    session_id=session_id,
                                    event_type="transcription",
                                    timestamp=recv_time,
                                    data={"text": text}
                                ))
                                
                            elif response.get("type") == "translation_result":
                                text = response.get("translated")
                                metrics.translations.append(text)
                                timeline_events.append(TimelineEvent(
                                    session_id=session_id,
                                    event_type="translation",
                                    timestamp=recv_time,
                                    data={"text": text}
                                ))
                                
                            elif response.get("type") == "final_metrics":
                                m = response.get("metrics", {})
                                metrics.stt_latencies.append(m.get("stt_time", 0.0))
                                metrics.mt_latencies.append(m.get("mt_time", 0.0))
                                metrics.tts_latencies.append(m.get("tts_time", 0.0))
                                
                            elif response.get("type") == "error":
                                metrics.error = f"Server error: {response.get('message')}"
                                break
                                
                        elif isinstance(raw_response, bytes):
                            # TTS audio received
                            metrics.last_tts_received = recv_time
                            timeline_events.append(TimelineEvent(
                                session_id=session_id,
                                event_type="tts_audio",
                                timestamp=recv_time,
                                data={"size_bytes": len(raw_response)}
                            ))
                            
                    except asyncio.TimeoutError:
                        pass
                    except websockets.exceptions.ConnectionClosedOK:
                        break
                    except Exception as e:
                        logging.error(f"Session {session_id}: Error in receive: {e}")
                        break
            
            receive_task = asyncio.create_task(receive_messages())
            
            # Stream audio chunks
            for i, chunk in enumerate(audio_chunks):
                if chunk.size > 0:
                    send_time = time.perf_counter()
                    if metrics.first_audio_sent is None:
                        metrics.first_audio_sent = send_time
                    
                    await websocket.send(chunk.tobytes())
                    timeline_events.append(TimelineEvent(
                        session_id=session_id,
                        event_type="audio_sent",
                        timestamp=send_time,
                        data={"chunk_index": i, "chunk_size": chunk.size}
                    ))
                    
                await asyncio.sleep(AUDIO_CHUNK_LENGTH_SECONDS * 0.8)
            
            # Send stop
            await websocket.send(json.dumps({"type": "stop"}))
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Cancel receiver
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            
            metrics.status = "completed"
            
    except Exception as e:
        metrics.error = f"Unexpected error: {e}"
        logging.error(f"Session {session_id}: {metrics.error}")
    
    finally:
        metrics.end_time = time.perf_counter()
        
        # Calculate E2E latency
        if metrics.first_audio_sent and metrics.last_tts_received:
            metrics.e2e_latency = metrics.last_tts_received - metrics.first_audio_sent
        
        timeline_events.append(TimelineEvent(
            session_id=session_id,
            event_type="session_end",
            timestamp=metrics.end_time
        ))
        
        if progress_bar:
            progress_bar.update(1)
        
        duration = metrics.end_time - metrics.start_time
        logging.info(f"Session {session_id}: Finished in {duration:.2f}s (Status: {metrics.status})")
        
        return metrics


async def run_concurrent_test(
    num_users: int,
    audio_files: List[str],
    tts_model: str = "piper",
    ramp_up_strategy: str = "random",
    ramp_up_duration: float = 10.0
) -> Tuple[List[SessionMetrics], List[TimelineEvent], ResourceMonitor]:
    """
    Runs concurrent test with specified number of users and ramp-up strategy.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting concurrent test: {num_users} users, {tts_model} TTS")
    logging.info(f"Ramp-up: {ramp_up_strategy} over {ramp_up_duration}s")
    logging.info(f"{'='*80}\n")
    
    timeline_events = []
    resource_monitor = ResourceMonitor()
    
    # Start resource monitoring
    monitor_task = None
    if PSUTIL_AVAILABLE:
        monitor_task = asyncio.create_task(resource_monitor.start())
    
    # Create progress bar
    progress_bar = None
    if TQDM_AVAILABLE:
        progress_bar = tqdm(total=num_users, desc=f"Testing {num_users} users")
    
    # Create session tasks
    tasks = []
    for i in range(num_users):
        session_id = i + 1
        audio_file = audio_files[i % len(audio_files)]
        
        # Calculate start delay based on strategy
        start_delay = 0
        if ramp_up_strategy == "linear":
            start_delay = i * (ramp_up_duration / num_users)
        elif ramp_up_strategy == "random":
            start_delay = random.uniform(0, ramp_up_duration)
            
        task = asyncio.create_task(
            simulate_user_session(
                session_id=session_id,
                audio_file_path=audio_file,
                source_lang="en",
                target_lang="sk",
                tts_model_choice=tts_model,
                timeline_events=timeline_events,
                progress_bar=progress_bar,
                start_delay=start_delay
            )
        )
        tasks.append(task)
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop monitoring
    if monitor_task:
        resource_monitor.stop()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    if progress_bar:
        progress_bar.close()
    
    # Filter results
    metrics_list = [r for r in results if isinstance(r, SessionMetrics)]
    
    return metrics_list, timeline_events, resource_monitor


def generate_summary_report(
    all_metrics: Dict[int, List[SessionMetrics]],
    all_timelines: Dict[int, List[TimelineEvent]],
    resource_stats: Dict[int, Dict[str, Any]]
):
    """
    Generates a comprehensive Markdown summary report.
    """
    report_path = os.path.join(OUTPUT_DIR, "summary_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Concurrent Performance Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Overall summary
        f.write("## Test Summary\n\n")
        for num_users, metrics_list in sorted(all_metrics.items()):
            successful = [m for m in metrics_list if m.status == "completed"]
            failed = [m for m in metrics_list if m.status != "completed"]
            
            f.write(f"### {num_users} Concurrent Users\n\n")
            f.write(f"- **Total Sessions:** {len(metrics_list)}\n")
            f.write(f"- **Successful:** {len(successful)} ({len(successful)/len(metrics_list)*100:.1f}%)\n")
            f.write(f"- **Failed:** {len(failed)}\n\n")
            
            if successful:
                # Calculate statistics
                all_e2e = [m.e2e_latency for m in successful if m.e2e_latency]
                all_stt = [lat for m in successful for lat in m.stt_latencies]
                all_mt = [lat for m in successful for lat in m.mt_latencies]
                all_tts = [lat for m in successful for lat in m.tts_latencies]
                
                f.write("**Latency Statistics:**\n\n")
                f.write("| Metric | Average | Min | Max | Std Dev |\n")
                f.write("|--------|---------|-----|-----|---------|\n")
                
                if all_e2e:
                    f.write(f"| E2E Latency | {np.mean(all_e2e):.2f}s | "
                           f"{np.min(all_e2e):.2f}s | {np.max(all_e2e):.2f}s | "
                           f"{np.std(all_e2e):.2f}s |\n")
                if all_stt:
                    f.write(f"| STT Time | {np.mean(all_stt):.2f}s | "
                           f"{np.min(all_stt):.2f}s | {np.max(all_stt):.2f}s | "
                           f"{np.std(all_stt):.2f}s |\n")
                if all_mt:
                    f.write(f"| MT Time | {np.mean(all_mt):.2f}s | "
                           f"{np.min(all_mt):.2f}s | {np.max(all_mt):.2f}s | "
                           f"{np.std(all_mt):.2f}s |\n")
                if all_tts:
                    f.write(f"| TTS Time | {np.mean(all_tts):.2f}s | "
                           f"{np.min(all_tts):.2f}s | {np.max(all_tts):.2f}s | "
                           f"{np.std(all_tts):.2f}s |\n")
                
                f.write("\n")
                
                # Resource usage
                if num_users in resource_stats and resource_stats[num_users]:
                    stats = resource_stats[num_users]
                    f.write("**Resource Usage:**\n\n")
                    f.write(f"- **CPU:** Avg {stats['cpu_avg']:.1f}%, "
                           f"Max {stats['cpu_max']:.1f}%\n")
                    f.write(f"- **Memory:** Avg {stats['memory_avg_mb']:.0f}MB, "
                           f"Max {stats['memory_max_mb']:.0f}MB\n\n")
            
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        max_users_tested = max(all_metrics.keys())
        max_success_rate = max(
            len([m for m in metrics if m.status == "completed"]) / len(metrics) * 100
            for metrics in all_metrics.values()
        )
        
        if max_success_rate >= 95:
            f.write(f"✅ System successfully handled {max_users_tested} concurrent users "
                   f"with {max_success_rate:.1f}% success rate.\n\n")
            f.write(f"💡 Consider testing with more users to find the upper limit.\n\n")
        else:
            f.write(f"⚠️  Success rate dropped to {max_success_rate:.1f}% at "
                   f"{max_users_tested} users.\n\n")
            f.write(f"💡 Recommend optimizing or limiting to fewer concurrent users.\n\n")
    
    logging.info(f"Summary report generated: {report_path}")


def export_timeline_csv(timeline_events: List[TimelineEvent], num_users: int):
    """Export timeline events to CSV"""
    csv_path = os.path.join(OUTPUT_DIR, f"timeline_{num_users}_users.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'event_type', 'timestamp', 'data'])
        
        for event in sorted(timeline_events, key=lambda x: x.timestamp):
            writer.writerow([
                event.session_id,
                event.event_type,
                event.timestamp,
                json.dumps(event.data) if event.data else ''
            ])
    
    logging.info(f"Timeline CSV exported: {csv_path}")


def generate_timeline_gantt(
    metrics_list: List[SessionMetrics],
    num_users: int
):
    """Generate a Gantt chart showing session timelines"""
    if not PLOTTING_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, num_users * 0.4)))
    
    # Get the earliest start time for relative timeline
    base_time = min(m.start_time for m in metrics_list)
    
    for metrics in sorted(metrics_list, key=lambda x: x.session_id):
        y_pos = metrics.session_id
        start = metrics.start_time - base_time
        duration = (metrics.end_time or metrics.start_time) - metrics.start_time
        
        # Choose color based on status
        color = 'green' if metrics.status == "completed" else 'red'
        
        # Draw session bar
        ax.barh(y_pos, duration, left=start, height=0.6, 
               color=color, alpha=0.7, label=metrics.status if y_pos == 1 else "")
        
        # Add session label
        ax.text(start + duration/2, y_pos, f"S{metrics.session_id}", 
               ha='center', va='center', fontsize=8, weight='bold')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Session ID', fontsize=12)
    ax.set_title(f'Session Timeline - {num_users} Concurrent Users', fontsize=14, weight='bold')
    ax.set_yticks(range(1, num_users + 1))
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"timeline_gantt_{num_users}_users.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    logging.info(f"Timeline Gantt chart generated: {chart_path}")


def generate_latency_distributions(
    metrics_list: List[SessionMetrics],
    num_users: int
):
    """Generate box plots for latency distributions"""
    if not PLOTTING_AVAILABLE:
        return
    
    successful = [m for m in metrics_list if m.status == "completed"]
    if not successful:
        return
    
    # Collect data
    stt_data = [lat for m in successful for lat in m.stt_latencies]
    mt_data = [lat for m in successful for lat in m.mt_latencies]
    tts_data = [lat for m in successful for lat in m.tts_latencies]
    e2e_data = [m.e2e_latency for m in successful if m.e2e_latency]
    
    # Create box plots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    
    if stt_data:
        data_to_plot.append(stt_data)
        labels.append(f'STT\n(n={len(stt_data)})')
    if mt_data:
        data_to_plot.append(mt_data)
        labels.append(f'MT\n(n={len(mt_data)})')
    if tts_data:
        data_to_plot.append(tts_data)
        labels.append(f'TTS\n(n={len(tts_data)})')
    if e2e_data:
        data_to_plot.append(e2e_data)
        labels.append(f'E2E\n(n={len(e2e_data)})')
    
    # Skip if no data to plot
    if not data_to_plot:
        logging.warning(f"No latency data to plot for {num_users} users - skipping boxplot")
        return
    
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title(f'Latency Distributions - {num_users} Concurrent Users', 
                fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"latency_distributions_{num_users}_users.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    logging.info(f"Latency distributions chart generated: {chart_path}")


async def main():
    """Main test execution"""
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    logging.info("Concurrent Performance Test")
    logging.info("=" * 80)
    logging.info(f"Output directory: {OUTPUT_DIR}\n")
    
    # Verify audio files exist
    for audio_file in TEST_AUDIO_FILES:
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found: {audio_file}")
            return
    
    # Test configurations
    user_counts = [12, 15]
    tts_model = "piper"
    ramp_up_strategy = "random"
    ramp_up_duration = 15.0 # Spread starts over 15 seconds
    
    all_metrics = {}
    all_timelines = {}
    all_resource_stats = {}
    
    # Run tests
    for num_users in user_counts:
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing with {num_users} concurrent users")
        logging.info(f"{'='*80}\n")
        
        metrics_list, timeline_events, resource_monitor = await run_concurrent_test(
            num_users=num_users,
            audio_files=TEST_AUDIO_FILES,
            tts_model=tts_model,
            ramp_up_strategy=ramp_up_strategy,
            ramp_up_duration=ramp_up_duration
        )
        
        all_metrics[num_users] = metrics_list
        all_timelines[num_users] = timeline_events
        all_resource_stats[num_users] = resource_monitor.get_stats()
        
        # Generate visualizations
        generate_timeline_gantt(metrics_list, num_users)
        generate_latency_distributions(metrics_list, num_users)
        export_timeline_csv(timeline_events, num_users)
        
        # Save metrics to JSON
        json_path = os.path.join(OUTPUT_DIR, f"metrics_{num_users}_users.json")
        with open(json_path, 'w') as f:
            json.dump([asdict(m) for m in metrics_list], f, indent=2)
        logging.info(f"Metrics JSON saved: {json_path}")
        
        # Wait between tests
        if num_users < max(user_counts):
            logging.info(f"\nWaiting 10 seconds before next test...\n")
            await asyncio.sleep(10)
    
    # Generate summary report
    generate_summary_report(all_metrics, all_timelines, all_resource_stats)
    
    logging.info(f"\n{'='*80}")
    logging.info("All tests completed!")
    logging.info(f"Results saved to: {OUTPUT_DIR}")
    logging.info(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
