import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import os
import time
import logging
from collections import deque
import sys # Import sys for sys.exit
from typing import List, Dict, Any, Tuple, Optional
import ssl # Import ssl for WebSocket connection

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
WS_URL = "wss://localhost:8000/ws" # Use wss for secure WebSocket
CERT_PATH = "certs/cert.pem" # Path to the self-signed certificate
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD and STT
VAD_FRAME_DURATION = 20 # ms - matching backend
AUDIO_CHUNK_LENGTH_SECONDS = VAD_FRAME_DURATION / 1000 # Send audio in VAD frame durations

# Test audio files (from the 'test/' directory)
TEST_AUDIO_FILES = [
    "test/Can you hear me_.wav",
    "test/Hello.wav",
    "test/My test speech_xtts_speaker_clean.wav",
]

# Output directory for test results and generated charts
OUTPUT_DIR = "test_output/concurrent_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def simulate_user_session(
    session_id: int,
    audio_file_path: str,
    source_lang: str,
    target_lang: str,
    tts_model_choice: str,
    speaker_wav_path: Optional[str] = None,
    speaker_lang: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulates a single user session, connecting to the WebSocket,
    sending audio, and collecting performance metrics.
    """
    session_results = {
        "session_id": session_id,
        "audio_file": audio_file_path,
        "status": "failed",
        "error": None,
        "metrics": {
            "stt_latencies": [],
            "mt_latencies": [],
            "tts_latencies": [],
            "e2e_latencies": [], # From first audio chunk sent to final TTS audio received
            "total_session_time": 0.0
        },
        "transcriptions": [],
        "translations": [],
        "tts_audio_chunks_received": 0
    }

    logging.info(f"Session {session_id}: Starting simulation for {audio_file_path}")
    session_start_time = time.perf_counter()

    # Create SSL context to trust the self-signed certificate
    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(CERT_PATH)
    ssl_context.check_hostname = False # Disable hostname check for localhost self-signed cert

    try:
        async with websockets.connect(WS_URL, ssl=ssl_context) as websocket:
            logging.info(f"Session {session_id}: WebSocket connected.")

            # Send initial config update
            config_message = {
                "type": "config_update",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "tts_model_choice": tts_model_choice,
                "speaker_wav_path": speaker_wav_path,
                "speaker_lang": speaker_lang
            }
            await websocket.send(json.dumps(config_message))
            logging.debug(f"Session {session_id}: Sent initial config update.")

            # Wait for config update status
            response_str = await asyncio.wait_for(websocket.recv(), timeout=10)
            response = json.loads(response_str)
            if response.get("status") == "error":
                session_results["error"] = f"Config update failed: {response.get('message')}"
                logging.error(f"Session {session_id}: {session_results['error']}")
                return session_results
            logging.info(f"Session {session_id}: Config update response: {response.get('message')}")

            # Send 'start' command
            await websocket.send(json.dumps({"type": "start"}))
            logging.debug(f"Session {session_id}: Sent 'start' command.")

            # Load audio file
            audio_data, sample_rate = sf.read(audio_file_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample audio to AUDIO_SAMPLE_RATE if necessary
            if sample_rate != AUDIO_SAMPLE_RATE:
                from scipy.signal import resample
                num_samples = int(len(audio_data) * AUDIO_SAMPLE_RATE / sample_rate)
                audio_data = resample(audio_data, num_samples)
                sample_rate = AUDIO_SAMPLE_RATE
            
            logging.debug(f"Session {session_id}: Loaded audio file with {audio_data.size} samples at {sample_rate}Hz.")

            # Chunk the audio for streaming
            chunk_size_samples = int(sample_rate * AUDIO_CHUNK_LENGTH_SECONDS)
            audio_chunks = [audio_data[i:i + chunk_size_samples] for i in range(0, len(audio_data), chunk_size_samples)]

            first_audio_sent_time = None
            last_tts_audio_received_time = None

            # Task to receive messages from the server
            async def receive_messages():
                nonlocal last_tts_audio_received_time
                while True:
                    try:
                        # Receive raw data, could be text or bytes
                        raw_response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        
                        if isinstance(raw_response, str):
                            response = json.loads(raw_response)
                            # Handle JSON messages as before
                            if response.get("type") == "transcription_result":
                                session_results["transcriptions"].append(response.get("transcribed"))
                                logging.debug(f"Session {session_id}: Transcription: {response.get('transcribed')}")
                            elif response.get("type") == "translation_result":
                                session_results["translations"].append(response.get("translated"))
                                logging.debug(f"Session {session_id}: Translation: {response.get('translated')}")
                            elif response.get("type") == "final_metrics":
                                metrics = response.get("metrics", {})
                                session_results["metrics"]["stt_latencies"].append(metrics.get("stt_time", 0.0))
                                session_results["metrics"]["mt_latencies"].append(metrics.get("mt_time", 0.0))
                                session_results["metrics"]["tts_latencies"].append(metrics.get("tts_time", 0.0))
                                # E2E latency will be calculated at the end of the session
                                logging.debug(f"Session {session_id}: Metrics: {metrics}")
                            elif response.get("type") == "error":
                                session_results["error"] = f"Server Error: {response.get('message')}"
                                logging.error(f"Session {session_id}: {session_results['error']}")
                                break
                            elif response.get("type") == "status":
                                logging.debug(f"Session {session_id}: Status: {response.get('message')}")
                        elif isinstance(raw_response, bytes):
                            # This is likely TTS audio
                            session_results["tts_audio_chunks_received"] += 1
                            last_tts_audio_received_time = time.perf_counter()
                            logging.debug(f"Session {session_id}: Received TTS audio chunk (bytes).")
                        else:
                            logging.warning(f"Session {session_id}: Received unknown message type: {type(raw_response)}")

                    except asyncio.TimeoutError:
                        pass
                    except json.JSONDecodeError as e:
                        logging.error(f"Session {session_id}: JSONDecodeError: {e} - Raw response: {raw_response[:100]}...")
                        session_results["error"] = f"JSON decoding error: {e}"
                        break
                    except websockets.exceptions.ConnectionClosedOK:
                        logging.info(f"Session {session_id}: WebSocket closed by server during receive.")
                        break
                    except Exception as e:
                        session_results["error"] = f"Error receiving message: {e}"
                        logging.error(f"Session {session_id}: {session_results['error']}")
                        break

            receive_task = asyncio.create_task(receive_messages())

            # Stream audio chunks
            for i, chunk in enumerate(audio_chunks):
                if chunk.size > 0:
                    if first_audio_sent_time is None:
                        first_audio_sent_time = time.perf_counter()
                    await websocket.send(chunk.tobytes())
                await asyncio.sleep(AUDIO_CHUNK_LENGTH_SECONDS * 0.8) # Simulate real-time input with slight buffer

            # Send 'stop' command
            await websocket.send(json.dumps({"type": "stop"}))
            logging.debug(f"Session {session_id}: Sent 'stop' command.")

            # Give some time for final messages to be processed
            await asyncio.sleep(5) # Wait for 5 seconds for final processing

            receive_task.cancel() # Cancel the receive task
            try:
                await receive_task # Await to catch any cancellation exceptions
            except asyncio.CancelledError:
                logging.debug(f"Session {session_id}: Receive task cancelled.")

            session_results["status"] = "completed"

    except websockets.exceptions.ConnectionClosedError as e:
        session_results["error"] = f"Connection closed unexpectedly: {e}"
        logging.error(f"Session {session_id}: {session_results['error']}")
    except OSError as e:
        session_results["error"] = f"OS Error: {e}. Is the FastAPI server running at {WS_URL}?"
        logging.error(f"Session {session_id}: {session_results['error']}")
    except Exception as e:
        session_results["error"] = f"An unexpected error occurred: {e}"
        logging.error(f"Session {session_id}: {session_results['error']}")
    finally:
        session_results["metrics"]["total_session_time"] = time.perf_counter() - session_start_time
        
        # Calculate E2E latency once at the end of the session
        if first_audio_sent_time is not None and last_tts_audio_received_time is not None:
            e2e_latency = last_tts_audio_received_time - first_audio_sent_time
            session_results["metrics"]["e2e_latencies"].append(e2e_latency)
            logging.info(f"Session {session_id}: Final E2E Latency (calculated at end): {e2e_latency:.2f}s")
        elif session_results["status"] == "completed": # If session completed but no TTS audio, it's a partial success
            logging.warning(f"Session {session_id}: Completed but no TTS audio received. E2E latency not calculated.")
            session_results["status"] = "completed_no_tts_audio" # Mark as partial success
        
        logging.info(f"Session {session_id}: Finished in {session_results['metrics']['total_session_time']:.2f}s with status: {session_results['status']}")
        return session_results

async def run_concurrent_test(num_users: int, audio_files: List[str], tts_model: str = "piper"):
    """
    Runs concurrent translation tests for a specified number of users.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting concurrent test for {num_users} users with {tts_model} TTS")
    logging.info(f"{'='*80}")

    tasks = []
    for i in range(num_users):
        # Loop through audio files for diversity
        audio_file = audio_files[i % len(audio_files)]
        tasks.append(
            simulate_user_session(
                session_id=i + 1,
                audio_file_path=audio_file,
                source_lang="en",
                target_lang="sk", # Assuming English to Slovak for now
                tts_model_choice=tts_model,
                speaker_wav_path=None, # Not used for Piper
                speaker_lang=None # Not used for Piper
            )
        )
    
    all_results = await asyncio.gather(*tasks, return_exceptions=True) # Collect results, including exceptions

    successful_results = [r for r in all_results if isinstance(r, dict) and r.get("status") == "completed"]
    failed_results = [r for r in all_results if not (isinstance(r, dict) and r.get("status") == "completed")]

    logging.info(f"\n--- Concurrent Test Summary for {num_users} Users ---")
    logging.info(f"Total sessions: {len(all_results)}")
    logging.info(f"Successful sessions: {len(successful_results)}")
    logging.info(f"Failed sessions: {len(failed_results)}")

    if successful_results:
        all_e2e_latencies = [m for r in successful_results for m in r["metrics"]["e2e_latencies"]]
        all_stt_latencies = [m for r in successful_results for m in r["metrics"]["stt_latencies"]]
        all_mt_latencies = [m for r in successful_results for m in r["metrics"]["mt_latencies"]]
        all_tts_latencies = [m for r in successful_results for m in r["metrics"]["tts_latencies"]]
        all_total_session_times = [r["metrics"]["total_session_time"] for r in successful_results]

        logging.info(f"Average E2E Latency: {np.mean(all_e2e_latencies):.2f}s (Std Dev: {np.std(all_e2e_latencies):.2f}s)")
        logging.info(f"Max E2E Latency: {np.max(all_e2e_latencies):.2f}s")
        logging.info(f"Min E2E Latency: {np.min(all_e2e_latencies):.2f}s")
        
        logging.info(f"Average STT Latency: {np.mean(all_stt_latencies):.2f}s")
        logging.info(f"Average MT Latency: {np.mean(all_mt_latencies):.2f}s")
        logging.info(f"Average TTS Latency: {np.mean(all_tts_latencies):.2f}s")
        logging.info(f"Average Total Session Time: {np.mean(all_total_session_times):.2f}s")
    else:
        logging.warning("No successful sessions to report metrics.")

    if failed_results:
        logging.error("\n--- Failed Sessions Details ---")
        for r in failed_results:
            logging.error(f"Session {r.get('session_id', 'N/A')}: Status={r.get('status', 'N/A')}, Error={r.get('error', 'Unknown error')}")
    
    return successful_results, failed_results

def generate_latency_chart(results: List[Dict[str, Any]], num_users: int, tts_model: str):
    """
    Generates a bar chart for average component latencies and E2E latency.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logging.error("Matplotlib and Seaborn are required for chart generation. Please install them: pip install matplotlib seaborn")
        return

    if not results:
        logging.warning("No data to generate chart.")
        return

    avg_stt = np.mean([m for r in results for m in r["metrics"]["stt_latencies"]])
    avg_mt = np.mean([m for r in results for m in r["metrics"]["mt_latencies"]])
    avg_tts = np.mean([m for r in results for m in r["metrics"]["tts_latencies"]])
    avg_e2e = np.mean([m for r in results for m in r["metrics"]["e2e_latencies"]])

    labels = ['STT', 'MT', 'TTS', 'E2E']
    averages = [avg_stt, avg_mt, avg_tts, avg_e2e]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=averages, palette='viridis')
    plt.title(f'Average Latency for {num_users} Concurrent Users ({tts_model} TTS)')
    plt.ylabel('Latency (seconds)')
    plt.xlabel('Component')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_filename = os.path.join(OUTPUT_DIR, f"latency_chart_{num_users}_users_{tts_model}.png")
    plt.savefig(chart_filename)
    logging.info(f"Generated latency chart: {chart_filename}")
    plt.close()

async def main():
    # Clear terminal before starting tests
    os.system('cls' if os.name == 'nt' else 'clear')
    logging.info("Terminal cleared. Waiting 2 seconds before starting tests...")
    await asyncio.sleep(2) # Wait 2 seconds as requested

    logging.info("Ensuring test audio files exist...")
    for f in TEST_AUDIO_FILES:
        if not os.path.exists(f):
            logging.error(f"Required audio file not found: {f}. Please ensure it exists in the 'test/' directory.")
            sys.exit(1)
    
    user_counts = [5, 10] # Test for 5 and 10 concurrent users
    tts_model = "piper" # As per user's request

    all_successful_results = []

    for num_users in user_counts:
        logging.info(f"\n--- Running concurrent test for {num_users} users ---")
        successful, failed = await run_concurrent_test(num_users, TEST_AUDIO_FILES, tts_model)
        all_successful_results.extend(successful)
        if successful:
            generate_latency_chart(successful, num_users, tts_model)
        
        # Add a small delay between different user count tests to allow resources to clear
        logging.info(f"Waiting 10 seconds before next test run for {num_users} users...")
        await asyncio.sleep(10)

    logging.info("\nAll concurrent tests completed.")
    logging.info(f"Results and charts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Ensure torchaudio and torch are imported for resampling if needed by audio_utils
    try:
        import torch
        # No direct torchaudio import needed here, as scipy.signal.resample is used
    except ImportError:
        logging.warning("Torch not found. Resampling might be less optimized if audio_utils relies on it.")
    
    # Run the main async function
    asyncio.run(main())
