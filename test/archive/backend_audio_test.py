import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import os
import time
import logging

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
WS_URL = "ws://localhost:8000/ws" # Adjust if your FastAPI server runs on a different port/host
AUDIO_FILE_PATH = "test/Can you hear me_.wav" # Path to your test audio file
OUTPUT_DIR = "test_output" # Directory to save processed audio segments

async def run_test_pipeline():
    if not os.path.exists(AUDIO_FILE_PATH):
        logging.error(f"Audio file not found: {AUDIO_FILE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        async with websockets.connect(WS_URL) as websocket:
            logging.info("WebSocket connected for testing.")

            # Send initial config update (optional, but good practice)
            config_message = {
                "type": "config_update",
                "source_lang": "en",
                "target_lang": "sk",
                "tts_model_choice": "piper"
            }
            await websocket.send(json.dumps(config_message))
            logging.info("Sent initial config update.")

            # Wait for config update status
            response = json.loads(await websocket.recv())
            logging.info(f"Config update response: {response}")
            if response.get("status") == "error":
                logging.error(f"Failed to update config: {response.get('message')}")
                return

            # Send 'start' command
            await websocket.send(json.dumps({"type": "start"}))
            logging.info("Sent 'start' command.")

            # Load audio file
            audio_data, sample_rate = sf.read(AUDIO_FILE_PATH, dtype='float32')
            
            # Ensure audio is mono if it's stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample if necessary (assuming backend expects 16kHz)
            if sample_rate != 16000:
                logging.info(f"Resampling audio from {sample_rate}Hz to 16000Hz.")
                # Using torchaudio for resampling, need to convert to tensor
                import torch
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_data = resampler(torch.from_numpy(audio_data)).numpy()
                sample_rate = 16000
            
            logging.info(f"Loaded audio file: {AUDIO_FILE_PATH} with {audio_data.size} samples at {sample_rate}Hz.")

            # Simulate streaming audio chunks (e.g., 30ms chunks for VAD)
            chunk_size_samples = int(sample_rate * 0.030) # 30ms chunks
            total_chunks = (audio_data.size + chunk_size_samples - 1) // chunk_size_samples

            for i in range(total_chunks):
                start_idx = i * chunk_size_samples
                end_idx = min((i + 1) * chunk_size_samples, audio_data.size)
                chunk = audio_data[start_idx:end_idx]

                if chunk.size > 0:
                    await websocket.send(chunk.tobytes())
                    # Listen for messages from the server in between sending chunks
                    try:
                        # Use a timeout to prevent blocking indefinitely if no message is sent
                        response_str = await asyncio.wait_for(websocket.recv(), timeout=0.01) 
                        response = json.loads(response_str)
                        logging.debug(f"Received from server: {response.get('type')}")
                        if response.get("type") == "transcription_result":
                            logging.info(f"Transcription: {response.get('transcribed')} (Final: {response.get('is_final')})")
                        elif response.get("type") == "translation_result":
                            logging.info(f"Translation: {response.get('translated')} (Final: {response.get('is_final')})")
                        elif response.get("type") == "final_metrics":
                            logging.info(f"Metrics: {response.get('metrics')}")
                        elif response.get("type") == "audio_level":
                            pass # Ignore audio level for now
                        elif response.get("type") == "status":
                            logging.info(f"Status: {response.get('message')}")
                        elif response.get("type") == "error":
                            logging.error(f"Server Error: {response.get('message')}")
                            break # Stop on error
                    except asyncio.TimeoutError:
                        pass # No message received within the timeout, continue sending audio
                    except websockets.exceptions.ConnectionClosedOK:
                        logging.info("WebSocket closed by server during chunk sending.")
                        break
                    except Exception as e:
                        logging.error(f"Error receiving message during chunk sending: {e}")
                        break
                await asyncio.sleep(0.025) # Simulate real-time audio input (30ms chunk, 25ms delay)

            # Send 'stop' command to process any remaining audio
            await websocket.send(json.dumps({"type": "stop"}))
            logging.info("Sent 'stop' command.")

            # Listen for final responses after 'stop'
            while True:
                try:
                    response_str = await asyncio.wait_for(websocket.recv(), timeout=2.0) # Longer timeout for final processing
                    response = json.loads(response_str)
                    logging.info(f"Final response: {response.get('type')}")
                    if response.get("type") == "transcription_result":
                        logging.info(f"Final Transcription: {response.get('transcribed')} (Final: {response.get('is_final')})")
                    elif response.get("type") == "translation_result":
                        logging.info(f"Final Translation: {response.get('translated')} (Final: {response.get('is_final')})")
                    elif response.get("type") == "final_metrics":
                        logging.info(f"Final Metrics: {response.get('metrics')}")
                    elif response.get("type") == "status":
                        logging.info(f"Final Status: {response.get('message')}")
                    elif response.get("type") == "error":
                        logging.error(f"Final Server Error: {response.get('message')}")
                        break
                except asyncio.TimeoutError:
                    logging.info("No more messages from server after 'stop' command.")
                    break
                except websockets.exceptions.ConnectionClosedOK:
                    logging.info("WebSocket closed by server after 'stop' command.")
                    break
                except Exception as e:
                    logging.error(f"Error receiving final message: {e}")
                    break

    except websockets.exceptions.ConnectionClosedError:
        logging.error(f"Connection closed unexpectedly. Is the FastAPI server running at {WS_URL}?")
    except OSError as e:
        if "Connection refused" in str(e):
            logging.error(f"Connection refused. Is the FastAPI server running at {WS_URL}?")
        else:
            raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure torchaudio is imported for resampling
    try:
        import torchaudio
        import torch
    except ImportError:
        logging.error("torchaudio and torch are required for audio resampling in the test script. Please install them: pip install torch torchaudio")
        exit(1)
    
    asyncio.run(run_test_pipeline())
