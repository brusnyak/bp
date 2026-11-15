# Connect translated speech to a room or call

Run Live Speech Translation on a processing machine. Choose an audio source, then send translated audio to a meeting application's microphone, an AV mixer, or listeners' browsers. No particular room system is required. Start with generic Piper voices and one isolated speaker.

## Route audio through a virtual device

Install and configure a virtual audio device appropriate to the host. The app does not install these drivers. Confirm the route with the meeting application's input meter before admitting listeners.

| Host | Virtual route | Live page **Output** | Meeting application microphone |
|---|---|---|---|
| macOS | BlackHole | The installed BlackHole device, such as **BlackHole 2ch** | The same BlackHole device |
| Windows | VB-Cable | **CABLE Input (VB-Audio Virtual Cable)**, the playback end | **CABLE Output (VB-Audio Virtual Cable)**, the recording end |
| Linux | PipeWire or PulseAudio null sink | The virtual sink's playback device, if exposed to the browser | The sink's monitor source, selected in the call app or OS routing controls |

Device names depend on the installed driver and OS. For Linux, create the virtual sink in the host's audio routing tools and connect its monitor to the call input; browser visibility differs by session and audio service.

1. Open the live page in Chrome or Edge. Prepare the chosen language route.
2. In the top controls, select the physical or AV source under **Microphone**. Use **Test microphone** and verify the level meter.
3. Select the virtual device under **Output** beside **Microphone**. Output is in the top controls in the current interface. Open **More** for the section **Sending the translation into a meeting?**, which explains the cable route.
4. In the meeting application, choose the cable's recording end as its microphone. The table above distinguishes the two VB-Cable ends.
5. Start translation and verify the remote listener hears the translated voice. Stop the session to confirm audio and queued playback end.

Keep meeting playback on headphones or an isolated mixer bus. Do not feed translated output or the call's return audio into the source microphone. A virtual device being listed does not prove that the meeting app is listening to it. Safari and Firefox do not provide the required output selection; use OS routing if browser output selection is unavailable.

For physical AV equipment, send the selected output to a dedicated mixer input or interface channel. Agree level, connector, channel mapping and echo isolation with the AV team.

## Captions and audience phones

The live page shows source and translated captions. Press **Pop out subtitles** in the **Translation** panel for a floating subtitle window where the browser supports Picture-in-Picture. In-page captions remain available when it does not.

For one speaker and many listeners:

1. Prepare the speaker's route and open **More → Audience broadcast (room mode) → Start room**.
2. Share **Copy audience link**, **Open audience link**, or the displayed QR code. Use the generated room; typing an arbitrary room code does not create one.
3. Listeners open the link on their phones, choose their language and start listening with earbuds. Each requested language needs an installed translation model and voice. The listener page obtains a guest token automatically when guest access is enabled.
4. The speaker starts translation and keeps the speaker connection open. A room ends when its speaker disconnects. One translation and voice are generated per requested language, then shared among listeners.

For access from phones or other computers on a trusted LAN, run from the repository root:

```sh
BP_HTTPS=1 BP_HOST=0.0.0.0 make run
```

This exposes port 8000 on all interfaces and uses the certificate and key in `certs/` prepared by setup. Use the processing machine's LAN address in the audience link, not `localhost`. Permit that port only on the intended network. Provide a trusted HTTPS certificate or explicitly trust the local certificate on each device. HTTPS is needed for microphone capture from a phone or any non-loopback browser; listeners do not capture audio and can use plain HTTP on a trusted LAN, but HTTPS is recommended for shared use and protects tokens in transit.

The command above uses a POSIX shell. Windows PowerShell equivalent:

```powershell
$env:BP_HTTPS="1"
$env:BP_HOST="0.0.0.0"
.venv\Scripts\python.exe app.py
```

Room mode is experimental. Model-free tests cover room logic, not the quality or capacity of a real meeting. Rehearse with the actual language set, listener count, phones and network before an event. Guest access can be disabled with `BP_GUEST=0`; disabling it also prevents automatic guest joining from the listener page. Review registration, access and network policy before exposing a shared host.

## HTTP and WebSocket integration

The current API is experimental. The following wire format follows `app.py` and `backend/main.py`. See [sdk.md](sdk.md) for the Python facade and expanded reference; its older examples may describe different defaults. The running server's `/docs` describes the HTTP endpoints.

| Step | Contract |
|---|---|
| Inspect | `GET /api/routes` for declared routes and installed assets; `GET /api/capabilities` for available engines; `GET /api/health` for health |
| Start a session | `POST /api/guest` is always enabled; returns `{token, guest_id}` with a distinct random ID and a two-hour signed token |
| Save a reference | `POST /api/voice-reference`, multipart `file`, Bearer token; WAV/M4A/MP3/WebM, max 15 MB and 60 s, at most five local files; returns `{name}` |
| List references | `GET /api/voice-reference`, Bearer token; returns `[{name}]` for this machine |
| Delete a reference | `DELETE /api/voice-reference?name=FILENAME`, Bearer token; returns `{name}` |
| Connect | `ws://HOST:8000/ws?token=TOKEN`, or `wss://` under HTTPS. Authenticate before model loading |
| Configure | Send JSON `{"type":"config_update","source_lang":"en","target_lang":"sk","tts_model_choice":"piper"}`; wait for `models_loading_status` and inspect route readiness |
| Start | Send `{"type":"start"}`, then binary **16 kHz mono float32 PCM** frames. Do not send a compressed MediaRecorder container or int16 samples |
| Receive | JSON `transcription_result` (`transcribed`) and `translation_result` (`translated`), plus binary WAV audio. Decode each WAV's own sample rate; do not assume a fixed output rate |
| Stop | Send `{"type":"stop"}` and clear the client's playback queue. Late work must not be played |
| Create a room | Authenticated `POST /api/room` returns `room_id`, `listen_path` and `listen_url`. Bearer token required |
| Listen | `/ws/listen?room=ID&lang=sk&token=TOKEN`; receives `room_caption` JSON and binary WAV. JSON `{"type":"set_lang","lang":"de"}` changes listener language |

Select a reference by passing its returned `name` as `speaker_wav_path` during configuration. References are local files shared across anonymous sessions; nothing is tied to an identity database.

Keep tokens out of logs and shared URLs. Preinstall assets before capture; a socket connection does not prepare the route. A successful request or loaded model does not establish language quality.

## Deployment checklist

Complete this table with the AV/IT team before choosing the final route.

| Question | Deployment answer |
|---|---|
| What supplies input: lectern microphone, mixer bus, USB interface or meeting return? | To fill in |
| What receives output: virtual cable, mixer channel, call microphone or audience browsers? | To fill in |
| Which software joins the call, and which machine runs it? | To fill in |
| Which machine hosts BP? OS, CPU, RAM and NVIDIA GPU model/VRAM, if present? | To fill in |
| Which source/target languages and voices are installed and tested? | To fill in |
| How many listeners and distinct output languages must run at once? | To fill in |
| Can devices reach the host? Wi-Fi isolation, firewall, port, HTTPS and certificate trust? | To fill in |
| Who administers the host, local recordings, model updates and recovery? | To fill in |
| What consent, privacy, recording and transcript-retention rules apply? | To fill in |
| Who checks model/voice licences and permission to use any cloned voice? | To fill in |
| What is the fallback if translation or the network fails? | To fill in |

## Troubleshooting

- **Echo or feedback:** use headphones; isolate the speaker's input from translated output and meeting return. Check that the cable is not also the source microphone. Mute one path at a time to identify the loop.
- **No microphone level:** select the device explicitly under **Microphone** and run **Test microphone**. The OS default may be an unplugged jack input. Check browser permission, OS permission, mixer mute and interface channel.
- **Translation audible locally but absent from the call:** check **Output**, the cable's two ends, and the call app's input meter. Stop/restart the call's sound check after changing its microphone.
- **Lag keeps growing:** inspect **More → Diagnostics: latency and activity log**. Separate slow recognition or voice generation from playback backlog. Try Piper, reduce simultaneous languages or competing compute, and use **Translation speed → Auto: speed up only when behind**. Stop to clear queued playback before retesting.
- **Phone cannot join:** confirm the host binds to the LAN, the link uses its LAN address, the firewall allows the port and Wi-Fi does not isolate clients. Resolve certificate trust. Check guest access and that the speaker's room is still open.
- **One listener language stays silent:** confirm that exact source-to-target pair and its assets are installed. A language listed elsewhere does not imply every pair exists.

[Product site](site/index.html) · [Documentation index](INDEX.md) · [Language evidence](LANGUAGES.md)

Created by Yegor Brusnyak, MIT licence
