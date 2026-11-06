# BP product contract

Updated: 2026-11-04. Source: owner instructions in this chat, project constitution and the reviewed implementation plan. Product target is approved; LHC's actual interface remains an assumption.

## Intended outcome

An operator prepares audio once, selects a qualified language/voice route, checks the real input/output, and runs continuous translated speech and captions for a hybrid university meeting. Stop ends capture, synthesis and queued playback. A listener understands the translation; a maintainer can install, observe, recover and restore the system.

The complete specification includes personal cloned speech. Generic Piper is a diagnostic baseline, not completion of the full product. English↔Slovak is mandatory; Czech is evaluated alongside it. Other languages earn support through per-route evidence. Better processing hardware is an owner-reported requirement for the planned full cloning path; qualify the exact engine, revision and workload on that hardware.

## People and journey

- Operator: prepare direction/profile → sign in → verify installed assets → select speech input and translation output → local sound check → confirm headphones/isolated channels → Start → watch persistent route/capture state → recover from faults → Stop.
- Listener: readable source/translated captions and intelligible translated speech, without feedback, unexplained source loss or growing playback delay.
- Voice owner: record/re-record → explicitly upload to the BP host → process → authorize training separately → listen and grade → qualify clone on target hardware. Recording/uploading is not clone readiness.
- Maintainer: provision offline assets and permitted devices, document access/retention/licences, monitor capacity and timing, restore after failure and own support.

## Surfaces and visual authority

Keep DESIGN.md's berry/navy palette, Playfair headings and Montserrat body. Home explains the task and limits with one primary preparation CTA; thesis/research material is secondary. Live speech is an operating surface: preparation and persistent status before captions/playback, engine experiments behind advanced controls. Enrollment is separate from a first translation. Voice Lab is private research/evidence, not an implied public-media permission.

## Integration hypothesis

BP runs beside existing LHC infrastructure and feeds translated audio through a selected virtual output device. Exact room input, return channels, conference software, processing host, install permissions and supervisor acceptance are not yet known. A virtual mic does not prove input capture, echo isolation, sustained service or embedding. Add a host-specific adapter only after an actual interface contract exists. MCP, AIS, Moodle and SSO are not established requirements.

## Acceptance and non-claims

Every promoted route needs current assets/configuration, source recognition, reviewed translation meaning, target speech intelligibility, clone identity where required, runtime policy/licence review, real output artifacts and measured live delay/queue stability. WER, chrF++, isolated RTF, model loading and environment-variable presence cannot individually grant route acceptance.

Run a continuous meeting/presentation rehearsal on intended hardware with the actual listener path; define p50/p95 speech-end→first-audio and maximum queue delay before testing. Record interruption, device loss, expired-auth, reconnect and terminal-Stop results. Human grades and capacity targets require an acceptance owner; no universal commercial chrF++ threshold is asserted.

The November 4 Kaggle test measures whole-phrase OmniVoice generation and synthetic scheduling only. It excludes STT/MT, true streaming and actual meeting transport. Remain a prototype until full-route, clean-machine/per-OS, privacy/licence, recovery and named pilot checks pass. No completed LHC integration or commercial release is claimed.
