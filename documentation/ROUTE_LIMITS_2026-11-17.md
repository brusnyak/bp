# Voice route hardware notes

Numbers below come from files in `documentation/evidence/2026-11-15/`. One run each unless stated; test audio is generated, not a live speaker.

## Slovak summary

prototyp prekladu reči v reálnom čase beží lokálne na mojom MacBooku (M1 Pro, 16 GB): angličtina → slovenčina / čeština / nemčina, preklad sa začína ozývať počas vety, s mediánom oneskorenia 0,8 s (p90 1,4 s) na 60 s testu, chybovosť rozpoznávania 5 %. Hlas je zatiaľ generický.

Na testoch na GPU (Kaggle T4, 16 GB) som overil modely na klonovanie hlasu:
- XTTS v2: prvý zvuk po 0,45 s, ale nepodporuje slovenčinu (len češtinu a angličtinu).
- X-Voice: slovenčina aj čeština, chybovosť do 11 %, ale 2,2–2,7 s na vetu, čo je na T4 príliš pomalé.
- Higgs Audio v3 (4B): potrebuje GPU s podporou bf16 a približne 24 GB, na T4 nevyprodukoval zrozumiteľný zvuk.

Na plnú verziu (hlas mojím vlastným hlasom, slovenský vstup a viac poslucháčov naraz) potrebujem lepší výpočtový výkon: jednu NVIDIA GPU s 24 GB (generácia Ampere alebo novšia, napr. RTX 3090/4090, A10, L4), s prístupom na testovanie. Na tom by som dokončil testy a pripravil integráciu do konferenčných aplikácií.

## English (same content, for checking)

The prototype runs locally on a MacBook (M1 Pro, 16 GB): English to Slovak / Czech / German, voice starts during the sentence, median lag 0.8 s (p90 1.4 s) on a 60 s test, recognition error 5%. Voice is still generic.
On GPU tests (Kaggle T4, 16 GB) of voice-cloning models: XTTS v2 first audio at 0.45 s but no Slovak; X-Voice has Slovak and Czech with error up to 11% but needs 2.2-2.7 s per sentence on a T4; Higgs Audio v3 (4B) needs bf16 and about 24 GB, and produced no intelligible audio on a T4.
For the full version (own voice, Slovak input, several listeners) I need one NVIDIA GPU with 24 GB (Ampere or newer), ideally a processing server with remote access, to finish testing and prepare integration into the conference-room system.

## Limits
- Not claimed: Slovak speech input under 2 s (best measured 2.8 s on T4 simulation), cloned voice at live speed.
