# Recording script v2 — phonetic coverage in minimum minutes

Why a new script: the v1 demo sentences teach tech vocabulary, not sounds.
VITS learns timbre from phonetic coverage — every missing phoneme is a sound
it mumbles forever. Model: the Rainbow Passage (standard English, all phonemes,
used by VCTK for all 110 speakers) + an original Slovak narrative built on the
same principle, in the ElevenLabs-IVC spirit (casual story, varied prosody).

Record each passage in ONE session, steady pace, short pauses at punctuation.
Peak ~0.5 (v1 takes peaked 0.14 — clean but quiet). Total new material ≈ 4 min
per language including the existing v1 demo sentences as session 2.

## EN — The Rainbow Passage (Fairbanks, public domain, ~300 words ≈ 2.5 min)

When the sunlight strikes raindrops in the air, they act as a prism and form
a rainbow. The rainbow is a division of white light into many beautiful colors.
These take the shape of a long round arch, with its path high above, and its
two ends apparently beyond the horizon. There is, according to legend, a
boiling pot of gold at one end. People look, but no one ever finds it. When a
man looks for something beyond his reach, his friends say he is looking for
the pot of gold at the end of the rainbow. Throughout the centuries people
have explained the rainbow in various ways. Some have accepted it as a miracle
without physical explanation. To the Hebrews it was a token that there would
be no more universal floods. The Greeks used to imagine that it was a sign
from the gods to foretell war or heavy rain. The Norsemen considered the
rainbow as a bridge over which the gods passed from earth to their home in
the sky. Others have tried to explain the phenomenon physically. Aristotle
thought that the rainbow was caused by reflection of the sun's rays by the
rain. Since then physicists have found that it is not reflection, but
refraction by the raindrops which causes the rainbow. Many complicated ideas
about the rainbow have been formed. The difference in the rainbow depends
considerably upon the size of the drops, and the width of the rainbow
increases as the size of the drops increases. The actual primary rainbow
observed is said to be the effect of superposition of a number of bows. If the
red of the second bow falls upon the green of the first, the result is to give
a bow with an abnormally wide yellow band, since red and green light when
mixed form yellow.

## SK — Trhové ráno (original, ~200 words ≈ 1.5–2 min)

Phonetic checklist covered: ä (deväť), ô (vôňa, môžem), ia/ie (piatok,
priateľa), ľ/ĺ (Ľubo, ľalie, kĺb), ŕ (vŕba, vŕzgali), dz (medzi), dž (džbán),
dž (hádzali), ch (chlieb), long á/é/í/ú/ý, questions, exclamation, dialogue.
Rhythmic law respected (no long+long inside words).

Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. Stretol som tam
starého priateľa Ľuba, ktorý predával med a syry. „Ahoj! Kde si sa túľal celé
mesiace?" spýtal sa ma s úsmevom. Porozprával som mu o svojej práci a o dlhej
ceste vlakom cez hory a doliny, až ma bolel každý kĺb.

Po chvíli sme si sadli na lavičku pod veľkú vŕbu a dali sme si kávu. „Môžem
prísť aj zajtra?" spýtal som sa. Okolo nás pobehovali deti a hádzali si
farebnú loptu. Jedno dievčatko nieslo džbán plný malín a smialo sa na celé
kolo. Vôňa čerstvého pečiva sa niesla vzduchom a ja som si spomenul na detstvo
u starých rodičov. V diaľke vŕzgali brzdy električky.

Ľubo mi potom ukázal fotografie zo svojej záhrady. Medzi riadkami cibule a
cesnaku rástli ľalie, pivonky a neskutočne veľké tekvice. „Pozri, táto váži
vyše deväť kíl!" pochválil sa. Neveril som vlastným očiam. „Musím už ísť, ale
v piatok prídem zas," povedal som a pobrali sme sa domov.

Ešte dnes cítim chuť toho chleba s maslom a medom. Bolo to jedno z
najkrajších rán tohto jesenného mesiaca. Koľko takýchto dní nám ešte život
prinesie?

## Recording spec (both)

- Quiet room matters more than the microphone. One session per passage.
- Read naturally, not performed — the ElevenLabs parrot passage works because it
  is casual varied speech, not announcer voice. Same energy here.
- Numbers are spelled out (match the transcript exactly — transcript lies become
  training lies).
- Leave 0.5s silence at start/end of each take for the silence splitter.
- After recording: segment (`scripts/segment_sk_dataset.py` pattern), QC in lab,
  then fine-tune per `documentation/voice_build_runbook.md`.

## CZ — Staroměstské odpoledne (optional third language, ~150 words ≈ 1.5 min)

Why Czech: closest phonetic neighbor (mutually intelligible, shared pipeline),
XTTS already speaks it (cross-checks), and it adds ř/ě/ů/ou the SK set lacks.
Same male-speaker warmstart logic applies (jirka is literally this language).

Včera odpoledne jsem šel na Staroměstské náměstí. Potkal jsem tam starého
kamaráda, který prodával housky a koláče. „Tři rohlíky a čtyři koláče,
prosím!" poprosil jsem. „Ahoj! Kde ses toulal celé měsíce?" zeptal se mě.
Vypravoval jsem mu o své práci a o dlouhé cestě vlakem přes hory a údolí.

Pak jsme si sedli na lavičku a dali si kávu. Okolo běhaly děti a házely si
míčem. V pekárně voněla čerstvá mouka a pečivo. „Věřil bys, že tohle místo
znám už čtyřicet let?" řekl a ukázal na věž. Musel jsem se smát.

Když se setmělo, vydali jsme se domů přes Karlův most. Kousek od mostu stojí
můj starý dům. Nad řekou kroužili racci a z hospody se ozývala hudba. Bylo to
jedno z nejhezčích odpolední tohoto měsíce. Kdy se sem zase vrátím?
