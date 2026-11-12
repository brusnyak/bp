# Reading script: English, Slovak, German (2026-11-07)

Read **one language per recording**, top to bottom, in this order: A, B, C, D. Leave about 3 seconds of silence between sections. If you stumble, stop, wait 2 seconds and read the sentence again from its start (the scoring tool keeps the last attempt).

How to record (this matters more than the microphone): quiet room, mouth 20-30 cm from the microphone, speak at normal conversation volume, and **raise the input gain until the loudest words peak around -6 to -3 dB** (the old recordings peaked at -17 dB, which cost recognition accuracy). Save as `en_script_2.m4a`, `sk_script_2.m4a`, `de_script_1.m4a` in `~/Downloads` (any name works, tell me the path).

Section C tests pauses. Read it as written: `/` is a short pause (about 0.3 s), `//` is a long pause (about 1 s). Do not rush them.

Score a recording with `python scripts/score_reading.py <audio> <en|sk|de>` (word error per section, pause statistics).

## English (EN)

A1. Good morning, and welcome to this live demonstration of speech translation.
A2. I am speaking English, and you will hear German or Slovak within a second or two.
A3. The system listens through the microphone and decides when a phrase has ended.
A4. First the words are recognised, then they are translated, then they are spoken again.
A5. Everything runs on this computer, without a cloud service.
A6. The delay is the time between my words and the voice you hear.
A7. Our goal is a delay short enough for a natural conversation.
A8. A quiet room matters more than an expensive microphone.
A9. Please speak at a steady pace, with short pauses between sentences.
A10. Names, numbers and technical terms need extra care.
A11. Questions from the audience are translated in the opposite direction.
A12. Thank you for listening, and I look forward to your questions.
B1. The meeting starts at 9:45 on the 14th of March, in room 2.17.
B2. Mr. Novák from Bratislava will present the results of 3,250 measurements.
B3. The latency target is 800 milliseconds, and the error rate must stay below 10 percent.
B4. Our servers run CUDA 12 on a Tesla T4 with 16 gigabytes of memory.
B5. Please send the report to the CERN office in Geneva by Friday.
B6. The model has 600 million parameters and runs at twice real time.
C1. Good morning. // Today I want to show you three things. / First, / how the system listens. // Second, / how it translates. // And third, / why the pauses matter. // When I stop speaking, / the voice should stop too. // And when I pause, / you should hear a pause.
D1. Can you hear me?
D2. What time does the session start?
D3. Could you repeat that, please?
D4. I think we need a short break.
D5. Let me explain that with an example.

## Slovak (SK)

A1. Dobré ráno a vitajte na tejto živej ukážke prekladu reči.
A2. Hovorím po anglicky a o sekundu či dve budete počuť nemčinu alebo slovenčinu.
A3. Systém počúva cez mikrofón a rozhoduje, kedy sa veta skončila.
A4. Najprv sa slová rozpoznajú, potom sa preložia a nakoniec znova vyslovia.
A5. Všetko beží na tomto počítači, bez cloudovej služby.
A6. Oneskorenie je čas medzi mojimi slovami a hlasom, ktorý počujete.
A7. Naším cieľom je oneskorenie dosť krátke na prirodzený rozhovor.
A8. Tichá miestnosť je dôležitejšia ako drahý mikrofón.
A9. Hovorte, prosím, rovnomerným tempom a s krátkymi prestávkami medzi vetami.  
A10. Mená, čísla a odborné výrazy vyžadujú osobitnú pozornosť.  
A11. Otázky z publika sa prekladajú opačným smerom.  
A12. Ďakujem za pozornosť a teším sa na vaše otázky.  
B1. Stretnutie sa začína o 9:45 štrnásteho marca v miestnosti 2.17.  
B2. Pán Novák z Bratislavy predstaví výsledky 3 250 meraní.  
B3. Cieľové oneskorenie je 800 milisekúnd a chybovosť musí zostať pod desať percent.  
B4. Naše servery bežia na CUDA 12 s grafickou kartou Tesla T4 so 16 gigabajtmi pamäte.  
B5. Pošlite, prosím, správu do kancelárie CERN v Ženeve do piatka.  
B6. Model má 600 miliónov parametrov a beží dvakrát rýchlejšie ako v reálnom čase.  
C1. Dobré ráno. // Dnes vám chcem ukázať tri veci. / Po prvé, / ako systém počúva. // Po druhé, / ako prekladá. // A po tretie, / prečo záleží na prestávkach. // Keď prestanem hovoriť, / hlas by mal prestať tiež. // A keď spravím pauzu, / mali by ste počuť pauzu.  
D1. Počujete ma?  
D2. O koľkej sa začína relácia?  
D3. Mohli by ste to, prosím, zopakovať?  
D4. Myslím, že potrebujeme krátku prestávku.  
D5. Dovoľte mi to vysvetliť na príklade.  


## Czech (CS)

A1. Dobré ráno a vítejte u této živé ukázky překladu řeči.
A2. Mluvím anglicky a za vteřinu či dvě uslyšíte němčinu nebo slovenštinu.
A3. Systém poslouchá přes mikrofon a určuje, kdy věta skončila.
A4. Nejprve se slova rozpoznají, poté se přeloží a nakonec se znovu vysloví.
A5. Vše probíhá na tomto počítači, bez cloudové služby.
A6. Zpoždění je časový odstup mezi mými slovy a hlasem, který slyšíte.
A7. Naším cílem je, aby zpoždění bylo dostatečně krátké pro přirozený rozhovor.
A8. Tichá místnost je důležitější než drahý mikrofon.
B1. Schůzka začíná ve 9:45 čtrnáctého března v místnosti 2.17.
B2. Pan Novák z Bratislavy představí výsledky 3 250 měření.
B3. Cílové zpoždění je 800 milisekund a chybovost musí zůstat pod deset procent.
C1. Dobré ráno. // Dnes vám chci ukázat tři věci. / Za prvé, / jak systém poslouchá. // Za druhé, / jak překládá. // A za třetí, / proč záleží na pauzách. // Když přestanu mluvit, / hlas by měl přestat také. // A když udělám pauzu, / měli byste slyšet pauzu.

## German (DE)

A1. Guten Morgen und willkommen zu dieser Live-Vorführung der Sprachübersetzung.
A2. Ich spreche Englisch, und Sie hören nach ein bis zwei Sekunden Deutsch oder Slowakisch.
A3. Das System hört über das Mikrofon zu und entscheidet, wann ein Satz zu Ende ist.
A4. Zuerst werden die Wörter erkannt, dann übersetzt und schließlich wieder gesprochen.
A5. Alles läuft auf diesem Computer, ohne Cloud-Dienst.
A6. Die Verzögerung ist die Zeit zwischen meinen Worten und der Stimme, die Sie hören.
A7. Unser Ziel ist eine Verzögerung, die kurz genug für ein natürliches Gespräch ist.
A8. Ein ruhiger Raum ist wichtiger als ein teures Mikrofon.
A9. Bitte sprechen Sie in gleichmäßigem Tempo und mit kurzen Pausen zwischen den Sätzen.
A10. Namen, Zahlen und Fachbegriffe brauchen besondere Sorgfalt.
A11. Fragen aus dem Publikum werden in die Gegenrichtung übersetzt.
A12. Vielen Dank fürs Zuhören, ich freue mich auf Ihre Fragen.
B1. Das Treffen beginnt am 14. März um 9:45 Uhr in Raum 2.17.
B2. Herr Novák aus Bratislava präsentiert die Ergebnisse von 3.250 Messungen.
B3. Das Latenzziel liegt bei 800 Millisekunden, und die Fehlerrate muss unter zehn Prozent bleiben.
B4. Unsere Server laufen mit CUDA 12 auf einer Tesla T4 mit 16 Gigabyte Speicher.
B5. Bitte schicken Sie den Bericht bis Freitag an das CERN-Büro in Genf.
B6. Das Modell hat 600 Millionen Parameter und läuft doppelt so schnell wie in Echtzeit.
C1. Guten Morgen. // Heute möchte ich Ihnen drei Dinge zeigen. / Erstens, / wie das System zuhört. // Zweitens, / wie es übersetzt. // Und drittens, / warum die Pausen wichtig sind. // Wenn ich aufhöre zu sprechen, / sollte die Stimme auch aufhören. // Und wenn ich eine Pause mache, / sollten Sie eine Pause hören.
D1. Können Sie mich hören?
D2. Um wie viel Uhr beginnt die Sitzung?
D3. Könnten Sie das bitte wiederholen?
D4. Ich glaube, wir brauchen eine kurze Pause.
D5. Lassen Sie mich das an einem Beispiel erklären.
