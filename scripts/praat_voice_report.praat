form Voice report
  sentence wav_path _
endform

Read from file: wav_path$
snd = selected("Sound")
pitch = To Pitch: 0, 75, 600
f0mean = Get mean: 0, 0, "Hertz"
removeObject: pitch
selectObject: snd
pp = To PointProcess (periodic, cc): 75, 500
jit = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
selectObject: snd
plusObject: pp
shim = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6
selectObject: snd
harm = To Harmonicity (cc): 0.01, 75, 0.1, 4.5
hnr = Get mean: 0, 0
writeInfoLine: "F0=", fixed$(f0mean, 1), " JIT=", fixed$(jit * 100, 2), "% SHIM=", fixed$(shim * 100, 2), "% HNR=", fixed$(hnr, 1), "dB"
