# Run 20261007-main: 37 ground-truth boxes, IoU threshold 0.5 unless stated

F1/P/R are strict (every unmatched prediction is a false positive). 'F1 border-tolerant' ignores unmatched predictions within 10 px of the image border. 'Best IoU' is the mean over ground-truth boxes of the best IoU with any prediction. F1@0.5 shows mean [min-max] over repeats. 'F1 any convention' is a diagnostic, not the main score: per call, the best F1@0.5 over six coordinate conventions (x/y order, 0-1000 / pixels / 0-1), i.e. what the model would score if its own format were accepted. A large gap to F1@0.5 means the model located tubules but did not follow the requested coordinate format.

| Model | Runs | Pred | F1@0.5 | P@0.5 | R@0.5 | F1@0.75 | F1@0.3 | Best IoU | F1 border-tolerant | F1 any convention | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 Flash | 3/3 | 39.3 | 0.796 [0.750-0.838] | 0.775 | 0.820 | 0.499 | 0.813 | 0.656 | 0.813 | 0.796 [0.750-0.838] | - |
| GPT-6.1 Sol | 3/3 | 53.7 | 0.735 [0.719-0.761] | 0.621 | 0.901 | 0.419 | 0.787 | 0.697 | 0.749 | 0.735 [0.719-0.761] | - |
| Gemini 3.8 Flash | 3/3 | 35.3 | 0.534 [0.111-0.761] | 0.546 | 0.523 | 0.369 | 0.645 | 0.477 | 0.551 | 0.747 [0.730-0.761] | convention? yxyx_norm1000 |
| Qwen3.8 Max (0902) | 3/3 | 35.7 | 0.370 [0.147-0.795] | 0.363 | 0.378 | 0.079 | 0.519 | 0.393 | 0.373 | 0.370 [0.147-0.795] | - |
| GPT-6 Luna | 3/3 | 50.7 | 0.260 [0.215-0.296] | 0.227 | 0.306 | 0.015 | 0.450 | 0.354 | 0.264 | 0.260 [0.215-0.296] | - |
| GLM 5.3 Flash | 3/3 | 21.0 | 0.232 [0.000-0.394] | 0.252 | 0.216 | 0.038 | 0.389 | 0.224 | 0.232 | 0.232 [0.000-0.394] | parse:none; truncated |
| Kimi K3 | 3/3 | 35.0 | 0.228 [0.149-0.312] | 0.232 | 0.225 | 0.062 | 0.441 | 0.291 | 0.230 | 0.635 [0.595-0.704] | convention? xyxy_px |
| DeepSeek V4.1 Flash | 3/3 | 52.7 | 0.132 [0.071-0.174] | 0.111 | 0.162 | 0.014 | 0.364 | 0.314 | 0.135 | 0.178 [0.151-0.217] | - |

## Speed and price

Measured on this run's calls (all answered calls, including unusable answers). 'Latency' is the request that produced the answer, without retries, median [min-max] s; calls ran up to 4 at a time, different models in parallel. 'Out tok/s' is completion tokens (reasoning included) divided by that latency, so it is end-to-end speed, not decode speed. 'TTFT' is OpenRouter's own time-to-first-token record (median); for reasoning models it can include thinking time before the first token. 'List $/M' is the model's listed input/output price; '$/call' is what OpenRouter billed. '$ per match' is mean $/call divided by mean boxes matched at IoU 0.5.

| Model | List $/M in/out | Provider | Out tok (reasoning) | Latency s | TTFT s | Out tok/s | $/call | Total $ | $ per match | F1@0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 Flash | 0.15 / 0.47 | Alibaba | 4316 (2987) | 44.4 [30.5-58.7] | 4.1 | 95 | $0.0021 | $0.0063 | $0.00007 | 0.796 |
| GPT-6.1 Sol | 2 / 10 | OpenAI | 2532 (1451) | 55.5 [53.8-55.8] | 11.1 | 47 | $0.0264 | $0.0793 | $0.00079 | 0.735 |
| Gemini 3.8 Flash | 0.75 / 3.75 | Google | 2904 (1633) | 19.5 [17.2-31.1] | 5.4 | 120 | $0.0119 | $0.0358 | $0.00062 | 0.534 |
| Qwen3.8 Max (0902) | 2 / 6 | Alibaba | 10664 (9384) | 172.6 [156.3-192.4] | 3.3 | 61 | $0.0650 | $0.195 | $0.0046 | 0.370 |
| GPT-6 Luna | 0.1 / 0.5 | OpenAI | 5295 (4273) | 44.3 [42.0-52.4] | 16.3 | 109 | $0.0027 | $0.0080 | $0.00023 | 0.260 |
| GLM 5.3 Flash | 0.15 / 0.5 | GMICloud, Modal, Wafer | 15273 (11038) | 236.4 [26.3-272.2] | 0.9 (n=1/3) | 53 | $0.0070 | $0.0209 | $0.00087 | 0.232 |
| Kimi K3 | 0.62 / 15 | Amazon Bedrock, Modal | 8278 (6914) | 79.8 [74.0-141.5] | 2.9 | 88 | $0.128 | $0.383 | $0.0153 | 0.228 |
| DeepSeek V4.1 Flash | 0.05 / 1.2 | AtlasCloud, Parasail, Together | 12368 (10943) | 35.8 [26.1-96.8] | 1.0 | 283 | $0.0114 | $0.0341 | $0.0019 | 0.132 |

- Not beaten on F1 vs $/call (no other model is at least as accurate and cheaper): Qwen3.8 Flash.
- Not beaten on F1 vs median latency (no other model is at least as accurate and faster): Qwen3.8 Flash, Gemini 3.8 Flash.
