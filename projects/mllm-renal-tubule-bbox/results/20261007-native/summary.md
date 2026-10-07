# Run 20261007-native: 37 ground-truth boxes, IoU threshold 0.5 unless stated

F1/P/R are strict (every unmatched prediction is a false positive). 'F1 border-tolerant' ignores unmatched predictions within 10 px of the image border. 'Best IoU' is the mean over ground-truth boxes of the best IoU with any prediction. F1@0.5 shows mean [min-max] over repeats. 'F1 any convention' is a diagnostic, not the main score: per call, the best F1@0.5 over six coordinate conventions (x/y order, 0-1000 / pixels / 0-1), i.e. what the model would score if its own format were accepted. A large gap to F1@0.5 means the model located tubules but did not follow the requested coordinate format.

| Model | Runs | Pred | F1@0.5 | P@0.5 | R@0.5 | F1@0.75 | F1@0.3 | Best IoU | F1 border-tolerant | F1 any convention | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash (native format) | 3/3 | 35.3 | 0.764 [0.735-0.789] | 0.784 | 0.748 | 0.494 | 0.792 | 0.606 | 0.786 | 0.764 [0.735-0.789] | - |
| Kimi K3 (native format) | 3/3 | 28.0 | 0.526 [0.067-0.769] | 0.589 | 0.477 | 0.445 | 0.602 | 0.442 | 0.526 | 0.549 [0.136-0.769] | - |

## Speed and price

Measured on this run's calls (all answered calls, including unusable answers). 'Latency' is the request that produced the answer, without retries, median [min-max] s; calls ran up to 4 at a time, different models in parallel. 'Out tok/s' is completion tokens (reasoning included) divided by that latency, so it is end-to-end speed, not decode speed. 'TTFT' is OpenRouter's own time-to-first-token record (median); for reasoning models it can include thinking time before the first token. 'List $/M' is the model's listed input/output price; '$/call' is what OpenRouter billed. '$ per match' is mean $/call divided by mean boxes matched at IoU 0.5.

| Model | List $/M in/out | Provider | Out tok (reasoning) | Latency s | TTFT s | Out tok/s | $/call | Total $ | $ per match | F1@0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash (native format) | 0.75 / 3.75 | Google | 2417 (1147) | 27.2 [18.9-28.4] | 7.5 (n=2/3) | 90 | $0.0101 | $0.0303 | $0.00036 | 0.764 |
| Kimi K3 (native format) | 0.62 / 15 | Amazon Bedrock, Modal, Morph | 5236 (4272) | 81.1 [53.4-93.9] | - | 81 | $0.0781 | $0.234 | $0.0044 | 0.526 |

- Not beaten on F1 vs $/call (no other model is at least as accurate and cheaper): Gemini 3.8 Flash (native format).
- Not beaten on F1 vs median latency (no other model is at least as accurate and faster): Gemini 3.8 Flash (native format).
