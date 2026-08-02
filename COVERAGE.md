# PointPillars coverage matrix

This matrix is a release gate: every algorithm block has a matching interactive chapter and a nearby source-class badge. Each chapter carries Observe, Operate, Inspect, Why, Alternative, Stress-test, and Information-survival fields in `app/curriculum.ts`.

| Source topic | Chapter | Interactive evidence |
|---|---:|---|
| LiDAR return and unordered sets | 01 | persistent source sweep, point colouring, camera frames |
| nuScenes frame and transforms | 02 | verified tutorial keyframe, view switching, provenance |
| Detection bounds | 03 | live range, half-open boundary box |
| XY pillar quantization | 04 | metric grid and adjustable resolution |
| Sparse point grouping | 05 | occupied pillar overlay |
| Pillar/point capacity | 06 | capacity control, truncation/padding reasoning |
| Point decoration | 07 | centroid and cell-centre references, 9D caveat |
| Stacked tensor | 08 | D/P/N glossary and shape ledger |
| Linear + BN + ReLU | 09 | learned point-response overlay |
| Symmetric max pooling | 10 | feature winners and pooling alternative control |
| Scatter | 11 | same-frame BEV feature tiles |
| Top-down backbone | 12 | feature planes and stride ledger |
| Upsampling and concatenation | 13 | aligned multi-scale planes |
| SSD head | 14 | scene-registered candidate boxes |
| Anchor templates | 15 | oriented physical priors |
| IoU matching | 16 | assignment reasoning and thresholds |
| Box residual coding | 17 | normalized equation and anchor/target geometry |
| Direction bins | 18 | orientation-aware boxes and alternative encoding |
| Focal/Smooth-L1/direction losses | 19 | loss contributions and live focal parameters |
| Database sampling and transforms | 20 | ghosted transformed objects |
| Forward/backward training | 21 | directional layer flow |
| Sigmoid, threshold, decoding | 22 | live score filter and candidates |
| NMS | 23 | retained/suppressed boxes and IoU control |
| Output and evaluation | 24 | final scene boxes and protocol distinction |
| Speed and information tradeoffs | 25 | full-frame compression ledger |

Pure-function checks cover quantization boundaries, translation behaviour, pooling permutation invariance, box coding, IoU, and NMS in `tests/algorithm.test.mjs`.
