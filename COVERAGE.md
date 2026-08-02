# Visual coverage

Every step transforms the same nuScenes point cloud instead of switching to a detached diagram.

| Step | Core operation | Scene treatment |
|---:|---|---|
| 01 | Raw LiDAR points | black returns on white |
| 02 | XY floor quantization | metric grid over the complete frame |
| 03 | Point-to-pillar grouping | translucent blue cell; assigned points remain black |
| 04 | Fixed pillar center | grid-derived blue cross and exact coordinates |
| 05 | Point-set mean | data-derived black ring and exact mean |
| 06 | Point decoration | selected point with both reference vectors |
| 07 | Shared point encoding and max pooling | per-channel responses beside the physical pillar |
| 08 | Scatter to pseudo-image | occupied BEV cells over the original frame |
| 09 | 2D backbone and fusion | aligned translucent feature planes |
| 10 | Anchor design | physical class-shaped box priors |
| 11 | Ground-truth matching | overlapping target and anchor geometry |
| 12 | Composite training loss | spatial hypotheses plus loss contributions |
| 13 | NMS | retained and faded duplicate boxes |
| 14 | Final predictions | remaining boxes over the original points |

Pure-function tests cover grid boundaries, cluster-offset translation behavior, max-pooling permutation invariance, box coding, IoU, and NMS.
