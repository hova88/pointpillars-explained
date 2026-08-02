# Visual coverage

Every chapter transforms the same nuScenes point cloud instead of switching to a detached diagram.

| Chapter | Core operation | Scene treatment |
|---:|---|---|
| 01 | Raw LiDAR returns | black measurements on white; every point is inspectable |
| 02 | XY quantization and ground truth | metric grid plus official red nuScenes jelly boxes |
| 03 | Complete pillar construction | all 406 occupied cells, one inspectable pillar, fixed center, point mean, decoration, capacity, and `(D,P,N)` stacking |
| 04 | Pillar Feature Net | shared Linear–BN–ReLU and channel-wise max over the point axis |
| 05 | Scatter to BEV | compact pillar row plus paired coordinate copied into an inspectable dense pseudo-image address |
| 06 | 2D backbone and fusion | modular `B₁ → B₂ → B₃`, animated Conv2D flow, transposed convolutions, and 384-channel concatenation |
| 07 | Truck anchor placement | readable truck-prior crop followed by detail focus |
| 08 | Ground-truth matching and loss | positive/ignored/negative assignments, box residuals, classification, localization, and direction loss |
| 09 | CenterHead branch | later anchor-free Gaussian center target and gathered attribute heads |
| 10 | Decode, filter, NMS, and return | local duplicate resolution followed by a camera callback to Chapter 2; red official GT and blue deterministic teaching predictions remain separately inspectable |

Pure-function tests cover grid boundaries, cluster-offset translation behavior, max-pooling permutation invariance, scatter address reconstruction, backbone scale alignment, box encode/decode round trips, IoU, NMS, and CenterHead Gaussian construction.
