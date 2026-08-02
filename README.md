# PointPillars Geometry Lab

A focused, full-screen 3D explanation of PointPillars. One verified nuScenes LiDAR keyframe remains visible while the presentation moves through XY quantization, pillar formation, the fixed cell center, the point-set mean, feature offsets, `(D,P,N)` stacking, shared point encoding, max pooling, BEV scatter, the 2D backbone, anchors, matching, losses, NMS, and a final callback comparing predictions with the ground truth introduced in Chapter 2.

The teaching pillar is calculated from real returns in the embedded frame. Its geometric center is derived from fixed grid bounds; its cluster mean is recomputed from the points assigned to the cell.

The final chapter keeps the official nuScenes annotations visually distinct from deterministic teaching predictions. The blue boxes illustrate decoding, suppression, and localization residuals; they are not outputs from a trained checkpoint and must not be interpreted as benchmark performance.

The companion Chinese long-form article is available at [`articles/a-bitter-history-for-3d-detection-1-pointpillars.zh-CN.md`](articles/a-bitter-history-for-3d-detection-1-pointpillars.zh-CN.md).

## Development

Requires Node 22 and pnpm.

```bash
pnpm install
pnpm dev
pnpm run check
pnpm run build:pages
```

## Data provenance

The browser asset is a deterministic 1-in-6 reduction of the 34,688-return nuScenes tutorial frame. Run `scripts/prepare-nuscenes.mjs` against the original `.pcd.bin` to reproduce it. See [NOTICE.md](NOTICE.md) for citations and licensing.
