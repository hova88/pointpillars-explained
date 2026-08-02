# PointPillars Geometry Lab

A focused, full-screen 3D explanation of PointPillars. One verified nuScenes LiDAR keyframe remains visible while the presentation moves through XY quantization, pillar formation, the fixed cell center, the point-set mean, feature offsets, pillar encoding, BEV scatter, the 2D backbone, anchors, matching, losses, NMS, and final detections.

The teaching pillar is calculated from real returns in the embedded frame. Its geometric center is derived from fixed grid bounds; its cluster mean is recomputed from the points assigned to the cell.

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
