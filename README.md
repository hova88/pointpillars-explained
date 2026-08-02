# Interactive PointPillars

A complete, animation-driven explanation of PointPillars. The site follows one verified nuScenes LiDAR keyframe from raw returns through pillarization, learned point features, pseudo-image construction, a 2D backbone, anchor-based training, box decoding, and NMS.

Every chapter answers what changes, how it is computed, why the choice exists, why a simpler alternative is insufficient, and what information is gained or lost.

## Development

Requires Node 22 and pnpm.

```bash
pnpm install
pnpm dev
```

Validation and static GitHub Pages build:

```bash
pnpm run check
pnpm run build:pages
```

## Data provenance

The browser asset is a deterministic 1-in-6 reduction of the 34,688-return nuScenes tutorial frame. Run `scripts/prepare-nuscenes.mjs` against the original `.pcd.bin` to reproduce it. See [NOTICE.md](NOTICE.md) for citations and licensing.

## Source labels

The interface distinguishes original-paper content, earlier work reused by PointPillars, common implementation conventions, nuScenes-specific context, and teaching-only models. Interactive feature removal demonstrates representational consequences; it does not claim retrained accuracy.
