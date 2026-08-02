# Data and research notices

This is an independent, non-commercial educational project. It is not affiliated with the PointPillars authors, Motional, nuScenes, or OpenMMLab.

The reduced point-cloud teaching asset in `public/data/nuscenes-lidar-demo.json` is derived from nuScenes sample `ca9a282c9e77460f8360f564131a8af5`, sample-data token `9d9bf11fb0e144c8b446d54a8a00184f`. The original binary is mirrored in the MMDetection3D demonstration assets. `public/data/nuscenes-boxes.json` contains all 69 official annotations attached to the same sample, transformed from global coordinates into its native `LIDAR_TOP` frame (`+x` right, `+y` forward, `+z` up). The red ego envelope is explicitly teaching geometry, not a nuScenes ground-truth annotation. nuScenes-derived data is used under the nuScenes Dataset Terms and CC BY-NC-SA 4.0 for non-commercial education. See https://www.nuscenes.org/terms-of-use.

Primary research citation:

> Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. “PointPillars: Fast Encoders for Object Detection from Point Clouds.” CVPR 2019.

Dataset citation:

> Holger Caesar et al. “nuScenes: A multimodal dataset for autonomous driving.” CVPR 2020.
