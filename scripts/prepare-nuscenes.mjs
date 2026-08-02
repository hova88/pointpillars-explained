import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error("Usage: node prepare-nuscenes.mjs input.bin output.json");

const buffer = readFileSync(input);
const floats = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
const stride = 5;
const total = floats.length / stride;
const target = 5200;
const step = Math.max(1, Math.floor(total / target));
const points = [];

for (let i = 0; i < total; i += step) {
  const j = i * stride;
  const x = floats[j];
  const y = floats[j + 1];
  const z = floats[j + 2];
  const intensity = floats[j + 3] / 255;
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
    points.push([
      Math.round(x * 100) / 100,
      Math.round(y * 100) / 100,
      Math.round(z * 100) / 100,
      Math.round(intensity * 1000) / 1000,
    ]);
  }
}

const payload = {
  source: "nuScenes v1.0-mini / MMDetection3D demo mirror",
  sampleToken: "ca9a282c9e77460f8360f564131a8af5",
  sampleDataToken: "9d9bf11fb0e144c8b446d54a8a00184f",
  filename: "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
  originalPointCount: total,
  sampling: `deterministic every ${step}th point`,
  fields: ["x", "y", "z", "intensity"],
  points,
};

mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(payload)}\n`);
console.log(`Wrote ${points.length} of ${total} points to ${output}`);
