export function pillarIndex(point, range, size) {
  const [x, y] = point;
  const [xmin, ymin, xmax, ymax] = range;
  if (x < xmin || x >= xmax || y < ymin || y >= ymax) return null;
  return [Math.floor((x - xmin) / size), Math.floor((y - ymin) / size)];
}

export function decoratePoint(point, points, pillarCenter) {
  const mean = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]], [0, 0, 0])
    .map((v) => v / points.length);
  const [x, y, z, r = 0] = point;
  return [x, y, z, r, x - mean[0], y - mean[1], z - mean[2], x - pillarCenter[0], y - pillarCenter[1]];
}

export function maxPool(channels) {
  if (!channels.length) return [];
  return channels[0].map((_, c) => Math.max(...channels.map((row) => row[c])));
}

export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function focalLoss(probability, target, alpha = 0.25, gamma = 2) {
  const p = Math.min(1 - 1e-7, Math.max(1e-7, probability));
  const pt = target ? p : 1 - p;
  const at = target ? alpha : 1 - alpha;
  return -at * Math.pow(1 - pt, gamma) * Math.log(pt);
}

export function smoothL1(value, beta = 1 / 9) {
  const a = Math.abs(value);
  return a < beta ? (0.5 * a * a) / beta : a - 0.5 * beta;
}

export function boxIou2d(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w / 2, b.x + b.w / 2) - Math.max(a.x - a.w / 2, b.x - b.w / 2));
  const iy = Math.max(0, Math.min(a.y + a.l / 2, b.y + b.l / 2) - Math.max(a.y - a.l / 2, b.y - b.l / 2));
  const intersection = ix * iy;
  return intersection / (a.w * a.l + b.w * b.l - intersection || 1);
}

export function nms(boxes, threshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  while (sorted.length) {
    const current = sorted.shift();
    kept.push(current);
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (boxIou2d(current, sorted[i]) > threshold) sorted.splice(i, 1);
    }
  }
  return kept;
}

export function encodeBox(box, anchor) {
  const diagonal = Math.hypot(anchor.w, anchor.l);
  return [
    (box.x - anchor.x) / diagonal,
    (box.y - anchor.y) / diagonal,
    (box.z - anchor.z) / anchor.h,
    Math.log(box.w / anchor.w),
    Math.log(box.l / anchor.l),
    Math.log(box.h / anchor.h),
    box.yaw - anchor.yaw,
  ];
}
