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

function rotatedCorners(box) {
  const c=Math.cos(box.yaw),s=Math.sin(box.yaw),hx=box.l/2,hy=box.w/2;
  return [[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]].map(([x,y])=>[box.x+x*c-y*s,box.y+x*s+y*c]);
}

function cross(a,b,c) {
  return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
}

function lineIntersection(start,end,a,b) {
  const rx=end[0]-start[0],ry=end[1]-start[1],sx=b[0]-a[0],sy=b[1]-a[1];
  const denominator=rx*sy-ry*sx;
  if(Math.abs(denominator)<1e-9)return end;
  const t=((a[0]-start[0])*sy-(a[1]-start[1])*sx)/denominator;
  return [start[0]+t*rx,start[1]+t*ry];
}

function clipPolygon(subject,clip) {
  let output=subject;
  for(let i=0;i<clip.length;i+=1){
    const a=clip[i],b=clip[(i+1)%clip.length],input=output;output=[];
    if(!input.length)break;
    let start=input[input.length-1];
    for(const end of input){
      const endInside=cross(a,b,end)>=-1e-9,startInside=cross(a,b,start)>=-1e-9;
      if(endInside){if(!startInside)output.push(lineIntersection(start,end,a,b));output.push(end)}
      else if(startInside)output.push(lineIntersection(start,end,a,b));
      start=end;
    }
  }
  return output;
}

function polygonArea(points) {
  if(points.length<3)return 0;
  let area=0;
  for(let i=0;i<points.length;i+=1){const next=points[(i+1)%points.length];area+=points[i][0]*next[1]-next[0]*points[i][1]}
  return Math.abs(area)/2;
}

export function rotatedBoxIou2d(a,b) {
  const intersection=polygonArea(clipPolygon(rotatedCorners(a),rotatedCorners(b)));
  return intersection/(a.w*a.l+b.w*b.l-intersection||1);
}

export function gaussianRadius([height,width],minOverlap=.5) {
  const a1=1,b1=height+width,c1=width*height*(1-minOverlap)/(1+minOverlap);
  const r1=(b1+Math.sqrt(b1*b1-4*a1*c1))/(2*a1);
  const a2=4,b2=2*(height+width),c2=(1-minOverlap)*width*height;
  const r2=(b2+Math.sqrt(b2*b2-4*a2*c2))/(2*a2);
  const a3=4*minOverlap,b3=-2*minOverlap*(height+width),c3=(minOverlap-1)*width*height;
  const r3=(b3+Math.sqrt(b3*b3-4*a3*c3))/(2*a3);
  return Math.min(r1,r2,r3);
}

export function gaussianValue(dx,dy,sigma) {
  return Math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma));
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
