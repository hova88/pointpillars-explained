import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { boxIou2d, decoratePoint, encodeBox, maxPool, nms, pillarIndex } from "../lib/algorithm.mjs";

test("pillar indexing uses half-open boundaries",()=>{
  assert.deepEqual(pillarIndex([0,0,0,0],[-1,-1,1,1],.5),[2,2]);
  assert.equal(pillarIndex([1,0,0,0],[-1,-1,1,1],.5),null);
});

test("cluster offsets are translation invariant",()=>{
  const a=[[1,2,0,1],[2,4,1,1]];
  const b=a.map(p=>[p[0]+10,p[1]-7,p[2]+3,p[3]]);
  const da=decoratePoint(a[0],a,[1.5,2.5]);
  const db=decoratePoint(b[0],b,[11.5,-4.5]);
  assert.deepEqual(da.slice(4),db.slice(4));
  assert.notDeepEqual(da.slice(0,3),db.slice(0,3));
});

test("teaching pillar keeps the fixed cell center distinct from its point mean",()=>{
  const demo=JSON.parse(fs.readFileSync(new URL("../public/data/nuscenes-lidar-demo.json",import.meta.url),"utf8"));
  const bounds={x0:-7.5,x1:-6,y0:-9,y1:-7.5,z0:-2.5,z1:3.5};
  const center=[(bounds.x0+bounds.x1)/2,(bounds.y0+bounds.y1)/2,(bounds.z0+bounds.z1)/2];
  const points=demo.points.filter(([x,y,z])=>x>=bounds.x0&&x<bounds.x1&&y>=bounds.y0&&y<bounds.y1&&z>=bounds.z0&&z<bounds.z1);
  const mean=[0,1,2].map(k=>points.reduce((sum,p)=>sum+p[k],0)/points.length);
  assert.equal(points.length,38);
  assert.deepEqual(center,[-6.75,-8.25,.5]);
  assert.ok(Math.abs(mean[0]+6.8042)<1e-3);
  assert.ok(Math.abs(mean[1]+8.1674)<1e-3);
  assert.notDeepEqual(mean,center);
});

test("max pooling ignores point order",()=>{
  const x=[[1,7,2],[4,2,3],[0,5,9]];
  assert.deepEqual(maxPool(x),maxPool([...x].reverse()));
});

test("box coding is zero for a matching anchor",()=>{
  const box={x:2,y:3,z:1,w:2,l:4,h:1.5,yaw:.2,score:1};
  assert.deepEqual(encodeBox(box,box),[0,0,0,0,0,0,0]);
});

test("IoU and NMS remove lower-scored duplicate",()=>{
  const a={x:0,y:0,z:0,w:2,l:4,h:1,yaw:0,score:.9};
  const b={...a,x:.1,score:.7};
  assert.ok(boxIou2d(a,b)>.8);
  assert.deepEqual(nms([b,a],.5),[a]);
});
