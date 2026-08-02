import test from "node:test";
import assert from "node:assert/strict";
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
