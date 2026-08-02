import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { boxIou2d, decoratePoint, encodeBox, gaussianRadius, gaussianValue, maxPool, nms, pillarIndex, rotatedBoxIou2d, scatterPillarFeatures } from "../lib/algorithm.mjs";

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

test("the displayed nuScenes frame produces the real pillar population",()=>{
  const demo=JSON.parse(fs.readFileSync(new URL("../public/data/nuscenes-lidar-demo.json",import.meta.url),"utf8"));
  const cells=new Map();
  for(const [x,y,z] of demo.points){
    if(x< -24||x>=24||y< -24||y>=24||z< -2.5||z>=3.5)continue;
    const key=`${Math.floor((x+24)/1.5)},${Math.floor((y+24)/1.5)}`;
    cells.set(key,(cells.get(key)??0)+1);
  }
  assert.equal(cells.size,406);
  assert.equal(Math.max(...cells.values()),927);
  assert.equal([...cells.values()].filter(count=>count>100).length,5);
});

test("max pooling ignores point order",()=>{
  const x=[[1,7,2],[4,2,3],[0,5,9]];
  assert.deepEqual(maxPool(x),maxPool([...x].reverse()));
});

test("scatter uses each compact pillar row's paired BEV address",()=>{
  const features=[[1,2],[3,4]];
  const coordinates=[[0,1],[2,0]];
  const image=scatterPillarFeatures(features,coordinates,3,3);
  assert.deepEqual([image[0][0][1],image[1][0][1]],[1,2]);
  assert.deepEqual([image[0][2][0],image[1][2][0]],[3,4]);
  assert.deepEqual(image[0][1],[0,0,0]);
  assert.deepEqual(
    scatterPillarFeatures([...features].reverse(),[...coordinates].reverse(),3,3),
    image,
  );
});

test("the three PointPillars backbone scales align before concatenation",()=>{
  const topDown=[[16,16],[8,8],[4,4]];
  const upsample=[1,2,4];
  assert.deepEqual(topDown.map((shape,i)=>shape.map(value=>value*upsample[i])),[[16,16],[16,16],[16,16]]);
  assert.equal(128*3,384);
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

test("rotated IoU respects box orientation",()=>{
  const a={x:0,y:0,z:0,w:2,l:8,h:2,yaw:0,score:1};
  assert.ok(Math.abs(rotatedBoxIou2d(a,{...a})-1)<1e-9);
  assert.ok(rotatedBoxIou2d(a,{...a,yaw:Math.PI/2})<.15);
  assert.ok(rotatedBoxIou2d(a,{...a,x:1})>.7);
});

test("CenterHead Gaussian peaks at the object center",()=>{
  const radius=Math.floor(gaussianRadius([10.201/.4,2.877/.4],.1));
  const sigma=(2*radius+1)/6;
  assert.equal(radius,13);
  assert.equal(gaussianValue(0,0,sigma),1);
  assert.equal(gaussianValue(3,-2,sigma),gaussianValue(-3,2,sigma));
  assert.ok(gaussianValue(radius,0,sigma)<.02);
});

test("the teaching truck crop preserves all three anchor assignment states",()=>{
  const gt={x:-4.499,y:15.253,z:.396,w:2.877,l:10.201,h:3.595,yaw:1.5952,score:1};
  const anchors=[];
  for(const x of [-10.5,-7.5,-4.5,-1.5,1.5])for(const y of [9,12,15,18,21])for(const yaw of [0,Math.PI/2]){
    const anchor={x,y,z:.3,w:2.8,l:9.6,h:3.4,yaw,score:1};
    anchors.push({anchor,iou:rotatedBoxIou2d(anchor,gt)});
  }
  assert.equal(anchors.length,50);
  assert.equal(anchors.filter(item=>item.iou>=.55).length,1);
  assert.equal(anchors.filter(item=>item.iou>=.4&&item.iou<.55).length,2);
  const best=anchors.sort((a,b)=>b.iou-a.iou)[0];
  assert.ok(Math.abs(best.iou-.8995313)<1e-6);
  assert.deepEqual(encodeBox(gt,best.anchor).map(value=>Number(value.toFixed(4))),[.0001,.0253,.0282,.0271,.0607,.0558,.0244]);
});
