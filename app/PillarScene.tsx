"use client";

import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type Point=[number,number,number,number];
type DemoData={points:Point[];originalPointCount:number};
export type PillarMetrics={count:number;center:[number,number,number];mean:[number,number,number];selected:Point};

const CELL={x0:-7.5,x1:-6,y0:-9,y1:-7.5,z0:-2.5,z1:3.5};
const FIXED_CENTER:[number,number,number]=[(CELL.x0+CELL.x1)/2,(CELL.y0+CELL.y1)/2,(CELL.z0+CELL.z1)/2];
const PAPER="#f8f6ed";

const VectorLine=({from,to,color,width=1}:{from:[number,number,number];to:[number,number,number];color:string;width?:number})=>{
  const positions=useMemo(()=>new Float32Array([...from,...to]),[from,to]);
  return <line><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry><lineBasicMaterial color={color} linewidth={width}/></line>;
};

function StoryCamera({step}:{step:number}){
  const {camera,gl}=useThree();
  const controls=useRef<OrbitControls|null>(null);
  const transition=useRef(0);
  const destination=useMemo(()=>{
    const look=new THREE.Vector3(...FIXED_CENTER);
    if(step===0)return {position:new THREE.Vector3(31,27,25),look:new THREE.Vector3(0,0,0)};
    if(step===1)return {position:new THREE.Vector3(0,0,46),look:new THREE.Vector3(0,0,0)};
    if(step===2)return {position:look.clone().add(new THREE.Vector3(8,8,7)),look};
    if(step<=5)return {position:look.clone().add(new THREE.Vector3(5.2,5.2,4.2)),look};
    if(step===6)return {position:look.clone().add(new THREE.Vector3(9,8,7)),look};
    if(step===7)return {position:new THREE.Vector3(0,0,46),look:new THREE.Vector3(0,0,-2)};
    if(step===8)return {position:new THREE.Vector3(27,23,24),look:new THREE.Vector3(0,1,5)};
    return {position:new THREE.Vector3(25,21,20),look:new THREE.Vector3(-3,1,0)};
  },[step]);
  useEffect(()=>{
    const orbit=new OrbitControls(camera,gl.domElement);
    orbit.enableDamping=true;orbit.dampingFactor=.065;orbit.enablePan=false;orbit.minDistance=2;orbit.maxDistance=80;
    controls.current=orbit;
    return()=>orbit.dispose();
  },[camera,gl]);
  useEffect(()=>{transition.current=0},[step]);
  useFrame(()=>{
    const orbit=controls.current;
    if(transition.current<.995){
      camera.position.lerp(destination.position,.07);
      orbit?.target.lerp(destination.look,.07);
      transition.current+=.07*(1-transition.current);
    }
    camera.up.set(0,0,1);orbit?.update();
  });
  return null;
}

function MetricGrid({local=false}:{local?:boolean}){
  const positions=useMemo(()=>{
    const p:number[]=[];const extent=local?4.5:24;const step=1.5;
    const cx=local?FIXED_CENTER[0]:0,cy=local?FIXED_CENTER[1]:0;
    for(let d=-extent;d<=extent+.01;d+=step){p.push(cx-extent,cy+d,-2.49,cx+extent,cy+d,-2.49,cx+d,cy-extent,-2.49,cx+d,cy+extent,-2.49)}
    return new Float32Array(p);
  },[local]);
  return <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry><lineBasicMaterial color={local?"#c4d9e3":"#d9d9d5"} transparent opacity={local?.8:.65}/></lineSegments>;
}

function Pillar({step}:{step:number}){
  const group=useRef<THREE.Group>(null);
  useFrame(()=>{if(group.current){group.current.scale.z=THREE.MathUtils.lerp(group.current.scale.z,1,.085);group.current.rotation.z=THREE.MathUtils.lerp(group.current.rotation.z,0,.085)}});
  return <group ref={group} scale={[1,1,.03]} rotation={[0,0,.06]}>
    <mesh position={FIXED_CENTER} renderOrder={2}>
      <boxGeometry args={[CELL.x1-CELL.x0,CELL.y1-CELL.y0,CELL.z1-CELL.z0]}/>
      <meshPhysicalMaterial color="#b9ddeb" transparent opacity={step>=2?.22:.08} roughness={.18} metalness={.02} transmission={.28} thickness={1.4} depthWrite={false} side={THREE.DoubleSide}/>
    </mesh>
    <mesh position={FIXED_CENTER} renderOrder={3}>
      <boxGeometry args={[CELL.x1-CELL.x0,CELL.y1-CELL.y0,CELL.z1-CELL.z0]}/>
      <meshBasicMaterial color="#72a9c2" transparent opacity={.56} wireframe/>
    </mesh>
    <PillarHatching/>
  </group>;
}

function PillarHatching(){
  const lines=useMemo(()=>{
    const p:number[]=[];const slope=1.25;const eps=.012;
    for(let base=CELL.z0-1.8;base<CELL.z1;base+=.22){
      const xa=Math.max(CELL.x0,CELL.x0+(CELL.z0-base)/slope),xb=Math.min(CELL.x1,CELL.x0+(CELL.z1-base)/slope);
      if(xa<xb){const za=base+slope*(xa-CELL.x0),zb=base+slope*(xb-CELL.x0);p.push(xa,CELL.y0-eps,za,xb,CELL.y0-eps,zb,xa,CELL.y1+eps,za,xb,CELL.y1+eps,zb)}
      const ya=Math.max(CELL.y0,CELL.y0+(CELL.z0-base)/slope),yb=Math.min(CELL.y1,CELL.y0+(CELL.z1-base)/slope);
      if(ya<yb){const za=base+slope*(ya-CELL.y0),zb=base+slope*(yb-CELL.y0);p.push(CELL.x0-eps,ya,za,CELL.x0-eps,yb,zb,CELL.x1+eps,ya,za,CELL.x1+eps,yb,zb)}
    }
    return new Float32Array(p);
  },[]);
  return <lineSegments renderOrder={4}><bufferGeometry><bufferAttribute attach="attributes-position" args={[lines,3]}/></bufferGeometry><lineBasicMaterial color="#78abc1" transparent opacity={.46} depthWrite={false}/></lineSegments>;
}

function CenterMarker({position,kind}:{position:[number,number,number];kind:"fixed"|"mean"}){
  const color=kind==="fixed"?"#65a8c6":"#111111";
  return <group position={position}>
    <mesh><sphereGeometry args={[.095,20,20]}/><meshBasicMaterial color={color}/></mesh>
    <mesh rotation={[Math.PI/2,0,0]}><ringGeometry args={[.16,.19,32]}/><meshBasicMaterial color={color} side={THREE.DoubleSide}/></mesh>
    {kind==="fixed"&&<><mesh><boxGeometry args={[.65,.022,.022]}/><meshBasicMaterial color={color}/></mesh><mesh><boxGeometry args={[.022,.65,.022]}/><meshBasicMaterial color={color}/></mesh></>}
  </group>;
}

function AnchorLayer(){
  const anchors=[{p:[-6.7,-8.2,.5] as [number,number,number],s:[1.8,4.2,1.6] as [number,number,number],r:.15},{p:[-3.5,10.5,.5] as [number,number,number],s:[1.8,4.2,1.6] as [number,number,number],r:-.2},{p:[-14.2,1,.4] as [number,number,number],s:[.8,.8,1.7] as [number,number,number],r:0}];
  return <group>{anchors.map((a,i)=><group key={i} position={a.p} rotation={[0,0,a.r]}><mesh><boxGeometry args={a.s}/><meshPhysicalMaterial color="#b9ddeb" transparent opacity={i===0?.24:.12} roughness={.2} transmission={.22} depthWrite={false}/></mesh><mesh><boxGeometry args={a.s}/><meshBasicMaterial color="#5b9ab8" transparent opacity={i===0?.9:.45} wireframe/></mesh></group>)}</group>;
}

function PseudoImageLayer({points}:{points:Point[]}){
  const cells=useMemo(()=>{
    const map=new Map<string,{x:number;y:number;n:number}>();
    points.forEach(([x,y])=>{if(Math.abs(x)>24||Math.abs(y)>24)return;const ix=Math.floor((x+24)/3),iy=Math.floor((y+24)/3),k=`${ix},${iy}`;const c=map.get(k)??{x:-22.5+ix*3,y:-22.5+iy*3,n:0};c.n++;map.set(k,c)});
    return [...map.values()].filter(c=>c.n>1).slice(0,320);
  },[points]);
  return <group>{cells.map((c,i)=><mesh key={i} position={[c.x,c.y,-2.42]}><boxGeometry args={[2.82,2.82,.1]}/><meshBasicMaterial color={c.n>9?"#70abc6":"#d8eaf1"} transparent opacity={Math.min(.8,.18+c.n*.035)}/></mesh>)}</group>;
}

function BackboneLayer(){
  return <group>{[
    {z:4,s:[34,27] as [number,number],c:"#d8eaf1"},{z:7,s:[25,19] as [number,number],c:"#a9cfdf"},{z:10,s:[17,12] as [number,number],c:"#70abc6"}
  ].map((p,i)=><mesh key={i} position={[0,1,p.z]}><planeGeometry args={[p.s[0],p.s[1],12-i*3,9-i*2]}/><meshPhysicalMaterial color={p.c} transparent opacity={.14+i*.05} roughness={.2} transmission={.2} wireframe side={THREE.DoubleSide} depthWrite={false}/></mesh>)}</group>;
}

function MatchingLayer({mode}:{mode:"match"|"loss"|"nms"|"final"}){
  const boxes=[
    {p:[-6.7,-8.2,.5] as [number,number,number],s:[1.8,4.2,1.6] as [number,number,number],r:.15},
    {p:[-6.3,-7.9,.5] as [number,number,number],s:[1.9,4.1,1.6] as [number,number,number],r:.09},
    {p:[-3.5,10.5,.5] as [number,number,number],s:[1.8,4.2,1.6] as [number,number,number],r:-.2},
  ];
  return <group>{boxes.map((a,i)=>{
    const suppressed=mode==="nms"&&i===1;const final=mode==="final"&&i===1;
    if(final)return null;
    return <group key={i} position={a.p} rotation={[0,0,a.r]}><mesh><boxGeometry args={a.s}/><meshBasicMaterial color={suppressed?"#b9b9b5":i===0?"#65a8c6":"#111111"} transparent opacity={suppressed?.18:.75} wireframe/></mesh></group>
  })}{mode==="loss"&&[.6,1.2,2,3.1].map((h,i)=><mesh key={`l${i}`} position={[-1+i*.7,-8,-2.5+h/2]}><boxGeometry args={[.42,.42,h]}/><meshBasicMaterial color={i<2?"#b9ddeb":"#65a8c6"} transparent opacity={.72}/></mesh>)}</group>;
}

function SceneContent({data,step,onMetrics}:{data:DemoData;step:number;onMetrics:(m:PillarMetrics)=>void}){
  const inside=useMemo(()=>data.points.filter(([x,y,z])=>x>=CELL.x0&&x<CELL.x1&&y>=CELL.y0&&y<CELL.y1&&z>=CELL.z0&&z<CELL.z1),[data]);
  const outside=useMemo(()=>data.points.filter(([x,y,z])=>!(x>=CELL.x0&&x<CELL.x1&&y>=CELL.y0&&y<CELL.y1&&z>=CELL.z0&&z<CELL.z1)),[data]);
  const mean=useMemo<[number,number,number]>(()=>[0,1,2].map(k=>inside.reduce((s,p)=>s+p[k],0)/inside.length) as [number,number,number],[inside]);
  const [selectedIndex,setSelectedIndex]=useState(0);
  const selected=inside[selectedIndex]??inside[0];
  useEffect(()=>{if(selected)onMetrics({count:inside.length,center:FIXED_CENTER,mean,selected})},[inside,mean,selected,onMetrics]);
  const outsidePositions=useMemo(()=>new Float32Array(outside.flatMap(p=>p.slice(0,3))),[outside]);
  const insidePositions=useMemo(()=>new Float32Array(inside.flatMap(p=>p.slice(0,3))),[inside]);
  return <>
    <StoryCamera step={step}/>
    <points>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[outsidePositions,3]}/></bufferGeometry>
      <pointsMaterial color="#111111" size={step>=2?.055:.075} transparent opacity={step>=2?.12:.78} sizeAttenuation/>
    </points>
    <points onPointerDown={(e:ThreeEvent<PointerEvent>)=>{e.stopPropagation();if(typeof e.index==="number")setSelectedIndex(e.index)}}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[insidePositions,3]}/></bufferGeometry>
      <pointsMaterial color="#050505" size={step>=2?.15:.085} transparent opacity={1} sizeAttenuation/>
    </points>
    {step===1&&<MetricGrid/>}
    {step>=2&&step<=6&&<><MetricGrid local/><Pillar key={step} step={step}/></>}
    {step>=3&&step<=6&&<CenterMarker position={FIXED_CENTER} kind="fixed"/>}
    {step>=4&&step<=6&&<CenterMarker position={mean} kind="mean"/>}
    {step>=5&&step<=6&&selected&&<><mesh position={[selected[0],selected[1],selected[2]]}><sphereGeometry args={[.12,20,20]}/><meshBasicMaterial color="#050505"/></mesh><VectorLine from={[selected[0],selected[1],selected[2]]} to={FIXED_CENTER} color="#65a8c6"/><VectorLine from={[selected[0],selected[1],selected[2]]} to={mean} color="#111111"/></>}
    {step===6&&Array.from({length:12},(_,i)=><mesh key={i} position={[FIXED_CENTER[0]+(i%4-1.5)*.24,FIXED_CENTER[1]+1.45,FIXED_CENTER[2]+(Math.floor(i/4)-1)*.3]}><boxGeometry args={[.18,.05,.18]}/><meshBasicMaterial color={i%3===0?"#65a8c6":"#111111"} transparent opacity={.74}/></mesh>)}
    {step===7&&<PseudoImageLayer points={data.points}/>} 
    {step===8&&<BackboneLayer/>}
    {step===9&&<AnchorLayer/>}
    {step===10&&<><AnchorLayer/><MatchingLayer mode="match"/></>}
    {step===11&&<MatchingLayer mode="loss"/>}
    {step===12&&<MatchingLayer mode="nms"/>}
    {step===13&&<MatchingLayer mode="final"/>}
  </>;
}

export function PillarScene({step,onMetrics}:{step:number;onMetrics:(m:PillarMetrics)=>void}){
  const [data,setData]=useState<DemoData|null>(null);
  useEffect(()=>{const base=process.env.NODE_ENV==="production"?"/pointpillars-explained":"";fetch(`${base}/data/nuscenes-lidar-demo.json`).then(r=>r.json()).then(setData)},[]);
  return <div className="pp-canvas" aria-label="Interactive PointPillars geometry">
    {data?<Canvas camera={{position:[31,27,25],fov:44,near:.05,far:200}} dpr={[1,1.7]} gl={{antialias:true}}>
      <color attach="background" args={[PAPER]}/><ambientLight intensity={2.2}/><directionalLight position={[8,9,14]} intensity={2.8}/>
      <SceneContent data={data} step={step} onMetrics={onMetrics}/>
    </Canvas>:<div className="pp-loading">Loading the LiDAR frame</div>}
  </div>;
}
