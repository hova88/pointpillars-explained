"use client";

import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type Point=[number,number,number,number];
type PointRecord={point:Point;index:number};
type PillarCell={key:string;ix:number;iy:number;x0:number;x1:number;y0:number;y1:number;center:[number,number,number];mean:[number,number,number];records:PointRecord[]};
type DemoData={points:Point[];originalPointCount:number};
type BoxAnnotation={category:string;center:[number,number,number];dimensions:[number,number,number];yaw:number;numLidarPoints:number;numRadarPoints:number;token:string};
type BoxData={boxes:BoxAnnotation[]};
export type PillarMetrics={count:number;center:[number,number,number];mean:[number,number,number];selected:Point;cellIndex:[number,number];nonEmptyPillars:number;maxPoints:number};
export type SceneSelection=
  |{kind:"point";index:number;point:Point;range:number;cell:[number,number];insideTeachingPillar:boolean}
  |{kind:"box";source:"nuScenes ground truth"|"teaching geometry";category:string;center:[number,number,number];dimensions:[number,number,number];yaw:number;numLidarPoints:number;numRadarPoints:number;token:string}
  |{kind:"cell";stage:"quantization grid"|"pseudo-image";index:[number,number];center:[number,number];bounds:[number,number,number,number];size:number;pointCount:number};

const GRID_EXTENT=24,PILLAR_SIZE=1.5,Z_MIN=-2.5,Z_MAX=3.5,DEFAULT_PILLAR_KEY="11,10";
const DEFAULT_CENTER:[number,number,number]=[-6.75,-8.25,.5];
const PAPER="#f8f6ed";
const EGO_BOX:BoxAnnotation={category:"ego vehicle envelope",center:[0,0,-1],dimensions:[4.6,1.95,1.5],yaw:Math.PI/2,numLidarPoints:0,numRadarPoints:0,token:"ego-teaching-envelope"};

const VectorLine=({from,to,color,width=1}:{from:[number,number,number];to:[number,number,number];color:string;width?:number})=>{
  const positions=useMemo(()=>new Float32Array([...from,...to]),[from,to]);
  return <line><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry><lineBasicMaterial color={color} linewidth={width}/></line>;
};

function StoryCamera({step,focus}:{step:number;focus:[number,number,number]}){
  const {camera,gl}=useThree();
  const controls=useRef<OrbitControls|null>(null);
  const transition=useRef(0);
  const destination=useMemo(()=>{
    const look=new THREE.Vector3(...focus);
    if(step===0)return {position:new THREE.Vector3(31,27,25),look:new THREE.Vector3(0,0,0)};
    if(step===1)return {position:new THREE.Vector3(0,8,72),look:new THREE.Vector3(0,8,0)};
    if(step===2)return {position:new THREE.Vector3(34,32,38),look:new THREE.Vector3(0,4,0)};
    if(step===3)return {position:look.clone().add(new THREE.Vector3(8.5,8.5,7.5)),look};
    if(step===4)return {position:new THREE.Vector3(0,0,46),look:new THREE.Vector3(0,0,-2)};
    if(step===5)return {position:new THREE.Vector3(27,23,24),look:new THREE.Vector3(0,1,5)};
    return {position:new THREE.Vector3(25,21,20),look:new THREE.Vector3(-3,1,0)};
  },[step,focus]);
  useEffect(()=>{
    const orbit=new OrbitControls(camera,gl.domElement);
    orbit.enableDamping=true;orbit.dampingFactor=.065;orbit.enablePan=false;orbit.minDistance=2;orbit.maxDistance=190;
    controls.current=orbit;
    return()=>orbit.dispose();
  },[camera,gl]);
  useEffect(()=>{transition.current=0},[step,focus]);
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

function MetricGrid({local=false,center=DEFAULT_CENTER}:{local?:boolean;center?:[number,number,number]}){
  const positions=useMemo(()=>{
    const p:number[]=[];const extent=local?4.5:24;const step=1.5;
    const cx=local?center[0]:0,cy=local?center[1]:0;
    for(let d=-extent;d<=extent+.01;d+=step){p.push(cx-extent,cy+d,-2.49,cx+extent,cy+d,-2.49,cx+d,cy-extent,-2.49,cx+d,cy+extent,-2.49)}
    return new Float32Array(p);
  },[local,center]);
  return <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry><lineBasicMaterial color={local?"#c4d9e3":"#d9d9d5"} transparent opacity={local?.8:.65}/></lineSegments>;
}

function SelectedPillar({cell}:{cell:PillarCell}){
  return <group>
    <mesh position={cell.center} renderOrder={2}>
      <boxGeometry args={[PILLAR_SIZE,PILLAR_SIZE,Z_MAX-Z_MIN]}/>
      <meshPhysicalMaterial color="#b9ddeb" transparent opacity={.24} roughness={.18} metalness={.02} transmission={.3} thickness={1.4} clearcoat={1} clearcoatRoughness={.2} depthWrite={false} side={THREE.DoubleSide}/>
    </mesh>
    <mesh position={cell.center} renderOrder={3}>
      <boxGeometry args={[PILLAR_SIZE,PILLAR_SIZE,Z_MAX-Z_MIN]}/>
      <meshBasicMaterial color="#72a9c2" transparent opacity={.56} wireframe/>
    </mesh>
    <PillarHatching cell={cell}/>
  </group>;
}

function PillarHatching({cell}:{cell:PillarCell}){
  const lines=useMemo(()=>{
    const p:number[]=[];const slope=1.25;const eps=.012;
    for(let base=Z_MIN-1.8;base<Z_MAX;base+=.22){
      const xa=Math.max(cell.x0,cell.x0+(Z_MIN-base)/slope),xb=Math.min(cell.x1,cell.x0+(Z_MAX-base)/slope);
      if(xa<xb){const za=base+slope*(xa-cell.x0),zb=base+slope*(xb-cell.x0);p.push(xa,cell.y0-eps,za,xb,cell.y0-eps,zb,xa,cell.y1+eps,za,xb,cell.y1+eps,zb)}
      const ya=Math.max(cell.y0,cell.y0+(Z_MIN-base)/slope),yb=Math.min(cell.y1,cell.y0+(Z_MAX-base)/slope);
      if(ya<yb){const za=base+slope*(ya-cell.y0),zb=base+slope*(yb-cell.y0);p.push(cell.x0-eps,ya,za,cell.x0-eps,yb,zb,cell.x1+eps,ya,za,cell.x1+eps,yb,zb)}
    }
    return new Float32Array(p);
  },[cell]);
  return <lineSegments renderOrder={4}><bufferGeometry><bufferAttribute attach="attributes-position" args={[lines,3]}/></bufferGeometry><lineBasicMaterial color="#78abc1" transparent opacity={.46} depthWrite={false}/></lineSegments>;
}

function ManyPillarsLayer({cells,selectedKey,muted=false,onSelect}:{cells:PillarCell[];selectedKey:string;muted?:boolean;onSelect:(key:string)=>void}){
  const mesh=useRef<THREE.InstancedMesh>(null);
  const progress=useRef(.02);
  const dummy=useMemo(()=>new THREE.Object3D(),[]);
  useEffect(()=>{progress.current=.02},[cells]);
  useEffect(()=>{
    const target=mesh.current;if(!target)return;
    cells.forEach((cell,index)=>{const density=Math.min(1,Math.log2(cell.records.length+1)/6);target.setColorAt(index,cell.key===selectedKey?new THREE.Color("#65a8c6"):new THREE.Color().setHSL(.55,.38,.78-density*.17))});
    if(target.instanceColor)target.instanceColor.needsUpdate=true;
  },[cells,selectedKey]);
  useFrame(()=>{
    const target=mesh.current;if(!target)return;
    if(progress.current>.999)return;
    progress.current=THREE.MathUtils.lerp(progress.current,1,.085);
    cells.forEach((cell,index)=>{
      dummy.position.set(cell.center[0],cell.center[1],Z_MIN+(Z_MAX-Z_MIN)*progress.current/2);
      dummy.scale.set(PILLAR_SIZE*.91,PILLAR_SIZE*.91,(Z_MAX-Z_MIN)*progress.current);
      dummy.updateMatrix();target.setMatrixAt(index,dummy.matrix);
    });
    target.instanceMatrix.needsUpdate=true;
  });
  return <instancedMesh ref={mesh} args={[undefined,undefined,cells.length]} onPointerDown={e=>{if(typeof e.instanceId==="number")onSelect(cells[e.instanceId].key)}} onPointerOver={e=>{e.stopPropagation();document.body.style.cursor="pointer"}} onPointerOut={()=>{document.body.style.cursor="default"}}>
    <boxGeometry args={[1,1,1]}/><meshPhysicalMaterial vertexColors color="#b9ddeb" transparent opacity={muted?.045:.105} transmission={.24} thickness={.7} roughness={.22} clearcoat={.75} clearcoatRoughness={.24} depthWrite={false} side={THREE.DoubleSide}/>
  </instancedMesh>;
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

function JellyBox({box,ego=false,selected,onSelect}:{box:BoxAnnotation;ego?:boolean;selected:boolean;onSelect:(selection:SceneSelection)=>void}){
  const {center,dimensions,yaw}=box;
  const group=useRef<THREE.Group>(null);
  const geometry=useMemo(()=>{
    const solid=new THREE.BoxGeometry(...dimensions);
    const edges=new THREE.EdgesGeometry(solid);
    solid.dispose();
    return edges;
  },[dimensions]);
  useEffect(()=>()=>geometry.dispose(),[geometry]);
  useFrame(()=>{if(group.current){group.current.scale.x=THREE.MathUtils.lerp(group.current.scale.x,1,.1);group.current.scale.y=THREE.MathUtils.lerp(group.current.scale.y,1,.1);group.current.scale.z=THREE.MathUtils.lerp(group.current.scale.z,1,.12)}});
  const choose=(e:ThreeEvent<PointerEvent>)=>{e.stopPropagation();onSelect({kind:"box",source:ego?"teaching geometry":"nuScenes ground truth",...box})};
  return <group ref={group} position={center} rotation={[0,0,yaw]} scale={[.84,.84,.06]} onPointerDown={choose} onPointerOver={e=>{e.stopPropagation();document.body.style.cursor="pointer"}} onPointerOut={()=>{document.body.style.cursor="default"}}>
    <mesh renderOrder={5}>
      <boxGeometry args={dimensions}/>
      <meshPhysicalMaterial color={selected?"#ef8279":"#e96860"} transparent opacity={selected?.25:ego?.18:.14} transmission={.48} thickness={1.15} roughness={.12} metalness={0} clearcoat={1} clearcoatRoughness={.16} side={THREE.DoubleSide} depthWrite={false}/>
    </mesh>
    <lineSegments geometry={geometry} renderOrder={6}>
      <lineBasicMaterial color="#c9332b" transparent opacity={selected?1:ego?.92:.72} depthWrite={false}/>
    </lineSegments>
  </group>;
}

function GroundTruthLayer({boxes,selection,onSelect}:{boxes:BoxAnnotation[];selection:SceneSelection|null;onSelect:(selection:SceneSelection)=>void}){
  return <group>
    <JellyBox box={EGO_BOX} ego selected={selection?.kind==="box"&&selection.token===EGO_BOX.token} onSelect={onSelect}/>
    {boxes.map(box=><JellyBox key={box.token} box={box} selected={selection?.kind==="box"&&selection.token===box.token} onSelect={onSelect}/>)}
  </group>;
}

function ClickableBev({points,size,stage,selection,onSelect}:{points:Point[];size:number;stage:"quantization grid"|"pseudo-image";selection:SceneSelection|null;onSelect:(selection:SceneSelection)=>void}){
  const extent=24;
  const occupancy=useMemo(()=>{
    const counts=new Map<string,number>();
    points.forEach(([x,y])=>{if(x< -extent||x>=extent||y< -extent||y>=extent)return;const ix=Math.floor((x+extent)/size),iy=Math.floor((y+extent)/size),key=`${ix},${iy}`;counts.set(key,(counts.get(key)??0)+1)});
    return counts;
  },[points,size]);
  const chosen=selection?.kind==="cell"&&selection.stage===stage?selection:null;
  const choose=(e:ThreeEvent<PointerEvent>)=>{
    e.stopPropagation();
    const ix=Math.floor((e.point.x+extent)/size),iy=Math.floor((e.point.y+extent)/size);
    if(ix<0||iy<0||ix>=48/size||iy>=48/size)return;
    const x0=-extent+ix*size,y0=-extent+iy*size;
    onSelect({kind:"cell",stage,index:[iy,ix],center:[x0+size/2,y0+size/2],bounds:[x0,x0+size,y0,y0+size],size,pointCount:occupancy.get(`${ix},${iy}`)??0});
  };
  return <group>
    <mesh position={[0,0,-2.4]} onPointerDown={choose} onPointerOver={()=>{document.body.style.cursor="crosshair"}} onPointerOut={()=>{document.body.style.cursor="default"}}>
      <planeGeometry args={[48,48]}/><meshBasicMaterial transparent opacity={0} side={THREE.DoubleSide} depthWrite={false}/>
    </mesh>
    {chosen&&<mesh position={[chosen.center[0],chosen.center[1],-2.32]} renderOrder={7}>
      <planeGeometry args={[size*.92,size*.92]}/><meshBasicMaterial color="#e67818" transparent opacity={.28} side={THREE.DoubleSide} depthWrite={false}/>
    </mesh>}
  </group>;
}

function SceneContent({data,boxData,step,selection,onSelection,onMetrics}:{data:DemoData;boxData:BoxData;step:number;selection:SceneSelection|null;onSelection:(selection:SceneSelection|null)=>void;onMetrics:(m:PillarMetrics)=>void}){
  const cells=useMemo<PillarCell[]>(()=>{
    const grouped=new Map<string,PointRecord[]>();
    data.points.forEach((point,index)=>{
      const [x,y,z]=point;if(x< -GRID_EXTENT||x>=GRID_EXTENT||y< -GRID_EXTENT||y>=GRID_EXTENT||z<Z_MIN||z>=Z_MAX)return;
      const ix=Math.floor((x+GRID_EXTENT)/PILLAR_SIZE),iy=Math.floor((y+GRID_EXTENT)/PILLAR_SIZE),key=`${ix},${iy}`;
      const records=grouped.get(key)??[];records.push({point,index});grouped.set(key,records);
    });
    return [...grouped.entries()].map(([key,records])=>{
      const [ix,iy]=key.split(",").map(Number),x0=-GRID_EXTENT+ix*PILLAR_SIZE,y0=-GRID_EXTENT+iy*PILLAR_SIZE;
      const mean=[0,1,2].map(axis=>records.reduce((sum,record)=>sum+record.point[axis],0)/records.length) as [number,number,number];
      const center:[number,number,number]=[x0+PILLAR_SIZE/2,y0+PILLAR_SIZE/2,(Z_MIN+Z_MAX)/2];
      return {key,ix,iy,x0,x1:x0+PILLAR_SIZE,y0,y1:y0+PILLAR_SIZE,center,mean,records};
    }).sort((a,b)=>a.iy-b.iy||a.ix-b.ix);
  },[data]);
  const [selectedKey,setSelectedKey]=useState(DEFAULT_PILLAR_KEY);
  const selectedCell=cells.find(cell=>cell.key===selectedKey)??cells[0];
  const [selectedIndex,setSelectedIndex]=useState(0);
  useEffect(()=>{setSelectedIndex(0)},[selectedKey]);
  const insideRecords=selectedCell.records;
  const insideIndices=useMemo(()=>new Set(insideRecords.map(record=>record.index)),[insideRecords]);
  const outsideRecords=useMemo(()=>data.points.map((point,index)=>({point,index})).filter(record=>!insideIndices.has(record.index)),[data,insideIndices]);
  const inside=useMemo(()=>insideRecords.map(record=>record.point),[insideRecords]);
  const outside=useMemo(()=>outsideRecords.map(record=>record.point),[outsideRecords]);
  const selected=inside[selectedIndex]??inside[0];
  const maxPoints=useMemo(()=>Math.max(...cells.map(cell=>cell.records.length)),[cells]);
  useEffect(()=>{if(selected)onMetrics({count:inside.length,center:selectedCell.center,mean:selectedCell.mean,selected,cellIndex:[selectedCell.iy,selectedCell.ix],nonEmptyPillars:cells.length,maxPoints})},[inside,selectedCell,selected,maxPoints,cells.length,onMetrics]);
  const outsidePositions=useMemo(()=>new Float32Array(outside.flatMap(point=>point.slice(0,3))),[outside]);
  const insidePositions=useMemo(()=>new Float32Array(inside.flatMap(point=>point.slice(0,3))),[inside]);
  const choosePoint=(item:PointRecord,insideTeachingPillar:boolean,e:ThreeEvent<PointerEvent>)=>{
    e.stopPropagation();const [x,y,z]=item.point;
    onSelection({kind:"point",index:item.index,point:item.point,range:Math.hypot(x,y,z),cell:[Math.floor((y+GRID_EXTENT)/PILLAR_SIZE),Math.floor((x+GRID_EXTENT)/PILLAR_SIZE)],insideTeachingPillar});
  };
  const choosePillar=(key:string)=>{setSelectedKey(key);onSelection(null)};
  const focus=useMemo<[number,number,number]>(()=>[selectedCell.center[0],selectedCell.center[1],selectedCell.mean[2]],[selectedCell]);
  const teaching=step===2||step===3;
  const fixedReference:[number,number,number]=[selectedCell.center[0],selectedCell.center[1],selected[2]];
  return <>
    <StoryCamera step={step} focus={focus}/>
    {step===1&&<GroundTruthLayer boxes={boxData.boxes} selection={selection} onSelect={onSelection}/>}
    <points onPointerDown={e=>{if(typeof e.index==="number")choosePoint(outsideRecords[e.index],false,e)}}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[outsidePositions,3]}/></bufferGeometry>
      <pointsMaterial color="#111111" size={teaching?.105:step>=4?.1:.16} transparent opacity={teaching?(step===2?.34:.13):step>=4?.15:.86} sizeAttenuation/>
    </points>
    <points onPointerDown={(e:ThreeEvent<PointerEvent>)=>{if(typeof e.index==="number"){setSelectedIndex(e.index);choosePoint(insideRecords[e.index],true,e)}}}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[insidePositions,3]}/></bufferGeometry>
      <pointsMaterial color="#050505" size={teaching?.25:.17} transparent opacity={1} sizeAttenuation/>
    </points>
    {selection?.kind==="point"&&<group position={[selection.point[0],selection.point[1],selection.point[2]]} renderOrder={8}><mesh><sphereGeometry args={[.16,18,18]}/><meshBasicMaterial color="#e67818"/></mesh><mesh rotation={[Math.PI/2,0,0]}><ringGeometry args={[.24,.29,30]}/><meshBasicMaterial color="#e67818" transparent opacity={.8} side={THREE.DoubleSide}/></mesh></group>}
    {step===1&&<><MetricGrid/><ClickableBev points={data.points} size={PILLAR_SIZE} stage="quantization grid" selection={selection} onSelect={onSelection}/></>}
    {teaching&&<>
      <ManyPillarsLayer cells={cells} selectedKey={selectedCell.key} muted={step===3} onSelect={choosePillar}/>
      <MetricGrid local={step===3} center={selectedCell.center}/><SelectedPillar key={`${step}-${selectedCell.key}`} cell={selectedCell}/>
      <VectorLine from={[selectedCell.center[0],selectedCell.center[1],Z_MIN]} to={[selectedCell.center[0],selectedCell.center[1],Z_MAX]} color="#65a8c6"/>
      <CenterMarker position={fixedReference} kind="fixed"/><CenterMarker position={selectedCell.mean} kind="mean"/>
      <mesh position={[selected[0],selected[1],selected[2]]}><sphereGeometry args={[.15,20,20]}/><meshBasicMaterial color="#050505"/></mesh>
      <VectorLine from={[selected[0],selected[1],selected[2]]} to={fixedReference} color="#65a8c6"/><VectorLine from={[selected[0],selected[1],selected[2]]} to={selectedCell.mean} color="#111111"/>
    </>}
    {step===4&&<><PseudoImageLayer points={data.points}/><ClickableBev points={data.points} size={3} stage="pseudo-image" selection={selection} onSelect={onSelection}/></>}
    {step===5&&<BackboneLayer/>}
    {step===6&&<AnchorLayer/>}
    {step===7&&<><AnchorLayer/><MatchingLayer mode="match"/></>}
    {step===8&&<MatchingLayer mode="loss"/>}
    {step===9&&<MatchingLayer mode="nms"/>}
    {step===10&&<MatchingLayer mode="final"/>}
  </>;
}

export function PillarScene({step,selection,onSelection,onMetrics}:{step:number;selection:SceneSelection|null;onSelection:(selection:SceneSelection|null)=>void;onMetrics:(m:PillarMetrics)=>void}){
  const [data,setData]=useState<DemoData|null>(null);
  const [boxData,setBoxData]=useState<BoxData|null>(null);
  useEffect(()=>{
    const base=process.env.NODE_ENV==="production"?"/pointpillars-explained":"";
    Promise.all([
      fetch(`${base}/data/nuscenes-lidar-demo.json`).then(r=>r.json()),
      fetch(`${base}/data/nuscenes-boxes.json`).then(r=>r.json())
    ]).then(([cloud,boxes])=>{setData(cloud);setBoxData(boxes)});
  },[]);
  return <div className="pp-canvas" aria-label="Interactive PointPillars geometry">
    {data&&boxData?<Canvas camera={{position:[31,27,25],fov:44,near:.05,far:200}} dpr={[1,1.7]} gl={{antialias:true}} onCreated={({raycaster})=>{raycaster.params.Points.threshold=.34}} onPointerMissed={()=>onSelection(null)}>
      <color attach="background" args={[PAPER]}/><ambientLight intensity={2.2}/><directionalLight position={[8,9,14]} intensity={2.8}/>
      <SceneContent data={data} boxData={boxData} step={step} selection={selection} onSelection={onSelection} onMetrics={onMetrics}/>
    </Canvas>:<div className="pp-loading">Loading the LiDAR frame</div>}
  </div>;
}
