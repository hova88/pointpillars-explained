"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type DemoData = { points: [number,number,number,number][]; originalPointCount:number; sampling:string };
type ViewMode = "perspective" | "bev" | "ego";

function storyCamera(chapter:number){
  if(chapter<=2)return {position:[33,29,30],look:[0,0,0]};
  if(chapter===3)return {position:[28,28,42],look:[0,0,0]};
  if(chapter<=6)return {position:[13,13,10],look:[5.4,7,1]};
  if(chapter===7)return {position:[10,11,7],look:[5.4,7,1.2]};
  if(chapter<=10)return {position:[11,12,8],look:[5.4,7,1.3]};
  if(chapter===11)return {position:[0,0,40],look:[0,2,0]};
  if(chapter<=13)return {position:[29,25,26],look:[0,1,7]};
  if(chapter<=20)return {position:[21,19,17],look:[2,4,1]};
  if(chapter===21)return {position:[34,29,31],look:[0,3,3]};
  if(chapter<=24)return {position:[18,18,15],look:[2,4,1]};
  return {position:[35,30,34],look:[0,0,2]};
}

function CameraRig({ mode, chapter }: { mode: ViewMode; chapter:number }) {
  const { camera, gl } = useThree();
  const controls=useRef<OrbitControls|null>(null);
  const transition=useRef(0);
  const story=useMemo(()=>storyCamera(chapter),[chapter]);
  const destination=useMemo(()=>{
    if(mode==="bev")return {position:[0,0,58],look:[0,0,0]};
    if(mode==="ego")return {position:[-1,-18,5],look:[0,16,1]};
    return story;
  },[mode,story]);
  useEffect(()=>{
    const orbit=new OrbitControls(camera,gl.domElement);
    orbit.enableDamping=true; orbit.dampingFactor=.07; orbit.enablePan=false; orbit.minDistance=2.5; orbit.maxDistance=90;
    controls.current=orbit;
    return()=>orbit.dispose();
  },[camera,gl]);
  useEffect(()=>{transition.current=0},[chapter,mode]);
  useFrame(() => {
    const orbit=controls.current;
    const position=new THREE.Vector3(...destination.position);
    const look=new THREE.Vector3(...destination.look);
    if(transition.current<.995){
      camera.position.lerp(position,.075);
      orbit?.target.lerp(look,.075);
      transition.current+=.075*(1-transition.current);
    }
    camera.up.set(0, 0, 1);
    orbit?.update();
  });
  return null;
}

function Grid({ size, visible }: { size:number; visible:boolean }) {
  const lines = useMemo(() => {
    const positions:number[]=[];
    const extent=24;
    const step=Math.max(1.2,size*5);
    for(let i=-extent;i<=extent;i+=step){ positions.push(-extent,i,0, extent,i,0, i,-extent,0, i,extent,0); }
    return new Float32Array(positions);
  },[size]);
  if(!visible) return null;
  return <lineSegments>
    <bufferGeometry><bufferAttribute attach="attributes-position" args={[lines,3]} /></bufferGeometry>
    <lineBasicMaterial color="#26475a" transparent opacity={0.48}/>
  </lineSegments>;
}

const Box=({position,size,color="#f5f5f0",opacity=.8,rotation=0}:{position:[number,number,number];size:[number,number,number];color?:string;opacity?:number;rotation?:number})=><mesh position={position} rotation={[0,0,rotation]}><boxGeometry args={size}/><meshBasicMaterial color={color} transparent opacity={opacity} wireframe/></mesh>;

const Segment=({from,to,color}:{from:[number,number,number];to:[number,number,number];color:string})=>{
  const points=useMemo(()=>new Float32Array([...from,...to]),[from,to]);
  return <line><bufferGeometry><bufferAttribute attach="attributes-position" args={[points,3]}/></bufferGeometry><lineBasicMaterial color={color}/></line>;
};

function NarrativeContent({chapter,size,range}:{chapter:number;size:number;range:number}){
  const cells=Array.from({length:16},(_,i)=>({x:(i%4-1.5)*1.55+5.4,y:(Math.floor(i/4)-1.5)*1.55+7,h:.6+(i%5)*.42}));
  const anchors=[
    {p:[7,10,1.1] as [number,number,number],s:[4.4,1.9,1.6] as [number,number,number],r:-.16},
    {p:[-9,4,1] as [number,number,number],s:[4.1,1.8,1.5] as [number,number,number],r:.22},
    {p:[2,-10,1.2] as [number,number,number],s:[4.6,2,1.7] as [number,number,number],r:-.08},
  ];
  if(chapter<=2)return <group>{Array.from({length:4},(_,i)=><mesh key={i} rotation={[0,0,i*.42]}><ringGeometry args={[7+i*4,7.04+i*4,96]}/><meshBasicMaterial color="#7aa7c7" transparent opacity={.18}/></mesh>)}</group>;
  if(chapter===3)return <Box position={[0,0,1.5]} size={[range*2,range*2,3]} color="#f5f5f0" opacity={.5}/>;
  if(chapter<=6)return <group>{cells.map((c,i)=><Box key={i} position={[c.x,c.y,c.h/2]} size={[size*5,size*5,c.h]} color={i===9?"#f2b544":"#7aa7c7"} opacity={i===9?1:.35}/>)}</group>;
  if(chapter===7)return <group><Box position={[5.4,7,3]} size={[size*5,size*5,6]} color="#f2b544"/><mesh position={[5.4,7,1.3]}><sphereGeometry args={[.16,16,16]}/><meshBasicMaterial color="#f2b544"/></mesh><mesh position={[5.78,7.35,1.75]}><sphereGeometry args={[.12,16,16]}/><meshBasicMaterial color="#e9573f"/></mesh><Segment from={[5.78,7.35,1.75]} to={[5.4,7,1.3]} color="#e9573f"/><Segment from={[5.78,7.35,1.75]} to={[5.4,7,0]} color="#7aa7c7"/></group>;
  if(chapter<=10)return <group>{cells.slice(0,10).map((c,i)=><mesh key={i} position={[c.x,c.y,c.h/2]}><cylinderGeometry args={[.08,.08,c.h,8]}/><meshBasicMaterial color={i%3===0?"#f2b544":"#7aa7c7"} transparent opacity={.75}/></mesh>)}</group>;
  if(chapter===11)return <group>{cells.map((c,i)=><mesh key={i} position={[c.x,c.y,.04]}><boxGeometry args={[size*5,size*5,.08]}/><meshBasicMaterial color={new THREE.Color().setHSL(.57,.38,.25+i/55)} transparent opacity={.82}/></mesh>)}</group>;
  if(chapter<=13)return <group>{[0,1,2].map((z)=><mesh key={z} position={[0,1,7+z*2.2]}><planeGeometry args={[30-z*5,22-z*4,12,8]}/><meshBasicMaterial color={z===2?"#f2b544":"#7aa7c7"} transparent opacity={.09+z*.04} wireframe/></mesh>)}</group>;
  if(chapter<=18)return <group>{anchors.map((a,i)=><Box key={i} position={a.p} size={a.s} rotation={a.r} color={i===0?"#f2b544":"#7aa7c7"} opacity={i===0?1:.55}/>)}</group>;
  if(chapter===19)return <group>{anchors.map((a,i)=><Box key={i} position={a.p} size={a.s} rotation={a.r} color={i===1?"#e9573f":"#f2b544"}/>) }{[0,1,2,3].map((i)=><mesh key={i} position={[-4+i*2,11,1+i*.75]}><boxGeometry args={[.7,.7,2+i*1.5]}/><meshBasicMaterial color={i>1?"#e9573f":"#7aa7c7"} transparent opacity={.65}/></mesh>)}</group>;
  if(chapter===20)return <group>{anchors.slice(0,2).map((a,i)=><Box key={i} position={[a.p[0]+2,a.p[1]-2,a.p[2]]} size={a.s} rotation={a.r+.18} color={i?"#7aa7c7":"#e9573f"} opacity={.6}/>)}</group>;
  if(chapter===21)return <group>{[0,1,2,3].map((i)=><mesh key={i} position={[-12+i*8,14,3+i*.8]} rotation={[0,0,-.15]}><boxGeometry args={[5.5,.22,.22]}/><meshBasicMaterial color={i%2?"#e9573f":"#7aa7c7"} transparent opacity={.75}/></mesh>)}</group>;
  if(chapter<=23)return <group>{anchors.map((a,i)=><Box key={i} position={a.p} size={a.s} rotation={a.r} color={chapter===23&&i===1?"#e9573f":"#f2b544"} opacity={chapter===23&&i===1?.25:.9}/>)}</group>;
  if(chapter===24)return <group>{anchors.filter((_,i)=>i!==1).map((a,i)=><Box key={i} position={a.p} size={a.s} rotation={a.r} color="#f2b544" opacity={1}/>)}</group>;
  return <group><Box position={[0,0,1.5]} size={[range*2,range*2,3]} color="#7aa7c7" opacity={.22}/>{anchors.map((a,i)=><Box key={i} position={a.p} size={a.s} rotation={a.r} color="#f2b544" opacity={.7}/>)}</group>;
}

function NarrativeLayers(props:{chapter:number;size:number;range:number}){
  const layer=useRef<THREE.Group>(null);
  useFrame(()=>{if(layer.current){layer.current.position.z=THREE.MathUtils.lerp(layer.current.position.z,0,.09);layer.current.scale.lerp(new THREE.Vector3(1,1,1),.09)}});
  return <group ref={layer} position={[0,0,-1.8]} scale={[.94,.94,.94]}><NarrativeContent {...props}/></group>;
}

function Cloud({ data, colorMode, range, size, showGrid, chapter }:{data:DemoData;colorMode:string;range:number;size:number;showGrid:boolean;chapter:number}){
  const group=useRef<THREE.Group>(null);
  const positions=useMemo(()=>new Float32Array(data.points.flatMap(p=>[p[0],p[1],p[2]])),[data]);
  const colors=useMemo(()=>{
    const a:number[]=[];
    data.points.forEach(([x,y,z,r])=>{
      const inside=Math.abs(x)<range&&Math.abs(y)<range;
      let c:THREE.Color;
      if(!inside)c=new THREE.Color("#24333b");
      else if(colorMode==="height")c=new THREE.Color().setHSL(.48+(z+2)*.035,.82,.58);
      else if(colorMode==="intensity")c=new THREE.Color().setHSL(.14,.82,.22+r*.55);
      else c=new THREE.Color().setHSL(.54+Math.min(1,Math.hypot(x,y)/50)*.15,.85,.62);
      a.push(c.r,c.g,c.b);
    });
    return new Float32Array(a);
  },[data,colorMode,range]);
  useFrame(({clock})=>{ if(group.current&&chapter<=2)group.current.rotation.z=Math.sin(clock.elapsedTime*.18)*.035; });
  return <group ref={group}>
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions,3]}/>
        <bufferAttribute attach="attributes-color" args={[colors,3]}/>
      </bufferGeometry>
      <pointsMaterial size={.105} vertexColors transparent opacity={.93} sizeAttenuation/>
    </points>
    <Grid size={size} visible={showGrid}/>
    <mesh position={[0,0,.35]}><boxGeometry args={[1.8,4.2,.7]}/><meshBasicMaterial color="#d8ff38" wireframe/></mesh>
    <NarrativeLayers key={chapter} chapter={chapter} size={size} range={range}/>
  </group>;
}

export function PointCloudScene({ chapter, gridSize, range }:{chapter:number;gridSize:number;range:number}){
  const [data,setData]=useState<DemoData|null>(null);
  const [view,setView]=useState<ViewMode>("perspective");
  useEffect(()=>{
    const base=process.env.NODE_ENV==="production"?"/pointpillars-explained":"";
    fetch(`${base}/data/nuscenes-lidar-demo.json`).then(r=>r.json()).then(setData);
  },[]);
  const narrative=["A physical laser sweep","A verified nuScenes keyframe","A finite detection world","XY quantization","Sparse grouping","Capacity and masking","Feature decoration","Stacked pillar tensor","Shared point transform","Symmetric pooling","Scatter to BEV","2D spatial context","Multi-scale fusion","Dense predictions","Anchor priors","Target assignment","Box residuals","Heading recovery","Loss contribution","Geometric augmentation","Training flow","Candidate decoding","Iterative suppression","Retained detections","The speed–information bargain"][chapter-1];
  return <div className="scene-shell" aria-label="Interactive nuScenes LiDAR point cloud">
    <div className="scene-toolbar">
      <div className="segmented" aria-label="Camera view">
        {(["perspective","bev","ego"] as ViewMode[]).map(v=><button key={v} className={view===v?"active":""} onClick={()=>setView(v)}>{v==="perspective"?"3D":v.toUpperCase()}</button>)}
      </div>
      <span className="gesture-hint">DRAG · SCROLL</span>
    </div>
    <div className="scene-canvas">
      {data ? <Canvas camera={{position:[33,29,30],fov:48,near:.1,far:300}} dpr={[1,1.5]}>
        <color attach="background" args={["#071014"]}/>
        <fog attach="fog" args={["#071014",45,100]}/>
        <CameraRig mode={view} chapter={chapter}/>
        <Cloud data={data} colorMode="distance" range={range} size={gridSize} showGrid={chapter>=3&&chapter<=13} chapter={chapter}/>
      </Canvas> : <div className="scene-loading">Loading verified LiDAR frame…</div>}
    </div>
    <div className="scene-narrative"><span>{String(chapter).padStart(2,"0")}</span><p>{narrative}</p></div>
    <div className="scene-hud">
      <span><b>{data?.originalPointCount.toLocaleString() ?? "—"}</b> source returns</span>
      <span><b>{data?.points.length.toLocaleString() ?? "—"}</b> rendered</span>
      <span><b>{gridSize.toFixed(2)} m</b> pillar</span>
    </div>
  </div>;
}
