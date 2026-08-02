"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { PillarScene, type DetectionMetrics, type PillarMetrics, type SceneSelection } from "./PillarScene";

const steps=[
  {title:"A point cloud",kicker:"Start with measurements",line:"Each black mark is one unordered LiDAR return. Click any point to inspect its measured coordinates and reflectance.",formula:"pᵢ = (xᵢ, yᵢ, zᵢ, rᵢ)",note:"No rows, columns, or semantic order—only measurements in the LIDAR_TOP sensor frame."},
  {title:"Quantize XY",kicker:"Choose a spatial address",line:"From BEV, floor division assigns every point to one cell. The translucent red boxes are this frame’s nuScenes ground truth.",formula:"iₓ = ⌊(x − xₘᵢₙ) / vₓ⌋",note:"Click any jelly box or grid square. Empty cells are valid addresses too."},
  {title:"Build the pillars",kicker:"Group · reference · decorate · stack",line:"Every occupied XY cell becomes a full-height pillar. Select one to trace its grid center, observed-point mean, 9D features, and exact place in the stacked tensor.",formula:"X[:,p,n] = [x,y,z,r,x−x̄,y−ȳ,z−z̄,x−xₚ,y−yₚ]",note:"The 406 blue columns are the exact occupied cells inside the shown teaching range; out-of-range points are intentionally not pillarized."},
  {title:"Encode each pillar",kicker:"The simplified PointNet",line:"Apply the same Linear, BatchNorm and ReLU to every point, then take a channel-wise maximum across its N points.",formula:"(D,P,N) → (C,P,N) →max N (C,P)",note:"Max is over the point axis N—not over feature channels. A different point can win each channel."},
  {title:"Scatter to BEV",kicker:"Return features to space",line:"Each encoded pillar returns to its XY address. Empty cells remain zero, forming a pseudo-image.",formula:"I[:, iᵧ, iₓ] = fₚ",note:"This looks like an image, but every pixel stores a learned LiDAR feature."},
  {title:"Build context",kicker:"A two-dimensional backbone",line:"Convolutions trade resolution for receptive field, then align several scales for the detector.",formula:"F = concat(up(F₁), up(F₂), up(F₃))",note:"Coarser layers see farther; shallow layers retain precise location."},
  {title:"Place truck anchors",kicker:"Panorama first · one prior at a time",line:"A readable crop of truck-shaped priors spans the road ahead. Click one: the camera moves from the field to that single geometric hypothesis.",formula:"a = (xₐ,yₐ,zₐ,wₐ,lₐ,hₐ,θₐ)",note:"Only a teaching crop is drawn. A production head tiles anchors over the full detection grid and configured classes."},
  {title:"Match + train",kicker:"One real truck · three supervision states",line:"Around the front ground-truth truck, rotated BEV IoU labels anchors positive, ignored, or negative; only then do classification and box residual losses receive targets.",formula:"IoU → label · t = encode(gt,a) · Lcls + Lloc + Ldir",note:"Thresholds shown here are explicit teaching settings—not universal PointPillars constants."},
  {title:"CenterHead branch",kicker:"Same BEV · no anchor boxes",line:"A later CenterPoint head replaces tiled boxes with a class heatmap: the truck center becomes a Gaussian peak, while separate channels recover its exact center and 3D attributes.",formula:"H(c)=Gaussian · q=[δx,δy,z,log d,sinθ,cosθ,v]",note:"CenterHead is a later anchor-free detector, not part of the 2019 PointPillars paper. It can consume pillar-based BEV features."},
  {title:"Suppress duplicates",kicker:"Keep one explanation",line:"NMS sorts candidates and removes lower-scoring boxes that overlap a selected box too strongly.",formula:"suppress bⱼ if IoU(b*, bⱼ) > τ",note:"Greedy and fast—not learned object reasoning."},
  {title:"Return detections",kicker:"Back to physical space",line:"The remaining class, score, position, dimensions, and yaw are rendered over the original point cloud.",formula:"d = (class, score, x,y,z,w,l,h,θ)",note:"The final box returns to the same frame where the story began."},
] as const;

function Matrix({rows,cols,tone="blue"}:{rows:number;cols:number;tone?:"blue"|"purple"|"ink"}){
  return <div className={`pp-matrix ${tone}`} style={{gridTemplateColumns:`repeat(${cols},1fr)`}}>{Array.from({length:rows*cols},(_,i)=><i key={i} className={(i*7+i%cols)%11===0?"marked":""}/>)}</div>;
}

function PillarConstructionLedger({metrics}:{metrics:PillarMetrics|null}){
  const count=metrics?.count??38,cell=metrics?.cellIndex??[10,11],center=metrics?.center??[-6.75,-8.25,.5],mean=metrics?.mean??[-6.92,-8.36,-.67],pillars=metrics?.nonEmptyPillars??406;
  return <div className="pp-ledger pp-build-ledger" aria-label="Pillar construction from points to stacked tensor">
    <section><div className="pp-dot-cluster">{Array.from({length:Math.min(18,count)},(_,i)=><i key={i}/>)}</div><b>Pillar [{cell[0]}, {cell[1]}]</b><span>{count} grouped points</span></section>
    <em data-note="two references">→</em>
    <section><div className="pp-center-sketch"><i/><i/></div><b>cₚ ≠ p̄</b><span>grid ({center[0].toFixed(2)}, {center[1].toFixed(2)})<br/>mean ({mean.map(v=>v.toFixed(2)).join(", ")})</span></section>
    <em data-note="decorate each point">→</em>
    <section><Matrix rows={5} cols={9}/><b>9 features</b><span>raw · cluster · center</span></section>
    <em data-note="cap N · mask padding">→</em>
    <section><div className="pp-stack"><Matrix rows={5} cols={9}/><Matrix rows={5} cols={9}/><Matrix rows={5} cols={9}/></div><b>9 × {pillars} × 100</b><span>D × P × N</span></section>
    <small><strong>{pillars}</strong> real non-empty pillars<br/><strong>max {metrics?.maxPoints??927}</strong> returns → cap 100<br/><strong>D = 9</strong> decorated features</small>
  </div>;
}

function EncodeLedger({metrics}:{metrics:PillarMetrics|null}){
  const pillars=metrics?.nonEmptyPillars??406;
  return <div className="pp-ledger pp-encode-ledger" aria-label="Pillar Feature Net transformation">
    <section><Matrix rows={5} cols={9}/><b>9 × {pillars} × 100</b><span>same stacked tensor</span></section>
    <em data-note="shared Linear · BN · ReLU">→</em>
    <section><Matrix rows={5} cols={8} tone="purple"/><b>64 × {pillars} × 100</b><span>point features</span></section>
    <em data-note="max over N">→</em>
    <section className="pooled"><Matrix rows={1} cols={8} tone="ink"/><b>64 × {pillars}</b><span>one vector / pillar</span></section>
  </div>;
}

function AnchorFieldLedger({metrics}:{metrics:DetectionMetrics|null}){
  return <div className="pp-detection-ledger pp-anchor-ledger" aria-label="Truck anchor field construction">
    <section><div className="pp-anchor-mini template"><i/></div><b>truck prior</b><span>9.6 × 2.8 × 3.4 m</span></section>
    <em data-note="rotate the same prior">→</em>
    <section><div className="pp-anchor-mini rotations"><i/><i/></div><b>2 headings</b><span>0° · 90°</span></section>
    <em data-note="tile readable crop">→</em>
    <section><div className="pp-anchor-grid-mini">{Array.from({length:25},(_,i)=><i key={i}/>)}</div><b>25 centers × 2</b><span>{metrics?.anchorCount??50} clickable anchors</span></section>
    <small><strong>Why a crop?</strong><br/>All classes across the full BEV would become visual noise. The detector logic is unchanged; only the camera’s teaching window is smaller.</small>
  </div>;
}

function TruckMatchLedger({metrics}:{metrics:DetectionMetrics|null}){
  const selected=metrics?.selected,best=metrics?.best;
  return <div className="pp-detection-ledger pp-match-ledger" aria-label="Anchor matching, box coding, and loss design">
    <section className="assignment"><div className="pp-iou-mini"><i/><i/><i/></div><b>rotated BEV IoU</b><span><mark>{metrics?.positiveCount??1} positive</mark> · {metrics?.ignoreCount??2} ignore · {metrics?.negativeCount??47} negative</span></section>
    <em data-note="assign supervision">→</em>
    <section className="residual"><div className="pp-residual-mini"><i/><i/></div><b>{selected?.status??"positive"} · IoU {(selected?.iou??.9).toFixed(3)}</b><span>Δx {(selected?.residual[0]??0).toFixed(3)} · Δy {(selected?.residual[1]??.025).toFixed(3)}<br/>Δθ {(selected?.residual[6]??.024).toFixed(3)} rad</span></section>
    <em data-note="predict residuals">→</em>
    <section className="losses"><div><i>cls</i><i>loc</i><i>dir</i></div><b>three learning signals</b><span>focal · Smooth L1 · direction CE</span></section>
    <small><strong>L = (Lcls + 2 Lloc + .2 Ldir) / Npos</strong><br/>Best positive IoU {(best?.iou??.9).toFixed(3)}. Teaching probe: p=.72 → FL {(metrics?.illustrative.classificationLoss??.00644).toFixed(4)}; t̂=0 → ΣSL1 {(metrics?.illustrative.zeroResidualLoss??.04305).toFixed(4)}.<br/>Negative: background cls. Ignore: no loss.</small>
  </div>;
}

function CenterHeadLedger({metrics}:{metrics:DetectionMetrics|null}){
  const center=metrics?.centerHead;
  return <div className="pp-detection-ledger pp-centerhead-ledger" aria-label="CenterPoint CenterHead target and regression branches">
    <section><div className="pp-gaussian-mini">{Array.from({length:81},(_,i)=><i key={i}/>)}</div><b>truck heatmap</b><span>r = {center?.radius??13} · σ = {(center?.sigma??4.5).toFixed(1)} cells</span></section>
    <em data-note="peak at integer cell">→</em>
    <section><div className="pp-offset-mini"><i/><i/><span/></div><b>recover sub-cell center</b><span>δ = ({(center?.offset[0]??.753).toFixed(3)}, {(center?.offset[1]??.133).toFixed(3)})</span></section>
    <em data-note="gather at the center">→</em>
    <section className="heads"><div><i>2</i><i>1</i><i>3</i><i>2</i><i>2</i></div><b>attribute heads</b><span>offset · z · dims · rot · velocity</span></section>
    <small><strong>CenterPoint 2021 branch</strong><br/>0.4 m output cell. Heatmap focal loss + 0.25 × gathered L1 regression in the official pillar config.</small>
  </div>;
}

function SelectionCard({selection,onClose}:{selection:SceneSelection;onClose:()=>void}){
  const f=(value:number,digits=2)=>value.toFixed(digits);
  const title=selection.kind==="point"?`Point ${selection.index}`:selection.kind==="cell"?`Cell [${selection.index[0]}, ${selection.index[1]}]`:selection.kind==="anchor"?`Truck anchor · ${selection.id.split("-").at(-1)}°`:selection.category.split(".").map(part=>part.replaceAll("_"," ")).join(" / ");
  const rows=selection.kind==="point"?[
    ["x · right",`${f(selection.point[0],3)} m`],["y · forward",`${f(selection.point[1],3)} m`],["z · up",`${f(selection.point[2],3)} m`],["reflectance",f(selection.point[3],3)],["range",`${f(selection.range)} m`],["pillar [y,x]",`[${selection.cell.join(", ")}]`]
  ]:selection.kind==="box"?[
    ["center (x,y,z)",`(${selection.center.map(v=>f(v)).join(", ")}) m`],["length × width × height",`${selection.dimensions.map(v=>f(v)).join(" × ")} m`],["yaw",`${f(selection.yaw*180/Math.PI,1)}°`],["LiDAR returns",String(selection.numLidarPoints)],["radar returns",String(selection.numRadarPoints)],["token",selection.token.slice(0,8)+"…"]
  ]:selection.kind==="anchor"?[
    ["assignment",selection.status],["loss mask",selection.status==="positive"?"cls + loc + direction":selection.status==="negative"?"background cls only":"none"],["rotated BEV IoU",f(selection.iou,4)],["center (x,y,z)",`(${selection.center.map(v=>f(v)).join(", ")}) m`],["length × width × height",`${selection.dimensions.map(v=>f(v)).join(" × ")} m`],["yaw",`${f(selection.yaw*180/Math.PI,1)}°`],["target (Δx,Δy,Δz)",`(${selection.residual.slice(0,3).map(v=>f(v,3)).join(", ")})`],["target (log w,l,h)",`(${selection.residual.slice(3,6).map(v=>f(v,3)).join(", ")})`],["target Δyaw",`${f(selection.residual[6],3)} rad`]
  ]:[
    ["index [y,x]",`[${selection.index.join(", ")}]`],["center (x,y)",`(${selection.center.map(v=>f(v)).join(", ")}) m`],["x bounds",`[${f(selection.bounds[0])}, ${f(selection.bounds[1])})`],["y bounds",`[${f(selection.bounds[2])}, ${f(selection.bounds[3])})`],["cell size",`${f(selection.size)} × ${f(selection.size)} m`],["raw points",String(selection.pointCount)]
  ];
  const label=selection.kind==="point"?"RAW LIDAR RETURN":selection.kind==="cell"?selection.stage.toUpperCase():selection.kind==="anchor"?`${selection.chapter.toUpperCase()} · TRUCK PRIOR`:selection.source.toUpperCase();
  return <aside className={`pp-inspector ${selection.kind}`} aria-live="polite">
    <button onClick={onClose} aria-label="Close information card">×</button><span>{label}</span><h2>{title}</h2>
    <div>{rows.map(([key,value])=><p key={key}><b>{key}</b><em>{value}</em></p>)}</div>
    <small>{selection.kind==="box"&&selection.source==="teaching geometry"?"Ego is a teaching envelope, not a dataset annotation.":selection.kind==="anchor"?"Residuals encode this real nuScenes truck relative to the selected prior.":"LIDAR_TOP · +x right · +y forward · +z up"}</small>
  </aside>;
}

export default function PillarExplainer(){
  const [step,setStep]=useState(0);const [playing,setPlaying]=useState(false);const [metrics,setMetrics]=useState<PillarMetrics|null>(null);const [detectionMetrics,setDetectionMetrics]=useState<DetectionMetrics|null>(null);const [selection,setSelection]=useState<SceneSelection|null>(null);
  const go=useCallback((next:number)=>{setStep(Math.max(0,Math.min(steps.length-1,next)));setPlaying(false)},[]);
  useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>setStep(s=>s===steps.length-1?0:s+1),4800);return()=>clearInterval(id)},[playing]);
  useEffect(()=>{setSelection(null);document.body.style.cursor="default"},[step]);
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.key==="ArrowRight")go(step+1);if(e.key==="ArrowLeft")go(step-1);if(e.key===" "){e.preventDefault();setPlaying(v=>!v)}};window.addEventListener("keydown",key);return()=>window.removeEventListener("keydown",key)},[go,step]);
  const s=steps[step];
  const gesture=step===2||step===3?"CLICK A BLUE PILLAR TO TRACE IT · CLICK A BLACK POINT TO INSPECT IT":step===6||step===7?"CLICK A TRUCK ANCHOR · THE CAMERA WILL FOLLOW · DRAG TO ORBIT":"DRAG TO ROTATE · SCROLL TO ZOOM · CLICK A POINT, BOX, OR BEV CELL";
  return <main className="pp-app">
    <header className="pp-header"><a href="#" className="pp-wordmark">PointPillars <span>/ geometry lab</span></a><div className="pp-stage-count">{String(step+1).padStart(2,"0")} <i/> {String(steps.length).padStart(2,"0")}</div></header>
    <section className="pp-stage">
      <PillarScene step={step} selection={selection} onSelection={setSelection} onMetrics={setMetrics} onDetectionMetrics={setDetectionMetrics}/>
      <article className="pp-story" key={step}><span>{s.kicker}<sup>{step+1}</sup></span><h1>{s.title}</h1><p>{s.line}</p><code>{s.formula}</code></article>
      {selection&&<SelectionCard selection={selection} onClose={()=>setSelection(null)}/>}
      {step===2&&<PillarConstructionLedger metrics={metrics}/>}
      {step===3&&<EncodeLedger metrics={metrics}/>}
      {step===6&&<AnchorFieldLedger metrics={detectionMetrics}/>}
      {step===7&&<TruckMatchLedger metrics={detectionMetrics}/>}
      {step===8&&<CenterHeadLedger metrics={detectionMetrics}/>}
      {(step===6||step===7||step===8)&&<div className={`pp-branch-label ${step===8?"center":"anchor"}`}>{step===8?"SAME BEV FEATURES → CENTERHEAD · LATER METHOD":"BEV FEATURES → ANCHOR HEAD"}</div>}
      <aside className={`pp-margin-note ${step>=6&&step<=8?"pp-detection-note":""}`} key={`n${step}`}><b>{step+1}.</b><p>{s.note}</p><i/></aside>
      {step===1&&<div className="pp-box-key"><i/>RED JELLY = NUSCENES GT · EGO = TEACHING ENVELOPE</div>}
      <div className="pp-gesture">{gesture}</div>
    </section>
    <footer className="pp-controls">
      <button onClick={()=>go(step-1)} disabled={step===0} aria-label="Previous step"><ChevronLeft/></button>
      <button className="pp-play" onClick={()=>setPlaying(v=>!v)} aria-label={playing?"Pause":"Play"}>{playing?<Pause fill="currentColor"/>:<Play fill="currentColor"/>}</button>
      <div className="pp-track" style={{gridTemplateColumns:`repeat(${steps.length}, 1fr)`}}>{steps.map((item,i)=><button key={item.title} onClick={()=>go(i)} className={i===step?"active":i<step?"passed":""} aria-label={`Step ${i+1}: ${item.title}`}><i/><span>{item.title}</span></button>)}</div>
      <button onClick={()=>go(step+1)} disabled={step===steps.length-1} aria-label="Next step"><ChevronRight/></button>
    </footer>
  </main>;
}
