"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { PillarScene, type DetectionMetrics, type PillarMetrics, type SceneSelection } from "./PillarScene";

const steps=[
  {title:"A point cloud",kicker:"Start with measurements",line:"Each black mark is one unordered LiDAR return. Click any point to inspect its measured coordinates and reflectance.",formula:"pᵢ = (xᵢ, yᵢ, zᵢ, rᵢ)",note:"No rows, columns, or semantic order—only measurements in the LIDAR_TOP sensor frame."},
  {title:"Quantize XY",kicker:"Choose a spatial address",line:"From BEV, floor division assigns every point to one cell. The translucent red boxes are this frame’s nuScenes ground truth.",formula:"iₓ = ⌊(x − xₘᵢₙ) / vₓ⌋",note:"Click any jelly box or grid square. Empty cells are valid addresses too."},
  {title:"Build the pillars",kicker:"Group · reference · decorate · stack",line:"Every occupied XY cell becomes a full-height pillar. Select one to trace its grid center, observed-point mean, 9D features, and exact place in the stacked tensor.",formula:"X[:,p,n] = [x,y,z,r,x−x̄,y−ȳ,z−z̄,x−xₚ,y−yₚ]",note:"The 406 blue columns are the exact occupied cells inside the shown teaching range; out-of-range points are intentionally not pillarized."},
  {title:"Encode each pillar",kicker:"The simplified PointNet",line:"Apply the same Linear, BatchNorm and ReLU to every point, then take a channel-wise maximum across its N points.",formula:"(D,P,N) → (C,P,N) →max N (C,P)",note:"Max is over the point axis N—not over feature channels. A different point can win each channel."},
  {title:"Scatter to BEV",kicker:"Compact list → spatial tensor",line:"The pillar-list index p is not a location. Its paired coordinate (iᵧ,iₓ) tells scatter exactly where to copy the complete 64D pillar feature in the BEV tensor.",formula:"I[:, coords[p]ᵧ, coords[p]ₓ] ← F[p, :]",note:"Scatter does not interpolate or learn a projection. It restores spatial addresses; cells with no pillar remain the all-zero vector."},
  {title:"Build context",kicker:"Dense Conv2D · multi-scale fusion",line:"Original PointPillars deliberately runs ordinary 3×3 2D convolutions over the pseudo-image, downsamples three times, then uses transposed convolutions to align and concatenate all scales.",formula:"B₁ → B₂ → B₃ · Deconv(×1,×2,×4) → concat",note:"Sparse convolution is a valid later alternative for efficiency, but it is not the original backbone—and preserving sparsity can also restrict information flow."},
  {title:"Place truck anchors",kicker:"Panorama first · one prior at a time",line:"A readable crop of truck-shaped priors spans the road ahead. Click one: the camera moves from the field to that single geometric hypothesis.",formula:"a = (xₐ,yₐ,zₐ,wₐ,lₐ,hₐ,θₐ)",note:"Only a teaching crop is drawn. A production head tiles anchors over the full detection grid and configured classes."},
  {title:"Match + train",kicker:"One real truck · three supervision states",line:"Around the front ground-truth truck, rotated BEV IoU labels anchors positive, ignored, or negative; only then do classification and box residual losses receive targets.",formula:"IoU → label · t = encode(gt,a) · Lcls + Lloc + Ldir",note:"Thresholds shown here are explicit teaching settings—not universal PointPillars constants."},
  {title:"CenterHead branch",kicker:"Same BEV · no anchor boxes",line:"A later CenterPoint head replaces tiled boxes with a class heatmap: the truck center becomes a Gaussian peak, while separate channels recover its exact center and 3D attributes.",formula:"H(c)=Gaussian · q=[δx,δy,z,log d,sinθ,cosθ,v]",note:"CenterHead is a later anchor-free detector, not part of the 2019 PointPillars paper. It can consume pillar-based BEV features."},
  {title:"Return to the scene",kicker:"Decode · filter · suppress · compare",line:"Decode scored boxes, remove weak and overlapping candidates, then return every survivor to the same point cloud—and the same red ground truth—introduced in Chapter 2.",formula:"logits → scores · decode(a,t̂) · NMS₀.₅ → {dₖ}",note:"Red is official nuScenes ground truth. Blue boxes are deterministic teaching predictions for geometry—not outputs from a trained checkpoint."},
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

function ScatterLedger({metrics}:{metrics:PillarMetrics|null}){
  const active=metrics?.nonEmptyPillars??406,total=32*32,ratio=active/total*100;
  return <div className="pp-detection-ledger pp-scatter-ledger" aria-label="Scatter compact pillar features into the BEV pseudo-image">
    <section><div className="pp-feature-list-mini">{Array.from({length:6},(_,i)=><i key={i}/>)}</div><b>F[p, :]</b><span>{active} rows × 64 channels</span></section>
    <em data-note="look up coords[p]">→</em>
    <section><div className="pp-coords-mini"><i>(10, 11)</i><i>(10, 12)</i><i>⋮</i><i>(29, 26)</i></div><b>K[p] = (iᵧ, iₓ)</b><span>one address / non-empty pillar</span></section>
    <em data-note="indexed copy · no blend">→</em>
    <section><div className="pp-scatter-grid-mini">{Array.from({length:64},(_,i)=><i key={i} className={(i*7+i%9)%5===0?"active":""}/>)}</div><b>I[:, iᵧ, iₓ]</b><span>64 × 32 × 32 teaching BEV</span></section>
    <small><strong>{active} / {total} active sites · {ratio.toFixed(1)}%</strong><br/>The compact p-axis has no spatial order. Coordinates rebuild the grid. The other {total-active} cells are exact zero vectors—not missing tensor entries.</small>
  </div>;
}

function BackboneLedger(){
  return <div className="pp-detection-ledger pp-backbone-ledger" aria-label="Original PointPillars 2D backbone and feature fusion">
    <section><div className="pp-backbone-input-mini"><i/><i/><i/><span/></div><b>pseudo-image</b><span>64 × H × W · sparse values in a dense tensor</span></section>
    <em data-note="3×3 Conv2D · BN · ReLU">→</em>
    <section className="blocks"><div><i>B₁<small>4 conv · /2 · 64</small></i><i>B₂<small>6 conv · /4 · 128</small></i><i>B₃<small>6 conv · /8 · 256</small></i></div><b>top-down blocks</b><span>resolution ↓ · receptive field ↑</span></section>
    <em data-note="transposed convolution">→</em>
    <section className="ups"><div><i>×1</i><i>×2</i><i>×4</i></div><b>align at stride 2</b><span>128 channels from each scale</span></section>
    <em data-note="concat channels">→</em>
    <section className="fusion"><div><i/><i/><i/></div><b>384 × H/2 × W/2</b><span>detail + context</span></section>
    <small><div className="pp-conv-choice-mini"><span><i/><b>dense 3×3</b></span><span><i/><b>SubM sparse</b></span></div><strong>Original PointPillars = dense Conv2D.</strong><br/>Dense kernels spread context into zero sites. SubM saves work and keeps the active mask, but can isolate nearby components.</small>
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

function FinalPredictionLedger(){
  return <div className="pp-detection-ledger pp-final-ledger" aria-label="Decode, confidence filter, non-maximum suppression, and final prediction comparison">
    <section><div className="pp-candidates-mini"><i/><i/><i/><i/></div><b>4 decoded boxes</b><span>same truck · different scores</span></section>
    <em data-note="score ≥ .30">→</em>
    <section><div className="pp-score-filter-mini"><i>.88</i><i>.71</i><i>.46</i><i>.24</i></div><b>3 candidates remain</b><span>weak evidence exits early</span></section>
    <em data-note="axis-aligned NMS · IoU .5">→</em>
    <section><div className="pp-nms-mini"><i/><i/><i/><b/></div><b>1 truck survives</b><span>greedy score order</span></section>
    <em data-note="decode to LiDAR frame">→</em>
    <section><div className="pp-compare-mini"><i/><i/><span/></div><b>Chapter 2, revisited</b><span>red GT · blue prediction</span></section>
    <small><strong>Visual truthfulness</strong><br/>The 69 red boxes are official annotations from this nuScenes keyframe. Six blue boxes are deterministic teaching geometry, paired to GT so you can inspect residual localization—not benchmark accuracy.</small>
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
  ]:selection.kind==="prediction"?[
    ["class",selection.category.replaceAll("_"," ")],["confidence",f(selection.score,2)],["center (x,y,z)",`(${selection.center.map(v=>f(v)).join(", ")}) m`],["length × width × height",`${selection.dimensions.map(v=>f(v)).join(" × ")} m`],["yaw",`${f(selection.yaw*180/Math.PI,1)}°`],["paired GT IoU",f(selection.matchIou,4)],["paired GT token",selection.gtToken.slice(0,8)+"…"]
  ]:selection.stage==="pseudo-image"?[
    ["BEV index [y,x]",`[${selection.index.join(", ")}]`],["site",selection.active?"active feature":"zero / empty"],["pillar-list index p",selection.pillarListIndex===null?"none":String(selection.pillarListIndex)],["scatter write",selection.pillarListIndex===null?"I[:,y,x] = 0":`I[:,${selection.index[0]},${selection.index[1]}] ← F[${selection.pillarListIndex},:]`],["feature channels",String(selection.featureChannels)],["source points",String(selection.pointCount)],["center (x,y)",`(${selection.center.map(v=>f(v)).join(", ")}) m`]
  ]:[
    ["index [y,x]",`[${selection.index.join(", ")}]`],["center (x,y)",`(${selection.center.map(v=>f(v)).join(", ")}) m`],["x bounds",`[${f(selection.bounds[0])}, ${f(selection.bounds[1])})`],["y bounds",`[${f(selection.bounds[2])}, ${f(selection.bounds[3])})`],["cell size",`${f(selection.size)} × ${f(selection.size)} m`],["raw points",String(selection.pointCount)]
  ];
  const label=selection.kind==="point"?"RAW LIDAR RETURN":selection.kind==="cell"?selection.stage.toUpperCase():selection.kind==="anchor"?`${selection.chapter.toUpperCase()} · TRUCK PRIOR`:selection.source.toUpperCase();
  return <aside className={`pp-inspector ${selection.kind}`} aria-live="polite">
    <button onClick={onClose} aria-label="Close information card">×</button><span>{label}</span><h2>{title}</h2>
    <div>{rows.map(([key,value])=><p key={key}><b>{key}</b><em>{value}</em></p>)}</div>
    <small>{selection.kind==="box"&&selection.source==="teaching geometry"?"Ego is a teaching envelope, not a dataset annotation.":selection.kind==="prediction"?"Teaching geometry only. It must not be interpreted as a measured detector score or benchmark result.":selection.kind==="anchor"?"Residuals encode this real nuScenes truck relative to the selected prior.":selection.kind==="cell"&&selection.stage==="pseudo-image"?"Scatter is an indexed copy: no interpolation, averaging, or learned weights.":"LIDAR_TOP · +x right · +y forward · +z up"}</small>
  </aside>;
}

export default function PillarExplainer(){
  const [step,setStep]=useState(0);const [playing,setPlaying]=useState(false);const [metrics,setMetrics]=useState<PillarMetrics|null>(null);const [detectionMetrics,setDetectionMetrics]=useState<DetectionMetrics|null>(null);const [selection,setSelection]=useState<SceneSelection|null>(null);
  const go=useCallback((next:number)=>{setStep(Math.max(0,Math.min(steps.length-1,next)));setPlaying(false)},[]);
  useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>setStep(s=>s===steps.length-1?0:s+1),4800);return()=>clearInterval(id)},[playing]);
  useEffect(()=>{setSelection(null);document.body.style.cursor="default"},[step]);
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.key==="ArrowRight")go(step+1);if(e.key==="ArrowLeft")go(step-1);if(e.key===" "){e.preventDefault();setPlaying(v=>!v)}};window.addEventListener("keydown",key);return()=>window.removeEventListener("keydown",key)},[go,step]);
  const s=steps[step];
  const gesture=step===2||step===3?"CLICK A BLUE PILLAR TO TRACE IT · CLICK A BLACK POINT TO INSPECT IT":step===4?"CLICK ANY BEV CELL · TRACE PILLAR-LIST ROW → SPATIAL ADDRESS":step===5?"FOLLOW THE ORANGE PULSES · DRAG TO ORBIT THE BACKBONE":step===6||step===7?"CLICK A TRUCK ANCHOR · THE CAMERA WILL FOLLOW · DRAG TO ORBIT":step===9?"WATCH NMS RESOLVE · CLICK A RED GT OR BLUE PREDICTION · DRAG TO ORBIT":"DRAG TO ROTATE · SCROLL TO ZOOM · CLICK A POINT, BOX, OR BEV CELL";
  return <main className="pp-app">
    <header className="pp-header"><a href="#" className="pp-wordmark">PointPillars <span>/ geometry lab</span></a><div className="pp-stage-count">{String(step+1).padStart(2,"0")} <i/> {String(steps.length).padStart(2,"0")}</div></header>
    <section className="pp-stage">
      <PillarScene step={step} selection={selection} onSelection={setSelection} onMetrics={setMetrics} onDetectionMetrics={setDetectionMetrics}/>
      <article className="pp-story" key={step}><span>{s.kicker}<sup>{step+1}</sup></span><h1>{s.title}</h1><p>{s.line}</p><code>{s.formula}</code></article>
      {selection&&<SelectionCard selection={selection} onClose={()=>setSelection(null)}/>}
      {step===2&&<PillarConstructionLedger metrics={metrics}/>}
      {step===3&&<EncodeLedger metrics={metrics}/>}
      {step===4&&<ScatterLedger metrics={metrics}/>}
      {step===5&&<BackboneLedger/>}
      {step===6&&<AnchorFieldLedger metrics={detectionMetrics}/>}
      {step===7&&<TruckMatchLedger metrics={detectionMetrics}/>}
      {step===8&&<CenterHeadLedger metrics={detectionMetrics}/>}
      {step===9&&<FinalPredictionLedger/>}
      {(step===6||step===7||step===8)&&<div className={`pp-branch-label ${step===8?"center":"anchor"}`}>{step===8?"SAME BEV FEATURES → CENTERHEAD · LATER METHOD":"BEV FEATURES → ANCHOR HEAD"}</div>}
      <aside className={`pp-margin-note ${step>=4&&step<=9?"pp-detection-note":""}`} key={`n${step}`}><b>{step+1}.</b><p>{s.note}</p><i/></aside>
      {step===1&&<div className="pp-box-key"><i/>RED JELLY = NUSCENES GT · EGO = TEACHING ENVELOPE</div>}
      {step===9&&<div className="pp-box-key pp-final-key"><i/>RED = CHAPTER 02 GT · <b/>BLUE = TEACHING PREDICTION</div>}
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
