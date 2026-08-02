"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { PillarScene, type PillarMetrics } from "./PillarScene";

const steps=[
  {title:"A point cloud",kicker:"Start with measurements",line:"Each black mark is one unordered LiDAR return: (x, y, z, reflectance).",formula:"P = {p₁, …, pₙ}",note:"No rows, columns, or semantic order—only measured locations."},
  {title:"Quantize XY",kicker:"Choose a spatial address",line:"Floor division assigns every accepted point to exactly one BEV cell.",formula:"iₓ = ⌊(x − xₘᵢₙ) / vₓ⌋",note:"The floor matters: the upper grid boundary stays half-open."},
  {title:"Form a pillar",kicker:"Keep Z continuous",line:"A pillar is one XY cell extended through the full detection height. Its black points share an address—not a height bin.",formula:"pillar[iᵧ,iₓ] = {pᵢ}",note:"38 returns fall inside this real 1.5 m × 1.5 m teaching cell."},
  {title:"Pillar center",kicker:"A fixed geometric reference",line:"The cell center comes from grid boundaries. It does not move when points move inside the pillar.",formula:"xₚ = xₘᵢₙ + (iₓ + ½)vₓ",note:"Grid-defined. Fixed. Do not confuse it with the observed-point mean."},
  {title:"Point-set mean",kicker:"A data-dependent reference",line:"Average the actual points in this pillar. Unlike the pillar center, this point follows the observed surface.",formula:"p̄ = (1/N) Σ pᵢ",note:"Move or remove a return and this center moves with the point set."},
  {title:"Decorate a point",kicker:"Two offsets, two meanings",line:"Blue locates the point within its fixed cell. Black describes its deviation from the observed local cluster.",formula:"[pᵢ, pᵢ − p̄, xᵢ − xₚ, yᵢ − yₚ]",note:"Same point, two reference frames. Neither vector replaces raw XYZ."},
  {title:"Stack the pillars",kicker:"Bound a sparse point set",line:"Retain at most P non-empty pillars and N points per pillar. Sample excess points; pad short pillars with masked zeros.",formula:"X ∈ ℝᴰ×ᴾ×ᴺ,  D = 9",note:"The article’s 9 × 12000 × 100 is one configuration—not a universal constant."},
  {title:"Encode each pillar",kicker:"The simplified PointNet",line:"Apply the same Linear, BatchNorm and ReLU to every point, then take a channel-wise maximum across its N points.",formula:"(D,P,N) → (C,P,N) →max N (C,P)",note:"Max is over the point axis N—not over feature channels. A different point can win each channel."},
  {title:"Scatter to BEV",kicker:"Return features to space",line:"Each encoded pillar returns to its XY address. Empty cells remain zero, forming a pseudo-image.",formula:"I[:, iᵧ, iₓ] = fₚ",note:"This looks like an image, but every pixel stores a learned LiDAR feature."},
  {title:"Build context",kicker:"A two-dimensional backbone",line:"Convolutions trade resolution for receptive field, then align several scales for the detector.",formula:"F = concat(up(F₁), up(F₂), up(F₃))",note:"Coarser layers see farther; shallow layers retain precise location."},
  {title:"Place anchors",kicker:"Hypotheses in physical space",line:"Anchors are class-shaped box priors attached to BEV locations. The network predicts corrections—not boxes from nothing.",formula:"box = anchor ⊕ residual",note:"Anchor dimensions encode expected object geometry before learning."},
  {title:"Match targets",kicker:"Decide what can teach",line:"BEV overlap assigns positive, negative, or ignored anchors before any detection loss is computed.",formula:"label(a) ← IoU(a, ground truth)",note:"Assignment decides which examples the network is allowed to learn from."},
  {title:"Balance the loss",kicker:"Rare objects, many backgrounds",line:"Focal classification, robust box regression, and direction classification contribute different learning signals.",formula:"L = βclsLcls + βlocLloc + βdirLdir",note:"Easy background anchors are numerous; focal weighting quiets them."},
  {title:"Suppress duplicates",kicker:"Keep one explanation",line:"NMS sorts candidates and removes lower-scoring boxes that overlap a selected box too strongly.",formula:"suppress bⱼ if IoU(b*, bⱼ) > τ",note:"Greedy and fast—not learned object reasoning."},
  {title:"Return detections",kicker:"Back to physical space",line:"The remaining class, score, position, dimensions, and yaw are rendered over the original point cloud.",formula:"d = (class, score, x,y,z,w,l,h,θ)",note:"The final box returns to the same frame where the story began."},
] as const;

const fmt=(v:number)=>v.toFixed(2);

function Matrix({rows,cols,tone="blue"}:{rows:number;cols:number;tone?:"blue"|"purple"|"ink"}){
  return <div className={`pp-matrix ${tone}`} style={{gridTemplateColumns:`repeat(${cols},1fr)`}}>{Array.from({length:rows*cols},(_,i)=><i key={i} className={(i*7+i%cols)%11===0?"marked":""}/>)}</div>;
}

function StackLedger({count}:{count:number}){
  return <div className="pp-ledger pp-stack-ledger" aria-label="Stacked pillar tensor construction">
    <section><Matrix rows={5} cols={9}/><b>{count} × 9</b><span>decorated points</span></section>
    <em data-note="sample / zero-pad">→</em>
    <section><div className="pp-stack"><Matrix rows={5} cols={9}/><Matrix rows={5} cols={9}/><Matrix rows={5} cols={9}/></div><b>D × P × N</b><span>stacked pillars</span></section>
    <small><strong>P</strong> capped non-empty pillars<br/><strong>N</strong> capped points per pillar<br/><strong>D = 9</strong> point features</small>
  </div>;
}

function EncodeLedger(){
  return <div className="pp-ledger pp-encode-ledger" aria-label="Pillar Feature Net transformation">
    <section><Matrix rows={5} cols={9}/><b>D × P × N</b><span>decorated tensor</span></section>
    <em data-note="shared Linear · BN · ReLU">→</em>
    <section><Matrix rows={5} cols={8} tone="purple"/><b>C × P × N</b><span>point features</span></section>
    <em data-note="max over N">→</em>
    <section className="pooled"><Matrix rows={1} cols={8} tone="ink"/><b>C × P</b><span>one vector / pillar</span></section>
  </div>;
}

export default function PillarExplainer(){
  const [step,setStep]=useState(0);const [playing,setPlaying]=useState(false);const [metrics,setMetrics]=useState<PillarMetrics|null>(null);
  const go=useCallback((next:number)=>{setStep(Math.max(0,Math.min(steps.length-1,next)));setPlaying(false)},[]);
  useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>setStep(s=>s===steps.length-1?0:s+1),4800);return()=>clearInterval(id)},[playing]);
  useEffect(()=>{const key=(e:KeyboardEvent)=>{if(e.key==="ArrowRight")go(step+1);if(e.key==="ArrowLeft")go(step-1);if(e.key===" "){e.preventDefault();setPlaying(v=>!v)}};window.addEventListener("keydown",key);return()=>window.removeEventListener("keydown",key)},[go,step]);
  const s=steps[step];
  const values=useMemo(()=>{
    if(!metrics)return[];
    if(step===2)return[["POINTS INSIDE",String(metrics.count)],["CELL SIZE","1.50 × 1.50 m"]];
    if(step===3)return[["FIXED CENTER",`(${fmt(metrics.center[0])}, ${fmt(metrics.center[1])})`],["DEPENDS ON","grid only"]];
    if(step===4)return[["POINT MEAN",`(${metrics.mean.map(fmt).join(", ")})`],["DEPENDS ON",`${metrics.count} points`]];
    if(step===5){const p=metrics.selected;return[["SELECTED POINT",`(${p.slice(0,3).map(fmt).join(", ")})`],["TO CELL CENTER",`(${fmt(p[0]-metrics.center[0])}, ${fmt(p[1]-metrics.center[1])})`],["TO POINT MEAN",`(${p.slice(0,3).map((v,i)=>fmt(v-metrics.mean[i])).join(", ")})`]]}
    return[];
  },[metrics,step]);
  return <main className="pp-app">
    <header className="pp-header"><a href="#" className="pp-wordmark">PointPillars <span>/ geometry lab</span></a><div className="pp-stage-count">{String(step+1).padStart(2,"0")} <i/> {String(steps.length).padStart(2,"0")}</div></header>
    <section className="pp-stage">
      <PillarScene step={step} onMetrics={setMetrics}/>
      <article className="pp-story" key={step}><span>{s.kicker}<sup>{step+1}</sup></span><h1>{s.title}</h1><p>{s.line}</p><code>{s.formula}</code></article>
      {!!values.length&&<div className="pp-values" key={`v${step}`}>{values.map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div>}
      {step===6&&<StackLedger count={metrics?.count??38}/>}
      {step===7&&<EncodeLedger/>}
      <aside className="pp-margin-note" key={`n${step}`}><b>{step+1}.</b><p>{s.note}</p><i/></aside>
      <div className="pp-gesture">DRAG TO ROTATE · SCROLL TO ZOOM · CLICK A PILLAR POINT</div>
    </section>
    <footer className="pp-controls">
      <button onClick={()=>go(step-1)} disabled={step===0} aria-label="Previous step"><ChevronLeft/></button>
      <button className="pp-play" onClick={()=>setPlaying(v=>!v)} aria-label={playing?"Pause":"Play"}>{playing?<Pause fill="currentColor"/>:<Play fill="currentColor"/>}</button>
      <div className="pp-track" style={{gridTemplateColumns:`repeat(${steps.length}, 1fr)`}}>{steps.map((item,i)=><button key={item.title} onClick={()=>go(i)} className={i===step?"active":i<step?"passed":""} aria-label={`Step ${i+1}: ${item.title}`}><i/><span>{item.title}</span></button>)}</div>
      <button onClick={()=>go(step+1)} disabled={step===steps.length-1} aria-label="Next step"><ChevronRight/></button>
    </footer>
  </main>;
}
