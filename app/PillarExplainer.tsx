"use client";

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { PillarScene, type PillarMetrics } from "./PillarScene";

const steps=[
  {title:"A point cloud",kicker:"Start with measurements",line:"Each black mark is one unordered LiDAR return: (x, y, z, reflectance).",formula:"P = {p₁, …, pₙ}"},
  {title:"Quantize XY",kicker:"Choose a spatial address",line:"Floor division assigns every accepted point to exactly one BEV cell.",formula:"iₓ = ⌊(x − xₘᵢₙ) / vₓ⌋"},
  {title:"Form a pillar",kicker:"Keep Z continuous",line:"A pillar is one XY cell extended through the full detection height. Its black points share an address—not a height bin.",formula:"pillar[iᵧ,iₓ] = {pᵢ}"},
  {title:"Pillar center",kicker:"A fixed geometric reference",line:"The cell center comes from grid boundaries. It does not move when points move inside the pillar.",formula:"xₚ = xₘᵢₙ + (iₓ + ½)vₓ"},
  {title:"Point-set mean",kicker:"A data-dependent reference",line:"Average the actual points in this pillar. Unlike the pillar center, this point follows the observed surface.",formula:"p̄ = (1/N) Σ pᵢ"},
  {title:"Decorate a point",kicker:"Two offsets, two meanings",line:"Blue locates the point within its fixed cell. Black describes its deviation from the observed local cluster.",formula:"[pᵢ, pᵢ − p̄, xᵢ − xₚ, yᵢ − yₚ]"},
  {title:"Encode the pillar",kicker:"Many points become one vector",line:"A shared point network transforms every decorated point; symmetric max pooling produces one feature per pillar.",formula:"fₚ[c] = maxᵢ MLP(pᵢ)[c]"},
  {title:"Scatter to BEV",kicker:"Return features to space",line:"Each encoded pillar returns to its XY address. Empty cells remain zero, forming a pseudo-image.",formula:"I[:, iᵧ, iₓ] = fₚ"},
  {title:"Build context",kicker:"A two-dimensional backbone",line:"Convolutions trade resolution for receptive field, then align several scales for the detector.",formula:"F = concat(up(F₁), up(F₂), up(F₃))"},
  {title:"Place anchors",kicker:"Hypotheses in physical space",line:"Anchors are class-shaped box priors attached to BEV locations. The network predicts corrections—not boxes from nothing.",formula:"box = anchor ⊕ residual"},
  {title:"Match targets",kicker:"Decide what can teach",line:"BEV overlap assigns positive, negative, or ignored anchors before any detection loss is computed.",formula:"label(a) ← IoU(a, ground truth)"},
  {title:"Balance the loss",kicker:"Rare objects, many backgrounds",line:"Focal classification, robust box regression, and direction classification contribute different learning signals.",formula:"L = βclsLcls + βlocLloc + βdirLdir"},
  {title:"Suppress duplicates",kicker:"Keep one explanation",line:"NMS sorts candidates and removes lower-scoring boxes that overlap a selected box too strongly.",formula:"suppress bⱼ if IoU(b*, bⱼ) > τ"},
  {title:"Return detections",kicker:"Back to physical space",line:"The remaining class, score, position, dimensions, and yaw are rendered over the original point cloud.",formula:"d = (class, score, x,y,z,w,l,h,θ)"},
] as const;

const fmt=(v:number)=>v.toFixed(2);

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
      <article className="pp-story" key={step}><span>{s.kicker}</span><h1>{s.title}</h1><p>{s.line}</p><code>{s.formula}</code></article>
      {!!values.length&&<div className="pp-values" key={`v${step}`}>{values.map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div>}
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
