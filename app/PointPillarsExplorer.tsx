"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, BookOpen, ChevronDown, Code2, Layers3, Pause, Play, RotateCcw, Sigma, SlidersHorizontal } from "lucide-react";
import { chapters, phases } from "./curriculum";
import { PointCloudScene } from "./PointCloudScene";

type LabState={gridSize:number;range:number;maxPoints:number;threshold:number;nmsThreshold:number;gamma:number;alpha:number;pool:"max"|"mean"|"sum"};

const defaultLab:LabState={gridSize:.25,range:24,maxPoints:32,threshold:.45,nmsThreshold:.3,gamma:2,alpha:.25,pool:"max"};
const sourceClass=(s:string)=>s.toLowerCase().replace(/ /g,"-");
const sourceUrl=(s:string)=>s==="NUSCENES"?"https://www.nuscenes.org/nuscenes":s==="IMPLEMENTATION"?"https://github.com/open-mmlab/OpenPCDet":"https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html";

function Reasoning({chapter}:{chapter:typeof chapters[number]}){
  const rows=[
    ["WHAT CHANGES",chapter.operate],
    ["WHY IT EXISTS",chapter.why],
    ["WHY NOT THE OBVIOUS ALTERNATIVE",chapter.compare],
    ["WHAT CAN BREAK",chapter.stress],
    ["WHAT SURVIVES",chapter.summary],
  ];
  return <div className="reasoning-grid">{rows.map(([k,v],i)=><article key={k} className={i===1?"primary-reason":""}><span>{k}</span><p>{v}</p></article>)}</div>;
}

function LabControls({chapter,lab,setLab}:{chapter:number;lab:LabState;setLab:(l:LabState)=>void}){
  const set=(key:keyof LabState,value:number|string)=>setLab({...lab,[key]:value});
  const controls=chapter<=6?[{k:"gridSize",label:"PILLAR SIZE",min:.16,max:.8,step:.01,unit:"m"},{k:"range",label:"XY RANGE",min:12,max:35,step:1,unit:"m"},{k:"maxPoints",label:"MAX POINTS N",min:8,max:100,step:4,unit:""}]:
    chapter===19?[{k:"gamma",label:"FOCAL γ",min:0,max:5,step:.1,unit:""},{k:"alpha",label:"FOCAL α",min:.05,max:.95,step:.05,unit:""}]:
    chapter>=22&&chapter<=24?[{k:"threshold",label:"SCORE τ",min:.05,max:.95,step:.05,unit:""},{k:"nmsThreshold",label:"NMS IoU",min:.05,max:.8,step:.05,unit:""}]:[];
  if(!controls.length&&chapter!==10)return null;
  return <div className="lab-controls">
    <div className="lab-title"><SlidersHorizontal size={15}/><span>LIVE PARAMETERS</span><button onClick={()=>setLab(defaultLab)} aria-label="Reset parameters"><RotateCcw size={14}/></button></div>
    {chapter===10&&<div className="segmented full">{(["max","mean","sum"] as const).map(p=><button className={lab.pool===p?"active":""} key={p} onClick={()=>set("pool",p)}>{p}</button>)}</div>}
    {controls.map(c=><label key={c.k}><span>{c.label}<b>{Number(lab[c.k as keyof LabState]).toFixed(c.step<1?2:0)}{c.unit}</b></span><input type="range" min={c.min} max={c.max} step={c.step} value={lab[c.k as keyof LabState] as number} onChange={e=>set(c.k as keyof LabState,Number(e.target.value))}/></label>)}
    {!controls.length&&chapter!==10&&<p className="control-note">This stage is inspected with the timeline and reasoning layers. Parameters appear only where a live change is mathematically honest.</p>}
  </div>;
}

export default function PointPillarsExplorer(){
  const [active,setActive]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [speed,setSpeed]=useState(1);
  const [lab,setLab]=useState(defaultLab);
  const [navOpen,setNavOpen]=useState(false);
  const chapter=chapters[active];
  const select=useCallback((index:number)=>{const i=(index+chapters.length)%chapters.length;setActive(i);setPlaying(false);window.history.replaceState(null,"",`#${chapters[i].id}`);window.scrollTo({top:document.querySelector(".chapter-shell")?.getBoundingClientRect().top!+window.scrollY-16,behavior:"smooth"});},[]);
  useEffect(()=>{const id=window.location.hash.slice(1);const i=chapters.findIndex(c=>c.id===id);if(i>=0)setActive(i);},[]);
  useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>setActive(v=>v===chapters.length-1?0:v+1),4200/speed);return()=>clearInterval(id);},[playing,speed]);
  useEffect(()=>{const onKey=(e:KeyboardEvent)=>{if((e.target as HTMLElement).tagName==="INPUT")return;if(e.key==="ArrowRight")select(active+1);if(e.key==="ArrowLeft")select(active-1);if(e.key===" "){e.preventDefault();setPlaying(v=>!v);}};window.addEventListener("keydown",onKey);return()=>window.removeEventListener("keydown",onKey);},[active,select]);
  const progress=((active+1)/chapters.length)*100;
  const phaseCounts=useMemo(()=>phases.map(p=>({p,n:chapters.filter(c=>c.phase===p).length})),[]);
  return <main>
    <header className="topbar">
      <a className="brand" href="#top"><span className="brand-mark">PP</span><span>POINTPILLARS<br/><i>INTERACTIVE PAPER</i></span></a>
      <nav><a href="#explorer">EXPLORER</a><a href="#curriculum">CURRICULUM</a><a href="#sources">SOURCES</a></nav>
      <a className="github-link" href="https://github.com/hova88/pointpillars-explained" target="_blank" rel="noreferrer"><Code2 size={16}/> SOURCE</a>
    </header>

    <section className="hero" id="top">
      <div className="hero-copy">
        <div className="eyebrow"><span/>CVPR 2019 · DECONSTRUCTED</div>
        <h1>SEE EVERY<br/><em>DECISION.</em></h1>
        <p>PointPillars turns an unordered 3D LiDAR sweep into fast 3D detections. This is not a summary. It is a reversible, inspectable journey through every representation—and every compromise.</p>
        <div className="hero-actions"><a href="#explorer" className="primary-button"><Play size={16} fill="currentColor"/> BEGIN THE PIPELINE</a><a href="#curriculum" className="text-link">VIEW ALL 25 CHAPTERS <ArrowRight size={15}/></a></div>
        <div className="hero-stats"><div><b>25</b><span>DEEP-DIVE<br/>CHAPTERS</span></div><div><b>34,688</b><span>NUSCENES<br/>RETURNS</span></div><div><b>7</b><span>REASONING<br/>LENSES</span></div></div>
      </div>
      <div className="hero-scene"><PointCloudScene chapter={active+1} gridSize={lab.gridSize} range={lab.range}/><div className="scene-caption"><span>LIVE / NUSCENES LIDAR_TOP</span><span>DRAG PARAMETERS · SWITCH VIEW</span></div></div>
    </section>

    <section className="thesis-strip"><p><span>THE CORE MOVE</span> Organize points into vertical columns, learn one feature vector per column, then use a standard 2D detector.</p><div className="mini-flow"><i>3D</i><span>→</span><i>P</i><span>→</span><i>2D</i></div></section>

    <section className="chapter-shell" id="explorer">
      <aside className={`chapter-nav ${navOpen?"open":""}`} aria-hidden="true">
        <button className="mobile-chapter-toggle" onClick={()=>setNavOpen(v=>!v)}>CHAPTER {chapter.n}: {chapter.title}<ChevronDown size={15}/></button>
        <div className="chapter-list">{phases.map(phase=><div key={phase}><h3>{phase}</h3>{chapters.map((c,i)=>c.phase===phase&&<button key={c.id} onClick={()=>{select(i);setNavOpen(false)}} className={i===active?"active":""}><span>{String(c.n).padStart(2,"0")}</span><p>{c.title}<small>{c.shape}</small></p></button>)}</div>)}</div>
      </aside>

      <div className="chapter-main">
        <div className="chapter-stage story-stage focus-stage">
          <div className="persistent-scene"><PointCloudScene chapter={chapter.n} gridSize={lab.gridSize} range={lab.range}/></div>
          <div className="focus-copy" key={chapter.id}><span>{String(chapter.n).padStart(2,"0")} / 25 · {chapter.phase}</span><h2>{chapter.title}</h2><p>{chapter.subtitle}</p><code>{chapter.equation}</code><small>{chapter.shape}</small></div>
          <LabControls chapter={chapter.n} lab={lab} setLab={setLab}/>
        </div>
        <div className="chapter-dots" aria-label="Chapter timeline">{chapters.map((c,i)=><button key={c.id} className={i===active?"active":""} onClick={()=>select(i)} aria-label={`Chapter ${c.n}: ${c.title}`}/>)}</div>
        <div className="chapter-controls">
          <button onClick={()=>select(active-1)} aria-label="Previous chapter"><ArrowLeft size={17}/></button>
          <button className="play-button" onClick={()=>setPlaying(v=>!v)}>{playing?<Pause size={16} fill="currentColor"/>:<Play size={16} fill="currentColor"/>}{playing?"PAUSE TOUR":"PLAY GUIDED TOUR"}</button>
          <label>SPEED<select value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value={.7}>0.7×</option><option value={1}>1×</option><option value={1.5}>1.5×</option></select></label>
          <div className="progress-line"><span style={{width:progress+"%"}}/></div>
          <button onClick={()=>select(active+1)} aria-label="Next chapter"><ArrowRight size={17}/></button>
        </div>
        <details className="deep-dive"><summary>Inspect the reasoning <span>{chapter.source} · {chapter.shape}</span></summary><div className="deep-dive-inner"><div className="inspect-row"><div><span>EQUATION</span><code>{chapter.equation}</code></div><div><span>SHAPE LEDGER</span><code>{chapter.shape}</code></div><div><span>OBSERVE</span><p>{chapter.observe}</p></div><div><span>INSPECT</span><p>{chapter.inspect}</p></div></div><Reasoning chapter={chapter}/><div className="symbol-glossary" aria-label="Symbol glossary"><span><b>N</b> points / capacity</span><span><b>P</b> non-empty pillars</span><span><b>D</b> decorated channels</span><span><b>C</b> learned channels</span><span><b>H × W</b> BEV grid</span><span><b>A</b> anchors per cell</span><a href={sourceUrl(chapter.source)} target="_blank" rel="noreferrer" className={`source-tag ${sourceClass(chapter.source)}`}>{chapter.source} ↗</a></div></div></details>
      </div>
    </section>

    <section className="curriculum" id="curriculum">
      <div className="section-heading"><span>COMPLETE COVERAGE</span><h2>Nothing is a<br/><em>black box.</em></h2><p>Every “standard” operation earns an explanation. Source badges distinguish the paper from prior work, common implementation choices, nuScenes adaptations, and teaching models.</p></div>
      <div className="phase-map">{phaseCounts.map(({p,n})=><div key={p}><span>{p}</span><b>{n}</b></div>)}</div>
      <div className="curriculum-grid">{chapters.map((c,i)=><button key={c.id} onClick={()=>select(i)}><span>{String(c.n).padStart(2,"0")}</span><div><small>{c.phase}</small><h3>{c.title}</h3><p>{c.summary}</p></div><ArrowRight size={17}/></button>)}</div>
    </section>

    <section className="principles">
      <article><BookOpen/><span>WHAT + HOW</span><h3>Traceable mechanics</h3><p>Every value can be followed from physical return to tensor cell and final box.</p></article>
      <article><Sigma/><span>WHY + WHY NOT</span><h3>Design, not dogma</h3><p>Each choice is placed beside credible alternatives and the property it buys.</p></article>
      <article><Layers3/><span>GAIN + LOSS</span><h3>An information ledger</h3><p>Every compression step states exactly what survives and what can never be recovered.</p></article>
    </section>

    <footer id="sources">
      <div><span className="brand-mark">PP</span><p>An independent, non-commercial educational visualization of PointPillars.</p></div>
      <div><h4>PRIMARY SOURCES</h4><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html">PointPillars · CVPR 2019</a><a href="https://www.nuscenes.org/nuscenes">nuScenes dataset</a><a href="https://github.com/nutonomy/nuscenes-devkit">nuScenes devkit</a></div>
      <div><h4>PROVENANCE</h4><p>Sample ca9a…a8af5<br/>LIDAR_TOP 9d9b…0184f<br/>CC BY-NC-SA 4.0 dataset terms</p></div>
      <div className="footer-end"><span>BUILT TO BE QUESTIONED.</span><a href="#top">BACK TO TOP ↑</a></div>
    </footer>
  </main>;
}
