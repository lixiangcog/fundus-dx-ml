import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  Activity, AlertTriangle, ArrowRight, BookOpen, BrainCircuit, Check,
  ChevronRight, Clock3, Database, Eye, FileCheck2, ImageIcon, LoaderCircle,
  Play, ScanLine, Server, ShieldAlert, Stethoscope, Waypoints,
} from 'lucide-react';

const STEPS = [
  { id:'case', label:'CASE TIMELINE', icon:Clock3 },
  { id:'tools', label:'IMAGING TOOLS', icon:ScanLine },
  { id:'vision', label:'MULTIMODAL AI', icon:Eye },
  { id:'rag', label:'EVIDENCE RAG', icon:Database },
  { id:'decision', label:'OPTION REVIEW', icon:Waypoints },
  { id:'report', label:'FINAL REPORT', icon:FileCheck2 },
];

const VISIT_LABELS = { baseline:'V0 / BASELINE', followup:'V1 / FOLLOW-UP' };
const MODALITIES = [
  ['oct','STRUCTURAL OCT'], ['octa','OCT ANGIOGRAPHY'], ['fundus','COLOR FUNDUS'],
];

function StatusBadge({ service }) {
  const ready = service?.status === 'ready';
  return <div className={`agent-service ${ready ? 'ready' : 'waiting'}`}>
    <span><Server size={14}/><i/></span>
    <div><small>LOCAL MLLM SERVICE</small><b>{ready ? 'QWEN2.5-VL / READY' : (service?.status || 'CHECKING').toUpperCase()}</b></div>
  </div>;
}

function SourceTag({ id }) {
  return <span className="source-tag">[{id}]</span>;
}

function AMDWorkspace({ apiUrl }) {
  const [config, setConfig] = useState(null);
  const [service, setService] = useState({status:'checking'});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [error, setError] = useState('');
  const caseData = config?.default_case;

  const refreshStatus = useCallback(() => fetch(`${apiUrl}/amd-agent/status`).then(r => r.json()).then(setService).catch(() => setService({status:'offline'})), [apiUrl]);
  useEffect(() => {
    fetch(`${apiUrl}/amd-agent/config`).then(r => r.json()).then((data) => { setConfig(data); setService(data.service); }).catch(() => setError('AMD Agent configuration is unavailable.'));
    const timer = setInterval(refreshStatus, 12000);
    return () => clearInterval(timer);
  }, [apiUrl, refreshStatus]);

  useEffect(() => {
    if (!loading) return;
    setProgressStep(0);
    const timers = [2200, 6500, 12500, 17000, 22000].map((delay, index) => setTimeout(() => setProgressStep(index + 1), delay));
    return () => timers.forEach(clearTimeout);
  }, [loading]);

  const runAgent = async () => {
    if (loading || service.status !== 'ready') return;
    setLoading(true); setResult(null); setError('');
    try {
      const response = await axios.post(`${apiUrl}/amd-agent/analyze-default`);
      setResult(response.data);
      setProgressStep(STEPS.length - 1);
      setService((current) => ({...current,status:'ready'}));
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Real inference did not complete. Check the GPU Agent status.');
      refreshStatus();
    } finally { setLoading(false); }
  };

  return <div className="amd-workspace">
    <section className="amd-hero">
      <div>
        <span className="eyebrow">LONGITUDINAL AMD FOLLOW-UP // EVIDENCE AGENT</span>
        <h1>LONGITUDINAL AMD FOLLOW-UP<br/><em>MULTIMODAL EVIDENCE AGENT</em></h1>
        <p>Paired OCT, OCTA and fundus images are bound to independently computed biomarkers, retrieved guideline evidence and an auditable candidate-option decision path.</p>
      </div>
      <StatusBadge service={service}/>
    </section>

    <section className="agent-flow">
      {STEPS.map((step,index) => {
        const Icon = step.icon;
        const complete = Boolean(result) || (loading && progressStep > index);
        const active = loading && progressStep === index;
        return <div key={step.id} className={`flow-step ${complete?'complete':''} ${active?'active':''}`}>
          <span>{complete ? <Check size={13}/> : active ? <LoaderCircle size={14}/> : <Icon size={14}/>}</span>
          <div><small>STEP // {String(index+1).padStart(2,'0')}</small><b>{step.label}</b></div>
          {index < STEPS.length-1 && <i><em/></i>}
        </div>;
      })}
    </section>

    {!result ? <section className="amd-console">
      <aside className="case-context">
        <div className="amd-panel-head"><span><Stethoscope size={14}/> CLINICAL CASE</span><b>SYNTHETIC DEMO</b></div>
        {caseData ? <>
          <h2>{caseData.case_id}</h2>
          <p className="case-title">{caseData.title}</p>
          <dl className="case-facts">
            <div><dt>PATIENT</dt><dd>{caseData.patient.age} / {caseData.patient.sex} / {caseData.patient.eye}</dd></div>
            <div><dt>PRIOR DIAGNOSIS</dt><dd>{caseData.patient.diagnosis}</dd></div>
            <div><dt>TREATMENT</dt><dd>{caseData.treatment.agent} / {caseData.treatment.injections} injections</dd></div>
            <div><dt>INTERVAL</dt><dd>{caseData.treatment.current_interval_weeks} weeks</dd></div>
          </dl>
          <p className="case-narrative">{caseData.context}</p>
        </> : <div className="case-loading"><LoaderCircle size={20}/>LOADING CASE...</div>}
        <div className="case-safety"><ShieldAlert size={15}/><span><b>DEMO DATA NOTICE</b>Public samples form a synthetic pair and are not a real longitudinal patient record.</span></div>
      </aside>

      <section className="visit-matrix">
        <div className="amd-panel-head"><span><ImageIcon size={14}/> MULTIMODAL TIMELINE</span><b>2 VISITS / 6 IMAGES</b></div>
        <div className="visit-grid">
          {(caseData?.visits || []).map((visit,index) => <div className="visit-column" key={visit.id}>
            <div className="visit-title"><span>{visit.id}</span><div><b>{visit.label}</b><small>{visit.date} / BCVA {visit.bcva_logmar} logMAR</small></div></div>
            {MODALITIES.map(([key,label]) => <div className="visit-image" key={key}>
              <img src={visit.images[key]} alt={`${visit.label} ${key}`}/>
              <span>{label}</span><i/>
            </div>)}
            {index === 0 && <div className="visit-arrow"><ArrowRight size={17}/><small>{caseData.treatment.current_interval_weeks} WEEKS</small></div>}
          </div>)}
        </div>
      </section>

      <aside className="agent-launch">
        <div className="amd-panel-head"><span><BrainCircuit size={14}/> AGENT RUNTIME</span><b>REAL EXECUTION</b></div>
        <div className="model-core"><span><BrainCircuit size={30}/><i/><em/></span><b>QWEN2.5-VL</b><small>3B / BF16 / A800 GPU</small></div>
        <ul>
          <li><Check size={12}/>Six images enter the vision encoder</li>
          <li><Check size={12}/>Three imaging tools execute independently</li>
          <li><Check size={12}/>Guideline retrieval constrains evidence</li>
          <li><Check size={12}/>Rules select the final action</li>
        </ul>
        <button className="agent-run" disabled={loading || service.status !== 'ready'} onClick={runAgent}>
          {loading ? <LoaderCircle size={17}/> : <Play size={17} fill="currentColor"/>}
          <span>{loading ? 'AGENT RUNNING' : service.status === 'ready' ? 'RUN EVIDENCE AGENT' : 'WAITING FOR GPU MODEL'}<small>{loading ? `STEP ${progressStep+1} / ${STEPS.length}` : 'RUN REAL MLLM PIPELINE'}</small></span>
        </button>
        {loading && <div className="agent-progress"><i style={{width:`${Math.max(8,(progressStep+1)/STEPS.length*100)}%`}}/></div>}
        <p><AlertTriangle size={13}/>No template or random report is returned if the GPU model is unavailable.</p>
      </aside>
    </section> : <AgentResult result={result} onReset={() => setResult(null)}/>}

    {error && <div className="error-banner amd-error"><AlertTriangle size={16}/>{error}</div>}
  </div>;
}

function AgentResult({ result, onReset }) {
  const report = result.report || {};
  const visitTools = result.tool_results?.visits || {};
  return <motion.section className="agent-result" initial={{opacity:0,y:12}} animate={{opacity:1,y:0}}>
    <div className="decision-banner">
      <span className="decision-icon"><FileCheck2 size={24}/></span>
      <div><small>PROGRAMMATIC FINAL SELECTION</small><h2>{result.decision.action}</h2><p>{result.report?.recommended_plan}</p></div>
      <div className="decision-score"><small>CONFIDENCE</small><strong>{result.decision.confidence_score}</strong><span>{result.decision.verdict}</span></div>
      <button onClick={onReset}>BACK TO CASE</button>
    </div>

    <div className="result-layout">
      <div className="result-main">
        <section className="report-card">
          <div className="amd-panel-head"><span><BrainCircuit size={14}/> MLLM CLINICAL SYNTHESIS</span><b>QWEN2.5-VL / VERIFIED</b></div>
          <div className="report-sections">
            <article><small>01 / CASE SUMMARY</small><p>{report.case_summary}</p></article>
            <article><small>02 / IMAGING INTERPRETATION</small><p>{report.imaging_interpretation}</p></article>
            <article><small>03 / EVIDENCE INTEGRATION</small><p>{report.evidence_integration}</p></article>
            <article><small>04 / FOLLOW-UP PLAN</small><p>{report.followup_schedule}</p></article>
          </div>
          <div className="safety-triggers"><span><ShieldAlert size={16}/>ESCALATION TRIGGERS</span>{(report.safety_triggers || []).map((item,index)=><p key={index}>{item}</p>)}</div>
          {report.uncertainty && <p className="uncertainty"><AlertTriangle size={13}/>{report.uncertainty}</p>}
        </section>

        <section className="biomarker-card">
          <div className="amd-panel-head"><span><Activity size={14}/> QUANTITATIVE BIOMARKERS</span><b>INDEPENDENT TOOLS</b></div>
          <div className="biomarker-visits">
            {Object.entries(visitTools).map(([visit,data]) => <div key={visit}>
              <h3>{VISIT_LABELS[visit]}</h3>
              <div className="tool-images">
                {['oct','octa','fundus'].map(key => <img key={key} src={data[key].overlay} alt={`${visit} ${key} output`}/>)}
              </div>
              <dl>
                <div><dt>OCT THICKNESS PROXY</dt><dd>{data.oct.thickness_proxy_px} px</dd></div>
                <div><dt>OCTA VESSEL DENSITY</dt><dd>{data.octa.vessel_density_percent}%</dd></div>
                <div><dt>FUNDUS TOP CLASS</dt><dd>{data.fundus.summary}</dd></div>
              </dl>
            </div>)}
          </div>
        </section>

        <section className="options-card">
          <div className="amd-panel-head"><span><Waypoints size={14}/> ALL CANDIDATE OPTIONS</span><b>CONSTRAINED EVALUATION</b></div>
          {result.options.map((option,index) => <div className={`option-row ${index===0?'selected':''}`} key={option.id}>
            <span>{String(index+1).padStart(2,'0')}</span>
            <div><b>{option.title}</b><small>{option.evidence_ids.map(id => <SourceTag id={id} key={id}/>)}</small></div>
            <em>{option.verdict}</em><strong>{option.score}</strong>
          </div>)}
        </section>
      </div>

      <aside className="result-side">
        <section className="trace-card">
          <div className="amd-panel-head"><span><Activity size={14}/> TOOL TRACE</span><b>{(result.runtime_ms/1000).toFixed(1)} S</b></div>
          {result.tool_trace.map((item,index) => <div className="trace-row" key={item.tool}>
            <span><Check size={11}/></span><div><small>CALL // {String(index+1).padStart(2,'0')}</small><b>{item.tool.replaceAll('_',' ')}</b><em>{item.runtime_ms ? `${item.runtime_ms} ms` : item.selection || `${item.documents || item.options} items`}</em></div>
          </div>)}
          <p><Check size={13}/>REAL MLLM INFERENCE<br/><small>Fallback generation: OFF</small></p>
        </section>
        <section className="evidence-card">
          <div className="amd-panel-head"><span><BookOpen size={14}/> EVIDENCE RETRIEVAL</span><b>TOP {result.evidence.length}</b></div>
          {result.evidence.map(item => <a href={item.url} target="_blank" rel="noreferrer" key={item.id}>
            <span>{item.id}</span><div><b>{item.title}</b><small>{item.source} / {item.year}</small></div><ChevronRight size={13}/>
          </a>)}
        </section>
      </aside>
    </div>
    <p className="agent-notice"><ShieldAlert size={14}/>{result.notice}</p>
  </motion.section>;
}

export default AMDWorkspace;

