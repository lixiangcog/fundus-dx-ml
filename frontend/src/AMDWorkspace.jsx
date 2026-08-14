import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  Activity, AlertTriangle, ArrowRight, BookOpen, BrainCircuit, Check,
  ChevronRight, Clock3, Database, Eye, FileCheck2, ImageIcon, LoaderCircle,
  Play, ScanLine, Server, ShieldAlert, Stethoscope, Waypoints,
} from 'lucide-react';

const STEPS = [
  { id:'case', label:'病例整理', icon:Clock3 },
  { id:'tools', label:'影像分析', icon:ScanLine },
  { id:'vision', label:'综合复核', icon:Eye },
  { id:'rag', label:'证据检索', icon:Database },
  { id:'decision', label:'方案评估', icon:Waypoints },
  { id:'report', label:'生成报告', icon:FileCheck2 },
];

const VISIT_LABELS = { baseline:'V0 / 基线', followup:'V1 / 随访' };
const MODALITIES = [
  ['oct','结构 OCT'], ['octa','OCTA'], ['fundus','眼底彩照'],
];

function StatusBadge({ service }) {
  const ready = service?.status === 'ready';
  return <div className={`agent-service ${ready ? 'ready' : 'waiting'}`}>
    <span><Server size={14}/><i/></span>
    <div><small>分析服务</small><b>{ready ? '已就绪' : service?.status === 'offline' ? '离线' : '检查中'}</b></div>
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
    fetch(`${apiUrl}/amd-agent/config`).then(r => r.json()).then((data) => { setConfig(data); setService(data.service); }).catch(() => setError('AMD 随访功能暂时不可用。'));
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
      setError(requestError.response?.data?.detail || '分析未完成，请检查服务状态后重试。');
      refreshStatus();
    } finally { setLoading(false); }
  };

  return <div className="amd-workspace">
    <section className="amd-hero">
      <div>
        <span className="eyebrow">纵向 AMD 随访</span>
        <h1>多模态影像对比，<em>辅助随访决策。</em></h1>
        <p>综合两次就诊的 OCT、OCTA、眼底彩照和视力变化，生成可追溯的随访建议。</p>
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
          <div><small>步骤 {String(index+1).padStart(2,'0')}</small><b>{step.label}</b></div>
          {index < STEPS.length-1 && <i><em/></i>}
        </div>;
      })}
    </section>

    {!result ? <section className="amd-console">
      <aside className="case-context">
        <div className="amd-panel-head"><span><Stethoscope size={14}/> 病例信息</span><b>论文去标识病例</b></div>
        {caseData ? <>
          <h2>{caseData.case_id}</h2>
          <p className="case-title">{caseData.title}</p>
          <dl className="case-facts">
            <div><dt>患者</dt><dd>{caseData.patient.age} 岁 / {caseData.patient.sex} / {caseData.patient.eye}</dd></div>
            <div><dt>诊断</dt><dd>{caseData.patient.diagnosis}</dd></div>
            <div><dt>治疗</dt><dd>{caseData.treatment.agent} / {caseData.treatment.injections} 次</dd></div>
            <div><dt>治疗间隔</dt><dd>{caseData.treatment.current_interval_weeks}</dd></div>
          </dl>
          <p className="case-narrative">{caseData.context}</p>
        </> : <div className="case-loading"><LoaderCircle size={20}/>正在加载病例</div>}
        <div className="case-safety"><ShieldAlert size={15}/><span><b>影像需复核</b>默认图像来自论文缩略图，仅用于流程演示；论文指标与本地分析结果分开显示。</span></div>
      </aside>

      <section className="visit-matrix">
        <div className="amd-panel-head"><span><ImageIcon size={14}/> 多模态随访影像</span><b>2 次就诊 · 6 张影像</b></div>
        <div className="visit-grid">
          {(caseData?.visits || []).map((visit,index) => <div className="visit-column" key={visit.id}>
            <div className="visit-title"><span>{visit.id}</span><div><b>{visit.label}</b><small>{visit.date} / 视力 {visit.bcva_decimal}</small></div></div>
            {MODALITIES.map(([key,label]) => <div className="visit-image" key={key}>
              <img src={visit.images[key]} alt={`${visit.label} ${key}`}/>
              <span>{label}</span><i/>
            </div>)}
            {index === 0 && <div className="visit-arrow"><ArrowRight size={17}/><small>3 个月</small></div>}
          </div>)}
        </div>
      </section>

      <aside className="agent-launch">
        <div className="amd-panel-head"><span><BrainCircuit size={14}/> 分析流程</span><b>真实推理</b></div>
        <div className="model-core"><span><BrainCircuit size={30}/><i/><em/></span><b>多模态分析</b><small>纵向影像与循证信息综合</small></div>
        <ul>
          <li><Check size={12}/>对比两次就诊的六张影像</li>
          <li><Check size={12}/>复核两次眼底彩照变化</li>
          <li><Check size={12}/>提取影像定量指标</li>
          <li><Check size={12}/>检索相关随访证据</li>
          <li><Check size={12}/>评估并选择随访方案</li>
        </ul>
        <button className="agent-run" disabled={loading || service.status !== 'ready'} onClick={runAgent}>
          {loading ? <LoaderCircle size={17}/> : <Play size={17} fill="currentColor"/>}
          <span>{loading ? '正在分析' : service.status === 'ready' ? '开始随访分析' : '等待分析服务'}<small>{loading ? `步骤 ${progressStep+1} / ${STEPS.length}` : '预计约 40 秒'}</small></span>
        </button>
        {loading && <div className="agent-progress"><i style={{width:`${Math.max(8,(progressStep+1)/STEPS.length*100)}%`}}/></div>}
        <p><AlertTriangle size={13}/>服务不可用时不会返回模板或随机报告。</p>
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
      <div><small>随访建议</small><h2>{result.decision.action}</h2><p>{result.report?.recommended_plan}</p></div>
      <div className="decision-score"><small>可信度</small><strong>{result.decision.confidence_score}</strong><span>{result.decision.verdict}</span></div>
      <button onClick={onReset}>返回病例</button>
    </div>

    <div className="result-layout">
      <div className="result-main">
        <section className="report-card">
          <div className="amd-panel-head"><span><BrainCircuit size={14}/> 综合分析报告</span><b>已完成</b></div>
          <div className="report-sections">
            <article><small>01 / 病例摘要</small><p>{report.case_summary}</p></article>
            <article><small>02 / 影像解读</small><p>{report.imaging_interpretation}</p></article>
            <article><small>03 / 证据综合</small><p>{report.evidence_integration}</p></article>
            <article><small>04 / 随访安排</small><p>{report.followup_schedule}</p></article>
          </div>
          <div className="safety-triggers"><span><ShieldAlert size={16}/>需要升级处理的情况</span>{(report.safety_triggers || []).map((item,index)=><p key={index}>{item}</p>)}</div>
          {report.uncertainty && <p className="uncertainty"><AlertTriangle size={13}/>{report.uncertainty}</p>}
        </section>

        <section className="biomarker-card">
          <div className="amd-panel-head"><span><Activity size={14}/> 本地定量结果</span><b>辅助参考</b></div>
          <div className="biomarker-visits">
            {Object.entries(visitTools).map(([visit,data]) => <div key={visit}>
              <h3>{VISIT_LABELS[visit]}</h3>
              <div className="tool-images">
                {['oct','octa','fundus'].map(key => <img key={key} src={data[key].overlay} alt={`${visit} ${key} output`}/>)}
              </div>
              <dl>
                <div><dt>OCT 厚度代理</dt><dd>{data.oct.thickness_proxy_px} px</dd></div>
                <div><dt>OCTA 血管密度</dt><dd>{data.octa.vessel_density_percent}%</dd></div>
                <div><dt>眼底筛查结果</dt><dd>{data.fundus.summary}</dd></div>
              </dl>
            </div>)}
          </div>
        </section>

        {result.reported_reference_biomarkers && <section className="biomarker-card reported-biomarkers">
          <div className="amd-panel-head"><span><BookOpen size={14}/> 论文报告指标</span><b>未在本地重算</b></div>
          <div className="reported-grid">
            <article><small>最佳矫正视力</small><strong>{result.reported_reference_biomarkers.bcva_decimal[0]} → {result.reported_reference_biomarkers.bcva_decimal[1]}</strong></article>
            <article><small>OCT 病灶面积</small><strong>{result.reported_reference_biomarkers.oct.candidate_lesion_area_mm2[0]} → {result.reported_reference_biomarkers.oct.candidate_lesion_area_mm2[1]} mm²</strong></article>
            <article><small>OCT 最大高度</small><strong>{result.reported_reference_biomarkers.oct.maximum_lesion_height_um[0]} → {result.reported_reference_biomarkers.oct.maximum_lesion_height_um[1]} μm</strong></article>
            <article><small>OCTA CNV 面积</small><strong>{result.reported_reference_biomarkers.octa.cnv_candidate_area_mm2[0]} → {result.reported_reference_biomarkers.octa.cnv_candidate_area_mm2[1]} mm²</strong></article>
            <article><small>眼底病灶面积</small><strong>{result.reported_reference_biomarkers.fundus.candidate_lesion_area_mm2[0]} → {result.reported_reference_biomarkers.fundus.candidate_lesion_area_mm2[1]} mm²</strong></article>
          </div>
          <p className="quality-warning"><AlertTriangle size={13}/>{result.case_quality?.label} · {result.case_quality?.reason}</p>
        </section>}

        <section className="options-card">
          <div className="amd-panel-head"><span><Waypoints size={14}/> 候选随访方案</span><b>综合评估</b></div>
          {result.options.map((option,index) => <div className={`option-row ${index===0?'selected':''}`} key={option.id}>
            <span>{String(index+1).padStart(2,'0')}</span>
            <div><b>{option.title}</b><small>{option.evidence_ids.map(id => <SourceTag id={id} key={id}/>)}</small></div>
            <em>{option.verdict}</em><strong>{option.score}</strong>
          </div>)}
        </section>
      </div>

      <aside className="result-side">
        <section className="trace-card">
          <div className="amd-panel-head"><span><Activity size={14}/> 分析记录</span><b>{(result.runtime_ms/1000).toFixed(1)} 秒</b></div>
          {result.tool_trace.map((item,index) => <div className="trace-row" key={item.tool}>
            <span><Check size={11}/></span><div><small>步骤 {String(index+1).padStart(2,'0')}</small><b>{STEPS[index]?.label || '完成'}</b><em>{item.runtime_ms ? `${(item.runtime_ms/1000).toFixed(1)} 秒` : '已完成'}</em></div>
          </div>)}
          <p><Check size={13}/>真实推理已完成</p>
        </section>
        <section className="evidence-card">
          <div className="amd-panel-head"><span><BookOpen size={14}/> 参考证据</span><b>{result.evidence.length} 条</b></div>
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
