import { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  Activity, AlertTriangle, ArrowRight, BookOpen, BrainCircuit, Check,
  ChevronRight, Clock3, Database, Eye, FileCheck2, ImageIcon, LoaderCircle,
  Play, ScanLine, Server, ShieldAlert, Stethoscope, Waypoints,
} from 'lucide-react';
import './amd-results.css';

const STEPS = [
  { id:'case', label:'病例整理', icon:Clock3 },
  { id:'tools', label:'影像分析', icon:ScanLine },
  { id:'vision', label:'综合复核', icon:Eye },
  { id:'rag', label:'依据核验', icon:Database },
  { id:'decision', label:'方案评估', icon:Waypoints },
  { id:'report', label:'生成报告', icon:FileCheck2 },
];

const VISIT_LABELS = { baseline:'V0 / 基线', followup:'V1 / 随访' };
const MODALITIES = [
  ['oct','结构 OCT'], ['octa','OCTA'], ['fundus','眼底彩照'],
];

const DELTA_METRICS = [
  ['oct_fluid_area_percent','OCT 液体面积','%'],
  ['oct_fluid_ratio_points','OCT 液体占比',' 个百分点'],
  ['octa_vessel_density_points','OCTA 血管密度',' 个百分点'],
  ['octa_skeleton_length_percent','OCTA 血管骨架','%'],
  ['octa_central_avascular_area_percent','中央无血管候选区','%'],
  ['fundus_lesion_ratio_points','彩照病灶占比',' 个百分点'],
  ['fundus_hemorrhage_area_percent','彩照出血面积','%'],
  ['amd_probability_points','AMD 筛查概率',' 个百分点'],
];

function showNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits });
}

function showDelta(value, unit) {
  if (value === null || value === undefined) return '基线为 0';
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${showNumber(numeric, 3)}${unit}`;
}

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
        <p>对两次就诊影像进行分割与定量比较，生成可追溯的随访建议和结构化操作规划。</p>
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
        <div className="amd-panel-head"><span><Stethoscope size={14}/> 病例信息</span><b>内置示例病例</b></div>
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
        <div className="case-safety"><ShieldAlert size={15}/><span><b>影像需复核</b>当前示例影像分辨率有限；历史记录与系统分析结果分开显示。</span></div>
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
        <div className="model-core"><span><BrainCircuit size={30}/><i/><em/></span><b>多模态分析</b><small>纵向影像与病历信息综合</small></div>
        <ul>
          <li><Check size={12}/>对比两次就诊的六张影像</li>
          <li><Check size={12}/>复核两次眼底彩照变化</li>
          <li><Check size={12}/>分割 OCT 液体、OCTA 血管与彩照病灶</li>
          <li><Check size={12}/>比较两次就诊的定量变化</li>
          <li><Check size={12}/>核验相关随访依据</li>
          <li><Check size={12}/>生成随访建议与操作规划</li>
        </ul>
        <button className="agent-run" disabled={loading || service.status !== 'ready'} onClick={runAgent}>
          {loading ? <LoaderCircle size={17}/> : <Play size={17} fill="currentColor"/>}
          <span>{loading ? '正在分析' : service.status === 'ready' ? '开始随访分析' : '等待分析服务'}<small>{loading ? `步骤 ${progressStep+1} / ${STEPS.length}` : '预计约 60 秒'}</small></span>
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
  const structured = report.structured_summary || {};
  const recommendation = report.recommendation || {};
  const procedure = report.procedure_plan || {};
  const visitTools = result.tool_results?.visits || {};
  const deltas = result.tool_results?.deltas || {};
  return <motion.section className="agent-result" initial={{opacity:0,y:12}} animate={{opacity:1,y:0}}>
    <div className="decision-banner">
      <span className="decision-icon"><FileCheck2 size={24}/></span>
      <div><small>随访建议</small><h2>{result.decision.action}</h2><p>{structured.treatment_response || report.recommended_plan}</p></div>
      <div className="decision-score"><small>可信度</small><strong>{result.decision.confidence_score}</strong><span>{result.decision.verdict}</span></div>
      <button onClick={onReset}>返回病例</button>
    </div>

    <div className="result-layout">
      <div className="result-main">
        <section className="report-card structured-report-card">
          <div className="amd-panel-head"><span><BrainCircuit size={14}/> 结构化随访报告</span><b>已完成</b></div>
          <div className="structured-summary-grid">
            <article className="summary-wide"><small>病例概览</small><p>{structured.case_overview || report.case_summary}</p></article>
            <article><small>当前状态</small><strong>{structured.disease_state || result.clinical_state?.activity}</strong></article>
            <article><small>治疗反应</small><p>{structured.treatment_response || report.treatment_response}</p></article>
            <article><small>影像综合</small><p>{structured.imaging_interpretation || report.imaging_interpretation}</p></article>
            <article><small>量化变化</small><p>{structured.quantitative_change || report.quantitative_change}</p></article>
            <article><small>随访安排</small><p>{recommendation.followup_schedule || report.followup_schedule}</p></article>
            <article><small>决策依据</small><p>{recommendation.evidence_integration || report.evidence_integration}</p></article>
          </div>
          <div className="safety-triggers"><span><ShieldAlert size={16}/>需要升级处理的情况</span>{(report.safety_triggers || []).map((item,index)=><p key={index}>{item}</p>)}</div>
          {report.uncertainty && <p className="uncertainty"><AlertTriangle size={13}/>{report.uncertainty}</p>}
        </section>

        <section className="segmentation-card">
          <div className="amd-panel-head"><span><ScanLine size={14}/> 分割、病灶定位与定量结果</span><b>真实推理</b></div>
          <div className="segmentation-visits">
            {Object.entries(visitTools).map(([visit,data]) => {
              const tiles = [
                ['OCT 层结构',data.oct.structure_overlay || data.oct.overlay],
                ['OCT 液体',data.oct.fluid_overlay],
                ['OCTA 血管',data.octa.overlay],
                ['彩照病灶',data.fundus.lesion_overlay || data.fundus.overlay],
              ];
              const metrics = [
                ['液体面积',data.oct.fluid_area_px,'px'],
                ['液体占比',data.oct.fluid_ratio_percent,'%'],
                ['液体最大高度',data.oct.max_fluid_height_px,'px'],
                ['血管密度',data.octa.vessel_density_percent,'%'],
                ['血管骨架',data.octa.skeleton_length_px,'px'],
                ['分支点',data.octa.branch_points,'个'],
                ['中央无血管候选区',data.octa.central_avascular_area_px2,'px²'],
                ['彩照病灶占比',data.fundus.lesion_ratio_percent,'%'],
                ['彩照出血面积',data.fundus.hemorrhage_area_px,'px'],
              ];
              return <article className="visit-segmentation" key={visit}>
                <header><span>{visit === 'baseline' ? 'V0' : 'V1'}</span><div><strong>{VISIT_LABELS[visit]}</strong><small>{visit === 'baseline' ? '治疗前基线' : '治疗后随访'}</small></div></header>
                <div className="segmentation-gallery">{tiles.map(([label,image]) => <figure key={label}><img src={image} alt={`${VISIT_LABELS[visit]} ${label}`}/><figcaption>{label}</figcaption></figure>)}</div>
                <div className="visit-quant-grid">{metrics.map(([label,value,unit]) => <div key={label}><small>{label}</small><strong>{showNumber(value)}<em>{unit}</em></strong></div>)}</div>
              </article>;
            })}
          </div>
          <div className="longitudinal-deltas">
            <div className="delta-heading"><span><ArrowRight size={15}/>基线至随访变化</span><small>正值表示增加，负值表示减少</small></div>
            <div>{DELTA_METRICS.map(([key,label,unit]) => <article key={key} className={Number(deltas[key]) > 0 ? 'increase' : Number(deltas[key]) < 0 ? 'decrease' : ''}><small>{label}</small><strong>{showDelta(deltas[key],unit)}</strong></article>)}</div>
          </div>
          <p className="segmentation-notice"><AlertTriangle size={13}/>{result.case_quality?.label}。像素指标用于同协议纵向比较，不等同于设备原生 μm、mm² 或体积测量。</p>
        </section>

        <section className="procedure-card">
          <div className="amd-panel-head"><span><Stethoscope size={14}/> 操作 / 手术规划</span><b>{procedure.status || '待专科确认'}</b></div>
          <div className="procedure-status"><span><FileCheck2 size={20}/></span><div><small>{procedure.title || 'AMD 操作规划'}</small><strong>{procedure.procedure_overview?.candidate_route}</strong></div></div>
          <p className="planning-rationale">{procedure.planning_rationale}</p>
          <div className="procedure-overview">
            <article><small>术眼</small><strong>{procedure.procedure_overview?.laterality}</strong></article>
            <article><small>目标</small><strong>{procedure.procedure_overview?.target}</strong></article>
            <article><small>时机</small><strong>{procedure.procedure_overview?.timing}</strong></article>
            <article><small>药物与剂量</small><strong>{procedure.procedure_overview?.drug_and_dose}</strong></article>
          </div>
          <div className="procedure-checklists">
            {[
              ['患者与影像要点',procedure.patient_specific_considerations],
              ['操作前核查',procedure.preoperative_checks],
              ['操作中要点',procedure.intraoperative_plan],
              ['操作后监测',procedure.postoperative_monitoring],
              ['升级与替代路径',procedure.escalation_and_alternatives],
              ['必须由专科决定',procedure.required_specialist_decisions],
            ].map(([title,items],groupIndex) => <section key={title}>
              <h3><span>{String(groupIndex+1).padStart(2,'0')}</span>{title}</h3>
              <ul>{(items || []).map((item,index)=><li key={index}><Check size={11}/><span>{item}</span></li>)}</ul>
            </section>)}
          </div>
          <p className="procedure-notice"><ShieldAlert size={14}/>{procedure.research_notice || '规划需由有资质的视网膜专科医生确认后方可执行。'}</p>
        </section>

        {result.reported_reference_biomarkers && <section className="biomarker-card reported-biomarkers">
          <div className="amd-panel-head"><span><BookOpen size={14}/> 历史随访指标</span><b>与系统结果分列</b></div>
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
          <div className="amd-panel-head"><span><BookOpen size={14}/> 决策依据</span><b>{result.evidence.length} 条</b></div>
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
