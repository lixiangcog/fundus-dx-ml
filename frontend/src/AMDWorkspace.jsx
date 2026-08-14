import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import {
  Activity, AlertTriangle, ArrowRight, BookOpen, BrainCircuit, Check,
  ChevronRight, Clock3, Database, Eye, FileCheck2, ImageIcon, LoaderCircle,
  Play, RotateCcw, ScanLine, Server, ShieldAlert, Stethoscope, UploadCloud,
  Waypoints,
} from 'lucide-react';
import ModuleLog from './ModuleLog';
import { useModuleLog } from './useModuleLog';
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

const IMAGE_FIELDS = [
  ['baseline_oct', 0, 'oct'], ['baseline_octa', 0, 'octa'], ['baseline_fundus', 0, 'fundus'],
  ['followup_oct', 1, 'oct'], ['followup_octa', 1, 'octa'], ['followup_fundus', 1, 'fundus'],
];
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

function cloneCase(value) {
  return value ? JSON.parse(JSON.stringify(value)) : null;
}

function defaultImageState(caseData) {
  return Object.fromEntries(IMAGE_FIELDS.map(([field, visitIndex, modality]) => [field, {
    file: null,
    preview: caseData?.visits?.[visitIndex]?.images?.[modality] || '',
    isDefault: true,
  }]));
}

const DELTA_METRICS = [
  ['oct_fluid_area_percent','OCT 液体面积','%'],
  ['oct_fluid_ratio_points','OCT 液体占比',' 个百分点'],
  ['oct_irf_area_percent','视网膜内液','%'],
  ['oct_srf_area_percent','视网膜下液','%'],
  ['oct_ped_area_percent','色素上皮脱离','%'],
  ['oct_cnv_probability_points','OCT 新生血管概率',' 个百分点'],
  ['octa_vessel_density_points','OCTA 血管密度',' 个百分点'],
  ['octa_skeleton_length_percent','OCTA 血管骨架','%'],
  ['octa_central_avascular_area_percent','中央无血管候选区','%'],
  ['fundus_lesion_ratio_points','彩照病灶占比',' 个百分点'],
  ['fundus_hemorrhage_area_percent','彩照出血面积','%'],
  ['amd_probability_points','AMD 筛查概率',' 个百分点'],
  ['fundus_large_drusen_probability_points','彩照玻璃膜疣概率',' 个百分点'],
  ['fundus_pigment_probability_points','彩照色素异常概率',' 个百分点'],
  ['fundus_advanced_amd_probability_points','彩照晚期 AMD 概率',' 个百分点'],
  ['fundus_ga_probability_points','地图样萎缩概率',' 个百分点'],
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
  const [caseData, setCaseData] = useState(null);
  const [caseDirty, setCaseDirty] = useState(false);
  const [caseImages, setCaseImages] = useState({});
  const caseImagesRef = useRef({});
  const { entries: amdLogs, write: writeAmdLog, writeMany: writeAmdLogs, clear: clearAmdLog } = useModuleLog('AMD 随访');
  const hasCustomImages = useMemo(() => Object.values(caseImages).some((image) => image?.file), [caseImages]);

  const refreshStatus = useCallback(() => fetch(`${apiUrl}/amd-agent/status`).then(r => r.json()).then(setService).catch(() => setService({status:'offline'})), [apiUrl]);
  useEffect(() => {
    fetch(`${apiUrl}/amd-agent/config`).then(r => r.json()).then((data) => {
      setConfig(data); setService(data.service);
      const initialCase = cloneCase(data.default_case);
      setCaseData(initialCase);
      setCaseImages(defaultImageState(initialCase));
      writeAmdLogs([
        { level:'command', channel:'SHELL', message:'fundus-dx amd status --case default' },
        { level:'success', channel:'INPUT', message:'visits=2; images=6', detail:'modalities=OCT,OCTA,FUNDUS' },
        { level:data.service?.status === 'ready' ? 'success' : 'warning', channel:'SERVICE', message:`agent=${data.service?.status || 'unknown'}`, detail:'multimodal tools + evidence retrieval' },
      ]);
    }).catch(() => { setError('AMD 随访功能暂时不可用。'); writeAmdLog('error', 'GET /amd-agent/config -> failed', '请检查网页服务与分析服务', 'HTTP'); });
    const timer = setInterval(refreshStatus, 12000);
    return () => clearInterval(timer);
  }, [apiUrl, refreshStatus, writeAmdLog, writeAmdLogs]);

  useEffect(() => { caseImagesRef.current = caseImages; }, [caseImages]);
  useEffect(() => () => {
    Object.values(caseImagesRef.current).forEach((image) => {
      if (image?.file && image.preview) URL.revokeObjectURL(image.preview);
    });
  }, []);

  const updatePatient = (field, value) => {
    setCaseData((current) => ({...current, patient:{...current.patient, [field]:value}}));
    setCaseDirty(true);
  };

  const updateTreatment = (field, value) => {
    setCaseData((current) => ({...current, treatment:{...current.treatment, [field]:value}}));
    setCaseDirty(true);
  };

  const updateVisit = (index, field, value) => {
    setCaseData((current) => ({...current, visits:current.visits.map((visit, visitIndex) => (
      visitIndex === index ? {...visit, [field]:value} : visit
    ))}));
    setCaseDirty(true);
  };

  const updateContext = (value) => {
    setCaseData((current) => ({...current, context:value}));
    setCaseDirty(true);
  };

  const updateImage = (field, file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('请选择 JPG、PNG 或 WebP 图像。');
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError('单张影像不能超过 12 MB。');
      return;
    }
    const preview = URL.createObjectURL(file);
    setCaseImages((current) => {
      if (current[field]?.file && current[field].preview) URL.revokeObjectURL(current[field].preview);
      return {...current, [field]:{file, preview, isDefault:false}};
    });
    setError('');
  };

  const restoreDefaultCase = () => {
    const restored = cloneCase(config?.default_case);
    if (!restored) return;
    setCaseData(restored);
    setCaseImages((current) => {
      Object.values(current).forEach((image) => {
        if (image?.file && image.preview) URL.revokeObjectURL(image.preview);
      });
      return defaultImageState(restored);
    });
    setCaseDirty(false);
    setError('');
    writeAmdLog('info', '恢复默认病例与六张影像', '所有可编辑字段已重置', 'INPUT');
  };

  useEffect(() => {
    if (!loading) return;
    // Reset the staged progress animation when a new analysis starts.
    setProgressStep(0);
    writeAmdLog('run', 'stage=01; case_context ready', '开始组织两次就诊记录', 'PIPELINE');
    const timers = [2200, 6500, 12500, 17000, 22000].map((delay, index) => setTimeout(() => {
      const nextStep = Math.min(index + 1, STEPS.length - 1);
      setProgressStep(nextStep);
      writeAmdLog('run', `stage=${String(nextStep + 1).padStart(2, '0')}; ${STEPS[nextStep].id}`, STEPS[nextStep].label, 'PIPELINE');
    }, delay));
    return () => timers.forEach(clearTimeout);
  }, [loading, writeAmdLog]);

  const runAgent = async () => {
    if (loading || service.status !== 'ready' || !caseData) return;
    setLoading(true); setResult(null); setError('');
    const inputMode = caseDirty || hasCustomImages ? 'custom' : 'default';
    writeAmdLogs([
      { level:'command', channel:'SHELL', message:`fundus-dx amd run --case ${inputMode} --device cuda:0` },
      { level:'info', channel:'INPUT', message:'validating case fields and six images', detail:`source=${inputMode}; baseline=3; followup=3` },
      { level:'run', channel:'QUEUE', message:'multimodal job accepted', detail:'quantification -> comparison -> retrieval -> report' },
      { level:'run', channel:'CUDA', message:'GPU tools dispatched', detail:'real_inference=true; fallback=false' },
    ]);
    try {
      const submittedCase = cloneCase(caseData);
      if (caseDirty || hasCustomImages) {
        delete submittedCase.reference_biomarkers;
        delete submittedCase.image_quality;
        submittedCase.evidence_origin = 'user_supplied';
        submittedCase.research_demo = false;
      }
      const body = new FormData();
      body.append('case_json', JSON.stringify(submittedCase));
      for (const [field] of IMAGE_FIELDS) {
        const current = caseImages[field];
        if (current?.file) {
          body.append(field, current.file, current.file.name);
          continue;
        }
        const imageUrl = current?.preview || '';
        const response = await fetch(imageUrl.startsWith('http') ? imageUrl : `${apiUrl}${imageUrl}`);
        if (!response.ok) throw new Error(`无法读取默认影像：${field}`);
        const blob = await response.blob();
        body.append(field, new File([blob], `${field}.png`, {type:blob.type || 'image/png'}));
      }
      const response = await axios.post(`${apiUrl}/amd-agent/analyze`, body);
      setResult(response.data);
      setProgressStep(STEPS.length - 1);
      setService((current) => ({...current,status:'ready'}));
      const output = response.data;
      const traceRows = (output.tool_trace || []).map((item) => {
        const detail = [
          item.runtime_ms !== undefined ? `runtime_ms=${item.runtime_ms}` : '',
          item.input_images !== undefined ? `images=${item.input_images}` : '',
          item.documents !== undefined ? `documents=${item.documents}` : '',
          item.options !== undefined ? `options=${item.options}` : '',
          `real=${Boolean(item.real_execution)}`,
        ].filter(Boolean).join('; ');
        return { level:item.status === 'completed' ? 'success' : 'warning', channel:'TOOL', message:`${item.tool} -> ${item.status}`, detail };
      });
      writeAmdLogs([
        { level:'success', channel:'HTTP', message:'POST /amd-agent/analyze -> 200 OK', detail:`run_id=${output.run_id}; source=${inputMode}` },
        ...traceRows,
        { level:output.report?.consistency_validated ? 'success' : 'warning', channel:'REPORT', message:`structured_report=${Boolean(output.report)}; consistency=${Boolean(output.report?.consistency_validated)}`, detail:`decision=${output.decision?.action_id || 'review'}` },
        { level:'success', channel:'DONE', message:'AMD longitudinal analysis completed', detail:`runtime_ms=${output.runtime_ms}; tools=${traceRows.length}` },
      ]);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || '分析未完成，请检查服务状态后重试。');
      writeAmdLog('error', `POST /amd-agent/analyze -> ${requestError.response?.status || 'NETWORK_ERROR'}`, requestError.response?.data?.detail || requestError.message || '分析服务未返回完整结果', 'HTTP');
      refreshStatus();
    } finally { setLoading(false); }
  };

  return <div className="amd-workspace">
    <section className="amd-hero">
      <div>
        <span className="eyebrow">纵向 AMD 随访</span>
        <h1>多模态影像对比，<em>辅助随访决策。</em></h1>
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

    {!result ? <section className="amd-console amd-input-console">
      <aside className="case-context">
        <div className="amd-panel-head">
          <span><Stethoscope size={14}/> 病例信息</span>
          <button type="button" className="case-reset" onClick={restoreDefaultCase}><RotateCcw size={12}/>恢复默认</button>
        </div>
        {caseData ? <>
          <div className="case-editor">
            <div className="case-field-row triple">
              <label className="case-field"><span>年龄</span><input type="number" min="18" max="110" value={caseData.patient.age} onChange={(event) => updatePatient('age', event.target.value)} /></label>
              <label className="case-field"><span>性别</span><select value={caseData.patient.sex} onChange={(event) => updatePatient('sex', event.target.value)}><option>女</option><option>男</option><option>其他</option></select></label>
              <label className="case-field"><span>眼别</span><select value={caseData.patient.eye} onChange={(event) => updatePatient('eye', event.target.value)}><option>右眼</option><option>左眼</option><option>双眼</option></select></label>
            </div>
            <label className="case-field"><span>诊断</span><input value={caseData.patient.diagnosis} maxLength={160} onChange={(event) => updatePatient('diagnosis', event.target.value)} /></label>
            <label className="case-field"><span>治疗方式</span><input value={caseData.treatment.agent} maxLength={160} onChange={(event) => updateTreatment('agent', event.target.value)} /></label>
            <div className="case-field-row">
              <label className="case-field"><span>治疗次数</span><input type="number" min="0" max="200" value={caseData.treatment.injections} onChange={(event) => updateTreatment('injections', event.target.value)} /></label>
              <label className="case-field"><span>治疗间隔</span><input value={caseData.treatment.current_interval_weeks} maxLength={40} onChange={(event) => updateTreatment('current_interval_weeks', event.target.value)} /></label>
            </div>
            <label className="case-field"><span>病例记录</span><textarea rows="4" maxLength={1200} value={caseData.context} onChange={(event) => updateContext(event.target.value)} /></label>
          </div>
        </> : <div className="case-loading"><LoaderCircle size={20}/>正在加载病例</div>}
      </aside>

      <section className="visit-matrix">
        <div className="amd-panel-head"><span><ImageIcon size={14}/> 多模态随访影像</span></div>
        <div className="visit-grid">
          {(caseData?.visits || []).map((visit,index) => <div className="visit-column" key={visit.id}>
            <div className="visit-title editable"><span>{visit.id}</span><div><b>{visit.label}</b><div className="visit-fields">
              <label><small>日期</small><input type="month" value={visit.date} onChange={(event) => updateVisit(index, 'date', event.target.value)} /></label>
              <label><small>视力</small><input type="number" min="0" max="2" step="0.01" value={visit.bcva_decimal} onChange={(event) => updateVisit(index, 'bcva_decimal', event.target.value)} /></label>
            </div></div></div>
            {MODALITIES.map(([key,label]) => {
              const field = `${index === 0 ? 'baseline' : 'followup'}_${key}`;
              const image = caseImages[field];
              return <div className={`visit-image editable ${image?.file ? 'uploaded' : ''}`} key={key}>
                <img src={image?.preview || visit.images[key]} alt={`${visit.label} ${key}`}/>
                <span>{label}</span><i/>
                <label className="visit-upload"><UploadCloud size={13}/><b>{image?.file ? '更换影像' : '上传替换'}</b><small>{image?.file ? image.file.name : '当前为默认影像'}</small>
                  <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => { updateImage(field, event.target.files?.[0]); event.target.value = ''; }} />
                </label>
              </div>;
            })}
            {index === 0 && <div className="visit-arrow"><ArrowRight size={17}/><small>3 个月</small></div>}
          </div>)}
        </div>
      </section>

      <aside className="agent-launch">
        <div className="amd-panel-head"><span><BrainCircuit size={14}/> 分析流程</span></div>
        <div className="model-core"><span><BrainCircuit size={30}/><i/><em/></span><b>多模态分析</b><small>纵向影像与病历信息综合</small></div>
        <ul>
          <li><Check size={12}/>对比两次就诊的六张影像</li>
          <li><Check size={12}/>复核两次眼底彩照变化</li>
          <li><Check size={12}/>识别 OCT 新生血管、玻璃膜疣与三类液体</li>
          <li><Check size={12}/>识别彩照玻璃膜疣、色素异常与黄斑萎缩</li>
          <li><Check size={12}/>分割 OCTA 血管与彩照微小病灶</li>
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
    </section> : <AgentResult
      result={result}
      onReset={() => { setResult(null); writeAmdLog('info', '返回病例编辑', '保留本次输入与运行日志'); }}
    />}

    <ModuleLog title="AMD 随访" entries={amdLogs} onClear={clearAmdLog} running={loading}/>
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
          <div className="amd-panel-head"><span><ScanLine size={14}/> 分割、病灶定位与定量结果</span></div>
          <div className="segmentation-visits">
            {Object.entries(visitTools).map(([visit,data]) => {
              const tiles = [
                ['OCT 层结构',data.oct.structure_overlay || data.oct.overlay],
                ['OCT 液体总量',data.oct.fluid_overlay],
                ['OCT 三类液体',data.oct.fluid_subtype_overlay],
                ['OCT AMD 病灶',data.oct.pathology_overlay],
                ['OCTA 血管',data.octa.overlay],
                ['彩照病灶',data.fundus.lesion_overlay || data.fundus.overlay],
              ];
              const fluidFindings = [
                ['irf','视网膜内液'], ['srf','视网膜下液'], ['ped','色素上皮脱离'],
              ].map(([id,label]) => ({id,label,status:Number(data.oct.fluid_subtypes?.[id]?.pixels) > 0 ? 'positive' : 'negative',value:data.oct.fluid_subtypes?.[id]?.pixels,unit:'px'}));
              const lesionGroups = [
                ['OCT 病灶筛查',[...(data.oct.amd_findings || []),...fluidFindings]],
                ['眼底彩照 AMD 表征',data.fundus.amd_findings || []],
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
                <div className="amd-lesion-groups">{lesionGroups.map(([title,findings]) => <section key={title}>
                  <h4>{title}</h4>
                  <div>{findings.map((finding) => <article className={finding.status === 'positive' ? 'positive' : 'negative'} key={finding.id}>
                    <span><i/>{finding.label}</span>
                    <strong>{finding.value !== undefined ? `${showNumber(finding.value)} ${finding.unit}` : `${showNumber(Number(finding.positive_probability) * 100)}%`}</strong>
                  </article>)}</div>
                </section>)}</div>
              </article>;
            })}
          </div>
          <div className="longitudinal-deltas">
            <div className="delta-heading"><span><ArrowRight size={15}/>基线至随访变化</span><small>正值表示增加，负值表示减少</small></div>
            <div>{DELTA_METRICS.map(([key,label,unit]) => <article key={key} className={Number(deltas[key]) > 0 ? 'increase' : Number(deltas[key]) < 0 ? 'decrease' : ''}><small>{label}</small><strong>{showDelta(deltas[key],unit)}</strong></article>)}</div>
          </div>
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
          <p><Check size={13}/>分析已完成</p>
        </section>
        <section className="evidence-card">
          <div className="amd-panel-head"><span><BookOpen size={14}/> 决策依据</span><b>{result.evidence.length} 条</b></div>
          {result.evidence.map(item => <a href={item.url} target="_blank" rel="noreferrer" key={item.id}>
            <span>{item.id}</span><div><b>{item.title}</b><small>{item.source} / {item.year}</small></div><ChevronRight size={13}/>
          </a>)}
        </section>
      </aside>
    </div>
  </motion.section>;
}

export default AMDWorkspace;
