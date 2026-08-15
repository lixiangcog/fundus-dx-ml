import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, Brain, Check, CircleAlert, Eye, FileImage, HeartPulse,
  LoaderCircle, Play, RotateCcw, ScanLine, ShieldCheck, UploadCloud,
} from 'lucide-react';
import ModuleLog from './ModuleLog';
import { useModuleLog } from './useModuleLog';
import './systemic.css';

const MAX_FILE_SIZE = 12 * 1024 * 1024;
const DEFAULT_STROKE_PROFILE = { age:67, sex:'male', systolic_bp:145, smoker:false, diabetes:true, atrial_fibrillation:false, antihypertensive:true, cardiovascular_disease:false };
const FALLBACK = {
  'eye-age': { id:'eye-age', number:'03', title:'眼龄', subtitle:'估算视网膜表观年龄，并与实际年龄对照', sample_age:57, sample_note:'', sample_url:'/systemic/sample/eye-age', published_validation:'公开测试集 MAE 5.09 岁' },
  'cardiovascular-retina': { id:'cardiovascular-retina', number:'04', title:'眼观心血管', subtitle:'联合彩照与 OCTA 评估冠心病相关视网膜风险', sample_note:'', sample_url:'/systemic/sample/cardiovascular-retina', sample_octa_url:'/systemic/sample/cardiovascular-retina?modality=octa', published_validation:'CFP + OCTA 联合风险评估' },
  'cerebrovascular-retina': { id:'cerebrovascular-retina', number:'05', title:'眼观脑血管', subtitle:'结合眼底影像与健康信息评估脑卒中风险', sample_profile:DEFAULT_STROKE_PROFILE, sample_note:'', sample_url:'/systemic/sample/cerebrovascular-retina', published_validation:'10 年首次卒中风险评估' },
};
const ICONS = { 'eye-age': Eye, 'cardiovascular-retina': HeartPulse, 'cerebrovascular-retina': Brain };
const STEPS = {
  'eye-age': ['彩照标准化', '眼龄估算', '年龄差对照'],
  'cardiovascular-retina': ['双模态输入', '微血管量化', '冠心病风险'],
  'cerebrovascular-retina': ['眼底解读', '血管量化', '风险评估'],
};

function SystemicWorkspace({ apiUrl, moduleId }) {
  const [config, setConfig] = useState(FALLBACK[moduleId]);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [octaFile, setOctaFile] = useState(null);
  const [octaPreview, setOctaPreview] = useState('');
  const [age, setAge] = useState(FALLBACK[moduleId]?.sample_age || '');
  const [strokeProfile, setStrokeProfile] = useState({ ...DEFAULT_STROKE_PROFILE });
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { entries: moduleLogs, write: writeModuleLog, writeMany: writeModuleLogs, clear: clearModuleLog } = useModuleLog(FALLBACK[moduleId]?.title || '系统模块');
  const fileInput = useRef(null);
  const octaInput = useRef(null);
  const Icon = ICONS[moduleId] || Eye;
  const updateStrokeProfile = (key, value) => setStrokeProfile((current) => ({ ...current, [key]:value }));

  const installFile = (nextFile, url) => {
    if (preview.startsWith('blob:')) URL.revokeObjectURL(preview);
    setFile(nextFile); setPreview(url); setResult(null); setActiveView(0); setError('');
  };
  const installOctaFile = (nextFile, url) => {
    if (octaPreview.startsWith('blob:')) URL.revokeObjectURL(octaPreview);
    setOctaFile(nextFile); setOctaPreview(url); setResult(null); setActiveView(0); setError('');
  };
  const loadDefault = async (nextConfig = config) => {
    try {
      const response = await fetch(`${apiUrl}${nextConfig.sample_url}`);
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      installFile(new File([blob], `${moduleId}-default.jpg`, { type:blob.type || 'image/jpeg' }), URL.createObjectURL(blob));
      if (moduleId === 'cardiovascular-retina' && nextConfig.sample_octa_url) {
        const octaResponse = await fetch(`${apiUrl}${nextConfig.sample_octa_url}`);
        if (!octaResponse.ok) throw new Error();
        const octaBlob = await octaResponse.blob();
        installOctaFile(new File([octaBlob], 'cardiovascular-default-octa.png', { type:octaBlob.type || 'image/png' }), URL.createObjectURL(octaBlob));
        writeModuleLog('info', `sample=cardiovascular-default-octa; mime=${octaBlob.type || 'image/png'}`, `bytes=${octaBlob.size}; ready=true`, 'INPUT');
      }
      if (nextConfig.sample_age) setAge(nextConfig.sample_age);
      if (nextConfig.sample_profile) setStrokeProfile({ ...nextConfig.sample_profile });
      writeModuleLog('info', `sample=${moduleId}-default; mime=${blob.type || 'image/jpeg'}`, `bytes=${blob.size}; ready=true`, 'INPUT');
    } catch { setError('默认研究样例加载失败，请刷新后重试。'); writeModuleLog('error', `GET ${nextConfig.sample_url} -> invalid image`, '默认研究样例加载失败', 'HTTP'); }
  };

  useEffect(() => {
    let current = true;
    fetch(`${apiUrl}/systemic/config`).then((response) => response.json()).then((payload) => {
      const next = payload.modules.find((item) => item.id === moduleId) || FALLBACK[moduleId];
      if (current) {
        setConfig(next); setAge(next.sample_age || '');
        writeModuleLogs([
          { level:'command', channel:'SHELL', message:`fundus-dx systemic status --module ${moduleId}` },
          { level:'success', channel:'CONFIG', message:`module=${moduleId}; ready=true`, detail:next.published_validation },
          { level:'success', channel:'CUDA', message:'gpu-worker=ready; real_inference=true', detail:'device=cuda:0' },
        ]);
        loadDefault(next);
      }
    }).catch(() => { if (current) { const next = FALLBACK[moduleId]; setConfig(next); setAge(next.sample_age || ''); writeModuleLog('warning', 'GET /systemic/config -> fallback', '已使用内置配置继续运行', 'HTTP'); loadDefault(next); } });
    return () => { current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moduleId]);
  useEffect(() => () => { if (preview.startsWith('blob:')) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => () => { if (octaPreview.startsWith('blob:')) URL.revokeObjectURL(octaPreview); }, [octaPreview]);

  const chooseFile = (nextFile) => {
    if (!nextFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP 彩照。'); writeModuleLog('warning', '彩照格式不受支持', '允许 JPG、PNG 或 WebP'); return; }
    if (nextFile.size > MAX_FILE_SIZE) { setError('影像不能超过 12 MB。'); writeModuleLog('warning', '彩照超过大小限制', '最大允许 12 MB'); return; }
    installFile(nextFile, URL.createObjectURL(nextFile));
    writeModuleLog('info', `upload accepted; type=${nextFile.type}`, `bytes=${nextFile.size}; module=${moduleId}`, 'INPUT');
  };
  const chooseOctaFile = (nextFile) => {
    if (!nextFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP OCTA。'); writeModuleLog('warning', 'OCTA 格式不受支持', '允许 JPG、PNG 或 WebP'); return; }
    if (nextFile.size > MAX_FILE_SIZE) { setError('影像不能超过 12 MB。'); writeModuleLog('warning', 'OCTA 超过大小限制', '最大允许 12 MB'); return; }
    installOctaFile(nextFile, URL.createObjectURL(nextFile));
    writeModuleLog('info', `octa upload accepted; type=${nextFile.type}`, `bytes=${nextFile.size}; module=${moduleId}`, 'INPUT');
  };
  const run = async () => {
    if (!file || loading || (moduleId === 'cardiovascular-retina' && !octaFile)) return;
    if (moduleId === 'eye-age' && (!age || Number(age) < 18 || Number(age) > 100)) { setError('请输入 18 至 100 岁的实际年龄。'); writeModuleLog('warning', '实际年龄校验未通过', '请输入 18 至 100 岁'); return; }
    if (moduleId === 'cerebrovascular-retina' && (Number(strokeProfile.age) < 55 || Number(strokeProfile.age) > 84 || Number(strokeProfile.systolic_bp) < 80 || Number(strokeProfile.systolic_bp) > 240)) {
      setError('请检查年龄和收缩压：年龄 55–84 岁，收缩压 80–240 mmHg。');
      writeModuleLog('warning', '卒中风险输入校验未通过', 'age=55..84; systolic_bp=80..240');
      return;
    }
    const data = new FormData(); data.append('file', file);
    if (moduleId === 'eye-age') data.append('chronological_age', age);
    if (moduleId === 'cardiovascular-retina') data.append('octa_file', octaFile);
    if (moduleId === 'cerebrovascular-retina') {
      data.append('risk_age', strokeProfile.age);
      data.append('risk_sex', strokeProfile.sex);
      data.append('systolic_bp', strokeProfile.systolic_bp);
      ['smoker','diabetes','atrial_fibrillation','antihypertensive','cardiovascular_disease'].forEach((key) => data.append(key, String(Boolean(strokeProfile[key]))));
    }
    setLoading(true); setResult(null); setError('');
    writeModuleLogs([
      { level:'command', channel:'SHELL', message:`fundus-dx systemic run --module ${moduleId} --device cuda:0` },
      { level:'info', channel:'INPUT', message:`mime=${file.type || 'image/unknown'}; bytes=${file.size}`, detail:moduleId === 'eye-age' ? `chronological_age=${age}` : moduleId === 'cerebrovascular-retina' ? `age=${strokeProfile.age}; sbp=${strokeProfile.systolic_bp}; sex=${strokeProfile.sex}` : `cfp_bytes=${file.size}; octa_bytes=${octaFile?.size || 0}` },
      { level:'run', channel:'QUEUE', message:'request accepted; GPU worker acquired', detail:`task=${moduleId === 'eye-age' ? 'eye_age' : 'retinal_vascular'}` },
      { level:'run', channel:'CUDA', message:'preprocess -> forward -> postprocess', detail:'inference_mode=true' },
    ]);
    try {
      const response = await axios.post(`${apiUrl}/systemic/analyze/${moduleId}`, data);
      setResult(response.data); setActiveView(0);
      const output = response.data;
      const metricRows = (output.metrics || []).map((metric) => ({ level:'info', channel:'METRIC', message:`${metric.label}=${metric.value}${metric.unit || ''}`, detail:metric.detail || '' }));
      const traceRows = (output.trace || []).map((step) => ({ level:'success', channel:'MODEL', message:`${step.tool} -> ${step.status}`, detail:`${step.model || step.method || 'executed'}${step.runtime_ms !== undefined ? `; runtime_ms=${step.runtime_ms}` : ''}` }));
      writeModuleLogs([
        { level:'success', channel:'HTTP', message:`POST /systemic/analyze/${moduleId} -> 200 OK` },
        { level:'success', channel:'BACKEND', message:`module=${output.module?.id || moduleId}; real_inference=${Boolean(output.real_inference)}`, detail:`views=${output.views?.length || 1}` },
        ...traceRows,
        ...metricRows,
        ...(output.quantified_feature_count !== undefined ? [{ level:'info', channel:'OUTPUT', message:`quantified_features=${output.quantified_feature_count}`, detail:'retinal vascular phenotype fields' }] : []),
        { level:output.quality?.status === 'review' ? 'warning' : 'success', channel:'QC', message:`status=${output.quality?.status || 'unknown'}`, detail:output.quality?.detail || '' },
        { level:'success', channel:'DONE', message:`${moduleId} completed`, detail:`runtime_ms=${output.runtime_ms}` },
      ]);
    } catch (requestError) { setError(requestError.response?.data?.detail || '推理服务暂时不可用，请稍后重试。'); writeModuleLog('error', `POST /systemic/analyze/${moduleId} -> ${requestError.response?.status || 'NETWORK_ERROR'}`, requestError.response?.data?.detail || '无法连接推理服务', 'HTTP'); }
    finally { setLoading(false); }
  };
  const shownImage = result?.views?.[activeView]?.image || result?.result_image;

  return <div className={`systemic-workspace systemic-${moduleId}`}>
    <section className="systemic-command">
      <div className="systemic-orbit" aria-hidden="true"><i/><i/><i/><span/><span/></div>
      <div className="systemic-title">
        <span className="systemic-index">SYSTEM / {config.number}</span>
        <h1>{config.title}</h1>
        <p>{config.subtitle}</p>
      </div>
      <div className="systemic-core" aria-hidden="true">
        <span className="core-halo halo-a"/><span className="core-halo halo-b"/>
        <div><Icon size={38}/><b>{config.number}</b><small>分析核心</small></div>
        <i className="core-sweep"/>
      </div>
      <div className="systemic-steps">
        {STEPS[moduleId].map((step,index) => <div key={step}><span>0{index+1}</span><i/><b>{step}</b><Check size={13}/></div>)}
      </div>
      <div className="systemic-validation">
        <span><ShieldCheck size={14}/>质量校验</span><b>{config.published_validation}</b>
      </div>
    </section>

    <section className="systemic-station">
      <header className="systemic-station-head">
        <div>{moduleId !== 'eye-age' && <span>分析工作台</span>}<h2>{moduleId === 'cardiovascular-retina' ? '上传彩照与 OCTA，查看' : '上传眼底彩照，查看'}<em>可复算结果</em></h2></div>
        {moduleId !== 'eye-age' && moduleId !== 'cerebrovascular-retina' && config.sample_note && <p>{config.sample_note}</p>}
      </header>
      <div className="systemic-grid">
        <section className="systemic-input">
          <div className="systemic-panel-head"><span><FileImage size={14}/>输入彩照</span><small>默认样例可直接运行</small></div>
          <div className={`systemic-image-frame ${moduleId === 'cardiovascular-retina' ? 'cardio-image-frame' : ''}`}>
            <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
            {moduleId === 'cardiovascular-retina' && <span className="systemic-modality-label">CFP 彩照</span>}
            {preview ? <img src={preview} alt="输入眼底彩照"/> : <div className="systemic-empty"><ScanLine size={28}/>等待彩照</div>}
            <i className="systemic-scan"/>
          </div>
          {moduleId === 'cardiovascular-retina' && <div className="systemic-image-frame cardio-image-frame">
            <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
            <span className="systemic-modality-label">OCTA</span>
            {octaPreview ? <img src={octaPreview} alt="输入 OCTA"/> : <div className="systemic-empty"><ScanLine size={28}/>等待 OCTA</div>}
            <i className="systemic-scan"/>
          </div>}
          {moduleId === 'eye-age' && <label className="age-input"><span>实际年龄</span><input type="number" min="18" max="100" value={age} onChange={(event) => setAge(event.target.value)}/><em>岁</em></label>}
          {moduleId === 'cerebrovascular-retina' && <div className="stroke-profile">
            <div className="stroke-profile-primary">
              <label><span>年龄</span><input type="number" min="55" max="84" value={strokeProfile.age} onChange={(event) => updateStrokeProfile('age', event.target.value)}/><em>岁</em></label>
              <label><span>性别</span><select value={strokeProfile.sex} onChange={(event) => updateStrokeProfile('sex', event.target.value)}><option value="male">男</option><option value="female">女</option></select></label>
              <label><span>收缩压</span><input type="number" min="80" max="240" value={strokeProfile.systolic_bp} onChange={(event) => updateStrokeProfile('systolic_bp', event.target.value)}/><em>mmHg</em></label>
            </div>
            <div className="stroke-profile-flags">
              {[
                ['smoker','吸烟'], ['diabetes','糖尿病'], ['atrial_fibrillation','房颤'],
                ['antihypertensive','降压治疗'], ['cardiovascular_disease','心血管病'],
              ].map(([key,label]) => <button type="button" key={key} className={strokeProfile[key] ? 'active' : ''} onClick={() => updateStrokeProfile(key, !strokeProfile[key])}><span>{strokeProfile[key] ? <Check size={11}/> : null}</span>{label}</button>)}
            </div>
          </div>}
          <div className="systemic-actions">
            <button onClick={() => fileInput.current?.click()}><UploadCloud size={15}/>上传彩照</button>
            <button onClick={() => loadDefault()}><RotateCcw size={14}/>恢复样例</button>
            <input ref={fileInput} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { if (event.target.files?.[0]) chooseFile(event.target.files[0]); event.target.value=''; }}/>
            {moduleId === 'cardiovascular-retina' && <button onClick={() => octaInput.current?.click()}><UploadCloud size={15}/>上传 OCTA</button>}
          </div>
            <input ref={octaInput} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { if (event.target.files?.[0]) chooseOctaFile(event.target.files[0]); event.target.value=''; }}/>
        </section>

        <div className={`systemic-bridge ${loading ? 'running' : ''}`} aria-hidden="true"><i/><span/><b/></div>

        <section className="systemic-output">
          <div className="systemic-panel-head"><span><Activity size={14}/>分析结果</span>{result && <small>{result.runtime_ms} 毫秒</small>}</div>
          <div className={`systemic-image-frame ${result ? 'complete' : ''}`}>
            <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
            <AnimatePresence mode="wait">
              {loading ? <motion.div key="loading" className="systemic-processing" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}><LoaderCircle size={32}/><b>正在分析</b><small>{moduleId === 'eye-age' ? '正在估算眼龄' : moduleId === 'cerebrovascular-retina' ? '正在进行眼底解读与风险评估' : '正在联合分析彩照与 OCTA'}</small><i/></motion.div>
                : result ? <motion.img key={shownImage} src={shownImage} alt="分析结果" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}/>
                : <motion.div key="waiting" className="systemic-waiting" initial={{opacity:0}} animate={{opacity:1}}><Icon size={30}/><b>等待分析</b><small>内置样例已就绪</small></motion.div>}
            </AnimatePresence>
            {result && <span className="systemic-done"><Check size={12}/>分析完成</span>}
          </div>
          {result?.views?.length > 1 && <nav className="view-switch">{result.views.map((view,index) => <button key={view.label} className={activeView === index ? 'active' : ''} onClick={() => { setActiveView(index); writeModuleLog('info', `切换结果视图：${view.label}`, '显示内容已更新'); }}>{view.label}</button>)}</nav>}
          <button className="systemic-run" onClick={run} disabled={!file || loading || (moduleId === 'cardiovascular-retina' && !octaFile)}><Play size={16} fill="currentColor"/>{loading ? '正在分析' : `开始${config.title}分析`}</button>
        </section>

        <aside className="systemic-metrics">
          <div className="systemic-panel-head"><span><Activity size={14}/>定量结果</span>{result && <small>{result.status_label}</small>}</div>
          {result ? <motion.div className="systemic-result-copy" initial={{opacity:0,y:6}} animate={{opacity:1,y:0}}>
            {moduleId !== 'eye-age' && moduleId !== 'cerebrovascular-retina' && <div className={`systemic-quality ${result.quality.status}`}><span><Check size={13}/>{result.quality.label}</span><small>{result.quality.detail}</small></div>}
            <h3>{result.summary}</h3>
            <div className="systemic-metric-list">{result.metrics.map((metric) => <div key={metric.label}><span>{metric.label}<small>{metric.detail}</small></span><strong>{metric.value}<em>{metric.unit}</em></strong></div>)}</div>
            <div className="systemic-findings">{result.sections.map((section) => <div key={section.title}><b>{section.title}</b><p>{section.text}</p></div>)}</div>
            {result.notice && <p className="systemic-notice"><CircleAlert size={13}/>{result.notice}</p>}
          </motion.div> : <div className="systemic-placeholder"><div>{[38,64,48,82,58,92,46,72,54,86].map((height,index) => <i key={index} style={{height:`${height}%`}}/>)}</div><p>运行后显示分割、量化与结果解读。</p></div>}
        </aside>
      </div>
      <ModuleLog title={config.title} entries={moduleLogs} onClear={clearModuleLog} running={loading}/>
      {error && <div className="systemic-error"><CircleAlert size={15}/>{error}</div>}
    </section>
  </div>;
}

export default SystemicWorkspace;
