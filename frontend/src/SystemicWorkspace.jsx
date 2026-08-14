import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, Brain, Check, CircleAlert, Eye, FileImage, HeartPulse,
  LoaderCircle, Play, RotateCcw, ScanLine, ShieldCheck, UploadCloud,
} from 'lucide-react';
import './systemic.css';

const MAX_FILE_SIZE = 12 * 1024 * 1024;
const FALLBACK = {
  'eye-age': { id:'eye-age', number:'03', title:'眼龄', subtitle:'估算视网膜表观年龄，并与实际年龄对照', sample_age:57, sample_note:'官方测试集留出样例', sample_url:'/systemic/sample/eye-age', published_validation:'公开测试集 MAE 5.09 岁' },
  'cardiovascular-retina': { id:'cardiovascular-retina', number:'04', title:'眼观心血管', subtitle:'提取与心血管研究相关的视网膜微血管表型', sample_note:'高分辨率眼底彩照研究样例', sample_url:'/systemic/sample/cardiovascular-retina', published_validation:'75 项血管表型可复算' },
  'cerebrovascular-retina': { id:'cerebrovascular-retina', number:'05', title:'眼观脑血管', subtitle:'分析与脑小血管研究相关的视网膜微循环表型', sample_note:'高分辨率眼底彩照研究样例', sample_url:'/systemic/sample/cerebrovascular-retina', published_validation:'75 项血管表型可复算' },
};
const ICONS = { 'eye-age': Eye, 'cardiovascular-retina': HeartPulse, 'cerebrovascular-retina': Brain };
const STEPS = {
  'eye-age': ['彩照标准化', '眼龄估算', '年龄差对照'],
  'cardiovascular-retina': ['动静脉分割', '形态量化', '表型归纳'],
  'cerebrovascular-retina': ['微血管分割', '分布量化', '表型归纳'],
};

function SystemicWorkspace({ apiUrl, moduleId }) {
  const [config, setConfig] = useState(FALLBACK[moduleId]);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [age, setAge] = useState(FALLBACK[moduleId]?.sample_age || '');
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInput = useRef(null);
  const Icon = ICONS[moduleId] || Eye;

  const installFile = (nextFile, url) => {
    if (preview.startsWith('blob:')) URL.revokeObjectURL(preview);
    setFile(nextFile); setPreview(url); setResult(null); setActiveView(0); setError('');
  };
  const loadDefault = async (nextConfig = config) => {
    try {
      const response = await fetch(`${apiUrl}${nextConfig.sample_url}`);
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      installFile(new File([blob], `${moduleId}-default.jpg`, { type:blob.type || 'image/jpeg' }), URL.createObjectURL(blob));
      if (nextConfig.sample_age) setAge(nextConfig.sample_age);
    } catch { setError('默认研究样例加载失败，请刷新后重试。'); }
  };

  useEffect(() => {
    let current = true;
    fetch(`${apiUrl}/systemic/config`).then((response) => response.json()).then((payload) => {
      const next = payload.modules.find((item) => item.id === moduleId) || FALLBACK[moduleId];
      if (current) { setConfig(next); setAge(next.sample_age || ''); loadDefault(next); }
    }).catch(() => { if (current) { const next = FALLBACK[moduleId]; setConfig(next); setAge(next.sample_age || ''); loadDefault(next); } });
    return () => { current = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moduleId]);
  useEffect(() => () => { if (preview.startsWith('blob:')) URL.revokeObjectURL(preview); }, [preview]);

  const chooseFile = (nextFile) => {
    if (!nextFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP 彩照。'); return; }
    if (nextFile.size > MAX_FILE_SIZE) { setError('影像不能超过 12 MB。'); return; }
    installFile(nextFile, URL.createObjectURL(nextFile));
  };
  const run = async () => {
    if (!file || loading) return;
    if (moduleId === 'eye-age' && (!age || Number(age) < 18 || Number(age) > 100)) { setError('请输入 18 至 100 岁的实际年龄。'); return; }
    const data = new FormData(); data.append('file', file);
    if (moduleId === 'eye-age') data.append('chronological_age', age);
    setLoading(true); setResult(null); setError('');
    try {
      const response = await axios.post(`${apiUrl}/systemic/analyze/${moduleId}`, data);
      setResult(response.data); setActiveView(0);
    } catch (requestError) { setError(requestError.response?.data?.detail || '推理服务暂时不可用，请稍后重试。'); }
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
        <span><ShieldCheck size={14}/>质量校验</span><b>{config.published_validation}</b><small>公开权重 · 真实推理</small>
      </div>
    </section>

    <section className="systemic-station">
      <header className="systemic-station-head">
        <div><span>分析工作台</span><h2>上传眼底彩照，查看<em>可复算结果</em></h2></div>
        <p>{config.sample_note}</p>
      </header>
      <div className="systemic-grid">
        <section className="systemic-input">
          <div className="systemic-panel-head"><span><FileImage size={14}/>输入彩照</span><small>默认样例可直接运行</small></div>
          <div className="systemic-image-frame">
            <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
            {preview ? <img src={preview} alt="输入眼底彩照"/> : <div className="systemic-empty"><ScanLine size={28}/>等待彩照</div>}
            <i className="systemic-scan"/>
          </div>
          {moduleId === 'eye-age' && <label className="age-input"><span>实际年龄</span><input type="number" min="18" max="100" value={age} onChange={(event) => setAge(event.target.value)}/><em>岁</em></label>}
          <div className="systemic-actions">
            <button onClick={() => fileInput.current?.click()}><UploadCloud size={15}/>上传彩照</button>
            <button onClick={() => loadDefault()}><RotateCcw size={14}/>恢复样例</button>
            <input ref={fileInput} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { if (event.target.files?.[0]) chooseFile(event.target.files[0]); event.target.value=''; }}/>
          </div>
        </section>

        <div className={`systemic-bridge ${loading ? 'running' : ''}`} aria-hidden="true"><i/><span/><b/></div>

        <section className="systemic-output">
          <div className="systemic-panel-head"><span><Activity size={14}/>分析结果</span>{result && <small>{result.runtime_ms} 毫秒</small>}</div>
          <div className={`systemic-image-frame ${result ? 'complete' : ''}`}>
            <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
            <AnimatePresence mode="wait">
              {loading ? <motion.div key="loading" className="systemic-processing" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}><LoaderCircle size={32}/><b>正在运行真实推理</b><small>{moduleId === 'eye-age' ? '正在估算眼龄' : '正在分割并计算血管表型'}</small><i/></motion.div>
                : result ? <motion.img key={shownImage} src={shownImage} alt="分析结果" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}/>
                : <motion.div key="waiting" className="systemic-waiting" initial={{opacity:0}} animate={{opacity:1}}><Icon size={30}/><b>等待分析</b><small>内置样例已就绪</small></motion.div>}
            </AnimatePresence>
            {result && <span className="systemic-done"><Check size={12}/>分析完成</span>}
          </div>
          {result?.views?.length > 1 && <nav className="view-switch">{result.views.map((view,index) => <button key={view.label} className={activeView === index ? 'active' : ''} onClick={() => setActiveView(index)}>{view.label}</button>)}</nav>}
          <button className="systemic-run" onClick={run} disabled={!file || loading}><Play size={16} fill="currentColor"/>{loading ? '正在分析' : `开始${config.title}分析`}</button>
        </section>

        <aside className="systemic-metrics">
          <div className="systemic-panel-head"><span><Activity size={14}/>定量结果</span>{result && <small>{result.status_label}</small>}</div>
          {result ? <motion.div className="systemic-result-copy" initial={{opacity:0,y:6}} animate={{opacity:1,y:0}}>
            <div className={`systemic-quality ${result.quality.status}`}><span><Check size={13}/>{result.quality.label}</span><small>{result.quality.detail}</small></div>
            <h3>{result.summary}</h3>
            <div className="systemic-metric-list">{result.metrics.map((metric) => <div key={metric.label}><span>{metric.label}<small>{metric.detail}</small></span><strong>{metric.value}<em>{metric.unit}</em></strong></div>)}</div>
            <div className="systemic-findings">{result.sections.map((section) => <div key={section.title}><b>{section.title}</b><p>{section.text}</p></div>)}</div>
            <p className="systemic-notice"><CircleAlert size={13}/>{result.notice}</p>
          </motion.div> : <div className="systemic-placeholder"><div>{[38,64,48,82,58,92,46,72,54,86].map((height,index) => <i key={index} style={{height:`${height}%`}}/>)}</div><p>运行后显示分割、量化与结果解读。</p></div>}
        </aside>
      </div>
      {error && <div className="systemic-error"><CircleAlert size={15}/>{error}</div>}
    </section>
  </div>;
}

export default SystemicWorkspace;
