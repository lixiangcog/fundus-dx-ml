import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, ArrowUpRight, BrainCircuit, Check, CircleAlert, CircleDot,
  FileImage, Github, ImagePlus, Layers3, LoaderCircle, Microscope,
  Network, Play, RotateCcw, ScanLine, ShieldAlert, Sparkles, UploadCloud,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);
const MAX_FILE_SIZE = 12 * 1024 * 1024;
const ICONS = { 'quality-enhancement': Sparkles, 'structure-segmentation': Layers3, 'lesion-recognition': ScanLine, 'vascular-quantification': Network };
const FALLBACK = [
  { id:'quality-enhancement',number:'01',title:'质量增强',english:'QUALITY ENHANCEMENT',default_modality:'OCT',modalities:['OCT','OCTA','眼底彩照'],engine:'Structure-preserving CPU baseline',engine_type:'algorithm',method:'CLAHE + 非局部均值去噪 + 结构锐化',sample_url:'/samples/ophthalmic/oct_quality.png',source_url:'https://github.com/opencv/opencv',license:'Apache-2.0',output:'增强影像 + 对比度/噪声代理指标' },
  { id:'structure-segmentation',number:'02',title:'结构分割',english:'STRUCTURE SEGMENTATION',default_modality:'OCT',modalities:['OCT'],engine:'ReLayNet · epoch 20',engine_type:'pretrained_model',method:'九类视网膜区域/层结构像素级分割',sample_url:'/samples/ophthalmic/oct_structure.png',source_url:'https://github.com/ai-med/relaynet_pytorch',license:'MIT',output:'结构叠加图 + 像素级分区统计' },
  { id:'lesion-recognition',number:'03',title:'病灶识别',english:'LESION RECOGNITION',default_modality:'眼底彩照',modalities:['眼底彩照'],engine:'FundusDx ResNet18 · v2',engine_type:'trained_model',method:'AMD / 白内障 / 糖网 / 正常四分类 + CAM 热力图',sample_url:'/samples/ophthalmic/fundus_lesion.jpg',source_url:'https://github.com/lixiangcog/fundus-dx-ml',license:'Project model',output:'分类概率 + 类激活定位图' },
  { id:'vascular-quantification',number:'04',title:'微血管定量',english:'MICROVASCULAR QUANTIFICATION',default_modality:'OCTA',modalities:['OCTA','眼底彩照'],engine:'Multi-scale vesselness + morphometry',engine_type:'algorithm',method:'多尺度 Hessian 血管响应、骨架化与分形/分支统计',sample_url:'/samples/ophthalmic/octa_vascular.png',source_url:'https://github.com/rmaphoh/AutoMorph',license:'Apache-2.0 / MIT references',output:'血管叠加图 + 密度/长度/分支/分形维数' },
];

function fileName(id) { return `demo_${id.replaceAll('-','_')}.png`; }
function engineType(type) { return type === 'pretrained_model' ? '预训练模型' : type === 'trained_model' ? '项目模型' : '实时算法'; }

function App() {
  const [capabilities, setCapabilities] = useState(FALLBACK);
  const [activeId, setActiveId] = useState('quality-enhancement');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [service, setService] = useState('checking');
  const [dragging, setDragging] = useState(false);
  const [isDefault, setIsDefault] = useState(true);
  const fileInput = useRef(null);
  const active = capabilities.find((item) => item.id === activeId) || capabilities[0];

  useEffect(() => {
    Promise.all([fetch(`${API_URL}/capabilities`).then((r) => r.json()), fetch(`${API_URL}/health`).then((r) => r.json())])
      .then(([catalog, health]) => { setCapabilities(catalog.capabilities); setService(health.status === 'ok' ? 'online' : 'degraded'); })
      .catch(() => setService('offline'));
  }, []);

  const installFile = (nextFile, nextPreview, defaultFlag = false) => {
    if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
    setFile(nextFile); setPreview(nextPreview); setIsDefault(defaultFlag); setResult(null); setError('');
  };

  const loadDefault = async (capability = active) => {
    try {
      const response = await fetch(capability.sample_url);
      if (!response.ok) throw new Error();
      const blob = await response.blob();
      installFile(new File([blob], fileName(capability.id), { type: blob.type || 'image/png' }), URL.createObjectURL(blob), true);
    } catch { setError('默认病例加载失败，请刷新页面后重试。'); }
  };

  useEffect(() => { loadDefault(active); }, [activeId]);
  useEffect(() => () => { if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview); }, [preview]);

  const chooseCapability = (id) => { if (loading || id === activeId) return; setActiveId(id); };
  const processFile = (nextFile) => {
    if (!nextFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP 影像。'); return; }
    if (nextFile.size > MAX_FILE_SIZE) { setError('影像不能超过 12 MB。'); return; }
    installFile(nextFile, URL.createObjectURL(nextFile));
  };
  const run = async () => {
    if (!file || loading) return;
    setLoading(true); setResult(null); setError('');
    const data = new FormData(); data.append('file', file);
    try {
      const response = await axios.post(`${API_URL}/analyze/${active.id}`, data);
      setResult(response.data); setService('online');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || '推理服务暂时不可用，请稍后重试。');
      if (!requestError.response) setService('offline');
    } finally { setLoading(false); }
  };

  return (
    <div className="app-shell">
      <div className="ambient-grid" aria-hidden="true"><i /><i /><i /><span /></div>
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><Microscope size={20} /></span><div><strong>RETINA<strong className="accent">SCOPE</strong></strong><small>多模态眼科影像智能分析平台</small></div></div>
        <div className="header-meta">
          <span><small>SUPPORTED MODALITY</small><b>OCT · OCTA · CFP</b></span>
          <span><small>ANALYSIS ENGINE</small><b>04 PIPELINES</b></span>
          <span className={`service ${service}`}><small>SYSTEM STATUS</small><b><i /> {service === 'online' ? 'ONLINE' : service === 'offline' ? 'OFFLINE' : 'CHECKING'}</b></span>
        </div>
      </header>

      <main>
        <section className="intro-row">
          <div><span className="eyebrow">OPHTHALMIC IMAGING WORKBENCH // V2.0</span><h1>从影像质量到微血管形态，<br /><em>一站式完成眼科影像分析。</em></h1></div>
          <p>支持 OCT、OCTA 与眼底彩照。每项能力均已配置默认病例、可运行引擎和来源说明，可直接开始实验。</p>
        </section>

        <nav className="pipeline-nav" aria-label="分析功能">
          {capabilities.map((item) => {
            const Icon = ICONS[item.id] || Activity; const selected = item.id === activeId;
            return <button key={item.id} className={selected ? 'active' : ''} onClick={() => chooseCapability(item.id)}>
              <span className="pipeline-no">// {item.number}</span><span className="pipeline-icon"><Icon size={18} /></span><span><strong>{item.title}</strong><small>{item.english}</small></span><i className="state-dot" />
            </button>;
          })}
        </nav>

        <section className="workbench">
          <aside className="engine-panel">
            <div className="panel-kicker"><CircleDot size={13} /> ACTIVE PIPELINE</div>
            <h2>{active.number}<span>/</span>{active.title}</h2><p className="english-title">{active.english}</p>
            <div className="modality-tags">{active.modalities.map((modality) => <span key={modality} className={modality === active.default_modality ? 'primary' : ''}>{modality}</span>)}</div>
            <dl>
              <div><dt>ENGINE</dt><dd>{active.engine}</dd></div><div><dt>METHOD</dt><dd>{active.method}</dd></div><div><dt>OUTPUT</dt><dd>{active.output}</dd></div>
            </dl>
            <a className="source-link" href={active.source_url} target="_blank" rel="noreferrer"><Github size={15} /><span><b>OPEN SOURCE REFERENCE</b><small>{active.license} · 查看代码来源</small></span><ArrowUpRight size={15} /></a>
            <div className="engine-type"><Check size={13} /><span>{engineType(active.engine_type)}</span><b>READY</b></div>
          </aside>

          <section className="visual-stage">
            <div className="stage-head"><span><FileImage size={14} /> INPUT / 输入影像</span><div><b>{isDefault ? 'DEFAULT CASE' : 'USER IMAGE'}</b><small>{active.default_modality}</small></div></div>
            <div className={`image-frame ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); processFile(event.dataTransfer.files?.[0]); }}>
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" /><span className="scan-beam" />
              {preview ? <img src={preview} alt={`${active.title}输入影像`} /> : <div className="empty-image"><ImagePlus size={30} />等待影像</div>}
              <div className="image-hud"><span>W {result?.input?.width || '---'}</span><span>H {result?.input?.height || '---'}</span><span>RGB / 8 BIT</span></div>
            </div>
            <div className="input-actions"><button onClick={() => fileInput.current?.click()}><UploadCloud size={15} />上传影像</button><button onClick={() => loadDefault()}><RotateCcw size={14} />恢复默认病例</button><input ref={fileInput} type="file" hidden accept="image/jpeg,image/png,image/webp" onChange={(event) => { if (event.target.files?.[0]) processFile(event.target.files[0]); event.target.value=''; }} /></div>
          </section>

          <div className={`data-bridge ${loading ? 'running' : ''}`} aria-hidden="true"><span /><i /><b>AI FLOW</b></div>

          <section className="visual-stage output-stage">
            <div className="stage-head"><span><BrainCircuit size={14} /> OUTPUT / 分析结果</span><div>{result && <><b>{result.runtime_ms} ms</b><small>RUNTIME</small></>}</div></div>
            <div className={`image-frame ${result ? 'has-result' : ''}`}>
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
              <AnimatePresence mode="wait">
                {loading ? <motion.div key="loading" className="processing" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}><span><LoaderCircle size={30} /></span><strong>ENGINE PROCESSING</strong><small>{active.engine}</small><i /></motion.div>
                  : result ? <motion.div key="result" className="result-image" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}><img src={result.result_image} alt={`${active.title}分析结果`} /><span className="result-scan" /><div className="result-label"><Check size={12} /> ANALYSIS COMPLETE</div></motion.div>
                  : <motion.div key="empty" className="waiting" initial={{opacity:0}} animate={{opacity:1}}><span className="radar"><Activity size={28} /></span><strong>等待执行分析</strong><small>选择默认病例或上传影像后启动引擎</small></motion.div>}
              </AnimatePresence>
            </div>
            <button className="run-button" onClick={run} disabled={!file || loading}><Play size={16} fill="currentColor" /><span>{loading ? '正在运行分析引擎' : `运行 ${active.title}`}</span><small>RUN PIPELINE</small></button>
          </section>

          <aside className="metrics-panel">
            <div className="panel-kicker"><Activity size={13} /> QUANTITATIVE OUTPUT</div>
            {result ? <motion.div className="result-data" initial={{opacity:0,y:5}} animate={{opacity:1,y:0}}>
              <h3>{result.summary}</h3><div className="metric-list">{result.metrics.map((metric,index) => <div key={`${metric.label}-${index}`}><span>{metric.label}<small>{metric.detail}</small></span><strong>{metric.value}<em>{metric.unit}</em></strong></div>)}</div>
              {result.probabilities && <div className="probability-mini">{Object.entries(result.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value]) => <div key={name}><span>{name.replaceAll('_',' ')}</span><i><b style={{width:`${value*100}%`}} /></i><strong>{(value*100).toFixed(1)}%</strong></div>)}</div>}
              <p className="result-notice"><CircleAlert size={14} />{result.notice}</p>
            </motion.div> : <div className="metric-placeholder"><div className="signal-bars">{[24,48,34,70,52,86,42,64,30,74].map((h,i)=><i key={i} style={{height:`${h}%`}} />)}</div><p>运行后将在这里显示结构化指标与模型说明。</p></div>}
          </aside>
        </section>

        {error && <div className="error-banner"><CircleAlert size={16} />{error}</div>}
        <footer><p><ShieldAlert size={14} />科研与教学演示系统，输出不构成临床诊断或治疗建议。</p><span>MODEL & CODE PROVENANCE AVAILABLE · BUILD 2026.08</span></footer>
      </main>
    </div>
  );
}

export default App;
