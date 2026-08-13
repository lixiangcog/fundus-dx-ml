import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, Archive, BrainCircuit, Check, CircleAlert, Database, Eye,
  FileImage, FileText, Github, Layers3, LoaderCircle, LocateFixed,
  Network, RotateCcw, ScanLine, ShieldCheck, UploadCloud, X,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);
const MAX_FILE_SIZE = 12 * 1024 * 1024;

const SAMPLES = [
  { src: '/samples/normal.jpg', label: '正常样例', filename: 'sample_normal.jpg' },
  { src: '/samples/cataract.jpg', label: '白内障样例', filename: 'sample_cataract.jpg' },
  { src: '/samples/diabetic_retinopathy.jpg', label: '糖网样例', filename: 'sample_dr.jpg' },
];

const CLASS_META = {
  amd: { label: '年龄相关性黄斑变性', short: 'AMD', note: '图像特征更接近模型训练集中的 AMD 样本。', next: '建议结合黄斑区检查、OCT 等临床信息由眼科医生复核。' },
  cataract: { label: '白内障', short: 'CATARACT', note: '图像特征更接近模型训练集中的白内障样本。', next: '建议结合裂隙灯检查、视力情况与晶状体混浊程度复核。' },
  diabetic_retinopathy: { label: '糖尿病视网膜病变', short: 'DR', note: '图像特征更接近模型训练集中的糖尿病视网膜病变样本。', next: '建议由眼科医生复核病变分期，并结合糖尿病病史制定随访方案。' },
  normal: { label: '未见模型已知异常', short: 'NORMAL', note: '在四个训练类别中，模型更倾向于正常类别。', next: '结果不能排除训练范围之外的眼病；如有症状，仍应接受专业检查。' },
};

const WORKFLOW = [
  { number: '01', label: '影像导入与预处理', icon: FileImage, ready: true },
  { number: '02', label: '智能眼底筛查', icon: BrainCircuit, ready: true },
  { number: '03', label: '病灶精准定位', icon: LocateFixed, ready: false },
  { number: '04', label: '生成临床报告', icon: FileText, ready: false },
];

const METRICS = [
  ['验证准确率', '97.7%', 'ACCURACY'],
  ['宏平均 F1', '97.5%', 'F1 SCORE'],
  ['验证样本', '871', 'VALIDATION'],
  ['输出类别', '4', 'CLASSES'],
];

function createCaseId() {
  const date = new Date().toISOString().slice(0, 10).replaceAll('-', '');
  return `FDX-${date}-${Math.floor(1000 + Math.random() * 9000)}`;
}

function confidenceMeta(confidence) {
  if (confidence >= 0.8) return { label: '高模型置信度', tone: 'high', message: '结果较集中，仍需结合临床资料复核。' };
  if (confidence >= 0.6) return { label: '中等模型置信度', tone: 'medium', message: '存在一定不确定性，建议人工复核。' };
  return { label: '低模型置信度', tone: 'low', message: '建议重新采集影像或直接转人工判读。' };
}

function App() {
  const [stationOpen, setStationOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [latency, setLatency] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [serviceStatus, setServiceStatus] = useState('checking');
  const [patientName, setPatientName] = useState('');
  const [caseId] = useState(createCaseId);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    fetch(`${API_URL}/health`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error('Service unavailable'); return response.json(); })
      .then((data) => setServiceStatus(data.model_loaded ? 'online' : 'degraded'))
      .catch((requestError) => { if (requestError.name !== 'AbortError') setServiceStatus('offline'); })
      .finally(() => clearTimeout(timer));
    return () => { clearTimeout(timer); controller.abort(); };
  }, []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    document.body.classList.toggle('station-active', stationOpen);
    return () => document.body.classList.remove('station-active');
  }, [stationOpen]);

  const processFile = (selectedFile) => {
    if (!selectedFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP 格式的眼底图像。'); return; }
    if (selectedFile.size > MAX_FILE_SIZE) { setError('图像不能超过 12 MB，请压缩后重试。'); return; }
    setFile(selectedFile); setPreview(URL.createObjectURL(selectedFile)); setPrediction(null); setLatency(null); setError(null);
  };
  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) processFile(selectedFile);
    event.target.value = '';
  };
  const loadSample = async (sample) => {
    try {
      const response = await fetch(sample.src);
      const blob = await response.blob();
      processFile(new File([blob], sample.filename, { type: 'image/jpeg' }));
    } catch { setError('示例图像加载失败，请稍后重试。'); }
  };
  const runScreening = async () => {
    if (!file || loading) return;
    setLoading(true); setPrediction(null); setError(null);
    const formData = new FormData(); formData.append('file', file);
    const startedAt = performance.now();
    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setLatency(Math.round(performance.now() - startedAt));
      setPrediction(response.data); setServiceStatus('online');
    } catch (requestError) {
      setError(requestError.response?.data?.detail || '无法连接筛查引擎，请稍后重试。');
      if (!requestError.response) setServiceStatus('offline');
    } finally { setLoading(false); }
  };
  const reset = () => { setFile(null); setPreview(null); setPrediction(null); setLatency(null); setError(null); };
  const sortedProbabilities = prediction ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a) : [];
  const resultMeta = prediction ? CLASS_META[prediction.prediction] : null;
  const confidence = prediction ? confidenceMeta(prediction.confidence) : null;

  return (
    <div className="holo-shell">
      <div className="ambient-field" aria-hidden="true">
        <span className="orb orb-a" /><span className="orb orb-b" /><span className="perspective-grid" />
        <span className="hud-corner corner-tl" /><span className="hud-corner corner-br" />
      </div>

      <header className="hud-header">
        <div className="hud-identity">
          <span className="eye-seal"><Eye size={25} /></span>
          <div><strong className="tech-font">Fundus DX</strong><span>视界守望者 · 智能眼底筛查平台</span></div>
        </div>
        <div className="hud-data">
          <div><small>MODEL ARCH</small><strong>ResNet18</strong></div>
          <div><small>INPUT MATRIX</small><strong>224 × 224 RGB</strong></div>
          <div><small>SYSTEM STATUS</small><strong className={`status-value ${serviceStatus}`}><i />{serviceStatus === 'online' ? 'ONLINE' : serviceStatus.toUpperCase()}</strong></div>
        </div>
      </header>

      <main className="dashboard">
        <a className="archive-chip" href="https://github.com/lixiangcog/fundus-dx-ml" target="_blank" rel="noreferrer">
          <Archive size={17} /><span>CLINICAL ARCHIVE</span><Github size={15} />
        </a>

        <section className="dashboard-stage">
          <div className="core-column">
            <div className="module-caption"><span>AI ANALYSIS CORE</span><i /></div>
            <button className="radar-core" onClick={() => setStationOpen(true)} aria-label="进入诊断工作站">
              <span className="radar-ring radar-ring-outer" /><span className="radar-ring radar-ring-mid" /><span className="radar-ring radar-ring-inner" />
              <span className="radar-cross cross-x" /><span className="radar-cross cross-y" /><span className="scan-pointer" />
              <span className="core-copy"><small>FUNDUS ANALYSIS CORE</small><strong className="tech-font">ResNet18</strong><em>眼底影像分类引擎</em><b><Activity size={12} /> 点击进入工作站</b></span>
            </button>
            <div className="core-specs"><span><small>DEVICE</small><b>CPU SERVICE</b></span><span><small>WEIGHTS</small><b>44.8 MB</b></span><span><small>MODE</small><b>4-CLASS</b></span></div>
          </div>

          <div className="flow-column" aria-label="双流特征管线">
            <div className="flow-label top"><Network size={15} /><span><b>GLOBAL STREAM</b>全局结构流</span></div>
            <div className="flow-track"><i /><i /></div><div className="flow-junction"><span /><b>FEATURE<br />FUSION</b></div>
            <div className="flow-track lower"><i /><i /></div>
            <div className="flow-label bottom"><ScanLine size={15} /><span><b>LOCAL STREAM</b>局部纹理流</span></div>
          </div>

          <div className="tree-column">
            <div className="module-caption"><span>DIAGNOSIS PIPELINE</span><i /></div>
            <div className="pipeline-tree"><div className="tree-trunk"><span className="trunk-pulse" /></div>
              <div className="tree-nodes">
                {WORKFLOW.map((step) => {
                  const Icon = step.icon;
                  return (
                    <button className={`tree-node ${step.ready ? 'ready' : 'planned'}`} key={step.number} onClick={() => setStationOpen(true)}>
                      <span className="branch-line" /><span className="node-index tech-font">{step.number}</span><span className="node-icon"><Icon size={20} /></span>
                      <span className="node-copy"><small>STEP // {step.number}</small><strong>{step.label}</strong><em>{step.ready ? 'MODULE READY' : 'NEXT PHASE'}</em></span>
                      {step.ready && <Check className="node-check" size={15} />}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="performance-deck">
          <div className="performance-title"><span>VALIDATION PERFORMANCE</span><small>HELD-OUT REPORT / MODEL V1.1.0</small></div>
          <div className="metric-rack">{METRICS.map(([label, value, english]) => <div className="metric-cell" key={english}><small>{english}</small><strong className="tech-font">{value}</strong><span>{label}</span></div>)}</div>
          <div className="system-console">
            <p><span>[SYSTEM]</span> 模型权重加载完毕...</p><p><span>[RESNET]</span> 残差特征引擎就绪...</p>
            <p><span>[STATUS]</span> 点击核心或节点进入筛查工作站<span className="cursor">_</span></p>
          </div>
          <button className="launch-button" onClick={() => setStationOpen(true)}><BrainCircuit size={18} /><span>LAUNCH WORKSTATION</span></button>
        </section>
        <p className="dashboard-disclaimer"><ShieldCheck size={14} /> EDUCATIONAL RESEARCH SYSTEM · NOT A MEDICAL DEVICE</p>
      </main>

      <AnimatePresence>
        {stationOpen && (
          <motion.div className="station-layer" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <motion.section className="clinical-station" initial={{ scale: 0.985, y: 12 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.985, y: 12 }}>
              <header className="station-header">
                <div><span className="station-logo"><Eye size={20} /></span><div><strong>Fundus DX 临床筛查工作站</strong><small>RESNET18 FUNDUS CLASSIFICATION CONSOLE</small></div></div>
                <div className="station-header-meta"><span><i /> ENGINE ONLINE</span><b>{caseId}</b><button onClick={() => setStationOpen(false)} aria-label="关闭工作站"><X size={20} /></button></div>
              </header>

              <div className="patient-strip">
                <label><span>PATIENT NAME / 患者姓名</span><input value={patientName} onChange={(event) => setPatientName(event.target.value)} placeholder="输入患者姓名或匿名" /></label>
                <label><span>CASE ID / 病例编号</span><input value={caseId} readOnly /></label>
                <div><span>EXAMINATION</span><strong>彩色眼底照相 · 4 类 AI 筛查</strong></div><div><span>MODEL</span><strong>ResNet18 · v1.1.0</strong></div>
              </div>

              <div className="station-workbench">
                <section className="imaging-viewport">
                  <div className="viewport-title"><span>ORIGINAL</span><small>原始眼底影像</small>{file && <button onClick={reset}><RotateCcw size={13} /> RESET</button>}</div>
                  <div className={`image-stage ${preview ? 'loaded' : ''} ${dragging ? 'dragging' : ''}`}
                    onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                    onDrop={(event) => { event.preventDefault(); setDragging(false); processFile(event.dataTransfer.files?.[0]); }}
                    onClick={() => !preview && fileInputRef.current?.click()} role="button" tabIndex={0}>
                    <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleFileChange} hidden /><span className="scan-line" />
                    {preview ? <><img src={preview} alt="待分析眼底照片" /><button className="replace-image" onClick={(event) => { event.stopPropagation(); fileInputRef.current?.click(); }}>更换影像</button></>
                      : <div className="upload-prompt"><UploadCloud size={38} /><strong>[ 1 ] 导入眼底影像</strong><p>拖放或点击选择 JPG / PNG / WEBP</p><small>MAX FILE SIZE 12 MB</small></div>}
                  </div>
                  <div className="sample-bank"><span>DEMO DATA</span>{SAMPLES.map((sample) => <button key={sample.filename} onClick={() => loadSample(sample)}><img src={sample.src} alt="" /><b>{sample.label}</b></button>)}</div>
                </section>

                <section className="analysis-viewport">
                  <div className="viewport-title"><span>ANALYSIS</span><small>AI 智能分析</small><em>SOFTMAX · 4 CLASSES</em></div>
                  <div className="analysis-screen"><AnimatePresence mode="wait">
                    {loading ? <motion.div className="engine-state" key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <span className="engine-orbit"><BrainCircuit size={34} /></span><strong>正在进行特征解码...</strong><p>RESIDUAL FEATURE EXTRACTION IN PROGRESS</p><div className="engine-progress"><i /></div>
                    </motion.div> : prediction ? <motion.div className="diagnosis-readout" key="result" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                      <div className="diagnosis-hero"><div><small>AI SCREENING CONCLUSION</small><h2>{resultMeta.label}</h2><p>{resultMeta.note}</p></div>
                        <div className="score-ring" style={{ '--score': `${prediction.confidence * 360}deg` }}><span><b>{(prediction.confidence * 100).toFixed(1)}</b>%<small>CONFIDENCE</small></span></div></div>
                      <div className={`certainty-alert ${confidence.tone}`}><CircleAlert size={15} /><span><b>{confidence.label}</b>{confidence.message}</span></div>
                      <div className="class-output"><div className="output-head"><span>CLASS PROBABILITY</span><small>MODEL OUTPUT</small></div>
                        {sortedProbabilities.map(([className, probability]) => {
                          const meta = CLASS_META[className]; const top = className === prediction.prediction;
                          return <div className={`class-row ${top ? 'top' : ''}`} key={className}><span><b>{meta.label}</b><small>{meta.short}</small></span><div><i style={{ width: `${Math.max(0.5, probability * 100)}%` }} /></div><strong>{(probability * 100).toFixed(1)}%</strong></div>;
                        })}
                      </div>
                      <div className="clinical-advice"><ShieldCheck size={18} /><span><b>临床复核建议</b><p>{resultMeta.next}</p></span></div>
                      <div className="runtime-row"><span>API {latency} ms</span><span>INFERENCE {prediction.inference_ms} ms</span><span>MODEL {prediction.model_version}</span></div>
                    </motion.div> : <motion.div className="engine-state waiting" key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                      <span className="engine-orbit"><Layers3 size={34} /></span><strong>等待引擎分析...</strong><p>IMPORT FUNDUS IMAGE TO INITIALIZE</p>
                      <div className="idle-spectrum">{[42,68,36,74,52,82,45,62,34,70].map((height,index) => <i key={index} style={{ height: `${height}%` }} />)}</div>
                    </motion.div>}
                  </AnimatePresence></div>
                </section>
              </div>

              {error && <div className="station-error" role="alert"><CircleAlert size={15} />{error}</div>}
              <div className="station-controls">
                <button className={file ? 'complete' : 'active'} onClick={() => fileInputRef.current?.click()}><span>[ 1 ]</span><FileImage size={16} />影像导入</button>
                <button className={prediction ? 'complete' : file ? 'active primary' : ''} onClick={runScreening} disabled={!file || loading}><span>[ 2 ]</span>{loading ? <LoaderCircle className="spin" size={16} /> : <BrainCircuit size={16} />}智能筛查</button>
                <button disabled><span>[ 3 ]</span><LocateFixed size={16} />病灶定位<em>PHASE 02</em></button>
                <button disabled><span>[ 4 ]</span><FileText size={16} />生成报告<em>PHASE 03</em></button>
              </div>
              <footer className="station-footer"><p><CircleAlert size={14} /> 本系统仅供科研与教学演示，输出不构成临床诊断或治疗建议。</p><span className="tech-font"><Database size={13} /> FUNDUS DX CORE · SYSTEM READY</span></footer>
            </motion.section>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
export default App;
