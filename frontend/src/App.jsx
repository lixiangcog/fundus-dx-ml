import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Check,
  CircleAlert,
  Eye,
  FileImage,
  FileText,
  Github,
  ImagePlus,
  Layers3,
  LoaderCircle,
  LocateFixed,
  RotateCcw,
  ShieldCheck,
  UploadCloud,
} from 'lucide-react';
import HowItWorks from './HowItWorks';

const API_URL = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);
const MAX_FILE_SIZE = 12 * 1024 * 1024;

const SAMPLES = [
  { src: '/samples/normal.jpg', label: '正常样例', filename: 'sample_normal.jpg' },
  { src: '/samples/cataract.jpg', label: '白内障样例', filename: 'sample_cataract.jpg' },
  {
    src: '/samples/diabetic_retinopathy.jpg',
    label: '糖网样例',
    filename: 'sample_diabetic_retinopathy.jpg',
  },
];

const CLASS_META = {
  amd: {
    label: '年龄相关性黄斑变性',
    short: 'AMD',
    note: '图像特征更接近模型训练集中的 AMD 样本。',
    next: '建议结合黄斑区检查、OCT 等临床信息由眼科医生复核。',
  },
  cataract: {
    label: '白内障',
    short: 'Cataract',
    note: '图像特征更接近模型训练集中的白内障样本。',
    next: '建议结合裂隙灯检查、视力情况与晶状体混浊程度复核。',
  },
  diabetic_retinopathy: {
    label: '糖尿病视网膜病变',
    short: 'DR',
    note: '图像特征更接近模型训练集中的糖尿病视网膜病变样本。',
    next: '建议由眼科医生复核病变分期，并结合糖尿病病史制定随访方案。',
  },
  normal: {
    label: '未见模型已知异常',
    short: 'Normal',
    note: '在四个训练类别中，模型更倾向于正常类别。',
    next: '结果不能排除训练范围之外的眼病；如有症状，仍应接受专业检查。',
  },
};

const WORKFLOW = [
  { number: '01', label: '影像导入', icon: FileImage, phase: 1 },
  { number: '02', label: '智能筛查', icon: BrainCircuit, phase: 1 },
  { number: '03', label: '病灶定位', icon: LocateFixed, phase: 2 },
  { number: '04', label: '临床报告', icon: FileText, phase: 3 },
];

function createCaseId() {
  const date = new Date().toISOString().slice(0, 10).replaceAll('-', '');
  return `FDX-${date}-${Math.floor(1000 + Math.random() * 9000)}`;
}

function confidenceMeta(confidence) {
  if (confidence >= 0.8) {
    return { label: '高模型置信度', tone: 'high', message: '四分类结果较集中，仍需专业人员结合临床资料复核。' };
  }
  if (confidence >= 0.6) {
    return { label: '中等模型置信度', tone: 'medium', message: '结果存在一定不确定性，建议检查图像质量并人工复核。' };
  }
  return { label: '低模型置信度', tone: 'low', message: '结果不确定，建议重新采集影像或直接转人工判读。' };
}

function App() {
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
      .then((response) => {
        if (!response.ok) throw new Error('Service unavailable');
        return response.json();
      })
      .then((data) => setServiceStatus(data.model_loaded ? 'online' : 'degraded'))
      .catch((requestError) => {
        if (requestError.name !== 'AbortError') setServiceStatus('offline');
      })
      .finally(() => clearTimeout(timer));

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, []);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const processFile = (selectedFile) => {
    if (!selectedFile?.type?.startsWith('image/')) {
      setError('请选择 JPG、PNG 或 WebP 格式的眼底图像。');
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('图像不能超过 12 MB，请压缩后重试。');
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setPrediction(null);
    setLatency(null);
    setError(null);
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) processFile(selectedFile);
    event.target.value = '';
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const droppedFile = event.dataTransfer.files?.[0];
    if (droppedFile) processFile(droppedFile);
  };

  const loadSample = async (sample) => {
    try {
      const response = await fetch(sample.src);
      const blob = await response.blob();
      processFile(new File([blob], sample.filename, { type: 'image/jpeg' }));
    } catch {
      setError('示例图像加载失败，请稍后重试。');
    }
  };

  const runScreening = async () => {
    if (!file || loading) return;
    setLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    const startedAt = performance.now();

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setLatency(Math.round(performance.now() - startedAt));
      setPrediction(response.data);
      setServiceStatus('online');
    } catch (requestError) {
      const detail = requestError.response?.data?.detail;
      setError(detail || '无法连接筛查引擎。请确认后端已启动，然后重试。');
      if (!requestError.response) setServiceStatus('offline');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setLatency(null);
    setError(null);
  };

  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a)
    : [];
  const confidence = prediction ? confidenceMeta(prediction.confidence) : null;
  const resultMeta = prediction ? CLASS_META[prediction.prediction] : null;
  const completedStep = prediction ? 2 : file ? 1 : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Fundus DX 首页">
          <span className="brand-mark"><Eye size={20} strokeWidth={1.8} /></span>
          <span>
            <strong>Fundus DX</strong>
            <small>智能眼底筛查工作站</small>
          </span>
        </a>

        <div className="topbar-actions">
          <span className={`service-status ${serviceStatus}`}>
            <i />
            {serviceStatus === 'online' && 'AI 引擎在线'}
            {serviceStatus === 'degraded' && '模型未就绪'}
            {serviceStatus === 'offline' && 'AI 引擎离线'}
            {serviceStatus === 'checking' && '正在连接引擎'}
          </span>
          <a
            className="icon-link"
            href="https://github.com/lixiangcog/fundus-dx-ml"
            target="_blank"
            rel="noreferrer"
            aria-label="查看 GitHub 源码"
          >
            <Github size={18} />
            <span>源码</span>
          </a>
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow"><Activity size={14} /> FUNDUS PHOTOGRAPHY · AI SCREENING</span>
            <h1>让一张眼底照片，进入<br />清晰、可复核的筛查流程。</h1>
            <p>
              基于现有 ResNet18 四分类模型搭建的第一阶段工作站。支持彩色眼底照片导入、
              模型推理、完整概率分布与不确定性提示。
            </p>
            <a className="hero-cta" href="#workstation">
              进入筛查工作站 <ArrowRight size={17} />
            </a>
          </div>

          <div className="model-card" aria-label="模型信息">
            <div className="model-card-head">
              <span>MODEL CORE</span>
              <ShieldCheck size={18} />
            </div>
            <strong>ResNet18</strong>
            <p>ImageNet 迁移学习 · 4 类眼底图像分类</p>
            <dl>
              <div><dt>验证准确率</dt><dd>97.7%</dd></div>
              <div><dt>验证集</dt><dd>871 images</dd></div>
              <div><dt>模型输入</dt><dd>224 × 224</dd></div>
              <div><dt>部署形态</dt><dd>FastAPI</dd></div>
            </dl>
            <span className="model-disclaimer">研究演示模型 · 非医疗器械</span>
          </div>
        </section>

        <section className="workflow" aria-label="筛查流程">
          {WORKFLOW.map((step, index) => {
            const StepIcon = step.icon;
            const active = step.phase === 1 && completedStep < 2 && index === completedStep;
            const complete = step.phase === 1 && completedStep > index;
            const planned = step.phase > 1;
            return (
              <div className={`workflow-step ${active ? 'active' : ''} ${complete ? 'complete' : ''}`} key={step.number}>
                <span className="workflow-icon">
                  {complete ? <Check size={17} /> : <StepIcon size={17} />}
                </span>
                <span><small>STEP {step.number}</small><strong>{step.label}</strong></span>
                {planned && <em>阶段 {step.phase}</em>}
              </div>
            );
          })}
        </section>

        <section className="workstation" id="workstation">
          <div className="section-heading">
            <div>
              <span className="eyebrow">CLINICAL WORKSTATION · PHASE 01</span>
              <h2>智能眼底筛查</h2>
            </div>
            <p>本阶段仅接收彩色眼底照片，不支持 OCT 影像。</p>
          </div>

          <div className="station-grid">
            <div className="case-panel">
              <div className="panel-title">
                <span><FileImage size={17} /> 病例与影像</span>
                {file && <button className="text-button" onClick={reset}><RotateCcw size={14} /> 重置</button>}
              </div>

              <div className="patient-fields">
                <label>
                  <span>患者姓名 <em>可选</em></span>
                  <input
                    value={patientName}
                    onChange={(event) => setPatientName(event.target.value)}
                    placeholder="请输入或匿名"
                    maxLength={40}
                  />
                </label>
                <label>
                  <span>病例编号</span>
                  <input value={caseId} readOnly />
                </label>
              </div>

              <div
                className={`upload-stage ${preview ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => !preview && fileInputRef.current?.click()}
                onKeyDown={(event) => {
                  if (!preview && (event.key === 'Enter' || event.key === ' ')) {
                    event.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                role="button"
                tabIndex={0}
                aria-label="上传彩色眼底照片"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  hidden
                />
                {preview ? (
                  <>
                    <img src={preview} alt="待分析的眼底照片" />
                    <div className="image-overlay">
                      <span>ORIGINAL FUNDUS</span>
                      <button onClick={(event) => { event.stopPropagation(); fileInputRef.current?.click(); }}>
                        更换影像
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="upload-empty">
                    <span className="upload-symbol"><UploadCloud size={29} /></span>
                    <strong>导入彩色眼底照片</strong>
                    <p>拖放至此处，或点击选择本地文件</p>
                    <small>JPG / PNG / WEBP · 最大 12 MB</small>
                  </div>
                )}
              </div>

              <div className="sample-row">
                <span>无影像？使用脱敏演示样例</span>
                <div>
                  {SAMPLES.map((sample) => (
                    <button key={sample.filename} onClick={() => loadSample(sample)} title={sample.label}>
                      <img src={sample.src} alt="" />
                      <span>{sample.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {error && (
                <div className="inline-error" role="alert">
                  <CircleAlert size={16} /> <span>{error}</span>
                </div>
              )}

              <button className="primary-action" onClick={runScreening} disabled={!file || loading}>
                {loading ? <><LoaderCircle className="spin" size={18} /> AI 引擎分析中</> : <><BrainCircuit size={18} /> 开始智能筛查</>}
              </button>
            </div>

            <div className="analysis-panel">
              <div className="panel-title">
                <span><Layers3 size={17} /> AI 分析结果</span>
                <span className="panel-mode">4-CLASS CLASSIFIER</span>
              </div>

              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.div
                    className="analyzing-state"
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    aria-live="polite"
                  >
                    <div className="analysis-orbit"><BrainCircuit size={33} /></div>
                    <strong>正在提取眼底图像特征</strong>
                    <p>完成标准化预处理后，模型将输出四个类别的概率分布。</p>
                    <div className="progress-line"><i /></div>
                  </motion.div>
                ) : prediction ? (
                  <motion.div
                    className="result-view"
                    key="result"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <div className="result-summary">
                      <div>
                        <span className="result-kicker">AI SCREENING RESULT</span>
                        <h3>{resultMeta.label}</h3>
                        <p>{resultMeta.note}</p>
                      </div>
                      <div className="confidence-dial" style={{ '--score': `${prediction.confidence * 360}deg` }}>
                        <div><strong>{(prediction.confidence * 100).toFixed(1)}</strong><span>%</span><small>置信度</small></div>
                      </div>
                    </div>

                    <div className={`confidence-note ${confidence.tone}`}>
                      <span><CircleAlert size={16} /> {confidence.label}</span>
                      <p>{confidence.message}</p>
                    </div>

                    <div className="probability-list">
                      <div className="subheading"><span>完整概率分布</span><small>SOFTMAX OUTPUT</small></div>
                      {sortedProbabilities.map(([className, probability]) => {
                        const isTop = className === prediction.prediction;
                        const meta = CLASS_META[className];
                        return (
                          <div className={`probability-row ${isTop ? 'top' : ''}`} key={className}>
                            <div className="probability-label">
                              <span>{meta.label}</span>
                              <small>{meta.short}</small>
                            </div>
                            <div className="probability-track"><i style={{ width: `${Math.max(0.5, probability * 100)}%` }} /></div>
                            <strong>{(probability * 100).toFixed(1)}%</strong>
                          </div>
                        );
                      })}
                    </div>

                    <div className="review-note">
                      <ShieldCheck size={18} />
                      <div><strong>建议下一步</strong><p>{resultMeta.next}</p></div>
                    </div>

                    <div className="inference-meta">
                      <span>病例 {caseId}</span>
                      <span>API 往返 {latency} ms</span>
                      {prediction.inference_ms != null && <span>模型推理 {prediction.inference_ms} ms</span>}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div className="empty-result" key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <span className="empty-result-icon"><ImagePlus size={28} /></span>
                    <strong>等待影像分析</strong>
                    <p>导入一张彩色眼底照片并启动筛查后，这里将显示预测类别、置信度和完整概率分布。</p>
                    <div className="empty-bars" aria-hidden="true">
                      {[72, 49, 35, 22].map((width) => <i key={width} style={{ width: `${width}%` }} />)}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <div className="station-footer">
            <CircleAlert size={17} />
            <p><strong>重要提示：</strong>本系统用于科研与教学演示，输出是模型筛查结果，不构成临床诊断或治疗建议。</p>
          </div>
        </section>

        <HowItWorks />
      </main>

      <footer className="footer">
        <div><strong>Fundus DX</strong><span>从可运行分类器，逐步演进为可信的眼底 AI 工作站。</span></div>
        <span>ResNet18 · PyTorch · FastAPI · React</span>
      </footer>
    </div>
  );
}

export default App;
