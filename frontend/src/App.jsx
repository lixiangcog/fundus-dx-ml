import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, BrainCircuit, Check, CircleAlert, CircleDot,
  FileImage, ImagePlus, Layers3, LoaderCircle, Microscope,
  Network, Play, RotateCcw, ScanLine, Sparkles, UploadCloud,
} from 'lucide-react';
import AMDWorkspace from './AMDWorkspace';
import SystemicWorkspace from './SystemicWorkspace';
import ModuleLog from './ModuleLog';
import { useModuleLog } from './useModuleLog';
import './overview.css';
import './theme-refresh.css';

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);
const MAX_FILE_SIZE = 12 * 1024 * 1024;
const ICONS = { 'quality-enhancement': Sparkles, 'structure-segmentation': Layers3, 'disease-screening': ScanLine, 'vascular-quantification': Network, 'fundus-lesion-quantification': CircleDot, 'oct-fluid-quantification': Activity, 'amd-oct-pathology': Microscope, 'amd-fundus-risk-factors': BrainCircuit };
const DESCRIPTIONS = {
  'quality-enhancement':'改善噪声与清晰度',
  'structure-segmentation':'标记视网膜层结构与液体区域',
  'disease-screening':'给出常见眼底疾病的筛查概率',
  'vascular-quantification':'提取血管并计算密度、长度与分支',
  'fundus-lesion-quantification':'定位并量化四类常见眼底病灶',
  'oct-fluid-quantification':'定位液体区域并计算面积与高度',
  'amd-oct-pathology':'识别四类 AMD 相关病灶并生成注意力图',
  'amd-fundus-risk-factors':'评估玻璃膜疣、色素与萎缩风险因子',
};
const CLASS_NAMES = { amd:'年龄相关性黄斑变性', cataract:'白内障', diabetic_retinopathy:'糖尿病视网膜病变', normal:'未见已知异常' };
const SYSTEMIC_WORKSPACES = {
  'eye-age': 'eye-age',
  cardiovascular: 'cardiovascular-retina',
  cerebrovascular: 'cerebrovascular-retina',
};
const FALLBACK = [
  { id:'quality-enhancement',number:'01',title:'质量增强',english:'QUALITY ENHANCEMENT',default_modality:'眼底彩照',modalities:['OCT','OCTA','眼底彩照'],engine:'多模态质量增强 · v2',engine_type:'pretrained_model',method:'OCT 专用模型 + OCTA / 眼底彩照自适应增强',sample_id:'amd-v0-fundus',sample_url:'/research-samples/amd-v0-fundus',source_url:'https://github.com/DeweiHu/OCT_DDPM',license:'MIT',output:'增强影像 + 配对 PSNR / SSIM / 边缘保持' },
  { id:'structure-segmentation',number:'02',title:'OCT 结构分割',english:'OCT STRUCTURE SEGMENTATION',default_modality:'OCT',modalities:['OCT'],engine:'Duke residual U-Net · v1',engine_type:'trained_model',method:'8 层结构 + 液体的十类像素级分割',sample_id:'oct-structure-duke-s03-4',sample_url:'/research-samples/oct-structure-duke-s03-4',source_url:'https://github.com/ClinicalAI/MIRAGE',license:'Research model / CC BY 4.0 data release',output:'层结构叠加 + Dice / IoU / 厚度代理' },
  { id:'oct-fluid-quantification',number:'03',title:'OCT 液体定位量化',english:'OCT FLUID QUANTIFICATION',default_modality:'OCT',modalities:['OCT'],engine:'Duke residual U-Net · v1',engine_type:'trained_model',method:'视网膜液体像素定位、组件与面积比例量化',sample_id:'oct-fluid-duke-s01-5',sample_url:'/research-samples/oct-fluid-duke-s01-5',source_url:'https://github.com/ClinicalAI/MIRAGE',license:'Research model / CC BY 4.0 data release',output:'液体热区 + Dice / IoU / 面积 / 最大高度' },
  { id:'amd-oct-pathology',number:'04',title:'OCT AMD 病灶分类',english:'OCT AMD PATHOLOGY',default_modality:'OCT',modalities:['OCT'],engine:'EfficientNet-B3 OCT classifier',engine_type:'pretrained_model',method:'脉络膜新生血管 / 水肿 / 玻璃膜疣 / 正常四分类 + 注意力定位',sample_id:'amd-v0-oct',sample_url:'/research-samples/amd-v0-oct',source_url:'https://huggingface.co/tomalmog/oct-retinal-classifier',license:'Research model / Kermany dataset',output:'四类病灶概率 + 注意力图' },
  { id:'vascular-quantification',number:'05',title:'OCTA 微血管定量',english:'OCTA VASCULAR QUANTIFICATION',default_modality:'OCTA',modalities:['OCTA'],engine:'Pretrained DynUNet · epoch 30',engine_type:'pretrained_model',method:'深度血管分割 + 骨架、分支、密度量化',sample_id:'octa-vessels-sgan-232653',sample_url:'/research-samples/octa-vessels-sgan-232653',source_url:'https://github.com/aiforvision/OCTA-autosegmentation',license:'MIT',output:'血管掩膜 + Dice / IoU + 微血管形态学' },
  { id:'disease-screening',number:'06',title:'眼底疾病筛查',english:'FUNDUS DISEASE SCREENING',default_modality:'眼底彩照',modalities:['眼底彩照'],engine:'FundusDx ResNet18 · v2',engine_type:'trained_model',method:'AMD / 白内障 / 糖网 / 正常四分类 + CAM',sample_id:'fundus-screen-idrid-67',sample_url:'/research-samples/fundus-screen-idrid-67',source_url:'https://github.com/lixiangcog/fundus-dx-ml',license:'Project model',output:'筛查概率 + CAM（非病灶掩膜）' },
  { id:'fundus-lesion-quantification',number:'07',title:'彩照病灶定位量化',english:'FUNDUS LESION QUANTIFICATION',default_modality:'眼底彩照',modalities:['眼底彩照'],engine:'U-Net · SE-ResNeXt50',engine_type:'pretrained_model',method:'棉絮斑 / 硬性渗出 / 出血 / 微动脉瘤像素分割',sample_id:'fundus-lesions-idrid-67',sample_url:'/research-samples/fundus-lesions-idrid-67',source_url:'https://github.com/ClementP/fundus-lesions-segmentation',license:'MIT',output:'四类病灶掩膜 + 面积 / 组件 / Dice / IoU' },
  { id:'amd-fundus-risk-factors',number:'08',title:'彩照 AMD 风险因子',english:'FUNDUS AMD RISK FACTORS',default_modality:'眼底彩照',modalities:['眼底彩照'],engine:'DeepSeeNet five-head ONNX',engine_type:'pretrained_model',method:'玻璃膜疣 / 色素异常 / 晚期 AMD / 地图样萎缩五项风险因子',sample_id:'amd-v0-fundus',sample_url:'/research-samples/amd-v0-fundus',source_url:'https://github.com/ncbi-nlp/DeepSeeNet',license:'Research use / NCBI DeepSeeNet',output:'五项风险概率 + 病灶候选定位' },
]

function fileName(id) { return `demo_${id.replaceAll('-','_')}.png`; }
function App() {
  const [workspace, setWorkspace] = useState('imaging');
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
  const { entries: imagingLogs, write: writeImagingLog, writeMany: writeImagingLogs, clear: clearImagingLog } = useModuleLog('影像分析');
  const fileInput = useRef(null);
  const workbenchRef = useRef(null);
  const active = capabilities.find((item) => item.id === activeId) || capabilities[0];

  useEffect(() => {
    fetch(`${API_URL}/compute/ensure`, { method:'POST' }).catch(() => null);
    Promise.all([fetch(`${API_URL}/capabilities`).then((r) => r.json()), fetch(`${API_URL}/health`).then((r) => r.json())])
      .then(([catalog, health]) => {
        setCapabilities(catalog.capabilities); setService(health.status === 'ok' ? 'online' : 'degraded');
        writeImagingLogs([
          { level:'command', channel:'SHELL', message:'fundus-dx status --format terminal' },
          { level:'success', channel:'API', message:`catalog loaded; capabilities=${catalog.capabilities.length}`, detail:`version=${catalog.version || health.version || 'unknown'}` },
          { level:health.imaging_service?.status === 'ready' ? 'success' : 'warning', channel:'CUDA', message:`device=${health.imaging_service?.live?.device || health.device || 'unknown'}; service=${health.imaging_service?.status || 'unknown'}`, detail:`job=${health.imaging_service?.live?.job_id || 'n/a'}; pipelines=${health.pipelines_ready || catalog.capabilities.length}` },
        ]);
      })
      .catch(() => { setService('offline'); writeImagingLog('error', 'GET /health -> connection failed', '请检查网页服务和 SSH 转发', 'HTTP'); });
  }, [writeImagingLog, writeImagingLogs]);

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
      writeImagingLog('info', `sample=${capability.sample_id || capability.id}`, `modality=${capability.default_modality}; bytes=${blob.size}`, 'INPUT');
    } catch { setError('默认病例加载失败，请刷新页面后重试。'); writeImagingLog('error', `GET ${capability.sample_url} -> invalid image`, '默认病例加载失败', 'HTTP'); }
  };

  // The sample URL is the stable identity of the selected default case.
  // eslint-disable-next-line react-hooks/exhaustive-deps, react-hooks/set-state-in-effect
  useEffect(() => { loadDefault(active); }, [activeId, active?.sample_url]);
  useEffect(() => () => { if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview); }, [preview]);

  const chooseCapability = (id) => {
    if (loading || id === activeId) return;
    const selected = capabilities.find((item) => item.id === id);
    setActiveId(id); writeImagingLog('command', `fundus-dx select --task ${id}`, selected?.title || id, 'SHELL');
  };
  const enterCapability = (id) => {
    if (loading) return;
    if (id !== activeId) {
      const selected = capabilities.find((item) => item.id === id);
      setActiveId(id); writeImagingLog('command', `fundus-dx select --task ${id}`, selected?.title || id, 'SHELL');
    }
    window.requestAnimationFrame(() => workbenchRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  };
  const processFile = (nextFile) => {
    if (!nextFile?.type?.startsWith('image/')) { setError('请选择 JPG、PNG 或 WebP 影像。'); writeImagingLog('warning', '输入影像格式不受支持', '允许 JPG、PNG 或 WebP'); return; }
    if (nextFile.size > MAX_FILE_SIZE) { setError('影像不能超过 12 MB。'); writeImagingLog('warning', '输入影像超过大小限制', '最大允许 12 MB'); return; }
    installFile(nextFile, URL.createObjectURL(nextFile));
    writeImagingLog('info', `upload accepted; type=${nextFile.type}`, `bytes=${nextFile.size}; task=${active.id}`, 'INPUT');
  };
  const run = async () => {
    if (!file || loading) return;
    setLoading(true); setResult(null); setError('');
    writeImagingLogs([
      { level:'command', channel:'SHELL', message:`fundus-dx infer --task ${active.id} --device cuda:0` },
      { level:'info', channel:'INPUT', message:`source=${isDefault ? 'research-sample' : 'user-upload'}; mime=${file.type || 'image/unknown'}`, detail:`bytes=${file.size}; modality=${active.default_modality}` },
      { level:'run', channel:'QUEUE', message:'request accepted; dispatching to GPU worker', detail:`pipeline=${active.id}` },
      { level:'run', channel:'CUDA', message:'inference_mode=true; device=cuda:0', detail:'preprocess -> forward -> postprocess' },
    ]);
    const data = new FormData(); data.append('file', file);
    if (isDefault && active.sample_id) data.append('sample_id', active.sample_id);
    try {
      const response = await axios.post(`${API_URL}/analyze/${active.id}`, data);
      setResult(response.data); setService('online');
      const output = response.data;
      const metricRows = (output.metrics || []).map((metric) => ({
        level:'info', channel:'METRIC', message:`${metric.label}=${String(metric.value)}${metric.unit || ''}`, detail:metric.detail || '',
      }));
      writeImagingLogs([
        { level:'success', channel:'HTTP', message:`POST /analyze/${active.id} -> 200 OK` },
        { level:'info', channel:'INPUT', message:`decoded=${output.input?.width || '?'}x${output.input?.height || '?'}`, detail:`reference_applied=${Boolean(output.input?.reference_applied)}` },
        { level:'success', channel:'BACKEND', message:`pipeline=${output.pipeline?.id || active.id}; real_inference=${Boolean(output.real_inference)}`, detail:`model_version=${output.model_version || 'unknown'}` },
        ...metricRows,
        { level:output.quality?.status === 'failed' ? 'warning' : 'success', channel:'QC', message:`status=${output.quality?.status || 'unverified'}`, detail:output.quality?.label || 'quality metadata unavailable' },
        { level:'success', channel:'DONE', message:`${active.id} completed`, detail:`runtime_ms=${output.runtime_ms}; output=image+metrics` },
      ]);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || '推理服务暂时不可用，请稍后重试。');
      writeImagingLog('error', `POST /analyze/${active.id} -> ${requestError.response?.status || 'NETWORK_ERROR'}`, requestError.response?.data?.detail || '无法连接推理服务', 'HTTP');
      if (!requestError.response) setService('offline');
    } finally { setLoading(false); }
  };

  return (
    <div className="app-shell">
      <div className="ambient-grid" aria-hidden="true"><i /><i /><i /><span /></div>
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><Microscope size={20} /></span><div><strong>视界<strong className="accent">智析</strong></strong><small>多模态眼科影像分析平台</small></div></div>
        <nav className="workspace-switch">
          <button className={workspace === 'imaging' ? 'active' : ''} onClick={() => setWorkspace('imaging')}>影像分析</button>
          <button className={workspace === 'amd' ? 'active' : ''} onClick={() => setWorkspace('amd')}>AMD 随访</button>
          <button className={workspace === 'eye-age' ? 'active' : ''} onClick={() => setWorkspace('eye-age')}>眼龄</button>
          <button className={workspace === 'cardiovascular' ? 'active' : ''} onClick={() => setWorkspace('cardiovascular')}>眼观心血管</button>
          <button className={workspace === 'cerebrovascular' ? 'active' : ''} onClick={() => setWorkspace('cerebrovascular')}>眼观脑血管</button>
        </nav>
        <div className="header-meta">
          <span><small>当前模块</small><b>{workspace === 'imaging' ? '影像分析' : workspace === 'amd' ? '纵向随访' : workspace === 'eye-age' ? '眼龄评估' : workspace === 'cardiovascular' ? '心血管表型' : '脑血管表型'}</b></span>
          <span className={`service ${service}`}><small>系统状态</small><b><i /> {service === 'online' ? '在线' : service === 'offline' ? '离线' : '检查中'}</b></span>
        </div>
      </header>

      <main>
        {workspace === 'amd' ? <AMDWorkspace apiUrl={API_URL}/> : SYSTEMIC_WORKSPACES[workspace] ? <SystemicWorkspace key={workspace} apiUrl={API_URL} moduleId={SYSTEMIC_WORKSPACES[workspace]} /> : <>
        <section className="command-overview">
          <div className="overview-lines" aria-hidden="true"><i /><i /><i /><span /><span /></div>
          <div className="analysis-core">
            <div className="core-ring ring-one" aria-hidden="true"><i /><i /><i /></div>
            <div className="core-ring ring-two" aria-hidden="true" />
            <div className="core-center">
              <small>OCT · OCTA · 眼底彩照</small>
              <strong>多模态影像分析内核</strong>
              <span>增强 · 分割 · 识别 · 定量</span>
            </div>
            <div className="core-scan" aria-hidden="true" />
          </div>

          <div className="capability-map">
            <div className="overview-heading"><span>分析功能</span><b>选择节点进入工作台</b></div>
            <div className="capability-spine" aria-hidden="true" />
            <div className="capability-grid">
              {capabilities.map((item) => {
                const Icon = ICONS[item.id] || Activity; const selected = item.id === activeId;
                return <button key={item.id} className={selected ? 'active' : ''} onClick={() => enterCapability(item.id)}>
                  <span className="capability-number">{item.number}</span>
                  <span className="capability-symbol"><Icon size={18} /></span>
                  <span className="capability-copy"><strong>{item.title}</strong><small>{item.default_modality}</small></span>
                  <i className="state-dot" />
                </button>;
              })}
            </div>
          </div>

        </section>

        <section className="analysis-station" ref={workbenchRef}>
          <div className="station-heading">
            <div><span className="eyebrow">分析工作台</span><h1>选择功能，上传影像，<em>获得可量化结果。</em></h1></div>
          </div>

          <nav className="pipeline-nav" aria-label="分析功能">
            {capabilities.map((item) => {
              const Icon = ICONS[item.id] || Activity; const selected = item.id === activeId;
              return <button key={item.id} className={selected ? 'active' : ''} onClick={() => chooseCapability(item.id)}>
                <span className="pipeline-no">{item.number}</span><span className="pipeline-icon"><Icon size={18} /></span><span><strong>{item.title}</strong></span><i className="state-dot" />
              </button>;
            })}
          </nav>

          <section className="workbench">
          <aside className="engine-panel">
            <div className="panel-kicker"><CircleDot size={13} /> 当前功能</div>
            <h2>{active.number}<span>/</span>{active.title}</h2>
            <div className="modality-tags">{active.modalities.map((modality) => <span key={modality} className={modality === active.default_modality ? 'primary' : ''}>{modality}</span>)}</div>
            <p className="task-summary">{DESCRIPTIONS[active.id]}</p>
            <div className="engine-type"><Check size={13} /><span>功能已就绪</span></div>
          </aside>

          <section className="visual-stage">
            <div className="stage-head"><span><FileImage size={14} /> 输入影像</span><div><b>{isDefault ? '默认病例' : '上传影像'}</b><small>{active.default_modality}</small></div></div>
            <div className={`image-frame ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); processFile(event.dataTransfer.files?.[0]); }}>
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" /><span className="scan-beam" />
              {preview ? <img src={preview} alt={`${active.title}输入影像`} /> : <div className="empty-image"><ImagePlus size={30} />等待影像</div>}
              <div className="image-hud"><span>宽 {result?.input?.width || '---'}</span><span>高 {result?.input?.height || '---'}</span><span>影像预览</span></div>
            </div>
            <div className="input-actions"><button onClick={() => fileInput.current?.click()}><UploadCloud size={15} />上传影像</button><button onClick={() => loadDefault()}><RotateCcw size={14} />恢复默认病例</button><input ref={fileInput} type="file" hidden accept="image/jpeg,image/png,image/webp" onChange={(event) => { if (event.target.files?.[0]) processFile(event.target.files[0]); event.target.value=''; }} /></div>
          </section>

          <div className={`data-bridge ${loading ? 'running' : ''}`} aria-hidden="true"><span /><i /></div>

          <section className="visual-stage output-stage">
            <div className="stage-head"><span><BrainCircuit size={14} /> 分析结果</span><div>{result && <><b>{result.runtime_ms} 毫秒</b><small>运行耗时</small></>}</div></div>
            <div className={`image-frame ${result ? 'has-result' : ''}`}>
              <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
              <AnimatePresence mode="wait">
                {loading ? <motion.div key="loading" className="processing" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}><span><LoaderCircle size={30} /></span><strong>正在分析</strong><small>请稍候</small><i /></motion.div>
                  : result ? <motion.div key="result" className="result-image" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}><img src={result.result_image} alt={`${active.title}分析结果`} /><span className="result-scan" /><div className="result-label"><Check size={12} /> 分析完成</div></motion.div>
                  : <motion.div key="empty" className="waiting" initial={{opacity:0}} animate={{opacity:1}}><span className="radar"><Activity size={28} /></span><strong>等待分析</strong><small>选择默认病例或上传影像后开始</small></motion.div>}
              </AnimatePresence>
            </div>
            <button className="run-button" onClick={run} disabled={!file || loading}><Play size={16} fill="currentColor" /><span>{loading ? '正在分析' : `开始${active.title}`}</span></button>
          </section>

          <aside className="metrics-panel">
            <div className="panel-kicker"><Activity size={13} /> 定量结果</div>
            {result ? <motion.div className="result-data" initial={{opacity:0,y:5}} animate={{opacity:1,y:0}}>
              <h3>{result.summary}</h3>
              {result.quality && <div className={`quality-gate ${result.quality.status}`}>
                <span>{result.quality.status === 'passed' ? <Check size={13}/> : <CircleAlert size={13}/>}<b>{result.quality.label}</b></span>
              </div>}
              <div className="metric-list">{result.metrics.map((metric,index) => <div key={`${metric.label}-${index}`}><span>{metric.label}<small>{metric.detail}</small></span><strong>{String(metric.value)}<em>{metric.unit}</em></strong></div>)}</div>
              {result.probabilities && <div className="probability-mini">{Object.entries(result.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value]) => <div key={name}><span>{CLASS_NAMES[name] || name}</span><i><b style={{width:`${value*100}%`}} /></i><strong>{(value*100).toFixed(1)}%</strong></div>)}</div>}
              <p className="result-notice"><CircleAlert size={14} />{result.notice}</p>
            </motion.div> : <div className="metric-placeholder"><div className="signal-bars">{[24,48,34,70,52,86,42,64,30,74].map((h,i)=><i key={i} style={{height:`${h}%`}} />)}</div><p>运行后在这里查看定量结果。</p></div>}
          </aside>
          </section>
        </section>

        <ModuleLog title={active.title} entries={imagingLogs} onClear={clearImagingLog} running={loading}/>
        {error && <div className="error-banner"><CircleAlert size={16} />{error}</div>}
        </>}
      </main>
    </div>
  );
}

export default App;
