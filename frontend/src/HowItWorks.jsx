import { BrainCircuit, FileCheck2, LocateFixed, ScanEye } from 'lucide-react';

const ROADMAP = [
  {
    number: '01',
    title: '分类工作站',
    status: '本次完成',
    icon: BrainCircuit,
    body: '病例信息、影像导入、服务状态、四分类推理、概率分布与不确定性提示。',
  },
  {
    number: '02',
    title: '病灶可解释性',
    status: '下一阶段',
    icon: LocateFixed,
    body: '接入 Grad-CAM 热力图、原图叠加、关键区域标注与图像质量控制。',
  },
  {
    number: '03',
    title: '临床报告',
    status: '规划中',
    icon: FileCheck2,
    body: '生成结构化 PDF 报告，包含筛查结论、图像、概率、复核意见与审计信息。',
  },
];

function HowItWorks() {
  return (
    <section className="system-section" id="roadmap">
      <div className="section-heading">
        <div>
          <span className="eyebrow">SYSTEM EVOLUTION</span>
          <h2>从分类模型到完整工作站</h2>
        </div>
        <p>每个阶段都以真实能力为准，不展示尚未实现的医学结果。</p>
      </div>

      <div className="roadmap-grid">
        {ROADMAP.map((item) => {
          const Icon = item.icon;
          return (
            <article className={item.number === '01' ? 'current' : ''} key={item.number}>
              <div className="roadmap-head"><span>{item.number}</span><Icon size={20} /></div>
              <small>{item.status}</small>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          );
        })}
      </div>

      <div className="boundary-card">
        <div className="boundary-icon"><ScanEye size={25} /></div>
        <div>
          <span className="eyebrow">CURRENT SYSTEM BOUNDARY</span>
          <h3>当前处理的是彩色眼底照片，不是 OCT。</h3>
          <p>
            现有模型可在 AMD、白内障、糖尿病视网膜病变和正常四个类别间进行分类。
            青光眼、视网膜脱离等范围外疾病不能被可靠排除；上线临床场景前还需要外部验证、
            图像质量控制、阈值校准、人工复核和合规审批。
          </p>
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;
