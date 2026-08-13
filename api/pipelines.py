"""Runnable ophthalmic imaging pipelines for the research workbench."""
from __future__ import annotations

import base64
import io
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party" / "relaynet_pytorch"
RELAYNET_CHECKPOINT = THIRD_PARTY_ROOT / "models" / "Exp01" / "relaynet_epoch20.model"
if str(THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

CAPABILITIES = [
    {"id": "quality-enhancement", "number": "01", "title": "质量增强", "english": "QUALITY ENHANCEMENT", "modalities": ["OCT", "OCTA", "眼底彩照"], "default_modality": "OCT", "engine": "Structure-preserving CPU baseline", "engine_type": "algorithm", "method": "CLAHE + 非局部均值去噪 + 结构锐化", "license": "Apache-2.0", "source_url": "https://github.com/opencv/opencv", "sample_url": "/samples/ophthalmic/oct_quality.png", "output": "增强影像 + 对比度/噪声代理指标", "status": "ready"},
    {"id": "structure-segmentation", "number": "02", "title": "结构分割", "english": "STRUCTURE SEGMENTATION", "modalities": ["OCT"], "default_modality": "OCT", "engine": "ReLayNet · epoch 20", "engine_type": "pretrained_model", "method": "九类视网膜区域/层结构像素级分割", "license": "MIT", "source_url": "https://github.com/ai-med/relaynet_pytorch", "sample_url": "/samples/ophthalmic/oct_structure.png", "output": "结构叠加图 + 像素级分区统计", "status": "ready"},
    {"id": "lesion-recognition", "number": "03", "title": "病灶识别", "english": "LESION RECOGNITION", "modalities": ["眼底彩照"], "default_modality": "眼底彩照", "engine": "FundusDx ResNet18 · v2", "engine_type": "trained_model", "method": "AMD / 白内障 / 糖网 / 正常四分类 + CAM 热力图", "license": "Project model", "source_url": "https://github.com/lixiangcog/fundus-dx-ml", "sample_url": "/samples/ophthalmic/fundus_lesion.jpg", "output": "分类概率 + 类激活定位图", "status": "ready"},
    {"id": "vascular-quantification", "number": "04", "title": "微血管定量", "english": "MICROVASCULAR QUANTIFICATION", "modalities": ["OCTA", "眼底彩照"], "default_modality": "OCTA", "engine": "Multi-scale vesselness + morphometry", "engine_type": "algorithm", "method": "多尺度 Hessian 血管响应、骨架化与分形/分支统计", "license": "Apache-2.0 / MIT references", "source_url": "https://github.com/rmaphoh/AutoMorph", "sample_url": "/samples/ophthalmic/octa_vascular.png", "output": "血管叠加图 + 密度/长度/分支/分形维数", "status": "ready"},
]
PIPELINE_INDEX = {item["id"]: item for item in CAPABILITIES}
CLASS_LABELS = {"amd": "年龄相关性黄斑变性", "cataract": "白内障", "diabetic_retinopathy": "糖尿病视网膜病变", "normal": "未见模型已知异常"}
SEGMENT_COLORS = np.array([[20,24,34],[0,235,255],[38,166,255],[111,255,179],[255,219,77],[255,121,198],[174,108,255],[255,134,76],[54,78,104]], dtype=np.uint8)
SEGMENT_NAMES = ["玻璃体区", "ILM", "NFL–IPL", "INL", "OPL", "ONL–ISM", "ISE", "OS–RPE", "RPE 下区"]
_relaynet_model = None


def _limit_image(image: Image.Image, longest: int = 1200) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= longest:
        return image
    ratio = longest / max(image.size)
    return image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)


def _png_data_url(array: np.ndarray) -> str:
    array = np.clip(array, 0, 255).astype(np.uint8)
    encoded_image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB")
    buffer = io.BytesIO()
    encoded_image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _metric(label: str, value, unit: str = "", detail: str = "") -> dict:
    return {"label": label, "value": value, "unit": unit, "detail": detail}


def _gray(image: Image.Image, max_side: int = 900) -> np.ndarray:
    return cv2.cvtColor(np.asarray(_limit_image(image, max_side)), cv2.COLOR_RGB2GRAY)


def quality_enhancement(image: Image.Image, **_) -> dict:
    started = time.perf_counter()
    rgb = np.asarray(_limit_image(image, 1000))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    denoised = cv2.fastNlMeansDenoising(luminance, None, h=5, templateWindowSize=7, searchWindowSize=21)
    equalized = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(denoised)
    blurred = cv2.GaussianBlur(equalized, (0, 0), 1.15)
    sharpened = cv2.addWeighted(equalized, 1.42, blurred, -0.42, 0)
    lab[:, :, 0] = sharpened
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    before_span = float(np.percentile(luminance, 99) - np.percentile(luminance, 1))
    after_span = float(np.percentile(sharpened, 99) - np.percentile(sharpened, 1))
    before_noise = float(np.median(np.abs(cv2.Laplacian(luminance, cv2.CV_32F))))
    after_noise = float(np.median(np.abs(cv2.Laplacian(denoised, cv2.CV_32F))))
    return {"summary": "已完成亮度均衡、噪声抑制与局部结构增强", "result_image": _png_data_url(enhanced), "auxiliary_images": [{"label": "增强前", "image": _png_data_url(rgb)}], "metrics": [_metric("动态范围提升", round((after_span-before_span)/max(before_span,1)*100,1), "%", "1–99 百分位灰度跨度"), _metric("噪声代理下降", round((before_noise-after_noise)/max(before_noise,.01)*100,1), "%", "拉普拉斯残差代理值"), _metric("输出分辨率", f"{enhanced.shape[1]}×{enhanced.shape[0]}", "px")], "runtime_ms": round((time.perf_counter()-started)*1000,1), "notice": "CPU 实时增强基线；不能恢复采集时已经丢失的组织信息。"}


def _get_relaynet() -> torch.nn.Module:
    global _relaynet_model
    if _relaynet_model is None:
        from relaynet_pytorch import relay_net
        from relaynet_pytorch.net_api import sub_module
        if not RELAYNET_CHECKPOINT.is_file():
            raise RuntimeError("ReLayNet checkpoint is not installed")
        # The 2018 checkpoint was pickled from a pre-package module layout.
        legacy_root = types.ModuleType("networks")
        legacy_root.__path__ = []
        legacy_api = types.ModuleType("networks.net_api")
        legacy_api.__path__ = []
        legacy_root.relay_net = relay_net
        legacy_root.net_api = legacy_api
        legacy_api.sub_module = sub_module
        sys.modules.setdefault("networks", legacy_root)
        sys.modules.setdefault("networks.relay_net", relay_net)
        sys.modules.setdefault("networks.net_api", legacy_api)
        sys.modules.setdefault("networks.net_api.sub_module", sub_module)
        _relaynet_model = torch.load(RELAYNET_CHECKPOINT, map_location="cpu", weights_only=False)
        _relaynet_model.eval()
    return _relaynet_model


def structure_segmentation(image: Image.Image, **_) -> dict:
    started = time.perf_counter()
    scan = cv2.resize(_gray(image, 1000), (768, 512), interpolation=cv2.INTER_AREA)
    normalized = scan.astype(np.float32) / 255.0
    normalized = (normalized-normalized.mean()) / max(normalized.std(), .05)
    with torch.inference_mode():
        labels = torch.argmax(_get_relaynet()(torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)), dim=1)[0].numpy().astype(np.uint8)
    base = cv2.cvtColor(scan, cv2.COLOR_GRAY2RGB)
    color_mask = SEGMENT_COLORS[np.clip(labels, 0, len(SEGMENT_COLORS)-1)]
    overlay = cv2.addWeighted(base, .58, color_mask, .42, 0)
    boundaries = np.zeros_like(labels)
    boundaries[1:] |= labels[1:] != labels[:-1]
    overlay[boundaries > 0] = np.array([0,245,255], dtype=np.uint8)
    present = [{"label": name, "pixels": int(np.count_nonzero(labels == i)), "color": SEGMENT_COLORS[i].tolist()} for i,name in enumerate(SEGMENT_NAMES) if np.any(labels == i)]
    retinal = (labels >= 1) & (labels <= 7)
    valid = retinal.sum(axis=0); valid = valid[valid > 0]
    thickness = float(np.median(valid)) if valid.size else 0.0
    return {"summary": f"ReLayNet 检出 {len(present)} 个结构区域", "result_image": _png_data_url(overlay), "auxiliary_images": [{"label":"分割标签", "image":_png_data_url(color_mask)}], "metrics": [_metric("检出结构区域",len(present),"类"), _metric("视网膜厚度代理",round(thickness,1),"px","逐列分割厚度中位数，非物理标定值"), _metric("有效视网膜占比",round(float(retinal.mean()*100),1),"%")], "regions":present, "runtime_ms":round((time.perf_counter()-started)*1000,1), "notice":"原始 ReLayNet Duke SD-OCT 权重；跨设备结果需要重新验证与标定。"}


def lesion_recognition(image: Image.Image, model=None, transform=None, class_names=None, **_) -> dict:
    if model is None or transform is None or class_names is None:
        raise RuntimeError("Fundus classifier is unavailable")
    started = time.perf_counter(); rgb_image = _limit_image(image,1000); activation = {}
    hook = model.layer4.register_forward_hook(lambda _m,_i,output: activation.update(layer4=output.detach()))
    try:
        with torch.inference_mode():
            model_device = next(model.parameters()).device
            probabilities = torch.softmax(model(transform(rgb_image).unsqueeze(0).to(model_device)),dim=1)[0].cpu(); index = int(torch.argmax(probabilities))
    finally:
        hook.remove()
    class_name = class_names[index]; features = activation["layer4"][0]; weights = model.fc.weight[index].detach().cpu()
    cam = torch.einsum("c,chw->hw",weights,features.cpu()).numpy(); cam = np.maximum(cam,0); cam = (cam-cam.min())/max(float(cam.max()-cam.min()),1e-7)
    rgb = np.asarray(rgb_image); cam = cv2.resize(cam,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb,.64,heatmap,.36,0); all_probs = {class_names[i]:float(probabilities[i]) for i in range(len(class_names))}; ranked = sorted(all_probs.items(),key=lambda item:item[1],reverse=True)
    return {"summary":CLASS_LABELS.get(class_name,class_name),"prediction":class_name,"confidence":float(probabilities[index]),"probabilities":all_probs,"result_image":_png_data_url(overlay),"auxiliary_images":[{"label":"原始彩照","image":_png_data_url(rgb)}],"metrics":[_metric("最高类别概率",round(float(probabilities[index])*100,1),"%"),_metric("次高类别",CLASS_LABELS.get(ranked[1][0],ranked[1][0])),_metric("类别数",len(class_names),"类")],"runtime_ms":round((time.perf_counter()-started)*1000,1),"notice":"CAM 仅表示分类模型关注区域，不等同于病灶像素级标注。"}


def _vesselness(gray: np.ndarray) -> np.ndarray:
    image = gray.astype(np.float32)/255.0; responses=[]
    for sigma in (.8,1.4,2.2,3.2):
        dxx=ndimage.gaussian_filter(image,sigma=sigma,order=(0,2))*sigma*sigma; dxy=ndimage.gaussian_filter(image,sigma=sigma,order=(1,1))*sigma*sigma; dyy=ndimage.gaussian_filter(image,sigma=sigma,order=(2,0))*sigma*sigma
        disc=np.sqrt(np.maximum((dxx-dyy)**2+4*dxy*dxy,0)); first=.5*(dxx+dyy-disc); second=.5*(dxx+dyy+disc); swap=np.abs(first)>np.abs(second); l1=np.where(swap,second,first); l2=np.where(swap,first,second)
        rb=np.abs(l1)/(np.abs(l2)+1e-6); strength=np.sqrt(l1*l1+l2*l2); c=max(float(np.percentile(strength,95)),1e-4); response=np.exp(-(rb*rb)/(2*.55**2))*(1-np.exp(-(strength*strength)/(2*c*c))); response[l2>0]=0; responses.append(response)
    return cv2.normalize(np.max(np.stack(responses),axis=0),None,0,1,cv2.NORM_MINMAX)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    current=(mask>0).astype(np.uint8)*255; skeleton=np.zeros_like(current); element=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    for _ in range(max(current.shape)):
        opened=cv2.morphologyEx(current,cv2.MORPH_OPEN,element); skeleton=cv2.bitwise_or(skeleton,cv2.subtract(current,opened)); current=cv2.erode(current,element)
        if cv2.countNonZero(current)==0: break
    return skeleton>0


def _fractal(mask: np.ndarray) -> float:
    binary=mask.astype(bool); sizes=[]; counts=[]; size=2
    while size<=min(binary.shape)//2:
        reduced=np.add.reduceat(np.add.reduceat(binary,np.arange(0,binary.shape[0],size),axis=0),np.arange(0,binary.shape[1],size),axis=1); count=int(np.count_nonzero(reduced))
        if count: sizes.append(size); counts.append(count)
        size*=2
    return float(-np.polyfit(np.log(sizes),np.log(counts),1)[0]) if len(counts)>=2 else 0.0


def vascular_quantification(image: Image.Image, **_) -> dict:
    started=time.perf_counter(); gray=cv2.resize(_gray(image,768),(512,512),interpolation=cv2.INTER_AREA); enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray); response=_vesselness(enhanced); nonzero=response[response>0]; threshold=float(np.percentile(nonzero,70)) if nonzero.size else .3
    mask=(response>=max(.12,threshold)).astype(np.uint8); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8)); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)); count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8); clean=np.zeros_like(mask)
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>=18: clean[labels==label]=1
    skeleton=_skeleton(clean); neighbours=ndimage.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode="constant")-skeleton; branch_mask=skeleton&(neighbours>=3); branch_groups,_=ndimage.label(branch_mask); vessel_pixels=int(clean.sum()); skeleton_pixels=int(skeleton.sum())
    base=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB); overlay=base.copy(); overlay[clean>0]=(0,236,255); overlay=cv2.addWeighted(base,.5,overlay,.5,0); overlay[cv2.dilate(branch_mask.astype(np.uint8),np.ones((5,5),np.uint8))>0]=(255,71,180)
    return {"summary":"微血管网络分割与形态计量完成","result_image":_png_data_url(overlay),"auxiliary_images":[{"label":"血管响应","image":_png_data_url(np.uint8(np.clip(response,0,1)*255))}],"metrics":[_metric("血管密度",round(vessel_pixels/clean.size*100,2),"%"),_metric("骨架总长度",skeleton_pixels,"px"),_metric("分支点",int(branch_groups.max()),"个"),_metric("端点",int(np.count_nonzero(skeleton&(neighbours==1))),"个"),_metric("平均管径代理",round(vessel_pixels/max(skeleton_pixels,1),2),"px"),_metric("分形维数",round(_fractal(skeleton),3))],"runtime_ms":round((time.perf_counter()-started)*1000,1),"notice":"像素单位形态学基线；物理单位定量需设备像素间距、分层范围和人工质控。"}


PIPELINES = {"quality-enhancement":quality_enhancement,"structure-segmentation":structure_segmentation,"lesion-recognition":lesion_recognition,"vascular-quantification":vascular_quantification}
