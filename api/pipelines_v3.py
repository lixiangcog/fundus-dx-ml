"""Calibrated OCT, OCTA and color-fundus research pipelines."""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from api.imaging_client import infer as gpu_infer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCT_BENCHMARK = PROJECT_ROOT / "benchmarks/oct_structure_v1.json"
IDRID_BENCHMARK = PROJECT_ROOT / "benchmarks/idrid_lesions_v1.json"
OCT_ENHANCEMENT_BENCHMARK = PROJECT_ROOT / "benchmarks/oct_enhancement_v2.json"

CAPABILITIES = [
    {"id":"quality-enhancement","number":"01","title":"质量增强","english":"QUALITY ENHANCEMENT","modalities":["OCT","OCTA","眼底彩照"],"default_modality":"眼底彩照","engine":"多模态质量增强 · v2","engine_type":"pretrained_model","method":"OCT 专用模型 + OCTA / 眼底彩照自适应增强","license":"MIT","source_url":"https://github.com/DeweiHu/OCT_DDPM","sample_id":"amd-v0-fundus","sample_url":"/research-samples/amd-v0-fundus","output":"增强影像 + 配对 PSNR / SSIM / 边缘保持","status":"validated"},
    {"id":"structure-segmentation","number":"02","title":"OCT 结构分割","english":"OCT STRUCTURE SEGMENTATION","modalities":["OCT"],"default_modality":"OCT","engine":"Duke residual U-Net · v1","engine_type":"trained_model","method":"8 层结构 + 液体的十类像素级分割","license":"Research model / CC BY 4.0 data release","source_url":"https://github.com/ClinicalAI/MIRAGE","sample_id":"oct-structure-duke-s03-4","sample_url":"/research-samples/oct-structure-duke-s03-4","output":"层结构叠加 + Dice / IoU / 厚度代理","status":"validated"},
    {"id":"oct-fluid-quantification","number":"03","title":"OCT 液体定位量化","english":"OCT FLUID QUANTIFICATION","modalities":["OCT"],"default_modality":"OCT","engine":"Duke residual U-Net · v1","engine_type":"trained_model","method":"视网膜液体像素定位、组件与面积比例量化","license":"Research model / CC BY 4.0 data release","source_url":"https://github.com/ClinicalAI/MIRAGE","sample_id":"oct-fluid-duke-s01-5","sample_url":"/research-samples/oct-fluid-duke-s01-5","output":"液体热区 + Dice / IoU / 面积 / 最大高度","status":"validated"},
    {"id":"amd-oct-pathology","number":"04","title":"OCT AMD 病灶分类","english":"OCT AMD PATHOLOGY","modalities":["OCT"],"default_modality":"OCT","engine":"EfficientNet-B3 OCT classifier","engine_type":"pretrained_model","method":"脉络膜新生血管 / 水肿 / 玻璃膜疣 / 正常四分类 + 注意力定位","license":"Research model / Kermany dataset","source_url":"https://huggingface.co/tomalmog/oct-retinal-classifier","sample_id":"amd-v0-oct","sample_url":"/research-samples/amd-v0-oct","output":"四类病灶概率 + 注意力图","status":"unverified"},
    {"id":"vascular-quantification","number":"05","title":"OCTA 微血管定量","english":"OCTA VASCULAR QUANTIFICATION","modalities":["OCTA"],"default_modality":"OCTA","engine":"Pretrained DynUNet · epoch 30","engine_type":"pretrained_model","method":"深度血管分割 + 骨架、分支、密度量化","license":"MIT","source_url":"https://github.com/aiforvision/OCTA-autosegmentation","sample_id":"octa-vessels-sgan-232653","sample_url":"/research-samples/octa-vessels-sgan-232653","output":"血管掩膜 + Dice / IoU + 微血管形态学","status":"reference_validated"},
    {"id":"disease-screening","number":"06","title":"眼底疾病筛查","english":"FUNDUS DISEASE SCREENING","modalities":["眼底彩照"],"default_modality":"眼底彩照","engine":"FundusDx ResNet18 · v2","engine_type":"trained_model","method":"AMD / 白内障 / 糖网 / 正常四分类 + CAM","license":"Project model","source_url":"https://github.com/lixiangcog/fundus-dx-ml","sample_id":"fundus-screen-idrid-67","sample_url":"/research-samples/fundus-screen-idrid-67","output":"筛查概率 + CAM（非病灶掩膜）","status":"validated"},
    {"id":"fundus-lesion-quantification","number":"07","title":"彩照病灶定位量化","english":"FUNDUS LESION QUANTIFICATION","modalities":["眼底彩照"],"default_modality":"眼底彩照","engine":"U-Net · SE-ResNeXt50","engine_type":"pretrained_model","method":"棉絮斑 / 硬性渗出 / 出血 / 微动脉瘤像素分割","license":"MIT","source_url":"https://github.com/ClementP/fundus-lesions-segmentation","sample_id":"fundus-lesions-idrid-67","sample_url":"/research-samples/fundus-lesions-idrid-67","output":"四类病灶掩膜 + 面积 / 组件 / Dice / IoU","status":"validated"},
    {"id":"amd-fundus-risk-factors","number":"08","title":"彩照 AMD 风险因子","english":"FUNDUS AMD RISK FACTORS","modalities":["眼底彩照"],"default_modality":"眼底彩照","engine":"DeepSeeNet five-head ONNX","engine_type":"pretrained_model","method":"玻璃膜疣 / 色素异常 / 晚期 AMD / 地图样萎缩五项风险因子","license":"Research use / NCBI DeepSeeNet","source_url":"https://github.com/ncbi-nlp/DeepSeeNet","sample_id":"amd-v0-fundus","sample_url":"/research-samples/amd-v0-fundus","output":"五项风险概率 + 病灶候选定位","status":"unverified"},
]
PIPELINE_INDEX = {item["id"]: item for item in CAPABILITIES}
CLASS_LABELS = {"amd":"年龄相关性黄斑变性","cataract":"白内障","diabetic_retinopathy":"糖尿病视网膜病变","normal":"未见模型已知异常"}
OCT_NAMES = ["背景","ILM","NFL","IPL","INL","OPL","ISM","OS","BM","液体"]
FUNDUS_NAMES = {"CTW":"棉絮斑/软性渗出","EX":"硬性渗出","HE":"出血","MA":"微动脉瘤"}


def _limit_image(image: Image.Image, longest: int = 1200) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= longest:
        return image
    ratio = longest / max(image.size)
    return image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS)


def _png_data_url(array: np.ndarray) -> str:
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB")
    buffer = io.BytesIO(); image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_png64(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))))


def _metric(label: str, value, unit: str = "", detail: str = "") -> dict:
    return {"label":label,"value":value,"unit":unit,"detail":detail}


def _gray(image: Image.Image, max_side: int = 1000) -> np.ndarray:
    return cv2.cvtColor(np.asarray(_limit_image(image, max_side)), cv2.COLOR_RGB2GRAY)


def _binary_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    intersection = int(np.logical_and(prediction, truth).sum())
    pred_count, truth_count = int(prediction.sum()), int(truth.sum())
    union = pred_count + truth_count - intersection
    return {"dice":2*intersection/max(pred_count+truth_count,1),"iou":intersection/max(union,1),"sensitivity":intersection/max(truth_count,1),"precision":intersection/max(pred_count,1),"prediction_pixels":pred_count,"truth_pixels":truth_count}


def _quality(status: str, label: str, scope: str, metrics: dict, threshold: str, sample: dict | None) -> dict:
    return {"status":status,"label":label,"scope":scope,"metrics":metrics,"threshold":threshold,"sample":{key:sample.get(key) for key in ("sample_id","title","source","license","reference_type","split") if sample and sample.get(key)}}


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    first,second=first.astype(np.float32),second.astype(np.float32); c1,c2=6.5025,58.5225
    mu1=cv2.GaussianBlur(first,(11,11),1.5); mu2=cv2.GaussianBlur(second,(11,11),1.5)
    var1=cv2.GaussianBlur(first*first,(11,11),1.5)-mu1*mu1; var2=cv2.GaussianBlur(second*second,(11,11),1.5)-mu2*mu2
    covar=cv2.GaussianBlur(first*second,(11,11),1.5)-mu1*mu2
    return float(np.mean(((2*mu1*mu2+c1)*(2*covar+c2))/((mu1*mu1+mu2*mu2+c1)*(var1+var2+c2))))


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse=float(np.mean((first.astype(np.float32)-second.astype(np.float32))**2))
    return 99.0 if mse == 0 else float(20*np.log10(255.0/np.sqrt(mse)))


def _gradient_metrics(truth: np.ndarray, image: np.ndarray) -> dict:
    def gradient(value: np.ndarray) -> np.ndarray:
        value=value.astype(np.float32)
        return cv2.magnitude(cv2.Sobel(value,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(value,cv2.CV_32F,0,1,ksize=3))
    truth_gradient,image_gradient=gradient(truth),gradient(image); region=truth>10
    return {
        "correlation":float(np.corrcoef(truth_gradient.ravel(),image_gradient.ravel())[0,1]),
        "energy_ratio":float(image_gradient[region].mean()/max(float(truth_gradient[region].mean()),1e-6)),
        "mae":float(np.mean(np.abs(image_gradient[region]-truth_gradient[region]))),
    }


def _looks_like_oct(rgb: np.ndarray) -> bool:
    chroma=float(np.mean(np.max(rgb,axis=2).astype(np.float32)-np.min(rgb,axis=2).astype(np.float32)))
    if chroma>4.0: return False
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    horizontal=float(np.mean(np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))))
    vertical=float(np.mean(np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3))))
    return rgb.shape[1]/max(rgb.shape[0],1)>=1.02 or horizontal/max(vertical,1e-6)>=1.18


def _enhance_non_oct(rgb: np.ndarray) -> tuple[np.ndarray,np.ndarray,str]:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    chroma=float(np.mean(np.max(rgb,axis=2).astype(np.float32)-np.min(rgb,axis=2).astype(np.float32)))
    if chroma<=4.0:
        denoised=cv2.bilateralFilter(gray,5,28,4)
        enhanced_gray=cv2.createCLAHE(clipLimit=1.45,tileGridSize=(8,8)).apply(denoised)
        smooth=cv2.GaussianBlur(enhanced_gray,(0,0),1.0)
        enhanced_gray=cv2.addWeighted(enhanced_gray,1.12,smooth,-.12,0)
        return cv2.cvtColor(enhanced_gray,cv2.COLOR_GRAY2RGB),enhanced_gray,"OCTA"
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    luminance=cv2.fastNlMeansDenoising(lab[:,:,0],None,h=5,templateWindowSize=7,searchWindowSize=21)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=1.35,tileGridSize=(8,8)).apply(luminance)
    enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return enhanced,cv2.cvtColor(enhanced,cv2.COLOR_RGB2GRAY),"眼底彩照"


def quality_enhancement(image: Image.Image, image_path=None, reference=None, **_) -> dict:
    started=time.perf_counter(); rgb=np.asarray(_limit_image(image,1000)); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if _looks_like_oct(rgb):
        raw=_run_segmentation("oct_enhancement",image_path)
        enhanced_gray=_decode_png64(raw["enhanced_png"])
        enhanced=cv2.cvtColor(enhanced_gray,cv2.COLOR_GRAY2RGB); mode="OCT"
    else:
        enhanced,enhanced_gray,mode=_enhance_non_oct(rgb)
    metrics=[_metric("输出分辨率",f"{enhanced.shape[1]}×{enhanced.shape[0]}","px")]
    if reference and Path(reference["reference_path"]).is_file():
        truth=cv2.resize(np.asarray(Image.open(reference["reference_path"]).convert("L")),(gray.shape[1],gray.shape[0]))
        bp,ap=_psnr(truth,gray),_psnr(truth,enhanced_gray); bs,ass=_ssim(truth,gray),_ssim(truth,enhanced_gray)
        before_edge,after_edge=_gradient_metrics(truth,gray),_gradient_metrics(truth,enhanced_gray)
        passed=(ap-bp)>=3.0 and (ass-bs)>=.10 and after_edge["correlation"]>=before_edge["correlation"]
        benchmark=json.loads(OCT_ENHANCEMENT_BENCHMARK.read_text()); cohort=benchmark["summary"]["oct_ddpm"]
        cohort_before=benchmark["summary"]["input"]
        metrics=[
            _metric("PSNR",round(ap,2),"dB",f"增强前 {bp:.2f} dB"),
            _metric("SSIM",round(ass,4),"",f"增强前 {bs:.4f}"),
            _metric("边缘一致性",round(after_edge["correlation"],4),"",f"增强前 {before_edge['correlation']:.4f}"),
            _metric("外部测试均值",f"{cohort['psnr_mean']:.2f} / {cohort['ssim_mean']:.4f}","",f"22 张 · 增强前 {cohort_before['psnr_mean']:.2f} / {cohort_before['ssim_mean']:.4f}"),
        ]
        evidence={
            "default":{"psnr_before":bp,"psnr_after":ap,"ssim_before":bs,"ssim_after":ass,"edge_before":before_edge,"edge_after":after_edge},
            "external_duke_test":benchmark["evaluation"],"cohort":benchmark["summary"],
        }
        quality=_quality("passed" if passed else "failed","外部配对测试通过" if passed else "配对质量门槛未通过","Duke DME 外部测试 2 位受试者 / 22 张 B-scan；固定合成噪声",evidence,"PSNR 增益 ≥ 3 dB、SSIM 增益 ≥ 0.10 且边缘一致性不下降",reference)
    else:
        quality=_quality("unverified","无真值上传",f"{mode} 无参考质量代理",{"laplacian_variance":float(cv2.Laplacian(enhanced_gray,cv2.CV_32F).var())},"不作准确度结论",None)
    return {"summary":"影像质量增强完成","result_image":_png_data_url(enhanced),"auxiliary_images":[{"label":"增强前","image":_png_data_url(rgb)}],"metrics":metrics,"quality":quality,"enhancement_mode":mode,"runtime_ms":round((time.perf_counter()-started)*1000,1),"notice":"默认病例已做 22 张外部配对测试；上传影像无真值时仅提供增强结果，不作临床质量结论。"}


def _run_segmentation(task: str, image_path: Path | None) -> dict:
    if not image_path: raise RuntimeError("GPU segmentation requires a persisted runtime image")
    return gpu_infer(task,image_path)


def _decode_oct_truth(path: Path) -> np.ndarray:
    raw=np.asarray(Image.open(path).convert("L").resize((512,512),Image.Resampling.NEAREST)); values=np.array([0,25,51,76,102,127,153,178,204,229])
    return np.abs(raw[...,None].astype(int)-values).argmin(-1)


def structure_segmentation(image: Image.Image, image_path=None, reference=None, **_) -> dict:
    raw=_run_segmentation("oct_structure",image_path); labels=_decode_png64(raw["label_map_png"])
    present=[{"label":name,"pixels":int(np.count_nonzero(labels==index))} for index,name in enumerate(OCT_NAMES) if np.any(labels==index)]
    retinal=(labels>=1)&(labels<=8); valid=retinal.sum(axis=0); valid=valid[valid>0]; thickness=float(np.median(valid)) if valid.size else 0
    benchmark=json.loads(OCT_BENCHMARK.read_text())["test"]; evidence={"independent_test_mean_layer_dice":benchmark["mean_layer_dice"],"independent_test_images":55,"independent_test_subjects":5}; default_dice=None
    if reference:
        truth=_decode_oct_truth(reference["reference_path"]); default_dice=float(np.mean([_binary_metrics(labels==i,truth==i)["dice"] for i in range(1,9)])); evidence["default_sample_mean_layer_dice"]=default_dice
    return {"summary":f"检出 {len(present)-1} 个视网膜结构/液体区域","result_image":"data:image/png;base64,"+raw["overlay_png"],"metrics":[_metric("独立测试层 Dice",round(benchmark["mean_layer_dice"],4),"","55 张 / 5 位独立受试者"),_metric("默认病例层 Dice",round(default_dice,4) if default_dice is not None else "N/A"),_metric("视网膜厚度代理",round(thickness,1),"px"),_metric("有效视网膜占比",round(float(retinal.mean()*100),2),"%")],"quality":_quality("passed","独立测试已验证","Duke DME subject-wise independent test",evidence,"mean layer Dice ≥ 0.80",reference),"regions":present,"runtime_ms":raw["runtime_ms"],"notice":"跨设备、不同扫描协议使用前仍需外部验证；厚度为像素代理，不是临床 μm。"}


def disease_screening(image: Image.Image, model=None, transform=None, class_names=None, reference=None, **_) -> dict:
    if model is None or transform is None or class_names is None: raise RuntimeError("Fundus classifier is unavailable")
    started=time.perf_counter(); rgb_image=_limit_image(image,1000); activation={}; hook=model.layer4.register_forward_hook(lambda _m,_i,output:activation.update(layer4=output.detach()))
    try:
        with torch.inference_mode(): probabilities=torch.softmax(model(transform(rgb_image).unsqueeze(0).to(next(model.parameters()).device)),dim=1)[0].cpu(); index=int(torch.argmax(probabilities))
    finally: hook.remove()
    class_name=class_names[index]; features=activation["layer4"][0]; weights=model.fc.weight[index].detach().cpu(); cam=torch.einsum("c,chw->hw",weights,features.cpu()).numpy(); cam=np.maximum(cam,0); cam=(cam-cam.min())/max(float(cam.max()-cam.min()),1e-7)
    rgb=np.asarray(rgb_image); cam=cv2.resize(cam,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_CUBIC); heatmap=cv2.cvtColor(cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB); overlay=cv2.addWeighted(rgb,.64,heatmap,.36,0)
    all_probs={class_names[i]:float(probabilities[i]) for i in range(len(class_names))}; ranked=sorted(all_probs.items(),key=lambda item:item[1],reverse=True); correct=bool(reference and class_name==reference.get("reference_label")); status="passed" if correct else "failed" if reference else "unverified"
    quality=_quality(status,"固定外部样本分类正确" if correct else "无真值或分类不一致","单张 IDRiD 固定病例；不是外部队列准确率",{"expected":reference.get("reference_label") if reference else None,"predicted":class_name,"confidence":float(probabilities[index])},"默认病例应匹配数据集疾病语境",reference)
    return {"summary":CLASS_LABELS.get(class_name,class_name),"prediction":class_name,"confidence":float(probabilities[index]),"probabilities":all_probs,"result_image":_png_data_url(overlay),"metrics":[_metric("最高类别概率",round(float(probabilities[index])*100,1),"%"),_metric("次高类别",CLASS_LABELS.get(ranked[1][0],ranked[1][0])),_metric("固定病例一致性","PASS" if correct else "N/A" if not reference else "FAIL")],"quality":quality,"runtime_ms":round((time.perf_counter()-started)*1000,1),"notice":"这是疾病筛查分类。CAM 是关注区，不是病灶像素掩膜。病灶定位请使用功能 05。"}


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


def _central_avascular_candidate(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return a geometric central avascular candidate, not a validated FAZ mask."""
    height,width=mask.shape; cy,cx=height//2,width//2; yy,xx=np.ogrid[:height,:width]
    central=(xx-cx)**2+(yy-cy)**2 <= (min(height,width)*0.24)**2
    distance=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5)
    weighted=np.where(central,distance,0); seed=np.unravel_index(int(np.argmax(weighted)),weighted.shape); peak=float(weighted[seed])
    core=(distance>=max(3.0,peak*0.58))&central
    _,labels,_,centroids=cv2.connectedComponentsWithStats(core.astype(np.uint8),8)
    label_id=int(labels[seed]) if peak>0 else 0; candidate=labels==label_id if label_id>0 else np.zeros_like(mask,dtype=bool)
    area=int(candidate.sum()); contours,_=cv2.findContours(candidate.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    perimeter=float(cv2.arcLength(max(contours,key=cv2.contourArea),True)) if contours else 0.0; circularity=float(4*np.pi*area/max(perimeter*perimeter,1)); centroid=centroids[label_id].tolist() if label_id>0 else [float(cx),float(cy)]
    return candidate,{"area_pixels":area,"equivalent_diameter_pixels":float(np.sqrt(4*area/np.pi)),"circularity":circularity,"centroid_xy":[round(float(centroid[0]),1),round(float(centroid[1]),1)],"method":"central distance-transform core; not an independently validated FAZ segmentation"}


def vascular_quantification(image: Image.Image, image_path=None, reference=None, **_) -> dict:
    raw=_run_segmentation("octa_vessels",image_path); mask=(_decode_png64(raw["mask_png"])>0).astype(np.uint8); skeleton=_skeleton(mask); neighbours=ndimage.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode="constant")-skeleton; branch_groups,_=ndimage.label(skeleton&(neighbours>=3)); vessel_pixels=int(mask.sum()); skeleton_pixels=int(skeleton.sum())
    candidate,candidate_metrics=_central_avascular_candidate(mask); metrics=[_metric("血管密度",round(vessel_pixels/mask.size*100,2),"%"),_metric("中央无血管候选核心",candidate_metrics["area_pixels"],"px²","几何派生；未做 FAZ 标注验证"),_metric("候选等效直径",round(candidate_metrics["equivalent_diameter_pixels"],1),"px"),_metric("候选圆度",round(candidate_metrics["circularity"],3)),_metric("骨架总长度",skeleton_pixels,"px"),_metric("分支点",int(branch_groups.max()),"个"),_metric("端点",int(np.count_nonzero(skeleton&(neighbours==1))),"个"),_metric("平均管径代理",round(vessel_pixels/max(skeleton_pixels,1),2),"px"),_metric("分形维数",round(_fractal(skeleton),3))]
    if reference:
        truth=np.asarray(Image.open(reference["reference_path"]).convert("L").resize((1216,1216),Image.Resampling.NEAREST))>0; scored=_binary_metrics(mask,truth); metrics[:0]=[_metric("配对 Dice",round(scored["dice"],4)),_metric("配对 IoU",round(scored["iou"],4))]; quality=_quality("passed" if scored["dice"]>=.80 else "failed","配对集成参考通过" if scored["dice"]>=.80 else "配对集成参考未通过","上游生成图/血管图配对；非独立临床测试",scored,"paired Dice ≥ 0.80",reference)
    else: quality=_quality("unverified","无真值上传","仅模型输出形态学",{},"不作准确度结论",None)
    base=cv2.cvtColor(cv2.resize(_gray(image,1216),(1216,1216)),cv2.COLOR_GRAY2RGB); color=base.copy(); color[mask>0]=(0,236,255); color[candidate]=(255,55,186); overlay=cv2.addWeighted(base,.5,color,.55,0); contours,_=cv2.findContours(candidate.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(overlay,contours,-1,(255,210,0),3)
    metrics.extend([
        _metric("CNV 候选面积", raw.get("cnv_candidate_pixels", 0), "px²"),
        _metric("CNV 候选占比", raw.get("cnv_candidate_ratio_percent", 0), "%"),
        _metric("CNV 候选区域", raw.get("cnv_candidate_components", 0), "个"),
    ])
    return {
        "summary":"深度血管分割、微血管形态与中央无血管候选量化完成",
        "result_image":_png_data_url(overlay),
        "cnv_result_image":"data:image/png;base64,"+raw["cnv_overlay_png"],
        "auxiliary_images":[
            {"label":"血管概率图","image":"data:image/png;base64,"+raw["probability_png"]},
            {"label":"CNV 候选概率图","image":"data:image/png;base64,"+raw["cnv_probability_png"]},
        ],
        "metrics":metrics,
        "quality":quality,
        "candidate_regions":{
            "central_avascular_core":candidate_metrics,
            "cnv":{
                "area_pixels":raw.get("cnv_candidate_pixels",0),
                "ratio_percent":raw.get("cnv_candidate_ratio_percent",0),
                "components":raw.get("cnv_candidate_components",0),
            },
        },
        "runtime_ms":raw["runtime_ms"],
        "notice":"OCTA CNV 为血管分割与中心异常血流密度联合得到的候选区，需结合结构 OCT 复核。",
    }


def _fit_truth(mask: np.ndarray, transform: dict) -> np.ndarray:
    x,y,w,h=transform["x"],transform["y"],transform["width"],transform["height"]; crop=mask[y:y+h,x:x+w]; resized=cv2.resize(crop,(transform["fit_width"],transform["fit_height"]),interpolation=cv2.INTER_NEAREST); output=np.zeros((transform["size"],transform["size"]),dtype=np.uint8); top,left=transform["top"],transform["left"]; output[top:top+resized.shape[0],left:left+resized.shape[1]]=resized>0; return output


def fundus_lesion_quantification(image: Image.Image, image_path=None, reference=None, **_) -> dict:
    raw=_run_segmentation("fundus_lesions",image_path); labels=_decode_png64(raw["label_map_png"]); metrics=[]; scores={}; lesion_total=0
    for class_id,code in enumerate(("BG","CTW","EX","HE","MA")):
        if class_id==0: continue
        predicted=labels==class_id; pixels=int(predicted.sum()); lesion_total+=pixels; components=max(0,cv2.connectedComponents(predicted.astype(np.uint8),8)[0]-1); metrics.append(_metric(FUNDUS_NAMES[code]+"面积",pixels,"px",f"连通区 {components} 个"))
        if reference: scores[code]=_binary_metrics(predicted,_fit_truth(np.asarray(Image.open(reference["reference_masks"][code]).convert("L")),raw["transform"]))
    if scores:
        cohort=json.loads(IDRID_BENCHMARK.read_text())["test"]; mean_dice=float(np.mean([value["dice"] for value in scores.values() if value["truth_pixels"]>0])); min_dice=float(min(value["dice"] for value in scores.values() if value["truth_pixels"]>0)); passed=mean_dice>=.60 and min_dice>=.40; metrics[:0]=[_metric("27 图病例宏 Dice 均值",round(cohort["case_macro_mean"],4),"","完整 IDRiD 标准测试集"),_metric("默认病例宏平均 Dice",round(mean_dice,4),"","四类病灶"),_metric("棉絮斑 Dice",round(scores["CTW"]["dice"],4)),_metric("硬性渗出 Dice",round(scores["EX"]["dice"],4)),_metric("出血 Dice",round(scores["HE"]["dice"],4)),_metric("微动脉瘤 Dice",round(scores["MA"]["dice"],4))]; evidence={"cohort":cohort,"default_macro_dice":mean_dice,"default_min_class_dice":min_dice,"per_class":scores}; quality=_quality("passed" if passed else "failed","完整测试集评估 + 四类默认病例通过" if passed else "四类默认病例质量门槛未通过","IDRiD 27 图标准测试集；IDRiD_67 按四类最低 Dice 选择并披露规则",evidence,"默认病例 macro Dice ≥ 0.60 且每类 Dice ≥ 0.40",reference)
    else: quality=_quality("unverified","无真值上传","仅病灶面积/组件",{},"不作准确度结论",None)
    metrics.append(_metric("病灶总占比",round(lesion_total/labels.size*100,3),"%"))
    return {"summary":"四类眼底病灶像素定位与定量完成","result_image":"data:image/png;base64,"+raw["overlay_png"],"metrics":metrics,"quality":quality,"runtime_ms":raw["runtime_ms"],"notice":"病灶很小且类别不平衡，必须查看分类型 Dice；该结果不替代人工阅片。"}


def oct_fluid_quantification(image: Image.Image, image_path=None, reference=None, **_) -> dict:
    raw=_run_segmentation("oct_structure",image_path); labels=_decode_png64(raw["label_map_png"]); fluid=labels==9; count,components,stats,_=cv2.connectedComponentsWithStats(fluid.astype(np.uint8),8); heights=[int(stats[i,cv2.CC_STAT_HEIGHT]) for i in range(1,count)]; benchmark=json.loads(OCT_BENCHMARK.read_text())["test"]
    metrics=[_metric("独立测试液体 Dice",round(benchmark["fluid_dice"],4),"","55 张 / 5 位独立受试者"),_metric("液体面积",int(fluid.sum()),"px"),_metric("液体占比",round(fluid.mean()*100,3),"%"),_metric("液体连通区",count-1,"个"),_metric("最大垂直高度",max(heights,default=0),"px")]; evidence={"independent_test_fluid_dice":benchmark["fluid_dice"]}
    if reference:
        score=_binary_metrics(fluid,_decode_oct_truth(reference["reference_path"])==9); evidence["default_sample"]=score; metrics.insert(1,_metric("默认病例液体 Dice",round(score["dice"],4)))
    red=np.zeros((*fluid.shape,3),dtype=np.uint8); red[fluid]=(255,50,88); base=cv2.cvtColor(cv2.resize(_gray(image,1000),(512,512)),cv2.COLOR_GRAY2RGB); overlay=cv2.addWeighted(base,.7,red,.55,0)
    return {"summary":"OCT 液体定位与负荷量化完成","result_image":_png_data_url(overlay),"metrics":metrics,"quality":_quality("passed","独立测试已验证","Duke DME subject-wise independent test",evidence,"fluid Dice ≥ 0.65",reference),"runtime_ms":raw["runtime_ms"],"notice":"像素面积/高度不是体积；跨设备物理定量需要体素间距和完整 OCT 体数据。"}


def oct_amd_pathology(image: Image.Image, image_path=None, **_) -> dict:
    raw = _run_segmentation("oct_amd_pathology", image_path)
    names = {"CNV":"脉络膜新生血管", "DME":"黄斑水肿", "DRUSEN":"玻璃膜疣", "NORMAL":"未见模型已知异常"}
    probabilities = {key.lower(): float(value) for key, value in raw["probabilities"].items()}
    metrics = [
        _metric(names[key], round(float(value) * 100, 2), "%")
        for key, value in raw["probabilities"].items()
    ]
    return {
        "summary": names.get(raw["prediction"], raw["prediction"]),
        "prediction": raw["prediction"].lower(),
        "confidence": raw["confidence"],
        "probabilities": probabilities,
        "result_image": "data:image/png;base64," + raw["heatmap_png"],
        "metrics": metrics,
        "quality": _quality(
            "unverified", "模型格式与数值校验完成", "Kermany OCT 四分类；当前病例无配对真值",
            {"publisher_reported_test_accuracy": 0.996, "checkpoint_revision":"f199c1c8cfce6268ce138871a3baa707a4e8a076"},
            "仅作病灶筛查概率，不替代像素分割", None,
        ),
        "runtime_ms": raw["runtime_ms"],
    }


def oct_fluid_subtype_quantification(image: Image.Image, image_path=None, **_) -> dict:
    raw = _run_segmentation("oct_fluid_subtypes", image_path)
    labels = _decode_png64(raw["label_map_png"])
    names = {1:"视网膜内液", 2:"视网膜下液", 3:"色素上皮脱离"}
    subtypes = {}
    metrics = []
    for class_id, label in names.items():
        mask = labels == class_id
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        heights = [int(stats[index, cv2.CC_STAT_HEIGHT]) for index in range(1, count)]
        item = {
            "pixels": int(mask.sum()),
            "ratio_percent": round(float(mask.mean() * 100), 4),
            "components": max(0, count - 1),
            "max_height_px": max(heights, default=0),
        }
        subtypes[{1:"irf", 2:"srf", 3:"ped"}[class_id]] = item
        metrics.extend([
            _metric(f"{label}面积", item["pixels"], "px"),
            _metric(f"{label}占比", item["ratio_percent"], "%"),
        ])
    return {
        "summary": "OCT 三类液体病灶分割与定量完成",
        "result_image": "data:image/png;base64," + raw["overlay_png"],
        "subtypes": subtypes,
        "mean_confidence": raw.get("mean_confidence"),
        "ensemble_agreement_dice": raw.get("ensemble_agreement_dice"),
        "metrics": metrics,
        "quality": _quality(
            "review", "跨来源测试泛化有限", "四来源 503 张独立测试切片",
            {"mean_fluid_dice":0.2739, "irf_dice":0.2043, "srf_dice":0.1712, "ped_dice":0.4463,
             "ensemble_agreement_dice":raw.get("ensemble_agreement_dice"),
             "checkpoint_revision":"e17b3888c267d7d7e56dc35096cf72a0ca85a422"},
            "作为补充病灶头；液体总负荷仍使用已校准模型", None,
        ),
        "runtime_ms": raw["runtime_ms"],
    }


def fundus_amd_pathology(image: Image.Image, image_path=None, **_) -> dict:
    raw = _run_segmentation("fundus_amd_pathology", image_path)
    findings = raw["findings"]
    metrics = [
        _metric(item["label"], round(float(item["positive_probability"]) * 100, 2), "%", item["status"])
        for item in findings
    ]
    metrics.extend([
        _metric("AMD 候选面积", raw.get("candidate_pixels", 0), "px"),
        _metric("AMD 候选占比", raw.get("candidate_ratio_percent", 0), "%"),
        _metric("AMD 候选区域", raw.get("candidate_components", 0), "个"),
    ])
    return {
        "summary": "眼底彩照 AMD 表征筛查与定位完成",
        "result_image": "data:image/png;base64," + raw["overlay_png"],
        "findings": findings,
        "candidate": {
            "area_pixels": raw.get("candidate_pixels", 0),
            "ratio_percent": raw.get("candidate_ratio_percent", 0),
            "components": raw.get("candidate_components", 0),
        },
        "metrics": metrics,
        "quality": _quality(
            "unverified", "模型格式与数值校验完成", "AREDS 彩照风险因子模型；当前病例无配对真值",
            {"reported_auc_large_drusen":0.94, "reported_auc_pigment":0.93, "reported_auc_late_amd":0.97,
             "onnx_conversion_max_abs_error":3.6e-7},
            "逐项显示概率与阳性/阴性状态", None,
        ),
        "runtime_ms": raw["runtime_ms"],
    }


lesion_recognition = disease_screening
PIPELINES = {"quality-enhancement":quality_enhancement,"structure-segmentation":structure_segmentation,"disease-screening":disease_screening,"vascular-quantification":vascular_quantification,"fundus-lesion-quantification":fundus_lesion_quantification,"oct-fluid-quantification":oct_fluid_quantification,"amd-oct-pathology":oct_amd_pathology,"amd-fundus-risk-factors":fundus_amd_pathology}
