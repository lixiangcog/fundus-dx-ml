# Third-party model and sample provenance

This project is a research and teaching workbench. It is not a medical device.
Every integrated pipeline declares whether it is a trained model or a reference
algorithm in the API and UI.

## Duke OCT structure and fluid segmentation

- Data/source: https://github.com/ClinicalAI/MIRAGE (CC BY 4.0 release metadata)
- Integrated artifact: locally trained MONAI residual U-Net at
  `models/oct-structure/duke_unet_v1.pth`.
- Split: subjects 6-9 development, subject 10 checkpoint selection, subjects
  1-5 independent test. Results are recorded in
  `benchmarks/oct_structure_v1.json` and `CALIBRATION_REPORT_V4.md`.
- The released ReLayNet epoch-20 checkpoint was evaluated and rejected for this
  deployment because it did not meet the local Duke quality gate.

## FundusDx lesion recognition

- Repository: https://github.com/lixiangcog/fundus-dx-ml
- Artifact: project `best_model.pth`
- Task: four-class fundus photograph screening with a CAM attention overlay.
- Note: CAM is an attention visualization, not a pixel-level lesion annotation.

## OCTA microvascular quantification references

- AutoMorph: https://github.com/rmaphoh/AutoMorph (Apache-2.0)
- OCTA autosegmentation domain/sample source:
  https://github.com/aiforvision/OCTA-autosegmentation (MIT), pinned commit
  `9cdc3137b6f55ae766dcae76c166ccf9774daf2b`
- Current deployable engine: upstream epoch-30 DynUNet checkpoint on the GPU,
  followed by connected-component cleanup, skeletonization and morphometry.
- The central avascular candidate is explicitly a geometry-derived visual aid,
  not an independently validated FAZ segmentation.
- Note: measurements are in pixels. Physical units require device pixel spacing,
  layer/slab metadata and clinical quality control.

## OCT quality enhancement

- Repository: https://github.com/DeweiHu/OCT_DDPM (MIT), pinned source commit
  `8dfb2e6`.
- Artifact: public checkpoint `DDPM_oct_dataset2_2021-07-08.pt`, stored locally
  at `models/oct-enhancement/`.
- Deployment: the OCT-specific diffusion network predicts the noise component
  at fixed timestep `14`; it is not an unconstrained image generator.
- Calibration: 22 Duke DME B-scans from two external-test subjects with fixed
  paired synthetic noise. Metrics and selection rules are stored in
  `benchmarks/oct_enhancement_v2.json` and `CALIBRATION_REPORT_V4.md`.
- OpenCV (Apache-2.0) remains the conservative modality-aware path for OCTA and
  color-fundus uploads without paired truth.
- Note: no enhancement can recreate tissue information lost during acquisition.

## Fundus lesion localization and default examples

- Model: https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d
  (MIT), U-Net with SE-ResNeXt50 encoder.
- Full 27-image standard IDRiD test-set measurements and the disclosed default
  case selection rule are stored in `benchmarks/idrid_lesions_v1.json`.
- The fixed fundus example is IDRiD_67 (CC BY 4.0); the fixed OCT examples are
  independent-test Duke subjects; the OCTA example is an upstream paired
  integration reference. See `CALIBRATION_REPORT_V4.md` for evidence levels.

## VisionUnite V1 fundus specialist

- Repository: https://github.com/HUANGLIZI/VisionUnite (MIT code), pinned source snapshot `3dab080ef21d946c4dfaab26572de3828c598090`.
- The vendored inference subset removes the unused bundled ImageBind module; it is not imported or executed by the deployed service.
- A deployment patch fixes five copy/paste threshold comparisons in the upstream six-head abnormality flag post-processing; each head is now compared against its own normal logit.
- Artifact: public `checkpoint-VisionUniteV1.pth` linked by the authors in the
  repository README; the project cites VisionUnite when this artifact is used.
- Scope in this workbench: independent review of baseline and follow-up color
  fundus photographs, including the model's six abnormality signal heads and
  generated descriptions. It is not used to interpret OCT or OCTA.
- The README separately restricts access to **further MMFundus pretrained
  models** to academic research and excludes commercial use/second development.
  Those further weights are not downloaded or integrated here.
- The service accepts the public V1 checkpoint only when every model parameter
  is present in that checkpoint. It does not download a separate LLaMA weight
  file; tokenizer compatibility files are pinned to the original LLaMA-7B
  format. The integration remains research-only under the V1 usage context.
- Limitations: VisionUnite is a research model, not a medical device; its output
  is preserved as a separately attributed specialist observation and cannot
  select the final clinical action.

## Retinal age estimation

- Repository: https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation
  (MIT).
- Public weights: https://huggingface.co/mehmetaytugyuruk/retina-resnet-age-estimation
  (MIT), `resnet101-nonfiltered.pth`, SHA-256
  `f78a8712f2326ba49f62bd62081f64321a8bd99e8a6f7139325cc176ea827fd0`.
- Default example: `img00509.jpg`, age 57, from the project's published
  retinal-age test split. The deployed checkpoint predicts 57.30 years for
  this held-out example (absolute error 0.30 years).
- Published non-filtered ResNet101 test performance: MAE 5.09 years,
  age-category accuracy 0.8431 and F1 0.7132.
- Scope: retinal apparent age and age difference for research. The output is
  not biological age and the authors state that the model is not clinically
  validated.

## VascX retinal vascular phenotyping

- Repository: https://github.com/Eyened/retinalysis-vascx (Apache-2.0 code).
- Public weights: https://huggingface.co/Eyened/vascx (AGPL-3.0 weights).
  The deployment pins the May-2026 vessel, optic-disc and fovea weights plus
  the fine-tuned artery/vein weights under `models/vascx/`.
- Tasks: retinal field preprocessing, vessel and artery/vein segmentation,
  optic-disc segmentation, fovea localization and the `full_v3` set of 75
  numeric vascular features.
- Default examples: the high-resolution fundus examples distributed with the
  VascX repository. They demonstrate feature extraction and are not labelled
  cardiovascular or cerebrovascular outcome cases.
- Local acceptance gate: retinal field, vessel map, optic disc and fovea must
  all be measurable. Both deployed default examples pass 4/4 checks. The
  system then reports caliber, density, tortuosity, bifurcation, sparsity and
  quadrant distribution without manufacturing disease probabilities.
- Scope: explainable retinal microvascular phenotypes for association studies
  and longitudinal analysis. Disease-specific cardiovascular and stroke heads
  were not integrated because no compatible, task-trained public checkpoints
  were available with the reviewed repositories.

## AMD-specific OCT screening

- Public model: https://huggingface.co/tomalmog/oct-retinal-classifier (MIT),
  pinned revision `f199c1c8cfce6268ce138871a3baa707a4e8a076`.
- Artifact: `pytorch_model.bin`, loaded with the published EfficientNet-B3
  architecture. The AMD workbench reports CNV, drusen, DME and normal
  probabilities plus a gradient-based localization heatmap.
- The publisher reports 99.6% accuracy on the 968-image Kermany test set. This
  number is provenance metadata, not a locally reproduced external result.

## OCT IRF / SRF / PED segmentation

- Code: https://github.com/Animesh-Kr/OCT-Fluid-Segmentation (MIT).
- Public model: https://huggingface.co/animeshakr/oct-fluid-segmentation,
  pinned revision `e17b3888c267d7d7e56dc35096cf72a0ca85a422`.
- Artifact: `deployment/slot2_v2l_seed123.onnx` with its external data file.
  The deployed supplementary head reports separate IRF, SRF and PED masks,
  pixel areas, ratios, component counts and maximum heights.
- Quality gate: the publisher reports mean fluid Dice 0.2739 on a 503-slice,
  four-source test set (IRF 0.2043, SRF 0.1712, PED 0.4463). Because this is
  substantially below the validation score, this head is marked for review and
  does not replace the locally calibrated Duke total-fluid model.

## DeepSeeNet color-fundus AMD factors

- Official code: https://github.com/ncbi-nlp/DeepSeeNet (public NCBI research
  release). The five model artifacts were mirrored from the public Hugging Face
  exports at fixed revisions: drusen `cacd45e5f737f6fe6d0ca8ca0da294576ffac481`,
  pigment `9c9dbd67b8ac58bbb5bc9955822c2e97d5c84945`, late AMD
  `2d24c38b3128fbe1ba67480076ebcf9919a5dc94`, GA
  `8c6590b6d754da14e6ca4a1585a6f776ada97dd2`, and central GA
  `1a483a181db7059aea11162909a168a237ae66c6`.
- The original TensorFlow SavedModels were converted once to ONNX. Conversion
  parity was verified on both built-in fundus images; maximum absolute output
  error was `3.6e-7`.
- Outputs: drusen size, pigmentary abnormality, late AMD, geographic atrophy
  and central geographic atrophy, each with complete class probabilities.
- The DeepSeeNet paper reports AUC 0.94 for large drusen, 0.93 for pigmentary
  abnormalities and 0.97 for late AMD. The built-in longitudinal figure crops
  have no pixel-level ground truth, so their predictions remain unverified.

## OCTA CNV research audit

- CNV-Net code is public at
  https://github.com/mahsavali/CNV-Segmentation-Classification-OCTA and reports
  CNV mask segmentation plus activity criteria. The reviewed repository does
  not publish a deployable trained checkpoint.
- RBGNet's repository currently contains only a README and no released weights.
  Consequently the system retains the verified OCTA vessel checkpoint and
  quantitative morphology, and does not label a vessel-density map as a CNV
  segmentation model.
