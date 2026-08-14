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
