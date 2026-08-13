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

## Quality enhancement reference

- OpenCV: https://github.com/opencv/opencv (Apache-2.0)
- Current deployable engine: calibrated non-local means denoising. Its fixed
  synthetic-noise example is checked with paired PSNR and SSIM gains.
- Note: the algorithm cannot recreate tissue information lost during acquisition.

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
