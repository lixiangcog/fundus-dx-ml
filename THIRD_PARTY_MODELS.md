# Third-party model and sample provenance

This project is a research and teaching workbench. It is not a medical device.
Every integrated pipeline declares whether it is a trained model or a reference
algorithm in the API and UI.

## ReLayNet OCT structure segmentation

- Repository: https://github.com/ai-med/relaynet_pytorch
- Pinned commit: `40ae1aa56e426da14ddca37e06c2f31966febea5`
- License: MIT (copied to `third_party/relaynet_pytorch/LICENSE`)
- Integrated artifact: `models/Exp01/relaynet_epoch20.model`
- Note: historical checkpoint compatibility shims are applied at load time.
  Cross-device use requires independent validation and calibration.

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
- Current deployable engine: CPU multiscale Hessian vessel response, connected
  component cleanup, morphological skeletonization, and morphometry.
- Note: measurements are in pixels. Physical units require device pixel spacing,
  layer/slab metadata and clinical quality control.

## Quality enhancement reference

- OpenCV: https://github.com/opencv/opencv (Apache-2.0)
- Current deployable engine: CLAHE, non-local means denoising and conservative
  unsharp masking.
- Note: the algorithm cannot recreate tissue information lost during acquisition.

## Default examples

- OCT B-scan: `Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET`,
  file `OCT/OCT1/1221_OD_o_2.jpg`; see the repository dataset citation and terms.
- OCTA image: `aiforvision/OCTA-autosegmentation`, MIT, pinned commit above.
- Fundus photograph: existing project teaching sample.
