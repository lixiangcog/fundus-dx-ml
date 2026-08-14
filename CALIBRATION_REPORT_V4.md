# Calibration report v4

All results are for research validation only and are not clinical claims.
Weights, reference images and annotations remain outside Git.

## Evidence levels

- **Independent test:** subjects/images excluded from both gradient updates and
  checkpoint selection.
- **External fixed sample:** one annotated sample from a separate public dataset;
  useful for regression, not a cohort performance estimate.
- **Paired integration reference:** image/label pair from the upstream model
  repository; not independent clinical validation.
- **Unverified upload:** no ground truth; only output measurements are reported.

## OCT layer and fluid model

Source data release: https://github.com/ClinicalAI/MIRAGE. A MONAI 2D residual
U-Net was trained with seed `20260814`: subjects 6-9 development (33 images),
subject 10 selection (11), subjects 1-5 independent test (55).

- Independent-test mean layer Dice: `0.827450`.
- Independent-test fluid Dice: `0.679633`.
- Default layer case Subject_03_4 mean layer Dice: `0.8683`.
- Default fluid case Subject_01_5 fluid Dice: `0.7026`.

The released ReLayNet epoch-20 checkpoint was rejected after local Duke tests
failed to reach a deployable threshold. Physical thickness and volume require
device spacing/volume metadata and are not inferred from a single B-scan.

## OCTA vessels

Model/source: https://github.com/aiforvision/OCTA-autosegmentation, pinned source
commit `9cdc3137b6f55ae766dcae76c166ccf9774daf2b`, epoch-30 DynUNet checkpoint.
On the fixed generated image/graph pair G_20230216_232653: Dice `0.8221`, IoU
`0.6979`. This is a paired integration reference, not an independent clinical
test. Morphometry remains in pixel/% units without scan-width metadata.

## Color-fundus lesions

Model: https://huggingface.co/ClementP/fundus-lesions-segmentation-unet_seresnext50_32x4d
(MIT), U-Net with SE-ResNeXt50 encoder and upstream 1024-pixel autofit.
The complete 27-image standard IDRiD test split was evaluated. Mean per-case
macro Dice is `0.5146`; class means on positive cases are soft exudate `0.4795`,
hard exudate `0.6438`, hemorrhage `0.5451`, and microaneurysm `0.3829`.
IDRiD_67 is the fixed UI example because it maximizes the minimum class Dice
among cases containing all four lesion classes (selection rule disclosed):
macro Dice `0.6171`, soft exudate `0.6030`, hard exudate `0.6266`, hemorrhage
`0.7531`, and microaneurysm `0.4856`. The gate requires macro Dice >= `0.60`
and every present class Dice >= `0.40`; every class remains individually shown.

## Quality enhancement

The deployed OCT path uses the public OCT_DDPM checkpoint from source commit
`8dfb2e6` at the fixed, pre-calibrated timestep `14`. It was evaluated on 22
central macular B-scans from Duke DME Subjects 09 and 10 with deterministic
Gaussian degradation (`sigma=12`, seed base `20260814`). This Duke set is
external to the checkpoint repository but remains a paired synthetic-noise
test, not clinical acquisition validation.

- Mean PSNR: `27.11 -> 30.15 dB` (`+3.04 dB`).
- Mean SSIM: `0.6217 -> 0.7567` (`+0.1350`).
- Mean edge correlation: `0.8775 -> 0.9005`.
- Mean gradient error: `31.04 -> 24.86` (`-19.9%`).
- The previous NLM baseline reached PSNR `29.76 dB`, SSIM `0.6713`, and retained
  only `44.3%` of reference gradient energy; it was rejected for visible
  over-smoothing. OCT_DDPM retained `80.6%`.

The default example is Subject 10 B-scan 32. It was fixed before enhancement
scoring because it has the largest manual fluid burden among the two evaluated
subjects. Its paired result is PSNR `26.99 -> 30.00 dB`, SSIM
`0.6453 -> 0.7590`, and edge correlation `0.8630 -> 0.8845`. The gate requires
PSNR gain >= 3 dB, SSIM gain >= 0.10, and non-decreasing edge correlation.
OCTA and color-fundus uploads retain conservative modality-aware enhancement;
without paired truth their quality state remains unverified.

## Disease screening

The project FundusDx ResNet18 classifies the fixed IDRiD_55 case as diabetic
retinopathy with probability `0.960`. This is a single-case consistency check;
the CAM overlay is explicitly not a pixel lesion mask.

## Longitudinal AMD default case

CASE_001 is a de-identified Figure 3 paper case: March-June 2024, decimal BCVA
`0.3 -> 0.5`, OCT candidate lesion area `2.38 -> 1.28 mm2`, maximum height
`413.4 -> 354.9 um`, fundus lesion area `8.58 -> 6.58 mm2`, and OCTA CNV
candidate area `1.39 -> 0.08 mm2`.

These numbers are stored as `paper_reported_not_locally_recomputed`. Six figure
crops have distinct SHA-256 hashes but only about `93 x 99` native pixels, so
the case quality gate is `review`. Qwen2.5-VL, VisionUnite, and local imaging
