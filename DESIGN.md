# Design

Target visual system for the RetinaAI frontend (register: product, personality: clinical-calm precision).

## Theme

Light. Scene: a recruiter opens the link at a desk in daylight and judges credibility in under a minute. Warm paper-tinted neutrals, not sterile white; the uploaded fundus photograph (orange/amber) is the most saturated object on screen.

## Color

OKLCH, restrained strategy: tinted neutrals plus one accent under 10% of surface area.

- `--paper`: oklch(0.975 0.004 80) — page background, warm paper
- `--surface`: oklch(0.99 0.003 80) — panel background
- `--ink`: oklch(0.24 0.012 60) — primary text, warm near-black
- `--ink-secondary`: oklch(0.45 0.012 60)
- `--ink-muted`: oklch(0.60 0.010 60)
- `--line`: oklch(0.88 0.006 70) — hairline borders
- `--line-strong`: oklch(0.78 0.008 70)
- `--accent`: oklch(0.50 0.11 45) — retina sienna (deep burnt amber, drawn from fundus imagery)
- `--accent-soft`: oklch(0.93 0.025 55) — accent tint for fills
- `--ok`: oklch(0.52 0.10 150) — success/normal, used sparingly
- `--warn`: oklch(0.55 0.12 70) — mid confidence

No #000/#fff. No gradients on text or buttons. No teal.

## Typography

- UI/body: Plus Jakarta Sans (400/500/600/700)
- Data, labels, microcopy: IBM Plex Mono (400/500) — uppercase tracked labels, tabular numerals for probabilities and confidence
- Headline scale ratio ≥1.25; body max width 70ch
- The wordmark "RetinaAI" is set in sans, solid ink, no gradient

## Components

- Panels: single 1px `--line` border, flat `--surface`, 2–6px radius (instrument, not bubble). No glassmorphism, no drop-shadow stacks; at most one 1–2px shadow for elevation on hover.
- Upload zone: hairline dashed border with crosshair corner registration marks; mono microcopy for accepted formats.
- Probability readout: rows with mono numerals, thin (4px) bars; accent color only on the predicted class, neutral bars elsewhere.
- Buttons: solid `--ink` primary (paper text), quiet bordered secondary. No glow, no translate-Y hover; background/border shifts only.
- Sample-image chips: small thumbnails with hairline borders so visitors can try the model without their own fundus photos.

## Motion

Confirms state changes only (result reveal, bar fill). ease-out-quart/expo, 200–500ms. No infinite loops, no particles, no scan lines, no pulsing. `prefers-reduced-motion` disables all of it.

## Layout

Compact top bar (wordmark left, status/links right), one restrained headline block, then a worksheet: upload and readout as two columns separated by a hairline rather than floating cards. Footer carries the educational-use disclaimer and model facts (ResNet18, 4 classes, val accuracy) in mono.
