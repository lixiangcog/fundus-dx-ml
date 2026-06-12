# Product

## Register

product

## Users

Recruiters and hiring managers opening the live demo link from a resume, at a desk in daylight, giving it 30 to 60 seconds. Secondary: engineers and curious visitors who actually upload an image. The job to be done: judge in under a minute whether the author ships credible, precise, production-quality work.

## Product Purpose

RetinaAI is a portfolio demonstration of an end-to-end ML system: a ResNet18 fundus-image classifier (AMD, cataract, diabetic retinopathy, normal) served by FastAPI and consumed by this React UI. The interface's success criterion is credibility: the visitor tries one prediction, understands the output instantly, and walks away thinking "this person sweats the details."

## Brand Personality

Clinical-calm precision. Three words: instrument, composed, exact. The interface should feel like a well-calibrated piece of diagnostic equipment, not a tech demo. Quiet confidence over spectacle; the most colorful object on screen should be the fundus photograph itself.

## Anti-references

- Generic AI-demo template: gradient text, glassmorphism cards, floating particles, mesh gradients, scan-line effects.
- First-order healthcare reflex: sterile white + teal "telehealth startup" look.
- Second-order reflex: dark sci-fi "AI scanner" aesthetic with neon glows.
- SaaS hero-metric layouts and identical icon-card grids.

## Design Principles

1. The image is the color: the interface stays in calm tinted neutrals so the uploaded retina photo carries the visual weight.
2. Precision is the decoration: alignment, hairlines, tabular numerals, and exact spacing do the aesthetic work that effects used to do.
3. One glance, one answer: prediction and confidence must be readable in under two seconds, with detail (per-class probabilities) available below.
4. Honest instrument: state limitations plainly (educational demo, not a medical device); credibility comes from restraint, not claims.
5. Nothing moves without a reason: motion confirms state changes only; respect reduced-motion preferences.

## Accessibility & Inclusion

WCAG AA contrast minimum for all text and meaningful UI. `prefers-reduced-motion` honored for all animation. Color is never the only carrier of meaning (prediction is labeled in text, not just bar color). Keyboard-operable upload flow.
