import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const PIPELINE_STEPS = [
  { label: 'Input image', note: '(H, W, C)' },
  { label: 'Resize 224×224 + ImageNet normalize', note: null },
  { label: 'ResNet18 forward pass', note: null },
  { label: 'Logits', note: '(1×4)' },
  { label: 'Softmax', note: null },
  { label: 'Probability distribution', note: null },
];

const Block = ({ index, title, children }) => (
  <div>
    <h3 className="flex items-baseline gap-3 text-base font-semibold">
      <span className="mono-data text-sm font-normal" style={{ color: 'var(--accent-ink)' }}>
        {String(index).padStart(2, '0')}
      </span>
      {title}
    </h3>
    <div className="mt-3 space-y-3 text-sm leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
      {children}
    </div>
  </div>
);

const HowItWorks = () => {
  const [activeTab, setActiveTab] = useState('simple');

  const tabs = [
    { id: 'simple', label: 'Plain English' },
    { id: 'technical', label: 'Technical' },
  ];

  return (
    <section className="mt-16">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
        <div>
          <span className="mono-label">03 · Notes</span>
          <h2 className="text-2xl font-semibold tracking-tight mt-1.5">How it works</h2>
        </div>

        <div
          className="flex border self-start sm:self-auto"
          style={{ borderColor: 'var(--line)' }}
          role="tablist"
          aria-label="Explanation level"
        >
          {tabs.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={active}
                onClick={() => setActiveTab(tab.id)}
                className="px-4 py-2 text-sm font-medium transition-colors cursor-pointer"
                style={
                  active
                    ? { background: 'var(--ink)', color: 'var(--paper)' }
                    : { background: 'transparent', color: 'var(--ink-secondary)' }
                }
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="border p-6 sm:p-8" style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}>
        <AnimatePresence mode="wait">
          {activeTab === 'simple' ? (
            <motion.div
              key="simple"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="grid md:grid-cols-2 gap-x-12 gap-y-8"
            >
              <Block index={1} title="Pattern recognition">
                <p>
                  Imagine an expert analyzing thousands of eye scans. Over time, patterns emerge,
                  such as tiny spots or specific vessel changes, that correlate with certain conditions.
                </p>
                <p>
                  The model works on the same principle. By processing thousands of examples of
                  healthy and affected eyes (including AMD, cataracts, and diabetic retinopathy),
                  it learned the visual features associated with each condition.
                </p>
              </Block>

              <Block index={2} title="Instant analysis">
                <p>
                  When you upload an image, the model breaks it down into pixels and compares
                  them against the patterns it knows.
                </p>
                <p>
                  In under a second it produces a confidence score. A 98% cataract score means the
                  image looks 98% similar to the confirmed cataract cases it studied during training.
                </p>
              </Block>

              <div className="md:col-span-2 pt-6 border-t" style={{ borderColor: 'var(--line)' }}>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                  {['Upload image', 'Analysis', 'Pattern matching', 'Result'].map((step, i, arr) => (
                    <React.Fragment key={step}>
                      <span className="text-sm" style={{ color: 'var(--ink-secondary)' }}>
                        <span className="mono-data mr-2" style={{ color: 'var(--ink-muted)' }}>{i + 1}</span>
                        {step}
                      </span>
                      {i < arr.length - 1 && (
                        <span aria-hidden="true" style={{ color: 'var(--line-strong)' }}>→</span>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="technical"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="grid md:grid-cols-2 gap-x-12 gap-y-8"
            >
              <div className="space-y-8">
                <Block index={1} title="Model architecture">
                  <p>
                    A <strong className="font-semibold" style={{ color: 'var(--ink)' }}>ResNet18</strong>:
                    an 18-layer convolutional network with skip connections, which let gradients
                    flow through deep layers during training and avoid the vanishing-gradient problem.
                  </p>
                </Block>

                <Block index={2} title="Transfer learning">
                  <p>
                    The network starts from ImageNet weights (1.2M images), so it already detects
                    edges and textures. The final fully connected layer is replaced to output the
                    4 target classes: normal, AMD, cataract, and diabetic retinopathy.
                  </p>
                </Block>
              </div>

              <div className="space-y-8">
                <Block index={3} title="Training pipeline">
                  <ul className="space-y-2">
                    <li><strong className="font-medium" style={{ color: 'var(--ink)' }}>Optimizer:</strong> SGD, momentum 0.9, learning rate 0.001</li>
                    <li><strong className="font-medium" style={{ color: 'var(--ink)' }}>Loss:</strong> cross-entropy over 4 classes</li>
                    <li><strong className="font-medium" style={{ color: 'var(--ink)' }}>Augmentation:</strong> random horizontal flips, rotations (±10°)</li>
                    <li><strong className="font-medium" style={{ color: 'var(--ink)' }}>Early stop:</strong> at target validation accuracy</li>
                  </ul>
                </Block>

                <Block index={4} title="Inference">
                  <div
                    className="mono-data text-xs leading-relaxed border p-4"
                    style={{ borderColor: 'var(--line)', background: 'var(--surface-sunken)', color: 'var(--ink-secondary)' }}
                  >
                    {PIPELINE_STEPS.map((step, i) => (
                      <div key={step.label} className="flex gap-2">
                        <span style={{ color: 'var(--ink-muted)' }}>{i === 0 ? ' ' : '↓'}</span>
                        <span>
                          {step.label}
                          {step.note && <span style={{ color: 'var(--ink-muted)' }}> {step.note}</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                </Block>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};

export default HowItWorks;
