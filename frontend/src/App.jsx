import React, { useState, useRef } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import HowItWorks from './HowItWorks';

const SAMPLES = [
  { src: '/samples/normal.jpg', label: 'normal' },
  { src: '/samples/cataract.jpg', label: 'cataract' },
  { src: '/samples/diabetic_retinopathy.jpg', label: 'retinopathy' },
];

const CLASS_LABELS = {
  amd: 'AMD',
  cataract: 'Cataract',
  diabetic_retinopathy: 'Diabetic retinopathy',
  normal: 'Normal',
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [latency, setLatency] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const processFile = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setPrediction(null);
    setError(null);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) processFile(selectedFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      processFile(droppedFile);
    }
  };

  const loadSample = async (sample) => {
    const response = await fetch(sample.src);
    const blob = await response.blob();
    processFile(new File([blob], `sample_${sample.label}.jpg`, { type: 'image/jpeg' }));
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    const started = performance.now();
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${apiUrl}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setLatency(Math.round(performance.now() - started));
      setPrediction(response.data);
    } catch (err) {
      setError('The classifier could not be reached. The first request can take a moment while the backend wakes; try again shortly.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  const sortedProbs = prediction
    ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      <header className="border-b" style={{ borderColor: 'var(--line)' }}>
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <span className="text-lg font-bold tracking-tight">RetinaAI</span>
            <span className="mono-label hidden sm:inline">Fundus classifier</span>
          </div>
          <a
            href="https://github.com/wuchris-ch/fundus-dx-ml"
            target="_blank"
            rel="noreferrer"
            className="mono-label transition-colors hover:text-[var(--accent-ink)]"
            style={{ color: 'var(--ink-secondary)' }}
          >
            Source ↗
          </a>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 w-full flex-1">
        {/* Intro */}
        <div className="pt-14 pb-10 max-w-[44rem]">
          <h1 className="text-3xl sm:text-[2.5rem] leading-[1.15] font-semibold tracking-tight">
            Four-class retinal screening from a single fundus photograph.
          </h1>
          <p className="mt-4 text-base leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
            A ResNet18 fine-tuned on 4,355 labeled fundus images distinguishes AMD, cataract,
            diabetic retinopathy, and healthy eyes. Upload a photograph or load a sample;
            inference runs live on the deployed model.
          </p>
        </div>

        {/* Worksheet */}
        <section
          className="grid lg:grid-cols-2 border"
          style={{ borderColor: 'var(--line)', background: 'var(--surface)' }}
        >
          {/* Specimen column */}
          <div className="p-6 sm:p-8 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <span className="mono-label">01 · Specimen</span>
              {file && (
                <button
                  onClick={reset}
                  className="mono-label transition-colors hover:text-[var(--accent-ink)] cursor-pointer"
                  style={{ color: 'var(--ink-secondary)' }}
                >
                  Clear
                </button>
              )}
            </div>

            <div
              className="reg-zone flex-1 min-h-[300px] border border-dashed flex items-center justify-center cursor-pointer transition-colors"
              style={{ borderColor: 'var(--line-strong)' }}
              data-active={!!preview}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              role="button"
              tabIndex={0}
              aria-label="Upload a fundus image"
            >
              <span className="reg-mark" />
              <span className="reg-mark" />
              <span className="reg-mark" />
              <span className="reg-mark" />
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept="image/*"
              />
              {preview ? (
                <img
                  src={preview}
                  alt="Selected fundus"
                  className="max-h-[320px] max-w-full object-contain p-3"
                />
              ) : (
                <div className="text-center px-8 py-12">
                  <p className="font-medium" style={{ color: 'var(--ink-secondary)' }}>
                    Drop a fundus image here, or click to browse
                  </p>
                  <p className="mono-label mt-3">JPG · PNG · resized to 224 × 224</p>
                </div>
              )}
            </div>

            <div>
              <span className="mono-label">No image at hand · try a sample</span>
              <div className="mt-2.5 flex gap-2.5">
                {SAMPLES.map((sample) => (
                  <button
                    key={sample.label}
                    onClick={() => loadSample(sample)}
                    className="group text-left cursor-pointer"
                    aria-label={`Load ${sample.label} sample image`}
                  >
                    <img
                      src={sample.src}
                      alt=""
                      className="w-16 h-16 object-cover border transition-colors group-hover:border-[var(--accent)]"
                      style={{ borderColor: 'var(--line)' }}
                    />
                    <span className="mono-label block mt-1.5 !text-[0.625rem]">{sample.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="w-full py-3.5 font-semibold text-[0.9375rem] transition-colors cursor-pointer disabled:cursor-not-allowed"
              style={
                !file
                  ? { background: 'var(--surface-sunken)', color: 'var(--ink-muted)' }
                  : { background: 'var(--ink)', color: 'var(--paper)' }
              }
            >
              {loading ? 'Analyzing…' : 'Run classification'}
            </button>
          </div>

          {/* Readout column */}
          <div
            className="p-6 sm:p-8 border-t lg:border-t-0 lg:border-l flex flex-col"
            style={{ borderColor: 'var(--line)' }}
          >
            <span className="mono-label">02 · Readout</span>

            <div className="flex-1 mt-5">
              <AnimatePresence mode="wait">
                {loading ? (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="space-y-6"
                    aria-live="polite"
                    aria-busy="true"
                  >
                    <div className="skeleton h-4 w-24" />
                    <div className="skeleton h-10 w-3/5" />
                    <div className="space-y-4 pt-4">
                      {[0, 1, 2, 3].map((i) => (
                        <div key={i} className="space-y-2">
                          <div className="skeleton h-3 w-40" />
                          <div className="skeleton h-1 w-full" />
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ) : error ? (
                  <motion.div
                    key="error"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="border p-5"
                    style={{ borderColor: 'var(--line-strong)' }}
                    role="alert"
                  >
                    <p className="mono-label" style={{ color: 'var(--accent-ink)' }}>
                      Request failed
                    </p>
                    <p className="mt-2 text-sm leading-relaxed" style={{ color: 'var(--ink-secondary)' }}>
                      {error}
                    </p>
                  </motion.div>
                ) : prediction ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <p className="mono-label">Prediction</p>
                    <p className="mt-1.5 text-3xl sm:text-4xl font-semibold tracking-tight">
                      {CLASS_LABELS[prediction.prediction] || prediction.prediction}
                    </p>
                    <p className="mono-data mt-2 text-sm" style={{ color: 'var(--ink-secondary)' }}>
                      {(prediction.confidence * 100).toFixed(1)}% confidence
                      {latency != null && <span style={{ color: 'var(--ink-muted)' }}> · {latency} ms</span>}
                    </p>

                    <div className="mt-8 space-y-4">
                      {sortedProbs.map(([className, prob]) => {
                        const isTop = className === prediction.prediction;
                        return (
                          <div key={className}>
                            <div className="flex justify-between items-baseline mb-1.5">
                              <span
                                className="text-sm"
                                style={{
                                  color: isTop ? 'var(--ink)' : 'var(--ink-secondary)',
                                  fontWeight: isTop ? 600 : 400,
                                }}
                              >
                                {CLASS_LABELS[className] || className}
                              </span>
                              <span
                                className="mono-data text-sm"
                                style={{ color: isTop ? 'var(--ink)' : 'var(--ink-muted)' }}
                              >
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="prob-track">
                              <div
                                className="prob-fill"
                                data-top={isTop}
                                style={{ width: `${Math.max(prob * 100, 0.5)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <p className="mono-label mt-8 !normal-case !tracking-normal" style={{ color: 'var(--ink-muted)' }}>
                      Softmax over 4 classes · ResNet18 · 224×224 input
                    </p>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="h-full flex flex-col justify-center"
                  >
                    <p className="text-sm leading-relaxed max-w-[36ch]" style={{ color: 'var(--ink-muted)' }}>
                      Load an image and run the classifier. The model returns a probability
                      for each of the four classes; the full distribution appears here.
                    </p>
                    <div className="mt-6 space-y-4" aria-hidden="true">
                      {['Normal', 'Cataract', 'Diabetic retinopathy', 'AMD'].map((name) => (
                        <div key={name}>
                          <div className="flex justify-between items-baseline mb-1.5">
                            <span className="text-sm" style={{ color: 'var(--ink-muted)' }}>{name}</span>
                            <span className="mono-data text-sm" style={{ color: 'var(--ink-muted)' }}>—</span>
                          </div>
                          <div className="prob-track" />
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </section>

        <HowItWorks />
      </main>

      {/* Footer */}
      <footer className="border-t mt-16" style={{ borderColor: 'var(--line)' }}>
        <div className="max-w-5xl mx-auto px-6 py-5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <p className="text-sm" style={{ color: 'var(--ink-secondary)' }}>
            Educational demonstration. Not a medical device.
          </p>
          <p className="mono-label">ResNet18 · PyTorch · FastAPI · 98.0% val acc (n=871)</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
