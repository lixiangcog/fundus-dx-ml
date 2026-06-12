import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BookOpen, Code, Brain, Database, Layers, GitBranch, Cpu, Microscope, Zap } from 'lucide-react';

const HowItWorks = () => {
    const [activeTab, setActiveTab] = useState('simple');

    const tabVariants = {
        hidden: { opacity: 0, y: 20, scale: 0.95 },
        visible: {
            opacity: 1,
            y: 0,
            scale: 1,
            transition: { duration: 0.4, ease: "easeOut" }
        },
        exit: {
            opacity: 0,
            y: -20,
            scale: 0.95,
            transition: { duration: 0.3 }
        }
    };

    return (
        <motion.section
            className="w-full max-w-5xl mt-24 mb-12"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
        >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
                <h2 className="text-2xl md:text-3xl font-bold text-slate-800 flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-rose-500/15 to-orange-500/15 flex items-center justify-center border border-rose-500/20">
                        <Brain className="w-6 h-6 text-rose-500" />
                    </div>
                    <span>
                        <span className="font-display italic gradient-text">How</span>
                        {" "}it Works
                    </span>
                </h2>

                <div className="bg-slate-100 p-1.5 rounded-xl flex items-center border border-slate-200">
                    <button
                        onClick={() => setActiveTab('simple')}
                        className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-2 relative ${activeTab === 'simple'
                            ? 'text-white'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        {activeTab === 'simple' && (
                            <motion.div
                                layoutId="activeTab"
                                className="absolute inset-0 bg-gradient-to-r from-rose-500 to-orange-500 rounded-lg shadow-lg shadow-rose-500/25"
                                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                            />
                        )}
                        <span className="relative z-10 flex items-center gap-2">
                            <BookOpen className="w-4 h-4" />
                            Plain English
                        </span>
                    </button>
                    <button
                        onClick={() => setActiveTab('technical')}
                        className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-2 relative ${activeTab === 'technical'
                            ? 'text-white'
                            : 'text-slate-500 hover:text-slate-700'
                            }`}
                    >
                        {activeTab === 'technical' && (
                            <motion.div
                                layoutId="activeTab"
                                className="absolute inset-0 bg-gradient-to-r from-rose-500 to-orange-500 rounded-lg shadow-lg shadow-rose-500/25"
                                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                            />
                        )}
                        <span className="relative z-10 flex items-center gap-2">
                            <Code className="w-4 h-4" />
                            Technical
                        </span>
                    </button>
                </div>
            </div>

            <div className="glass-card rounded-3xl p-8 relative overflow-hidden">
                {/* Decorative background element */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-bl from-rose-500/5 to-transparent rounded-bl-full pointer-events-none" />
                <div className="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-orange-500/5 to-transparent rounded-tr-full pointer-events-none" />

                <AnimatePresence mode="wait">
                    {activeTab === 'simple' ? (
                        <motion.div
                            key="simple"
                            variants={tabVariants}
                            initial="hidden"
                            animate="visible"
                            exit="exit"
                            className="space-y-8 relative"
                        >
                            <div className="grid md:grid-cols-2 gap-8">
                                <motion.div
                                    className="space-y-4 group"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.1 }}
                                >
                                    <div className="w-14 h-14 bg-gradient-to-br from-rose-500/10 to-rose-500/5 rounded-xl flex items-center justify-center border border-rose-500/20 mb-4 group-hover:scale-110 transition-transform duration-300">
                                        <Microscope className="w-7 h-7 text-rose-500" />
                                    </div>
                                    <h3 className="text-xl font-bold text-slate-800">Pattern Recognition</h3>
                                    <p className="text-slate-500 leading-relaxed">
                                        Imagine an expert analyzing thousands of eye scans. Over time, patterns emerge, such as tiny spots or specific vessel changes, that correlate with certain conditions.
                                    </p>
                                    <p className="text-slate-500 leading-relaxed">
                                        Our model operates on a similar principle. By processing thousands of examples of healthy and affected eyes (including AMD, cataracts, and diabetic retinopathy), it has been trained to identify visual features associated with each condition.
                                    </p>
                                </motion.div>

                                <motion.div
                                    className="space-y-4 group"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                >
                                    <div className="w-14 h-14 bg-gradient-to-br from-amber-500/10 to-amber-500/5 rounded-xl flex items-center justify-center border border-amber-500/20 mb-4 group-hover:scale-110 transition-transform duration-300">
                                        <Zap className="w-7 h-7 text-amber-500" />
                                    </div>
                                    <h3 className="text-xl font-bold text-slate-800">Instant Analysis</h3>
                                    <p className="text-slate-500 leading-relaxed">
                                        When you upload an image, the AI breaks it down into millions of tiny pixels. It compares these pixels against the patterns it knows.
                                    </p>
                                    <p className="text-slate-500 leading-relaxed">
                                        In less than a second, it calculates a "confidence score." If it says 98% Cataract, it means the image looks 98% similar to the confirmed cataract cases it studied during training.
                                    </p>
                                </motion.div>
                            </div>

                            {/* Visual process flow */}
                            <motion.div
                                className="mt-8 pt-8 border-t border-slate-200"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 }}
                            >
                                <div className="flex flex-wrap items-center justify-center gap-4 text-sm">
                                    {['Upload Image', 'AI Analysis', 'Pattern Matching', 'Results'].map((step, i) => (
                                        <React.Fragment key={step}>
                                            <div className="flex items-center gap-2 px-4 py-2 bg-slate-50 rounded-full border border-slate-200">
                                                <span className="w-6 h-6 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 flex items-center justify-center text-xs font-bold text-white">
                                                    {i + 1}
                                                </span>
                                                <span className="text-slate-600">{step}</span>
                                            </div>
                                            {i < 3 && (
                                                <div className="hidden sm:block w-8 h-px bg-gradient-to-r from-rose-500/50 to-orange-500/50" />
                                            )}
                                        </React.Fragment>
                                    ))}
                                </div>
                            </motion.div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="technical"
                            variants={tabVariants}
                            initial="hidden"
                            animate="visible"
                            exit="exit"
                            className="space-y-8 relative"
                        >
                            <div className="grid md:grid-cols-2 gap-8">
                                <motion.div
                                    className="space-y-6"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.1 }}
                                >
                                    <div className="group">
                                        <h3 className="text-base font-bold text-rose-500 mb-3 flex items-center gap-2">
                                            <div className="w-8 h-8 rounded-lg bg-rose-500/10 flex items-center justify-center border border-rose-500/20">
                                                <Layers className="w-4 h-4" />
                                            </div>
                                            Model Architecture
                                        </h3>
                                        <p className="text-slate-500 text-sm leading-relaxed pl-10">
                                            We utilize a <span className="text-rose-600 font-semibold">ResNet-18</span> (Residual Neural Network) architecture. This is a Convolutional Neural Network (CNN) that is 18 layers deep. It uses "skip connections" to allow gradients to flow more easily during training, enabling the model to learn complex features without the vanishing gradient problem.
                                        </p>
                                    </div>

                                    <div className="group">
                                        <h3 className="text-base font-bold text-blue-500 mb-3 flex items-center gap-2">
                                            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center border border-blue-500/20">
                                                <Database className="w-4 h-4" />
                                            </div>
                                            Transfer Learning
                                        </h3>
                                        <p className="text-slate-500 text-sm leading-relaxed pl-10">
                                            Instead of training from scratch, we use <span className="text-blue-600 font-semibold">Transfer Learning</span>. The model was pre-trained on ImageNet (1.2 million images) to learn basic visual features (edges, textures). We then replaced the final fully connected layer (<code className="text-amber-600 bg-amber-500/10 px-1.5 py-0.5 rounded text-xs">fc</code>) to output our 4 specific classes: Normal, AMD, Cataract, and Diabetic Retinopathy.
                                        </p>
                                    </div>
                                </motion.div>

                                <motion.div
                                    className="space-y-6"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.2 }}
                                >
                                    <div className="group">
                                        <h3 className="text-base font-bold text-rose-500 mb-3 flex items-center gap-2">
                                            <div className="w-8 h-8 rounded-lg bg-rose-500/10 flex items-center justify-center border border-rose-500/20">
                                                <GitBranch className="w-4 h-4" />
                                            </div>
                                            Training Pipeline
                                        </h3>
                                        <ul className="space-y-2.5 text-sm text-slate-500 pl-10">
                                            <li className="flex items-start gap-2">
                                                <span className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 mt-2 shrink-0" />
                                                <span><span className="text-slate-700 font-medium">Optimizer:</span> SGD with Momentum (0.9) and learning rate 0.001</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 mt-2 shrink-0" />
                                                <span><span className="text-slate-700 font-medium">Loss Function:</span> CrossEntropyLoss for multi-class classification</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-rose-500 to-orange-500 mt-2 shrink-0" />
                                                <span><span className="text-slate-700 font-medium">Augmentation:</span> Random horizontal flips and rotations (±10°)</span>
                                            </li>
                                        </ul>
                                    </div>

                                    <div className="group">
                                        <h3 className="text-base font-bold text-amber-500 mb-3 flex items-center gap-2">
                                            <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                                                <Cpu className="w-4 h-4" />
                                            </div>
                                            Inference Process
                                        </h3>
                                        <div className="bg-slate-50 p-4 rounded-xl font-mono text-xs text-slate-500 border border-slate-200 ml-10 relative overflow-hidden">
                                            <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-rose-500 via-blue-500 to-amber-500" />
                                            <div className="space-y-1.5 pl-3">
                                                <div><span className="text-slate-400">// Input</span></div>
                                                <div>Image <span className="text-slate-400">(H, W, C)</span></div>
                                                <div className="text-rose-500">↓ <span className="text-slate-500">Resize (224×224) & Normalize</span></div>
                                                <div className="text-blue-500">↓ <span className="text-slate-500">ResNet-18 Forward Pass</span></div>
                                                <div className="text-blue-500">↓ <span className="text-slate-500">Logits <span className="text-slate-400">(1×4 Vector)</span></span></div>
                                                <div className="text-amber-500">↓ <span className="text-slate-500">Softmax Function</span></div>
                                                <div><span className="text-rose-500">→</span> <span className="text-rose-600 font-semibold">Probability Distribution</span></div>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </motion.section>
    );
};

export default HowItWorks;
