DeepGuard Detection Strategy
Phase 1 — Transfer Learning Approach (Building the Intelligence)

The biggest challenge in detecting Sora and Runway Gen-3 videos is that they are new technologies with very limited public data. Training a model directly on them would lead to severe overfitting and failure in real-world conditions.

Therefore, we adopt a Transfer Learning strategy, which follows the principle:

“First teach the model to walk, then teach it to run.”

Step A — Foundation Training (Physics of Real Motion)

We first train the model on large-scale real-world action datasets such as Kinetics-700 and UCF-101.

Purpose:
These datasets teach the model how real humans move — including eye blinking, head rotation, facial muscle motion, and how objects obey gravity and inertia.

Outcome:
The model develops an internal understanding of normal physical motion and biological behavior.

Step B — Specialization (Forgery Awareness)

Next, the model is fine-tuned on FaceForensics++.

Purpose:
To explicitly teach the model the difference between authentic videos and deepfake-manipulated videos.

Outcome:
The model learns how visual and motion patterns change when a video is artificially generated or manipulated.

Step C — Advanced Threat Evaluation (Sora & Runway)

Finally, the trained model is tested on a custom dataset containing Sora and Runway Gen-3 videos.

Purpose:
To verify that the model does not only detect traditional deepfakes, but can also generalize to modern generative video systems.

Outcome:
The system proves future-robustness against new AI video generators.

Phase 2 — Physics-Guided Validation (DeepGuard’s Unique Advantage)

Most existing systems analyze only pixel-level artifacts.
DeepGuard goes beyond that by analyzing motion physics.

Core Idea

When a person speaks or moves, the eyes, lips, cheeks, and facial muscles follow a biologically and physically constrained pattern.

In many deepfake or AI-generated videos:

The mouth moves

But the rest of the face remains unnaturally frozen

Or moves inconsistently with physics

DeepGuard’s Method

We compute Optical Flow motion vectors across the face and body.

If the motion violates:

natural smoothness

facial biomechanics

or Newtonian motion consistency

then the video is flagged as FAKE — even if it looks visually perfect.

Physics does not lie, even when pixels do.

Phase 3 — Hybrid Detection Architecture

DeepGuard uses a multi-stream decision pipeline:

Input:

User uploads a video.

Stream 1 — Visual Forensics (CNN)

A ResNet-based CNN checks for:

Blurring

Texture inconsistencies

GAN or diffusion artifacts

Stream 2 — Physics & Motion (Optical Flow + PINN)

An optical flow engine measures:

Facial motion

Lip-eye synchronization

Temporal smoothness

Fusion Rule

The outputs are combined:

If CNN says Real but Physics says Fake → Final decision = FAKE

Because physics-based violations cannot be visually faked reliably.

Implementation Roadmap
Week 1 — Data Setup

Create the full dataset folder structure

Download small samples of:

FaceForensics++

Sora videos

Prepare metadata.csv

Week 2 — System Skeleton

Run main.py and the inference pipeline

Verify that:

Video uploads work

Backend receives and processes files

Week 3 — Lightweight Training

Train the model on 50–100 videos only

Purpose is Proof of Concept, not full-scale training

Week 4 — Frontend Integration

Connect React dashboard to the FastAPI backend

User uploads video → DeepGuard returns:

Real/Fake verdict

Heatmaps and confidence score

Final Strategy Summary

We first teach the system physics, then we teach it to detect lies, and finally we challenge it with the newest AI criminals.
