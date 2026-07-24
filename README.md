# SignFlow Studio — Peak AI Sign Language Platform

![SignFlow Status](https://img.shields.io/badge/Status-Peak%20Platform-success?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Stack-MediaPipe%20%2B%20PyTorch%20%2B%20FastAPI%20%2B%20Three.js%20%2B%20WebSpeech-blueviolet?style=for-the-badge)

SignFlow Studio is a peak, full-spectrum AI Sign Language platform combining **Continuous Recognition (CSLR)**, **Reverse 3D Avatar Production (SLP)**, **Live Voice-to-3D Sign Dictation**, **Joint Flexion Spatial Inspector**, and **Gamified ASL Pose Trainer**.

---

## 🔥 Peak Features Matrix

### 1. 🖐️ Continuous Sign Recognition (Camera → Text/Speech)
- **Unsegmented CTC Sequence Recognition**: Powered by PyTorch BiLSTM + `CTCLoss`.
- **Mobile Hotspot IP Phone Support**: Streams wireless phone cameras (`http://10.120.195.42:8080/video`) via same-origin local proxy (`/proxy_frame`).
- **Strict Geometric Classifier**: 0 false positives when hands are still or missing.
- **Web Speech Synthesis (TTS)**: Reads recognized sentences out loud.

### 2. 🎙️ Live Voice Microphone Dictation to 3D Avatar (Speech → 3D Sign)
- Speak into your laptop/phone microphone using `SpeechRecognition`.
- Automatically translates spoken English into an ASL Gloss sequence and animates the **Three.js 3D Avatar** in real time!

### 3. 📐 Real-Time Spatial Joint Flexion Inspector
- Analyzes fingertip flexion angles ($\text{Index: 85}^\circ$, $\text{Middle: 90}^\circ$, etc.) in real time below the video viewport for diagnostic hand pose inspection.

### 4. 🎓 Gamified ASL Practice & Pose Trainer
- Interactive challenge mode ("Duolingo for ASL").
- Prompts target signs (`PEACE`, `OPEN HAND`, `I LOVE YOU`, `L-SHAPE`), scores your hand pose in real-time, and gives instant accuracy feedback.

---

## 🚀 How to Run

1. **Launch Studio Platform**:
   Open **[http://localhost:8000](http://localhost:8000)** in your browser.

2. **Explore Studio Modes**:
   - **Tab 1: `1. Camera Recognition (CSLR)`**: Wireless IP phone camera sign translation.
   - **Tab 2: `2. 3D Avatar (Text/Voice → 3D)`**: Type text or click **"🎙️ Mic Dictate"** to speak into your microphone and watch the 3D avatar sign!
   - **Tab 3: `3. ASL Trainer Studio`**: Practice ASL signs with live pose scoring.
