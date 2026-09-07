# LLM Security Analyzer — Prompt Injection Detector

Multi-layered, model-agnostic security assessment platform for LLM applications. 

This repository contains **Detector #1 (Prompt Injection & Jailbreak Prevention)** implementing a 4-stage pipeline combining pre-processing deobfuscation, lightweight sequence classification, regex intent filtering, and LLM-as-a-Judge semantic evaluation.

---

## 🏗️ Architecture

```text
Raw User Prompt
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 1: Pre-processing & Deobfuscation                 │
│ (Base64 multi-pass, HTML entity unescape, NFKC, zero-width)│
└──────────────────────────┬───────────────────────────────┘
                           │ Clean Text
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 2: DeBERTa-v2 Sequence Classifier                 │
│ (Local Fine-Tuned Transformer Model)                    │
└──────────────────────────┬───────────────────────────────┘
                           │ Injection Probability Score [0.0 - 1.0]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 2.5: Regex Intent & Fast-Path Filter               │
│ (Fast-paths explicit attacks / Detects technical intent) │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
  CONFIRMED_INJECTION                ROUTE_TO_JUDGE
               │                           │
               ▼                           ▼
       [Flag Injection]         ┌──────────────────────────┐
                                │ Layer 3: LLM Judge       │
                                │ (Groq / OpenAI / Gemini) │
                                └──────────┬───────────────┘
                                           │
                                    Final Verdict
```

---

## 📥 Model Weights Download

Due to file size limits, the fine-tuned DeBERTa model weights (`428 MB`) are hosted on Google Drive.

1. **Download the model zip:** [Download `injection-classifier.zip` from Google Drive](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)
2. Extract the zip into the project root directory as `model_tmp/`:

```powershell
# Directory structure must look like:
final_project/
└── model_tmp/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/HemashreeJelli/LLM_Security_Analyser-.git
cd LLM_Security_Analyser-
```

### 2. Install Dependencies
```bash
pip install torch transformers sentencepiece protobuf python-dotenv openai pytest
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and enter your preferred LLM Judge API key (Groq, OpenAI, or Gemini):

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Groq (Recommended - Free & Fast from console.groq.com)
GROQ_API_KEY=gsk_your_groq_key_here
```

---

## 🚀 Running the Pipeline

Launch the interactive REPL:

```bash
python run_pipeline.py
```

### Interactive Commands
- **Type any prompt:** Processes the prompt through all 4 layers in real-time.
- **Type `demo`:** Runs 5 built-in test cases demonstrating obfuscation decoding, false-positive resolution, and LLM judge routing.
- **Type `quit`:** Exits the application.

---

## 📁 Repository Structure

```text
.
├── detectors/
│   ├── base.py                   # BaseDetector interface contract
│   └── prompt_injection/         # Detector #1 Package
│       ├── preprocessing.py      # Layer 1: Obfuscation stripper
│       ├── classifier.py         # Layer 2: DeBERTa model wrapper
│       ├── intent_filter.py      # Layer 2.5: Intent & pattern filter
│       ├── llm_judge.py          # Layer 3: Semantic judge evaluator
│       └── detector.py           # Top-level PromptInjectionDetector class
├── llm_judge/
│   └── backends/                 # API adapters (Groq, OpenAI, Gemini)
├── run_pipeline.py               # Interactive CLI tester
├── .env.example                  # Environment variable template
└── .gitignore                    # Excludes .env and model weights
```
