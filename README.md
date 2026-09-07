# LLM Security Analyzer

A multi-layered, modular security assessment platform for Large Language Models (LLMs), designed to scan prompts and responses for injection attacks, jailbreaks, and sensitive data leakage in real-time.

---

## 🚀 Features & Detectors Completed (Phases 1-3)

### 1. Prompt Injection Detector (Layered)
* **Layer 1:** Advanced Pre-processing & Deobfuscation (Base64 decoding, HTML entity unescaping, Zero-width character stripping).
* **Layer 2:** DeBERTa-v2 Sequence Classifier (Local HuggingFace model).
* **Layer 2.5:** Regex Intent & Fast-Path Filter.
* **Layer 3:** LLM-as-a-Judge for semantic analysis of "gray zone" attacks.

### 2. Jailbreak Detector (Layered)
* **Layer 1:** Rules & Regex Engine (Detects persona adoption, fictional framing, DAN patterns).
* **Layer 2:** HuggingFace Transformer Classifier (Fine-tuned specifically on jailbreak methodologies).
* **Layer 3:** Multi-Turn Escalation detection.
* **Layer 4:** LLM-as-a-Judge (Human-in-the-Loop Proxy) to catch "Goal Displacement" and confident false-negatives.

### 3. Data Leakage & PII Scanner
* **Layer 1:** Regex & Checksums (Detects secrets, API keys, and uses Luhn/Verhoeff algorithms for Credit Cards & Indian Aadhaar/PAN cards).
* **Layer 2:** Microsoft Presidio & SpaCy NER (Named Entity Recognition for natural language PII).
* **Layer 3:** Prompt Leakage Detection (N-gram overlap and fuzzy sequence matching to ensure the LLM hasn't leaked its system prompt).

### 4. Async Orchestrator & Risk Scoring Engine
* **Concurrency:** Runs all 3 detectors simultaneously using `asyncio.gather`, achieving an average latency of ~250ms (well under the 1.5s PRD limit).
* **Logistic Regression Scoring:** Instead of a simple linear average that dilutes critical flags, the scoring engine uses a Sigmoid activation curve `1 / (1 + exp(-logit))`. If *any* detector flags a high-confidence threat, the composite risk score mathematically snaps to >95/100 (CRITICAL).

---

## 🧠 Model Weights Setup

Due to GitHub's file size limits, the fine-tuned Hugging Face models are excluded from tracking. You must ensure the following directories exist in your project root before running:

1. `model_tmp/` (Prompt Injection Weights - ~428 MB)
2. `jailbreak-classifier/` (Jailbreak Weights - ~560 MB)

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
# Requires Python 3.10+
pip install python-dotenv torch transformers sentencepiece protobuf presidio-analyzer presidio-anonymizer spacy openai

# Download the SpaCy NLP model required for Presidio NER
python -m spacy download en_core_web_lg
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and enter your preferred LLM Judge API key (Groq is set as default for low latency):

```env
GROQ_API_KEY=gsk_your_groq_key_here
```

---

## ⚡ Running the Pipeline

We have replaced the individual test scripts with a master asynchronous orchestrator. To launch the interactive REPL and test all three detectors simultaneously:

```bash
python run_pipeline_async.py
```

### Interactive Commands
- **Type any prompt:** Processes your prompt through the Prompt Injection, Jailbreak, and Data Leakage layers simultaneously.
- **`quit` or `exit`:** Shuts down the pipeline.

The console will output the individual sub-scores, inner evidence structure (JSON), and the final **Composite Risk Score** and **Severity Band**.

---

## 🏗️ Repository Structure

```text
.
├── detectors/
│   ├── base.py                   # BaseDetector interface contract
│   ├── data_leakage/             # Detector #3 Package (PII, Secrets, Presidio)
│   ├── jailbreak/                # Detector #2 Package (Jailbreak, Escalation)
│   └── prompt_injection/         # Detector #1 Package (Obfuscation, DeBERTa)
├── llm_judge/
│   └── backends/                 # API adapters (Groq, OpenAI, Gemini)
├── run_pipeline_async.py         # Async Master Pipeline & REPL
├── run_jailbreak.py              # Legacy standalone test script
├── run_leakage.py                # Legacy standalone test script
├── .env.example                  # Environment variable template
└── .gitignore                    # Excludes .env and heavy model weights
```
