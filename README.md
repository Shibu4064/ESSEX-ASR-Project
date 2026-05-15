# ESSEX-ASR-Project

<h2>Literature Review</h2>
<href>https://docs.google.com/spreadsheets/d/10UlfkrQj_z7OrkhekFt-zFD4gF8Z_hLCIQSyNADnq7M/edit?gid=0#gid=0</href>

<h2>Meeting Minutes</h2>
<href>https://essexuniversity-my.sharepoint.com/personal/ad25534_essex_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fad25534%5Fessex%5Fac%5Fuk%2FDocuments%2FCE903%5FGroup%5FMeetings&ga=1</href>

<h2>Dataset Links</h2>
<h3>LibriSpeech</h3>
<href>https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus</href>

<h3>Mozilla Common Voice</h3>
<href>https://www.kaggle.com/datasets/eddiehoogewerf/mozilla-commonvoice</href>



# Omnilanguage_ASR — Cross-Domain Automatic Speech Recognition (ASR)

This repository contains an end-to-end **cross-domain ASR pipeline** with experiments across **LibriSpeech (LS)** and **Mozilla Common Voice (MCV)** using multiple model families:
- **Whisper** (incl. **LoRA** parameter-efficient fine-tuning)
- **XLS-R / wav2vec 2.0** (HF Transformers)
- **Conformer-CTC** baselines (custom + **NVIDIA NeMo**)
- **QuartzNet-5*5 baselines**

The focus is **domain transfer** (e.g., LS → MCV, MCV → LS), **robustness**, and **reproducible evaluation** via WER/CER across in-domain and cross-domain settings.

---

## Repo Contents (What’s here)

### Key documents
- `ASR_initial_roadmap.docx` — initial plan/roadmap for the project
- `CE-903_Team-7_Assignment-1.pdf` — course submission / assignment artifact
- `README.md` — you’re reading it 🙂

### Notebooks (main work)
| Notebook | Purpose (high level) |
|---|---|
| `Omnilanguage_ASR` | Omnilingual ASR 7 language integrated software application | Whisper Model |
| `eda-of-mcv-and-ls.ipynb` | EDA of MCV + LS (duration, transcripts, OOV, WPS, etc.) |
| `librispeech-asr-corpus.ipynb` |QuartzNet Model | LibriSpeech data prep / exploration utilities |
| `cross-domain-asr-pipeline.ipynb` | Full Whisper Model pipeline glue: preprocessing → training → evaluation (cross-domain) |
| `whisper-small-mcv.ipynb` | Whisper-small fine-tuning / baseline on MCV |
| `whisper-lora-cross-domain.ipynb` | Whisper + Novel LoRA cross-domain training/eval |
| `Conformer-CTC-mcv.ipynb` | Conformer-CTC baseline (MCV) |
| `NVIDIA_NeMo_Conformer-CTC-Small_mcv.ipynb` | NeMo Conformer-CTC-Small fine-tuned/tested on MCV |
| `xls-r-mcv.ipynb` | XLS-R training/eval on MCV |
| `cross-domain-xls-r-300m.ipynb` | Cross-domain training/eval with XLS-R 300M |
| `xlsr300m-fixed.ipynb` | Stabilized/fixed version of XLS-R 300M notebook |
| `xlsr300m-fixed__2_.ipynb` | Variant/iteration of fixed XLS-R 300M workflow |
| `crossdomainasr-facebook-wav2vec2-conformer-large.ipynb` | Wav2Vec2/Conformer-large cross-domain experiment |

### Result snapshots (quick visual references)
- `Screenshot_2026-02-20_at_12.48.32_PM.png` — Whisper LoRA training on LS (result snapshot)
- `Screenshot_2026-02-20_at_12.49.03_PM.png` — Whisper LoRA testing on CV (result snapshot)

---

## What this project does (Workflow)

1. **Data ingestion & normalization**
   - Load LS / MCV metadata
   - Standardize transcript format, filter invalid/empty samples
   - (Optional) compute OOV stats, transcript/lexical analysis

2. **Feature / token preparation**
   - HF datasets + tokenization (Whisper tokenizer / wav2vec2/XLS-R tokenizer)
   - CTC label prep for Conformer/XLS-R style models

3. **Model training**
   - Full fine-tuning (baseline)
   - Parameter-efficient tuning (LoRA for Whisper)
   - NeMo Conformer training where applicable

4. **Evaluation**
   - WER/CER (primary)
   - Cross-domain generalization checks (train in A, test in B)

---

## Setup

### Recommended environment
- Python **3.9+**
- A CUDA-capable GPU is strongly recommended (P100/T4/V100/A100 etc.)
- Tested best with: `torch`, `transformers`, `datasets`, `accelerate`, `jiwer`

### Install (typical)
Create a virtual env, then:

```bash
pip install -U pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate evaluate jiwer librosa soundfile tqdm pandas numpy matplotlib
