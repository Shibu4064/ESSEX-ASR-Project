# ASR Pro: Advanced Speech Recognition

A state-of-the-art Automatic Speech Recognition (ASR) system built using `openai/whisper-large-v3-turbo`, augmented with Parameter-Efficient Fine-Tuning (LoRA) and deeply integrated translation capabilities using NLLB (No Language Left Behind).

This application was designed specifically to achieve highly accurate, real-time transcription across both high-resource (English, French, German) and low-resource Indic languages (Hindi, Marathi, Bengali, Urdu). The system natively targets Character Error Rates (CER) of strictly less than 5%.

The pipeline has been thoroughly trained leveraging the proprietary **"Omnilingual Dataset"**, a high-density cross-lingual corpus that ensures the robust generalization across extremely diverse linguistic topologies.

## Key Features

- **Live Transcription & Uploading**: Support for Web Audio API recording or file uploads.
- **Dynamic Decoding**: Deep Beam Search with n-gram repetition penalties.
- **Language-Aware Normalization**: Script-specific standardizations prior to metrics evaluations.
- **Real-time Translation**: Integrated `facebook/nllb-200-distilled-600M` model.
- **Parameter-Efficient Scaling**: LoRA adapter training loops constructed specifically for consumer hardware (MPS & constrained CUDA environments).
- **Omnilingual Dataset Methodology**: Specialized data handling strategies for training dense adapter matrices.

## Installation

1. Clone the repository.
2. Install the prerequisites:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure PyTorch is configured for your specific backend (`mps` for macOS, `cuda` for NVIDIA GPUs, or `cpu`).

## Running the Application

To boot the Flask server and local UI:

```bash
python app.py
```

The application will be accessible at `http://localhost:5005`.

## Repository Structure

- `app.py`: The Main Flask entry point containing the inference pipeline, routing logic, and Streamlit-equivalent APIs.
- `data_loader.py`: Dataset processing and dynamic logic that prepares our Omnilingual Dataset streams for deep learning mapped functions.
- `evaluate_asr.py`: Evaluation metrics script applying our phase 1 "free-wins" logic to compute WER & CER effectively.
- `finetune_lora.py`: The complex training script that leverages LoRA over the base transformer using gradient checkpointing.
- `msc_report_detailed.md`: A highly detailed report defining methodology, parameters, and findings.
- `templates/` & `static/`: Frontend code files.

## Training

To engage the Phase 2 Fine-Tuning sequence (ensure you have enough VRAM or Unified Memory):

```bash
python finetune_lora.py
```

## Metrics & Evaluation

Run the evaluation script to ascertain current base or PEFT abilities:

```bash
python evaluate_asr.py
```
