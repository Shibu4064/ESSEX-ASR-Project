import os
import re
import torch
import librosa
import tempfile
import jiwer
from flask import Flask, render_template, request, jsonify
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# --- Model Loading Logic ---

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
}

def load_pipe():
    """
    Initializes the HuggingFace ASR pipeline. 
    It checks for valid hardware backends (MPS for Mac Apple Silicon, CUDA for Nvidia, fallback CPU) 
    and determines if local LoRA (Parameter-Efficient) weights exist. If they do, it structurally 
    merges the low-rank matrices into the foundation model for rapid, zero-overhead inference.
    
    Returns:
        Pipeline: The instantiated ASR pipeline ready for generation.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if device == "mps" else (torch.float16 if torch.cuda.is_available() else torch.float32)
    base_model_id = "openai/whisper-large-v3-turbo"
    peft_model_id = "./whisper-large-v3-turbo-lora"
    
    if os.path.exists(peft_model_id):
        print(f"LoRA weights found at {peft_model_id}. Loading PEFT Model...")
        processor = WhisperProcessor.from_pretrained(base_model_id, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(model, peft_model_id)
        model = model.merge_and_unload()
        model.to(device)
        model.to(torch_dtype)
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=model, 
            tokenizer=processor.tokenizer, 
            feature_extractor=processor.feature_extractor, 
            device=device, 
            torch_dtype=torch_dtype, 
            chunk_length_s=30
        )
    else:
        print("No LoRA weights found. Loading base Large-v3-Turbo...")
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=base_model_id, 
            device=device, 
            torch_dtype=torch_dtype, 
            chunk_length_s=30
        )
    return pipe

# --- Translation Logic ---

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_LANG_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
    "Urdu": "urd_Arab",
    "French": "fra_Latn",
    "German": "deu_Latn",
}

def load_translator():
    """
    Initializes the Meta NLLB (No Language Left Behind) sequence-to-sequence translation model.
    Used for on-the-fly cross-lingual translation of the transcribed ASR text payloads.
    
    Returns:
        dict: A dictionary containing the loaded model, tokenizer, and active hardware device.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32 if device == "mps" else (torch.float16 if torch.cuda.is_available() else torch.float32)
    print(f"Loading translation model: {NLLB_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID, torch_dtype=torch_dtype).to(device)
    return {"model": model, "tokenizer": tokenizer, "device": device}

# Initialize models
transcription_pipe = load_pipe()
translator_pipe = None # Lazy load if needed to save memory initially or load now

# --- Routes ---

@app.route('/')
def index():
    """Serves the primary Single Page Application (SPA) frontend."""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Main API endpoint for processing audio payloads.
    It expects an audio file from the frontend, securely writes it to a temporary environment, 
    processes it via librosa, executes the ASR generation kwargs (with Deep Beam Search enabled), 
    and handles optional downstream tasks like Jiwer metrics scoring (WER/CER) and NLLB Translation.
    
    Returns:
        JSON Response: containing transcriptions, metrics, translation payloads, and metadata.
    """
    global translator_pipe
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'Auto-detect')
    target_lang = request.form.get('target_language', 'None')
    reference_text = request.form.get('reference_text', '').strip()
    
    # Advanced Params
    beam_size = int(request.form.get('beam_size', 5))
    temperature = float(request.form.get('temperature', 0.0))
    # Note: VAD and other low-level params might require switching from channel pipeline to model.generate
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio_array, sr = librosa.load(tmp_path, sr=16000)
        duration = librosa.get_duration(y=audio_array, sr=sr)
        
        # Transcription Logic
        lang_code = LANGUAGES.get(language)
        generate_kwargs = {
            "task": "transcribe", 
            "num_beams": beam_size, 
            "no_repeat_ngram_size": 3,
            "temperature": temperature,
        }
        
        if lang_code:
            generate_kwargs["language"] = lang_code

        # Use return_timestamps if requested
        return_timestamps = request.form.get('timestamps', 'segment')
        
        result = transcription_pipe(
            {"array": audio_array, "sampling_rate": sr}, 
            batch_size=8, 
            generate_kwargs=generate_kwargs, 
            return_timestamps=(return_timestamps == 'word' or return_timestamps == 'segment')
        )
        
        transcription = result["text"].strip()
        chunks = result.get("chunks", [])

        # Auto-detect language if not provided
        # The pipeline doesn't return detected language easily, but we can try to guess for NLLB
        detected_lang_name = language
        if language == "Auto-detect":
            # Simple heuristic: if we have Indic characters, try to pick an Indic src for NLLB
            # In a more robust setup, we'd use langdetect or look at model tokens
            if re.search(r'[\u0900-\u097F]', transcription): # Devanagari (Hindi/Marathi)
                 detected_lang_name = "Hindi" # Fallback guess for NLLB
            elif re.search(r'[\u0980-\u09FF]', transcription): # Bengali
                 detected_lang_name = "Bengali"
            else:
                 detected_lang_name = "English"

        # Translation
        translation = None
        if target_lang != "None" and target_lang in NLLB_LANG_MAP:
            if translator_pipe is None:
                translator_pipe = load_translator()
            
            tgt_lang_code = NLLB_LANG_MAP[target_lang]
            src_lang_code = NLLB_LANG_MAP.get(detected_lang_name, "eng_Latn")
            
            try:
                tokenizer = translator_pipe["tokenizer"]
                model = translator_pipe["model"]
                device = translator_pipe["device"]

                tokenizer.src_lang = src_lang_code
                inputs = tokenizer(transcription, return_tensors="pt").to(device)
                
                translated_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code], 
                    max_length=448
                )
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except Exception as te:
                print(f"Translation Error: {str(te)}")
                translation = f"[Translation Error: {str(te)}]"

        # Metrics
        wer, cer = None, None
        if reference_text:
            ref = re.sub(r'\s+', ' ', reference_text).strip()
            pred = re.sub(r'\s+', ' ', transcription).strip()
            wer = jiwer.wer(ref, pred) * 100
            cer = jiwer.cer(ref, pred) * 100

        # Fake confidence for now or try to extract from model if possible
        # Real confidence requires model.generate(return_dict_in_generate=True, output_scores=True)
        confidence = 0.92  # Placeholder until low-level transition

        return jsonify({
            "transcription": transcription,
            "translation": translation,
            "wer": round(wer, 2) if wer is not None else None,
            "cer": round(cer, 2) if cer is not None else None,
            "duration": round(duration, 2),
            "confidence": confidence,
            "chunks": chunks
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
