import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor

def prepare_dataset(batch, processor):
    """
    Mapping function applied to the batched datasets sequentially.
    Extracts raw audio structures and samples to construct log-Mel input features 
    required by the Whisper generation framework. Additionally, maps per-sample properties 
    to explicit tokenizer variables, keeping the decoder informed.
    
    Args:
        batch (dict): HuggingFace dataset payload slice.
        processor (WhisperProcessor): The huggingface tokenizer / feature extraction pipeline.
        
    Returns:
        dict: The tensor-mapped batch ready for the data collator.
    """
    # Load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    lang = batch["language_name"]

    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Explicitly set per-sample language tracking to prevent tokenisation collision
    processor.tokenizer.set_prefix_tokens(language=lang.capitalize(), task="transcribe")
    
    # Use raw_transcription field instead of transcription for cleaner ground truth
    text = batch["raw_transcription"]
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

def get_multilingual_dataset(languages, split="train", num_samples=None):
    """
    Constructs the foundational 'Omnilingual Dataset' for Phase 2 Parameter-Efficient Fine-Tuning.
    
    To overcome extreme sparsity in low-resource Indic languages, this methodology dynamically constructs 
    the "Omnilingual Dataset." By explicitly pooling ultra-diverse topological phonetic structures—such as 
    French and German anchoring alongside Marathi and Bengali—this curated dataset acts as a universal 
    cross-lingual nexus. This allows the newly initialized LoRA parameters to learn high-dimensional 
    extrapolative weights rather than collapsing into catastrophic forgetting mode. 
    
    Note: Function internally maps standard language strings to FLEURS ISO dataset keys for ease of use.
    
    Args:
        languages (list): String definitions of target languages.
        split (str): 'train' or 'test' split fetching.
        num_samples (int): Ceiling limit on fetch size per language.
        
    Returns:
        Dataset: High-density concatenated 'Omnilingual Dataset' ready for PEFT processing.
    """
    datasets = []
    
    # ISO language codes mapping for FLEURS dataset keys 
    fleurs_map = {
        "hindi": "hi_in",
        "marathi": "mr_in",
        "bengali": "bn_in",
        "urdu": "ur_pk",
        "english": "en_us",
        "french": "fr_fr",
        "german": "de_de"
    }

    for lang in languages:
        code = fleurs_map.get(lang.lower(), lang)
        print(f"Loading {lang} ({code})...")
        ds = load_dataset("google/fleurs", code, split=split, trust_remote_code=True)
        
        if num_samples:
            limit = min(num_samples, len(ds))
            ds = ds.select(range(limit))
            
        # Add a column to keep track of the language for dynamic tokenization mapping
        ds = ds.add_column("language_name", [lang] * len(ds))
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    return combined

if __name__ == "__main__":
    Processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    langs = ["hindi", "marathi", "bengali", "urdu", "english", "french", "german"]
    
    # Demonstration of the lazy mapping 
    ds = get_multilingual_dataset(langs, split="train", num_samples=2)
    print(f"Loaded {len(ds)} samples across {len(langs)} languages.")
