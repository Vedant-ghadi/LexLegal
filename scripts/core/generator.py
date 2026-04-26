import os
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

class T5Generator:
    """Wrapper for the QLoRA fine-tuned Flan-T5 model."""
    def __init__(self, adapter_path: str = None, device: str = 'cuda:0'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        
        if adapter_path and os.path.exists(adapter_path):
            self._load_model(adapter_path)
        else:
            print(f"Warning: No valid adapter found at {adapter_path}. Proposition Extraction and HyDE will be disabled.")
            
    def _load_model(self, adapter_path):
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        print(f"Loading Flan-T5-Large generator with LoRA adapter from {adapter_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large', legacy=False)
        
        if self.device != 'cpu':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = T5ForConditionalGeneration.from_pretrained(
                'google/flan-t5-large', 
                quantization_config=bnb_config,
                device_map={'': self.device}
            )
        else:
            base_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
            
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("Generator loaded successfully.")

    def ft_gen(self, prompt: str, ml: int = 256) -> str:
        """Runs the fine-tuned generation."""
        if not self.model or not self.tokenizer:
            return "" # Fallback stub if models aren't present
            
        ids = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids, 
                max_new_tokens=ml, 
                num_beams=3, 
                early_stopping=True
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
