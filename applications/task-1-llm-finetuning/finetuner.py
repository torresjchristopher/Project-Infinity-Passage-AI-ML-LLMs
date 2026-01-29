"""
LLM Fine-Tuning Framework
Production-grade framework for fine-tuning large language models
Supports LoRA, QLoRA, and full model fine-tuning with distributed training
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, QLoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning"""
    model_name: str = "gpt2"
    method: str = "lora"  # lora, qlora, full
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    
    # QLoRA config (quantization)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    
    # Training
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    use_flash_attention: bool = True
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"


class CustomDataset(Dataset):
    """Custom dataset for fine-tuning"""
    
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[str]:
        """Load data from JSON or JSONL"""
        data = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        text = item.get('text', '') or item.get('content', '')
                    else:
                        text = str(item)
                    if text:
                        data.append(text)
        
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            text = item.get('text', '') or item.get('content', '')
                        else:
                            text = str(item)
                        if text:
                            data.append(text)
        
        elif data_path.endswith('.txt'):
            with open(data_path, 'r') as f:
                content = f.read()
                data = content.split('\n\n')
        
        return [d for d in data if d.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class LLMFineTuner:
    """Main fine-tuning framework"""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load model with quantization if specified"""
        print(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization for QLoRA
        bnb_config = None
        if self.config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Prepare for training
        if self.config.method == "qlora":
            self.model = prepare_model_for_kbit_training(self.model)
        
        print(f"Model loaded successfully")
        print(f"Model size: {self.model.get_memory_footprint() / 1e9:.2f} GB")
    
    def setup_lora(self):
        """Setup LoRA adapters"""
        print("Setting up LoRA adapters...")
        
        if self.config.method == "qlora":
            lora_config = QLoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.lora_target_modules,
            )
        else:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.lora_target_modules,
            )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total params: {total_params:,}")
    
    def train(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Fine-tune the model"""
        print("Preparing training data...")
        
        # Load model
        self.load_model()
        
        # Setup LoRA if specified
        if self.config.method in ["lora", "qlora"]:
            self.setup_lora()
        
        # Load datasets
        train_dataset = CustomDataset(
            train_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        eval_dataset = None
        if eval_data_path:
            eval_dataset = CustomDataset(
                eval_data_path,
                self.tokenizer,
                self.config.max_seq_length
            )
        
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Eval samples: {len(eval_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.eval_strategy,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.mixed_precision == "fp16",
            bf16=self.config.mixed_precision == "bf16",
            remove_unused_columns=False,
            logging_steps=10,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Training complete!")
        return trainer
    
    def evaluate(self, eval_data_path: str) -> Dict[str, float]:
        """Evaluate the model"""
        print("Evaluating model...")
        
        eval_dataset = CustomDataset(
            eval_data_path,
            self.tokenizer,
            self.config.max_seq_length
        )
        
        data_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity.item(),
        }
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        self.model.eval()
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    
    def save_model(self, output_path: str):
        """Save the fine-tuned model"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        if self.config.method in ["lora", "qlora"]:
            self.model.save_pretrained(output_path)
            print(f"LoRA adapters saved to {output_path}")
        else:
            self.model.save_pretrained(output_path)
            print(f"Model saved to {output_path}")
        
        self.tokenizer.save_pretrained(output_path)
        print(f"Tokenizer saved to {output_path}")
    
    def export_model(self, output_path: str, format: str = "onnx"):
        """Export model to different formats"""
        print(f"Exporting model to {format}...")
        
        if format == "onnx":
            # Export to ONNX requires additional setup
            print("ONNX export requires additional dependencies")
            print("Install: pip install optimum onnx onnxruntime")
        
        elif format == "torchscript":
            traced_model = torch.jit.trace(self.model, self.model.dummy_inputs)
            traced_model.save(output_path)
            print(f"Model exported to {output_path}")


def main():
    """Example usage"""
    config = FineTuneConfig(
        model_name="gpt2",
        method="lora",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
    )
    
    tuner = LLMFineTuner(config)
    
    # Train
    # tuner.train("train_data.jsonl", "eval_data.jsonl")
    
    # Evaluate
    # metrics = tuner.evaluate("eval_data.jsonl")
    # print(f"Evaluation metrics: {metrics}")
    
    # Generate
    # text = tuner.generate("Once upon a time")
    # print(f"Generated text: {text}")
    
    # Save
    # tuner.save_model("./fine_tuned_model")


if __name__ == "__main__":
    main()
