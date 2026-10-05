import argparse
import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    default_data_collator,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


def getenv_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': os.getenv('TRAIN_OUTPUT_DIR', f'{OUTPUT_DIR}/gpt2-1b-russian'),
    'optim': os.getenv('TRAIN_OPTIM', 'adamw_torch_fused'),
    'num_train_epochs': 1,
    'per_device_train_batch_size': int(os.getenv('TRAIN_BATCH_SIZE', '4')),
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': float(os.getenv('TRAIN_LR', '5e-5')),
    'weight_decay': 0.01,
    'warmup_steps': int(os.getenv('TRAIN_WARMUP_STEPS', '200')),
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'lr_scheduler_type': os.getenv('TRAIN_LR_SCHEDULER', 'linear'),
    'bf16': getenv_bool('TRAIN_BF16', True),
    'tf32': getenv_bool('TRAIN_TF32', True),
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': int(os.getenv('TRAIN_GRAD_ACCUM', '1')),
    'dataloader_num_workers': 4,
    'torch_compile': getenv_bool('TRAIN_TORCH_COMPILE', False),
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized[LABELS] = [ids.copy() for ids in tokenized[INPUT_IDS]]
    return {
        INPUT_IDS: tokenized[INPUT_IDS],
        ATTENTION_MASK: tokenized[ATTENTION_MASK],
        LABELS: tokenized[LABELS],
    }


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard_path = os.path.join(output_dir, f"{index:05d}.parquet")
        shard.to_parquet(shard_path)
        print(f"Saved shard {index + 1}/{num_shards}: {shard_path} ({len(shard)} rows)")


def prepare_dataset():
    """
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    tokenizer = prepare_tokenizer()
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing Wikipedia RU",
    )
    save_as_parquets(tokenized_dataset)


def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    - List only parquet files in data_dir, sorted by filename
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Tokenized data directory '{data_dir}' was not found. "
            "Run `python your_solution.py prepare` first."
        )

    parquet_files = sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in '{data_dir}'. "
            "Run `python your_solution.py prepare` first."
        )

    return load_dataset("parquet", data_files=parquet_files, split="train")


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    if dataset_size <= validation_size:
        raise ValueError(
            f"Dataset has {dataset_size} samples, but validation_size={validation_size}. "
            "Need more tokenized samples for the required train/test split."
        )

    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def train_model():
    """
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    model = create_model(tokenizer)

    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
        )
    trainer.train()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()
    trainer.save_model(TRAINING_CONFIG["output_dir"])
    tokenizer.save_pretrained(TRAINING_CONFIG["output_dir"])

    metrics_path = os.path.join(TRAINING_CONFIG["output_dir"], "final_eval_results.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(eval_results, fp, ensure_ascii=False, indent=2)

    prompt = "Москва является одним из крупнейших городов"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generation_path = os.path.join(TRAINING_CONFIG["output_dir"], "sample_generation.txt")
    with open(generation_path, "w", encoding="utf-8") as fp:
        fp.write(f"Prompt: {prompt}\n\n{generated_text}\n")
    print(f"Sample generation:\n{generated_text}")
    print(f"Saved final metrics to {metrics_path}")
    print(f"Saved sample generation to {generation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-pretrain Qwen3-1B on Russian Wikipedia.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=("prepare", "train", "all"),
        help="Use 'prepare' once to tokenize data, 'train' to train from parquet shards, or 'all' for both.",
    )
    args = parser.parse_args()

    if args.mode in ("prepare", "all"):
        prepare_dataset()
    if args.mode in ("train", "all"):
        train_model()
