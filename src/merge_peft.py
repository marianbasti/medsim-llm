import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Merge a PEFT adapter into a base model.")
parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model (e.g., HuggingFaceTB/SmolLM2-360M)")
parser.add_argument("--peft_model", type=str, required=True, help="Path to the PEFT adapter checkpoint directory")
parser.add_argument("--output_dir", type=str, default="/media/marian/Kingston-2Tb/medsim-llm/models/merged_model", help="Directory to save the merged model and tokenizer")
args = parser.parse_args()

# --- Configuration ---
base_model_name = args.base_model
peft_model_path = args.peft_model
merged_model_save_path = args.output_dir # Where to save the merged model

# --- Load Base Model and Original Tokenizer ---
print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
print(f"Loading original tokenizer: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- Add New Tokens from Checkpoint ---
added_tokens_path = os.path.join(peft_model_path, "special_tokens_map.json")
num_new_tokens = 0
if os.path.exists(added_tokens_path):
    print(f"Found added tokens file: {added_tokens_path}")
    with open(added_tokens_path, "r") as f:
        added_tokens_dict = json.load(f)

        tokens_to_add = []
        # Extract from additional_special_tokens list
        if "additional_special_tokens" in added_tokens_dict and isinstance(added_tokens_dict["additional_special_tokens"], list):
            for token_info in added_tokens_dict["additional_special_tokens"]:
                if isinstance(token_info, dict) and "content" in token_info:
                    tokens_to_add.append(token_info["content"])
                elif isinstance(token_info, str): # Handle case where it might just be a list of strings
                     tokens_to_add.append(token_info)

        # Handle other special tokens (bos, eos, pad) if defined with content
        for token_key in ["bos_token", "eos_token", "pad_token"]:
             if token_key in added_tokens_dict:
                 token_val = added_tokens_dict[token_key]
                 if isinstance(token_val, dict) and "content" in token_val:
                     tokens_to_add.append(token_val["content"])
                 elif isinstance(token_val, str): # Handle case where it's just a string
                     tokens_to_add.append(token_val)

        # Remove duplicates before adding
        unique_tokens_to_add = list(set(tokens_to_add))
        if unique_tokens_to_add:
            print(f"Attempting to add tokens: {unique_tokens_to_add}")
            num_new_tokens = tokenizer.add_tokens(unique_tokens_to_add)
            print(f"Added {num_new_tokens} new token(s) to the tokenizer.")
        else:
            print("No valid tokens found to add from special_tokens_map.json")

else:
    print(f"No added tokens file found at {added_tokens_path}. Assuming vocab size didn't change or tokens are already in base.")

# --- Resize Model Embeddings if Tokens Were Added ---
if num_new_tokens > 0:
    print(f"Resizing model token embeddings to {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))
    # Ensure the model's config also reflects the new vocab size if needed
    base_model.config.vocab_size = len(tokenizer)
else:
     # Check if vocab size in checkpoint config differs anyway (might indicate direct vocab modification)
    adapter_config_path = os.path.join(peft_model_path, "adapter_config.json")
    try:
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
            # You might need to inspect adapter_config or other files in the checkpoint
            # to determine the *intended* final vocab size if added_tokens.json is missing.
            # For now, we assume the tokenizer length is the source of truth.
            print(f"Model vocab size will remain {base_model.config.vocab_size}")
    except FileNotFoundError:
        print(f"Adapter config not found at {adapter_config_path}")


# --- Load PEFT Adapter ---
# Now load the adapter state dict. The tokenizer mismatch should be resolved.
print(f"Loading PEFT adapter from: {peft_model_path}")
# Note: Pass is_trainable=False if you only intend to merge and infer.
# Set is_trainable=True if you plan further training BEFORE merging.
peft_model = PeftModel.from_pretrained(base_model, peft_model_path, is_trainable=False)
print("PEFT adapter loaded successfully.")

# --- Merge and Unload ---
print("Merging adapter weights into the base model...")
merged_model = peft_model.merge_and_unload()
print("Merging complete.")

# --- Save Merged Model and Tokenizer ---
print(f"Saving merged model to: {merged_model_save_path}")
merged_model.save_pretrained(merged_model_save_path)
print(f"Saving tokenizer to: {merged_model_save_path}")
tokenizer.save_pretrained(merged_model_save_path)

print("Merged model and tokenizer saved successfully!")