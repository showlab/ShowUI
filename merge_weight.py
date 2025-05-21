import argparse
import os
import sys

import pdb
import json
import torch
from peft import LoraConfig, get_peft_model

from transformers import AutoProcessor

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    # arg_url
    parser.add_argument('--exp_dir', 
        type=str, 
        default="/blob/v-lqinghong/experiments/ShowUI-RL/showui2-1e-4/2025-02-04_23-09-35/")
    # Env
    # parser.add_argument(
    #     "--precision",
    #     default="bf16",
    #     type=str,
    #     choices=["fp32", "bf16", "fp16"],
    #     help="precision for inference",
    # )

    # # Model
    # parser.add_argument("--version", default="Qwen/Qwen2-VL-2B-Instruct")
    # parser.add_argument("--out_dim", default=256, type=int)
    # parser.add_argument("--model_max_length", default=4096, type=int)

    # # Lora
    # parser.add_argument("--lora_r", default=8, type=int)
    # parser.add_argument("--lora_alpha", default=16, type=int)
    # parser.add_argument("--lora_dropout", default=0.05, type=float)
    # parser.add_argument("--lora_target_modules", default="qkv_proj", type=str)

    # # Training and save
    # parser.add_argument("--weight", type=str, required=True)
    return parser.parse_args(args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=["self_attn", "lm_head"], verbose=True):
    linear_cls = torch.nn.modules.Linear
    lora_module_names = []
    # lora_namespan_exclude += ["vision_model", "img_projection", "visual_model"]
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def load_sharded_weights(weight_dir):
    """Load sharded weights"""
    print(f"Loading sharded weights from {weight_dir}")
    
    # Check if index file exists
    index_file = os.path.join(weight_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found at {index_file}")
        
    # Read index file
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    # Get weight mapping
    weight_map = index_data['weight_map']
        
    # Initialize state dict
    state_dict = {}
    # Load weights based on mapping
    for param_name, filename in weight_map.items():
        # Load shard containing this parameter
        shard_path = os.path.join(weight_dir, filename)
        print(f"Loading weight {param_name} from shard: {filename}")
        
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Weight shard not found at {shard_path}")
            
         # Load shard
        shard_state = torch.load(shard_path, map_location="cpu")
        
        # Extract parameter from shard
        if param_name in shard_state:
            state_dict[param_name] = shard_state[param_name]
        else:
            print(f"Warning: Parameter {param_name} not found in shard {filename}")
    
    return state_dict

def main(args):
    args = parse_args(args)
    json_url = os.path.join(args.exp_dir, 'args.json')
    with open(json_url, 'r') as f:
        json_args = json.load(f)
    for key, value in json_args.items():
        setattr(args, key, value)

    args.save_path = args.exp_dir + "/ckpt_model/merged_model"
    args.weight_url = args.exp_dir + "/ckpt_model/pytorch_model.bin"

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    if args.model_id in ['showlab/ShowUI-2B']:
        from model.showui.processing_showui import ShowUIProcessor

        model_url = args.model_id
        processor = ShowUIProcessor.from_pretrained(
                                                    "showlab/ShowUI-2B",
                                                    min_pixels=args.min_visual_tokens *28*28, 
                                                    max_pixels=args.max_visual_tokens *28*28,
                                                    model_max_length=args.model_max_length,
                                                    uigraph_train=args.uigraph_train, uigraph_test=args.uigraph_test,
                                                    uigraph_diff=args.uigraph_diff,  uigraph_rand=args.uigraph_rand,
                                                    uimask_pre=args.uimask_pre, uimask_ratio=args.uimask_ratio, uimask_rand=args.uimask_rand
                                                    )

        from model.utils import parse_layer_type
        from model.showui.modeling_showui import ShowUIForConditionalGeneration

        lm_qwen_layer = 28
        vis_qwen_layer = 32
        lm_skip_layer = parse_layer_type(args.lm_skip_layer, lm_qwen_layer)
        vis_skip_layer = parse_layer_type(args.vis_skip_layer, vis_qwen_layer)

        model = ShowUIForConditionalGeneration.from_pretrained(
            model_url,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            # quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
            lm_skip_layer=lm_skip_layer,
            lm_skip_ratio=args.lm_skip_ratio,
        )
    elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
        from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        model_id = args.model_id.replace("Qwen/", "")
        model_url = args.model_id
        processor = Qwen2VLProcessor.from_pretrained(
                                                    model_url,
                                                    min_pixels=args.min_visual_tokens *28*28, 
                                                    max_pixels=args.max_visual_tokens *28*28,
                                                    model_max_length=args.model_max_length,
                                                   )

        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_url,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            # quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
        )
    model.config.use_cache = False
    model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=["visual"])
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load sharded weights
        try:
            print(f"Loading sharded weights from {args.weight_url}")
            state_dict = load_sharded_weights(args.weight_url)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise

    # Merge LoRA weights
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

        # Save merged model
        print(f"Saving merged model to {args.save_path}")
        model.save_pretrained(
            args.save_path,
            max_shard_size="10GB",  # Can be adjusted based on requirements
            safe_serialization=True
        )
        processor.save_pretrained(args.save_path)




if __name__ == "__main__":
    main(sys.argv[1:])