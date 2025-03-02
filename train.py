import argparse
import re
import os
import shutil
import sys
import pdb
import time
import json
import wandb
from functools import partial
from datetime import datetime

import deepspeed
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoProcessor, BitsAndBytesConfig      # do not remove this line
from main.trainer import train
from main.eval_aitw import validate_aitw
from main.eval_mind2web import validate_mind2web
from main.eval_screenspot import validate_screenspot
from main.evaluator import validate as validate_default

from model.utils import find_target_linear_names
from data.dataset import HybridDataset, collate_fn
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from utils.utils import save_args_to_json, create_log_dir

def env_init(distributed=True):
    print("Init Env for Distributed Training")
    if distributed:
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", 'localhost')
            os.environ['MASTER_PORT']  = os.environ.get("MASTER_PORT", "12875")
            os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
            os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
            os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

            print(f"OMPI_COMM_WORLD_SIZE: {os.environ['OMPI_COMM_WORLD_SIZE']}")
            print(f"OMPI_COMM_WORLD_RANK: {os.environ['OMPI_COMM_WORLD_RANK']}")
            print(f"OMPI_COMM_WORLD_LOCAL_RANK: {os.environ['OMPI_COMM_WORLD_LOCAL_RANK']}")
            print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
            print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        elif 'WORLD_SIZE' in os.environ:
            os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", 'localhost')
            os.environ['MASTER_PORT']  = os.environ.get("MASTER_PORT", "12875")

            print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
            print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
        else:
            return
    else:
        return

# a tricky way to broadcast timestamp to all ranks
def broadcast_timestamp(src=0, local_rank=0):
    if dist.get_rank() == src:
        timestamp = torch.tensor([datetime.now().timestamp()], dtype=torch.float64).to(f'cuda:{local_rank}')
    else:
        timestamp = torch.zeros(1, dtype=torch.float64).to(f'cuda:{local_rank}')

    dist.broadcast(timestamp, src=src)
    time_str = datetime.fromtimestamp(timestamp.item()).strftime('%Y-%m-%d_%H-%M-%S')
    return time_str

def parse_args(args):
    parser = argparse.ArgumentParser(description="ShowUI Training Pipeline")
    # Environment
    parser.add_argument("--wandb_key", default=None, type=str, help="wandb key to monitor training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--ds_zero", choices=['zero1', 'zero2', 'zero3'], default='zero2', help="deepspeed zero stage")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--attn_imple", choices=["eager", "flash_attention_2", "sdpa"], default="eager")
    parser.add_argument("--liger_kernel", action="store_true", default=False)

    # Model & Ckpt
    parser.add_argument("--model_id", default="showlab/ShowUI-2B", choices=["showlab/ShowUI-2B", "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", \
                                                                            "Qwen/Qwen2.5-VL-3B-Instruct"])
    parser.add_argument("--version", default="showlab/ShowUI-2B")
    parser.add_argument("--max_new_tokens", default=128, type=int, help="max. generated token length")
    parser.add_argument("--local_weight", action="store_true", default=False)
    parser.add_argument("--local_weight_dir",  default=".", help="default path to load the model weight")
    # Visual Encoder Training strategy
    parser.add_argument("--tune_visual_encoder", action="store_true", default=False)
    parser.add_argument("--tune_visual_encoder_projector", action="store_true", default=False)
    parser.add_argument("--freeze_lm_embed", action="store_true", default=False)

    # Training / Validation Data
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--train_dataset", default="showui", type=str)
    parser.add_argument("--train_json", default="hf_train", type=str)
    parser.add_argument("--train_ratio", default="1", type=str)
    parser.add_argument("--val_dataset", default="screenspot", type=str)
    parser.add_argument("--val_json", default="hf_test_full", type=str)
    parser.add_argument("--val_ratio", default="1", type=str)
    parser.add_argument("--uniform_sample", action="store_true", default=False)
    parser.add_argument("--random_sample", action="store_true", default=False)
    parser.add_argument("--record_sample", action="store_true", default=False)
    
    ### ShowUI Preprocessor
    # 0. Common setups
    parser.add_argument("--min_visual_tokens", default=256, type=int)
    parser.add_argument("--max_visual_tokens", default=1280, type=int)
    parser.add_argument("--model_max_length", default=8192, type=int)
    # 1. Screenshot -> Graph
    parser.add_argument("--uigraph_train", action="store_false", default=True, help="Enable ui graph during training")
    parser.add_argument("--uigraph_test", action="store_true", default=False, help="Enable ui graph during inference")
    parser.add_argument("--uigraph_diff", default=1, type=int, help="Pixel difference used for constructing ui graph")
    parser.add_argument("--uigraph_rand", action="store_true", default=False, help="Enable random graph construction")
    # 2. Graph -> Mask 
    parser.add_argument("--uimask_pre", action="store_false", default=True, help="Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency")
    parser.add_argument("--uimask_ratio", default=0.5, type=float, help="Specify the percentage of patch tokens to skip per component")
    parser.add_argument("--uimask_rand", action="store_true", default=False, help="Enable random token selection instead of uniform selection")
    ### ShowUI Model
    # 0 is without layer token selection, 1 is with layer token selection. Below we provide examples:
    # [1,28,1] means that all LM layers use token selection; [1,28,0] means that do not.
    # Interleaved layer-wise '[2,2,1],[4,4,1],[6,6,1],[8,8,1],[10,10,1],[12,12,1],[14,14,1],[16,16,1],[18,18,1],[20,20,1],[22,22,1],[24,24,1],[26,26,1]'
    parser.add_argument("--lm_skip_ratio", default=0, type=float)
    parser.add_argument("--lm_skip_layer", default='[1,28,0]', type=str)
    parser.add_argument("--vis_skip_ratio", default=0, type=float)
    parser.add_argument("--vis_skip_layer", default='[1,32,0]', type=str)
    # Pretrain / Supervised Fine-tuning
    parser.add_argument("--showui_data", default="hf_train", type=str)
    parser.add_argument("--amex_data", default="hf_train", type=str)
    parser.add_argument("--guiact_data", default="hf_train_web-single_v2", type=str)
    parser.add_argument("--ricosca_data", default="hf_train_ricosca", type=str)
    parser.add_argument("--widget_data", default="hf_train_widget", type=str)
    parser.add_argument("--screencap_data", default="hf_train_screencap", type=str)
    # Downstream train. set
    parser.add_argument("--aitw_data", default="hf_train", type=str)
    parser.add_argument("--mind2web_data", default="hf_train", type=str)
    parser.add_argument("--miniwob_data", default="hf_train", type=str)
    # Downstream val. set
    parser.add_argument("--val_aitw_data", default="hf_test", type=str)
    parser.add_argument("--val_mind2web_data", default="hf_test_full", type=str)
    parser.add_argument("--val_screenspot_data", default="hf_test_full", type=str)

    # Grounding setting
    parser.add_argument("--num_turn", default=1, type=int, help="Interleaved Query-Action setting")
    parser.add_argument("--shuffle_image_token", action="store_true", default=False, help="shuffle image token for training")
    parser.add_argument("--uniform_prompt", action="store_true", default=False)
    parser.add_argument("--text2point", default=1, type=float)
    parser.add_argument("--text2bbox", default=0, type=float)
    parser.add_argument("--point2text", default=0, type=float)
    parser.add_argument("--bbox2text", default=0, type=float)
    parser.add_argument("--crop_min", default=1, type=float)
    parser.add_argument("--crop_max", default=1, type=float)
    parser.add_argument("--xy_int", action="store_true", default=False)

    # Navigation setting
    parser.add_argument("--num_history", default=4, type=int)
    parser.add_argument("--interleaved_history", default='tttt',  choices=['tttt', 'vvvv', 'vtvt', 'tvtv', 'vvtt', 'ttvv'], help="Interleaved Vision-Action setting")
    parser.add_argument("--skip_readme_train", action="store_true", default=False)
    parser.add_argument("--skip_readme_test", action="store_true", default=False)

    # Lora
    parser.add_argument("--use_qlora", action="store_true", default=False)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="qkv_proj", type=str)

    # Training
    parser.add_argument("--log_base_dir", default="../runs", type=str)
    parser.add_argument("--exp_id", default="debug", type=str)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--warmup_type", default="linear", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    # Model Checkpoint or Evaluation strategies
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--debug", action="store_true", default=False, help="for debugging, will not save model and monitor")
    return parser.parse_args(args)

def main(args):
    print("\033[34m##########################################################\033[0m")
    print("\033[34m############ 💻 Building GUI Agents with ShowUI ##########\033[0m")
    print("\033[34m##########################################################\033[0m")

    env_init()
    args = parse_args(args)
    args.global_rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.attn_imple in ["eager", "sdpa"]:
        # suggested by https://github.com/Lightning-AI/litgpt/issues/327
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if args.global_rank == 0 else None

    args.distributed = args.world_size > 1
    # ensure all rank share the same timestamp
    if args.distributed:
        print(f"Using distributed training with {args.world_size} GPUs, with rank {os.environ['RANK']}")
        deepspeed.init_distributed(dist_backend="nccl", rank=args.global_rank, world_size=args.world_size)
        timestamp = broadcast_timestamp(0, args.local_rank)

    args.log_dir = os.path.join(args.log_base_dir, args.exp_id, timestamp)
    args.tmp_dir = os.path.join(args.log_dir, "tmp")

    # must provide wandb-key
    assert args.wandb_key is not None
    wandb.login(key=args.wandb_key)

    writer = None
    if args.global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.tmp_dir, exist_ok=True)
        save_args_to_json(args, os.path.join(args.log_dir, "args.json"))        # save args to json
        if not args.debug:
            writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))       # init. tensorboard writer
            # init. wandb monitor
            wandb.init(
                project="ShowUI",
                group=args.exp_id,
                name=f'{args.exp_id}_{timestamp}',
                dir=args.log_dir,
                config=args
            )
    print(f"Start Job: {args.exp_id}")

    # Create processor
    if args.model_id in ["showlab/ShowUI-2B"]:
        from model.showui.processing_showui import ShowUIProcessor
        model_id = args.model_id
        if args.local_weight:
            model_url = f"{args.local_weight_dir}/{model_id}"
        else:
            model_url = args.model_id
        processor = ShowUIProcessor.from_pretrained(
                                                    "Qwen/Qwen2-VL-2B-Instruct",
                                                    min_pixels=args.min_visual_tokens *28*28, 
                                                    max_pixels=args.max_visual_tokens *28*28,
                                                    model_max_length=args.model_max_length,
                                                    uigraph_train=args.uigraph_train, uigraph_test=args.uigraph_test,
                                                    uigraph_diff=args.uigraph_diff,  uigraph_rand=args.uigraph_rand,
                                                    uimask_pre=args.uimask_pre, uimask_ratio=args.uimask_ratio, uimask_rand=args.uimask_rand
                                                    )
    elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
        from model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor  # AutoProcessor
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        model_id = args.model_id.replace("Qwen/", "")
        if args.local_weight:
            model_url = f"{args.local_weight_dir}/{model_id}"
        else:
            model_url = args.model_id
        processor = Qwen2VLProcessor.from_pretrained(
                                                    "Qwen/Qwen2-VL-2B-Instruct",
                                                    min_pixels=args.min_visual_tokens *28*28, 
                                                    max_pixels=args.max_visual_tokens *28*28,
                                                    model_max_length=args.model_max_length,
                                                   )
    elif args.model_id in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
        from model.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        model_id = args.model_id.replace("Qwen/", "")
        if args.local_weight:
            model_url = f"{args.local_weight_dir}/{model_id}"
        else:
            model_url = args.model_id
        processor = Qwen2_5_VLProcessor.from_pretrained(
                                                        model_url,
                                                        min_pixels=args.min_visual_tokens *28*28, 
                                                        max_pixels=args.max_visual_tokens *28*28,
                                                        model_max_length=args.model_max_length,
                                                        )
        
    CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    processor.chat_template = CHAT_TEMPLATE
    processor.tokenizer.chat_template = CHAT_TEMPLATE

    # Create model
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["img_projection"],
                ) if args.use_qlora else None

    # Create model
    if args.local_weight:
        model_id = args.model_id.replace("Qwen/", "")
        model_url = f"{args.local_weight_dir}/{model_id}"
    else:
        model_url = args.model_id
        
    if args.model_id in ["showlab/ShowUI-2B"]:
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
            quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
            lm_skip_layer=lm_skip_layer,
            lm_skip_ratio=args.lm_skip_ratio,
        )
    elif args.model_id in ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"]:
        from model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_url,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
        )
    elif args.model_id in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
        from model.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_url,
            low_cpu_mem_usage=True,
            _attn_implementation=args.attn_imple,
            quantization_config=bnb_config,
            device_map=f"cuda:{args.local_rank}",
        )

    # load model checkpoint
    if args.version != args.model_id:
        state_dict = torch.load(args.version, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    model.config.use_cache = False
    
    if args.liger_kernel:
        # https://github.com/linkedin/Liger-Kernel
        print("Apply liger kernel to ShowUI for efficiency")
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
        apply_liger_kernel_to_qwen2_vl()

    # During evaluation mode, no need to load lora
    if args.eval_only:
        print("evaluation mode, thus set the `lora_r' as zero.")
        args.lora_r = 0
    if not args.eval_only and args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # Config lora using peft library
    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        exclude_module = ["visual"] if not args.tune_visual_encoder else []
        exclude_module += ["lm_head"] if args.freeze_lm_embed else exclude_module
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=exclude_module)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if args.global_rank == 0:
            model.print_trainable_parameters()

        model_child = model.model.model
    else:
        model_child = model.model

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if not args.tune_visual_encoder:
        if args.lora_r > 0:
            for p in model.base_model.model.visual.parameters():
                p.requires_grad = False
        elif args.lora_r == 0:
            for p in model.visual.parameters():
                p.requires_grad = False
            
    if args.tune_visual_encoder_projector:
        for k, p in model.named_parameters():
            if 'visual.merger' in k:
                p.requires_grad = True

    if args.freeze_lm_embed:
        if args.lora_r > 0:
            for p in model_child.embed_tokens.parameters():
                p.requires_grad = False
        elif args.lora_r == 0:
            for p in model_child.embed_tokens.parameters():
                p.requires_grad = False

    # Check trainable parameters
    list_of_params_to_optimize = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if args.global_rank == 0:
                print("[Name]", n, " [Shape]", p.shape)
            list_of_params_to_optimize.append(p)

    # Create dataset
    args.samples_per_epoch = args.batch_size    \
                    * args.grad_accumulation_steps  \
                    * args.steps_per_epoch  \
                    * args.world_size

    train_dataset = HybridDataset(
        processor,
        inference=False,
        args=args
    )
    val_dataset = HybridDataset(
        processor,
        inference=True,
        args=args
    )

    if args.val_dataset == "mind2web":
        validate = validate_mind2web
    elif args.val_dataset == "screenspot":
        validate = validate_screenspot
    elif args.val_dataset == "aitw":
        validate = validate_aitw
    else:
        validate = validate_default

    if not args.random_sample:
        args.steps_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)

    # Build deepspeed config and initialize deepspeed
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "warmup_type": args.warmup_type,
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        }
    }

    config_url = f'ds_configs/{args.ds_zero}.json'
    with open(config_url, 'r') as file:
        ds_json = json.load(file)
    ds_config.update(ds_json)
    
    # lora tuning
    if lora_r > 0:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=list_of_params_to_optimize,
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
            config=ds_config,
        )
    # full tunning
    elif lora_r == 0 and not args.eval_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=list_of_params_to_optimize,
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
            config=ds_config,
        )
    # evaluation
    elif args.eval_only:
        for param in model.parameters():
            param.requires_grad = False 
        model_engine = model
    else:
        raise ValueError("Invalid setting")

    # Resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        if args.global_rank == 0:
            print(
                "resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                processor=processor
            ),
        )
    else:
        val_loader = None

    if args.eval_only:
        local_rank = args.local_rank
        model_engine = model_engine.to(f'cuda:{local_rank}')
        validate(val_loader, model_engine, processor, 0, 0, writer, args)
        exit()

    train_iter = iter(train_loader)
    best_score = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter, global_step = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False and val_loader is not None:
            score = validate(val_loader, model_engine, processor, epoch, global_step, writer, args)
            is_best = score > best_score
            best_score = max(score, best_score)
        else:
            is_best = True
            best_score = 0

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        save_dir,
                        "meta_log_epo{:.0f}_score{:.2f}.pth".format(
                            epoch, best_score
                        ),
                    ),
                )
            torch.distributed.barrier()
            try:
                model_engine.save_checkpoint(save_dir)
            except Exception as e:
                print("Failed to save checkpoint (): ", e)
    
    if args.global_rank == 0:
        if not args.debug:
            wandb.finish()
            writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
