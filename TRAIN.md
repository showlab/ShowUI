# 🚀ShowUI Training Instruction
## 🔧Install Environment

```
conda create -n showui python=3.10
conda activate showui
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --user
pip install -r requirements.txt --user
```

## 📦Setup Datasets
### Grounding datasets
- Download grounding training dataset -- [ShowUI-desktop](https://huggingface.co/datasets/showlab/ShowUI-desktop-8K) and [ShowUI-Web](https://huggingface.co/datasets/showlab/ShowUI-web).
- Download [AMEX](https://huggingface.co/datasets/Yuxiang007/AMEX) then use our `prepare/hf_amex.py` to create metadata.
- Download grounding evaluation dataset -- [ScreenSpot](https://huggingface.co/datasets/KevinQHLin/ScreenSpot)

You can use huggingface-cli to download these datasets easily.
```
cd $_DATA_DIR
huggingface-cli download showlab/ShowUI-desktop --repo-type dataset --local-dir .
huggingface-cli download KevinQHLin/ScreenSpot --repo-type dataset --local-dir .
```

### Navigtion datasets
- Download [GUIAct](https://huggingface.co/datasets/yiye2023/GUIAct) then use our `prepare/hf_guiact.ipynb` to create metadata for each split (i.e., web, mobile).

- Set up Mind2Web, AITW, Miniwob follow [SeeClick's Instruction](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md). Then use our `prepare/hf_mind2web/aitw/miniwob.py` to process them and get the metadata.

Then, the dataset should be organized as following:
```
$_DATA_DIR
    - ScreenSpot
        - images
        - metadata
    - AMEX
        - images
        - metadata
    - ShowUI-web
        - images
        - metadata
    - ShowUI-desktop
        - images
        - metadata
    - GUI_Course
        - GUIAct
            - images
            - metadata
    - Mind2Web
        - images
        - metadata
    - AITW
        - images
        - metadata
    - MiniWob
        - images
        - metadata
```

## ⚙️Define Dataloader
You can simply re-use existed implementation of `dset_shared_grounding.py` for UI grounding;
or `dset_shared_navigation.py` for UI navigation;

For grounding, you just need to define the dataset_mapping for path identification such as `"showui": "hf_train.json"`

Please organize the UI grounding metadata as following:
```
"""
sample = {
        "img_url": "c12b572ebccfae5052fe62826615c58d.png",
        "img_size": [
            1920,
            1080
        ],
        "element": [
            {
                "instruction": "Galerie",
                "bbox": [
                    0.6125,
                    0.35648148148148145,
                    0.6817708333333333,
                    0.375
                ],
                "data_type": "text",
                "point": [
                    0.65,
                    0.37
                ]
            },
            {
                "instruction": "Coiffure",
                "bbox": [
                    0.30416666666666664,
                    0.35648148148148145,
                    0.3770833333333333,
                    0.375
                ],
                "data_type": "text",
                "point": [
                    0.34,
                    0.37
                ]
            }],
        "element_size": 2
}
"""
```

For navigation, you need to define the dataset_mapping as above;
Beside, you need to define the action space in `template/shared_navigation.py` for your customized scenario.

## 〽️Start Grounding Training
Below are instruction for training on grounding then evaluation on screenspot grounding;

Please keep the `bsz` as 1, if you want to enlarge the bsz, just increase the `grad_accumulation_steps`.

Our codebase use [Wandb](https://wandb.ai/) to monitor training process, please provide your own Wandb API key by `$WANDB_KEY`.

```
deepspeed --include localhost:1 --master_port 5678 train.py \
  --wandb_key=$WANDB_KEY \
  --model_id='showlab/ShowUI-2B' \
  --version='showlab/ShowUI-2B' \
  --dataset_dir=$_DATA_DIR \
  --log_base_dir=$_SAVE_DIR \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="debug" \
  --train_ratio="1"  \
  --train_dataset="showui-desktop"  \
  --train_json="hf_train"   \
  --val_dataset="screenspot"  \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=32 \
  --lora_alpha=64  \
  --min_visual_tokens=256  \
  --max_visual_tokens=1344  \
  --num_turn=100 \
  --crop_min=0.5 \
  --crop_max=1.5 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt  \
  --ds_zero="zero2" \
  --gradient_checkpointing  \
  --lm_skip_ratio=0.5   \
  --lm_skip_layer='[1,28,0]'
```
Then, the model checkpoints will be saved under `$_SAVE_DIR/$exp_id`

We have provided evaluation script for screenspot in `main/eval_screenspot.py`.
If you want to evaluate on your own setting, you need to define the evaluation function and place it under `main/eval_X.py`

You should able monitor the training information in wandb panel.

**Note:** If for evaluation, please apply `--eval_only` and change the `--lora_r=0`. Otherwise, the lora will change the model behavior.

## 〽️Start Navigation Training
### **Pretrained on GUI-Act (Optional)**
The code below utilizes GUI-Act for pre-training a Qwen2VL, followed by evaluation on AITW.
We have set `num_history` to 2 with `interleaved_history='tttt'`.

If you have access to greater GPU memory, feel free to switch to `vtvt` and increase the history length.
```
deepspeed --include localhost:1 --master_port 5678 train.py \
  --wandb_key=$WANDB_KEY \
  --model_id='Qwen/Qwen2-VL-2B-Instruct' \
  --version='Qwen/Qwen2-VL-2B-Instruct' \
  --dataset_dir=$_DATA_DIR \
  --log_base_dir=$_SAVE_DIR \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="debug" \
  --train_ratio="1,1,1"  \
  --train_dataset="guiact,guiact,guiact"  \
  --train_json="hf_train_smartphone,hf_train_web-multi,hf_train_web-single"   \
  --val_dataset="aitw"  \
  --val_json="hf_test"    \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=32 \
  --lora_alpha=64  \
  --min_visual_tokens=256  \
  --max_visual_tokens=1344  \
  --num_turn=100 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt  \
  --ds_zero="zero2" \
  --gradient_checkpointing  \
  --lm_skip_ratio=0.5   \
  --lm_skip_layer='[1,28,0]'    \
  --num_history=2    \
  --interleaved_history='tttt'
```

For Mind2Web AITW zero-shot evaluation, we may encounter action mismatches (e.g., `TYPE`, `INPUT`), leading to unstable scores. To mitigate this, we pretrain on navigation data and monitor intermediate scores throughout pretraining, reporting the best-performing result.

### **Fine-tuned on Downstream Tasks**
The code below utilizes downstream training data for fine-tuning our ShowUI.

To ensure a better performance, we enlarge the `min_visual_tokens` to 1344 and `max_visual_tokens` to 1680 during fine-tuning stage.

You can easily replace the training `train_dataset` / validation dataset `val_dataset` to `aitw` or `mind2web`, and replace the `train_json` or `val_json` if needed.
- For Mind2Web, `train_dataset=mind2web`, `train_json='hf_train'` and `val_json='hf_test_full'`.
- For AITW, `train_dataset=aitw`, `train_json='hf_train'` and `val_json='hf_test'`.
- For Miniwob, `train_dataset=miniwob`, `train_json='hf_miniwob`. Follow the [SeeClick](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md#evaluation-on-miniwob) to set up the evaluation environment.

```
deepspeed --include localhost:1 --master_port 5678 train.py \
  --wandb_key=$WANDB_KEY \
  --model_id='showlab/ShowUI-2B' \
  --version='showlab/ShowUI-2B' \
  --dataset_dir=$_DATA_DIR \
  --log_base_dir=$_SAVE_DIR \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="debug" \
  --train_ratio="1"  \
  --train_dataset="aitw"  \
  --train_json="hf_train"   \
  --val_dataset="aitw"  \
  --val_json="hf_test"    \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=32 \
  --lora_alpha=64  \
  --min_visual_tokens=1344  \
  --max_visual_tokens=1680  \
  --num_turn=100 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt  \
  --ds_zero="zero2" \
  --gradient_checkpointing  \
  --lm_skip_ratio=0.5   \
  --lm_skip_layer='[1,28,0]'    \
  --num_history=4    \
  --interleaved_history='tttt'
```

**Note:** If for evaluation, please apply `--eval_only` and change the `--lora_r=0`. Otherwise, the lora will change the model behavior.

## 〽️Multi-Task Co-Training
Below is the instruction to use both grounding and navigation data for co-training. Training on multiple nodes (e.g. 32 GPUs) is recommended.

You can easily add or delete any training data `train_dataset` and adjust the `train_ratio`.

```
deepspeed --include localhost:1 --master_port 5678 train.py \
  --wandb_key=$WANDB_KEY \
  --model_id='Qwen/Qwen2-VL-2B-Instruct' \
  --version='Qwen/Qwen2-VL-2B-Instruct' \
  --dataset_dir=$_DATA_DIR \
  --log_base_dir=$_SAVE_DIR \
  --epochs=50 \
  --steps_per_epoch=100 \
  --batch_size=1 \
  --grad_accumulation_steps=2 \
  --model_max_length=8192 \
  --exp_id="debug" \
  --train_ratio="1,1,1,1,1,1"  \
  --train_dataset="showui-desktop, showui-web, amex, guiact, guiact, guiact,"  \
  --train_json="hf_train, hf_train, hf_train, hf_train_smartphone, hf_train_web-multi, hf_train_web-single"   \
  --val_dataset="screenspot"  \
  --val_json="hf_test"    \
  --precision="bf16" \
  --attn_imple="sdpa" \
  --workers=0 \
  --lora_r=32 \
  --lora_alpha=64  \
  --min_visual_tokens=256  \
  --max_visual_tokens=1344  \
  --num_turn=100 \
  --random_sample \
  --record_sample \
  --lr=0.0001 \
  --uniform_prompt  \
  --ds_zero="zero2" \
  --gradient_checkpointing  \
  --lm_skip_ratio=0.5   \
  --lm_skip_layer='[1,28,0]'    \
  --num_history=2    \
  --interleaved_history='tttt'
```

## ⬇️Save Model Checkpoints
Once you finished the training, you can use the following cmd to save the model checkpoint.

```bash
exp_dir="$_SAVE_DIR/$exp_id/2024-11-28_17-30-32/"
showui_dir=$(pwd)
ckpt_dir="${exp_dir}/ckpt_model/"
merge_dir="${ckpt_dir}/merged_model"

cd "$ckpt_dir" || { echo "Failed to cd to $ckpt_dir"; exit 1; }
python zero_to_fp32.py . pytorch_model.bin
mkdir -p merged_model

cd "$showui_dir"
python3 merge_weight.py --exp_dir="$exp_dir"

echo "$merge_dir"
```
