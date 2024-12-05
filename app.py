import base64
import json
from datetime import datetime
import gradio as gr
import torch
import spaces
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import ast
import os
from datetime import datetime
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

# Define constants
DESCRIPTION = "[ShowUI Demo](https://huggingface.co/showlab/ShowUI-2B)"
_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1344 * 28 * 28

# Specify the model repository and destination folder
model_repo = "showlab/ShowUI-2B"
destination_folder = "./showui-2b"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the repository
files = list_repo_files(repo_id=model_repo)

# Download each file to the destination folder
for file in files:
    file_path = hf_hub_download(repo_id=model_repo, filename=file, local_dir=destination_folder)
    print(f"Downloaded {file} to {file_path}")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./showui-2b",
    # "showlab/ShowUI-2B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# Helper functions
def draw_point(image_input, point=None, radius=5):
    """Draw a point on the image."""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = Image.fromarray(np.uint8(image_input))

    if point:
        x, y = point[0] * image.width, point[1] * image.height
        ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
    return image

def array_to_image_path(image_array):
    """Save the uploaded image and return its path."""
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    img = Image.fromarray(np.uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    return os.path.abspath(filename)

@spaces.GPU
def run_showui(image, query):
    """Main function for inference."""
    image_path = array_to_image_path(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SYSTEM},
                {"type": "image", "image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": query}
            ],
        }
    ]

    # Prepare inputs for the model

    global model

    model = model.to("cuda")
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse the output into coordinates
    click_xy = ast.literal_eval(output_text)

    # Draw the point on the image
    result_image = draw_point(image_path, click_xy, radius=10)
    return result_image, str(click_xy)

# Function to record votes
def record_vote(vote_type, image_path, query, action_generated):
    """Record a vote in a JSON file."""
    vote_data = {
        "vote_type": vote_type,
        "image_path": image_path,
        "query": query,
        "action_generated": action_generated,
        "timestamp": datetime.now().isoformat()
    }
    with open("votes.json", "a") as f:
        f.write(json.dumps(vote_data) + "\n")
    return f"Your {vote_type} has been recorded. Thank you!"

# Helper function to handle vote recording
def handle_vote(vote_type, image_path, query, action_generated):
    """Handle vote recording by using the consistent image path."""
    if image_path is None:
        return "No image uploaded. Please upload an image before voting."
    return record_vote(vote_type, image_path, query, action_generated)

# Load logo and encode to Base64
with open("./assets/showui.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")


# Define layout and UI
def build_demo(embed_mode, concurrency_count=1):
    with gr.Blocks(title="ShowUI Demo", theme=gr.themes.Default()) as demo:
        # State to store the consistent image path
        state_image_path = gr.State(value=None)

        if not embed_mode:
            gr.HTML(
                f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <!-- Image -->
                    <div style="display: flex; justify-content: center;">
                        <img src="data:image/png;base64,{base64_image}" alt="ShowUI" width="320" style="margin-bottom: 10px;"/>
                    </div>
            
                    <!-- Description -->
                    <p>ShowUI is a lightweight vision-language-action model for GUI agents.</p>
            
                    <!-- Links -->
                    <div style="display: flex; justify-content: center; gap: 15px; font-size: 20px;">
                        <a href="https://huggingface.co/showlab/ShowUI-2B" target="_blank">
                            <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ShowUI--2B-blue" alt="model"/>
                        </a>
                        <a href="https://arxiv.org/abs/2411.17465" target="_blank">
                            <img src="https://img.shields.io/badge/arXiv%20paper-2411.17465-b31b1b.svg" alt="arXiv"/>
                        </a>
                        <a href="https://github.com/showlab/ShowUI" target="_blank">
                            <img src="https://img.shields.io/badge/GitHub-ShowUI-black" alt="GitHub"/>
                        </a>
                    </div>
                </div>
                """
            )

        with gr.Row():
            with gr.Column(scale=3):
                # Input components
                imagebox = gr.Image(type="numpy", label="Input Screenshot")
                textbox = gr.Textbox(
                    show_label=True,
                    placeholder="Enter a query (e.g., 'Click Nahant')",
                    label="Query",
                )
                submit_btn = gr.Button(value="Submit", variant="primary")

                # Placeholder examples
                gr.Examples(
                    examples=[
                        ["./examples/app_store.png", "Download Kindle."],
                        ["./examples/ios_setting.png", "Turn off Do not disturb."],
                        ["./examples/apple_music.png", "Star to favorite."],
                        ["./examples/map.png", "Boston."],
                        ["./examples/wallet.png", "Scan a QR code."],
                        ["./examples/word.png", "More shapes."],
                        ["./examples/web_shopping.png", "Proceed to checkout."],
                        ["./examples/web_forum.png", "Post my comment."],
                        ["./examples/safari_google.png", "Click on search bar."],
                    ],
                    inputs=[imagebox, textbox],
                    examples_per_page=3
                )

            with gr.Column(scale=8):
                # Output components
                output_img = gr.Image(type="pil", label="Output Image")
                # Add a note below the image to explain the red point
                gr.HTML(
                    """
                    <p><strong>Note:</strong> The <span style="color: red;">red point</span> on the output image represents the predicted clickable coordinates.</p>
                    """
                )
                output_coords = gr.Textbox(label="Clickable Coordinates")

                # Buttons for voting, flagging, regenerating, and clearing
                with gr.Row(elem_id="action-buttons", equal_height=True):
                    vote_btn = gr.Button(value="👍 Vote", variant="secondary")
                    downvote_btn = gr.Button(value="👎 Downvote", variant="secondary")
                    flag_btn = gr.Button(value="🚩 Flag", variant="secondary")
                    regenerate_btn = gr.Button(value="🔄 Regenerate", variant="secondary")
                    clear_btn = gr.Button(value="🗑️ Clear", interactive=True)  # Combined Clear button

            # Define button actions
            def on_submit(image, query):
                """Handle the submit button click."""
                if image is None:
                    raise ValueError("No image provided. Please upload an image before submitting.")
                
                # Generate consistent image path and store it in the state
                image_path = array_to_image_path(image)
                return run_showui(image, query) + (image_path,)

            submit_btn.click(
                on_submit,
                [imagebox, textbox],
                [output_img, output_coords, state_image_path],
            )

            clear_btn.click(
                lambda: (None, None, None, None, None),
                inputs=None,
                outputs=[imagebox, textbox, output_img, output_coords, state_image_path],  # Clear all outputs
                queue=False
            )

            regenerate_btn.click(
                lambda image, query, state_image_path: run_showui(image, query),
                [imagebox, textbox, state_image_path],
                [output_img, output_coords],
            )

            # Record vote actions without feedback messages
            vote_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "upvote", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

            downvote_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "downvote", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

            flag_btn.click(
                lambda image_path, query, action_generated: handle_vote(
                    "flag", image_path, query, action_generated
                ),
                inputs=[state_image_path, textbox, output_coords],
                outputs=[],
                queue=False
            )

    return demo
# Launch the app
if __name__ == "__main__":
    demo = build_demo(embed_mode=False)
    demo.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        debug=True,
    )