# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import gradio as gr

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    def complete_prompt(prompt, max_gen_len, temperature, top_p):
        results = generator.generate(
            [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
        )
        return str(*results)

    if local_rank == 0:
        with gr.Blocks() as demo:
            with gr.Row():
                title = gr.Textbox(value="Llama model size 7B parameters: COMPLETE THE PROMPT",label="title", interactive=False)
            with gr.Row():
                description = gr.Textbox(
                    value="""Generations are bad! 
                    Keep in mind these models are not finetuned for question answering. As such, they should be prompted so that the expected answer is the natural continuation of the prompt.
                    Here are a few examples of prompts (from [issue#69](https://github.com/facebookresearch/llama/issues/69)) geared towards finetuned models, and how to modify them to get the expected results:
                    - Do not prompt with \"What is the meaning of life? Be concise and do not repeat yourself." but with \"I believe the meaning of life is\"
                    - Do not prompt with \"Explain the theory of relativity.\" but with \"Simply put, the theory of relativity states that"
                    - Do not prompt with \"Ten easy steps to build a website...\" but with \"Building a website can be done in 10 simple steps:\"
                    To be able to directly prompt the models with questions / instructions, you can either:
                    - Prompt it with few-shot examples so that the model understands the task you have in mind.
                    - Finetune the models on datasets of instructions to make them more robust to input prompts.
                    We've updated `example.py` with more sample prompts. Overall, always keep in mind that models are very sensitive to prompts (particularly when they have not been finetuned).
                    """, label="description", interactive=False
                )
            with gr.Row():
                input_prompt = gr.Textbox(label="input_prompt")
            with gr.Row():
                max_gen_len = gr.Number(
                    value=int(256), label="max_gen_len (Recommended max value is 256): ", precision=0, interactive=True
                )
            with gr.Row():
                temperature = gr.Number(
                    value=float(0.8), label="temperature: ", precision=4, interactive=True
                )
            with gr.Row():
                top_p = gr.Number(
                    value=float(0.95), label="top_p: ", precision=4, interactive=True
                )
            with gr.Row():
                submit_prompt_button = gr.Button("Complete Prompt")
            with gr.Row():
                output_prompt = gr.Textbox(label="output_prompt")
            
            submit_prompt_button.click(
                complete_prompt,
                inputs=[input_prompt, max_gen_len, temperature, top_p],
                outputs=[output_prompt]
            )
        # demo = gr.Interface(
        #     title="Llama model size 7B parameters",
        #     description="""Generations are bad! 

        #     Keep in mind these models are not finetuned for question answering. As such, they should be prompted so that the expected answer is the natural continuation of the prompt.
        #     Here are a few examples of prompts (from [issue#69](https://github.com/facebookresearch/llama/issues/69)) geared towards finetuned models, and how to modify them to get the expected results:
        #     - Do not prompt with \"What is the meaning of life? Be concise and do not repeat yourself." but with \"I believe the meaning of life is\"
        #     - Do not prompt with \"Explain the theory of relativity.\" but with \"Simply put, the theory of relativity states that"
        #     - Do not prompt with \"Ten easy steps to build a website...\" but with \"Building a website can be done in 10 simple steps:\"
        #     To be able to directly prompt the models with questions / instructions, you can either:
        #     - Prompt it with few-shot examples so that the model understands the task you have in mind.
        #     - Finetune the models on datasets of instructions to make them more robust to input prompts.
        #     We've updated `example.py` with more sample prompts. Overall, always keep in mind that models are very sensitive to prompts (particularly when they have not been finetuned).
        #     """,
        #     fn=complete_prompt, 
        #     inputs="text", 
        #     outputs="text")
        port=6789
        print(("Connect to Port: ", port))
        # sys.stdout.write("Connect to Port: ", os.getenv("GRADIO_SERVER_PORT"))
        sys.stdout.flush()
        demo.launch(server_name="0.0.0.0", server_port=int(port))


if __name__ == "__main__":
    fire.Fire(main)
