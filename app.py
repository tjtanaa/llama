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
import zmq


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def create_prompt_socket(context, local_rank, port):
    if local_rank == 0:
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://*:{port}")
        print(f"Prompt Socket = local_rank: {local_rank} create PUB Socket")
    else:
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 10)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(1)
        print(f"Prompt Socket = local_rank: {local_rank} create SUB Socket")

    return socket


def create_initialization_socket(context, local_rank, port):
    if local_rank == 0:
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 10)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.bind(f"tcp://*:{port}")
        print(f"Initialize Socket = local_rank: {local_rank} create SUB Socket")
    else:
        socket = context.socket(zmq.PUB)
        socket.connect(f"tcp://127.0.0.1:{port}")
        print(f"Initialize Socket = local_rank: {local_rank} create PUB Socket")
        time.sleep(1)

    return socket


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
    max_seq_len: int = 256,
    max_batch_size: int = 1,
    port=6789,
):
    local_rank, world_size = setup_model_parallel()
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, "w")

    context = zmq.Context(2)
    socket = create_prompt_socket(context, local_rank, port - 1)
    init_socket = create_initialization_socket(context, local_rank, port + 1)

    if local_rank == 0:
        # when the subscriber process has fully initialized
        # it will send a message to the master thread
        # when the master thread has ensured all the 
        # model weights are loaded, only then the app is launched
        subscriber_replies = []
        while len(subscriber_replies) < (world_size - 1):
            print("ready to initialize: len(subscriber_replies) " + str(len(subscriber_replies)) + "\t world_size: " + str(world_size))
            prompt = init_socket.recv_string()
            print(f"local_rank {local_rank}: {prompt}")
            subscriber_replies.append(subscriber_replies)
            # time.sleep(0.1)

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    def complete_prompt(prompt, max_gen_len, temperature, top_p):
        if max_gen_len > max_seq_len:
            return f"WARNING: Maximum value of `max_gen_len` is {max_seq_len}."
        if local_rank == 0:
            print(f"local rank: {local_rank} send {prompt}")
            socket.send_string(prompt, zmq.NOBLOCK)
        results = generator.generate(
            [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
        )
        return str(*results)

    model_size = ckpt_dir.split(os.sep)[-1].upper()

    if local_rank == 0:
        with gr.Blocks() as demo:
            with gr.Row():
                title = gr.Textbox(value=f"Llama model name {model_size}",label="title", interactive=False)
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
                    value=int(max_seq_len), label=f"max_gen_len (max value is {max_seq_len}): ", precision=0, interactive=True
                )
            with gr.Row():
                temperature=gr.Slider(
                    label="temperature",
                    value=0.8,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    interactive=True
                )
            with gr.Row():
                top_p =gr.Slider(
                    label="top_p",
                    value=0.95,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    interactive=True
                )
            with gr.Row():
                submit_prompt_button = gr.Button("Complete the Prompt")
            with gr.Row():
                output_prompt = gr.Textbox(label="output_prompt")
            
            submit_prompt_button.click(
                complete_prompt,
                inputs=[input_prompt, max_gen_len, temperature, top_p],
                outputs=[output_prompt]
            )
        sys.stdout.flush()
        demo.queue(max_size=4)
            
        print(("Connect to Port: ", port))
        demo.launch(server_name="0.0.0.0", server_port=int(port))
    else:
        while True:
            # print(f"local_rank:{local_rank} ready to receive prompt.")
            message = f"local_rank:{local_rank} ready to receive prompt."
            print(f"local_rank:{local_rank} ready to receive prompt.")
            init_socket.send_string(message, zmq.NOBLOCK)
            try:
                prompt = socket.recv_string()
                # prompt = socket.recv()
                print(f"local_rank:{local_rank} received: {prompt}")
                sys.stdout.flush()
                results = complete_prompt(prompt, max_seq_len, temperature, top_p)
                # print(f"local_rank:{local_rank} results: {results}")
                sys.stdout.flush()
            except Exception as e:
                print(repr(e))
                context.close()
                break


if __name__ == "__main__":
    fire.Fire(main)
