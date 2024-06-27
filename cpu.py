import argparse
import logging
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import gc
import torch
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    print('💽 Loading model and tokenizer...')
    print('model name: ', args.model_name)
    device = 'cpu'
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # model = LlamaForCausalLM.from_pretrained()
    
    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)

    # model.to(device) # 不再需要将模型移动到设备上
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print('💽 Loading harmful data...')    
    
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )
    
    print('💽 Loading normal data...')    

    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print('🚀 Model training...')    

    model.train()

    # Reference model for computing KL.
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    
    loss_history = {
        "bad_loss": [],
        "random_loss": [],
        "normal_loss": []
    }
    
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            bad_loss = get_answer_loss("ga", bad_batch, model, device=device) # 不再传入device参数

            random_loss = get_rand_ans_loss(
                bad_batch,
                tokenizer,
                normal_ans,
                model,
                K=5,
                device=device
            )
            
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device=device)

            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )
            
            loss_history["bad_loss"].append(-bad_loss.item())
            loss_history["random_loss"].append(random_loss.item())
            loss_history["normal_loss"].append(normal_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"random_loss: {random_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            if idx % args.save_every == 0:
            # if idx == 200 or idx == 500 or idx == 1000:
                model.save_pretrained(args.model_save_dir + f'_step{idx}', from_pt=True)
                print('📁 Archive model')
                
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history["bad_loss"], label="Loss on Unlearned Samples")
                plt.plot(loss_history["random_loss"], label="Random Mismatch Loss")
                plt.plot(loss_history["normal_loss"], label="Loss on Normal Samples")
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.title("Loss Curves During Training")
                plt.legend()
                plt.savefig(f'result/1.3b-loss_curves_opt_steps{idx}.png')
            
            if idx >= args.max_unlearn_steps or bad_loss >= args.max_bad_loss:
                break
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # model.save_pretrained(args.model_save_dir, from_pt=True)
    logging.info("Unlearning finished")

    print("🚀 Unlearning Process Finished!")

    # plt.figure(figsize=(10, 6))
    # plt.plot(loss_history["bad_loss"], label="Loss on Unlearned Samples")
    # plt.plot(loss_history["random_loss"], label="Random Mismatch Loss")
    # plt.plot(loss_history["normal_loss"], label="Loss on Normal Samples")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.title("Loss Curves During Training")
    # plt.legend()
    # plt.savefig("result/1.3b-loss_curves_opt.png")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="Unlearning LR."
    )
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
