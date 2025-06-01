import os
os.environ['HF_HOME'] = 'cache'

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from transformers.utils import is_torch_xpu_available
import gc

from typing import Literal, Union
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import torch.amp as amp
from torch import nn


cfg = {
    "project_name": "dpo-distil-logit",
    "dataset": {
        "name": "Intel/orca_dpo_pairs",
        "split": "train",
        "val_split": 0.05,          # % of samples held out for eval
        "seed": 42,
    },
    "models": {
        "student": "Qwen/Qwen2.5-1.5B-Instruct",
        "teacher": "Qwen2.5-3B-DPO-intel-orca/merge", # Dpo trained model
    },
    "tokenizer": {
        "pad_side": "left",
        "chat_template": None,      # keep default that ships with Qwen
    },
    "lora": {                       # DPO requires higher Vram so LORA is suitable for consumer based GPUS <= 24GB Vram
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": [
            "k_proj","v_proj","q_proj","o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    },
    "quant": {
        "bits": 4,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
    },
    "dpo": {
        "num_train_epochs": 3,
        "per_device_batch": 1,
        "grad_accum": 4,
        "lr": 5e-5,
        "logging_steps": 50,
        "save_steps": 1000,
        "max_length": 4096,
        # KD
        "kd_temperature": 1.0,
        "kd_weight": 1e-3,
    },
    "wandb": {
        "entity": "my-team",
        "name": "exp-name",  # run name
        "tags": ["dpo", "knowledge-distillation"],
    },
    "paths": {
        "output_dir": "path-to-save-ckpts",
        "final_merged": "final-lora-merged-model",
    }
}


def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the left.

    This function operates on a binary mask and any number of additional tensors with the same dimensions as the mask.
    For each row, non-zero values are shifted to the leftmost positions. Then, columns that contain only zeros across
    all rows are truncated from the mask and tensors. Visually, this operation can be represented as follows:

    ```
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
     [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    ```

    Args:

        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`)
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the left and trailing zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.

    Example:
    ```python
    >>> mask = torch.tensor([[0, 0, 1, 1, 1],
    ...                      [0, 1, 1, 0, 0]])
    >>> tensor = torch.tensor([[9, 9, 2, 3, 4],
    ...                        [9, 5, 6, 9, 9]])
    >>> new_mask, new_tensor = flush_left(mask, tensor)
    >>> print(new_mask)
    tensor([[1, 1, 1],
            [1, 1, 0]])
    >>> print(new_tensor)
    tensor([[2, 3, 4],
            [5, 6, 0]])
    ```
    """
    # Create copy of mask and tensors
    mask = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    for i in range(mask.size(0)):
        first_one_idx = torch.nonzero(mask[i])[0].item()
        mask[i] = torch.roll(mask[i], shifts=-first_one_idx)
        for tensor in tensors:
            tensor[i] = torch.roll(tensor[i], shifts=-first_one_idx)

    # Get the first column idx that is all zeros and remove every column after that
    empty_cols = torch.sum(mask, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else mask.size(1)
    mask = mask[:, :first_empty_col]
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor[:, :first_empty_col]

    if not tensors:
        return mask
    else:
        return mask, *tensors


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


class DPOTrainerWithKD(DPOTrainer):
    def __init__(
        self,
        model,
        teacher_model,
        args,
        train_dataset,
        processing_class,
        peft_config,
    ):
        super().__init__(model=model, args=args, train_dataset=train_dataset, processing_class=processing_class, peft_config=peft_config)
        self.teacher_model = teacher_model  # Teacher model will be used for KD

    def distillation_loss(self, student_logits, teacher_logits, temperature=1.0):
        """
        Compute Knowledge Distillation loss using KL Divergence.
        Args:
            student_logits (torch.Tensor): Logits from the student model (current model being trained).
            teacher_logits (torch.Tensor): Logits from the teacher model (pre-trained model).
            temperature (float): Temperature for softening the logits.
        Returns:
            torch.Tensor: KD loss.
        """
        # Softmax and log_softmax for KD
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        # KL Divergence loss
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        return loss
    
    def response_distillation_loss(self, student_chosen_logps, teacher_chosen_logps):
        """Distillation loss at the response level"""
        # Encourage similar likelihood assignments
        return F.mse_loss(student_chosen_logps, teacher_chosen_logps.detach())
    
    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output, _ = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output, _ = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        teacher_output, teacher_logits = self.concatenated_forward(self.teacher_model.eval(), batch, no_grad=True)
        model_output, logits = self.concatenated_forward(model, batch)

        logit_kd_loss = self.distillation_loss(logits, teacher_logits)

        # resp_kd_loss = self.response_distillation_loss(model_output["chosen_logps"], teacher_output["chosen_logps"])


        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        losses = losses + logit_kd_loss * 0.001

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )

        metrics[f"{prefix}logits/kd"] = (
            self.accelerator.gather_for_metrics(logit_kd_loss).detach().mean().item()
        )

        # metrics[f"{prefix}response/kd"] = (
        #     self.accelerator.gather_for_metrics(resp_kd_loss).detach().mean().item()
        # )

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics
    
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], no_grad=False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask


            if no_grad:
                with torch.no_grad():
                    outputs = self.teacher_model(input_ids, **model_kwargs)
            else:
                outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output, logits


def chatml_format(example, tokenizer):
    # Format system
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# load + shuffle
ds = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
ds = ds.shuffle(seed=cfg["dataset"]["seed"])
val_pct = cfg["dataset"]["val_split"]
if val_pct:
    ds = ds.train_test_split(test_size=val_pct, seed=cfg["dataset"]["seed"])
    train_ds, val_ds = ds["train"], ds["test"]
else:
    train_ds, val_ds = ds, None

# load tokenizer
tok_student = AutoTokenizer.from_pretrained(cfg["models"]["student"])
tok_student.pad_token = tok_student.eos_token
tok_student.padding_side = cfg["tokenizer"]["pad_side"]
if cfg["tokenizer"]["chat_template"]:
    tok_student.chat_template = cfg["tokenizer"]["chat_template"]

tok_teacher = AutoTokenizer.from_pretrained(cfg["models"]["teacher"])
tok_teacher.pad_token = tok_student.eos_token
tok_teacher.padding_side = cfg["tokenizer"]["pad_side"]

# mapping dataset to chatML format
train_ds = train_ds.map(
    lambda ex: chatml_format(ex, tok_student),
    remove_columns=train_ds.column_names
)
if val_ds:
    val_ds = val_ds.map(
        lambda ex: chatml_format(ex, tok_student),
        remove_columns=val_ds.column_names
)
    

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=cfg["quant"]["quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, cfg["quant"]["compute_dtype"]),
)

lora_cfg = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    task_type="CAUSAL_LM",
    bias="none",
    target_modules=cfg["lora"]["target_modules"],
)

student_model = AutoModelForCausalLM.from_pretrained(
    cfg["models"]["student"], quantization_config=bnb_cfg
)
student_model.config.use_cache = False        # required for gradient-checkpointing

teacher_model = AutoModelForCausalLM.from_pretrained(
    cfg["models"]["teacher"], quantization_config=bnb_cfg
)
teacher_model.eval()                          # we never train the teacher


train_cfg = DPOConfig(
    per_device_train_batch_size = cfg["dpo"]["per_device_batch"],
    gradient_accumulation_steps = cfg["dpo"]["grad_accum"],
    num_train_epochs = cfg["dpo"]["num_train_epochs"],
    learning_rate = cfg["dpo"]["lr"],
    logging_steps = cfg["dpo"]["logging_steps"],
    save_steps = cfg["dpo"]["save_steps"],
    output_dir = cfg["paths"]["output_dir"],
    bf16 = True,
    # report_to = "wandb",
)

trainer = DPOTrainerWithKD(
    model            = student_model,
    teacher_model    = teacher_model,
    args             = train_cfg,
    train_dataset    = train_ds,
    # eval_dataset     = val_ds,
    processing_class = tok_student,    # your trainer expects tokenizer here
    peft_config      = lora_cfg,
    # max_length, truncation_mode, etc. come from your custom class attributes
)

trainer.train()


trainer.model.save_pretrained(cfg["paths"]["output_dir"])
tok_student.save_pretrained(cfg["paths"]["output_dir"])

# ─ merge LoRA into full-precision base model ─
base_fp16 = AutoModelForCausalLM.from_pretrained(
    cfg["models"]["student"], torch_dtype=torch.float16, return_dict=True
)
merged = PeftModel.from_pretrained(base_fp16, cfg["paths"]["output_dir"])
merged = merged.merge_and_unload()
merged.save_pretrained(cfg["paths"]["final_merged"])
tok_student.save_pretrained(cfg["paths"]["final_merged"])

# tidy-up
del trainer, student_model, teacher_model, merged
gc.collect(); torch.cuda.empty_cache()