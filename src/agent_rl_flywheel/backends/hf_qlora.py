"""RTX 4070 SUPER 12GB 4-bit NF4 QLoRA backend (lazy imports for Mac compatibility)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from agent_rl_flywheel.env.proxy import TurnSample


class HFQLoRABackend:
    """Memory-lean HuggingFace + bitsandbytes 4-bit + PEFT LoRA backend for RTX 4070 12GB."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
        qlora_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        gradient_checkpointing: bool = True,
        max_seq_len: int = 2048,
    ) -> None:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "HFQLoRABackend requires `transformers`, `peft`, and `bitsandbytes`. "
                "Install `pip install -r requirements-4070.txt` on your Windows WSL2 / Linux 4070 environment."
            ) from exc

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = None
        if qlora_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if qlora_4bit:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=gradient_checkpointing)
        elif gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        targets = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.device = next(self.model.parameters()).device

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id or 0)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def trainable_parameters(self) -> list[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def sync_reference(self) -> None:
        """No-op: PEFT `disable_adapter()` supplies zero-VRAM reference logprobs."""
        return None

    def forward_logits(self, input_ids: torch.Tensor, use_ref: bool = False) -> torch.Tensor:
        self.model.config.use_cache = False
        ids = input_ids[:, -self.max_seq_len :].to(self.device)
        if use_ref:
            with self.model.disable_adapter(), torch.no_grad():
                return self.model(input_ids=ids).logits
        return self.model(input_ids=ids).logits

    def generate_with_logprobs(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 384,
        temperature: float = 0.7,
    ) -> TurnSample:
        """Generate exact token IDs and rollout logprobs with KV cache enabled."""
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        ctx_ids = list(prompt_token_ids[-(self.max_seq_len - max(1, max_new_tokens)) :])
        input_t = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_t,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(1e-4, temperature) if temperature > 0.0 else 1.0,
                    pad_token_id=self.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            gen_ids = out.sequences[0, len(ctx_ids) :].tolist()
            logprobs: list[float] = []
            for step_logits, tok_id in zip(out.scores, gen_ids):
                lp = F.log_softmax(step_logits[0].float(), dim=-1)[tok_id].item()
                logprobs.append(float(lp))
            text = self.decode(gen_ids)
            return TurnSample(text=text, token_ids=gen_ids, logprobs=logprobs)
        finally:
            self.model.config.use_cache = False
            if was_training:
                self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
