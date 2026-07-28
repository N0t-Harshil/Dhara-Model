from __future__ import annotations

import logging
from itertools import zip_longest
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ConstitutionalTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        constitution: List[str],
        device: Optional[torch.device] = None,
        max_length: int = 2048,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.constitution = constitution
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

    def build_constitutional_prompt(
        self,
        instruction: str,
        critique: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> str:
        constitution_block = "\n".join(f"- {p}" for p in self.constitution)
        parts = [f"### Constitution\n{constitution_block}", f"### Instruction\n{instruction}"]
        if critique:
            parts.append(f"### Critique\n{critique}")
        if revision:
            parts.append(f"### Revision\n{revision}")
        parts.append("### Response\n")
        return "\n\n".join(parts)

    @torch.no_grad()
    def critique(self, instruction: str, response: str) -> str:
        prompt = (
            f"### Instruction\nCritique the following response according to these principles:\n"
            f"{chr(10).join(f'- {p}' for p in self.constitution)}\n\n"
            f"Response to critique:\n{response}\n\n### Critique\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=min(self.max_length, self.max_length - inputs["input_ids"].shape[1]),
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def revise(self, instruction: str, response: str, critique: str) -> str:
        prompt = (
            f"### Instruction\n{instruction}\n\n"
            f"### Original Response\n{response}\n\n"
            f"### Critique\n{critique}\n\n"
            f"### Revised Response\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=min(self.max_length, self.max_length - inputs["input_ids"].shape[1]),
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_preference_pairs(
        self,
        instructions: List[str],
        num_critique_steps: int = 2,
    ) -> List[Dict[str, str]]:
        pairs: List[Dict[str, str]] = []
        for instruction in instructions:
            prompt = self.build_constitutional_prompt(instruction)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_length, self.max_length - inputs["input_ids"].shape[1]),
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
            )
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            revised = response
            for _ in range(num_critique_steps):
                critique = self.critique(instruction, revised)
                if not critique.strip():
                    critique = "The response is acceptable."
                revised = self.revise(instruction, revised, critique)

            pairs.append({
                "prompt": instruction,
                "chosen": revised,
                "rejected": response,
            })
            logger.info("Generated preference pair for: %.80s", instruction)

        return pairs

    def build_sft_with_constitution(
        self,
        instructions: List[str],
        responses: List[str],
    ) -> List[Dict[str, str]]:
        aligned: List[Dict[str, str]] = []
        for inst, resp in zip_longest(instructions, responses, fillvalue=""):
            critique = self.critique(inst, resp)
            revised = self.revise(inst, resp, critique)
            aligned.append({"instruction": inst, "output": revised})
            logger.debug("Constitutional alignment applied to: %.60s", inst)
        return aligned
