from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class RopeScalingConfig(BaseModel):
    type: str = "yarn"
    factor: float = 16.0
    target_max_length: int = 262144
    original_max_position_embeddings: int = 16384


class MoEConfig(BaseModel):
    num_experts: int = Field(default=8, ge=1)
    top_k: int = Field(default=2, ge=1)
    expert_capacity: Optional[int] = None
    shared_expert_count: int = Field(default=1, ge=0)
    shared_expert_gate: bool = True
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    aux_loss_coef: float = Field(default=0.01, ge=0.0)
    jitter_noise: float = Field(default=0.0, ge=0.0)
    router_aux_loss_coef: float = Field(default=0.001, ge=0.0)

    @model_validator(mode="after")
    def validate_top_k(self) -> "MoEConfig":
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")
        return self


class VisionConfig(BaseModel):
    enabled: bool = False
    vision_encoder: str = "google/siglip-so400m-patch14-384"
    image_size: int = Field(default=384, ge=1)
    patch_size: int = Field(default=14, ge=1)
    vision_hidden_size: int = Field(default=1152, ge=1)
    num_vision_layers: int = Field(default=27, ge=1)
    num_attention_heads: int = Field(default=16, ge=1)
    intermediate_size: int = Field(default=4304, ge=1)
    projection_dim: int = Field(default=5120, ge=1)
    freeze_vision_encoder: bool = True
    tie_vision_embeddings: bool = False
    image_token_id: Optional[int] = Field(default=128000, ge=0)
    max_images_per_sample: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_image_size(self) -> "VisionConfig":
        if self.enabled and self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be evenly divisible by patch_size ({self.patch_size})")
        return self


class MultimodalConfig(BaseModel):
    vision: VisionConfig = Field(default_factory=VisionConfig)


class NSLTConfig(BaseModel):
    """NSLT-specific architecture parameters."""
    d_state: int = 2048
    d_hidden: int = 7168
    n_ssm_layers: int = 4
    n_ode_steps: int = 8
    solver: Literal["euler", "rk4", "adjoint"] = "rk4"
    n_trajectories: int = 8
    n_sim_steps: int = 16
    use_efficient_sandbox: bool = False
    sparsity_pct: float = 1.0


class MethosV3Config(BaseModel):
    """MethosV3 architecture parameters."""
    d_state: int = Field(default=4096, ge=1)
    d_hidden: int = Field(default=10240, ge=1)
    n_ssm_layers: int = Field(default=6, ge=1)
    n_hssm_levels: int = Field(default=3, ge=1)
    n_ode_steps: int = Field(default=8, ge=1)
    n_trajectories: int = Field(default=7, ge=1)
    n_sim_steps: int = Field(default=16, ge=1)
    use_efficient_sandbox: bool = True
    sparsity_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    n_token_categories: int = Field(default=8, ge=1)
    n_languages: int = Field(default=16, ge=1)
    n_doc_roles: int = Field(default=8, ge=1)
    n_task_types: int = Field(default=8, ge=1)
    n_difficulty_levels: int = Field(default=5, ge=1)
    n_reasoning_types: int = Field(default=8, ge=1)
    max_subgoals: int = Field(default=64, ge=1)
    n_domains: int = Field(default=5, ge=1)
    max_reasoning_steps: int = Field(default=32, ge=1)
    n_semantic_concepts: int = Field(default=4096, ge=1)
    working_mem_capacity: int = Field(default=512, ge=1)
    max_episodes: int = Field(default=256, ge=1)
    n_context_adapter_blocks: int = Field(default=4, ge=1)
    n_language_groups: int = Field(default=8, ge=1)
    adaptive_top_k_min: int = Field(default=32, ge=1)
    adaptive_top_k_max: int = Field(default=2048, ge=1)
    n_experts: int = Field(default=7, ge=1)
    n_debate_rounds: int = Field(default=3, ge=1)
    max_refinement_passes: int = Field(default=5, ge=1)
    max_repair_iters: int = Field(default=3, ge=1)
    max_entities: int = Field(default=64, ge=1)
    n_relation_types: int = Field(default=16, ge=1)
    max_events: int = Field(default=32, ge=1)
    n_tool_types: int = Field(default=4, ge=1)
    enable_executive: bool = True
    enable_world_model: bool = True
    enable_learning_controller: bool = True
    enable_tools: bool = True
    enable_curiosity: bool = True
    enable_aux_losses: bool = True
    executive_gate_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    qa_max_passes: int = Field(default=5, ge=1)
    qa_converge_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    loss_weights: Optional[Dict[str, float]] = None

    @model_validator(mode="after")
    def validate_top_k_range(self) -> "MethosV3Config":
        if self.adaptive_top_k_min > self.adaptive_top_k_max:
            raise ValueError(f"adaptive_top_k_min ({self.adaptive_top_k_min}) cannot exceed adaptive_top_k_max ({self.adaptive_top_k_max})")
        return self


class ModelArchitectureConfig(BaseModel):
    model_type: Literal["llama", "mixtral", "qwen2_moe", "deepseek_v2", "nslt", "methos_v3"] = "nslt"
    hidden_size: int = Field(default=7168, ge=1)
    num_hidden_layers: int = Field(default=56, ge=1)
    num_attention_heads: int = Field(default=56, ge=1)
    num_key_value_heads: int = Field(default=8, ge=1)
    intermediate_size: int = Field(default=18432, ge=1)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    nslt: NSLTConfig = Field(default_factory=NSLTConfig)
    methos_v3: MethosV3Config = Field(default_factory=MethosV3Config)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    max_position_embeddings: int = Field(default=16384, ge=1)
    rope_theta: float = Field(default=10000000.0, ge=0.0)
    rope_scaling: Optional[RopeScalingConfig] = None
    attention_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "flash_attention_2"
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    hidden_act: Literal["silu", "gelu", "relu", "swish"] = "silu"
    rms_norm_eps: float = Field(default=1e-6, ge=1e-12)
    initializer_range: float = Field(default=0.02, ge=0.0)
    pretraining_tp: int = Field(default=1, ge=1)
    mlp_bias: bool = False
    gradient_checkpointing: bool = True
    activation_checkpointing: Optional[bool] = None
    use_compile: bool = False
    vocab_size: int = Field(default=128000, ge=1)

    @model_validator(mode="after")
    def validate_kv_heads(self) -> "ModelArchitectureConfig":
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        return self


class ModelConfig(BaseModel):
    name: str = "Methos Class Model"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "auto"
    train_from_scratch: bool = True
    architecture: ModelArchitectureConfig = Field(default_factory=ModelArchitectureConfig)
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    @model_validator(mode="after")
    def validate_loading(self) -> "ModelConfig":
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit cannot both be enabled")
        return self


class DPOConfig(BaseModel):
    learning_rate: float = 3e-7
    beta: float = 0.1
    batch_size: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100
    label_smoothing: float = 0.0
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid"


class ORPOConfig(BaseModel):
    learning_rate: float = 3e-7
    beta: float = 0.05
    batch_size: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100


class SimPOConfig(BaseModel):
    learning_rate: float = 3e-7
    gamma: float = 0.5
    beta: float = 2.0
    batch_size: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100


class KTOConfig(BaseModel):
    learning_rate: float = 3e-7
    beta: float = 0.1
    batch_size: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


class AlignmentMethodConfig(BaseModel):
    dpo: Optional[DPOConfig] = None
    orpo: Optional[ORPOConfig] = None
    simpo: Optional[SimPOConfig] = None
    kto: Optional[KTOConfig] = None


class AlignmentPhaseConfig(BaseModel):
    enabled: bool = False
    methods: List[Literal["dpo", "orpo", "simpo", "kto"]] = Field(default_factory=lambda: ["dpo", "kto"])
    method_configs: AlignmentMethodConfig = Field(default_factory=AlignmentMethodConfig)


class PretrainStageConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 5000
    weight_decay: float = 0.1
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 500000
    optimizer: Literal["adamw", "adamw_8bit", "adamw_fused", "sgd"] = "adamw_fused"
    data_mix: Dict[str, float] = Field(default_factory=lambda: {
        "code": 0.3, "web_text": 0.3, "books": 0.1, "math": 0.1, "science": 0.1, "other": 0.1
    })


class SFTStageConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 500
    weight_decay: float = 0.05
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    optimizer: Literal["adamw", "adamw_8bit", "adamw_fused", "sgd"] = "adamw_8bit"
    data_mix: Dict[str, float] = Field(default_factory=lambda: {
        "sft_instructions": 0.5, "conversations": 0.3, "code_tasks": 0.2
    })


class InstructionTuningConfig(BaseModel):
    enabled: bool = True
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 200
    weight_decay: float = 0.05
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 20000
    optimizer: Literal["adamw", "adamw_8bit", "adamw_fused", "sgd"] = "adamw_fused"


class RLHFConfig(BaseModel):
    enabled: bool = False
    learning_rate: float = 1e-6
    batch_size: int = 4
    max_steps: int = 10000
    ppo_epochs: int = 4
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1


class SafetyConfig(BaseModel):
    enabled: bool = True
    safety_training_steps: int = 5000
    harmlessness_loss_coef: float = 0.1
    refusal_training: bool = True
    honesty_training: bool = True
    constitution_training: bool = True
    red_teaming_iters: int = 1000
    red_team_model: Optional[str] = None


class TrainingConfig(BaseModel):
    max_seq_length: int = 8192
    response_only_loss: bool = True
    ignore_data_skip: bool = True
    save_steps: int = 1000
    save_total_limit: int = 5
    eval_strategy: Literal["steps", "epoch", "no"] = "steps"
    eval_steps: int = 500
    logging_steps: int = 10
    pretrain: PretrainStageConfig = Field(default_factory=PretrainStageConfig)
    sft: SFTStageConfig = Field(default_factory=SFTStageConfig)
    alignment: AlignmentPhaseConfig = Field(default_factory=AlignmentPhaseConfig)
    instruction_tuning: InstructionTuningConfig = Field(default_factory=InstructionTuningConfig)
    rlhf: RLHFConfig = Field(default_factory=RLHFConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class FSDPConfig(BaseModel):
    enabled: bool = True
    sharding_strategy: Literal["full_shard", "hybrid_shard", "no_shard"] = "full_shard"
    transformer_layer_cls: str = "MethosV3Model"
    backward_prefetch: Literal["backward_pre", "backward_post", "no_prefetch"] = "backward_pre"
    forward_prefetch: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    limit_all_gathers: bool = True
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    cpu_offload: bool = False


class DeepSpeedConfig(BaseModel):
    zero_stage: int = 3
    offload_optimizer: Optional[Literal["cpu", "nvme"]] = "cpu"
    offload_params: Optional[Literal["cpu", "nvme"]] = None


class DistributedConfig(BaseModel):
    strategy: Literal["fsdp", "deepspeed", "ddp", "none"] = "fsdp"
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)
    deepspeed: Optional[DeepSpeedConfig] = None

    @model_validator(mode="after")
    def validate_deepspeed(self) -> "DistributedConfig":
        if self.strategy == "deepspeed" and self.deepspeed is None:
            raise ValueError("deepspeed config is required when strategy is 'deepspeed'")
        return self


class DedupConfig(BaseModel):
    enabled: bool = True
    method: Literal["exact", "minhash", "embedding"] = "minhash"
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class ContaminationConfig(BaseModel):
    enabled: bool = True
    benchmarks: List[str] = Field(default_factory=lambda: ["human_eval", "mbpp", "mmlu", "gsm8k", "arc", "hellaswag", "truthfulqa"])


class QualityScoringConfig(BaseModel):
    enabled: bool = True
    method: Literal["heuristic", "perplexity", "classifier"] = "heuristic"


class LanguageDetectionConfig(BaseModel):
    enabled: bool = True


class ToxicityFilteringConfig(BaseModel):
    enabled: bool = True


class QualityPipelineConfig(BaseModel):
    min_length: int = 30
    max_length: int = 500000
    deduplication: DedupConfig = Field(default_factory=DedupConfig)
    contamination: ContaminationConfig = Field(default_factory=ContaminationConfig)
    quality_scoring: QualityScoringConfig = Field(default_factory=QualityScoringConfig)
    language_detection: LanguageDetectionConfig = Field(default_factory=LanguageDetectionConfig)
    toxicity_filtering: ToxicityFilteringConfig = Field(default_factory=ToxicityFilteringConfig)


class CurriculumStageConfig(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    dataset_indices: Optional[List[int]] = None


class CurriculumConfig(BaseModel):
    enabled: bool = False
    stages: List[CurriculumStageConfig] = Field(default_factory=list)


class DatasetEntryConfig(BaseModel):
    path: str
    max_samples: Optional[int] = None
    split: str = "train"
    name: Optional[str] = None
    data_dir: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None
    weight: float = 1.0


class DataConfig(BaseModel):
    cache_dir: str = "./hf_cache"
    streaming: bool = True
    max_cache_gb: int = 200
    num_download_workers: int = 16
    datasets: List[DatasetEntryConfig] = Field(default_factory=list)
    quality: QualityPipelineConfig = Field(default_factory=QualityPipelineConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)


class TokenizerConfig(BaseModel):
    source: Literal["custom", "huggingface"] = "custom"
    huggingface_model: str = "Xenova/claude-tokenizer"
    vocab_size: int = 128000
    max_samples: int = 100000
    type: Literal["bpe", "unigram", "wordpiece"] = "bpe"
    force: bool = False
    add_prefix_space: bool = False
    add_bos_token: bool = True
    add_eos_token: bool = True


class ConstitutionEntry(BaseModel):
    principle: str


class AlignmentConfig(BaseModel):
    prompt_style: Literal["standard", "constitutional", "hhh"] = "constitutional"
    constitution: List[str] = Field(default_factory=lambda: [
        "Be helpful, harmless, and honest.",
        "Provide accurate, well-reasoned responses with appropriate caveats.",
        "Refuse requests for harmful, illegal, or unethical content.",
        "Acknowledge uncertainty when you don't know something.",
        "Respect user privacy and avoid collecting personal information.",
        "Consider potential misuse of your capabilities.",
        "Respond to all languages at the same quality level.",
        "Do not generate code or instructions for weapons, malware, or surveillance.",
        "Correct your own mistakes when discovered.",
        "Avoid making up facts or citations.",
    ])


class ExperimentTrackingConfig(BaseModel):
    enabled: bool = False
    provider: Literal["wandb", "mlflow", "tensorboard", "none"] = "none"
    project: str = "methos-class-model"


class OutputConfig(BaseModel):
    model_dir: str = "./models/methos"
    data_dir: str = "./data"
    checkpoint_dir: str = "./models/methos/checkpoints"
    log_dir: str = "./logs"
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=32768, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repetition_penalty: float = Field(default=1.05, ge=0.0)
    do_sample: bool = True
    num_beams: int = Field(default=1, ge=1)


# MMLU benchmark config is handled by BenchmarkEntry below


class BenchmarkEntry(BaseModel):
    name: str
    split: str = "test"
    num_few_shot: int = 0
    max_samples: Optional[int] = None


class EvaluationConfig(BaseModel):
    benchmarks: List[str] = Field(default_factory=lambda: [
        "mmlu", "hellaswag", "arc", "human_eval", "mbpp",
        "gsm8k", "truthfulqa", "winogrande", "bbh",
    ])
    benchmark_configs: Dict[str, BenchmarkEntry] = Field(default_factory=dict)
    automated_report: bool = True
    report_dir: str = "./eval_reports"
    timeout: int = 60
    evaluate_during_training: bool = True
    eval_frequency: int = 5000


class Config(BaseModel):
    project: Optional[Dict[str, Any]] = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_v1_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "data_collection" in values:
            dc = values.pop("data_collection")
            if "datasets" not in values.get("data", {}):
                data = values.get("data", {})
                data["datasets"] = dc.get("datasets", [])
                data["cache_dir"] = dc.get("cache_dir", "./hf_cache")
                data["streaming"] = dc.get("streaming", True)
                data["max_cache_gb"] = dc.get("max_cache_gb", 50)
                data["num_download_workers"] = dc.get("num_download_workers", 8)
                values["data"] = data

        if "fsdp" in values and "distributed" not in values:
            fsdp_old = values.pop("fsdp")
            dist = {"strategy": "fsdp", "fsdp": {}}
            strat_map = {"full_shard": "full_shard", "hybrid_shard": "hybrid_shard", "no_shard": "no_shard"}
            fsdp_new = dist["fsdp"]
            fsdp_new["enabled"] = fsdp_old.get("enabled", True)
            fsdp_new["sharding_strategy"] = strat_map.get(fsdp_old.get("sharding_strategy", "full_shard"), "full_shard")
            fsdp_new["transformer_layer_cls"] = fsdp_old.get("transformer_layer_cls", "LlamaDecoderLayer")
            fsdp_new["backward_prefetch"] = fsdp_old.get("backward_prefetch", "backward_pre")
            fsdp_new["forward_prefetch"] = fsdp_old.get("forward_prefetch", True)
            values["distributed"] = dist

        if "languages" in values:
            values.pop("languages")

        arch = values.get("model", {}).get("architecture", {})
        if "rope_scaling_factor" in arch:
            factor = arch.pop("rope_scaling_factor")
            rtype = arch.pop("rope_scaling_type", "dynamic")
            if factor and factor > 1:
                arch["rope_scaling"] = {"type": rtype, "factor": factor, "target_max_length": 131072}

        return values


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return cfg.model_dump(mode="python")
