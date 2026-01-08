# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os


@dataclass
class LLMGenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95
    do_sample: bool = False
    repetition_penalty: float = 1.05


@dataclass
class HFLoadConfig:
    """Extra loading knobs for HF models (mainly for VRAM control)."""

    load_in_8bit: bool = False
    load_in_4bit: bool = False


class BaseExplainer:
    def explain(self, prompt: str, evidence: Dict[str, Any]) -> str:
        raise NotImplementedError


class TemplateExplainer(BaseExplainer):
    """Deterministic explanation without any LLM dependency."""

    def explain(self, prompt: str, evidence: Dict[str, Any]) -> str:
        # evidence is expected to include fields used in pipeline; we keep it robust.
        seg = evidence.get('segment', {})
        feats = evidence.get('features', [])
        dataset = evidence.get('dataset', 'DATA')
        window = evidence.get('window', {})
        lines = []
        lines.append("1) 异常摘要")
        lines.append(
            f"数据集 {dataset} 在窗口索引[{seg.get('start')},{seg.get('end')}] 期间出现异常，峰值窗口为 {seg.get('peak')}。")

        lines.append("\n2) 关键变量与证据（Top-K）")
        for i, fe in enumerate(feats, 1):
            name = fe.get('name', f"var_{fe.get('index')}")
            score = fe.get('score', None)
            atype = fe.get('pattern', {}).get('anomaly_type', 'unknown')
            direction = fe.get('pattern', {}).get('direction', 'unknown')
            stats = fe.get('stats', {})
            score_str = f"{float(score):.4f}" if score is not None else "N/A"
            min_v = stats.get("min", float("nan"))
            max_v = stats.get("max", float("nan"))
            last_v = stats.get("last", float("nan"))
            lines.append(
                f"- {i}. {name}: score={score_str}; type={atype} ({direction}); "
                f"range=[{min_v:.3f},{max_v:.3f}], last={last_v:.3f}"
            )

        lines.append("\n3) 异常形态判断")
        lines.append("本解释基于模型误差贡献与窗口内统计特征；若 Top-K 变量同时异常，可能属于多变量耦合异常。")

        lines.append("\n4) 可能原因（通用假设）")
        lines.append("- 传感器瞬时噪声/丢包导致的突变（需结合原始采集日志确认）")
        lines.append("- 系统工况切换导致的水平漂移（需结合控制指令/事件日志确认）")
        lines.append("- 设备性能退化导致趋势变化或波动增大（需结合历史维护记录确认）")

        lines.append("\n5) 建议检查项")
        lines.append("- 检查 Top-K 变量对应的原始数据是否存在缺失、卡死、异常尖峰")
        lines.append("- 对比异常发生前后窗口的统计量（均值/方差/极值）以及变量间相关性变化")
        lines.append("- 若可获取事件日志，核对异常窗口对应时间是否有操作/告警/工况切换")

        return "\n".join(lines)


class HuggingFaceCausalLMExplainer(BaseExplainer):
    """Use a local HuggingFace CausalLM model to generate explanations.

    This does **not** require fine-tuning: we rely on prompt engineering.

    Network/mirror tips:
    - You can set env HF_ENDPOINT / HF_HOME.
    - Or pass `hf_endpoint` / `cache_dir` explicitly (recommended on servers).
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: str = "cpu",
        torch_dtype: str = "auto",
        gen_cfg: Optional[LLMGenerationConfig] = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        hf_endpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_gpu: bool = False,
        hf_load_cfg: Optional[HFLoadConfig] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.gen_cfg = gen_cfg or LLMGenerationConfig()
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.force_gpu = bool(force_gpu)
        self.hf_load_cfg = hf_load_cfg or HFLoadConfig()

        self.hf_endpoint = (
            hf_endpoint
            or os.environ.get("HF_ENDPOINT")
            or os.environ.get("HUGGINGFACE_HUB_BASE_URL")
        )
        self.cache_dir = cache_dir or os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

        # lazy init
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def _lazy_load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise ImportError(
                "transformers is required for HuggingFaceCausalLMExplainer. "
                "Please pip install transformers>=4.38 and a compatible torch."
            ) from e

        # If user requested GPU, prefer CUDA when available.
        if self.force_gpu and (self.device in (None, "auto", "cpu")):
            if torch.cuda.is_available():
                self.device = "cuda"

        model_id = self.model_name_or_path
        if model_id == "gpt2":
            model_id = "openai-community/gpt2"

        # If we're in offline mode, prefer resolving the cached snapshot path and load from it.
        # This completely avoids any Hub metadata requests (e.g., model_info()) inside tokenizer helpers.
        if self.local_files_only:
            try:
                from huggingface_hub import snapshot_download

                local_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    local_files_only=True,
                    resume_download=False,
                )
                if local_path and os.path.isdir(local_path):
                    model_id = local_path
            except Exception:
                # Best effort: if snapshot resolution fails, fall back to model_id.
                pass

        # Ensure caches are consistent.
        if self.cache_dir:
            os.environ["HF_HOME"] = str(self.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", os.path.join(str(self.cache_dir), "hub"))

        # Strong offline mode: prevents any Hub metadata requests.
        # This is important because some tokenizers may call huggingface_hub.model_info()
        # even when local_files_only=True (e.g., template/regex detection).
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # IMPORTANT: On some torch+cudnn+cublas+transformers combos, GPT2 generation can
        # trigger a device-side assert (index out of bounds) in CUDA kernels.
        # A device-side assert poisons the CUDA context for the whole process.
        # Therefore we default to running the LLM explainer on CPU unless force_gpu=True.
        if (isinstance(self.device, str) and self.device.startswith("cuda")) and not self.force_gpu:
            self.device = "cpu"

        # Disable SDPA/flash attention fast paths. This is best-effort.
        os.environ.setdefault("TORCH_SDPA_DISABLE", "1")
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")

        # resolve dtype
        dtype = None
        if isinstance(self.torch_dtype, str):
            if self.torch_dtype.lower() == "auto":
                # Prefer fp16 on GPU for speed and VRAM; keep None on CPU.
                dtype = torch.float16 if (isinstance(self.device, str) and self.device.startswith("cuda")) else None
            elif self.torch_dtype.lower() in ["fp16", "float16"]:
                dtype = torch.float16
            elif self.torch_dtype.lower() in ["bf16", "bfloat16"]:
                dtype = torch.bfloat16
            elif self.torch_dtype.lower() in ["fp32", "float32"]:
                dtype = torch.float32

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = str(self.cache_dir)

        # Optional quantization via bitsandbytes.
        # NOTE: if enabled, model is already placed on GPU with device_map and should NOT be moved via .to().
        quantized = False
        if self.hf_load_cfg.load_in_4bit or self.hf_load_cfg.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise ImportError(
                    "To use 4-bit/8-bit loading, install bitsandbytes and a recent transformers. "
                    "pip install bitsandbytes"
                ) from e

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=bool(self.hf_load_cfg.load_in_4bit),
                load_in_8bit=bool(self.hf_load_cfg.load_in_8bit),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = "auto"  # respects CUDA_VISIBLE_DEVICES
            quantized = True

        try:
            tok = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token or tok.unk_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                **load_kwargs,
            )
        except OSError as e:
            hint = (
                "HuggingFace model files not found locally and network download failed. "
                "Fix options: (1) set --explain_hf_endpoint to a reachable mirror, (2) set --explain_hf_cache_dir/HF_HOME, "
                "(3) run with --explain_local_files_only True, or (4) use --explain_backend template."
            )
            raise OSError(
                f"{e}\n\n[Hint] {hint}\n[Debug] hf_endpoint={self.hf_endpoint}, cache_dir={self.cache_dir}, local_files_only={self.local_files_only}"
            ) from e

        # Move model when not quantized.
        if not quantized:
            target_device = self.device
            try:
                model.to(target_device)
            except Exception as e:
                # Provide a clear hint instead of silently falling back.
                msg = (
                    f"Failed to move model to device={target_device}. Falling back to CPU.\n"
                    f"Root error: {repr(e)}\n"
                    "Common causes: (1) GPU OOM (VRAM too small / already occupied), (2) CUDA not visible, "
                    "(3) CPU-only torch.\n"
                    "Tips: set CUDA_VISIBLE_DEVICES to a free GPU; or enable 4-bit loading (requires bitsandbytes)."
                )
                print("[Explain][HF][WARN] " + msg)
                model.to("cpu")
                self.device = "cpu"
        else:
            # device handled by device_map
            if torch.cuda.is_available():
                self.device = "cuda"

        model.eval()
        self._tokenizer = tok
        self._model = model

        # lightweight debug (helps confirm actual device used)
        try:
            print(f"[Explain][HF] Loaded LLM='{model_id}' on device='{self.device}' (force_gpu={self.force_gpu}, 4bit={self.hf_load_cfg.load_in_4bit}, 8bit={self.hf_load_cfg.load_in_8bit})")
        except Exception:
            pass

    def explain(self, prompt: str, evidence: Dict[str, Any]) -> str:
        self._lazy_load()
        import torch

        tok: Any = self._tokenizer
        model: Any = self._model
        assert tok is not None and model is not None

        want_zh = ("只用中文" in prompt or "中文" in prompt)

        # Prefer chat template only when the tokenizer actually has a chat_template configured.
        # Some tokenizers expose apply_chat_template() but do not ship with a template (e.g., GPT-2)
        # and will raise: "tokenizer.chat_template is not set".
        tok_chat_template = getattr(tok, "chat_template", None)
        use_chat_template = bool(tok_chat_template) and hasattr(tok, "apply_chat_template")

        if use_chat_template:
            system = (
                "你是时间序列异常诊断与解释助手。\n"
                "必须只用中文回答。\n"
                "只允许使用用户提供的证据，不要编造传感器含义。\n"
                "输出必须严格包含 1)~5) 五个小节。"
            )
            if not want_zh:
                system = (
                    "You are a time-series anomaly diagnosis assistant. "
                    "Use only the supplied evidence. "
                    "Output exactly 5 sections labeled 1) to 5)."
                )

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_for_tok = rendered
        else:
            # fallback: keep the original prompt
            prompt_for_tok = prompt

        # Respect model context length
        try:
            max_ctx = int(
                getattr(getattr(model, 'config', None), 'n_positions', 0)
                or getattr(getattr(model, 'config', None), 'max_position_embeddings', 0)
                or 0
            )
        except Exception:
            max_ctx = 0
        if not max_ctx or max_ctx <= 0:
            max_ctx = 2048

        reserve = int(self.gen_cfg.max_new_tokens) + 8
        max_prompt_tokens = max(16, max_ctx - reserve)

        inputs = tok(
            prompt_for_tok,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=int(self.gen_cfg.max_new_tokens),
            do_sample=bool(self.gen_cfg.do_sample),
            repetition_penalty=float(self.gen_cfg.repetition_penalty),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if bool(self.gen_cfg.do_sample):
            gen_kwargs.update(
                temperature=float(self.gen_cfg.temperature),
                top_p=float(self.gen_cfg.top_p),
            )

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        gen_part = out_ids[0, input_ids.shape[1]:]
        raw = tok.decode(gen_part, skip_special_tokens=True).strip()

        # Some chat models include tags; strip common junk / echoed instructions.
        text = raw
        # Remove echoed prompt prefix if the decoded text accidentally contains it
        if not use_chat_template and text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()

        # Strip common markers in our prompts that models tend to copy
        for bad in [
            "### 示例输出",
            "### 示例",
            "### 请结束生成",
            "### 结束语",
            "生成完成！",
            "生成解释",
        ]:
            if bad in text:
                text = text.split(bad, 1)[0].strip()

        # If model still outputs placeholders, keep only the first 5 sections area.
        # Heuristic: stop after section 5) if repeated.
        if "5)" in text:
            # keep up to the last occurrence of section 5) block
            idx = text.rfind("5)")
            # keep a reasonable tail after 5)
            text = text[: idx + 2 + 800].strip()

        # Optional debug: keep raw completion for inspection.
        if os.environ.get("STAMP_EXPLAIN_DEBUG_RAW", "").strip() in ("1", "true", "True"):
            return "[HF_RAW]\n" + raw + "\n\n[HF_CLEAN]\n" + (text or "<EMPTY>")

        return text or "<EMPTY>"


def build_explainer(
    backend: str,
    model_name_or_path: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    hf_endpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_gpu: bool = False,
    hf_load_in_8bit: bool = False,
    hf_load_in_4bit: bool = False,
) -> BaseExplainer:
    backend = (backend or "template").lower()
    if backend in ["none", "template", "rule", "rules"]:
        return TemplateExplainer()

    if backend in ["hf", "huggingface", "transformers"]:
        gen_cfg = LLMGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        load_cfg = HFLoadConfig(load_in_8bit=bool(hf_load_in_8bit), load_in_4bit=bool(hf_load_in_4bit))
        return HuggingFaceCausalLMExplainer(
            model_name_or_path=model_name_or_path,
            device=device,
            gen_cfg=gen_cfg,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            hf_endpoint=hf_endpoint,
            cache_dir=cache_dir,
            force_gpu=force_gpu,
            hf_load_cfg=load_cfg,
        )

    raise ValueError(f"Unknown LLM backend: {backend}. Use template|hf.")
