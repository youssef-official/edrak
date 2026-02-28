"""
Edrak Transformer Architecture
بنية ترانسفورمر إدراك المتقدمة

هذه البنية مخصصة للتعامل مع:
- البرمجة متعددة اللغات
- اللغة العربية بجميع لهجاتها
- الرياضيات والفيزياء
- الرسومات ثلاثية الأبعاد
"""

import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ═══════════════════════════════════════════════════════════════════════════════
# إعدادات النموذج
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EdrakConfig:
    """إعدادات نموذج Edrak"""
    vocab_size: int = 100000
    hidden_size: int = 4096
    num_layers: int = 48
    num_attention_heads: int = 64
    intermediate_size: int = 16384
    max_position_embeddings: int = 32768
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_function: str = "swiglu"
    layer_norm_eps: float = 1.0e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # إعدادات الرؤية
    vision_encoder: bool = True
    vision_hidden_size: int = 1024
    vision_num_layers: int = 24
    
    # إعدادات الكود
    code_vocab_size: int = 50000
    code_languages: Tuple[str, ...] = ("python", "javascript", "typescript", "html", "css", "cpp", "java", "rust")
    
    # إعدادات اللغة العربية
    arabic_vocab_size: int = 50000
    arabic_dialects: Tuple[str, ...] = ("msa", "egyptian", "gulf", "levantine", "maghrebi")

# ═══════════════════════════════════════════════════════════════════════════════
# طبقات الترانسفورمر الأساسية
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS Normalization - أكثر كفاءة من LayerNorm"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - لتحسين فهم المواقع النسبية"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """تدوير نصف الأبعاد"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """تطبيق الـ Rotary Position Embedding"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU Activation - أفضل أداء في LLMs الحديثة"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ═══════════════════════════════════════════════════════════════════════════════
# آلية الانتباه المتقدمة
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakAttention(nn.Module):
    """آلية انتباه متعددة الرؤوس مع تحسينات متقدمة"""
    
    def __init__(self, config: EdrakConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.num_heads // 4  # GQA - Grouped Query Attention
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)
        
        self.attention_dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # إسقاط الاستعلام والمفتاح والقيمة
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # إعادة التشكيل
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # تطبيق Rotary Position Embedding
        cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # إعادة استخدام المفتاح والقيمة السابقة (للتوليد التدريجي)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # توسيع المفاتيح والقيم لـ GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # حساب الانتباه
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # إعادة التشكيل
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value

# ═══════════════════════════════════════════════════════════════════════════════
# طبقة الترميز (Encoder Layer)
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakLayer(nn.Module):
    """طبقة ترميز واحدة من Edrak"""
    
    def __init__(self, config: EdrakConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # طبقة الانتباه الذاتي
        self.self_attn = EdrakAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # طبقة التغذية الأمامية
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        residual = hidden_states
        
        # طبقة الانتباه
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # طبقة MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value

# ═══════════════════════════════════════════════════════════════════════════════
# معالج الكود المتخصص
# ═══════════════════════════════════════════════════════════════════════════════

class CodeEncoder(nn.Module):
    """مشفر متخصص للأكواد البرمجية"""
    
    def __init__(self, config: EdrakConfig):
        super().__init__()
        self.config = config
        
        # تضمين اللغة
        self.language_embeddings = nn.Embedding(len(config.code_languages), config.hidden_size)
        
        # طبقات الترميز
        self.layers = nn.ModuleList([
            EdrakLayer(config, i) for i in range(config.num_layers // 2)
        ])
        
        # فك التشفير للكود
        self.code_head = nn.Linear(config.hidden_size, config.code_vocab_size, bias=False)
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        language_id: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # إضافة تضمين اللغة
        lang_emb = self.language_embeddings(language_id)
        hidden_states = hidden_states + lang_emb.unsqueeze(1)
        
        # تمرير عبر الطبقات
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

# ═══════════════════════════════════════════════════════════════════════════════
# المعالج العربي المتخصص
# ═══════════════════════════════════════════════════════════════════════════════

class ArabicEncoder(nn.Module):
    """مشفر متخصص للغة العربية ولهجاتها"""
    
    def __init__(self, config: EdrakConfig):
        super().__init__()
        self.config = config
        
        # تضمين اللهجة
        self.dialect_embeddings = nn.Embedding(len(config.arabic_dialects), config.hidden_size)
        
        # طبقات الترميز
        self.layers = nn.ModuleList([
            EdrakLayer(config, i) for i in range(config.num_layers // 2)
        ])
        
        # رأس اللغة العربية
        self.arabic_head = nn.Linear(config.hidden_size, config.arabic_vocab_size, bias=False)
        
        # طبقة التحليل النحوي
        self.grammar_analyzer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 128)  # 128 علامة نحوية
        )
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        dialect_id: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # إضافة تضمين اللهجة
        if dialect_id is not None:
            dialect_emb = self.dialect_embeddings(dialect_id)
            hidden_states = hidden_states + dialect_emb.unsqueeze(1)
        
        # تمرير عبر الطبقات
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # التحليل النحوي
        grammar_logits = self.grammar_analyzer(hidden_states)
        
        return {
            "hidden_states": hidden_states,
            "grammar_logits": grammar_logits,
            "arabic_logits": self.arabic_head(hidden_states)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# النموذج الرئيسي - Edrak Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakTransformer(nn.Module):
    """
    نموذج Edrak Transformer الرئيسي
    
    هذا النموذج يجمع بين:
    - فهم عام عميق (General Understanding)
    - ترميز متخصص للأكواد
    - معالجة متقدمة للغة العربية
    - دعم الرياضيات والفيزياء
    """
    
    def __init__(self, config: EdrakConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        
        # التضمينات الأساسية
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # طبقات الترميز الرئيسية
        self.layers = nn.ModuleList([
            EdrakLayer(config, i) for i in range(config.num_layers)
        ])
        
        # المشفر المتخصص للأكواد
        self.code_encoder = CodeEncoder(config)
        
        # المشفر المتخصص للعربية
        self.arabic_encoder = ArabicEncoder(config)
        
        # طبقة التطبيع النهائية
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # رأس اللغة الرئيسي
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # رأس الرياضيات
        self.math_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)  # للتعبيرات الرياضية
        )
        
        # رأس الفيزياء
        self.physics_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 512)  # للصيغ الفيزيائية
        )
        
        # رأس الرسومات 3D
        self.graphics3d_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1024)  # للإحداثيات ثلاثية الأبعاد
        )
        
        # بوابة التوجيه (Router) - لتحديد المسار المناسب
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 5),  # general, code, arabic, math, physics, 3d
            nn.Softmax(dim=-1)
        )
        
        # تهيئة الأوزان
        self._init_weights()
        
        # عدد المعاملات
        self.print_model_size()
    
    def _init_weights(self):
        """تهيئة الأوزان"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def print_model_size(self):
        """طباعة حجم النموذج"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║                    Edrak AI Model Stats                      ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Total Parameters:     {total_params:>15,} ({total_params/1e9:.2f}B)  ║")
        print(f"║  Trainable Parameters: {trainable_params:>15,} ({trainable_params/1e9:.2f}B)  ║")
        print(f"║  Layers:               {self.config.num_layers:>15}                ║")
        print(f"║  Hidden Size:          {self.config.hidden_size:>15}                ║")
        print(f"║  Attention Heads:      {self.config.num_attention_heads:>15}                ║")
        print(f"║  Vocab Size:           {self.config.vocab_size:>15}                ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        mode: str = "general",
        language_id: Optional[torch.Tensor] = None,
        dialect_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # استرجاع التضمينات
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # إنشاء قناع الانتباه
        if attention_mask is None:
            attention_mask = torch.ones(
                (hidden_states.size(0), hidden_states.size(1)),
                dtype=torch.long,
                device=hidden_states.device
            )
        
        # توسيع القناع للبعد الصحيح
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # التوجيه التلقائي
        router_logits = self.router(hidden_states.mean(dim=1))
        
        # المعالجة حسب الوضع
        if mode == "code" and language_id is not None:
            # استخدام مشفر الكود
            hidden_states = self.code_encoder(hidden_states, language_id, attention_mask)
        elif mode == "arabic":
            # استخدام المشفر العربي
            arabic_output = self.arabic_encoder(hidden_states, dialect_id, attention_mask)
            hidden_states = arabic_output["hidden_states"]
        else:
            # المعالجة العامة
            next_cache = () if use_cache else None
            
            for idx, layer in enumerate(self.layers):
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                
                if use_cache:
                    next_cache += (present_key_value,)
            
            hidden_states = self.norm(hidden_states)
        
        # المخرجات
        outputs = {"hidden_states": hidden_states}
        
        # اللغة الرئيسية
        outputs["logits"] = self.lm_head(hidden_states)
        
        # الرياضيات
        outputs["math_logits"] = self.math_head(hidden_states)
        
        # الفيزياء
        outputs["physics_logits"] = self.physics_head(hidden_states)
        
        # الرسومات 3D
        outputs["graphics3d_logits"] = self.graphics3d_head(hidden_states)
        
        # التوجيه
        outputs["router_logits"] = router_logits
        
        if use_cache:
            outputs["past_key_values"] = next_cache
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        **kwargs
    ) -> torch.Tensor:
        """
        توليد النصوص باستخدام الـ Transformer
        """
        batch_size = input_ids.shape[0]
        
        # توسيع الإدخال إذا طُلبت عدة تسلسلات
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
        
        # تجهيز الكاش
        past_key_values = None
        
        for _ in range(max_length):
            # التمرير الأمامي
            outputs = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.get("past_key_values")
            next_token_logits = outputs["logits"][:, -1, :]
            
            # تطبيق الحرارة
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-p sampling
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                    probs[indices_to_remove] = 0
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # إضافة التوكن الجديد
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # تحديث القناع
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
                ], dim=-1)
            
            # التحقق من انتهاء التسلسل
            if (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def save_pretrained(self, save_path: str):
        """حفظ النموذج"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # حفظ حالة النموذج
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, save_path / "edrak_model.pt")
        
        print(f"✅ تم حفظ النموذج في: {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """تحميل النموذج"""
        load_path = Path(load_path)
        checkpoint = torch.load(load_path / "edrak_model.pt", map_location='cpu')
        
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ تم تحميل النموذج من: {load_path}")
        return model

# ═══════════════════════════════════════════════════════════════════════════════
# نماذج مساعدة
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakRewardModel(nn.Module):
    """نموذج المكافآت للـ RLHF"""
    
    def __init__(self, config: EdrakConfig):
        super().__init__()
        self.config = config
        
        self.transformer = EdrakTransformer(config)
        self.reward_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.transformer(input_ids, attention_mask)
        # متوسط الحالات المخفية
        pooled = (outputs["hidden_states"] * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        reward = self.reward_head(pooled)
        return reward

class EdrakForSequenceClassification(nn.Module):
    """نموذج Edrak للتصنيف"""
    
    def __init__(self, config: EdrakConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.transformer = EdrakTransformer(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        outputs = self.transformer(input_ids, attention_mask)
        pooled = outputs["hidden_states"][:, 0]  # استخدام أول توكن
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs["hidden_states"]
        }
