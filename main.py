"""
Edrak AI - Main Application
إدراك - نظام الذكاء الاصطناعي المتقدم

المخترع: يوسف السيد الغريب (Youssef Elsayed Elghareeb)

Edrak AI هو نموذج ذكاء اصطناعي متخصص في:
- البرمجة وتطوير الويب
- التصميم والـ UI/UX
- اللغة العربية العامية والفصحى
- الرياضيات والفيزياء
- الرسومات ثلاثية الأبعاد
- التعلم المستمر من البشر
"""

import os
import sys
import json
import yaml
import asyncio
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from loguru import logger

# إضافة المسار للمكتبات المحلية
sys.path.append(str(Path(__file__).parent))

from src.model.edrak_transformer import EdrakTransformer, EdrakConfig
from src.model.arabic_processor import ArabicProcessor
from src.model.code_engine import CodeEngine
from src.model.math_solver import MathSolver
from src.model.physics_engine import PhysicsEngine
from src.model.graphics3d import Graphics3D
from src.learning.human_feedback import HumanFeedbackLearner
from src.learning.self_correction import SelfCorrection
from src.memory.context_manager import ContextManager
from src.utils.safety import SafetyChecker

# ═══════════════════════════════════════════════════════════════════════════════
# إعدادات النظام
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

EDRAK_CONFIG = CONFIG['edrak_ai']

# إعداد اللوجن
logger.add("logs/edrak_{time}.log", rotation="500 MB", encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# نماذج البيانات (Pydantic Models)
# ═══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str = Field(..., description="دور الرسالة: system, user, assistant")
    content: str = Field(..., description="محتوى الرسالة")
    name: Optional[str] = Field(None, description="اسم المرسل (اختياري)")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="edrak-1.0", description="اسم النموذج")
    messages: List[Message] = Field(..., description="قائمة الرسائل")
    temperature: float = Field(default=0.7, ge=0, le=2, description="درجة العشوائية")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Top-p sampling")
    n: int = Field(default=1, ge=1, le=10, description="عدد الردود")
    stream: bool = Field(default=False, description="تدفق الرد")
    max_tokens: Optional[int] = Field(default=4096, ge=1, le=32768, description="الحد الأقصى للتوكنات")
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    user: Optional[str] = Field(None, description="معرف المستخدم")
    # إعدادات خاصة بـ Edrak
    code_mode: bool = Field(default=False, description="وضع البرمجة المتقدم")
    arabic_dialect: Optional[str] = Field(default=None, description="اللهجة العربية المطلوبة")
    math_mode: bool = Field(default=False, description="وضع الرياضيات")
    physics_mode: bool = Field(default=False, description="وضع الفيزياء")
    graphics_3d: bool = Field(default=False, description="وضع الرسومات 3D")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class CodeGenerateRequest(BaseModel):
    prompt: str = Field(..., description="وصف الكود المطلوب")
    language: str = Field(default="python", description="لغة البرمجة")
    context: Optional[str] = Field(None, description="سياق إضافي")
    framework: Optional[str] = Field(None, description="الإطار المستخدم")
    style: str = Field(default="clean", description="أسلوب الكود")
    include_tests: bool = Field(default=False, description="تضمين الاختبارات")
    user: Optional[str] = Field(None)

class CodeReviewRequest(BaseModel):
    code: str = Field(..., description="الكود المراد مراجعته")
    language: str = Field(..., description="لغة البرمجة")
    review_type: str = Field(default="full", description="نوع المراجعة")

class MathSolveRequest(BaseModel):
    problem: str = Field(..., description="مسألة الرياضيات")
    problem_type: Optional[str] = Field(None, description="نوع المسألة")
    show_steps: bool = Field(default=True, description="إظهار خطوات الحل")
    language: str = Field(default="arabic", description="لغة الإجابة")

class ArabicProcessRequest(BaseModel):
    text: str = Field(..., description="النص العربي")
    task: str = Field(default="understand", description="المهمة: understand, correct, translate, poetry")
    dialect: Optional[str] = Field(None, description="اللهجة")
    target_language: Optional[str] = Field(None, description="اللغة المستهدفة للترجمة")

class FeedbackRequest(BaseModel):
    user_id: str
    interaction_id: str
    feedback_type: str = Field(..., description="positive, negative, correction")
    comment: Optional[str] = None
    correction: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = Field(default="edrak-embed")
    user: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════════════════
# نظام Edrak AI الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakAI:
    """
    النظام الرئيسي لـ Edrak AI
    يدير جميع المكونات ويواجهة موحدة للتفاعل
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("🚀 جاري تهيئة Edrak AI...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"📱 الجهاز المستخدم: {self.device}")
        
        # تهيئة الإعدادات
        self.config = EdrakConfig(**EDRAK_CONFIG['model'])
        
        # تهيئة النموذج الرئيسي
        self.model = self._load_model()
        
        # تهيئة المعالجات المتخصصة
        self.arabic_processor = ArabicProcessor()
        self.code_engine = CodeEngine()
        self.math_solver = MathSolver()
        self.physics_engine = PhysicsEngine()
        self.graphics3d = Graphics3D()
        
        # تهيئة أنظمة التعلم
        self.feedback_learner = HumanFeedbackLearner()
        self.self_correction = SelfCorrection()
        self.context_manager = ContextManager()
        self.safety_checker = SafetyChecker()
        
        # إحصائيات
        self.stats = {
            "total_requests": 0,
            "code_generations": 0,
            "arabic_interactions": 0,
            "math_solutions": 0,
            "start_time": datetime.now()
        }
        
        self._initialized = True
        logger.info("✅ تم تهيئة Edrak AI بنجاح!")
    
    def _load_model(self) -> EdrakTransformer:
        """تحميل أو تهيئة النموذج"""
        model_path = Path(__file__).parent / "models" / "edrak_main.pt"
        
        model = EdrakTransformer(self.config)
        
        if model_path.exists():
            logger.info("📥 جاري تحميل النموذج...")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ تم تحميل النموذج")
        else:
            logger.warning("⚠️ لم يتم العثور على نموذج مدرب، سيتم استخدام نموذج جديد")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    async def chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        معالجة طلبات الدردشة مع دعم كامل للميزات المتقدمة
        """
        self.stats["total_requests"] += 1
        
        # فحص الأمان
        for msg in request.messages:
            safety_check = self.safety_checker.check(msg.content)
            if not safety_check.is_safe:
                yield self._create_error_response("تم رفض الطلب: محتوى غير آمن")
                return
        
        # إدارة السياق
        context = self.context_manager.build_context(
            request.messages,
            user_id=request.user,
            max_tokens=request.max_tokens
        )
        
        # تحديد الوضع المناسب
        mode = self._detect_mode(request)
        
        # معالجة خاصة حسب الوضع
        if mode == "code":
            async for chunk in self._handle_code_mode(context, request):
                yield chunk
        elif mode == "arabic":
            async for chunk in self._handle_arabic_mode(context, request):
                yield chunk
        elif mode == "math":
            async for chunk in self._handle_math_mode(context, request):
                yield chunk
        elif mode == "physics":
            async for chunk in self._handle_physics_mode(context, request):
                yield chunk
        elif mode == "3d":
            async for chunk in self._handle_3d_mode(context, request):
                yield chunk
        else:
            async for chunk in self._handle_general_mode(context, request):
                yield chunk
    
    def _detect_mode(self, request: ChatCompletionRequest) -> str:
        """اكتشاف الوضع المناسب بناءً على الطلب"""
        if request.code_mode:
            return "code"
        if request.math_mode:
            return "math"
        if request.physics_mode:
            return "physics"
        if request.graphics_3d:
            return "3d"
        if request.arabic_dialect:
            return "arabic"
        
        # تحليل تلقائي للرسائل
        last_message = request.messages[-1].content.lower()
        
        code_keywords = ["كود", "code", "برمج", "function", "class", "برنامج", "سكربت"]
        math_keywords = ["حساب", "معادلة", "equation", "integral", "derivative", "جبر", "تفاضل"]
        physics_keywords = ["فيزياء", "physics", "force", "energy", "velocity", "acceleration"]
        
        for kw in code_keywords:
            if kw in last_message:
                return "code"
        for kw in math_keywords:
            if kw in last_message:
                return "math"
        for kw in physics_keywords:
            if kw in last_message:
                return "physics"
        
        return "general"
    
    async def _handle_code_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """معالجة وضع البرمجة المتقدم"""
        self.stats["code_generations"] += 1
        
        # استخدام محرك الأكواد المتخصص
        code_result = await self.code_engine.generate(
            prompt=context['last_message'],
            language=self._detect_language(context['last_message']),
            context=context['history'],
            temperature=request.temperature
        )
        
        response = self._format_code_response(code_result)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    async def _handle_arabic_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """معالجة الوضع العربي المتقدم"""
        self.stats["arabic_interactions"] += 1
        
        # معالجة اللهجة المطلوبة
        dialect = request.arabic_dialect or "egyptian"
        
        arabic_result = await self.arabic_processor.process(
            text=context['last_message'],
            task="respond",
            dialect=dialect,
            context=context['history']
        )
        
        response = self._format_arabic_response(arabic_result, dialect)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    async def _handle_math_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """معالجة وضع الرياضيات المتقدم"""
        self.stats["math_solutions"] += 1
        
        math_result = await self.math_solver.solve(
            problem=context['last_message'],
            show_steps=True,
            language="arabic"
        )
        
        response = self._format_math_response(math_result)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    async def _handle_physics_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """معالجة وضع الفيزياء"""
        physics_result = await self.physics_engine.solve(
            problem=context['last_message'],
            show_derivation=True
        )
        
        response = self._format_physics_response(physics_result)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    async def _handle_3d_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """معالجة وضع الرسومات 3D"""
        graphics_result = await self.graphics3d.generate(
            description=context['last_message'],
            output_format="three.js"
        )
        
        response = self._format_3d_response(graphics_result)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    async def _handle_general_mode(
        self, 
        context: Dict, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """المعالجة العامة باستخدام النموذج الرئيسي"""
        
        with torch.no_grad():
            with autocast(enabled=True):
                # توليد الرد
                output_ids = self.model.generate(
                    input_ids=context['input_ids'].to(self.device),
                    attention_mask=context['attention_mask'].to(self.device),
                    max_length=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    num_return_sequences=1
                )
        
        # فك التشفير
        response_text = self.model.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True
        )
        
        # تطبيق التصحيح الذاتي
        corrected = await self.self_correction.correct(response_text)
        
        response = self._format_general_response(corrected)
        
        if request.stream:
            for chunk in self._stream_response(response, request):
                yield chunk
        else:
            yield json.dumps(response, ensure_ascii=False)
    
    def _detect_language(self, text: str) -> str:
        """اكتشاف لغة البرمجة من النص"""
        text_lower = text.lower()
        
        language_map = {
            "python": ["python", "بايثون", "بيثون"],
            "javascript": ["javascript", "js", "جافاسكربت"],
            "typescript": ["typescript", "ts", "تايبسكربت"],
            "html": ["html", "اتش تي ام ال"],
            "css": ["css", "سي اس اس"],
            "cpp": ["c++", "cpp", "سي بلس بلس"],
            "java": ["java", "جافا"],
            "rust": ["rust", "رست"],
            "go": ["go", "جولانج"],
        }
        
        for lang, keywords in language_map.items():
            for kw in keywords:
                if kw in text_lower:
                    return lang
        
        return "python"  # افتراضي
    
    def _format_code_response(self, code_result: Dict) -> Dict:
        """تنسيق رد الكود"""
        return {
            "id": f"edrak-code-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "edrak-1.0-code",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": code_result['code'],
                    "metadata": {
                        "language": code_result.get('language', 'unknown'),
                        "explanation": code_result.get('explanation', ''),
                        "complexity": code_result.get('complexity', 'N/A'),
                        "suggestions": code_result.get('suggestions', [])
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": code_result.get('prompt_tokens', 0),
                "completion_tokens": code_result.get('completion_tokens', 0),
                "total_tokens": code_result.get('total_tokens', 0)
            }
        }
    
    def _format_arabic_response(self, result: Dict, dialect: str) -> Dict:
        """تنسيق الرد العربي"""
        return {
            "id": f"edrak-arabic-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": f"edrak-1.0-arabic-{dialect}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result['text'],
                    "metadata": {
                        "dialect": dialect,
                        "sentiment": result.get('sentiment', 'neutral'),
                        "entities": result.get('entities', [])
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": result.get('usage', {})
        }
    
    def _format_math_response(self, result: Dict) -> Dict:
        """تنسيق رد الرياضيات"""
        content = f"## الحل:\n\n{result['solution']}"
        if result.get('steps'):
            content += "\n\n### خطوات الحل:\n"
            for i, step in enumerate(result['steps'], 1):
                content += f"\n**الخطوة {i}:** {step}"
        
        return {
            "id": f"edrak-math-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "edrak-1.0-math",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "metadata": {
                        "problem_type": result.get('problem_type', 'unknown'),
                        "difficulty": result.get('difficulty', 'medium'),
                        "verification": result.get('verification', False)
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": result.get('usage', {})
        }
    
    def _format_physics_response(self, result: Dict) -> Dict:
        """تنسيق رد الفيزياء"""
        content = f"## الحل الفيزيائي:\n\n{result['solution']}"
        if result.get('derivation'):
            content += f"\n\n### الاشتقاق:\n{result['derivation']}"
        if result.get('formulas'):
            content += "\n\n### الصيغ المستخدمة:\n"
            for formula in result['formulas']:
                content += f"- {formula}\n"
        
        return {
            "id": f"edrak-physics-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "edrak-1.0-physics",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "metadata": {
                        "topic": result.get('topic', 'general'),
                        "units": result.get('units', {})
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": result.get('usage', {})
        }
    
    def _format_3d_response(self, result: Dict) -> Dict:
        """تنسيق رد الرسومات 3D"""
        return {
            "id": f"edrak-3d-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "edrak-1.0-3d",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get('description', 'تم إنشاء المشهد ثلاثي الأبعاد'),
                    "metadata": {
                        "code": result.get('code', ''),
                        "format": result.get('format', 'three.js'),
                        "vertices": result.get('vertices', 0),
                        "faces": result.get('faces', 0)
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": result.get('usage', {})
        }
    
    def _format_general_response(self, text: str) -> Dict:
        """تنسيق الرد العام"""
        return {
            "id": f"edrak-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "edrak-1.0",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(text.split()),
                "total_tokens": len(text.split())
            }
        }
    
    def _stream_response(self, response: Dict, request: ChatCompletionRequest):
        """تدفق الرد على شكل chunks"""
        content = response['choices'][0]['message']['content']
        words = content.split()
        
        for i, word in enumerate(words):
            chunk = {
                "id": response['id'],
                "object": "chat.completion.chunk",
                "created": response['created'],
                "model": response['model'],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            asyncio.sleep(0.02)  # محاكاة التدفق
        
        yield "data: [DONE]\n\n"
    
    def _create_error_response(self, message: str) -> str:
        """إنشاء رد خطأ"""
        return json.dumps({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "content_filter"
            }
        }, ensure_ascii=False)
    
    async def generate_code(self, request: CodeGenerateRequest) -> Dict:
        """توليد كود متقدم"""
        self.stats["code_generations"] += 1
        
        result = await self.code_engine.generate(
            prompt=request.prompt,
            language=request.language,
            framework=request.framework,
            style=request.style,
            include_tests=request.include_tests,
            context=request.context
        )
        
        return {
            "code": result['code'],
            "language": request.language,
            "explanation": result.get('explanation', ''),
            "tests": result.get('tests', '') if request.include_tests else None,
            "complexity_analysis": result.get('complexity', {}),
            "security_check": result.get('security', {}),
            "suggestions": result.get('suggestions', [])
        }
    
    async def review_code(self, request: CodeReviewRequest) -> Dict:
        """مراجعة الكود"""
        review = await self.code_engine.review(
            code=request.code,
            language=request.language,
            review_type=request.review_type
        )
        
        return {
            "score": review.get('score', 0),
            "issues": review.get('issues', []),
            "suggestions": review.get('suggestions', []),
            "security_concerns": review.get('security', []),
            "performance_notes": review.get('performance', []),
            "best_practices": review.get('best_practices', [])
        }
    
    async def solve_math(self, request: MathSolveRequest) -> Dict:
        """حل مسائل الرياضيات"""
        self.stats["math_solutions"] += 1
        
        result = await self.math_solver.solve(
            problem=request.problem,
            problem_type=request.problem_type,
            show_steps=request.show_steps,
            language=request.language
        )
        
        return result
    
    async def process_arabic(self, request: ArabicProcessRequest) -> Dict:
        """معالجة النصوص العربية"""
        self.stats["arabic_interactions"] += 1
        
        result = await self.arabic_processor.process(
            text=request.text,
            task=request.task,
            dialect=request.dialect,
            target_language=request.target_language
        )
        
        return result
    
    async def learn_from_feedback(self, request: FeedbackRequest) -> Dict:
        """التعلم من تغذية راجعة البشر"""
        success = await self.feedback_learner.process_feedback(
            user_id=request.user_id,
            interaction_id=request.interaction_id,
            feedback_type=request.feedback_type,
            comment=request.comment,
            correction=request.correction,
            rating=request.rating
        )
        
        if success:
            # تطبيق التصحيح الذاتي
            if request.correction:
                await self.self_correction.learn_correction(
                    interaction_id=request.interaction_id,
                    correction=request.correction
                )
        
        return {
            "success": success,
            "message": "تم استلام التغذية الراجعة بنجاح، شكراً لمشاركتك في تعليمي!" 
                      if success else "حدث خطأ في معالجة التغذية الراجعة"
        }
    
    def get_stats(self) -> Dict:
        """الحصول على إحصائيات النظام"""
        uptime = datetime.now() - self.stats["start_time"]
        
        return {
            "system": "Edrak AI",
            "version": EDRAK_CONFIG['version'],
            "creator": EDRAK_CONFIG['creator']['name_arabic'],
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": self.stats["total_requests"],
            "code_generations": self.stats["code_generations"],
            "arabic_interactions": self.stats["arabic_interactions"],
            "math_solutions": self.stats["math_solutions"],
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "capabilities": list(EDRAK_CONFIG['capabilities'].keys())
        }

# ═══════════════════════════════════════════════════════════════════════════════
# إنشاء تطبيق FastAPI
# ═══════════════════════════════════════════════════════════════════════════════

# تهيئة النظام العالمي
edrak_system: Optional[EdrakAI] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """إدارة دورة حياة التطبيق"""
    global edrak_system
    
    # بدء التشغيل
    logger.info("🚀 جاري تشغيل Edrak AI API...")
    edrak_system = EdrakAI()
    
    yield
    
    # الإغلاق
    logger.info("🛑 جاري إغلاق Edrak AI API...")
    # حفظ الحالة إذا لزم الأمر

app = FastAPI(
    title="Edrak AI API",
    description="""
    # إدراك AI - واجهة برمجة التطبيقات
    
    نموذج ذكاء اصطناعي متقدم متخصص في:
    - البرمجة وتطوير الويب
    - اللغة العربية العامية والفصحى
    - الرياضيات والفيزياء
    - الرسومات ثلاثية الأبعاد
    
    المخترع: يوسف السيد الغريب (Youssef Elsayed Elghareeb)
    """,
    version=EDRAK_CONFIG['version'],
    lifespan=lifespan
)

# إضافة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ═══════════════════════════════════════════════════════════════════════════════
# نقاط النهاية (Endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "name": "Edrak AI",
        "name_arabic": "إدراك",
        "version": EDRAK_CONFIG['version'],
        "creator": EDRAK_CONFIG['creator']['name_arabic'],
        "creator_english": EDRAK_CONFIG['creator']['name'],
        "message": "مرحباً بك في إدراك - نموذج الذكاء الاصطناعي المتقدم",
        "message_english": "Welcome to Edrak - Advanced AI Model",
        "documentation": "/docs",
        "status": "operational"
    }

@app.get("/v1/models")
async def list_models():
    """قائمة النماذج المتاحة"""
    return {
        "object": "list",
        "data": [
            {
                "id": "edrak-1.0",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "youssef-elghareeb",
                "permission": [],
                "root": "edrak-1.0",
                "parent": None,
            },
            {
                "id": "edrak-1.0-code",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "youssef-elghareeb",
                "description": "متخصص في توليد الأكواد"
            },
            {
                "id": "edrak-1.0-arabic",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "youssef-elghareeb",
                "description": "متخصص في اللغة العربية"
            },
            {
                "id": "edrak-1.0-math",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "youssef-elghareeb",
                "description": "متخصص في الرياضيات"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    نقطة النهاية الرئيسية للدردشة
    متوافقة مع OpenAI API
    """
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        if request.stream:
            return StreamingResponse(
                edrak_system.chat_completion(request),
                media_type="text/event-stream"
            )
        else:
            response_text = ""
            async for chunk in edrak_system.chat_completion(request):
                response_text = chunk
            return json.loads(response_text)
    
    except Exception as e:
        logger.error(f"خطأ في معالجة الطلب: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: dict):
    """نقطة نهاية Completions (للتوافق مع OpenAI)"""
    # تحويل إلى chat completions
    messages = [{"role": "user", "content": request.get("prompt", "")}]
    chat_request = ChatCompletionRequest(
        model=request.get("model", "edrak-1.0"),
        messages=[Message(**m) for m in messages],
        temperature=request.get("temperature", 0.7),
        max_tokens=request.get("max_tokens", 4096),
        stream=request.get("stream", False)
    )
    return await chat_completions(chat_request)

@app.post("/v1/code/generate")
async def generate_code(request: CodeGenerateRequest):
    """توليد كود متقدم"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        result = await edrak_system.generate_code(request)
        return result
    except Exception as e:
        logger.error(f"خطأ في توليد الكود: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/code/review")
async def review_code(request: CodeReviewRequest):
    """مراجعة الكود"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        result = await edrak_system.review_code(request)
        return result
    except Exception as e:
        logger.error(f"خطأ في مراجعة الكود: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/math/solve")
async def solve_math(request: MathSolveRequest):
    """حل مسائل الرياضيات"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        result = await edrak_system.solve_math(request)
        return result
    except Exception as e:
        logger.error(f"خطأ في حل المسألة الرياضية: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/arabic/process")
async def process_arabic(request: ArabicProcessRequest):
    """معالجة النصوص العربية"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        result = await edrak_system.process_arabic(request)
        return result
    except Exception as e:
        logger.error(f"خطأ في معالجة النص العربي: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/learn/feedback")
async def learn_feedback(request: FeedbackRequest):
    """التعلم من تغذية راجعة البشر"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        result = await edrak_system.learn_from_feedback(request)
        return result
    except Exception as e:
        logger.error(f"خطأ في معالجة التغذية الراجعة: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """إنشاء Embeddings"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    try:
        # TODO: تنفيذ نظام الـ Embeddings
        return {
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        logger.error(f"خطأ في إنشاء embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats")
async def get_stats():
    """إحصائيات النظام"""
    if edrak_system is None:
        raise HTTPException(status_code=503, detail="النظام غير جاهز بعد")
    
    return edrak_system.get_stats()

# WebSocket للتفاعل المباشر
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket للدردشة المباشرة"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            request = ChatCompletionRequest(**data)
            
            async for chunk in edrak_system.chat_completion(request):
                await websocket.send_text(chunk)
    
    except WebSocketDisconnect:
        logger.info("انقطع اتصال WebSocket")
    except Exception as e:
        logger.error(f"خطأ في WebSocket: {str(e)}")
        await websocket.close()

# ═══════════════════════════════════════════════════════════════════════════════
# تشغيل التطبيق
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # إنشاء مجلد اللوجات
    Path("logs").mkdir(exist_ok=True)
    
    # تشغيل الخادم
    uvicorn.run(
        "main:app",
        host=EDRAK_CONFIG['api']['host'],
        port=EDRAK_CONFIG['api']['port'],
        workers=1,  # يجب أن يكون 1 للنماذج الكبيرة
        reload=False,
        log_level="info"
    )
