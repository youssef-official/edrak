"""
Context Manager - مدير السياق

يدير:
- سياق المحادثات
- الذاكرة قصيرة المدى
- الذاكرة طويلة المدى
- تفضيلات المستخدمين
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# مدير السياق
# ═══════════════════════════════════════════════════════════════════════════════

class ContextManager:
    """مدير السياق للمحادثات"""
    
    def __init__(self, max_context_length: int = 4096):
        self.max_context_length = max_context_length
        self.conversations: Dict[str, List[Dict]] = {}
        self.user_preferences: Dict[str, Dict] = {}
        self.short_term_memory: List[Dict] = []
        self.max_short_term = 10000
    
    def build_context(
        self,
        messages: List[Dict],
        user_id: Optional[str] = None,
        max_tokens: int = 4096
    ) -> Dict:
        """
        بناء السياق للمحادثة
        
        Args:
            messages: قائمة الرسائل
            user_id: معرف المستخدم
            max_tokens: الحد الأقصى للتوكنات
        """
        
        # استرجاع سياق المستخدم
        user_context = []
        if user_id and user_id in self.conversations:
            user_context = self.conversations[user_id][-5:]  # آخر 5 تفاعلات
        
        # دمج السياق
        full_context = user_context + messages
        
        # تقليل السياق إذا لزم الأمر
        while len(str(full_context)) > max_tokens * 4 and len(full_context) > 1:
            full_context.pop(0)
        
        # استخراج الرسالة الأخيرة
        last_message = messages[-1]["content"] if messages else ""
        
        # تحويل إلى تنسيق مناسب للنموذج
        # TODO: تنفيذ التحويل الفعلي باستخدام الـ Tokenizer
        
        return {
            "messages": full_context,
            "last_message": last_message,
            "input_ids": None,  # TODO: تحويل إلى IDs
            "attention_mask": None,  # TODO: إنشاء قناع الانتباه
            "history": user_context
        }
    
    def add_to_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str
    ):
        """إضافة تفاعل إلى سجل المحادثة"""
        
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response
        }
        
        self.conversations[user_id].append(interaction)
        
        # الاحتفاظ فقط بآخر 100 تفاعل
        if len(self.conversations[user_id]) > 100:
            self.conversations[user_id] = self.conversations[user_id][-100:]
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """الحصول على سجل المحادثة"""
        
        if user_id not in self.conversations:
            return []
        
        return self.conversations[user_id][-limit:]
    
    def clear_conversation(self, user_id: str):
        """مسح سجل المحادثة"""
        
        if user_id in self.conversations:
            self.conversations[user_id] = []
    
    def set_user_preference(self, user_id: str, key: str, value: Any):
        """تعيين تفضيل المستخدم"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id][key] = {
            "value": value,
            "set_at": datetime.now().isoformat()
        }
    
    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """الحصول على تفضيل المستخدم"""
        
        if user_id in self.user_preferences:
            pref = self.user_preferences[user_id].get(key)
            if pref:
                return pref["value"]
        
        return default
    
    def add_to_short_term_memory(self, item: Dict):
        """إضافة إلى الذاكرة قصيرة المدى"""
        
        item["timestamp"] = datetime.now().isoformat()
        self.short_term_memory.append(item)
        
        # الاحتفاظ فقط بأحدث العناصر
        if len(self.short_term_memory) > self.max_short_term:
            self.short_term_memory = self.short_term_memory[-self.max_short_term:]
    
    def search_short_term_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """البحث في الذاكرة قصيرة المدى"""
        
        # TODO: تنفيذ البحث الفعلي
        
        return self.short_term_memory[-top_k:]
    
    def get_relevant_context(self, query: str, user_id: Optional[str] = None) -> List[Dict]:
        """الحصول على سياق ذو صلة"""
        
        relevant = []
        
        # البحث في محادثات المستخدم
        if user_id and user_id in self.conversations:
            for interaction in reversed(self.conversations[user_id][-10:]):
                if any(word in interaction["user_message"] for word in query.split()):
                    relevant.append(interaction)
        
        # البحث في الذاكرة قصيرة المدى
        memory_results = self.search_short_term_memory(query)
        relevant.extend(memory_results)
        
        return relevant[:10]


# ═══════════════════════════════════════════════════════════════════════════════
# نظام الذاكرة طويلة المدى
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermMemory:
    """نظام الذاكرة طويلة المدى"""
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.facts: Dict[str, Dict] = {}
        self.concepts: Dict[str, Dict] = {}
        
        self._load_data()
    
    def _load_data(self):
        """تحميل البيانات"""
        
        facts_file = self.storage_path / "facts.json"
        if facts_file.exists():
            with open(facts_file, 'r', encoding='utf-8') as f:
                self.facts = json.load(f)
        
        concepts_file = self.storage_path / "concepts.json"
        if concepts_file.exists():
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.concepts = json.load(f)
    
    def _save_data(self):
        """حفظ البيانات"""
        
        facts_file = self.storage_path / "facts.json"
        with open(facts_file, 'w', encoding='utf-8') as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        
        concepts_file = self.storage_path / "concepts.json"
        with open(concepts_file, 'w', encoding='utf-8') as f:
            json.dump(self.concepts, f, ensure_ascii=False, indent=2)
    
    def store_fact(self, fact: str, category: str = "general", confidence: float = 1.0):
        """تخزين حقيقة"""
        
        fact_id = hash(fact) % 10000000
        
        self.facts[str(fact_id)] = {
            "fact": fact,
            "category": category,
            "confidence": confidence,
            "learned_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        
        self._save_data()
    
    def retrieve_fact(self, query: str) -> Optional[str]:
        """استرجاع حقيقة"""
        
        # TODO: تنفيذ البحث الفعلي
        
        for fact_id, fact_data in self.facts.items():
            if query.lower() in fact_data["fact"].lower():
                fact_data["access_count"] += 1
                fact_data["last_accessed"] = datetime.now().isoformat()
                self._save_data()
                return fact_data["fact"]
        
        return None
    
    def store_concept(self, concept: str, definition: str, related: List[str] = None):
        """تخزين مفهوم"""
        
        self.concepts[concept] = {
            "definition": definition,
            "related": related or [],
            "learned_at": datetime.now().isoformat()
        }
        
        self._save_data()
    
    def get_concept(self, concept: str) -> Optional[Dict]:
        """الحصول على مفهوم"""
        
        return self.concepts.get(concept)
