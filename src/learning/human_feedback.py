"""
Human Feedback Learner - نظام التعلم من تغذية راجعة البشر

هذا النظام يتيح لـ Edrak AI:
- تلقي التغذية الراجعة من المستخدمين
- التعلم من التصحيحات
- تحسين الأداء مع الوقت
- تخزين المعرفة المكتسبة
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# نظام التعلم من البشر
# ═══════════════════════════════════════════════════════════════════════════════

class HumanFeedbackLearner:
    """نظام التعلم من تغذية راجعة البشر"""
    
    def __init__(self, storage_path: str = "data/feedback"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # قاعدة بيانات التغذية الراجعة
        self.feedback_db = []
        self.corrections_db = {}
        
        # تحميل البيانات المخزنة
        self._load_data()
    
    def _load_data(self):
        """تحميل البيانات المخزنة"""
        
        feedback_file = self.storage_path / "feedback.json"
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                self.feedback_db = json.load(f)
        
        corrections_file = self.storage_path / "corrections.json"
        if corrections_file.exists():
            with open(corrections_file, 'r', encoding='utf-8') as f:
                self.corrections_db = json.load(f)
    
    def _save_data(self):
        """حفظ البيانات"""
        
        feedback_file = self.storage_path / "feedback.json"
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_db, f, ensure_ascii=False, indent=2)
        
        corrections_file = self.storage_path / "corrections.json"
        with open(corrections_file, 'w', encoding='utf-8') as f:
            json.dump(self.corrections_db, f, ensure_ascii=False, indent=2)
    
    async def process_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback_type: str,
        comment: Optional[str] = None,
        correction: Optional[str] = None,
        rating: Optional[int] = None
    ) -> bool:
        """
        معالجة التغذية الراجعة من المستخدم
        
        Args:
            user_id: معرف المستخدم
            interaction_id: معرف التفاعل
            feedback_type: نوع التغذية (positive, negative, correction)
            comment: تعليق إضافي
            correction: تصحيح
            rating: تقييم (1-5)
        """
        
        feedback_entry = {
            "id": self._generate_id(),
            "user_id": user_id,
            "interaction_id": interaction_id,
            "feedback_type": feedback_type,
            "comment": comment,
            "correction": correction,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        # إضافة إلى قاعدة البيانات
        self.feedback_db.append(feedback_entry)
        
        # معالجة التصحيح
        if correction:
            await self._learn_correction(interaction_id, correction)
        
        # تحديث الإحصائيات
        self._update_statistics(feedback_entry)
        
        # حفظ البيانات
        self._save_data()
        
        return True
    
    async def _learn_correction(self, interaction_id: str, correction: str):
        """التعلم من التصحيح"""
        
        # تخزين التصحيح
        self.corrections_db[interaction_id] = {
            "correction": correction,
            "learned_at": datetime.now().isoformat(),
            "times_applied": 0
        }
        
        # TODO: تطبيق التعلم الفعلي على النموذج
        # هذا يتطلب تدريب إضافي أو تحديث للأوزان
    
    def _update_statistics(self, feedback_entry: Dict):
        """تحديث إحصائيات التغذية الراجعة"""
        
        # يمكن إضافة إحصائيات أكثر تفصيلاً هنا
        pass
    
    def get_correction(self, interaction_id: str) -> Optional[str]:
        """الحصول على تصحيح لتفاعل معين"""
        
        correction_data = self.corrections_db.get(interaction_id)
        if correction_data:
            correction_data["times_applied"] += 1
            self._save_data()
            return correction_data["correction"]
        
        return None
    
    def get_similar_corrections(self, query: str, top_k: int = 5) -> List[Dict]:
        """الحصول على تصحيحات مشابهة"""
        
        # TODO: تنفيذ البحث عن التصحيحات المشابهة
        
        return []
    
    def get_statistics(self) -> Dict:
        """الحصول على إحصائيات التعلم"""
        
        total_feedback = len(self.feedback_db)
        positive_feedback = sum(1 for f in self.feedback_db if f["feedback_type"] == "positive")
        negative_feedback = sum(1 for f in self.feedback_db if f["feedback_type"] == "negative")
        corrections_count = len(self.corrections_db)
        
        avg_rating = 0
        ratings = [f["rating"] for f in self.feedback_db if f["rating"] is not None]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
        
        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "corrections_learned": corrections_count,
            "average_rating": round(avg_rating, 2),
            "satisfaction_rate": round(positive_feedback / total_feedback * 100, 2) if total_feedback > 0 else 0
        }
    
    def _generate_id(self) -> str:
        """توليد معرف فريد"""
        
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# نظام التصحيح الذاتي
# ═══════════════════════════════════════════════════════════════════════════════

class SelfCorrection:
    """نظام التصحيح الذاتي لـ Edrak AI"""
    
    def __init__(self):
        self.correction_rules = self._load_correction_rules()
        self.learned_corrections = {}
    
    def _load_correction_rules(self) -> Dict:
        """تحميل قواعد التصحيح"""
        
        return {
            "arabic": {
                "common_errors": {
                    "الى": "إلى",
                    "انه": "أنه",
                    "انها": "أنها",
                    "الذي": "الذي",
                    "التي": "التي",
                },
                "grammar_rules": [
                    # قواعد نحوية
                ]
            },
            "code": {
                "common_errors": {
                    "==": "===",  # في JavaScript
                    "!=": "!==",
                },
                "best_practices": [
                    # أفضل الممارسات
                ]
            },
            "math": {
                "common_errors": {
                    # أخطاء رياضية شائعة
                }
            }
        }
    
    async def correct(self, text: str, context: str = "general") -> str:
        """
        تصحيح النص تلقائياً
        
        Args:
            text: النص المراد تصحيحه
            context: سياق النص (arabic, code, math, general)
        """
        
        corrected = text
        
        # تطبيق قواعد التصحيح
        if context in self.correction_rules:
            rules = self.correction_rules[context]
            
            # تصحيح الأخطاء الشائعة
            if "common_errors" in rules:
                for error, correction in rules["common_errors"].items():
                    corrected = corrected.replace(error, correction)
        
        # تطبيق التصحيحات المكتسبة
        for pattern, replacement in self.learned_corrections.items():
            corrected = corrected.replace(pattern, replacement)
        
        return corrected
    
    async def learn_correction(self, original: str, corrected: str, context: str = "general") -> bool:
        """
        تعلم تصحيح جديد
        
        Args:
            original: النص الأصلي
            corrected: النص المصحح
            context: السياق
        """
        
        # تخزين التصحيح
        self.learned_corrections[original] = corrected
        
        # TODO: تحديث قواعد التصحيح بشكل أكثر ذكاء
        
        return True
    
    def get_learned_corrections(self) -> Dict:
        """الحصول على التصحيحات المكتسبة"""
        
        return self.learned_corrections.copy()
