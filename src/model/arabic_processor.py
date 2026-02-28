"""
Arabic Processor - المعالج العربي المتقدم

يدعم:
- جميع لهجات العربية (مصرية، خليجية، شامية، مغربية، فصحى)
- الشعر العربي
- التصحيح النحوي
- تحليل المشاعر
- الترجمة من والى العربية
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════════
# قواعد وقواميس اللهجات العربية
# ═══════════════════════════════════════════════════════════════════════════════

ARABIC_DIALECTS = {
    "egyptian": {
        "name": "اللهجة المصرية",
        "patterns": {
            r"\b(انا|إنا)\b": "أنا",
            r"\b(انت|إنت)\b": "أنت",
            r"\b(انتي|إنتي)\b": "أنتِ",
            r"\b(احنا|إحنا)\b": "نحن",
            r"\b(انتو|إنتو)\b": "أنتم",
            r"\b(هم|هما)\b": "هم",
            r"\b(ده|دي)\b": "هذا/هذه",
            r"\b(كده|كدا)\b": "هكذا",
            r"\b(عايز|عاوز)\b": "أريد",
            r"\b(عايزة|عاوزة)\b": "أريد",
            r"\b(مش|ماش)\b": "لا",
            r"\b(ليه|لية)\b": "لماذا",
            r"\b(ازاي|إزاي)\b": "كيف",
            r"\b(فين|فين)\b": "أين",
            r"\b(امتى|إمتى)\b": "متى",
            r"\b(كام|قده)\b": "كم",
            r"\b(كتير|كتير)\b": "كثير",
            r"\b(شوية|شويه)\b": "قليل",
            r"\b(دلوقتي|دلوقت)\b": "الآن",
            r"\b(بكرة|بكره)\b": "غداً",
            r"\b(مبارح|امبارح)\b": "أمس",
            r"\b(دلوقتي|دلوقت)\b": "الآن",
            r"\b(هناك|هنا)\b": "هناك",
            r"\b(كويس|كويسة)\b": "جيد",
            r"\b(وحش|وحشة)\b": "سيء",
            r"\b(جميل|جميلة)\b": "جميل",
            r"\b(صعب|صعبة)\b": "صعب",
            r"\b(سهل|سهلة)\b": "سهل",
            r"\b(كبير|كبيرة)\b": "كبير",
            r"\b(صغير|صغيرة)\b": "صغير",
            r"\b(جديد|جديدة)\b": "جديد",
            r"\b(قديم|قديمة)\b": "قديم",
        },
        "suffixes": ["ة", "ين", "ون", "ات", "وا", "ي", "ك", "ه", "هم", "ها"],
        "common_phrases": [
            "يا عم", "يا باشا", "يا فندم", "يا سيدي", 
            "بص يا سيدي", "ماشي", "تمام", " oró", "خلاص",
            "يالا بينا", "استنى", "استني", "هات", "هاتي",
            "روح", "روحي", "تعالى", "تعالي", "شوف", "شوفي"
        ]
    },
    
    "gulf": {
        "name": "اللهجة الخليجية",
        "patterns": {
            r"\b(انا|أنا)\b": "أنا",
            r"\b(انت|أنت)\b": "أنت",
            r"\b(احنا|إحنا)\b": "نحن",
            r"\b(انتم|أنتم)\b": "أنتم",
            r"\b(هذا|هذي|هاذا|هاذي)\b": "هذا/هذه",
            r"\b(وش|ويش)\b": "ماذا",
            r"\b(وشو|ويشو)\b": "ماذا",
            r"\b(ليش|ليشو)\b": "لماذا",
            r"\b(كيف|كيفك)\b": "كيف",
            r"\b(وين|وينك)\b": "أين",
            r"\b(متى|متى)\b": "متى",
            r"\b(كم|قد)\b": "كم",
            r"\b(كثير|كثر)\b": "كثير",
            r"\b(شوي|شوية)\b": "قليل",
            r"\b(حين|الحين)\b": "الآن",
            r"\b(بكره|بكرة)\b": "غداً",
            r"\b(امس|امبارح)\b": "أمس",
            r"\b(هنا|هناك)\b": "هناك",
            r"\b(زين|طيب)\b": "جيد",
            r"\b(سيء|وحش)\b": "سيء",
            r"\b(حلو|جميل)\b": "جميل",
            r"\b(صعب|عسير)\b": "صعب",
            r"\b(سهل|ساهل)\b": "سهل",
            r"\b(كبير|عظيم)\b": "كبير",
            r"\b(صغير|صغير)\b": "صغير",
            r"\b(جديد|جديد)\b": "جديد",
            r"\b(قديم|قديم)\b": "قديم",
            r"\b(بس|بس)\b": "لكن",
            r"\b(يلا|يله)\b": "هيا",
            r"\b(خلاص|خلاص)\b": "انتهى",
        },
        "suffixes": ["ة", "ين", "ون", "ات", "وا", "ي", "ك", "ه", "هم", "ها", "نا"],
        "common_phrases": [
            "هلا والله", "هلا بك", "شلونك", "شلونچ", "شخبارك",
            "الله يعافيك", "يا هلا", "يا مرحبا", "حياك الله",
            "الله يحفظك", "ان شاء الله", "ما شاء الله",
            "يسلمك", "يعطيك العافية", "بالتوفيق"
        ]
    },
    
    "levantine": {
        "name": "اللهجة الشامية (الlevantine)",
        "patterns": {
            r"\b(انا|أنا)\b": "أنا",
            r"\b(انت|إنت)\b": "أنت",
            r"\b(انتي|إنتي)\b": "أنتِ",
            r"\b(نحنا|إحنا)\b": "نحن",
            r"\b(انتو|إنتو)\b": "أنتم",
            r"\b(هيدا|هاد|هاي)\b": "هذا/هذه",
            r"\b(شو|شو)\b": "ماذا",
            r"\b(ليش|لش)\b": "لماذا",
            r"\b(كيف|كيف)\b": "كيف",
            r"\b(وين|وين)\b": "أين",
            r"\b(إمتى|امتى)\b": "متى",
            r"\b(قديش|قد ايش)\b": "كم",
            r"\b(كتير|كتير)\b": "كثير",
            r"\b(شوي|شوية)\b": "قليل",
            r"\b(هلق|هلقيت)\b": "الآن",
            r"\b(بكرا|بكرة)\b": "غداً",
            r"\b(مبارح|امبارح)\b": "أمس",
            r"\b(منيح|منيحة)\b": "جيد",
            r"\b(مو منيح|مش منيح)\b": "سيء",
            r"\b(حلو|حلوة)\b": "جميل",
            r"\b(صعب|صعبة)\b": "صعب",
            r"\b(سهل|سهلة)\b": "سهل",
            r"\b(كبير|كبيرة)\b": "كبير",
            r"\b(صغير|صغيرة)\b": "صغير",
            r"\b(جديد|جديدة)\b": "جديد",
            r"\b(قديم|قديمة)\b": "قديم",
            r"\b(بس|بس)\b": "لكن",
            r"\b(يلا|يلا)\b": "هيا",
        },
        "suffixes": ["ة", "ين", "ون", "ات", "وا", "ي", "ك", "ه", "هم", "ها", "نا"],
        "common_phrases": [
            "شو اخبارك", "كيف صحتك", "الله معك", "يعطيك العافية",
            "صباح الخير", "تصبح على خير", "اهلا وسهلا", "حياك الله",
            "الله يسعدك", "ان شاء الله", "ما شاء الله"
        ]
    },
    
    "maghrebi": {
        "name": "اللهجة المغربية (الدارجة)",
        "patterns": {
            r"\b(انا|ننا)\b": "أنا",
            r"\b(نتا|نتي)\b": "أنت/أنتِ",
            r"\b(حنا|حنا)\b": "نحن",
            r"\b(نتوما|نتما)\b": "أنتم",
            r"\b(هادا|هادي)\b": "هذا/هذه",
            r"\b(اش\b|اشنو)\b": "ماذا",
            r"\b(علاش|علاش)\b": "لماذا",
            r"\b(كيفاش|كيف)\b": "كيف",
            r"\b(فين|فين)\b": "أين",
            r"\b(فوقاش|فوقاش)\b": "متى",
            r"\b(شحال|شحال)\b": "كم",
            r"\b(بزاف|بزاف)\b": "كثير",
            r"\b(شوية|شويا)\b": "قليل",
            r"\b(دابا|دابا)\b": "الآن",
            r"\b(غدا|غدا)\b": "غداً",
            r"\b(البارح|لبارح)\b": "أمس",
            r"\b(مزيان|مزيانة)\b": "جيد",
            r"\b(خايب|خايبة)\b": "سيء",
            r"\b(زوين|زوينة)\b": "جميل",
            r"\b(صعيب|صعيبة)\b": "صعب",
            r"\b(ساهل|ساهلة)\b": "سهل",
            r"\b(كبير|كبيرة)\b": "كبير",
            r"\b(صغير|صغيرة)\b": "صغير",
            r"\b(جديد|جديدة)\b": "جديد",
            r"\b(قديم|قديمة)\b": "قديم",
            r"\b(غير|غير)\b": "لكن",
            r"\b(يلا|يلا)\b": "هيا",
        },
        "suffixes": ["ة", "ين", "ون", "ات", "وا", "ي", "ك", "ه", "هم", "ها", "نا"],
        "common_phrases": [
            "كيف داير", "لا باس", "بخير", "الحمد لله",
            "صباح الخير", "مساء الخير", "الله يحفظك",
            "برك الله فيك", "الله يعطيك الصحة", "ان شاء الله"
        ]
    },
    
    "msa": {
        "name": "الفصحى (العربية المعيارية الحديثة)",
        "patterns": {},
        "suffixes": ["ة", "ين", "ون", "ات", "وا", "ي", "ك", "ه", "هم", "ها", "نا"],
        "common_phrases": [
            "كيف حالك", "ما اسمك", "من أين أنت", "أهلاً وسهلاً",
            "صباح الخير", "مساء الخير", "تصبح على خير",
            "شكراً لك", "عفواً", "مع السلامة"
        ]
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# أوزان الشعر العربي
# ═══════════════════════════════════════════════════════════════════════════════

POETRY_METERS = {
    "طويل": {"pattern": "فعولن مفاعيلن فعولن مفاعلن", "feet": 4},
    "مديد": {"pattern": "فاعلاتن فاعلاتن فاعلاتن", "feet": 3},
    "بسيط": {"pattern": "مستفعلن فاعلن مستفعلن فاعلن", "feet": 4},
    "وافر": {"pattern": "مفاعلتن مفاعلتن فعولن", "feet": 3},
    "كامل": {"pattern": "متفاعلن متفاعلن متفاعلن", "feet": 3},
    "هزج": {"pattern": "مفاعيلن مفاعيلن", "feet": 2},
    "رجز": {"pattern": "مستفعلن مستفعلن مستفعلن", "feet": 3},
    "رمل": {"pattern": "فاعلاتن فاعلاتن فاعلاتن", "feet": 3},
    "سريع": {"pattern": "مستفعلن مستفعلن فاعلن", "feet": 3},
    "منسرح": {"pattern": "مستفعلن مافاعيلن مستفعلن", "feet": 3},
    "خبب": {"pattern": "فاعلاتن فاعلاتن", "feet": 2},
    "مجتث": {"pattern": "مستفعلن فاعلاتن مستفعلن", "feet": 3},
    "متقارب": {"pattern": "فعولن فعولن فعولن فعول", "feet": 4},
    "محذوف": {"pattern": "فاعلاتن فاعلاتن فاعلن", "feet": 3},
    "مضارع": {"pattern": "مفاعلن فاعلاتن مفاعلن", "feet": 3},
    "مقتضب": {"pattern": "مفاعيلن فاعلاتن", "feet": 2},
    "متدارك": {"pattern": "فاعلن فاعلن فاعلن فاعلن", "feet": 4},
}

# ═══════════════════════════════════════════════════════════════════════════════
# المعالج العربي الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class ArabicProcessor:
    """المعالج العربي المتقدم لـ Edrak AI"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.dialects = ARABIC_DIALECTS
        self.poetry_meters = POETRY_METERS
        
        # تحميل النموذج إذا كان متوفراً
        self.model = None
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """تحميل النموذج المتخصص"""
        try:
            # TODO: تحميل النموذج الفعلي
            pass
        except Exception as e:
            print(f"Warning: Could not load Arabic model: {e}")
    
    async def process(
        self,
        text: str,
        task: str = "understand",
        dialect: Optional[str] = None,
        target_language: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        معالجة النص العربي
        
        Args:
            text: النص المراد معالجته
            task: المهمة (understand, correct, translate, poetry, sentiment)
            dialect: اللهجة المحددة
            target_language: لغة الترجمة المستهدفة
            context: السياق السابق
        """
        
        if task == "understand":
            return await self._understand(text, dialect, context)
        elif task == "correct":
            return await self._correct(text)
        elif task == "translate":
            return await self._translate(text, target_language or "english")
        elif task == "poetry":
            return await self._analyze_poetry(text)
        elif task == "sentiment":
            return await self._analyze_sentiment(text)
        elif task == "respond":
            return await self._generate_response(text, dialect, context)
        else:
            return {"error": f"Unknown task: {task}"}
    
    async def _understand(
        self, 
        text: str, 
        dialect: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> Dict:
        """فهم النص العربي"""
        
        # اكتشاف اللهجة
        detected_dialect = dialect or self._detect_dialect(text)
        
        # تحليل الكيانات
        entities = self._extract_entities(text)
        
        # تحليل النية
        intent = self._analyze_intent(text)
        
        # تحليل المشاعر
        sentiment = self._analyze_sentiment_quick(text)
        
        return {
            "text": text,
            "dialect": detected_dialect,
            "dialect_name": self.dialects.get(detected_dialect, {}).get("name", "غير معروف"),
            "entities": entities,
            "intent": intent,
            "sentiment": sentiment,
            "normalized": self._normalize_to_msa(text, detected_dialect),
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": 0, "total_tokens": len(text.split())}
        }
    
    async def _correct(self, text: str) -> Dict:
        """تصحيح النص العربي"""
        
        # تصحيح الأخطاء الإملائية
        spelling_corrected = self._correct_spelling(text)
        
        # تصحيح النحو
        grammar_corrected = self._correct_grammar(spelling_corrected)
        
        # تصحيح الإعراب
        iarab_corrected = self._correct_iarab(grammar_corrected)
        
        return {
            "original": text,
            "corrected": iarab_corrected,
            "spelling_corrections": self._get_spelling_corrections(text, spelling_corrected),
            "grammar_corrections": self._get_grammar_corrections(spelling_corrected, grammar_corrected),
            "confidence": 0.95,
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": len(iarab_corrected.split()), "total_tokens": len(text.split()) + len(iarab_corrected.split())}
        }
    
    async def _translate(self, text: str, target_language: str) -> Dict:
        """الترجمة من/إلى العربية"""
        
        # اكتشاف لغة المصدر
        source_language = self._detect_language(text)
        
        if source_language == "arabic":
            # ترجمة من العربية
            translated = self._translate_from_arabic(text, target_language)
        else:
            # ترجمة إلى العربية
            translated = self._translate_to_arabic(text, source_language)
        
        return {
            "original": text,
            "translated": translated,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": 0.92,
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": len(translated.split()), "total_tokens": len(text.split()) + len(translated.split())}
        }
    
    async def _analyze_poetry(self, text: str) -> Dict:
        """تحليل الشعر العربي"""
        
        # اكتشاف البحر
        meter = self._detect_poetry_meter(text)
        
        # تحليل القافية
        rhyme = self._detect_rhyme(text)
        
        # تحليل الوزن
        scansion = self._analyze_scansion(text)
        
        # عدد التفعيلات
        feet_count = self._count_feet(text, meter)
        
        return {
            "text": text,
            "meter": meter,
            "meter_name": self.poetry_meters.get(meter, {}).get("pattern", "غير معروف"),
            "rhyme": rhyme,
            "scansion": scansion,
            "feet_count": feet_count,
            "is_valid": self._validate_poetry(text, meter),
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": 0, "total_tokens": len(text.split())}
        }
    
    async def _analyze_sentiment(self, text: str) -> Dict:
        """تحليل المشاعر المتقدم"""
        
        quick_sentiment = self._analyze_sentiment_quick(text)
        
        # تحليل أعمق
        emotions = self._detect_emotions(text)
        
        # تحليل النبرة
        tone = self._detect_tone(text)
        
        return {
            "text": text,
            "sentiment": quick_sentiment,
            "emotions": emotions,
            "tone": tone,
            "intensity": self._calculate_intensity(text),
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": 0, "total_tokens": len(text.split())}
        }
    
    async def _generate_response(
        self, 
        text: str, 
        dialect: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> Dict:
        """توليد رد باللهجة المطلوبة"""
        
        detected_dialect = dialect or self._detect_dialect(text)
        
        # فهم السؤال/الطلب
        understanding = await self._understand(text, detected_dialect, context)
        
        # توليد الرد
        response = self._generate_in_dialect(understanding, detected_dialect, context)
        
        return {
            "text": response,
            "dialect": detected_dialect,
            "sentiment": "helpful",
            "entities": understanding.get("entities", []),
            "usage": {"prompt_tokens": len(text.split()), "completion_tokens": len(response.split()), "total_tokens": len(text.split()) + len(response.split())}
        }
    
    def _detect_dialect(self, text: str) -> str:
        """اكتشاف اللهجة العربية"""
        
        scores = {dialect: 0 for dialect in self.dialects.keys()}
        text_lower = text.lower()
        
        for dialect, data in self.dialects.items():
            # فحص الأنماط
            for pattern in data.get("patterns", {}).keys():
                if re.search(pattern, text_lower):
                    scores[dialect] += 1
            
            # فحص العبارات الشائعة
            for phrase in data.get("common_phrases", []):
                if phrase.lower() in text_lower:
                    scores[dialect] += 2
        
        # إرجاع اللهجة الأعلى درجة
        max_dialect = max(scores, key=scores.get)
        return max_dialect if scores[max_dialect] > 0 else "msa"
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """استخراج الكيانات المسماة"""
        
        entities = []
        
        # أسماء الأشخاص
        person_pattern = r"\b(محمد|أحمد|علي|عمر|خالد|يوسف|ياسين|إدريس|فاطمة|عائشة|خديجة|مريم|ليلى|سارة|نور|ريم)\b"
        for match in re.finditer(person_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PERSON",
                "start": match.start(),
                "end": match.end()
            })
        
        # أماكن
        place_pattern = r"\b(القاهرة|الرياض|دبي|بغداد|دمشق|بيروت|تونس|الرباط|الجزائر|طرابلس|الخرطوم|صنعاء|مسقط|المنامة|الكويت|الدوحة|عمان|عمان|القدس|غزة|الخليل|نابلس|القدس)\b"
        for match in re.finditer(place_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "LOCATION",
                "start": match.start(),
                "end": match.end()
            })
        
        # تواريخ
        date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|يوم\s+\w+|شهر\s+\w+|سنة\s+\d{4})\b"
        for match in re.finditer(date_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "DATE",
                "start": match.start(),
                "end": match.end()
            })
        
        return entities
    
    def _analyze_intent(self, text: str) -> Dict:
        """تحليل نية المستخدم"""
        
        text_lower = text.lower()
        
        intents = {
            "question": ["ما", "ماذا", "كيف", "لماذا", "لم", "أين", "متى", "من", "هل", "أ"],
            "command": ["اعمل", "اكتب", "حل", "شرح", "فسر", "ترجم", "حول"],
            "greeting": ["مرحبا", "السلام", "صباح", "مساء", "أهلا", "هلا"],
            "gratitude": ["شكرا", "أشكرك", "متشكر", "يسلمك", "جزاك"],
            "farewell": ["مع السلامة", "في أمان", "إلى اللقاء", "باي"],
            "request": ["عايز", "عاوز", "أريد", "أحب", "ممكن", "لو سمحت"]
        }
        
        scores = {intent: 0 for intent in intents}
        
        for intent, keywords in intents.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1
        
        max_intent = max(scores, key=scores.get)
        confidence = scores[max_intent] / max(sum(scores.values()), 1)
        
        return {
            "intent": max_intent if scores[max_intent] > 0 else "unknown",
            "confidence": confidence,
            "all_scores": scores
        }
    
    def _analyze_sentiment_quick(self, text: str) -> str:
        """تحليل سريع للمشاعر"""
        
        positive_words = [
            "جميل", "رائع", "ممتاز", "جيد", "حلو", "كويس", "ممتاز", "عظيم",
            "مبهج", "سعيد", "فرحان", "مبسوط", "مسرور", "متحمس", "فخور",
            "حب", "أحب", "يعجبني", "فضل", "أفضل", "ممتن", "شاكر"
        ]
        
        negative_words = [
            "سيء", "وحش", "مزعج", "مؤلم", "محبط", "حزين", "زعلان", "متضايق",
            "غاضب", "عصبي", "منزعج", "خايب", "مقرف", "ممل", "صعب", "معقد"
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _normalize_to_msa(self, text: str, dialect: str) -> str:
        """تحويل النص إلى الفصحى"""
        
        normalized = text
        
        if dialect in self.dialects:
            patterns = self.dialects[dialect].get("patterns", {})
            for pattern, replacement in patterns.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _correct_spelling(self, text: str) -> str:
        """تصحيح الأخطاء الإملائية"""
        
        # قائمة الأخطاء الشائعة
        common_errors = {
            "الى": "إلى",
            "انه": "أنه",
            "انها": "أنها",
            "هذا": "هذا",
            "هذه": "هذه",
            "اللذي": "الذي",
            "اللتي": "التي",
        }
        
        corrected = text
        for error, correction in common_errors.items():
            corrected = corrected.replace(error, correction)
        
        return corrected
    
    def _correct_grammar(self, text: str) -> str:
        """تصحيح الأخطاء النحوية"""
        
        # قواعد نحوية أساسية
        # TODO: تنفيذ قواعد أكثر تقدماً
        
        return text
    
    def _correct_iarab(self, text: str) -> str:
        """تصحيح الإعراب"""
        
        # TODO: تنفيذ تصحيح الإعراب
        
        return text
    
    def _get_spelling_corrections(self, original: str, corrected: str) -> List[Dict]:
        """استخراج التصحيحات الإملائية"""
        
        corrections = []
        orig_words = original.split()
        corr_words = corrected.split()
        
        for i, (orig, corr) in enumerate(zip(orig_words, corr_words)):
            if orig != corr:
                corrections.append({
                    "original": orig,
                    "corrected": corr,
                    "position": i,
                    "type": "spelling"
                })
        
        return corrections
    
    def _get_grammar_corrections(self, original: str, corrected: str) -> List[Dict]:
        """استخراج التصحيحات النحوية"""
        
        # TODO: تنفيذ استخراج التصحيحات النحوية
        
        return []
    
    def _detect_language(self, text: str) -> str:
        """اكتشاف لغة النص"""
        
        # فحص الحروف العربية
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        if arabic_pattern.search(text):
            return "arabic"
        
        # فحص الحروف اللاتينية
        latin_pattern = re.compile(r'[a-zA-Z]')
        if latin_pattern.search(text):
            return "english"
        
        return "unknown"
    
    def _translate_from_arabic(self, text: str, target_language: str) -> str:
        """الترجمة من العربية"""
        
        # TODO: تنفيذ الترجمة الفعلية
        
        return f"[Translated to {target_language}]: {text}"
    
    def _translate_to_arabic(self, text: str, source_language: str) -> str:
        """الترجمة إلى العربية"""
        
        # TODO: تنفيذ الترجمة الفعلية
        
        return f"[مترجم من {source_language}]: {text}"
    
    def _detect_poetry_meter(self, text: str) -> str:
        """اكتشاف بحر الشعر"""
        
        # TODO: تنفيذ اكتشاف البحر الشعري
        
        return "unknown"
    
    def _detect_rhyme(self, text: str) -> str:
        """اكتشاف القافية"""
        
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return ""
        
        # استخراج الحرف الأخير من كل بيت
        last_chars = [line.strip()[-1] if line.strip() else '' for line in lines]
        
        # البحث عن التكرار
        from collections import Counter
        char_counts = Counter(last_chars)
        most_common = char_counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        
        return ""
    
    def _analyze_scansion(self, text: str) -> str:
        """تحليل الوزن العروضي"""
        
        # TODO: تنفيذ تحليل الوزن
        
        return ""
    
    def _count_feet(self, text: str, meter: str) -> int:
        """عدد التفعيلات"""
        
        meter_info = self.poetry_meters.get(meter, {})
        return meter_info.get("feet", 0)
    
    def _validate_poetry(self, text: str, meter: str) -> bool:
        """التحقق من صحة الشعر"""
        
        # TODO: تنفيذ التحقق
        
        return True
    
    def _detect_emotions(self, text: str) -> List[str]:
        """اكتشاف المشاعر المتعددة"""
        
        emotions = {
            "joy": ["سعيد", "فرح", "مبسوط", "مسرور", "متحمس"],
            "sadness": ["حزين", "زعلان", "محبط", "مكتئب"],
            "anger": ["غاضب", "عصبي", "منزعج", "متضايق"],
            "fear": ["خائف", "قلق", "متردد", "خجلان"],
            "surprise": ["مندهش", "متفاجئ", "صدم"],
            "love": ["حب", "عشق", "هيام", "غرام"],
            "disgust": ["مقرف", "م disgusted", "منزعج"]
        }
        
        detected = []
        text_lower = text.lower()
        
        for emotion, keywords in emotions.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(emotion)
                    break
        
        return detected
    
    def _detect_tone(self, text: str) -> str:
        """اكتشاف النبرة"""
        
        tones = {
            "formal": ["سيدي", "فضلك", "لو سمحت", "أرجو", "أطلب"],
            "informal": ["يا", "أنت", "إنت", "شوف", "بص"],
            "poetic": ["يا", "أيها", "هيا", "ألا"],
            "technical": ["نظام", "برنامج", "كود", "دالة", "متغير"]
        }
        
        scores = {tone: 0 for tone in tones}
        text_lower = text.lower()
        
        for tone, keywords in tones.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[tone] += 1
        
        max_tone = max(scores, key=scores.get)
        return max_tone if scores[max_tone] > 0 else "neutral"
    
    def _calculate_intensity(self, text: str) -> float:
        """حساب شدة المشاعر"""
        
        # علامات التعجب
        exclamation_count = text.count('!')
        
        # الكلمات المؤكدة
        intensifiers = ["جدا", "كثير", "للغاية", "بشدة", "تماما"]
        intensifier_count = sum(text.lower().count(iw) for iw in intensifiers)
        
        # حساب الشدة
        intensity = min((exclamation_count * 0.2) + (intensifier_count * 0.15), 1.0)
        
        return intensity
    
    def _generate_in_dialect(
        self, 
        understanding: Dict, 
        dialect: str,
        context: Optional[List[Dict]] = None
    ) -> str:
        """توليد رد باللهجة المحددة"""
        
        # بناء الرد بناءً على الفهم
        intent = understanding.get("intent", {}).get("intent", "unknown")
        entities = understanding.get("entities", [])
        sentiment = understanding.get("sentiment", "neutral")
        
        # قوالب ردود حسب اللهجة
        responses = {
            "egyptian": {
                "greeting": "أهلاً بيك! إزيك؟",
                "question": "أنا هنا عشان أساعدك، إسأل براحتك.",
                "gratitude": "العفو يا فندم! أي حاجة تانية؟",
                "farewell": "مع السلامة! متنساش ترجع تاني.",
                "default": "تمام، فهمتك. خليني أساعدك."
            },
            "gulf": {
                "greeting": "هلا والله! كيف حالك؟",
                "question": "أنا حاضر أساعدك، تفضل اسأل.",
                "gratitude": "الله يعافيك! شي ثاني؟",
                "farewell": "مع السلامة! تراكم على خير.",
                "default": "زين، فهمت عليك. خليني أساعدك."
            },
            "levantine": {
                "greeting": "أهلاً وسهلاً! كيفك؟",
                "question": "أنا جاهز أساعدك، إسألني.",
                "gratitude": "العفو! شي تاني بدك إياه؟",
                "farewell": "مع السلامة! الله معك.",
                "default": "منيح، فهمت عليك. خليني ساعدك."
            },
            "maghrebi": {
                "greeting": "مرحبا بيك! كيف داير؟",
                "question": "أنا هنا باش نعاونك، سول براحتك.",
                "gratitude": "العفو! واخرا؟",
                "farewell": "السلامة! نتلاقاو.",
                "default": "مزيان، فهمتك. خليني نعاونك."
            },
            "msa": {
                "greeting": "أهلاً وسهلاً! كيف حالك؟",
                "question": "أنا هنا لمساعدتك، فلا تتردد في السؤال.",
                "gratitude": "العفو! هل تحتاج مساعدة أخرى؟",
                "farewell": "مع السلامة! إلى اللقاء.",
                "default": "حسناً، فهمت طلبك. دعني أساعدك."
            }
        }
        
        dialect_responses = responses.get(dialect, responses["msa"])
        response = dialect_responses.get(intent, dialect_responses["default"])
        
        # إضافة تفاصيل من الكيانات
        if entities:
            entity_names = [e["text"] for e in entities[:3]]
            response += f" (بتكلم عن: {', '.join(entity_names)})"
        
        return response
