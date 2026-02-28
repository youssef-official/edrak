"""
Safety Checker - فاحص الأمان

يفحص:
- المحتوى الضار
- الأكواد الخبيثة
- التحيز
- المحتوى العربي غير اللائق
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# قوائم الكلمات والأنماط
# ═══════════════════════════════════════════════════════════════════════════════

# كلمات ممنوعة عالمية
GLOBAL_BLOCKED_WORDS = [
    # العنف
    "kill", "murder", "attack", "bomb", "terrorist", "terrorism",
    "قتل", "اغتيال", "هجوم", "قنبلة", "إرهابي", "إرهاب",
    
    # الكراهية
    "hate speech", "racist", "nazi", "supremacy",
    "خطاب كراهية", "عنصري", "نازي", "تفوق",
    
    # المحتوى الجنسي
    "porn", "sexual", "explicit",
    "إباحي", "جنسي", "فاضح",
    
    # الاحتيال
    "scam", "fraud", "phishing",
    "احتيال", "نصب", "تصيد",
]

# أنماط أكواد خطرة
DANGEROUS_CODE_PATTERNS = {
    "python": [
        r"os\.system\s*\(",
        r"subprocess\.call\s*\(\s*shell\s*=\s*True",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(\s*['\"]os['\"]",
        r"pickle\.loads?\s*\(",
        r"yaml\.load\s*\(",
        r"input\s*\(\s*\).*eval",
    ],
    "javascript": [
        r"eval\s*\(",
        r"Function\s*\(",
        r"setTimeout\s*\(\s*['\"]",
        r"setInterval\s*\(\s*['\"]",
        r"document\.write\s*\(",
        r"innerHTML\s*=",
        r"location\.href\s*=",
    ],
    "sql": [
        r"DROP\s+TABLE",
        r"DELETE\s+FROM\s+\w+\s*(?!WHERE)",
        r"TRUNCATE\s+TABLE",
    ]
}

# أنماط تحيز
BIAS_PATTERNS = {
    "gender": [
        r"\b(men are|women are)\s+(better|superior|inferior)",
        r"\b(all men|all women)\s+",
    ],
    "religion": [
        r"\b(all (muslims|christians|jews|hindus))\s+(are|should)",
        r"\b(religion of (peace|terror|hate))",
    ],
    "nationality": [
        r"\b(all (arabs|americans|europeans|asians|africans))\s+(are|should)",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# نتيجة فحص الأمان
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyCheckResult:
    is_safe: bool
    violations: List[Dict]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "violations": self.violations,
            "confidence": self.confidence
        }

# ═══════════════════════════════════════════════════════════════════════════════
# فاحص الأمان
# ═══════════════════════════════════════════════════════════════════════════════

class SafetyChecker:
    """فاحص الأمان لـ Edrak AI"""
    
    def __init__(self):
        self.blocked_words = GLOBAL_BLOCKED_WORDS
        self.dangerous_patterns = DANGEROUS_CODE_PATTERNS
        self.bias_patterns = BIAS_PATTERNS
    
    def check(self, content: str, content_type: str = "text") -> SafetyCheckResult:
        """
        فحص المحتوى للتأكد من أمانه
        
        Args:
            content: المحتوى المراد فحصه
            content_type: نوع المحتوى (text, code, arabic)
        """
        
        violations = []
        
        # فحص الكلمات الممنوعة
        word_violations = self._check_blocked_words(content)
        violations.extend(word_violations)
        
        # فحص الأكواد الخطرة
        if content_type in ["code", "python", "javascript"]:
            code_violations = self._check_dangerous_code(content, content_type)
            violations.extend(code_violations)
        
        # فحص التحيز
        bias_violations = self._check_bias(content)
        violations.extend(bias_violations)
        
        # فحص المحتوى العربي
        if content_type == "arabic" or self._is_arabic(content):
            arabic_violations = self._check_arabic_content(content)
            violations.extend(arabic_violations)
        
        # حساب النتيجة
        is_safe = len(violations) == 0
        confidence = 1.0 - (len(violations) * 0.1)
        
        return SafetyCheckResult(
            is_safe=is_safe,
            violations=violations,
            confidence=max(0.0, confidence)
        )
    
    def _check_blocked_words(self, content: str) -> List[Dict]:
        """فحص الكلمات الممنوعة"""
        
        violations = []
        content_lower = content.lower()
        
        for word in self.blocked_words:
            if word.lower() in content_lower:
                # العثور على موقع الكلمة
                start = content_lower.find(word.lower())
                violations.append({
                    "type": "blocked_word",
                    "severity": "high",
                    "message": f"Blocked word detected: '{word}'",
                    "position": {"start": start, "end": start + len(word)},
                    "suggestion": "Please remove or replace this word"
                })
        
        return violations
    
    def _check_dangerous_code(self, code: str, language: str) -> List[Dict]:
        """فحص الأكواد الخطرة"""
        
        violations = []
        
        # فحص حسب اللغة
        patterns = self.dangerous_patterns.get(language, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append({
                    "type": "dangerous_code",
                    "severity": "high",
                    "message": f"Potentially dangerous code pattern detected",
                    "line": line_num,
                    "pattern": pattern,
                    "suggestion": "Review this code for security vulnerabilities"
                })
        
        return violations
    
    def _check_bias(self, content: str) -> List[Dict]:
        """فحص التحيز"""
        
        violations = []
        content_lower = content.lower()
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    violations.append({
                        "type": "bias",
                        "bias_type": bias_type,
                        "severity": "medium",
                        "message": f"Potentially biased content detected ({bias_type})",
                        "suggestion": "Please ensure content is neutral and inclusive"
                    })
        
        return violations
    
    def _check_arabic_content(self, content: str) -> List[Dict]:
        """فحص المحتوى العربي"""
        
        violations = []
        
        # كلمات ممنوعة بالعربية
        arabic_blocked = [
            "سب", "شتم", "قذف", "تحرش",
            "كفر", "إلحاد", "مهرطق",
            "عنصري", "عنصرية",
        ]
        
        for word in arabic_blocked:
            if word in content:
                violations.append({
                    "type": "inappropriate_arabic_content",
                    "severity": "high",
                    "message": f"Inappropriate Arabic content detected",
                    "suggestion": "Please use respectful language"
                })
        
        return violations
    
    def _is_arabic(self, content: str) -> bool:
        """التحقق مما إذا كان النص يحتوي على عربية"""
        
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(content))
    
    def check_code_security(self, code: str, language: str) -> Dict:
        """فحص أمان الكود بشكل شامل"""
        
        result = {
            "is_secure": True,
            "issues": [],
            "score": 100,
            "recommendations": []
        }
        
        # فحص الأنماط الخطرة
        dangerous = self._check_dangerous_code(code, language)
        if dangerous:
            result["is_secure"] = False
            result["issues"].extend(dangerous)
            result["score"] -= len(dangerous) * 10
        
        # فحص الأسرار المكتوبة مباشرة
        secrets = self._check_hardcoded_secrets(code)
        if secrets:
            result["is_secure"] = False
            result["issues"].extend(secrets)
            result["score"] -= len(secrets) * 15
        
        # فحص حقن SQL
        sql_injection = self._check_sql_injection(code)
        if sql_injection:
            result["is_secure"] = False
            result["issues"].extend(sql_injection)
            result["score"] -= len(sql_injection) * 20
        
        # التوصيات
        result["recommendations"] = self._generate_security_recommendations(code, language)
        
        result["score"] = max(0, result["score"])
        
        return result
    
    def _check_hardcoded_secrets(self, code: str) -> List[Dict]:
        """فحص الأسرار المكتوبة مباشرة"""
        
        violations = []
        
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
            (r'aws_access_key_id\s*=\s*["\'][^"\']+["\']', "Hardcoded AWS key"),
        ]
        
        for pattern, message in secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append({
                    "type": "hardcoded_secret",
                    "severity": "critical",
                    "message": message,
                    "line": line_num,
                    "suggestion": "Use environment variables or a secrets manager"
                })
        
        return violations
    
    def _check_sql_injection(self, code: str) -> List[Dict]:
        """فحص حقن SQL"""
        
        violations = []
        
        sql_patterns = [
            r'execute\s*\(\s*["\'].*%s.*["\']',
            r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
            r'cursor\.execute\s*\(\s*f["\']',
            r'\.format\s*\(',
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                violations.append({
                    "type": "sql_injection",
                    "severity": "critical",
                    "message": "Potential SQL injection vulnerability",
                    "line": line_num,
                    "suggestion": "Use parameterized queries or ORM"
                })
        
        return violations
    
    def _generate_security_recommendations(self, code: str, language: str) -> List[str]:
        """توليد توصيات الأمان"""
        
        recommendations = []
        
        if language == "python":
            if "import os" in code and "os.system" in code:
                recommendations.append("Avoid using os.system(). Use subprocess.run() with proper arguments instead.")
            
            if "pickle" in code:
                recommendations.append("Be cautious with pickle. It can execute arbitrary code. Consider using JSON for serialization.")
            
            if "yaml.load" in code:
                recommendations.append("Use yaml.safe_load() instead of yaml.load() to prevent code execution.")
        
        elif language == "javascript":
            if "eval(" in code:
                recommendations.append("Avoid using eval(). It can execute arbitrary code and is a security risk.")
            
            if "innerHTML" in code:
                recommendations.append("Be careful with innerHTML. It can lead to XSS attacks. Consider using textContent or proper sanitization.")
        
        return recommendations
