"""
Code Engine - محرك الأكواد المتقدم

يدعم:
- توليد أكواد بجودة عالية
- مراجعة الأكواد
- تحليل التعقيد
- فحص الأمان
- اقتراح تحسينات
"""

import re
import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# قوالب الأكواد والأنماط
# ═══════════════════════════════════════════════════════════════════════════════

CODE_PATTERNS = {
    "python": {
        "web_frameworks": ["django", "flask", "fastapi", "tornado", "bottle"],
        "data_science": ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "tensorflow", "pytorch"],
        "automation": ["selenium", "requests", "beautifulsoup4", "scrapy"],
        "testing": ["pytest", "unittest", "mock"],
        "patterns": {
            "singleton": """
class {class_name}:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
""",
            "factory": """
class {factory_name}:
    @staticmethod
    def create_{product_type}(type):
        if type == "{type1}":
            return {Class1}()
        elif type == "{type2}":
            return {Class2}()
        raise ValueError(f"Unknown type: {type}")
""",
            "observer": """
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class Observer:
    def update(self, subject):
        pass
""",
            "decorator": """
def {decorator_name}(func):
    def wrapper(*args, **kwargs):
        # Before
        result = func(*args, **kwargs)
        # After
        return result
    return wrapper
""",
            "context_manager": """
class {manager_name}:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
""",
        }
    },
    
    "javascript": {
        "web_frameworks": ["react", "vue", "angular", "svelte", "next.js", "nuxt.js"],
        "backend": ["express", "koa", "fastify", "nest.js"],
        "testing": ["jest", "mocha", "cypress", "playwright"],
        "patterns": {
            "singleton": """
class {class_name} {{
    static instance = null;
    
    static getInstance() {{
        if (!{class_name}.instance) {{
            {class_name}.instance = new {class_name}();
        }}
        return {class_name}.instance;
    }}
}}
""",
            "factory": """
class {factory_name} {{
    static create{type}(type) {{
        switch(type) {{
            case '{type1}': return new {Class1}();
            case '{type2}': return new {Class2}();
            default: throw new Error(`Unknown type: ${{type}}`);
        }}
    }}
}}
""",
            "observer": """
class Subject {{
    constructor() {{
        this.observers = [];
    }}
    
    attach(observer) {{
        this.observers.push(observer);
    }}
    
    notify() {{
        this.observers.forEach(observer => observer.update(this));
    }}
}}
""",
            "async_await": """
async function {function_name}() {{
    try {{
        const result = await {async_operation};
        return result;
    }} catch (error) {{
        console.error('Error:', error);
        throw error;
    }}
}}
""",
        }
    },
    
    "typescript": {
        "patterns": {
            "interface": """
interface {interface_name} {{
    {property1}: {type1};
    {property2}: {type2};
}}
""",
            "generic_class": """
class {class_name}<T> {{
    private data: T[] = [];
    
    add(item: T): void {{
        this.data.push(item);
    }}
    
    get(index: number): T | undefined {{
        return this.data[index];
    }}
}}
""",
            "decorator": """
function {decorator_name}(target: any, propertyKey: string, descriptor: PropertyDescriptor) {{
    const originalMethod = descriptor.value;
    descriptor.value = function(...args: any[]) {{
        // Before
        const result = originalMethod.apply(this, args);
        // After
        return result;
    }};
    return descriptor;
}}
""",
        }
    },
    
    "html": {
        "templates": {
            "basic": """<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {head_content}
</head>
<body>
    {body_content}
</body>
</html>""",
            "responsive": """<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 0 10px;
            }}
        }}
    </style>
</head>
<body>
    {body_content}
</body>
</html>""",
        }
    },
    
    "css": {
        "patterns": {
            "flexbox_center": """
.{class_name} {{
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}}
""",
            "grid_layout": """
.{class_name} {{
    display: grid;
    grid-template-columns: repeat({columns}, 1fr);
    gap: {gap}px;
}}
""",
            "animation": """
@keyframes {animation_name} {{
    0% {{ {property}: {start_value}; }}
    100% {{ {property}: {end_value}; }}
}}

.{class_name} {{
    animation: {animation_name} {duration}s {easing} {iteration};
}}
""",
            "responsive": """
.{class_name} {{
    /* Base styles */
}}

@media (max-width: 768px) {{
    .{class_name} {{
        /* Tablet styles */
    }}
}}

@media (max-width: 480px) {{
    .{class_name} {{
        /* Mobile styles */
    }}
}}
""",
        }
    },
    
    "cpp": {
        "patterns": {
            "class": """
class {class_name} {{
private:
    // Private members
    
public:
    {class_name}() = default;
    ~{class_name}() = default;
    
    // Public methods
}};
""",
            "template": """
template<typename T>
class {class_name} {{
private:
    T data;
    
public:
    void setData(T value) {{ data = value; }}
    T getData() const {{ return data; }}
}};
""",
        }
    },
    
    "java": {
        "patterns": {
            "class": """
public class {class_name} {{
    // Fields
    
    // Constructor
    public {class_name}() {{
    }}
    
    // Methods
}}
""",
            "singleton": """
public class {class_name} {{
    private static {class_name} instance;
    
    private {class_name}() {{}}
    
    public static synchronized {class_name} getInstance() {{
        if (instance == null) {{
            instance = new {class_name}();
        }}
        return instance;
    }}
}}
""",
        }
    },
    
    "rust": {
        "patterns": {
            "struct": """
pub struct {struct_name} {{
    field: {type},
}}

impl {struct_name} {{
    pub fn new() -> Self {{
        Self {{
            field: Default::default(),
        }}
    }}
}}
""",
            "trait": """
pub trait {trait_name} {{
    fn method(&self) -> {return_type};
}}

impl {trait_name} for {type} {{
    fn method(&self) -> {return_type} {{
        // Implementation
    }}
}}
""",
        }
    },
    
    "go": {
        "patterns": {
            "struct": """
type {struct_name} struct {{
    Field {type}
}}

func New{struct_name}() *{struct_name} {{
    return &{struct_name}{{}}
}}

func (s *{struct_name}) Method() {{
    // Implementation
}}
""",
            "interface": """
type {interface_name} interface {{
    Method()
}}
""",
        }
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# محرك الأكواد الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class CodeEngine:
    """محرك الأكواد المتقدم لـ Edrak AI"""
    
    def __init__(self):
        self.patterns = CODE_PATTERNS
        self.security_patterns = self._load_security_patterns()
        self.best_practices = self._load_best_practices()
    
    def _load_security_patterns(self) -> Dict:
        """تحميل أنماط الأمان"""
        return {
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%s.*['\"]",
                r"cursor\.execute\s*\(\s*['\"].*\+.*['\"]",
                r"SELECT\s+.*\s+FROM\s+.*WHERE\s+.*=\s*['\"].*\+.*['\"]",
            ],
            "xss": [
                r"innerHTML\s*=",
                r"document\.write\s*\(",
                r"eval\s*\(",
            ],
            "command_injection": [
                r"os\.system\s*\(\s*['\"].*\+.*['\"]",
                r"subprocess\.call\s*\(\s*['\"].*\+.*['\"]",
                r"exec\s*\(",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]",
            ],
            "insecure_deserialization": [
                r"pickle\.loads?\s*\(",
                r"yaml\.load\s*\(",
                r"eval\s*\(",
            ],
        }
    
    def _load_best_practices(self) -> Dict:
        """تحميل أفضل الممارسات"""
        return {
            "python": {
                "naming": {
                    "classes": "PascalCase",
                    "functions": "snake_case",
                    "constants": "UPPER_SNAKE_CASE",
                    "variables": "snake_case",
                },
                "docstrings": "Google style",
                "type_hints": True,
                "max_line_length": 88,
            },
            "javascript": {
                "naming": {
                    "classes": "PascalCase",
                    "functions": "camelCase",
                    "constants": "UPPER_SNAKE_CASE",
                    "variables": "camelCase",
                },
                "semicolons": True,
                "quotes": "single",
            },
            "typescript": {
                "naming": {
                    "interfaces": "PascalCase with I prefix",
                    "types": "PascalCase",
                    "enums": "PascalCase",
                },
                "strict": True,
            },
        }
    
    async def generate(
        self,
        prompt: str,
        language: str = "python",
        framework: Optional[str] = None,
        style: str = "clean",
        include_tests: bool = False,
        context: Optional[str] = None
    ) -> Dict:
        """
        توليد كود عالي الجودة
        
        Args:
            prompt: وصف الكود المطلوب
            language: لغة البرمجة
            framework: الإطار المستخدم
            style: أسلوب الكود
            include_tests: تضمين الاختبارات
            context: سياق إضافي
        """
        
        # تحليل الطلب
        analysis = self._analyze_prompt(prompt, language)
        
        # توليد الكود
        code = self._generate_code(analysis, language, framework, style)
        
        # تنسيق الكود
        formatted_code = self._format_code(code, language, style)
        
        # توليد الاختبارات إذا طُلبت
        tests = ""
        if include_tests:
            tests = self._generate_tests(formatted_code, language, framework)
        
        # تحليل التعقيد
        complexity = self._analyze_complexity(formatted_code, language)
        
        # فحص الأمان
        security = self._check_security(formatted_code, language)
        
        # اقتراحات التحسين
        suggestions = self._suggest_improvements(formatted_code, language)
        
        # الشرح
        explanation = self._generate_explanation(formatted_code, language)
        
        return {
            "code": formatted_code,
            "language": language,
            "framework": framework,
            "explanation": explanation,
            "tests": tests,
            "complexity": complexity,
            "security": security,
            "suggestions": suggestions,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(formatted_code.split()),
            "total_tokens": len(prompt.split()) + len(formatted_code.split())
        }
    
    async def review(
        self,
        code: str,
        language: str,
        review_type: str = "full"
    ) -> Dict:
        """
        مراجعة الكود
        
        Args:
            code: الكود المراد مراجعته
            language: لغة البرمجة
            review_type: نوع المراجعة (full, security, style, performance)
        """
        
        score = 100
        issues = []
        suggestions = []
        
        # مراجعة الأمان
        if review_type in ["full", "security"]:
            security_issues = self._check_security(code, language)
            for issue in security_issues.get("issues", []):
                issues.append(issue)
                score -= 10
        
        # مراجعة الأسلوب
        if review_type in ["full", "style"]:
            style_issues = self._check_style(code, language)
            for issue in style_issues:
                issues.append(issue)
                score -= 5
        
        # مراجعة الأداء
        if review_type in ["full", "performance"]:
            perf_issues = self._check_performance(code, language)
            for issue in perf_issues:
                issues.append(issue)
                score -= 5
        
        # أفضل الممارسات
        best_practices = self._check_best_practices(code, language)
        
        # اقتراحات
        suggestions = self._suggest_improvements(code, language)
        
        return {
            "score": max(0, score),
            "issues": issues,
            "suggestions": suggestions,
            "security": security_issues if review_type in ["full", "security"] else {},
            "best_practices": best_practices,
            "performance_notes": perf_issues if review_type in ["full", "performance"] else [],
        }
    
    def _analyze_prompt(self, prompt: str, language: str) -> Dict:
        """تحليل طلب المستخدم"""
        
        analysis = {
            "intent": "general",
            "components": [],
            "requirements": [],
            "constraints": []
        }
        
        prompt_lower = prompt.lower()
        
        # اكتشاف النية
        if any(word in prompt_lower for word in ["create", "create", "generate", "write", "اكتب", "أنشئ", "عمل"]):
            analysis["intent"] = "create"
        elif any(word in prompt_lower for word in ["fix", "debug", "solve", "حل", "أصلح", "صحح"]):
            analysis["intent"] = "fix"
        elif any(word in prompt_lower for word in ["optimize", "improve", "enhance", "حسن", "طور"]):
            analysis["intent"] = "optimize"
        
        # استخراج المكونات
        component_keywords = {
            "class": ["class", "فئة", "صنف"],
            "function": ["function", "method", "دالة", "وظيفة"],
            "api": ["api", "endpoint", "route", "نقطة نهاية"],
            "database": ["database", "db", "query", "قاعدة بيانات"],
            "ui": ["ui", "interface", "component", "واجهة"],
        }
        
        for component, keywords in component_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                analysis["components"].append(component)
        
        return analysis
    
    def _generate_code(
        self, 
        analysis: Dict, 
        language: str, 
        framework: Optional[str],
        style: str
    ) -> str:
        """توليد الكود بناءً على التحليل"""
        
        # استخدام قوالب جاهزة إذا وجدت
        if language in self.patterns:
            lang_patterns = self.patterns[language]
            
            # توليد class
            if "class" in analysis["components"]:
                pattern = lang_patterns.get("patterns", {}).get("class", "")
                if pattern:
                    return pattern.format(class_name="MyClass")
            
            # توليد function
            if "function" in analysis["components"]:
                if language == "python":
                    return self._generate_python_function(analysis)
                elif language == "javascript":
                    return self._generate_javascript_function(analysis)
        
        # توليد افتراضي
        return self._generate_default_code(analysis, language)
    
    def _generate_python_function(self, analysis: Dict) -> str:
        """توليد دالة بايثون"""
        return '''def process_data(data: list) -> dict:
    """
    معالجة البيانات وإرجاع النتائج.
    
    Args:
        data: قائمة البيانات المدخلة
        
    Returns:
        قاموس يحتوي على النتائج
    """
    results = {}
    
    for item in data:
        # معالجة كل عنصر
        processed = item.strip().lower()
        results[processed] = len(processed)
    
    return results
'''
    
    def _generate_javascript_function(self, analysis: Dict) -> str:
        """توليد دالة جافاسكربت"""
        return '''/**
 * معالجة البيانات وإرجاع النتائج
 * @param {Array} data - قائمة البيانات
 * @returns {Object} - النتائج
 */
function processData(data) {
    const results = {};
    
    data.forEach(item => {
        const processed = item.trim().toLowerCase();
        results[processed] = processed.length;
    });
    
    return results;
}
'''
    
    def _generate_default_code(self, analysis: Dict, language: str) -> str:
        """توليد كود افتراضي"""
        
        templates = {
            "python": "# TODO: Implement\npass",
            "javascript": "// TODO: Implement\nfunction main() {\n    // Your code here\n}",
            "typescript": "// TODO: Implement\nfunction main(): void {\n    // Your code here\n}",
            "html": "<!-- TODO: Implement -->\n<div>Your content here</div>",
            "css": "/* TODO: Implement */\n.class {\n    /* Your styles here */\n}",
            "cpp": "// TODO: Implement\nint main() {\n    return 0;\n}",
            "java": "// TODO: Implement\npublic class Main {\n    public static void main(String[] args) {\n    }\n}",
            "rust": "// TODO: Implement\nfn main() {\n}",
            "go": "// TODO: Implement\npackage main\n\nfunc main() {\n}",
        }
        
        return templates.get(language, f"// TODO: Implement in {language}")
    
    def _format_code(self, code: str, language: str, style: str) -> str:
        """تنسيق الكود"""
        
        # إزالة المسافات الزائدة
        lines = code.split('\n')
        formatted_lines = []
        
        indent_level = 0
        for line in lines:
            stripped = line.strip()
            
            # تقليل المسافة البادئة للأقواس المغلقة
            if stripped.startswith('}') or stripped.startswith(']'):
                indent_level = max(0, indent_level - 1)
            
            # إضافة المسافة البادئة
            formatted_line = '    ' * indent_level + stripped
            formatted_lines.append(formatted_line)
            
            # زيادة المسافة البادئة للأقواس المفتوحة
            if stripped.endswith('{') or stripped.endswith('['):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _generate_tests(self, code: str, language: str, framework: Optional[str]) -> str:
        """توليد اختبارات للكود"""
        
        if language == "python":
            return self._generate_python_tests(code, framework)
        elif language == "javascript":
            return self._generate_javascript_tests(code, framework)
        
        return "# Tests not available for this language yet"
    
    def _generate_python_tests(self, code: str, framework: Optional[str]) -> str:
        """توليد اختبارات بايثون"""
        
        return '''import unittest
from your_module import process_data

class TestDataProcessing(unittest.TestCase):
    def test_process_data_basic(self):
        """Test basic data processing"""
        input_data = ["Hello", "World"]
        result = process_data(input_data)
        self.assertEqual(result["hello"], 5)
        self.assertEqual(result["world"], 5)
    
    def test_process_data_empty(self):
        """Test with empty input"""
        result = process_data([])
        self.assertEqual(result, {})
    
    def test_process_data_whitespace(self):
        """Test whitespace handling"""
        result = process_data(["  test  "])
        self.assertEqual(result["test"], 4)

if __name__ == '__main__':
    unittest.main()
'''
    
    def _generate_javascript_tests(self, code: str, framework: Optional[str]) -> str:
        """توليد اختبارات جافاسكربت"""
        
        if framework == "jest":
            return '''const { processData } = require('./yourModule');

describe('processData', () => {
    test('basic data processing', () => {
        const input = ['Hello', 'World'];
        const result = processData(input);
        expect(result['hello']).toBe(5);
        expect(result['world']).toBe(5);
    });
    
    test('empty input', () => {
        const result = processData([]);
        expect(result).toEqual({});
    });
    
    test('whitespace handling', () => {
        const result = processData(['  test  ']);
        expect(result['test']).toBe(4);
    });
});
'''
        
        return "// Tests not available for this framework"
    
    def _analyze_complexity(self, code: str, language: str) -> Dict:
        """تحليل تعقيد الكود"""
        
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # حساب التعقيد الحلقي
        loop_count = len(re.findall(r'\b(for|while|do)\b', code))
        
        # حساب التعقيد الشرطي
        condition_count = len(re.findall(r'\b(if|else|switch|case)\b', code))
        
        # عدد الدوال
        function_count = len(re.findall(r'\b(def|function|func)\b', code))
        
        # عدد الفئات
        class_count = len(re.findall(r'\b(class|struct|interface)\b', code))
        
        # التعقيد الإدراكي التقريبي
        cyclomatic = 1 + loop_count + condition_count
        
        return {
            "lines_of_code": len(non_empty_lines),
            "blank_lines": len(lines) - len(non_empty_lines),
            "loops": loop_count,
            "conditions": condition_count,
            "functions": function_count,
            "classes": class_count,
            "cyclomatic_complexity": cyclomatic,
            "complexity_level": "low" if cyclomatic <= 10 else "medium" if cyclomatic <= 20 else "high"
        }
    
    def _check_security(self, code: str, language: str) -> Dict:
        """فحص أمان الكود"""
        
        issues = []
        
        for issue_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        "type": issue_type,
                        "severity": "high" if issue_type in ["sql_injection", "command_injection"] else "medium",
                        "line": code[:match.start()].count('\n') + 1,
                        "message": f"Potential {issue_type.replace('_', ' ').title()} vulnerability detected",
                        "suggestion": self._get_security_suggestion(issue_type)
                    })
        
        return {
            "issues": issues,
            "score": max(0, 100 - len(issues) * 10),
            "passed": len(issues) == 0
        }
    
    def _get_security_suggestion(self, issue_type: str) -> str:
        """الحصول على اقتراح أمان"""
        
        suggestions = {
            "sql_injection": "Use parameterized queries or ORM instead of string concatenation",
            "xss": "Use proper escaping or templating engines to prevent XSS",
            "command_injection": "Avoid shell=True and use lists of arguments instead of string concatenation",
            "hardcoded_secrets": "Use environment variables or secret management systems",
            "insecure_deserialization": "Use safe serialization formats like JSON instead of pickle",
        }
        
        return suggestions.get(issue_type, "Review and fix the security issue")
    
    def _check_style(self, code: str, language: str) -> List[Dict]:
        """فحص أسلوب الكود"""
        
        issues = []
        
        if language == "python":
            # فحص تسمية الدوال
            function_pattern = r'def\s+([A-Z][a-zA-Z0-9]*)\s*\('
            for match in re.finditer(function_pattern, code):
                issues.append({
                    "type": "naming",
                    "severity": "low",
                    "line": code[:match.start()].count('\n') + 1,
                    "message": f"Function '{match.group(1)}' should use snake_case",
                    "suggestion": f"Rename to '{self._to_snake_case(match.group(1))}'"
                })
        
        return issues
    
    def _check_performance(self, code: str, language: str) -> List[Dict]:
        """فحص أداء الكود"""
        
        issues = []
        
        # فحص الحلقات غير الفعالة
        if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', code):
            issues.append({
                "type": "performance",
                "severity": "medium",
                "message": "Use enumerate() instead of range(len())",
                "suggestion": "Replace 'for i in range(len(x))' with 'for i, item in enumerate(x)'"
            })
        
        # فحص القوائم في الحلقات
        if re.search(r'for\s+\w+\s+in\s+\[.*?\]', code):
            issues.append({
                "type": "performance",
                "severity": "low",
                "message": "Consider using a set for O(1) lookup",
                "suggestion": "Convert list to set if only checking membership"
            })
        
        return issues
    
    def _check_best_practices(self, code: str, language: str) -> List[Dict]:
        """فحص أفضل الممارسات"""
        
        practices = []
        
        if language == "python":
            # فحص وجود docstrings
            if 'def ' in code and '"""' not in code and "'''" not in code:
                practices.append({
                    "practice": "documentation",
                    "status": "missing",
                    "message": "Add docstrings to functions for better documentation"
                })
            
            # فحص type hints
            if 'def ' in code and '->' not in code:
                practices.append({
                    "practice": "type_hints",
                    "status": "missing",
                    "message": "Consider adding type hints for better code clarity"
                })
        
        return practices
    
    def _suggest_improvements(self, code: str, language: str) -> List[str]:
        """اقتراح تحسينات"""
        
        suggestions = []
        
        if language == "python":
            if 'try:' not in code and any(op in code for op in ['open(', 'read(', 'write(']):
                suggestions.append("Add error handling with try-except blocks for file operations")
            
            if 'logging' not in code:
                suggestions.append("Consider adding logging instead of print statements")
            
            if 'if __name__' not in code and 'def ' in code:
                suggestions.append("Add 'if __name__ == \"__main__\":' guard for script execution")
        
        return suggestions
    
    def _generate_explanation(self, code: str, language: str) -> str:
        """توليد شرح للكود"""
        
        lines = code.split('\n')
        explanation_parts = []
        
        # شرح عام
        explanation_parts.append(f"This {language} code performs the following operations:")
        
        # تحليل الكود سطر بسطر
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('def ') or stripped.startswith('function '):
                explanation_parts.append(f"\n**Line {i}**: Defines a function")
            elif stripped.startswith('class '):
                explanation_parts.append(f"\n**Line {i}**: Defines a class")
            elif 'for ' in stripped and ' in ' in stripped:
                explanation_parts.append(f"\n**Line {i}**: Iterates over a collection")
            elif 'if ' in stripped:
                explanation_parts.append(f"\n**Line {i}**: Conditional check")
            elif 'return ' in stripped:
                explanation_parts.append(f"\n**Line {i}**: Returns a value")
        
        return '\n'.join(explanation_parts)
    
    def _to_snake_case(self, name: str) -> str:
        """تحويل إلى snake_case"""
        
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
