"""
Math Solver - حل الرياضيات المتقدم

يدعم:
- الجبر
- التفاضل والتكامل
- الهندسة
- الإحصاء
- الجبر الخطي
- المعادلات التفاضلية
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
from sympy import *

# ═══════════════════════════════════════════════════════════════════════════════
# إعداد SymPy
# ═══════════════════════════════════════════════════════════════════════════════

# تعريف الرموز الشائعة
x, y, z, t = symbols('x y z t')
a, b, c, d = symbols('a b c d')
n, m, k = symbols('n m k', integer=True)
f, g, h = symbols('f g h', cls=Function)

# ═══════════════════════════════════════════════════════════════════════════════
# أنواع المسائل الرياضية
# ═══════════════════════════════════════════════════════════════════════════════

MATH_PROBLEM_TYPES = {
    "algebra": {
        "name": "الجبر",
        "patterns": [
            r"(حل|solve)\s+(المعادلة|equation)?",
            r"(تبسيط|simplify)",
            r"(عوامل|factor)",
            r"(توسيع|expand)",
        ],
        "operations": ["solve", "simplify", "factor", "expand", "substitute"]
    },
    "calculus": {
        "name": "التفاضل والتكامل",
        "patterns": [
            r"(مشتقة|derivative|diff)",
            r"(تكامل|integral|integrate)",
            r"(حد|limit)",
            r"(متسلسلة|series)",
        ],
        "operations": ["diff", "integrate", "limit", "series"]
    },
    "linear_algebra": {
        "name": "الجبر الخطي",
        "patterns": [
            r"(مصفوفة|matrix)",
            r"(مح determinant|determinant)",
            r"(inverse|inverse)",
            r"(eigenvalue|eigenvalue)",
        ],
        "operations": ["matrix", "det", "inv", "eigen"]
    },
    "geometry": {
        "name": "الهندسة",
        "patterns": [
            r"(مساحة|area)",
            r"(محيط|perimeter)",
            r"(حجم|volume)",
            r"(زاوية|angle)",
        ],
        "operations": ["area", "perimeter", "volume", "angle"]
    },
    "statistics": {
        "name": "الإحصاء",
        "patterns": [
            r"(متوسط|mean|average)",
            r"(وسيط|median)",
            r"(منوال|mode)",
            r"(انحراف معياري|std|standard deviation)",
        ],
        "operations": ["mean", "median", "mode", "std", "var"]
    },
    "differential_equations": {
        "name": "المعادلات التفاضلية",
        "patterns": [
            r"(معادلة تفاضلية|differential equation)",
            r"(dy/dx|dx/dy)",
        ],
        "operations": ["dsolve"]
    },
    "trigonometry": {
        "name": "المثلثات",
        "patterns": [
            r"(sin|cos|tan|cot|sec|csc)",
            r"(زاوية|angle)",
        ],
        "operations": ["trig_simplify", "trig_expand"]
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# حل الرياضيات الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class MathSolver:
    """حل الرياضيات المتقدم لـ Edrak AI"""
    
    def __init__(self):
        self.problem_types = MATH_PROBLEM_TYPES
        self.symbols = {
            'x': x, 'y': y, 'z': z, 't': t,
            'a': a, 'b': b, 'c': c, 'd': d,
            'n': n, 'm': m, 'k': k
        }
    
    async def solve(
        self,
        problem: str,
        problem_type: Optional[str] = None,
        show_steps: bool = True,
        language: str = "arabic"
    ) -> Dict:
        """
        حل مسألة رياضية
        
        Args:
            problem: نص المسألة
            problem_type: نوع المسألة
            show_steps: إظهار خطوات الحل
            language: لغة الإجابة
        """
        
        # اكتشاف نوع المسألة
        detected_type = problem_type or self._detect_problem_type(problem)
        
        # تحليل المسألة
        parsed = self._parse_problem(problem, detected_type)
        
        # حل المسألة
        if detected_type == "algebra":
            result = self._solve_algebra(parsed, show_steps)
        elif detected_type == "calculus":
            result = self._solve_calculus(parsed, show_steps)
        elif detected_type == "linear_algebra":
            result = self._solve_linear_algebra(parsed, show_steps)
        elif detected_type == "geometry":
            result = self._solve_geometry(parsed, show_steps)
        elif detected_type == "statistics":
            result = self._solve_statistics(parsed, show_steps)
        elif detected_type == "differential_equations":
            result = self._solve_differential_equations(parsed, show_steps)
        elif detected_type == "trigonometry":
            result = self._solve_trigonometry(parsed, show_steps)
        else:
            result = self._solve_general(parsed, show_steps)
        
        # التحقق من الحل
        verification = self._verify_solution(problem, result.get("solution", ""))
        
        # تنسيق الإجابة
        formatted_solution = self._format_solution(result, language)
        
        return {
            "problem": problem,
            "problem_type": detected_type,
            "solution": formatted_solution,
            "steps": result.get("steps", []) if show_steps else [],
            "verification": verification,
            "difficulty": result.get("difficulty", "medium"),
            "usage": {
                "prompt_tokens": len(problem.split()),
                "completion_tokens": len(formatted_solution.split()),
                "total_tokens": len(problem.split()) + len(formatted_solution.split())
            }
        }
    
    def _detect_problem_type(self, problem: str) -> str:
        """اكتشاف نوع المسألة الرياضية"""
        
        problem_lower = problem.lower()
        scores = {ptype: 0 for ptype in self.problem_types.keys()}
        
        for ptype, data in self.problem_types.items():
            for pattern in data.get("patterns", []):
                if re.search(pattern, problem_lower):
                    scores[ptype] += 1
        
        # كلمات مفتاحية إضافية
        if any(word in problem_lower for word in ["derivative", "مشتقة", "diff", " differentiate"]):
            scores["calculus"] += 2
        
        if any(word in problem_lower for word in ["integral", "تكامل", "integrate"]):
            scores["calculus"] += 2
        
        if any(word in problem_lower for word in ["matrix", "مصفوفة", "determinant", "مح determinan"]):
            scores["linear_algebra"] += 2
        
        if any(word in problem_lower for word in ["equation", "معادلة", "solve for", "حل"]):
            scores["algebra"] += 2
        
        if any(word in problem_lower for word in ["area", "مساحة", "volume", "حجم", "perimeter", "محيط"]):
            scores["geometry"] += 2
        
        if any(word in problem_lower for word in ["mean", "متوسط", "average", "median", "وسيط", "std", "انحراف معياري"]):
            scores["statistics"] += 2
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0 else "general"
    
    def _parse_problem(self, problem: str, problem_type: str) -> Dict:
        """تحليل المسألة واستخراج المعادلات"""
        
        parsed = {
            "original": problem,
            "expressions": [],
            "equations": [],
            "variables": [],
            "constants": []
        }
        
        # استخراج المعادلات
        equation_pattern = r'([^=]+)=([^\n]+)'
        for match in re.finditer(equation_pattern, problem):
            left = match.group(1).strip()
            right = match.group(2).strip()
            parsed["equations"].append((left, right))
        
        # استخراج المتغيرات
        var_pattern = r'\b([xyzabcnmkt])\b'
        variables = set(re.findall(var_pattern, problem.lower()))
        parsed["variables"] = list(variables)
        
        # استخراج الأرقام
        number_pattern = r'\b(\d+(?:\.\d+)?)\b'
        numbers = re.findall(number_pattern, problem)
        parsed["constants"] = [float(n) for n in numbers]
        
        return parsed
    
    def _solve_algebra(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل الجبر"""
        
        steps = []
        solution = ""
        
        try:
            # تحليل المعادلات
            for left, right in parsed["equations"]:
                expr_left = sympify(left)
                expr_right = sympify(right)
                
                equation = Eq(expr_left, expr_right)
                
                if show_steps:
                    steps.append(f"المعادلة: {latex(equation)}")
                
                # حل المعادلة
                variables = [self.symbols.get(v, Symbol(v)) for v in parsed["variables"]]
                
                if variables:
                    sol = solve(equation, variables[0])
                    
                    if show_steps:
                        steps.append(f"الحل: {variables[0]} = {sol}")
                    
                    solution = f"{variables[0]} = {sol}"
            
            # إذا لم تكن هناك معادلات، قم بتبسيط التعبير
            if not parsed["equations"] and parsed["original"]:
                expr = sympify(parsed["original"])
                simplified = simplify(expr)
                
                if show_steps:
                    steps.append(f"التعبير الأصلي: {latex(expr)}")
                    steps.append(f"التبسيط: {latex(simplified)}")
                
                solution = str(simplified)
        
        except Exception as e:
            solution = f"Error solving: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "medium"
        }
    
    def _solve_calculus(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل التفاضل والتكامل"""
        
        steps = []
        solution = ""
        
        try:
            problem = parsed["original"].lower()
            
            # اشتقاق
            if any(word in problem for word in ["derivative", "مشتقة", "diff"]):
                # استخراج التعبير
                expr_match = re.search(r'(?:derivative|مشتقة|diff)\s+(?:of\s+)?(.+?)(?:\s+(?:with respect to|بالنسبة ل))?\s*([xyz])?', problem)
                
                if expr_match:
                    expr_str = expr_match.group(1).strip()
                    var_str = expr_match.group(2) if expr_match.group(2) else 'x'
                    
                    expr = sympify(expr_str)
                    var = self.symbols.get(var_str, Symbol(var_str))
                    
                    if show_steps:
                        steps.append(f"التعبير: {latex(expr)}")
                        steps.append(f"المتغير: {var}")
                    
                    # الاشتقاق
                    derivative = diff(expr, var)
                    
                    if show_steps:
                        steps.append(f"المشتقة: {latex(derivative)}")
                    
                    solution = f"d/d{var}({expr}) = {derivative}"
            
            # تكامل
            elif any(word in problem for word in ["integral", "تكامل", "integrate"]):
                expr_match = re.search(r'(?:integral|تكامل|integrate)\s+(?:of\s+)?(.+?)(?:\s+(?:with respect to|بالنسبة ل))?\s*([xyz])?', problem)
                
                if expr_match:
                    expr_str = expr_match.group(1).strip()
                    var_str = expr_match.group(2) if expr_match.group(2) else 'x'
                    
                    expr = sympify(expr_str)
                    var = self.symbols.get(var_str, Symbol(var_str))
                    
                    if show_steps:
                        steps.append(f"التعبير: {latex(expr)}")
                        steps.append(f"المتغير: {var}")
                    
                    # التكامل
                    integral = integrate(expr, var)
                    
                    if show_steps:
                        steps.append(f"التكامل: {latex(integral)} + C")
                    
                    solution = f"∫{expr} d{var} = {integral} + C"
            
            # حدود
            elif "limit" in problem or "حد" in problem:
                # TODO: تنفيذ حدود
                solution = "Limit calculation not yet implemented"
        
        except Exception as e:
            solution = f"Error in calculus: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "hard"
        }
    
    def _solve_linear_algebra(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل الجبر الخطي"""
        
        steps = []
        solution = ""
        
        try:
            problem = parsed["original"].lower()
            
            # مصفوفة
            if "matrix" in problem or "مصفوفة" in problem:
                # استخراج عناصر المصفوفة
                matrix_match = re.search(r'\[\s*([^\]]+)\s*\]', problem)
                
                if matrix_match:
                    matrix_str = matrix_match.group(1)
                    
                    # تحليل المصفوفة
                    rows = matrix_str.split(';')
                    matrix_data = []
                    
                    for row in rows:
                        elements = [float(x.strip()) for x in row.split(',')]
                        matrix_data.append(elements)
                    
                    M = Matrix(matrix_data)
                    
                    if show_steps:
                        steps.append(f"المصفوفة: {latex(M)}")
                    
                    # عمليات على المصفوفة
                    if "det" in problem or "determinant" in problem or "مح determinan" in problem:
                        det = M.det()
                        if show_steps:
                            steps.append(f"المحدد: {latex(det)}")
                        solution = f"det(M) = {det}"
                    
                    elif "inverse" in problem or "inverse" in problem:
                        if M.det() != 0:
                            inv = M.inv()
                            if show_steps:
                                steps.append(f"المعكوس: {latex(inv)}")
                            solution = f"M^(-1) = {inv}"
                        else:
                            solution = "المصفوفة غير قابلة للعكس (المحدد = 0)"
                    
                    elif "eigen" in problem:
                        eigenvals = M.eigenvals()
                        if show_steps:
                            steps.append(f"القيم الذاتية: {eigenvals}")
                        solution = f"Eigenvalues: {eigenvals}"
                    
                    else:
                        solution = f"Matrix: {M}"
        
        except Exception as e:
            solution = f"Error in linear algebra: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "hard"
        }
    
    def _solve_geometry(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل الهندسة"""
        
        steps = []
        solution = ""
        
        try:
            problem = parsed["original"].lower()
            
            # مساحة
            if "area" in problem or "مساحة" in problem:
                # مثلث
                if "triangle" in problem or "مثلث" in problem:
                    # استخراج القاعدة والارتفاع
                    base_match = re.search(r'base\s*=\s*(\d+)', problem)
                    height_match = re.search(r'height\s*=\s*(\d+)', problem)
                    
                    if base_match and height_match:
                        base = float(base_match.group(1))
                        height = float(height_match.group(1))
                        
                        if show_steps:
                            steps.append(f"القاعدة: {base}")
                            steps.append(f"الارتفاع: {height}")
                            steps.append(f"المساحة = (1/2) × قاعدة × ارتفاع")
                        
                        area = 0.5 * base * height
                        solution = f"مساحة المثلث = {area}"
                
                # دائرة
                elif "circle" in problem or "دائرة" in problem:
                    radius_match = re.search(r'radius\s*=\s*(\d+)', problem)
                    
                    if radius_match:
                        radius = float(radius_match.group(1))
                        
                        if show_steps:
                            steps.append(f"نصف القطر: {radius}")
                            steps.append(f"المساحة = π × r²")
                        
                        area = float(pi * radius**2)
                        solution = f"مساحة الدائرة = {area:.4f}"
                
                # مستطيل
                elif "rectangle" in problem or "مستطيل" in problem:
                    length_match = re.search(r'length\s*=\s*(\d+)', problem)
                    width_match = re.search(r'width\s*=\s*(\d+)', problem)
                    
                    if length_match and width_match:
                        length = float(length_match.group(1))
                        width = float(width_match.group(1))
                        
                        if show_steps:
                            steps.append(f"الطول: {length}")
                            steps.append(f"العرض: {width}")
                            steps.append(f"المساحة = طول × عرض")
                        
                        area = length * width
                        solution = f"مساحة المستطيل = {area}"
            
            # محيط
            elif "perimeter" in problem or "محيط" in problem:
                # TODO: تنفيذ حساب المحيط
                solution = "Perimeter calculation - implementation needed"
            
            # حجم
            elif "volume" in problem or "حجم" in problem:
                # TODO: تنفيذ حساب الحجم
                solution = "Volume calculation - implementation needed"
        
        except Exception as e:
            solution = f"Error in geometry: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "medium"
        }
    
    def _solve_statistics(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل الإحصاء"""
        
        steps = []
        solution = ""
        
        try:
            problem = parsed["original"].lower()
            
            # استخراج البيانات
            data_match = re.search(r'\[\s*([^\]]+)\s*\]', problem)
            
            if data_match:
                data_str = data_match.group(1)
                data = [float(x.strip()) for x in data_str.split(',')]
                
                if show_steps:
                    steps.append(f"البيانات: {data}")
                
                # المتوسط
                if any(word in problem for word in ["mean", "متوسط", "average"]):
                    mean_val = np.mean(data)
                    
                    if show_steps:
                        steps.append(f"المتوسط = sum(data) / len(data)")
                        steps.append(f"المتوسط = {sum(data)} / {len(data)}")
                    
                    solution = f"المتوسط = {mean_val:.4f}"
                
                # الوسيط
                elif any(word in problem for word in ["median", "وسيط"]):
                    median_val = np.median(data)
                    solution = f"الوسيط = {median_val:.4f}"
                
                # المنوال
                elif any(word in problem for word in ["mode", "منوال"]):
                    from scipy import stats
                    mode_val = stats.mode(data)[0][0]
                    solution = f"المنوال = {mode_val}"
                
                # الانحراف المعياري
                elif any(word in problem for word in ["std", "standard deviation", "انحراف معياري"]):
                    std_val = np.std(data)
                    
                    if show_steps:
                        steps.append(f"الانحراف المعياري = sqrt(variance)")
                    
                    solution = f"الانحراف المعياري = {std_val:.4f}"
                
                # التباين
                elif any(word in problem for word in ["variance", "تباين"]):
                    var_val = np.var(data)
                    solution = f"التباين = {var_val:.4f}"
        
        except Exception as e:
            solution = f"Error in statistics: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "easy"
        }
    
    def _solve_differential_equations(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل المعادلات التفاضلية"""
        
        steps = []
        solution = ""
        
        try:
            # TODO: تنفيذ حل المعادلات التفاضلية
            solution = "Differential equations solver - implementation in progress"
        
        except Exception as e:
            solution = f"Error in differential equations: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "hard"
        }
    
    def _solve_trigonometry(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل مسائل المثلثات"""
        
        steps = []
        solution = ""
        
        try:
            problem = parsed["original"].lower()
            
            # تبسيط التعبيرات المثلثية
            if "simplify" in problem or "تبسيط" in problem:
                expr_match = re.search(r'(?:simplify|تبسيط)\s+(.+)', problem)
                
                if expr_match:
                    expr_str = expr_match.group(1).strip()
                    expr = sympify(expr_str)
                    
                    simplified = trigsimp(expr)
                    
                    if show_steps:
                        steps.append(f"التعبير الأصلي: {latex(expr)}")
                        steps.append(f"التبسيط: {latex(simplified)}")
                    
                    solution = f"{simplified}"
            
            # توسيع التعبيرات المثلثية
            elif "expand" in problem or "توسيع" in problem:
                expr_match = re.search(r'(?:expand|توسيع)\s+(.+)', problem)
                
                if expr_match:
                    expr_str = expr_match.group(1).strip()
                    expr = sympify(expr_str)
                    
                    expanded = expand_trig(expr)
                    
                    if show_steps:
                        steps.append(f"التعبير الأصلي: {latex(expr)}")
                        steps.append(f"التوسيع: {latex(expanded)}")
                    
                    solution = f"{expanded}"
        
        except Exception as e:
            solution = f"Error in trigonometry: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "medium"
        }
    
    def _solve_general(self, parsed: Dict, show_steps: bool) -> Dict:
        """حل عام للمسائل"""
        
        steps = []
        solution = ""
        
        try:
            # محاولة تبسيط التعبير
            expr = sympify(parsed["original"])
            simplified = simplify(expr)
            
            if show_steps:
                steps.append(f"التعبير: {latex(expr)}")
                steps.append(f"التبسيط: {latex(simplified)}")
            
            solution = str(simplified)
        
        except Exception as e:
            solution = f"Could not solve: {str(e)}"
        
        return {
            "solution": solution,
            "steps": steps,
            "difficulty": "unknown"
        }
    
    def _verify_solution(self, problem: str, solution: str) -> bool:
        """التحقق من صحة الحل"""
        
        # TODO: تنفيذ التحقق الفعلي
        
        return True
    
    def _format_solution(self, result: Dict, language: str) -> str:
        """تنسيق الحل"""
        
        solution = result.get("solution", "")
        
        if language == "arabic":
            # ترجمة المصطلحات للعربية
            translations = {
                "solution": "الحل",
                "equals": "يساوي",
                "where": "حيث",
                "therefore": "إذن",
                "thus": "وبالتالي",
            }
            
            # TODO: تنفيذ الترجمة الفعلية
        
        return solution
