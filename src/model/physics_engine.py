"""
Physics Engine - محرك الفيزياء المتقدم

يدعم:
- الميكانيكا
- الكهرومغناطيسية
- الديناميكا الحرارية
- ميكانيكا الكم
- النسبية
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
from sympy import *

# ═══════════════════════════════════════════════════════════════════════════════
# الثوابت الفيزيائية
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICAL_CONSTANTS = {
    "speed_of_light": {"value": 299792458, "unit": "m/s", "symbol": "c"},
    "gravitational_constant": {"value": 6.67430e-11, "unit": "N⋅m²/kg²", "symbol": "G"},
    "planck_constant": {"value": 6.62607015e-34, "unit": "J⋅s", "symbol": "h"},
    "reduced_planck_constant": {"value": 1.054571817e-34, "unit": "J⋅s", "symbol": "ℏ"},
    "electron_mass": {"value": 9.10938356e-31, "unit": "kg", "symbol": "m_e"},
    "proton_mass": {"value": 1.67262192369e-27, "unit": "kg", "symbol": "m_p"},
    "neutron_mass": {"value": 1.67492749804e-27, "unit": "kg", "symbol": "m_n"},
    "elementary_charge": {"value": 1.602176634e-19, "unit": "C", "symbol": "e"},
    "boltzmann_constant": {"value": 1.380649e-23, "unit": "J/K", "symbol": "k_B"},
    "avogadro_number": {"value": 6.02214076e23, "unit": "mol⁻¹", "symbol": "N_A"},
    "gas_constant": {"value": 8.314462618, "unit": "J/(mol⋅K)", "symbol": "R"},
    "coulomb_constant": {"value": 8.9875517923e9, "unit": "N⋅m²/C²", "symbol": "k_e"},
    "vacuum_permittivity": {"value": 8.854187817e-12, "unit": "F/m", "symbol": "ε₀"},
    "vacuum_permeability": {"value": 1.25663706212e-6, "unit": "N/A²", "symbol": "μ₀"},
    "electron_volt": {"value": 1.602176634e-19, "unit": "J", "symbol": "eV"},
    "atomic_mass_unit": {"value": 1.66053906660e-27, "unit": "kg", "symbol": "u"},
    "bohr_radius": {"value": 5.29177210903e-11, "unit": "m", "symbol": "a₀"},
    "fine_structure_constant": {"value": 7.2973525693e-3, "unit": "", "symbol": "α"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# الصيغ الفيزيائية
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICS_FORMULAS = {
    "mechanics": {
        "newton_second_law": {
            "formula": "F = m * a",
            "variables": {"F": "force", "m": "mass", "a": "acceleration"},
            "unit": "N"
        },
        "kinetic_energy": {
            "formula": "KE = (1/2) * m * v²",
            "variables": {"KE": "kinetic energy", "m": "mass", "v": "velocity"},
            "unit": "J"
        },
        "potential_energy": {
            "formula": "PE = m * g * h",
            "variables": {"PE": "potential energy", "m": "mass", "g": "gravity", "h": "height"},
            "unit": "J"
        },
        "momentum": {
            "formula": "p = m * v",
            "variables": {"p": "momentum", "m": "mass", "v": "velocity"},
            "unit": "kg⋅m/s"
        },
        "work": {
            "formula": "W = F * d * cos(θ)",
            "variables": {"W": "work", "F": "force", "d": "distance", "θ": "angle"},
            "unit": "J"
        },
        "power": {
            "formula": "P = W / t",
            "variables": {"P": "power", "W": "work", "t": "time"},
            "unit": "W"
        },
        "circular_motion_velocity": {
            "formula": "v = 2 * π * r / T",
            "variables": {"v": "velocity", "r": "radius", "T": "period"},
            "unit": "m/s"
        },
        "centripetal_force": {
            "formula": "F = m * v² / r",
            "variables": {"F": "force", "m": "mass", "v": "velocity", "r": "radius"},
            "unit": "N"
        },
        "gravitational_force": {
            "formula": "F = G * m₁ * m₂ / r²",
            "variables": {"F": "force", "G": "gravitational constant", "m₁": "mass 1", "m₂": "mass 2", "r": "distance"},
            "unit": "N"
        },
        "hookes_law": {
            "formula": "F = -k * x",
            "variables": {"F": "force", "k": "spring constant", "x": "displacement"},
            "unit": "N"
        },
    },
    
    "electromagnetism": {
        "coulombs_law": {
            "formula": "F = k_e * q₁ * q₂ / r²",
            "variables": {"F": "force", "k_e": "Coulomb constant", "q₁": "charge 1", "q₂": "charge 2", "r": "distance"},
            "unit": "N"
        },
        "electric_field": {
            "formula": "E = F / q",
            "variables": {"E": "electric field", "F": "force", "q": "charge"},
            "unit": "N/C"
        },
        "ohms_law": {
            "formula": "V = I * R",
            "variables": {"V": "voltage", "I": "current", "R": "resistance"},
            "unit": "V"
        },
        "electric_power": {
            "formula": "P = V * I",
            "variables": {"P": "power", "V": "voltage", "I": "current"},
            "unit": "W"
        },
        "capacitance": {
            "formula": "C = Q / V",
            "variables": {"C": "capacitance", "Q": "charge", "V": "voltage"},
            "unit": "F"
        },
        "magnetic_force": {
            "formula": "F = q * v * B * sin(θ)",
            "variables": {"F": "force", "q": "charge", "v": "velocity", "B": "magnetic field", "θ": "angle"},
            "unit": "N"
        },
        "faradays_law": {
            "formula": "ε = -dΦ/dt",
            "variables": {"ε": "EMF", "Φ": "magnetic flux", "t": "time"},
            "unit": "V"
        },
    },
    
    "thermodynamics": {
        "ideal_gas_law": {
            "formula": "PV = nRT",
            "variables": {"P": "pressure", "V": "volume", "n": "moles", "R": "gas constant", "T": "temperature"},
            "unit": "Pa⋅m³"
        },
        "first_law": {
            "formula": "ΔU = Q - W",
            "variables": {"ΔU": "change in internal energy", "Q": "heat", "W": "work"},
            "unit": "J"
        },
        "entropy_change": {
            "formula": "ΔS = Q / T",
            "variables": {"ΔS": "entropy change", "Q": "heat", "T": "temperature"},
            "unit": "J/K"
        },
        "thermal_energy": {
            "formula": "Q = m * c * ΔT",
            "variables": {"Q": "heat", "m": "mass", "c": "specific heat", "ΔT": "temperature change"},
            "unit": "J"
        },
    },
    
    "quantum_mechanics": {
        "energy_photon": {
            "formula": "E = h * f",
            "variables": {"E": "energy", "h": "Planck constant", "f": "frequency"},
            "unit": "J"
        },
        "de_broglie_wavelength": {
            "formula": "λ = h / p",
            "variables": {"λ": "wavelength", "h": "Planck constant", "p": "momentum"},
            "unit": "m"
        },
        "heisenberg_uncertainty": {
            "formula": "Δx * Δp ≥ ℏ/2",
            "variables": {"Δx": "position uncertainty", "Δp": "momentum uncertainty", "ℏ": "reduced Planck constant"},
            "unit": "J⋅s"
        },
        "schrodinger_equation": {
            "formula": "iℏ ∂ψ/∂t = Ĥψ",
            "variables": {"ψ": "wave function", "Ĥ": "Hamiltonian", "t": "time"},
            "unit": ""
        },
    },
    
    "relativity": {
        "time_dilation": {
            "formula": "t' = γ * t",
            "variables": {"t'": "dilated time", "t": "proper time", "γ": "Lorentz factor"},
            "unit": "s"
        },
        "length_contraction": {
            "formula": "L' = L / γ",
            "variables": {"L'": "contracted length", "L": "proper length", "γ": "Lorentz factor"},
            "unit": "m"
        },
        "mass_energy": {
            "formula": "E = m * c²",
            "variables": {"E": "energy", "m": "mass", "c": "speed of light"},
            "unit": "J"
        },
        "lorentz_factor": {
            "formula": "γ = 1 / sqrt(1 - v²/c²)",
            "variables": {"γ": "Lorentz factor", "v": "velocity", "c": "speed of light"},
            "unit": ""
        },
    },
    
    "waves": {
        "wave_speed": {
            "formula": "v = f * λ",
            "variables": {"v": "velocity", "f": "frequency", "λ": "wavelength"},
            "unit": "m/s"
        },
        "wave_equation": {
            "formula": "y(x,t) = A * sin(kx - ωt + φ)",
            "variables": {"y": "displacement", "A": "amplitude", "k": "wave number", "ω": "angular frequency", "φ": "phase"},
            "unit": "m"
        },
        "doppler_effect": {
            "formula": "f' = f * (v ± v₀) / (v ∓ v_s)",
            "variables": {"f'": "observed frequency", "f": "source frequency", "v": "wave speed", "v₀": "observer speed", "v_s": "source speed"},
            "unit": "Hz"
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# محرك الفيزياء الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsEngine:
    """محرك الفيزياء المتقدم لـ Edrak AI"""
    
    def __init__(self):
        self.constants = PHYSICAL_CONSTANTS
        self.formulas = PHYSICS_FORMULAS
    
    async def solve(
        self,
        problem: str,
        show_derivation: bool = True,
        language: str = "arabic"
    ) -> Dict:
        """
        حل مسألة فيزيائية
        
        Args:
            problem: نص المسألة
            show_derivation: إظهار الاشتقاق
            language: لغة الإجابة
        """
        
        # تحليل المسألة
        analysis = self._analyze_problem(problem)
        
        # تحديد الموضوع
        topic = analysis.get("topic", "general")
        
        # حل المسألة
        if topic == "mechanics":
            result = self._solve_mechanics(problem, analysis, show_derivation)
        elif topic == "electromagnetism":
            result = self._solve_electromagnetism(problem, analysis, show_derivation)
        elif topic == "thermodynamics"::
            result = self._solve_thermodynamics(problem, analysis, show_derivation)
        elif topic == "quantum_mechanics":
            result = self._solve_quantum(problem, analysis, show_derivation)
        elif topic == "relativity":
            result = self._solve_relativity(problem, analysis, show_derivation)
        elif topic == "waves":
            result = self._solve_waves(problem, analysis, show_derivation)
        else:
            result = self._solve_general(problem, analysis, show_derivation)
        
        # تنسيق الإجابة
        formatted_solution = self._format_solution(result, language)
        
        return {
            "problem": problem,
            "topic": topic,
            "topic_arabic": self._get_topic_arabic(topic),
            "solution": formatted_solution,
            "derivation": result.get("derivation", "") if show_derivation else "",
            "formulas_used": result.get("formulas_used", []),
            "constants_used": result.get("constants_used", []),
            "units": result.get("units", {}),
            "usage": {
                "prompt_tokens": len(problem.split()),
                "completion_tokens": len(formatted_solution.split()),
                "total_tokens": len(problem.split()) + len(formatted_solution.split())
            }
        }
    
    def _analyze_problem(self, problem: str) -> Dict:
        """تحليل المسألة الفيزيائية"""
        
        problem_lower = problem.lower()
        
        analysis = {
            "topic": "general",
            "knowns": {},
            "unknowns": [],
            "formulas_needed": [],
            "constants_needed": []
        }
        
        # اكتشاف الموضوع
        topic_keywords = {
            "mechanics": ["force", "motion", "velocity", "acceleration", "mass", "energy", "work", "power", "momentum", "ق force", "حركة", "سرعة", "تسارع", "كتلة", "طاقة", "شغل", "قدرة"],
            "electromagnetism": ["electric", "magnetic", "charge", "current", "voltage", "resistance", "capacitor", "field", "كهربائي", "مغناطيسي", "شحنة", "تيار", "جهد", "مقاومة"],
            "thermodynamics": ["heat", "temperature", "entropy", "pressure", "volume", "gas", "thermal", "حرارة", "حرارة", "درجة حرارة", "إنتروبيا", "ضغط", "حجم", "غاز"],
            "quantum_mechanics": ["quantum", "photon", "electron", "wave function", "uncertainty", "كم", "فوتون", "إلكترون", "دالة موجية", "عدم يقين"],
            "relativity": ["relativistic", "time dilation", "length contraction", "mass-energy", "Lorentz", "نسبية", "تمدد زمني", "انكماش طول", "طاقة كتلة"],
            "waves": ["wave", "frequency", "wavelength", "amplitude", "interference", "diffraction", "موجة", "تردد", "طول موجة", "سعة", "تداخل", "حيود"],
        }
        
        scores = {topic: 0 for topic in topic_keywords.keys()}
        
        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in problem_lower:
                    scores[topic] += 1
        
        max_topic = max(scores, key=scores.get)
        analysis["topic"] = max_topic if scores[max_topic] > 0 else "general"
        
        # استخراج المعطيات
        value_pattern = r'(\w+)\s*=\s*([\d.]+)\s*(\w+)'
        for match in re.finditer(value_pattern, problem):
            var_name = match.group(1)
            value = float(match.group(2))
            unit = match.group(3)
            analysis["knowns"][var_name] = {"value": value, "unit": unit}
        
        # استخراج المطلوب
        unknown_pattern = r'(find|calculate|determine|احسب|أوجد|حدد)\s+(?:the\s+)?(\w+)'
        for match in re.finditer(unknown_pattern, problem_lower):
            analysis["unknowns"].append(match.group(2))
        
        return analysis
    
    def _solve_mechanics(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل الميكانيكا"""
        
        formulas_used = []
        constants_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # قانون نيوتن الثاني
        if any(word in problem_lower for word in ["force", "acceleration", "ق force", "تسارع"]):
            if "mass" in knowns and "acceleration" in knowns:
                m = knowns["mass"]["value"]
                a = knowns["acceleration"]["value"]
                F = m * a
                
                formulas_used.append(self.formulas["mechanics"]["newton_second_law"])
                
                if show_derivation:
                    derivation_steps.append(f"F = m × a")
                    derivation_steps.append(f"F = {m} kg × {a} m/s²")
                    derivation_steps.append(f"F = {F} N")
                
                solution = f"F = {F} N"
                units = {"F": "N", "m": "kg", "a": "m/s²"}
        
        # الطاقة الحركية
        elif any(word in problem_lower for word in ["kinetic energy", "طاقة حركية"]):
            if "mass" in knowns and "velocity" in knowns:
                m = knowns["mass"]["value"]
                v = knowns["velocity"]["value"]
                KE = 0.5 * m * v**2
                
                formulas_used.append(self.formulas["mechanics"]["kinetic_energy"])
                
                if show_derivation:
                    derivation_steps.append(f"KE = (1/2) × m × v²")
                    derivation_steps.append(f"KE = 0.5 × {m} kg × ({v} m/s)²")
                    derivation_steps.append(f"KE = {KE} J")
                
                solution = f"KE = {KE} J"
                units = {"KE": "J", "m": "kg", "v": "m/s"}
        
        # الطاقة الكامنة
        elif any(word in problem_lower for word in ["potential energy", "طاقة كامنة"]):
            if "mass" in knowns and "height" in knowns:
                m = knowns["mass"]["value"]
                h = knowns["height"]["value"]
                g = 9.81  # m/s²
                PE = m * g * h
                
                formulas_used.append(self.formulas["mechanics"]["potential_energy"])
                constants_used.append(self.constants["gravitational_constant"])
                
                if show_derivation:
                    derivation_steps.append(f"PE = m × g × h")
                    derivation_steps.append(f"PE = {m} kg × {g} m/s² × {h} m")
                    derivation_steps.append(f"PE = {PE} J")
                
                solution = f"PE = {PE} J"
                units = {"PE": "J", "m": "kg", "g": "m/s²", "h": "m"}
        
        # القوة الجاذبية
        elif any(word in problem_lower for word in ["gravitational force", "gravity", "جاذبية"]):
            if "mass1" in knowns and "mass2" in knowns and "distance" in knowns:
                m1 = knowns["mass1"]["value"]
                m2 = knowns["mass2"]["value"]
                r = knowns["distance"]["value"]
                G = self.constants["gravitational_constant"]["value"]
                
                F = G * m1 * m2 / r**2
                
                formulas_used.append(self.formulas["mechanics"]["gravitational_force"])
                constants_used.append(self.constants["gravitational_constant"])
                
                if show_derivation:
                    derivation_steps.append(f"F = G × m₁ × m₂ / r²")
                    derivation_steps.append(f"F = {G} × {m1} × {m2} / {r}²")
                    derivation_steps.append(f"F = {F:.4e} N")
                
                solution = f"F = {F:.4e} N"
                units = {"F": "N", "G": "N⋅m²/kg²", "m": "kg", "r": "m"}
        
        else:
            solution = "Mechanics problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": constants_used,
            "units": units
        }
    
    def _solve_electromagnetism(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل الكهرومغناطيسية"""
        
        formulas_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # قانون أوم
        if any(word in problem_lower for word in ["ohm", "voltage", "current", "resistance", "أوم", "جهد", "تيار", "مقاومة"]):
            if "voltage" in knowns and "current" in knowns:
                V = knowns["voltage"]["value"]
                I = knowns["current"]["value"]
                R = V / I
                
                formulas_used.append(self.formulas["electromagnetism"]["ohms_law"])
                
                if show_derivation:
                    derivation_steps.append(f"V = I × R")
                    derivation_steps.append(f"R = V / I")
                    derivation_steps.append(f"R = {V} V / {I} A")
                    derivation_steps.append(f"R = {R} Ω")
                
                solution = f"R = {R} Ω"
                units = {"V": "V", "I": "A", "R": "Ω"}
            
            elif "current" in knowns and "resistance" in knowns:
                I = knowns["current"]["value"]
                R = knowns["resistance"]["value"]
                V = I * R
                
                formulas_used.append(self.formulas["electromagnetism"]["ohms_law"])
                
                if show_derivation:
                    derivation_steps.append(f"V = I × R")
                    derivation_steps.append(f"V = {I} A × {R} Ω")
                    derivation_steps.append(f"V = {V} V")
                
                solution = f"V = {V} V"
                units = {"V": "V", "I": "A", "R": "Ω"}
            
            else:
                solution = "Ohm's law problem - need more data"
                units = {}
        
        # قانون كولوم
        elif any(word in problem_lower for word in ["coulomb", "electric force", "كولوم", "قوة كهربائية"]):
            if "charge1" in knowns and "charge2" in knowns and "distance" in knowns:
                q1 = knowns["charge1"]["value"]
                q2 = knowns["charge2"]["value"]
                r = knowns["distance"]["value"]
                k_e = self.constants["coulomb_constant"]["value"]
                
                F = k_e * q1 * q2 / r**2
                
                formulas_used.append(self.formulas["electromagnetism"]["coulombs_law"])
                
                if show_derivation:
                    derivation_steps.append(f"F = k_e × q₁ × q₂ / r²")
                    derivation_steps.append(f"F = {k_e:.4e} × {q1} × {q2} / {r}²")
                    derivation_steps.append(f"F = {F:.4e} N")
                
                solution = f"F = {F:.4e} N"
                units = {"F": "N", "q": "C", "r": "m"}
            
            else:
                solution = "Coulomb's law problem - need charges and distance"
                units = {}
        
        else:
            solution = "Electromagnetism problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": [],
            "units": units
        }
    
    def _solve_thermodynamics(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل الديناميكا الحرارية"""
        
        formulas_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # قانون الغاز المثالي
        if any(word in problem_lower for word in ["ideal gas", "gas law", "غاز مثالي", "قانون الغاز"]):
            if "pressure" in knowns and "volume" in knowns and "temperature" in knowns:
                P = knowns["pressure"]["value"]
                V = knowns["volume"]["value"]
                T = knowns["temperature"]["value"]
                R = self.constants["gas_constant"]["value"]
                
                n = (P * V) / (R * T)
                
                formulas_used.append(self.formulas["thermodynamics"]["ideal_gas_law"])
                
                if show_derivation:
                    derivation_steps.append(f"PV = nRT")
                    derivation_steps.append(f"n = PV / RT")
                    derivation_steps.append(f"n = ({P} × {V}) / ({R} × {T})")
                    derivation_steps.append(f"n = {n:.4f} mol")
                
                solution = f"n = {n:.4f} mol"
                units = {"P": "Pa", "V": "m³", "n": "mol", "T": "K"}
            
            else:
                solution = "Ideal gas law problem - need P, V, and T"
                units = {}
        
        # الطاقة الحرارية
        elif any(word in problem_lower for word in ["heat", "thermal energy", "حرارة", "طاقة حرارية"]):
            if "mass" in knowns and "specific_heat" in knowns and "temperature_change" in knowns:
                m = knowns["mass"]["value"]
                c = knowns["specific_heat"]["value"]
                delta_T = knowns["temperature_change"]["value"]
                
                Q = m * c * delta_T
                
                formulas_used.append(self.formulas["thermodynamics"]["thermal_energy"])
                
                if show_derivation:
                    derivation_steps.append(f"Q = m × c × ΔT")
                    derivation_steps.append(f"Q = {m} kg × {c} J/(kg⋅K) × {delta_T} K")
                    derivation_steps.append(f"Q = {Q} J")
                
                solution = f"Q = {Q} J"
                units = {"Q": "J", "m": "kg", "c": "J/(kg⋅K)", "ΔT": "K"}
            
            else:
                solution = "Thermal energy problem - need mass, specific heat, and temperature change"
                units = {}
        
        else:
            solution = "Thermodynamics problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": [],
            "units": units
        }
    
    def _solve_quantum(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل ميكانيكا الكم"""
        
        formulas_used = []
        constants_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # طاقة الفوتون
        if any(word in problem_lower for word in ["photon energy", "photon", "فوتون", "طاقة فوتون"]):
            if "frequency" in knowns:
                f = knowns["frequency"]["value"]
                h = self.constants["planck_constant"]["value"]
                
                E = h * f
                
                formulas_used.append(self.formulas["quantum_mechanics"]["energy_photon"])
                constants_used.append(self.constants["planck_constant"])
                
                if show_derivation:
                    derivation_steps.append(f"E = h × f")
                    derivation_steps.append(f"E = {h:.4e} J⋅s × {f} Hz")
                    derivation_steps.append(f"E = {E:.4e} J")
                    derivation_steps.append(f"E = {E / self.constants['electron_volt']['value']:.4f} eV")
                
                solution = f"E = {E:.4e} J = {E / self.constants['electron_volt']['value']:.4f} eV"
                units = {"E": "J", "h": "J⋅s", "f": "Hz"}
            
            else:
                solution = "Photon energy problem - need frequency"
                units = {}
        
        # طول موجة دي برولي
        elif any(word in problem_lower for word in ["de broglie", "wavelength", "دي برولي", "طول موجة"]):
            if "momentum" in knowns:
                p = knowns["momentum"]["value"]
                h = self.constants["planck_constant"]["value"]
                
                wavelength = h / p
                
                formulas_used.append(self.formulas["quantum_mechanics"]["de_broglie_wavelength"])
                constants_used.append(self.constants["planck_constant"])
                
                if show_derivation:
                    derivation_steps.append(f"λ = h / p")
                    derivation_steps.append(f"λ = {h:.4e} J⋅s / {p} kg⋅m/s")
                    derivation_steps.append(f"λ = {wavelength:.4e} m")
                
                solution = f"λ = {wavelength:.4e} m"
                units = {"λ": "m", "h": "J⋅s", "p": "kg⋅m/s"}
            
            else:
                solution = "De Broglie wavelength problem - need momentum"
                units = {}
        
        else:
            solution = "Quantum mechanics problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": constants_used,
            "units": units
        }
    
    def _solve_relativity(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل النسبية"""
        
        formulas_used = []
        constants_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # طاقة الكتلة
        if any(word in problem_lower for word in ["mass-energy", "einstein", "طاقة كتلة", "أينشتاين"]):
            if "mass" in knowns:
                m = knowns["mass"]["value"]
                c = self.constants["speed_of_light"]["value"]
                
                E = m * c**2
                
                formulas_used.append(self.formulas["relativity"]["mass_energy"])
                constants_used.append(self.constants["speed_of_light"])
                
                if show_derivation:
                    derivation_steps.append(f"E = m × c²")
                    derivation_steps.append(f"E = {m} kg × ({c} m/s)²")
                    derivation_steps.append(f"E = {E:.4e} J")
                
                solution = f"E = {E:.4e} J"
                units = {"E": "J", "m": "kg", "c": "m/s"}
            
            else:
                solution = "Mass-energy problem - need mass"
                units = {}
        
        # عامل لورنتز
        elif any(word in problem_lower for word in ["lorentz factor", "time dilation", "عامل لورنتز", "تمدد زمني"]):
            if "velocity" in knowns:
                v = knowns["velocity"]["value"]
                c = self.constants["speed_of_light"]["value"]
                
                if v < c:
                    gamma = 1 / np.sqrt(1 - v**2 / c**2)
                    
                    formulas_used.append(self.formulas["relativity"]["lorentz_factor"])
                    constants_used.append(self.constants["speed_of_light"])
                    
                    if show_derivation:
                        derivation_steps.append(f"γ = 1 / √(1 - v²/c²)")
                        derivation_steps.append(f"γ = 1 / √(1 - {v}²/{c}²)")
                        derivation_steps.append(f"γ = {gamma:.6f}")
                    
                    solution = f"γ = {gamma:.6f}"
                    units = {"γ": "", "v": "m/s", "c": "m/s"}
                else:
                    solution = "Velocity cannot exceed speed of light!"
                    units = {}
            
            else:
                solution = "Lorentz factor problem - need velocity"
                units = {}
        
        else:
            solution = "Relativity problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": constants_used,
            "units": units
        }
    
    def _solve_waves(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل مسائل الموجات"""
        
        formulas_used = []
        derivation_steps = []
        
        problem_lower = problem.lower()
        knowns = analysis.get("knowns", {})
        
        # سرعة الموجة
        if any(word in problem_lower for word in ["wave speed", "velocity", "سرعة موجة", "سرعة"]):
            if "frequency" in knowns and "wavelength" in knowns:
                f = knowns["frequency"]["value"]
                wavelength = knowns["wavelength"]["value"]
                
                v = f * wavelength
                
                formulas_used.append(self.formulas["waves"]["wave_speed"])
                
                if show_derivation:
                    derivation_steps.append(f"v = f × λ")
                    derivation_steps.append(f"v = {f} Hz × {wavelength} m")
                    derivation_steps.append(f"v = {v} m/s")
                
                solution = f"v = {v} m/s"
                units = {"v": "m/s", "f": "Hz", "λ": "m"}
            
            else:
                solution = "Wave speed problem - need frequency and wavelength"
                units = {}
        
        else:
            solution = "Waves problem - specific formula needed"
            units = {}
        
        return {
            "solution": solution,
            "derivation": "\n".join(derivation_steps) if show_derivation else "",
            "formulas_used": formulas_used,
            "constants_used": [],
            "units": units
        }
    
    def _solve_general(self, problem: str, analysis: Dict, show_derivation: bool) -> Dict:
        """حل عام للمسائل الفيزيائية"""
        
        return {
            "solution": "General physics problem - please specify the topic (mechanics, electromagnetism, thermodynamics, quantum mechanics, relativity, or waves)",
            "derivation": "",
            "formulas_used": [],
            "constants_used": [],
            "units": {}
        }
    
    def _get_topic_arabic(self, topic: str) -> str:
        """الحصول على اسم الموضوع بالعربية"""
        
        arabic_names = {
            "mechanics": "الميكانيكا",
            "electromagnetism": "الكهرومغناطيسية",
            "thermodynamics": "الديناميكا الحرارية",
            "quantum_mechanics": "ميكانيكا الكم",
            "relativity": "النسبية",
            "waves": "الموجات",
            "general": "عام"
        }
        
        return arabic_names.get(topic, topic)
    
    def _format_solution(self, result: Dict, language: str) -> str:
        """تنسيق الحل"""
        
        solution = result.get("solution", "")
        
        if language == "arabic":
            # ترجمة المصطلحات
            solution = solution.replace("=", "=")
        
        return solution
