"""
Edrak AI - إدراك
نظام الذكاء الاصطناعي المتقدم

المخترع: يوسف السيد الغريب (Youssef Elsayed Elghareeb)
"""

__version__ = "1.0.0-alpha"
__author__ = "Youssef Elsayed Elghareeb"
__author_arabic__ = "يوسف السيد الغريب"

from .model.edrak_transformer import EdrakTransformer, EdrakConfig
from .model.arabic_processor import ArabicProcessor
from .model.code_engine import CodeEngine
from .model.math_solver import MathSolver
from .model.physics_engine import PhysicsEngine
from .model.graphics3d import Graphics3D

__all__ = [
    "EdrakTransformer",
    "EdrakConfig",
    "ArabicProcessor",
    "CodeEngine",
    "MathSolver",
    "PhysicsEngine",
    "Graphics3D",
]
