"""
Graphics 3D - نظام الرسومات ثلاثية الأبعاد

يدعم:
- النمذجة ثلاثية الأبعاد
- التصيير (Rendering)
- الرسوم المتحركة
- Shaders
- تصدير إلى تنسيقات متعددة
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# أشكال أساسية ثلاثية الأبعاد
# ═══════════════════════════════════════════════════════════════════════════════

PRIMITIVE_SHAPES = {
    "cube": {
        "vertices": [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Front face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],      # Back face
        ],
        "faces": [
            [0, 1, 2, 3],  # Front
            [4, 5, 6, 7],  # Back
            [0, 1, 5, 4],  # Bottom
            [2, 3, 7, 6],  # Top
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5],  # Right
        ],
        "normals": [
            [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]
        ]
    },
    
    "sphere": {
        "type": "parametric",
        "function": "sphere",
        "params": {"radius": 1, "segments": 32, "rings": 16}
    },
    
    "cylinder": {
        "type": "parametric",
        "function": "cylinder",
        "params": {"radius": 1, "height": 2, "segments": 32}
    },
    
    "cone": {
        "type": "parametric",
        "function": "cone",
        "params": {"radius": 1, "height": 2, "segments": 32}
    },
    
    "torus": {
        "type": "parametric",
        "function": "torus",
        "params": {"major_radius": 1, "minor_radius": 0.4, "major_segments": 32, "minor_segments": 16}
    },
    
    "plane": {
        "vertices": [
            [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]
        ],
        "faces": [[0, 1, 2, 3]],
        "normals": [[0, 1, 0]]
    },
    
    "pyramid": {
        "vertices": [
            [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1],  # Base
            [0, 2, 0]  # Apex
        ],
        "faces": [
            [0, 1, 2, 3],  # Base
            [0, 1, 4],     # Side 1
            [1, 2, 4],     # Side 2
            [2, 3, 4],     # Side 3
            [3, 0, 4],     # Side 4
        ],
        "normals": [
            [0, -1, 0], [0, 0.707, 0.707], [0.707, 0.707, 0], [0, 0.707, -0.707], [-0.707, 0.707, 0]
        ]
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# قوالب Three.js
# ═══════════════════════════════════════════════════════════════════════════════

THREE_JS_TEMPLATES = {
    "scene_setup": """
// إعداد المشهد
const scene = new THREE.Scene();
scene.background = new THREE.Color({background_color});

// الكاميرا
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set({camera_x}, {camera_y}, {camera_z});

// المُصيِّر
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

// الإضاءة
const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(5, 10, 7);
directionalLight.castShadow = true;
scene.add(directionalLight);

// التحكم
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
""",
    
    "mesh_creation": """
// إنشاء {shape_name}
const {variable_name}Geometry = new THREE.{geometry_type}({geometry_params});
const {variable_name}Material = new THREE.{material_type}({{
    color: {color},
    {material_properties}
}});
const {variable_name} = new THREE.Mesh({variable_name}Geometry, {variable_name}Material);
{variable_name}.position.set({position_x}, {position_y}, {position_z});
{variable_name}.rotation.set({rotation_x}, {rotation_y}, {rotation_z});
{variable_name}.scale.set({scale_x}, {scale_y}, {scale_z});
{variable_name}.castShadow = true;
{variable_name}.receiveShadow = true;
scene.add({variable_name});
""",
    
    "animation_loop": """
// حلقة الرسوم المتحركة
function animate() {{
    requestAnimationFrame(animate);
    
    {animation_code}
    
    controls.update();
    renderer.render(scene, camera);
}}

animate();

// التعامل مع تغيير حجم النافذة
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
""",
}

# ═══════════════════════════════════════════════════════════════════════════════
# نظام الرسومات 3D الرئيسي
# ═══════════════════════════════════════════════════════════════════════════════

class Graphics3D:
    """نظام الرسومات ثلاثية الأبعاد لـ Edrak AI"""
    
    def __init__(self):
        self.primitives = PRIMITIVE_SHAPES
        self.templates = THREE_JS_TEMPLATES
    
    async def generate(
        self,
        description: str,
        output_format: str = "three.js",
        include_animation: bool = False,
        include_materials: bool = True,
        include_lighting: bool = True
    ) -> Dict:
        """
        توليد مشهد ثلاثي الأبعاد
        
        Args:
            description: وصف المشهد
            output_format: تنسيق الإخراج (three.js, babylon.js, unity, gltf)
            include_animation: تضمين رسوم متحركة
            include_materials: تضمين مواد
            include_lighting: تضمين إضاءة
        """
        
        # تحليل الوصف
        scene_description = self._parse_description(description)
        
        # توليد الكود حسب التنسيق
        if output_format == "three.js":
            code = self._generate_three_js(scene_description, include_animation, include_materials, include_lighting)
        elif output_format == "babylon.js":
            code = self._generate_babylon_js(scene_description, include_animation, include_materials, include_lighting)
        elif output_format == "unity":
            code = self._generate_unity(scene_description, include_animation)
        elif output_format == "gltf":
            code = self._generate_gltf(scene_description)
        else:
            code = self._generate_three_js(scene_description, include_animation, include_materials, include_lighting)
        
        # حساب الإحصائيات
        vertices = self._count_vertices(scene_description)
        faces = self._count_faces(scene_description)
        
        return {
            "code": code,
            "format": output_format,
            "description": self._generate_description(scene_description),
            "vertices": vertices,
            "faces": faces,
            "objects": len(scene_description.get("objects", [])),
            "materials": len(scene_description.get("materials", [])),
            "lights": len(scene_description.get("lights", [])),
            "usage": {
                "prompt_tokens": len(description.split()),
                "completion_tokens": len(code.split()),
                "total_tokens": len(description.split()) + len(code.split())
            }
        }
    
    def _parse_description(self, description: str) -> Dict:
        """تحليل وصف المشهد"""
        
        description_lower = description.lower()
        
        scene = {
            "objects": [],
            "materials": [],
            "lights": [],
            "camera": {"position": [0, 5, 10], "target": [0, 0, 0]},
            "background": "#000000",
            "animation": []
        }
        
        # اكتشاف الأشكال
        shape_keywords = {
            "cube": ["cube", "مكعب", "box", "صندوق"],
            "sphere": ["sphere", "كرة", "ball", "كرة"],
            "cylinder": ["cylinder", "أسطوانة", "tube", "أنبوب"],
            "cone": ["cone", "مخروط", "pyramid", "هرم"],
            "torus": ["torus", "حلقة", "donut", "دونات"],
            "plane": ["plane", "مستوى", "ground", "أرض"],
        }
        
        for shape, keywords in shape_keywords.items():
            for kw in keywords:
                if kw in description_lower:
                    scene["objects"].append({
                        "type": shape,
                        "name": f"{shape}_1",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "scale": [1, 1, 1],
                        "color": self._extract_color(description) or "#3498db",
                        "material": "standard"
                    })
                    break
        
        # إذا لم يُعثر على أشكال، أضف مكعب افتراضي
        if not scene["objects"]:
            scene["objects"].append({
                "type": "cube",
                "name": "default_cube",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "color": "#3498db",
                "material": "standard"
            })
        
        # اكتشاف الإضاءة
        if "light" in description_lower or "إضاءة" in description_lower:
            scene["lights"].append({
                "type": "directional",
                "position": [5, 10, 7],
                "color": "#ffffff",
                "intensity": 1
            })
        
        # اكتشاف الرسوم المتحركة
        if "rotate" in description_lower or "spin" in description_lower or "يدور" in description_lower:
            scene["animation"].append({
                "type": "rotation",
                "target": scene["objects"][0]["name"],
                "axis": [0, 1, 0],
                "speed": 0.01
            })
        
        return scene
    
    def _extract_color(self, description: str) -> Optional[str]:
        """استخراج اللون من الوصف"""
        
        color_map = {
            "red": "#e74c3c",
            "أحمر": "#e74c3c",
            "blue": "#3498db",
            "أزرق": "#3498db",
            "green": "#2ecc71",
            "أخضر": "#2ecc71",
            "yellow": "#f1c40f",
            "أصفر": "#f1c40f",
            "orange": "#e67e22",
            "برتقالي": "#e67e22",
            "purple": "#9b59b6",
            "بنفسجي": "#9b59b6",
            "white": "#ffffff",
            "أبيض": "#ffffff",
            "black": "#000000",
            "أسود": "#000000",
            "gray": "#95a5a6",
            "رمادي": "#95a5a6",
        }
        
        description_lower = description.lower()
        
        for color_name, hex_code in color_map.items():
            if color_name in description_lower:
                return hex_code
        
        return None
    
    def _generate_three_js(
        self,
        scene: Dict,
        include_animation: bool,
        include_materials: bool,
        include_lighting: bool
    ) -> str:
        """توليد كود Three.js"""
        
        code_parts = []
        
        # إعداد المشهد
        scene_setup = self.templates["scene_setup"].format(
            background_color=scene.get("background", "#000000"),
            camera_x=scene["camera"]["position"][0],
            camera_y=scene["camera"]["position"][1],
            camera_z=scene["camera"]["position"][2]
        )
        code_parts.append(scene_setup)
        
        # إنشاء الأشكال
        for obj in scene.get("objects", []):
            shape_type = obj["type"]
            
            # تحديد نوع الهندسة
            geometry_map = {
                "cube": "BoxGeometry",
                "sphere": "SphereGeometry",
                "cylinder": "CylinderGeometry",
                "cone": "ConeGeometry",
                "torus": "TorusGeometry",
                "plane": "PlaneGeometry",
                "pyramid": "ConeGeometry"
            }
            
            geometry_type = geometry_map.get(shape_type, "BoxGeometry")
            
            # تحديد معاملات الهندسة
            if shape_type == "cube":
                geometry_params = "2, 2, 2"
            elif shape_type == "sphere":
                geometry_params = "1, 32, 16"
            elif shape_type == "cylinder":
                geometry_params = "1, 1, 2, 32"
            elif shape_type == "cone":
                geometry_params = "1, 2, 32"
            elif shape_type == "torus":
                geometry_params = "1, 0.4, 16, 100"
            elif shape_type == "plane":
                geometry_params = "10, 10"
            else:
                geometry_params = "1, 1, 1"
            
            # تحديد المادة
            material_type = "MeshStandardMaterial" if include_materials else "MeshBasicMaterial"
            material_properties = "roughness: 0.5, metalness: 0.5" if include_materials else ""
            
            mesh_code = self.templates["mesh_creation"].format(
                shape_name=shape_type,
                variable_name=obj["name"],
                geometry_type=geometry_type,
                geometry_params=geometry_params,
                material_type=material_type,
                color=f"0x{obj['color'][1:]}" if obj['color'].startswith('#') else obj['color'],
                material_properties=material_properties,
                position_x=obj["position"][0],
                position_y=obj["position"][1],
                position_z=obj["position"][2],
                rotation_x=obj["rotation"][0],
                rotation_y=obj["rotation"][1],
                rotation_z=obj["rotation"][2],
                scale_x=obj["scale"][0],
                scale_y=obj["scale"][1],
                scale_z=obj["scale"][2]
            )
            code_parts.append(mesh_code)
        
        # حلقة الرسوم المتحركة
        animation_code = ""
        if include_animation and scene.get("animation"):
            for anim in scene["animation"]:
                if anim["type"] == "rotation":
                    animation_code += f"""
    {anim['target']}.rotation.x += {anim['speed']};
    {anim['target']}.rotation.y += {anim['speed']};"""
        
        animation_loop = self.templates["animation_loop"].format(
            animation_code=animation_code
        )
        code_parts.append(animation_loop)
        
        return "\n".join(code_parts)
    
    def _generate_babylon_js(
        self,
        scene: Dict,
        include_animation: bool,
        include_materials: bool,
        include_lighting: bool
    ) -> str:
        """توليد كود Babylon.js"""
        
        code = """// إعداد المشهد
const canvas = document.getElementById("renderCanvas");
const engine = new BABYLON.Engine(canvas, true);

const createScene = function() {
    const scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color3(0, 0, 0);
    
    // الكاميرا
    const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2.5, 10, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    
    // الإضاءة
    const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
    
"""
        
        # إضافة الأشكال
        for obj in scene.get("objects", []):
            shape_type = obj["type"]
            
            if shape_type == "cube":
                code += f"""
    // {shape_type}
    const {obj['name']} = BABYLON.MeshBuilder.CreateBox("{obj['name']}", {{size: 2}}, scene);
    {obj['name']}.position = new BABYLON.Vector3({obj['position'][0]}, {obj['position'][1]}, {obj['position'][2]});
"""
            elif shape_type == "sphere":
                code += f"""
    // {shape_type}
    const {obj['name']} = BABYLON.MeshBuilder.CreateSphere("{obj['name']}", {{diameter: 2}}, scene);
    {obj['name']}.position = new BABYLON.Vector3({obj['position'][0]}, {obj['position'][1]}, {obj['position'][2]});
"""
            elif shape_type == "cylinder":
                code += f"""
    // {shape_type}
    const {obj['name']} = BABYLON.MeshBuilder.CreateCylinder("{obj['name']}", {{height: 2, diameter: 2}}, scene);
    {obj['name']}.position = new BABYLON.Vector3({obj['position'][0]}, {obj['position'][1]}, {obj['position'][2]});
"""
            
            # إضافة المادة
            if include_materials:
                code += f"""
    const {obj['name']}Mat = new BABYLON.StandardMaterial("{obj['name']}Mat", scene);
    {obj['name']}Mat.diffuseColor = new BABYLON.Color3.FromHexString("{obj['color']}");
    {obj['name']}.material = {obj['name']}Mat;
"""
        
        code += """
    return scene;
};

const scene = createScene();

engine.runRenderLoop(function() {
    scene.render();
});

window.addEventListener("resize", function() {
    engine.resize();
});
"""
        
        return code
    
    def _generate_unity(self, scene: Dict, include_animation: bool) -> str:
        """توليد كود Unity (C#)"""
        
        code = """using UnityEngine;

public class SceneGenerator : MonoBehaviour
{
    void Start()
    {
        // إعداد الكاميرا
        Camera.main.transform.position = new Vector3(0, 5, -10);
        Camera.main.transform.LookAt(Vector3.zero);
        
        // إضاءة
        GameObject lightObj = new GameObject("Directional Light");
        Light light = lightObj.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1;
        lightObj.transform.rotation = Quaternion.Euler(50, -30, 0);
        
"""
        
        for obj in scene.get("objects", []):
            shape_type = obj["type"]
            
            code += f"""
        // إنشاء {shape_type}
        GameObject {obj['name']} = GameObject.CreatePrimitive(PrimitiveType.{shape_type.capitalize()});
        {obj['name']}.transform.position = new Vector3({obj['position'][0]}f, {obj['position'][1]}f, {obj['position'][2]}f);
        {obj['name']}.transform.localScale = new Vector3({obj['scale'][0]}f, {obj['scale'][1]}f, {obj['scale'][2]}f);
        
        // لون
        Renderer {obj['name']}Renderer = {obj['name']}.GetComponent<Renderer>();
        {obj['name']}Renderer.material.color = new Color({int(obj['color'][1:3], 16)/255}f, {int(obj['color'][3:5], 16)/255}f, {int(obj['color'][5:7], 16)/255}f);
"""
        
        code += """
    }
}
"""
        
        return code
    
    def _generate_gltf(self, scene: Dict) -> str:
        """توليد ملف glTF"""
        
        # تبسيط - إرجاع هيكل glTF أساسي
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "Edrak AI 3D Engine"
            },
            "scene": 0,
            "scenes": [{"nodes": list(range(len(scene.get("objects", []))))}],
            "nodes": [],
            "meshes": [],
            "buffers": [],
            "bufferViews": [],
            "accessors": []
        }
        
        for i, obj in enumerate(scene.get("objects", [])):
            gltf["nodes"].append({
                "mesh": i,
                "translation": obj["position"]
            })
            
            gltf["meshes"].append({
                "primitives": [{
                    "attributes": {"POSITION": i}
                }]
            })
        
        return json.dumps(gltf, indent=2)
    
    def _generate_description(self, scene: Dict) -> str:
        """توليد وصف نصي للمشهد"""
        
        objects_desc = []
        for obj in scene.get("objects", []):
            objects_desc.append(f"{obj['type']} باللون {obj['color']}")
        
        return f"مشهد ثلاثي الأبعاد يحتوي على: {', '.join(objects_desc)}"
    
    def _count_vertices(self, scene: Dict) -> int:
        """حساب عدد الرؤوس"""
        
        total = 0
        for obj in scene.get("objects", []):
            shape_type = obj["type"]
            if shape_type == "cube":
                total += 8
            elif shape_type == "sphere":
                total += 33 * 17  # تقريبي
            elif shape_type == "cylinder":
                total += 66  # تقريبي
            elif shape_type == "cone":
                total += 33  # تقريبي
            elif shape_type == "torus":
                total += 33 * 17  # تقريبي
            elif shape_type == "plane":
                total += 4
        
        return total
    
    def _count_faces(self, scene: Dict) -> int:
        """حساب عدد الأوجه"""
        
        total = 0
        for obj in scene.get("objects", []):
            shape_type = obj["type"]
            if shape_type == "cube":
                total += 12
            elif shape_type == "sphere":
                total += 33 * 16 * 2  # تقريبي
            elif shape_type == "cylinder":
                total += 64  # تقريبي
            elif shape_type == "cone":
                total += 32  # تقريبي
            elif shape_type == "torus":
                total += 32 * 16 * 2  # تقريبي
            elif shape_type == "plane":
                total += 2
        
        return total
    
    def generate_sphere(self, radius: float = 1, segments: int = 32, rings: int = 16) -> Dict:
        """توليد كرة"""
        
        vertices = []
        faces = []
        
        for ring in range(rings + 1):
            phi = np.pi * ring / rings
            for segment in range(segments + 1):
                theta = 2 * np.pi * segment / segments
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.cos(phi)
                z = radius * np.sin(phi) * np.sin(theta)
                
                vertices.append([x, y, z])
        
        for ring in range(rings):
            for segment in range(segments):
                current = ring * (segments + 1) + segment
                next_seg = current + 1
                next_ring = (ring + 1) * (segments + 1) + segment
                next_ring_seg = next_ring + 1
                
                faces.append([current, next_seg, next_ring_seg, next_ring])
        
        return {"vertices": vertices, "faces": faces}
    
    def generate_cylinder(self, radius: float = 1, height: float = 2, segments: int = 32) -> Dict:
        """توليد أسطوانة"""
        
        vertices = []
        faces = []
        
        # القاعدة السفلية
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            vertices.append([x, -height/2, z])
        
        # القاعدة العلوية
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            vertices.append([x, height/2, z])
        
        # المركز
        vertices.append([0, -height/2, 0])  # قاعدة سفلية
        vertices.append([0, height/2, 0])   # قاعدة علوية
        
        # الأوجه الجانبية
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i, next_i, next_i + segments, i + segments])
        
        # أوجه القاعدة السفلية
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i, next_i, 2 * segments])
        
        # أوجه القاعدة العلوية
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i + segments, next_i + segments, 2 * segments + 1])
        
        return {"vertices": vertices, "faces": faces}
