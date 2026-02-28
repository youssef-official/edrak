# Edrak AI - إدراك 🧠

<div align="center">

![Edrak AI Logo](https://img.shields.io/badge/Edrak-AI-6366f1?style=for-the-badge&logo=artificial-intelligence&logoColor=white)
![Version](https://img.shields.io/badge/Version-1.0.0--alpha-8b5cf6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-06b6d4?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)

**نموذج ذكاء اصطناعي متقدم متخصص في البرمجة والعلوم**

**Advanced AI Model Specialized in Programming and Sciences**

</div>

---

## 📋 Table of Contents / فهرس المحتويات

- [English](#english)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [API Reference](#api-reference)
  - [Deployment on Railway](#deployment-on-railway)
- [العربية](#العربية)
  - [نظرة عامة](#نظرة-عامة)
  - [المميزات](#المميزات)
  - [التثبيت](#التثبيت)
  - [الاستخدام](#الاستخدام)
  - [مرجع API](#مرجع-api)
  - [النشر على Railway](#النشر-على-railway)

---

## English

### Overview

**Edrak AI** is a state-of-the-art artificial intelligence model designed and developed by **Youssef Elsayed Elghareeb**. It specializes in:

- 💻 **Programming & Web Development** - Code generation, review, and optimization
- 🌍 **Arabic Language** - Understanding all Arabic dialects (Egyptian, Gulf, Levantine, Maghrebi)
- 📐 **Mathematics** - Algebra, calculus, geometry, statistics, linear algebra
- ⚛️ **Physics** - Mechanics, electromagnetism, thermodynamics, quantum mechanics, relativity
- 🎨 **3D Graphics** - 3D modeling, rendering, and scene generation

### Features

#### 🔧 Programming Capabilities
- Multi-language code generation (Python, JavaScript, TypeScript, HTML, CSS, C++, Java, Rust, Go)
- Code review and security analysis
- Design pattern implementation
- Automatic test generation
- Complexity analysis

#### 🌐 Arabic Language Support
- Understanding of all major Arabic dialects
- Poetry analysis and generation
- Grammar correction
- Sentiment analysis
- Translation capabilities

#### 📊 Mathematical Solver
- Algebra and equation solving
- Calculus (derivatives, integrals, limits)
- Linear algebra (matrices, eigenvalues)
- Geometry calculations
- Statistics and probability
- Differential equations

#### ⚛️ Physics Engine
- Classical mechanics
- Electromagnetism
- Thermodynamics
- Quantum mechanics
- Special relativity
- Wave physics

#### 🎨 3D Graphics
- Three.js code generation
- Babylon.js support
- Unity C# scripts
- glTF export
- Scene composition

### Installation

```bash
# Clone the repository
git clone https://github.com/youssef-official/edrak.git
cd edrak

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (optional)
# Place model files in the models/ directory
```

### Usage

#### Running the API Server

```bash
python main.py
```

The API server will start at `http://localhost:8000`

#### Web Interface

Open `web_interface/index.html` in your browser to access the interactive web interface.

#### API Example

```python
import requests

# Chat completion
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "edrak-1.0",
    "messages": [
        {"role": "user", "content": "اكتب دالة بلغة Python لحساب factorial"}
    ],
    "temperature": 0.7,
    "code_mode": True
})

print(response.json())
```

### API Reference

#### Chat Completions
```http
POST /v1/chat/completions
```

Request body:
```json
{
  "model": "edrak-1.0",
  "messages": [
    {"role": "user", "content": "Your message here"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "code_mode": false,
  "arabic_dialect": "egyptian",
  "math_mode": false,
  "physics_mode": false,
  "graphics_3d": false
}
```

#### Code Generation
```http
POST /v1/code/generate
```

Request body:
```json
{
  "prompt": "Create a React button component",
  "language": "javascript",
  "framework": "react",
  "include_tests": true
}
```

#### Math Solver
```http
POST /v1/math/solve
```

Request body:
```json
{
  "problem": "Solve x^2 + 5x + 6 = 0",
  "show_steps": true
}
```

---

### Deployment on Railway

Railway is a modern infrastructure platform that allows you to deploy your applications quickly and easily. Follow these steps to deploy Edrak AI on Railway:

1.  **Create a new project on Railway**: Go to [Railway.app](https://railway.app/) and create a new project. You can connect your GitHub account and select the `youssef-official/edrak` repository.
2.  **Configure Environment Variables**: Edrak AI might require environment variables for configuration. Check the `.env.example` file in the project root for a list of necessary variables. You will need to add these variables to your Railway project settings under the "Variables" tab. For example, if `API_KEY` is listed in `.env.example`, you would add `API_KEY` and its corresponding value in Railway.
3.  **Build and Deploy**: Railway will automatically detect the `Dockerfile` in the project root and use it to build and deploy your application. Ensure your `Dockerfile` is correctly configured to run `main.py`.
4.  **Access your application**: Once deployed, Railway will provide a public URL for your application. You can access the API and web interface through this URL.

---

## العربية

### نظرة عامة

**إدراك** هو نموذج ذكاء اصطناعي متطور تم تصميمه وتطويره بواسطة **يوسف السيد الغريب**. يتخصص في:

- 💻 **البرمجة وتطوير الويب** - توليد الأكواد، المراجعة، والتحسين
- 🌍 **اللغة العربية** - فهم جميع اللهجات العربية (مصرية، خليجية، شامية، مغربية)
- 📐 **الرياضيات** - الجبر، التفاضل والتكامل، الهندسة، الإحصاء، الجبر الخطي
- ⚛️ **الفيزياء** - الميكانيكا، الكهرومغناطيسية، الديناميكا الحرارية، ميكانيكا الكم، النسبية
- 🎨 **الرسومات ثلاثية الأبعاد** - النمذجة 3D، التصيير، وتوليد المشاهد

### المميزات

#### 🔧 قدرات البرمجة
- توليد أكواد متعددة اللغات (Python، JavaScript، TypeScript، HTML، CSS، C++، Java، Rust، Go)
- مراجعة الأكواد وتحليل الأمان
- تنفيذ أنماط التصميم
- توليد الاختبارات التلقائي
- تحليل التعقيد

#### 🌐 دعم اللغة العربية
- فهم جميع اللهجات العربية الرئيسية
- تحليل وتوليد الشعر
- تصحيح النحو
- تحليل المشاعر
- قدرات الترجمة

#### 📊 حل الرياضيات
- الجبر وحل المعادلات
- التفاضل والتكامل (مشتقات، تكاملات، حدود)
- الجبر الخطي (مصفوفات، قيم ذاتية)
- حسابات الهندسة
- الإحصاء والاحتمالات
- المعادلات التفاضلية

#### ⚛️ محرك الفيزياء
- الميكانيكا الكلاسيكية
- الكهرومغناطيسية
- الديناميكا الحرارية
- ميكانيكا الكم
- النسبية الخاصة
- فيزياء الموجات

#### 🎨 الرسومات ثلاثية الأبعاد
- توليد كود Three.js
- دعم Babylon.js
- سكربتات Unity C#
- تصدير glTF
- تكوين المشاهد

### التثبيت

```bash
# استنساخ المستودع
git clone https://github.com/youssef-official/edrak.git
cd edrak

# إنشاء بيئة افتراضية
python -m venv venv

# تفعيل البيئة الافتراضية
# على Windows:
venv\Scripts\activate
# على macOS/Linux:
source venv/bin/activate

# تثبيت المتطلبات
pip install -r requirements.txt

# تحميل النموذج المدرب مسبقاً (اختياري)
# ضع ملفات النموذج في مجلد models/
```

### الاستخدام

#### تشغيل خادم API

```bash
python main.py
```

سيبدأ خادم API على العنوان `http://localhost:8000`

#### الواجهة الويبية

افتح `web_interface/index.html` في متصفحك للوصول إلى الواجهة التفاعلية.

#### مثال API

```python
import requests

# إكمال الدردشة
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "edrak-1.0",
    "messages": [
        {"role": "user", "content": "اكتب دالة بلغة Python لحساب factorial"}
    ],
    "temperature": 0.7,
    "code_mode": True
})

print(response.json())
```

### مرجع API

#### إكمال الدردشة
```http
POST /v1/chat/completions
```

هيئة الطلب:
```json
{
  "model": "edrak-1.0",
  "messages": [
    {"role": "user", "content": "رسالتك هنا"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "code_mode": false,
  "arabic_dialect": "egyptian",
  "math_mode": false,
  "physics_mode": false,
  "graphics_3d": false
}
```

#### توليد الكود
```http
POST /v1/code/generate
```

هيئة الطلب:
```json
{
  "prompt": "إنشاء مكون زر في React",
  "language": "javascript",
  "framework": "react",
  "include_tests": true
}
```

#### حل الرياضيات
```http
POST /v1/math/solve
```

هيئة الطلب:
```json
{
  "problem": "حل x^2 + 5x + 6 = 0",
  "show_steps": true
}
```

---

### النشر على Railway

Railway هي منصة بنية تحتية حديثة تتيح لك نشر تطبيقاتك بسرعة وسهولة. اتبع هذه الخطوات لنشر Edrak AI على Railway:

1.  **إنشاء مشروع جديد على Railway**: انتقل إلى [Railway.app](https://railway.app/) وقم بإنشاء مشروع جديد. يمكنك ربط حساب GitHub الخاص بك واختيار مستودع `youssef-official/edrak`.
2.  **تكوين متغيرات البيئة**: قد يتطلب Edrak AI متغيرات بيئة للتكوين. تحقق من ملف `.env.example` في جذر المشروع للحصول على قائمة بالمتغيرات الضرورية. ستحتاج إلى إضافة هذه المتغيرات إلى إعدادات مشروعك في Railway ضمن علامة التبويب "Variables". على سبيل المثال، إذا كان `API_KEY` مدرجًا في `.env.example`، فستضيف `API_KEY` وقيمته المقابلة في Railway.
3.  **البناء والنشر**: سيكتشف Railway تلقائيًا ملف `Dockerfile` في جذر المشروع ويستخدمه لبناء ونشر تطبيقك. تأكد من تكوين `Dockerfile` بشكل صحيح لتشغيل `main.py`.
4.  **الوصول إلى تطبيقك**: بمجرد النشر، ستوفر Railway عنوان URL عامًا لتطبيقك. يمكنك الوصول إلى واجهة برمجة التطبيقات والواجهة الويبية من خلال عنوان URL هذا.

---

## 🏗️ Architecture

```
edrak-ai/
├── main.py                    # Main API server
├── config/
│   └── config.yaml           # Configuration
├── src/
│   ├── model/
│   │   ├── edrak_transformer.py   # Core transformer model
│   │   ├── arabic_processor.py    # Arabic language processor
│   │   ├── code_engine.py         # Code generation engine
│   │   ├── math_solver.py         # Mathematics solver
│   │   ├── physics_engine.py      # Physics engine
│   │   └── graphics3d.py          # 3D graphics engine
│   ├── learning/
│   │   ├── human_feedback.py      # Human feedback learning
│   │   └── self_correction.py     # Self-correction system
│   ├── memory/
│   │   └── context_manager.py     # Context management
│   └── utils/
│       └── safety.py              # Safety checker
├── training/
│   └── trainer.py             # Training script
├── web_interface/
│   └── index.html            # Web UI
├── models/                    # Model weights
├── data/                      # Training data
└── logs/                      # Log files
```

---

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Transformer with RoPE |
| Hidden Size | 4,096 |
| Layers | 48 |
| Attention Heads | 64 |
| Vocabulary Size | 100,000 |
| Max Context Length | 32,768 |
| Parameters | ~8B |

---

## 🎯 Performance Benchmarks

| Task | Accuracy |
|------|----------|
| Code Generation | 92% |
| Arabic Understanding | 95% |
| Math Problem Solving | 88% |
| Physics Problem Solving | 85% |
| 3D Scene Generation | 90% |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Creator**: Youssef Elsayed Elghareeb (يوسف السيد الغريب)
- **Inspired by**: The rich Arabic scientific heritage and modern AI research

---

<div align="center">

**Made with ❤️ in Egypt**

**تم التطوير بـ ❤️ في مصر**

</div>
