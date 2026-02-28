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
  - [Deployment on Railway (No Docker)](#deployment-on-railway-no-docker)
- [العربية](#العربية)
  - [نظرة عامة](#نظرة-عامة)
  - [المميزات](#المميزات)
  - [التثبيت](#التثبيت)
  - [الاستخدام](#الاستخدام)
  - [النشر على Railway (بدون Docker)](#النشر-على-railway-بدون-docker)

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
```

### Usage

#### Running the API Server

```bash
python main.py
```

The API server will start at `http://localhost:8000`

#### Web Interface

Open `web_interface/index.html` in your browser to access the interactive web interface.

---

### Deployment on Railway (No Docker)

To avoid image size limits, we deploy directly using Python. Railway will automatically detect the Python environment.

1.  **Create a new project on Railway**: Connect your GitHub and select `youssef-official/edrak`.
2.  **Configure Environment Variables**:
    - Add `PORT` variable (set it to `8000` or leave it, Railway provides it automatically).
    - Add any other variables from `.env.example`.
3.  **Set Start Command**:
    In Railway service settings, set the **Start Command** to:
    ```bash
    python main.py
    ```
4.  **Nixpacks**: Ensure Railway is using **Nixpacks** (default) to build the project. It will detect `requirements.txt` and install all dependencies.

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
```

### الاستخدام

#### تشغيل خادم API

```bash
python main.py
```

سيبدأ خادم API على العنوان `http://localhost:8000`

---

### النشر على Railway (بدون Docker)

لتجنب مشكلة حجم الصورة الكبيرة، سنقوم بالنشر مباشرة باستخدام Python. سيقوم Railway باكتشاف بيئة Python تلقائياً.

1.  **إنشاء مشروع جديد على Railway**: اربط حساب GitHub الخاص بك واختر مستودع `youssef-official/edrak`.
2.  **تكوين متغيرات البيئة**:
    - أضف متغير `PORT` (اجعله `8000` أو اتركه، Railway يوفره تلقائياً).
    - أضف أي متغيرات أخرى من ملف `.env.example`.
3.  **إعداد أمر التشغيل (Start Command)**:
    في إعدادات الخدمة في Railway، قم بتعيين **Start Command** إلى:
    ```bash
    python main.py
    ```
4.  **Nixpacks**: تأكد من أن Railway يستخدم **Nixpacks** (الافتراضي) لبناء المشروع. سيكتشف ملف `requirements.txt` ويقوم بتثبيت كافة المكتبات المطلوبة.

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

<div align="center">

**Made with ❤️ in Egypt**

**تم التطوير بـ ❤️ في مصر**

</div>
