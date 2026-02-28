"""
Edrak AI Trainer - نظام التدريب

يدرب نموذج Edrak AI على:
- بيانات البرمجة
- النصوص العربية
- الرياضيات
- الفيزياء
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import yaml

from src.model.edrak_transformer import EdrakTransformer, EdrakConfig

# ═══════════════════════════════════════════════════════════════════════════════
# مجموعات البيانات
# ═══════════════════════════════════════════════════════════════════════════════

class CodeDataset(Dataset):
    """مجموعة بيانات الأكواد"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # تحميل البيانات
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # تجهيز الإدخال
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class ArabicDataset(Dataset):
    """مجموعة بيانات اللغة العربية"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class MathDataset(Dataset):
    """مجموعة بيانات الرياضيات"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = f"### Problem:\n{item['problem']}\n\n### Solution:\n{item['solution']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# مدرب Edrak AI
# ═══════════════════════════════════════════════════════════════════════════════

class EdrakTrainer:
    """مدرب نموذج Edrak AI"""
    
    def __init__(
        self,
        model: EdrakTransformer,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # نقل النموذج إلى الجهاز
        self.model.to(device)
        
        # إعداد المُحسن
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.1)
        )
        
        # إعداد Scaler للـ Mixed Precision
        self.scaler = GradScaler()
        
        # إحصائيات
        self.global_step = 0
        self.epoch = 0
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 500,
        output_dir: str = "models/edrak"
    ):
        """تدريب النموذج"""
        
        # إعداد DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # إعداد الجدولة
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║                    Edrak AI Training                         ║")
        print(f"╠══════════════════════════════════════════════════════════════╣")
        print(f"║  Epochs: {num_epochs:>15}                                  ║")
        print(f"║  Batch Size: {batch_size:>11}                                  ║")
        print(f"║  Total Steps: {total_steps:>10}                                  ║")
        print(f"║  Device: {self.device:>15}                                  ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")
        
        # حلقة التدريب
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # نقل البيانات إلى الجهاز
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Mixed Precision Training
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # حساب الخسارة
                    logits = outputs['logits']
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    
                    # تقسيم الخسارة على خطوات التراكم
                    loss = loss / gradient_accumulation_steps
                
                # Backward Pass مع Scaler
                self.scaler.scale(loss).backward()
                
                # تحديث الأوزان
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # حفظ النموذج
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(output_dir)
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            
            # التقييم
            if eval_dataset:
                eval_loss = self.evaluate(eval_dataset, batch_size)
                print(f"Evaluation Loss: {eval_loss:.4f}")
        
        # حفظ النموذج النهائي
        self.save_checkpoint(output_dir, final=True)
        print("\n✅ Training completed!")
    
    def evaluate(self, eval_dataset: Dataset, batch_size: int = 8) -> float:
        """تقييم النموذج"""
        
        self.model.eval()
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs['logits']
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                total_loss += loss.item()
        
        return total_loss / len(eval_loader)
    
    def save_checkpoint(self, output_dir: str, final: bool = False):
        """حفظ checkpoint"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_name = "edrak_final.pt" if final else f"edrak_step_{self.global_step}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# دوال مساعدة
# ═══════════════════════════════════════════════════════════════════════════════

def create_sample_data():
    """إنشاء بيانات عينة للتدريب"""
    
    # بيانات البرمجة
    code_data = [
        {
            "instruction": "اكتب دالة بلغة Python لحساب factorial",
            "input": "",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)"
        },
        {
            "instruction": "Create a React component for a button",
            "input": "",
            "output": "import React from 'react';\n\nconst Button = ({ onClick, children }) => (\n    <button onClick={onClick}>{children}</button>\n);\n\nexport default Button;"
        },
    ]
    
    with open('data/code_data.json', 'w', encoding='utf-8') as f:
        json.dump(code_data, f, ensure_ascii=False, indent=2)
    
    # بيانات العربية
    arabic_data = [
        {"text": "مرحباً، كيف يمكنني مساعدتك اليوم؟"},
        {"text": "السلام عليكم ورحمة الله وبركاته"},
    ]
    
    with open('data/arabic_data.json', 'w', encoding='utf-8') as f:
        json.dump(arabic_data, f, ensure_ascii=False, indent=2)
    
    # بيانات الرياضيات
    math_data = [
        {
            "problem": "احتسب 2 + 2",
            "solution": "2 + 2 = 4"
        },
        {
            "problem": "ما مشتقة x²؟",
            "solution": "d/dx(x²) = 2x"
        },
    ]
    
    with open('data/math_data.json', 'w', encoding='utf-8') as f:
        json.dump(math_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Sample data created!")


if __name__ == "__main__":
    # إنشاء بيانات عينة
    create_sample_data()
