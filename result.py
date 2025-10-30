import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import time

# ==================== 1️⃣ SETUP DATA TEST ====================
data_dir = "all_data"  # folder kamu
img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes
print("Kelas:", class_names)

# Split: 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== 2️⃣ LOAD MODEL ====================
model = models.squeezenet1_1(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Conv2d(512, len(class_names), kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)

model.load_state_dict(torch.load("squeezenet_trained.pth", map_location="cpu"))
model.eval()
print("✅ Model berhasil dimuat!")

# ==================== 3️⃣ EVALUASI MODEL ====================
all_preds = []
all_labels = []
start_time = time.time()

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

test_time = time.time() - start_time

# ==================== 4️⃣ CONFUSION MATRIX ====================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ==================== 5️⃣ CLASSIFICATION REPORT ====================
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\n=== Classification Report ===\n")
print(report)

# ==================== 6️⃣ VISUALISASI TRAINING HISTORY  ====================
# masih dummy

epochs = np.arange(1, 31)
train_acc = np.linspace(0.2, 0.98, 30)
val_acc = np.random.uniform(0.1, 0.3, 30)
train_loss = np.exp(-epochs/5)
val_loss = np.random.uniform(4, 7, 30)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Train")
plt.plot(epochs, val_acc, label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Train")
plt.plot(epochs, val_loss, label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_graph.png")
plt.show()

print(f"\n🕒 Waktu testing: {test_time:.2f} detik")
