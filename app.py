from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# ── GhostAttentionCNN ─────────────────────────────────────────────────────────

class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2, kernel=1, stride=1):
        super().__init__()
        init_ch = out_ch // ratio
        cheap_ch = out_ch - init_ch
        self.primary   = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel, stride, kernel//2, bias=False),
            nn.BatchNorm2d(init_ch), nn.ReLU(inplace=True))
        self.cheap     = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, 3, 1, 1, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        p = self.primary(x)
        return torch.cat([p, self.cheap(p)], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch))

    def forward(self, x):
        a = self.fc(self.avg(x).flatten(1))
        m = self.fc(self.max(x).flatten(1))
        return x * torch.sigmoid(a + m).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], 1)
        return x * torch.sigmoid(self.conv(a))


class GhostAttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.ghost   = GhostModule(in_ch, out_ch, stride=stride)
        self.ca      = ChannelAttention(out_ch)
        self.sa      = SpatialAttention()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        return F.relu(self.sa(self.ca(self.ghost(x))) + self.shortcut(x))


class GhostAttentionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            GhostAttentionBlock(32,  64,  stride=2),
            GhostAttentionBlock(64,  128, stride=2),
            GhostAttentionBlock(128, 256, stride=2),
            GhostAttentionBlock(256, 512, stride=2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.drop(self.pool(x).flatten(1))
        return self.fc(x)


# ── Load model ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Cancer", "Normal"]   # adjust order to match your training labels

model = GhostAttentionCNN(num_classes=len(CLASS_NAMES))
model_path = "ghost_attention_bone_cancer.pt"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path} on {device.upper()}")
else:
    print(f"WARNING: '{model_path}' not found — model running with random weights.")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = torch.argmax(probs).item()

        label      = CLASS_NAMES[pred]
        confidence = probs[pred].item() * 100
        all_probs  = {name: round(probs[i].item() * 100, 2)
                      for i, name in enumerate(CLASS_NAMES)}

        return jsonify({
            'success':    True,
            'prediction': label,
            'confidence': round(confidence, 1),
            'all_probs':  all_probs
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device': device,
        'classes': CLASS_NAMES,
        'model_loaded': os.path.exists(model_path)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
