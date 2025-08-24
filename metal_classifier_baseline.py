# metal_classifier_baseline.py

import torch.nn as nn

class MetalClassifier(nn.Module):
    def __init__(self, dropout=0.1, negative_slope=0.2):
        super().__init__()
        
        # Simple CNN backbone
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            # Second conv block
            # nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    m = MetalClassifier()
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("This model has ",total," parameters")