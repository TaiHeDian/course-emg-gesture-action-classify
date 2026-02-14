# trainer.py
import torch


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, torch.argmax(y, dim=1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(y, dim=1)
        correct += (preds == labels).sum().item()
        total += X.size(0)
        
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, torch.argmax(y, dim=1))
            
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            
            correct += (preds == labels).sum().item()
            total += X.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return total_loss / total, correct / total, all_preds, all_labels
