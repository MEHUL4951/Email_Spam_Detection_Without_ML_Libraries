import random
from collections import defaultdict

def train_test_split(X, y, test_size=0.2, stratify=None):
    if stratify is None:
        # Regular shuffle split
        combined = list(zip(X, y))
        random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)
        
        split_idx = int(len(X_shuffled) * (1 - test_size))
        return (list(X_shuffled[:split_idx]), list(X_shuffled[split_idx:]),
                list(y_shuffled[:split_idx]), list(y_shuffled[split_idx:]))
    
    # Stratified split
    class_indices = defaultdict(list)
    for idx, label in enumerate(stratify):
        class_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    for label, indices in class_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    
    # Shuffle final splits to avoid any order bias
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
