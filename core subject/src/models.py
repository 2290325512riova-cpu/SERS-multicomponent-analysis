"""Model definitions for the SERS project."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    AUG_ENABLED,
    AUG_N,
    DL_BATCH,
    DL_EPOCHS,
    DL_LR,
    DL_PATIENCE,
    DL_SCHEDULER,
    DL_VAL_FRAC,
    DL_WEIGHT_DECAY,
    FOCAL_LOSS_GAMMA,
    KAN_GRID_SIZE,
    KAN_SPLINE_ORDER,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    MIXUP_ENABLED,
    SEED,
    TASKS,
    TASKS_MT_FULL,
    USE_CLASS_WEIGHT,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        target_matrix = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            target_matrix = target_matrix * (1.0 - self.label_smoothing) + smooth

        probs = torch.exp(log_probs)
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -target_matrix * focal_weight * log_probs

        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)

        return loss.sum(dim=-1).mean()


class OrdinalCoralLoss(nn.Module):
    """Focal loss + ordinal regularization: penalizes predictions far from true class."""
    def __init__(self, num_classes, gamma=2.0, weight=None, label_smoothing=0.0, ordinal_lambda=0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)
        self.num_classes = num_classes
        self.ordinal_lambda = ordinal_lambda

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        probs = F.softmax(logits, dim=-1)
        class_indices = torch.arange(self.num_classes, device=logits.device).float()
        expected_class = (probs * class_indices.unsqueeze(0)).sum(dim=-1)
        true_class = targets.float()
        ordinal_penalty = ((expected_class - true_class) ** 2).mean()
        return focal_loss + self.ordinal_lambda * ordinal_penalty


class CostSensitiveOrdinalLoss(nn.Module):
    """Cross-entropy weighted by ordinal distance: adjacent errors cost less than skip errors."""
    def __init__(self, num_classes, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        class_indices = torch.arange(self.num_classes, device=logits.device)
        dist = torch.abs(class_indices.unsqueeze(0) - targets.unsqueeze(1).float())
        cost_weights = 1.0 + 0.5 * dist  # normalized: correct=1, adjacent=1.5, skip=2
        loss = -log_probs * cost_weights
        if self.weight is not None:
            sample_weights = self.weight.to(logits.device)[targets]
            loss = loss * sample_weights.unsqueeze(1)
        return loss.sum(dim=-1).mean() / self.num_classes


def _compute_class_weights(y_encoded, method='simple'):
    classes, counts = np.unique(y_encoded, return_counts=True)
    weights = np.ones(int(classes.max()) + 1, dtype=np.float32)
    if method == 'effective_number':
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights[classes] = (1.0 - beta) / effective_num
    else:
        weights[classes] = len(y_encoded) / (len(classes) * counts)
    weights = weights / weights.max()
    return torch.tensor(weights, dtype=torch.float32)


def _default_train_cfg():
    return {
        'lr': DL_LR,
        'epochs': DL_EPOCHS,
        'patience': DL_PATIENCE,
        'batch_size': DL_BATCH,
        'weight_decay': DL_WEIGHT_DECAY,
        'val_frac': DL_VAL_FRAC,
        'scheduler': DL_SCHEDULER,
        'optimizer': 'adam',
        'step_size': max(DL_EPOCHS // 5, 1),
        'step_gamma': 0.7,
        'aug_enabled': AUG_ENABLED,
        'aug_n': AUG_N,
        'mixup_enabled': MIXUP_ENABLED,
        'mixup_alpha': MIXUP_ALPHA,
        'label_smoothing': LABEL_SMOOTHING,
        'focal_gamma': FOCAL_LOSS_GAMMA,
        'use_class_weight': USE_CLASS_WEIGHT,
    }


def _resolve_train_cfg(overrides=None):
    cfg = _default_train_cfg()
    if overrides:
        cfg.update(overrides)
    return cfg


def _build_optimizer(model, cfg):
    optimizer_name = str(cfg['optimizer']).lower()
    lr = float(cfg['lr'])
    weight_decay = float(cfg['weight_decay'])

    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, cfg):
    scheduler_name = str(cfg['scheduler']).lower()
    epochs = int(cfg['epochs'])

    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(cfg['step_size']), 1),
            gamma=float(cfg['step_gamma']),
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,
        factor=0.5,
    )


def _make_loss(y_encoded, num_classes, gamma=FOCAL_LOSS_GAMMA,
               label_smoothing=LABEL_SMOOTHING, use_class_weight=USE_CLASS_WEIGHT,
               loss_type='focal', class_balance='simple'):
    weight = None
    if use_class_weight:
        weight = _compute_class_weights(y_encoded, method=class_balance)
        if len(weight) < num_classes:
            padded = torch.ones(num_classes, dtype=torch.float32)
            padded[: len(weight)] = weight
            weight = padded

    if loss_type == 'ordinal_coral':
        return OrdinalCoralLoss(num_classes=num_classes, gamma=gamma, weight=weight,
                                label_smoothing=label_smoothing)
    elif loss_type == 'cost_sensitive_ordinal':
        return CostSensitiveOrdinalLoss(num_classes=num_classes, weight=weight)
    return FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)



def _balanced_sample_weight(y, class_weight):
    if class_weight is None:
        return None

    if class_weight == 'balanced':
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts)
        weight_map = dict(zip(classes, weights))
        return np.asarray([weight_map[label] for label in y], dtype=np.float64)

    if isinstance(class_weight, dict):
        return np.asarray([class_weight.get(label, 1.0) for label in y], dtype=np.float64)

    raise ValueError(f'Unsupported class_weight for HistGradientBoosting: {class_weight!r}')


def get_rf(**overrides):
    params = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': SEED,
        'n_jobs': -1,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def get_extra_trees(**overrides):
    params = {
        'n_estimators': 400,
        'max_depth': None,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': SEED,
        'n_jobs': -1,
    }
    params.update(overrides)
    return ExtraTreesClassifier(**params)


def get_svm(**overrides):
    params = {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'random_state': SEED,
    }
    params.update(overrides)
    return SVC(**params)


class _ScaledLDAClassifier:
    def __init__(self, solver='svd', shrinkage=None, standardize=True):
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X, y):
        X_fit = self.scaler.fit_transform(X) if self.standardize else X
        self.model.fit(X_fit, y)
        return self

    def predict(self, X):
        X_pred = self.scaler.transform(X) if self.standardize else X
        return self.model.predict(X_pred)


def get_lda(**overrides):
    params = {
        'solver': 'svd',
        'shrinkage': None,
        'standardize': True,
    }
    params.update(overrides)
    return _ScaledLDAClassifier(**params)


class _XGBoostWrapper:
    def __init__(self, **overrides):
        self.overrides = dict(overrides)
        self.model = None
        self.classes_ = None
        self.label_map = None
        self.inv_map = None

    def _build_model(self, n_classes: int):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is requested but not installed. Install requirements.txt or remove XGBoost from the model list."
            ) from exc

        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic' if n_classes == 2 else 'multi:softprob',
            'eval_metric': 'logloss' if n_classes == 2 else 'mlogloss',
            'tree_method': 'hist',
            'random_state': SEED,
            'n_jobs': -1,
        }
        if n_classes > 2:
            params['num_class'] = n_classes
        params.update(self.overrides)
        return XGBClassifier(**params)

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        self.label_map = {label: idx for idx, label in enumerate(self.classes_)}
        self.inv_map = {idx: label for label, idx in self.label_map.items()}
        y_encoded = np.array([self.label_map[value] for value in y], dtype=int)
        self.model = self._build_model(len(self.classes_))
        sample_weight = _balanced_sample_weight(y_encoded, 'balanced')
        self.model.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X):
        pred = self.model.predict(X).astype(int)
        return np.array([self.inv_map[int(value)] for value in pred])


def get_xgboost(**overrides):
    return _XGBoostWrapper(**overrides)


class PLSDA:
    def __init__(self, n_components=10):
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        Y = pd.get_dummies(y).values.astype(float)
        self.pls.fit(X, Y)
        return self

    def predict(self, X):
        probs = self.pls.predict(X)
        return self.classes_[np.argmax(probs, axis=1)]


def get_plsda(**overrides):
    params = {'n_components': 10}
    params.update(overrides)
    return PLSDA(**params)


class _ScaledKNNClassifier:
    def __init__(self, n_neighbors=7, weights='distance', metric='minkowski',
                 p=2, algorithm='auto', leaf_size=30, n_jobs=-1,
                 standardize=True):
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        X_fit = self.scaler.fit_transform(X) if self.standardize else X
        self.model.fit(X_fit, y)
        return self

    def predict(self, X):
        X_pred = self.scaler.transform(X) if self.standardize else X
        return self.model.predict(X_pred)


def get_knn(**overrides):
    params = {
        'n_neighbors': 7,
        'weights': 'distance',
        'metric': 'minkowski',
        'p': 2,
        'algorithm': 'auto',
        'leaf_size': 30,
        'n_jobs': -1,
        'standardize': True,
    }
    params.update(overrides)
    return _ScaledKNNClassifier(**params)


class _BalancedHistGradientBoostingClassifier:
    def __init__(self, learning_rate=0.05, max_iter=300, max_depth=6,
                 max_leaf_nodes=31, min_samples_leaf=3, l2_regularization=1e-3,
                 validation_fraction=0.2, n_iter_no_change=20,
                 early_stopping=True, class_weight='balanced', random_state=SEED):
        self.class_weight = class_weight
        self.model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    def fit(self, X, y):
        sample_weight = _balanced_sample_weight(y, self.class_weight)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


def get_hist_gradient_boosting(**overrides):
    params = {
        'learning_rate': 0.05,
        'max_iter': 300,
        'max_depth': 6,
        'max_leaf_nodes': 31,
        'min_samples_leaf': 3,
        'l2_regularization': 1e-3,
        'validation_fraction': 0.2,
        'n_iter_no_change': 20,
        'early_stopping': True,
        'class_weight': 'balanced',
        'random_state': SEED,
    }
    params.update(overrides)
    return _BalancedHistGradientBoostingClassifier(**params)


class _MLFeatureWrapper:
    """Wrap an sklearn classifier to use hand-crafted features (78-dim)."""
    is_feature_model = True

    def __init__(self, base_cls_fn):
        self.base_cls_fn = base_cls_fn
        self._clf = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        feats = self._extract(X)
        self._clf = self.base_cls_fn()
        self._clf.fit(feats, y)
        return self

    def predict(self, X):
        feats = self._extract(X)
        return self._clf.predict(feats)


def get_rf_feat():
    return _MLFeatureWrapper(get_rf)


def get_svm_feat():
    return _MLFeatureWrapper(get_svm)


def get_plsda_feat():
    return _MLFeatureWrapper(get_plsda)


class _SE1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = x.mean(dim=-1)
        weights = self.fc(weights).unsqueeze(-1)
        return x * weights


class _ConvBranch1D(nn.Module):
    def __init__(self, kernel_size, out_channels=16):
        super().__init__()
        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv1d(1, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
            nn.MaxPool1d(2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.branch(x).squeeze(-1)


class _CNN1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.se = _SE1D(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.se(self.block(x)) + x)


class _ResNet1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(_ResBlock1D(16), _ResBlock1D(16))
        self.down1 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=2),
            nn.BatchNorm1d(32),
        )
        self.stage2 = nn.Sequential(_ResBlock1D(32), _ResBlock1D(32))
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=2),
            nn.BatchNorm1d(64),
        )
        self.stage3 = nn.Sequential(_ResBlock1D(64), _ResBlock1D(64))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class _RamanNetLite(nn.Module):
    """Location-aware window MLP for fixed-wavenumber Raman peaks."""

    def __init__(self, input_dim, num_classes, window_size=64, stride=32, embed_dim=16):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        if input_dim <= window_size:
            self.n_windows = 1
        else:
            self.n_windows = ((input_dim - window_size + stride - 1) // stride) + 1
        self.pad_right = max((self.n_windows - 1) * stride + window_size - input_dim, 0)
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.window_weight = nn.Parameter(torch.randn(self.n_windows, window_size, embed_dim) * 0.02)
        self.window_bias = nn.Parameter(torch.zeros(self.n_windows, embed_dim))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.n_windows * embed_dim),
            nn.Linear(self.n_windows * embed_dim, 96),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.input_bn(x)
        if self.pad_right:
            x = F.pad(x, (0, self.pad_right))
        windows = x.unfold(dimension=1, size=self.window_size, step=self.stride)
        z = torch.einsum('bnw,nwe->bne', windows, self.window_weight) + self.window_bias
        z = self.dropout(F.gelu(z)).flatten(1)
        return self.classifier(z)


class _PatchTransformer1DNet(nn.Module):
    """Small patch transformer for interactions among spectral regions."""

    def __init__(self, input_dim, num_classes, patch_size=32, stride=32, d_model=48):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        if input_dim <= patch_size:
            self.n_patches = 1
        else:
            self.n_patches = ((input_dim - patch_size + stride - 1) // stride) + 1
        self.pad_right = max((self.n_patches - 1) * stride + patch_size - input_dim, 0)
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.25),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.pad_right:
            x = F.pad(x, (0, self.pad_right))
        z = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed[:, :z.size(1) + 1]
        z = self.encoder(z)
        return self.classifier(z[:, 0])


class _DLWrapper:
    def __init__(self, net_cls, lr=DL_LR, epochs=DL_EPOCHS, patience=DL_PATIENCE,
                 batch_size=DL_BATCH, train_cfg=None, seed=SEED):
        self.net_cls = net_cls
        overrides = {
            'lr': lr,
            'epochs': epochs,
            'patience': patience,
            'batch_size': batch_size,
        }
        if train_cfg is not None:
            overrides.update(train_cfg)
        self.train_cfg = _resolve_train_cfg(overrides)
        self.lr = self.train_cfg['lr']
        self.epochs = self.train_cfg['epochs']
        self.patience = self.train_cfg['patience']
        self.batch_size = self.train_cfg['batch_size']
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = None
        self.inv_map = None
        self.classes_ = None

    def fit(self, X, y, groups=None):
        cfg = self.train_cfg
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True

        self.classes_ = np.array(sorted(set(y)))
        self.label_map = {label: idx for idx, label in enumerate(self.classes_)}
        self.inv_map = {idx: label for label, idx in self.label_map.items()}
        yi = np.array([self.label_map[value] for value in y])
        num_classes = len(self.label_map)

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(yi, dtype=torch.long)

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=cfg['val_frac'], random_state=self.seed)
            t_idx, v_idx = next(gss.split(X, yi, groups))
        else:
            idx = np.random.permutation(len(Xt))
            n_val = max(int(len(Xt) * cfg['val_frac']), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if cfg['aug_enabled']:
            from src.dataset import augment_spectra, spectral_mixup

            X_tr_np, y_tr_np = X[t_idx], yi[t_idx]
            composition_keys = np.asarray(groups)[t_idx] if groups is not None else None
            X_tr_np, y_tr_np = augment_spectra(X_tr_np, y_tr_np, n_aug=int(cfg['aug_n']))
            if composition_keys is not None:
                composition_keys = np.tile(composition_keys, int(cfg['aug_n']) + 1)
            if cfg['mixup_enabled'] and composition_keys is not None:
                X_mix, y_mix = spectral_mixup(
                    X_tr_np,
                    y_tr_np,
                    n_mix=len(X[t_idx]),
                    alpha=float(cfg['mixup_alpha']),
                    composition_keys=composition_keys,
                )
                X_tr_np = np.concatenate([X_tr_np, X_mix], axis=0)
                y_tr_np = np.concatenate([y_tr_np, y_mix], axis=0)
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            yt_train = torch.tensor(y_tr_np, dtype=torch.long)
        else:
            Xt_train, yt_train = Xt[t_idx], yt[t_idx]

        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            TensorDataset(Xt_train, yt_train),
            batch_size=int(cfg['batch_size']),
            shuffle=True,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            TensorDataset(Xt[v_idx], yt[v_idx]),
            batch_size=256,
            shuffle=False,
            pin_memory=pin_memory,
        )

        self.model = self.net_cls(X.shape[1], num_classes).to(self.device)
        optimizer = _build_optimizer(
            self.model,
            cfg,
        )
        scheduler = _build_scheduler(optimizer, cfg)
        scheduler_name = str(cfg['scheduler']).lower()

        train_loss = _make_loss(
            yt_train.cpu().numpy(),
            num_classes,
            gamma=float(cfg['focal_gamma']),
            label_smoothing=float(cfg['label_smoothing']),
            use_class_weight=bool(cfg['use_class_weight']),
            loss_type=str(cfg.get('loss_type', 'focal')),
            class_balance=str(cfg.get('class_balance', 'simple')),
        )
        val_loss = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(int(cfg['epochs'])):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=pin_memory)
                yb = yb.to(self.device, non_blocking=pin_memory)
                optimizer.zero_grad()
                loss = train_loss(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device, non_blocking=pin_memory)
                    yb = yb.to(self.device, non_blocking=pin_memory)
                    epoch_val_loss += val_loss(self.model(xb), yb).item() * len(yb)
                    val_count += len(yb)

            epoch_val_loss /= max(val_count, 1)
            if scheduler_name in {'cosine', 'step'}:
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= int(cfg['patience']):
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[pred] for pred in preds])

    def predict_proba(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(Xt)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba


def get_cnn1d(train_cfg=None):
    return _DLWrapper(_CNN1DNet, train_cfg=train_cfg)


def get_resnet1d(train_cfg=None):
    return _DLWrapper(_ResNet1DNet, train_cfg=train_cfg)


def get_ramannet_lite(train_cfg=None):
    return _DLWrapper(_RamanNetLite, train_cfg=train_cfg)


def get_patch_transformer(train_cfg=None):
    return _DLWrapper(_PatchTransformer1DNet, train_cfg=train_cfg)


class _DLFeatureWrapper:
    """Run any DL model on handcrafted features instead of raw spectra."""
    is_feature_model = True

    def __init__(self, net_cls, train_cfg=None, **kwargs):
        if train_cfg is not None:
            kwargs = dict(kwargs)
            kwargs['train_cfg'] = train_cfg
        self._inner = _DLWrapper(net_cls, **kwargs)
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        feats = self._extract(X)
        return self._inner.fit(feats, y, groups=groups)

    def predict(self, X):
        feats = self._extract(X)
        return self._inner.predict(feats)


def get_cnn1d_feat(train_cfg=None):
    return _DLFeatureWrapper(_CNN1DNet, train_cfg=train_cfg)


def get_resnet1d_feat(train_cfg=None):
    return _DLFeatureWrapper(_ResNet1DNet, train_cfg=train_cfg)


class _KANCNN1DNet(nn.Module):
    """CNN backbone + KAN head (replaces FC classifier with KAN layers)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.kan_head = nn.Sequential(
            _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER),
            nn.Dropout(0.2),
            _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.kan_head(x)


def get_kan_cnn(train_cfg=None):
    return _DLWrapper(_KANCNN1DNet, train_cfg=train_cfg)


def get_kan_cnn_feat(train_cfg=None):
    return _DLFeatureWrapper(_KANCNN1DNet, train_cfg=train_cfg)


class _KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = 2.0 / grid_size
        grid = torch.linspace(
            -1 - step * spline_order,
            1 + step * spline_order,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid)

        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_bases) * 0.1
        )
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(
                -1.0 / in_features ** 0.5,
                1.0 / in_features ** 0.5,
            )
        )

    def forward(self, x):
        bases = self._compute_bases(x)
        spline_out = torch.einsum('bin,oin->bo', bases, self.spline_weight)
        base_out = F.silu(x) @ self.base_weight.t()
        return spline_out + base_out

    def _compute_bases(self, x):
        x = torch.tanh(x).unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1]
            t_left = self.grid[:n - 1]
            t_left_k = self.grid[k:k + n - 1]
            left = (x - t_left) / (t_left_k - t_left + 1e-8)
            t_right_k1 = self.grid[k + 1:k + 1 + n - 1]
            t_right = self.grid[1:1 + n - 1]
            right = (t_right_k1 - x) / (t_right_k1 - t_right + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases


class _MultiTaskCNN1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: nn.Linear(64, n_classes) for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskResNet1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.stage1 = nn.Sequential(_ResBlock1D(16), _ResBlock1D(16))
        self.down1 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=1, stride=2), nn.BatchNorm1d(32))
        self.stage2 = nn.Sequential(_ResBlock1D(32), _ResBlock1D(32))
        self.down2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, stride=2), nn.BatchNorm1d(64))
        self.stage3 = nn.Sequential(_ResBlock1D(64), _ResBlock1D(64))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: nn.Linear(64, n_classes) for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        features = self.gap(x).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskKANCNN1DNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), _SE1D(16), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), _SE1D(32), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), _SE1D(64),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({
            task_id: _KANLinear(64, n_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
            for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).squeeze(-1)
        features = self.dropout(features)
        return {task_id: head(features) for task_id, head in self.heads.items()}


class _MultiTaskDLWrapper:
    is_multitask = True

    def __init__(self, net_cls, tasks, lr=DL_LR, epochs=DL_EPOCHS,
                 patience=DL_PATIENCE, batch_size=DL_BATCH, seed=SEED):
        self.net_cls = net_cls
        self.tasks = tasks
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_maps = {}
        self.inv_maps = {}

    def fit(self, X, y_dict, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = True

        task_num_classes = []
        encoded_y = {}
        for task in self.tasks:
            task_id = task['id']
            labels = sorted(set(y_dict[task_id]))
            self.label_maps[task_id] = {label: idx for idx, label in enumerate(labels)}
            self.inv_maps[task_id] = {idx: label for label, idx in self.label_maps[task_id].items()}
            task_num_classes.append((task_id, len(labels)))
            encoded_y[task_id] = torch.tensor(
                [self.label_maps[task_id][value] for value in y_dict[task_id]],
                dtype=torch.long,
            )

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(X, encoded_y[self.tasks[0]['id']].numpy(), groups))
        else:
            idx = np.random.permutation(len(Xt))
            n_val = max(int(len(Xt) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            from src.dataset import augment_spectra_mt, spectral_mixup_mt

            X_tr_np = X[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}
            composition_keys = np.asarray(groups)[t_idx] if groups is not None else None
            X_tr_np, y_dict_tr = augment_spectra_mt(X_tr_np, y_dict_tr, n_aug=AUG_N)
            if composition_keys is not None:
                composition_keys = np.tile(composition_keys, AUG_N + 1)
            if MIXUP_ENABLED and composition_keys is not None:
                X_mix, y_mix = spectral_mixup_mt(
                    X_tr_np,
                    y_dict_tr,
                    n_mix=len(X[t_idx]),
                    alpha=MIXUP_ALPHA,
                    composition_keys=composition_keys,
                )
                X_tr_np = np.concatenate([X_tr_np, X_mix], axis=0)
                y_dict_tr = {
                    task_id: np.concatenate([y_dict_tr[task_id], y_mix[task_id]])
                    for task_id in y_dict_tr
                }
            Xt_train = torch.tensor(X_tr_np, dtype=torch.float32).unsqueeze(1)
            train_tensors = [Xt_train] + [
                torch.tensor(y_dict_tr[task['id']], dtype=torch.long) for task in self.tasks
            ]
        else:
            train_tensors = [Xt[t_idx]] + [encoded_y[task['id']][t_idx] for task in self.tasks]

        val_tensors = [Xt[v_idx]] + [encoded_y[task['id']][v_idx] for task in self.tasks]
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=256, shuffle=False)

        self.model = self.net_cls(X.shape[1], task_num_classes).to(self.device)
        n_tasks = len(self.tasks)
        log_vars = nn.Parameter(torch.zeros(n_tasks, device=self.device))
        task_losses = [
            _make_loss(train_tensors[i + 1].cpu().numpy(), len(self.label_maps[self.tasks[i]['id']]))
            for i in range(n_tasks)
        ]
        val_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [log_vars],
            lr=self.lr,
            weight_decay=DL_WEIGHT_DECAY,
        )
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=10,
                factor=0.5,
            )

        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                xb = batch[0].to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = sum(
                    torch.exp(-log_vars[i]) * task_losses[i](logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                    + 0.5 * log_vars[i]
                    for i in range(n_tasks)
                )
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0].to(self.device)
                    logits = self.model(xb)
                    batch_loss = sum(
                        val_loss(logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                        for i in range(n_tasks)
                    )
                    epoch_val_loss += batch_loss.item() * len(xb)
                    val_count += len(xb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        preds = {}
        with torch.no_grad():
            logits = self.model(Xt)
            for task_id, task_logits in logits.items():
                idx = task_logits.argmax(1).cpu().numpy()
                preds[task_id] = np.array([self.inv_maps[task_id][value] for value in idx])
        return preds


def get_mt_cnn():
    return _MultiTaskDLWrapper(_MultiTaskCNN1DNet, TASKS)


def get_mt_resnet():
    return _MultiTaskDLWrapper(_MultiTaskResNet1DNet, TASKS)


def get_mt_kan_cnn():
    return _MultiTaskDLWrapper(_MultiTaskKANCNN1DNet, TASKS)


def get_mt_cnn_3c():
    return _MultiTaskDLWrapper(_MultiTaskCNN1DNet, TASKS_MT_FULL)


def get_mt_resnet_3c():
    return _MultiTaskDLWrapper(_MultiTaskResNet1DNet, TASKS_MT_FULL)


def get_mt_kan_cnn_3c():
    return _MultiTaskDLWrapper(_MultiTaskKANCNN1DNet, TASKS_MT_FULL)


class _SpectrumKANNet(nn.Module):
    """Pure KAN on full spectrum (no CNN backbone)."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.kan1 = _KANLinear(input_dim, 128, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = _KANLinear(128, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan3 = _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan4 = _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.bn(x)
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        x = self.kan3(x)
        x = self.dropout3(x)
        return self.kan4(x)


def get_spectrum_kan(train_cfg=None):
    return _DLWrapper(_SpectrumKANNet, train_cfg=train_cfg)


class _ChemKANNet(nn.Module):
    """Chemistry-aware KAN model for targeted method development.

    Design:
      1. Three parallel convolution branches capture narrow / medium / broad
         spectral structures via kernel sizes 7 / 15 / 31.
      2. A global KAN branch keeps direct full-spectrum non-linear modeling.
      3. Local multi-scale features and global KAN features are fused by a
         KAN head, keeping the proposed model KAN-centric rather than
         attention-centric.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.branch_fine = _ConvBranch1D(7, out_channels=16)
        self.branch_medium = _ConvBranch1D(15, out_channels=16)
        self.branch_coarse = _ConvBranch1D(31, out_channels=16)
        self.branch_gate = nn.Sequential(
            nn.Linear(48, 24),
            nn.PReLU(),
            nn.Linear(24, 3),
        )
        self.local_bn = nn.BatchNorm1d(48)
        self.local_drop = nn.Dropout(0.2)

        self.global_kan1 = _KANLinear(
            input_dim,
            64,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER,
        )
        self.global_drop = nn.Dropout(0.2)
        self.global_kan2 = _KANLinear(
            64,
            32,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(80, 24),
            nn.PReLU(),
            nn.Linear(24, 2),
        )
        self.fusion_bn = nn.BatchNorm1d(80)
        self.fusion_kan1 = _KANLinear(
            80,
            64,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER,
        )
        self.fusion_drop1 = nn.Dropout(0.3)
        self.fusion_kan2 = _KANLinear(
            64,
            32,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER,
        )
        self.fusion_drop2 = nn.Dropout(0.2)
        self.classifier = _KANLinear(
            32,
            num_classes,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER,
        )

    def forward(self, x):
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
            x_vec = x
        else:
            x_seq = x
            x_vec = x.squeeze(1)

        x_vec = self.input_bn(x_vec)

        fine = self.branch_fine(x_seq)
        medium = self.branch_medium(x_seq)
        coarse = self.branch_coarse(x_seq)
        local = torch.cat([fine, medium, coarse], dim=1)
        branch_weights = torch.softmax(self.branch_gate(local), dim=1).unsqueeze(-1)
        local_stack = torch.stack([fine, medium, coarse], dim=1)
        local = (local_stack * branch_weights).reshape(local_stack.size(0), -1)
        local = self.local_bn(local)
        local = self.local_drop(local)

        global_feat = self.global_kan1(x_vec)
        global_feat = self.global_drop(global_feat)
        global_feat = self.global_kan2(global_feat)

        fusion_weights = torch.softmax(self.fusion_gate(torch.cat([local, global_feat], dim=1)), dim=1)
        local = local * fusion_weights[:, :1]
        global_feat = global_feat * fusion_weights[:, 1:2]
        fused = torch.cat([local, global_feat], dim=1)
        fused = self.fusion_bn(fused)
        fused = self.fusion_kan1(fused)
        fused = self.fusion_drop1(fused)
        fused = self.fusion_kan2(fused)
        fused = self.fusion_drop2(fused)
        return self.classifier(fused)


def get_chem_kan(train_cfg=None):
    return _DLWrapper(_ChemKANNet, train_cfg=train_cfg)


class _FeatureKANNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.kan1 = _KANLinear(input_dim, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan2 = _KANLinear(64, 32, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.kan3 = _KANLinear(32, num_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn(x)
        x = self.kan1(x)
        x = self.dropout1(x)
        x = self.kan2(x)
        x = self.dropout2(x)
        x = self.kan3(x)
        return x


class _FeatureKANWrapper:
    is_feature_model = True

    def __init__(self, lr=DL_LR, epochs=DL_EPOCHS, patience=DL_PATIENCE,
                 batch_size=DL_BATCH, seed=SEED):
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_map = None
        self.inv_map = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features

        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if wn is not None:
            self._wn = wn
        feats = self._extract(X)

        self.label_map = {label: idx for idx, label in enumerate(sorted(set(y)))}
        self.inv_map = {idx: label for label, idx in self.label_map.items()}
        yi = np.array([self.label_map[value] for value in y])
        num_classes = len(self.label_map)

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(feats, yi, groups))
        else:
            idx = np.random.permutation(len(feats))
            n_val = max(int(len(feats) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            rng = np.random.RandomState(self.seed)
            F_tr, y_tr = feats[t_idx], yi[t_idx]
            aug_feats = [F_tr]
            aug_labels = [y_tr]
            for _ in range(AUG_N):
                noisy = F_tr.copy()
                noisy += rng.normal(0, 0.02, noisy.shape) * np.std(noisy, axis=0, keepdims=True)
                noisy *= rng.uniform(0.95, 1.05, (len(noisy), 1))
                aug_feats.append(noisy)
                aug_labels.append(y_tr.copy())
            F_tr = np.concatenate(aug_feats, axis=0)
            y_tr = np.concatenate(aug_labels, axis=0)
        else:
            F_tr, y_tr = feats[t_idx], yi[t_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(F_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(feats[v_idx], dtype=torch.float32), torch.tensor(yi[v_idx], dtype=torch.long)),
            batch_size=256,
            shuffle=False,
        )

        self.model = _FeatureKANNet(feats.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=DL_WEIGHT_DECAY)
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_loss = _make_loss(y_tr, num_classes)
        val_loss = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = train_loss(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    epoch_val_loss += val_loss(self.model(xb), yb).item() * len(yb)
                    val_count += len(yb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        feats = self._extract(X)
        Xt = torch.tensor(feats, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).argmax(1).cpu().numpy()
        return np.array([self.inv_map[pred] for pred in preds])


def get_feature_kan():
    return _FeatureKANWrapper()


class _MTFeatureKANNet(nn.Module):
    def __init__(self, input_dim, task_num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.shared1 = _KANLinear(input_dim, 64, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.shared2 = _KANLinear(64, 48, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({
            task_id: _KANLinear(48, n_classes, grid_size=KAN_GRID_SIZE, spline_order=KAN_SPLINE_ORDER)
            for task_id, n_classes in task_num_classes
        })

    def forward(self, x):
        x = self.bn(x)
        x = self.shared1(x)
        x = self.dropout1(x)
        x = self.shared2(x)
        x = self.dropout2(x)
        return {task_id: head(x) for task_id, head in self.heads.items()}


class _MTFeatureKANWrapper:
    is_multitask = True
    is_feature_model = True

    def __init__(self, tasks, lr=DL_LR, epochs=DL_EPOCHS,
                 patience=DL_PATIENCE, batch_size=DL_BATCH, seed=SEED):
        self.tasks = tasks
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_maps = {}
        self.inv_maps = {}
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features

        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y_dict, groups=None, wn=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if wn is not None:
            self._wn = wn
        feats = self._extract(X)

        task_num_classes = []
        encoded_y = {}
        for task in self.tasks:
            task_id = task['id']
            labels = sorted(set(y_dict[task_id]))
            self.label_maps[task_id] = {label: idx for idx, label in enumerate(labels)}
            self.inv_maps[task_id] = {idx: label for label, idx in self.label_maps[task_id].items()}
            task_num_classes.append((task_id, len(labels)))
            encoded_y[task_id] = torch.tensor(
                [self.label_maps[task_id][value] for value in y_dict[task_id]],
                dtype=torch.long,
            )

        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=DL_VAL_FRAC, random_state=self.seed)
            t_idx, v_idx = next(gss.split(feats, encoded_y[self.tasks[0]['id']].numpy(), groups))
        else:
            idx = np.random.permutation(len(feats))
            n_val = max(int(len(feats) * DL_VAL_FRAC), 1)
            t_idx, v_idx = idx[n_val:], idx[:n_val]

        if AUG_ENABLED:
            rng = np.random.RandomState(self.seed)
            F_tr = feats[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}
            aug_feats = [F_tr]
            aug_labels = {task['id']: [y_dict_tr[task['id']]] for task in self.tasks}
            for _ in range(AUG_N):
                noisy = F_tr + rng.normal(0, 0.02, F_tr.shape) * np.std(F_tr, axis=0, keepdims=True)
                noisy *= rng.uniform(0.95, 1.05, (len(noisy), 1))
                aug_feats.append(noisy)
                for task in self.tasks:
                    aug_labels[task['id']].append(y_dict_tr[task['id']].copy())
            F_tr = np.concatenate(aug_feats, axis=0)
            y_dict_tr = {task_id: np.concatenate(parts) for task_id, parts in aug_labels.items()}
        else:
            F_tr = feats[t_idx]
            y_dict_tr = {task['id']: encoded_y[task['id']][t_idx].numpy() for task in self.tasks}

        train_tensors = [torch.tensor(F_tr, dtype=torch.float32)] + [
            torch.tensor(y_dict_tr[task['id']], dtype=torch.long) for task in self.tasks
        ]
        val_tensors = [torch.tensor(feats[v_idx], dtype=torch.float32)] + [
            encoded_y[task['id']][v_idx] for task in self.tasks
        ]
        train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=256, shuffle=False)

        self.model = _MTFeatureKANNet(feats.shape[1], task_num_classes).to(self.device)
        n_tasks = len(self.tasks)
        log_vars = nn.Parameter(torch.zeros(n_tasks, device=self.device))
        task_losses = [
            _make_loss(train_tensors[i + 1].cpu().numpy(), len(self.label_maps[self.tasks[i]['id']]))
            for i in range(n_tasks)
        ]
        val_loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + [log_vars],
            lr=self.lr,
            weight_decay=DL_WEIGHT_DECAY,
        )
        if DL_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_loss = float('inf')
        best_state = None
        wait = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                xb = batch[0].to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = sum(
                    torch.exp(-log_vars[i]) * task_losses[i](logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                    + 0.5 * log_vars[i]
                    for i in range(n_tasks)
                )
                loss.backward()
                optimizer.step()

            self.model.eval()
            epoch_val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch[0].to(self.device)
                    logits = self.model(xb)
                    batch_loss = sum(
                        val_loss(logits[self.tasks[i]['id']], batch[i + 1].to(self.device))
                        for i in range(n_tasks)
                    )
                    epoch_val_loss += batch_loss.item() * len(xb)
                    val_count += len(xb)

            epoch_val_loss /= max(val_count, 1)
            if DL_SCHEDULER == 'cosine':
                scheduler.step()
            else:
                scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                wait = 0
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        feats = self._extract(X)
        Xt = torch.tensor(feats, dtype=torch.float32).to(self.device)
        preds = {}
        with torch.no_grad():
            logits = self.model(Xt)
            for task_id, task_logits in logits.items():
                idx = task_logits.argmax(1).cpu().numpy()
                preds[task_id] = np.array([self.inv_maps[task_id][value] for value in idx])
        return preds


def get_mt_feature_kan():
    return _MTFeatureKANWrapper(TASKS)


def get_mt_feature_kan_3c():
    return _MTFeatureKANWrapper(TASKS_MT_FULL)


class _EnsembleWrapper:
    """Soft-voting ensemble: RF(features) + Feature-KAN(features)."""
    is_feature_model = True

    def __init__(self, seed=SEED):
        self.seed = seed
        self._rf = None
        self._kan = None
        self._wn = None

    def _extract(self, X):
        from src.feature_engineering import extract_all_features
        features, _ = extract_all_features(X, self._wn)
        return features

    def fit(self, X, y, groups=None, wn=None):
        if wn is not None:
            self._wn = wn
        self._rf = _MLFeatureWrapper(get_rf)
        self._rf.fit(X, y, groups=groups, wn=wn)
        self._kan = _FeatureKANWrapper()
        self._kan.fit(X, y, groups=groups, wn=wn)
        return self

    def predict(self, X):
        feats = self._extract(X)
        # RF: use predict_proba for soft voting
        rf_proba = self._rf._clf.predict_proba(feats)
        # KAN: get logits → softmax
        import torch
        self._kan.model.eval()
        Xt = torch.tensor(feats, dtype=torch.float32).to(self._kan.device)
        with torch.no_grad():
            kan_logits = self._kan.model(Xt)
            kan_proba = torch.softmax(kan_logits, dim=-1).cpu().numpy()
        # Align class dimensions (RF may have fewer classes)
        n_classes = max(rf_proba.shape[1], kan_proba.shape[1])
        if rf_proba.shape[1] < n_classes:
            pad = np.zeros((rf_proba.shape[0], n_classes - rf_proba.shape[1]))
            rf_proba = np.hstack([rf_proba, pad])
        if kan_proba.shape[1] < n_classes:
            pad = np.zeros((kan_proba.shape[0], n_classes - kan_proba.shape[1]))
            kan_proba = np.hstack([kan_proba, pad])
        # Average probabilities
        avg_proba = 0.5 * rf_proba + 0.5 * kan_proba
        pred_idx = np.argmax(avg_proba, axis=1)
        # Map back to original labels
        inv_map = self._kan.inv_map
        return np.array([inv_map[idx] for idx in pred_idx])


def get_ensemble():
    return _EnsembleWrapper()


MODEL_REGISTRY = {
    # ML full-spectrum
    'RF': get_rf,
    'ExtraTrees': get_extra_trees,
    'SVM': get_svm,
    'PLS-DA': get_plsda,
    'KNN': get_knn,
    'LDA': get_lda,
    'HistGradientBoosting': get_hist_gradient_boosting,
    'XGBoost': get_xgboost,
    # ML features
    'RF-feat': get_rf_feat,
    'SVM-feat': get_svm_feat,
    'PLS-DA-feat': get_plsda_feat,
    # DL full-spectrum
    '1D-CNN': get_cnn1d,
    '1D-ResNet': get_resnet1d,
    'RamanNet-Lite': get_ramannet_lite,
    'PatchTransformer': get_patch_transformer,
    'Spectrum-KAN': get_spectrum_kan,
    'Chem-KAN': get_chem_kan,
    'KAN-CNN': get_kan_cnn,
    # DL features
    '1D-CNN-feat': get_cnn1d_feat,
    '1D-ResNet-feat': get_resnet1d_feat,
    'Feature-KAN': get_feature_kan,
    'KAN-CNN-feat': get_kan_cnn_feat,
    # Ensemble
    'Ensemble': get_ensemble,
    # Multi-task (kept for optional use)
    'MT-CNN': get_mt_cnn,
    'MT-ResNet': get_mt_resnet,
    'MT-KAN-CNN': get_mt_kan_cnn,
    'MT-Feature-KAN': get_mt_feature_kan,
    'MT-CNN-3c': get_mt_cnn_3c,
    'MT-ResNet-3c': get_mt_resnet_3c,
    'MT-KAN-CNN-3c': get_mt_kan_cnn_3c,
    'MT-Feature-KAN-3c': get_mt_feature_kan_3c,
}
