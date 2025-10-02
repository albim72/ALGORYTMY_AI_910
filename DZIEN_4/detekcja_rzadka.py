# rare_local_pattern_detector.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1) SYNTHETYCZNE DANE
# -----------------------------
def synth_sequence(seq_len=512, pattern_len=21, prevalence=0.02, noise_std=0.5, seed=None):
    """
    Buduje sekwencję szumu + (opcjonalnie) lokalny rzadki wzorzec (np. pik + oscylacja).
    Zwraca: x (float), maskę binarną (gdzie pattern występuje).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    x = rng.normal(0.0, noise_std, size=(seq_len,))
    mask = np.zeros(seq_len, dtype=np.int32)

    # czy wstrzyknąć pattern w tej sekwencji?
    if rng.random() < prevalence:
        start = rng.integers(low=5, high=seq_len - pattern_len - 5)
        t = np.linspace(0, 1, pattern_len)
        # wzorzec: pik + lokalna oscylacja z dekadem amplitudy
        pattern = 2.5*np.exp(-8*(t-0.3)**2) + 1.0*np.sin(2*np.pi*5*t)*np.exp(-3*t)
        x[start:start+pattern_len] += pattern
        mask[start:start+pattern_len] = 1

    return x.astype(np.float32), mask

def make_dataset(n_sequences=4000, seq_len=512, pattern_len=21, prevalence=0.02, seed=123):
    rng = np.random.default_rng(seed)
    X = []
    M = []
    for i in range(n_sequences):
        x, m = synth_sequence(seq_len, pattern_len, prevalence, seed=int(rng.integers(1e9)))
        X.append(x)
        M.append(m)
    return np.stack(X), np.stack(M)

# -----------------------------
# 2) SLIDING WINDOW + ETYKIETY
# -----------------------------
def windows_from_sequences(X, M, win=64, stride=4, label_rule="any"):
    """
    Tworzy okna 1D. Etykieta = czy wzorzec występuje w oknie.
    label_rule: "any" (1 jeśli w oknie jest choć jedno 1 w masce)
                "center" (1 jeśli środek okna wypada w obszarze wzorca)
    """
    Xw, Yw = [], []
    half = win // 2
    for seq, mask in zip(X, M):
        for s in range(0, len(seq) - win + 1, stride):
            w = seq[s:s+win]
            m = mask[s:s+win]
            if label_rule == "any":
                y = 1 if m.any() else 0
            else:
                y = int(m[half] == 1)
            Xw.append(w)
            Yw.append(y)
    Xw = np.expand_dims(np.array(Xw, dtype=np.float32), axis=-1)   # (N, win, 1)
    Yw = np.array(Yw, dtype=np.int32)
    return Xw, Yw

# -----------------------------
# 3) FOCAL LOSS
# -----------------------------
def binary_focal_loss(gamma=2.0, alpha=0.95):
    """
    Focal loss dla silnej nierównowagi klas.
    """
    bce = keras.losses.BinaryCrossentropy(reduction="none")
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        b = bce(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        w = alpha * tf.pow(1 - p_t, gamma)
        return tf.reduce_mean(w * b)
    return loss

# -----------------------------
# 4) MODEL: DYLATOWANE CONV + RES + ATTENTION
# -----------------------------
def se_block(x, r=8):
    # Squeeze-&-Excitation (kanałowa atencja)
    c = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(c // r, 1), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1, c))(s)
    return layers.Multiply()([x, s])

def conv_block(x, filters, kernel=7, dilation=1):
    y = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(filters, 3, padding="same", dilation_rate=dilation)(y)
    y = layers.BatchNormalization()(y)
    # Residual (dopasuj kanały jeśli trzeba)
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)
    y = layers.Add()([x, y])
    y = layers.Activation("relu")(y)
    y = se_block(y)
    return y

def build_model(win=64):
    inp = layers.Input(shape=(win, 1))
    x = conv_block(inp, 32, kernel=7, dilation=1)
    x = conv_block(x, 32, kernel=7, dilation=2)
    x = conv_block(x, 64, kernel=7, dilation=4)
    x = conv_block(x, 64, kernel=7, dilation=8)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    return model

# -----------------------------
# 5) DANE + PODZIAŁ
# -----------------------------
SEQ_LEN = 512
PAT_LEN = 21
WIN = 64
STRIDE = 4
PREV = 0.02

X, M = make_dataset(n_sequences=4000, seq_len=SEQ_LEN, pattern_len=PAT_LEN, prevalence=PREV, seed=42)
# train/val/test po sekwencjach (ważne, by okna z tej samej sekwencji nie mieszały się)
n = len(X)
idx = np.arange(n)
np.random.default_rng(0).shuffle(idx)
train_idx, val_idx, test_idx = idx[:int(0.7*n)], idx[int(0.7*n):int(0.85*n)], idx[int(0.85*n):]

Xtr, Ytr = windows_from_sequences(X[train_idx], M[train_idx], win=WIN, stride=STRIDE, label_rule="any")
Xva, Yva = windows_from_sequences(X[val_idx],   M[val_idx],   win=WIN, stride=STRIDE, label_rule="any")
Xte, Yte = windows_from_sequences(X[test_idx],  M[test_idx],  win=WIN, stride=STRIDE, label_rule="any")

print("Shapes:", Xtr.shape, Xva.shape, Xte.shape)
pos_ratio = Ytr.mean()
print(f"Train positive ratio: {pos_ratio:.4f}")

# class weight (opcjonalnie — i tak mamy focal loss)
from collections import Counter
ctr = Counter(Ytr.tolist())
total = len(Ytr)
cw = {0: total/(2.0*ctr[0]), 1: total/(2.0*ctr[1])}
print("class_weight:", cw)

# tf.data
BATCH = 256
ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(8192).batch(BATCH).prefetch(tf.data.AUTOTUNE)
ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
ds_te = tf.data.Dataset.from_tensor_slices((Xte, Yte)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 6) TRENING
# -----------------------------
model = build_model(win=WIN)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=binary_focal_loss(gamma=2.0, alpha=0.95),
    metrics=[
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(curve="PR", name="auc_pr")
    ],
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", factor=0.5, patience=3, mode="max", verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_auc_pr", patience=8, mode="max", restore_best_weights=True, verbose=1)
]

hist = model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=40,
    class_weight=cw,
    callbacks=callbacks,
    verbose=2
)

# -----------------------------
# 7) DOBÓR PROGU (max F1 na walidacji)
# -----------------------------
val_probs = model.predict(ds_va, verbose=0).ravel()
val_y = Yva.astype(np.int32)

def best_f1_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_f1, best_p, best_r = 0.5, 0.0, 0.0, 0.0
    for t in thresholds:
        y_hat = (y_prob >= t).astype(np.int32)
        tp = np.sum((y_hat==1) & (y_true==1))
        fp = np.sum((y_hat==1) & (y_true==0))
        fn = np.sum((y_hat==0) & (y_true==1))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2*p*r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, p, r
    return best_t, best_f1, best_p, best_r

thr, f1, p, r = best_f1_threshold(val_y, val_probs)
print(f"[VAL] Best F1={f1:.3f} at threshold={thr:.2f} (P={p:.3f}, R={r:.3f})")

# -----------------------------
# 8) TEST
# -----------------------------
test_probs = model.predict(ds_te, verbose=0).ravel()
test_y = Yte.astype(np.int32)
y_hat = (test_probs >= thr).astype(np.int32)

tp = np.sum((y_hat==1) & (test_y==1))
fp = np.sum((y_hat==1) & (test_y==0))
fn = np.sum((y_hat==0) & (test_y==1))
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1_test = 2*precision*recall / (precision + recall + 1e-9)

auc_pr_metric = keras.metrics.AUC(curve="PR")
auc_pr_metric.update_state(test_y, test_probs)
auc_pr = auc_pr_metric.result().numpy()

print(f"[TEST] P={precision:.3f} R={recall:.3f} F1={f1_test:.3f} AUC-PR={auc_pr:.3f} @thr={thr:.2f}")

# -----------------------------
# 9) INFERENCJA PO CAŁEJ SEKWENCJI
# -----------------------------
def detect_in_sequence(seq, model, win=64, stride=4, threshold=0.5):
    """
    Zwraca listę (start, end, score) wykrytych obszarów po złączeniu sąsiadów.
    """
    windows = []
    idxs = []
    for s in range(0, len(seq) - win + 1, stride):
        windows.append(seq[s:s+win])
        idxs.append(s)
    Xw = np.expand_dims(np.array(windows, dtype=np.float32), axis=-1)
    probs = model.predict(Xw, verbose=0).ravel()
    hits = []
    for start, p in zip(idxs, probs):
        if p >= threshold:
            hits.append((start, start+win, p))
    # scal sąsiadujące
    if not hits:
        return []
    hits.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e, cur_max = hits[0]
    for s, e, p in hits[1:]:
        if s <= cur_e:  # zachodzi
            cur_e = max(cur_e, e)
            cur_max = max(cur_max, p)
        else:
            merged.append((cur_s, cur_e, cur_max))
            cur_s, cur_e, cur_max = s, e, p
    merged.append((cur_s, cur_e, cur_max))
    return merged

# Demo inferencji na jednej sekwencji testowej:
seq_demo = X[test_idx[0]]
dets = detect_in_sequence(seq_demo, model, win=WIN, stride=STRIDE, threshold=thr)
print("Wykrycia (start, end, score):", dets)
