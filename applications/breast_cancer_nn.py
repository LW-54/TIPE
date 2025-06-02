import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

from encephalon.nn import NN
from encephalon.types_and_functions import ReLU, Id, tanh, sigmoid
from encephalon.serial_utils import simulate_serial, SerialJSONInterface, handshake
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.representations import auto_subplot, graph_3d, decision_boundary


# ─── 1) LOAD THE BREAST CANCER DATASET ──────────────────────────────────────────
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
X_full = breast_cancer_dataset.data        # shape (569, 30)
y_full = breast_cancer_dataset.target      # shape (569, )

# We choose two features: index 0 = 'mean radius', index 1 = 'mean texture'
# (You can inspect breast_cancer_dataset.feature_names to pick other indices.)
X_raw = X_full[:, [0, 1]]                  # shape (569, 2)

# ─── 2) MIN–MAX NORMALIZE TO [0, 1] FOR EACH FEATURE ─────────────────────────
#   (so our NN sees inputs in roughly the same range)
mins = X_raw.min(axis=0)    # shape (2,)
maxs = X_raw.max(axis=0)    # shape (2,)
X_norm = (X_raw - mins) / (maxs - mins)

# ─── 3) WRAP DATA INTO LISTS OF FLOATS, AND TARGETS INTO [[0],[1]] FORMAT ─────
#    (our NN expects data as List[List[float]] and labels as List[List[int]] )
data_all = X_norm.tolist()                   # 569 × [x0, x1]
labels_all = [[int(label)] for label in y_full]  # 569 × [[0]] or [[1]]

# ─── 4) SPLIT INTO TRAIN AND TEST SETS ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data_all, labels_all, test_size=0.2, random_state=42, stratify=labels_all
)
# Now:
#   X_train: 455 samples  × 2
#   y_train: 455 samples  × 1
#   X_test: 114 samples   × 2
#   y_test: 114 samples   × 1

# ─── 5) BUILD/SIMULATE THE SERIAL INTERFACE ────────────────────────────────────

layers, f, g = [2, 3, 1], tanh, Id

if True :
    interface, sim = simulate_serial(make_arduino_simulator(layers, f=f, noise_amplitude=0))
    sim.start()
else :
    interface = SerialJSONInterface(port="/dev/ttyACM0", baud=9600, timeout=1.0)

handshake(interface)


# ─── 6) INITIALIZE THE NN ─────────────────────────────────────────────────────
bcn = NN(interface, layers, name="bcn", f=f, g=g, verbose=True)

# ─── 7) TRAIN ON THE TRAINING SET ─────────────────────────────────────────────
bcn.train(X_train, y_train, epochs=1000, batch_size=10, graphing=False)



# ─── 8) EVALUATE ON THE TEST SET ──────────────────────────────────────────────
correct = 0
for x_vec, y_true in zip(X_test, y_test):
    y_pred_raw = bcn.use(x_vec)[0]      # bcn.use returns a 1-element list [score]
    y_pred_label = 1 if (y_pred_raw >= 0.5) else 0
    if y_pred_label == y_true[0]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Test Accuracy: 89.47%

# ─── 9) PLOT 3D SURFACE & DECISION BOUNDARY ───────────────────────────────────
# Because we've normalized both features to [0,1], we can plot between 0 and 1 directly.
to_plot = [
    (
        graph_3d,
        dict(
            model=bcn,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            n=25,
        ),
    ),
    (
        decision_boundary,
        dict(
            model=bcn,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            n=50,
            boundary=0.5,
            # For visualization, we’ll color the positive class (y=1) vs. negative (y=0)
            data_0=[x for x, y in zip(X_norm, labels_all) if y == [0]],
            data_1=[x for x, y in zip(X_norm, labels_all) if y == [1]],
        ),
    ),
]

auto_subplot(1, 2, to_plot, figsize=(10, 5))

# ─── 10) CLEAN UP ─────────────────────────────────────────────────────────────
sim.stop()
interface.close()

