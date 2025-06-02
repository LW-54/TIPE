import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from encephalon.nn import NN
from encephalon.types_and_functions import tanh, Id
from encephalon.serial_utils import simulate_serial, SerialJSONInterface, handshake
from encephalon.arduino_sim import make_arduino_simulator
from encephalon.representations import auto_subplot, graph_3d, decision_boundary

# ─── 1) GENERATE A “CONCENTRIC CIRCLES” DATASET ───────────────────────────────
#    We’ll create 500 points in two concentric circles:
#      • inner circle (label = 0)
#      • outer ring   (label = 1)
#    The `factor` parameter controls the radius of the inner circle relative to the outer ring.
#    The entire dataset is returned as X (shape: 500×2) and y (shape: 500,).
X_raw, y_raw = make_circles(
    n_samples=500,
    noise=0.08,   # add a bit of Gaussian noise so it’s not perfectly separable
    factor=0.4,   # inner‐circle radius = 0.4 * outer radius
    random_state=42
)
# Now X_raw ∈ [−1,1]² (approximately), and y_raw ∈ {0,1}.

# ─── 2) (OPTIONAL) MIN–MAX NORMALIZE TO [0,1] ────────────────────────────────
#    We could leave the raw points in [−1,1], but it’s often easier to visualize
#    and train if everything is in [0,1]. We'll shift + scale:
mins = X_raw.min(axis=0)
maxs = X_raw.max(axis=0)
X_norm = (X_raw - mins) / (maxs - mins)
# Now X_norm ∈ [0,1]².

# ─── 3) FORMAT THE LABELS FOR `NN.train(...)` ────────────────────────────────
#    The encephalon `NN` expects labels as a list of single‐element lists,
#    e.g. [[0], [1], [0], …].
data_all   = X_norm.tolist()                   # 500 × [x0, x1]
labels_all = [[int(label)] for label in y_raw]  # 500 × [[0]] or [[1]]

# ─── 4) SPLIT INTO TRAIN / TEST ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data_all,
    labels_all,
    test_size=0.2,
    random_state=42,
    stratify=labels_all
)
# Now we have:
#   X_train: 400 points × 2
#   y_train: 400 labels × 1
#   X_test: 100 points × 2
#   y_test: 100 labels × 1

# ─── 5) BUILD / SIMULATE THE SERIAL INTERFACE ─────────────────────────────────
layers, f, g = [2, 3, 1], tanh, Id
# Note: I bumped hidden‐layer size to 8 to give the net more capacity to carve out a circular boundary.
if True:
    interface, sim = simulate_serial(
        make_arduino_simulator(layers, f=f, noise_amplitude=0)
    )
    sim.start()
else:
    interface = SerialJSONInterface(port="/dev/ttyACM0", baud=9600, timeout=1.0)

handshake(interface)

# ─── 6) INITIALIZE THE NEURAL NETWORK ─────────────────────────────────────────
circle_net = NN(interface, layers, name="circle_net", f=f, g=g, verbose=True)

# ─── 7) TRAIN ON THE TRAINING SET ─────────────────────────────────────────────
circle_net.train(
    X_train,
    y_train,
    epochs=1000,      # you can experiment with more/less epochs
    batch_size=10,
    graphing=True
)

# ─── 8) EVALUATE ON THE TEST SET ──────────────────────────────────────────────
num_correct = 0
for x_vec, y_true in zip(X_test, y_test):
    y_pred_raw   = circle_net.use(x_vec)[0]
    y_pred_label = 1 if (y_pred_raw >= 0.5) else 0
    if y_pred_label == y_true[0]:
        num_correct += 1

accuracy = num_correct / len(X_test)
print(f"Test Accuracy on concentric‐circles: {accuracy * 100:.2f}%")

# ─── 9) PLOT DECISION SURFACE + BOUNDARY ───────────────────────────────────────
# Since we normalized X into [0,1]², we set the plotting range accordingly.
to_plot = [
    (
        graph_3d,
        dict(
            model=circle_net,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            n=50,      # 50×50 grid for a smooth surface
        ),
    ),
    (
        decision_boundary,
        dict(
            model=circle_net,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            n=100,     # finer grid for a crisp boundary
            boundary=0.5,
            # pass all points (normalized) so we can color‐code inner (0) vs. outer (1)
            data_0=[x for x, y in zip(X_norm.tolist(), labels_all) if y == [1]],
            data_1=[x for x, y in zip(X_norm.tolist(), labels_all) if y == [0]],
        ),
    ),
]

auto_subplot(1, 2, to_plot, figsize=(12, 5))

# ─── 10) CLEAN UP ─────────────────────────────────────────────────────────────
sim.stop()
interface.close()
