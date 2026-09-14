import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.ticker import ScalarFormatter  
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict, Counter
from matplotlib.ticker import FuncFormatter


class visualization_utils:

    # ============================================================
    # LABEL NORMALIZATION
    # ============================================================
    @staticmethod
    def plot_lot_projection(Xtr, Ytr, Xte, Yte, tube_name="P1"):

        mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)
    
        proj_train = (Xtr - mu_all) @ v_hat
        proj_test  = (Xte - mu_all) @ v_hat
    
        sigma = np.std(proj_train, ddof=1)
    
        proj_train /= sigma
        proj_test /= sigma
    
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    
        for ax, proj, labels, title in zip(
            axes,
            [proj_train, proj_test],
            [Ytr, Yte],
            ["Train", "Test"]
        ):
    
            ax.hist(proj[labels == 0], bins=12, alpha=0.5, label="NBM")
            ax.hist(proj[labels == 1], bins=12, alpha=0.5, label="AML")
    
            ax.set_title(title)
    
        axes[0].set_ylabel("Density")
        axes[0].legend()
    
        plt.tight_layout()
        plt.show()
    @staticmethod
    def normalize_label(s):
        s = str(s).strip()

        if s.startswith("[b'") and s.endswith("']"):
            s = s[3:-2]
        if s.startswith("b'") and s.endswith("'"):
            s = s[2:-1]

        return s

    # ============================================================
    # LOAD LOT DATA (PATIENT LEVEL)
    # ============================================================
    @staticmethod
    def load_lot_by_tube(
        tube_name,
        h5_data_path,
        patient_ids=None,
        preprocess_id="1",
        embedding_name="patient0_emd2_map",
        dataset_name=None,
        subsampled_cell_count="1000",
    ):

        X_list, Y_list = [], []

        with h5py.File(h5_data_path, "r") as h5:

            # Check layout: Stage 2 format vs Legacy format
            tube_grp = None
            dataset_detected = dataset_name
            if dataset_detected is None:
                for candidate in ["BLAST110", "FLOWCAPII", "LAIP29"]:
                    if candidate in h5 and str(subsampled_cell_count) in h5[candidate]:
                        if tube_name in h5[candidate][str(subsampled_cell_count)]:
                            dataset_detected = candidate
                            break

            if dataset_detected and dataset_detected in h5 and str(subsampled_cell_count) in h5[dataset_detected]:
                tube_grp = h5[dataset_detected][str(subsampled_cell_count)][tube_name]
            elif str(subsampled_cell_count) in h5 and tube_name in h5[str(subsampled_cell_count)]:
                tube_grp = h5[str(subsampled_cell_count)][tube_name]
            else:
                raise ValueError(f"Tube {tube_name} not found in {h5_data_path}")

            # Case A: Stage 2 layout with lot_embeddings
            if "lot_embeddings" in tube_grp:
                prep_grp = tube_grp["lot_embeddings"].get(str(preprocess_id), None)
                if prep_grp is None and len(tube_grp["lot_embeddings"].keys()) > 0:
                    prep_grp = tube_grp["lot_embeddings"][list(tube_grp["lot_embeddings"].keys())[0]]

                if prep_grp is None:
                    raise ValueError(f"No lot_embeddings found for tube {tube_name}")

                if embedding_name in prep_grp:
                    emb_node = prep_grp[embedding_name]
                else:
                    emb_node = prep_grp[list(prep_grp.keys())[0]]

                embs = emb_node["embeddings"][:]
                raw_pids = [p.decode() if isinstance(p, bytes) else str(p) for p in emb_node["patient_ids"][:]]

                meta_pids = [p.decode() if isinstance(p, bytes) else str(p) for p in tube_grp["metadata"]["patient_ids"][:]]
                meta_labels = [l.decode() if isinstance(l, bytes) else str(l) for l in tube_grp["metadata"]["labels"][:]]
                label_dict = dict(zip(meta_pids, meta_labels))

                if patient_ids is not None:
                    target_pids = set(str(p) for p in patient_ids)
                else:
                    target_pids = None

                for emb, pid in zip(embs, raw_pids):
                    if target_pids is not None and pid not in target_pids:
                        continue
                    lbl = visualization_utils.normalize_label(label_dict.get(pid, ""))
                    if lbl in ["normal", "NBM", 0, "0"]:
                        y_val = 0
                    elif lbl in ["AML_Dx", "AML", 1, "1"]:
                        y_val = 1
                    else:
                        continue
                    X_list.append(emb)
                    Y_list.append(y_val)

            # Case B: Legacy layout with lot_per_patient
            elif "lot_per_patient" in tube_grp:
                lot_grp = tube_grp["lot_per_patient"]
                label_grp = tube_grp["labels_per_patient"]

                if patient_ids is None:
                    patient_ids = list(lot_grp.keys())

                for pid in patient_ids:
                    pid = str(pid)
                    if pid not in lot_grp:
                        continue

                    lot_vec = lot_grp[pid][()]
                    label = visualization_utils.normalize_label(label_grp[pid][()])

                    if label in ["normal", "NBM"]:
                        y_val = 0
                    elif label in ["AML_Dx", "AML"]:
                        y_val = 1
                    else:
                        continue

                    X_list.append(lot_vec)
                    Y_list.append(y_val)
            else:
                raise ValueError(f"Neither lot_embeddings nor lot_per_patient found in {tube_name}")

        if len(X_list) == 0:
            raise ValueError(f"No valid samples found for tube {tube_name}.")

        return np.vstack(X_list), np.array(Y_list)

    # ============================================================
    # LOAD ORIGINAL DATA (CELL LEVEL)
    # ============================================================
    @staticmethod
    def load_org_data_by_tube(
        tube_name,
        h5_data_path,
        patient_ids=None,
        preprocess_id="1",
        dataset_name=None,
        subsampled_cell_count="1000",
    ):

        X_list, Y_list = [], []

        with h5py.File(h5_data_path, "r") as h5:

            tube_grp = None
            dataset_detected = dataset_name
            if dataset_detected is None:
                for candidate in ["BLAST110", "FLOWCAPII", "LAIP29"]:
                    if candidate in h5 and str(subsampled_cell_count) in h5[candidate]:
                        if tube_name in h5[candidate][str(subsampled_cell_count)]:
                            dataset_detected = candidate
                            break

            if dataset_detected and dataset_detected in h5 and str(subsampled_cell_count) in h5[dataset_detected]:
                tube_grp = h5[dataset_detected][str(subsampled_cell_count)][tube_name]
            elif str(subsampled_cell_count) in h5 and tube_name in h5[str(subsampled_cell_count)]:
                tube_grp = h5[str(subsampled_cell_count)][tube_name]
            else:
                raise ValueError(f"Tube {tube_name} not found in {h5_data_path}")

            # Case A: Stage 2 layout with preprocess_1 or raw
            prep_key = f"preprocess_{preprocess_id}"
            if prep_key in tube_grp or "raw" in tube_grp:
                data_parent = tube_grp[prep_key] if prep_key in tube_grp else tube_grp["raw"]
                matrix_name = "processed_matrix" if prep_key in tube_grp else "raw_cell_matrix"

                meta_pids = [p.decode() if isinstance(p, bytes) else str(p) for p in tube_grp["metadata"]["patient_ids"][:]]
                meta_labels = [l.decode() if isinstance(l, bytes) else str(l) for l in tube_grp["metadata"]["labels"][:]]
                label_dict = dict(zip(meta_pids, meta_labels))

                if patient_ids is None:
                    patient_ids = [k for k in data_parent.keys() if isinstance(data_parent[k], h5py.Group)]

                for pid in patient_ids:
                    pid = str(pid)
                    if pid not in data_parent or not isinstance(data_parent[pid], h5py.Group):
                        continue
                    lbl = visualization_utils.normalize_label(label_dict.get(pid, ""))
                    if lbl in ["normal", "NBM", 0, "0"]:
                        y_val = 0
                    elif lbl in ["AML_Dx", "AML", 1, "1"]:
                        y_val = 1
                    else:
                        continue
                    pdata = data_parent[pid][matrix_name][()]
                    X_list.append(pdata)
                    Y_list.extend([y_val] * pdata.shape[0])

            # Case B: Legacy layout with pdata_aligned
            elif "pdata_aligned" in tube_grp:
                pdata_grp = tube_grp["pdata_aligned"]
                label_grp = tube_grp["labels_per_patient"]

                if patient_ids is None:
                    patient_ids = list(pdata_grp.keys())

                for pid in patient_ids:
                    pid = str(pid)
                    if pid not in pdata_grp:
                        continue

                    pdata = pdata_grp[pid][()]
                    label = visualization_utils.normalize_label(label_grp[pid][()])

                    if label in ["normal", "NBM"]:
                        y_val = 0
                    elif label in ["AML_Dx", "AML"]:
                        y_val = 1
                    else:
                        continue

                    X_list.append(pdata)
                    Y_list.extend([y_val] * pdata.shape[0])
            else:
                raise ValueError(f"No cell matrix data found in {tube_name}")

        if len(X_list) == 0:
            raise ValueError(f"No valid cell samples found for tube {tube_name}.")

        return np.vstack(X_list), np.array(Y_list)


    # ============================================================
    # SPLIT
    # ============================================================
    @staticmethod
    def stratified_split(X, Y):

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=42
        )

        tr, tt = next(splitter.split(X, Y))
        return X[tr], Y[tr], X[tt], Y[tt]

    # ============================================================
    # LOT → MATRIX
    # ============================================================
    @staticmethod
    def lot_vec_to_V(lot_vec, n_markers):
        n_temp = lot_vec.size // n_markers
        return lot_vec.reshape((n_temp, n_markers), order="F").T

    # ============================================================
    # DISCRIMINANT DIRECTION
    # ============================================================
    @staticmethod
    def compute_discriminant_direction(Xtr, Ytr):

        X0 = Xtr[Ytr == 0]
        X1 = Xtr[Ytr == 1]

        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        mu_all = Xtr.mean(axis=0)

        v = mu1 - mu0
        v_hat = v / np.linalg.norm(v)

        return mu0, v_hat, mu_all

    # ============================================================
    # MARKER MAPPING
    # ============================================================
    # @staticmethod
    # def get_marker_names_from_tube(tube_name):

    #     common_map = {
    #         'FSC-A': 'FSC-A',
    #         'FSC-H': 'FSC-H',
    #         'SSC-A': 'SSC-A',
    #         'SSC-H': 'SSC-H',
    #         'PerCP-A': 'CD34',
    #         'PC7-A': 'CD117',
    #         'APC-H7-A': 'HLA-DR',
    #         'Horizon V450-A': 'CD13',
    #         'Horizon V500-A': 'CD45'
    #     }

    #     tube_map = {
    #         "P1": {'FITC-A': 'CD7',  'PE-A': 'CD56', 'APC-A': 'CD33'},
    #         "P2": {'FITC-A': 'CD15', 'PE-A': 'CD22', 'APC-A': 'CD19'},
    #         "P3": {'FITC-A': 'CD36', 'PE-A': 'CD14', 'APC-A': 'CD11B'},
    #         "P4": {'FITC-A': 'CD2',  'PE-A': 'CD133','APC-A': 'CD33'}
    #     }

    #     channels = [
    #         'FSC-A','FSC-H','SSC-A','SSC-H',
    #         'FITC-A','PE-A','PerCP-A','PC7-A',
    #         'APC-A','APC-H7-A','Horizon V450-A','Horizon V500-A'
    #     ]

    #     return [
    #         common_map[ch] if ch in common_map
    #         else tube_map[tube_name].get(ch, ch)
    #         for ch in channels
    #     ]

    # ============================================================
    # SIMPLE PROJECTION PLOT
    # ============================================================
    @staticmethod
    def plot_projection(Xtr, Ytr, Xte, Yte, tube_name):

        mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)

        proj_train = (Xtr - mu_all) @ v_hat
        proj_test  = (Xte - mu_all) @ v_hat

        fig, ax = plt.subplots(1,2, figsize=(7,3))

        for a, proj, y, name in zip(ax, [proj_train, proj_test], [Ytr, Yte], ["Train","Test"]):

            a.hist(proj[y==0], bins=12, range=(-2,2), alpha=0.5, label="NBM")
            a.hist(proj[y==1], bins=12,range=(-2,2), alpha=0.5, label="AML")

            a.set_title(name)
            a.legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_lot_pairwise_evolution_nature_final(
        Xtr, Ytr, Xte, Yte,
        tube_name,
        marker_names=None,
        K_VALUES=None,
        use_hexbin=False
    ):
        if marker_names is None:
            marker_names = visualization_utils.get_marker_names_from_tube(tube_name)
        if K_VALUES is None:
            K_VALUES = np.arange(-1.5, 1.75, 0.75)

        # Candidate pairs: FSC-A vs SSC-A, CD117 vs HLA-DR (or first valid pairs)
        candidate_pairs = [(0, 2), (6, 8)]
        PAIRS = [(i, j) for (i, j) in candidate_pairs if i < len(marker_names) and j < len(marker_names)]
        if not PAIRS and len(marker_names) >= 2:
            PAIRS = [(0, 1)]

        n_rows = len(PAIRS)
        n_cols = len(K_VALUES)

        print(f"\n✅ Processing tube: {tube_name}")

    
    
    
        # -----------------------------------
        # Discriminant direction
        # -----------------------------------
        mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)
        proj_train = (Xtr - mu_all) @ v_hat
        sigma_parallel = np.std(proj_train, ddof=1)
    
        print(f"σ∥ (train) = {sigma_parallel:.4f}")
    
        # -----------------------------------
        # Tick formatter
        # -----------------------------------
        scale = 1e5
        def scaled_formatter(x, pos):
            return f"{x/scale:.1f}"
    
        formatter = FuncFormatter(scaled_formatter)
    
        # -----------------------------------
        # Figure (tight grid)
        # -----------------------------------
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            # figsize=(2.6 * n_cols, 2.2 * n_rows),
            squeeze=False,
            gridspec_kw={"wspace": 0.25, "hspace": 0.25}
        )
    
        # # -----------------------------------
        # # Global sigma labels
        # # -----------------------------------
        # for col_idx, k in enumerate(K_VALUES):
        #     fig.text(
        #         0.1 + col_idx / n_cols * 0.8 + 0.4 / n_cols,
        #         0.94,
        #         rf"$\sigma = {k}$",
        #         ha="center",
        #         fontsize=10,
        #         fontweight="bold"
        #     )
    
        # -----------------------------------
        # Loop over pairs
        # -----------------------------------
        for row_idx, (i, j) in enumerate(PAIRS):
    
            # ----------- GLOBAL LIMITS (KEY ADDITION) -----------
            all_x = []
            all_y = []
    
            for k in K_VALUES:
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(x_k, len(marker_names))
                all_x.append(V[i, :])
                all_y.append(V[j, :])
    
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
    
            # 1–99 percentile limits
            x_min, x_max = np.percentile(all_x, [1, 99])
            y_min, y_max = np.percentile(all_y, [1, 99])
    
            # Optional symmetric limits (better for LOT geometry)
            x_lim = max(abs(x_min), abs(x_max))
            y_lim = max(abs(y_min), abs(y_max))
    
            # ---------------------------------------------------
    
            for col_idx, k in enumerate(K_VALUES):
    
                ax = axes[row_idx, col_idx]
    
                # LOT slice
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(x_k, len(marker_names))
    
                x_vals = V[i, :]
                y_vals = V[j, :]
    
                # Plot
                if use_hexbin:
                    ax.hexbin(
                        x_vals, y_vals,
                        gridsize=40,
                        cmap="Blues",
                        mincnt=1
                    )
                else:
                    ax.scatter(
                        x_vals, y_vals,
                        s=10,
                        color="#4C72B0",
                        alpha=1,
                        edgecolors="none"
                    )
    
                # Apply SAME limits across σ
                ax.set_xlim(0, x_lim)
                ax.set_ylim(0, y_lim)
    
                # Formatting
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                ax.tick_params(labelsize=6)
    
                # Clean labeling
                if col_idx == 0:
                    ax.set_ylabel(marker_names[j], fontsize=8, fontweight="bold")
    
                # if row_idx != 100:
                ax.set_xlabel(marker_names[i], fontsize=8, fontweight="bold")
    
                if row_idx == 0:
                    ax.set_title(rf"$\sigma={k}$", fontsize=12, fontweight="bold",pad=2)
    
        # -----------------------------------
        # Row labels
        # -----------------------------------
        # for row_idx, (i, j) in enumerate(PAIRS):
            # axes[row_idx, 0].annotate(
            #     f"{marker_names[i]} vs {marker_names[j]}",
            #     xy=(-0.35, 0.5),
            #     xycoords="axes fraction",
            #     fontsize=8,
            #     rotation=90,
            #     va="center",
            #     ha="right"
            # )
    
        # -----------------------------------
        # Title
        # -----------------------------------
        fig.suptitle(
            "Pairwise LOT evolution across σ",
            fontsize=11,
            fontweight="bold",
            y=0.98
        )
    
        # # Tight control (NO tight_layout)
        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.06,
            top=0.90,
            wspace=0.15,
            hspace=0.15
        )
    
        fig.savefig(
            "pairwise_k_grid_final.pdf",
            dpi=600,
            bbox_inches="tight"
        )
    
        plt.show()
    @staticmethod
    def plot_lot_pairwise_evolution_nature(Xtr, Ytr, Xte, Yte, tube_name="P1", marker_names=None):
        if marker_names is None:
            marker_names = visualization_utils.get_marker_names_from_tube(tube_name)

        # ---------------------------------------------------
        # Selective marker pairs
        # ---------------------------------------------------
        
        MAX_COLS = 3  # Nature-friendly layout
    
        print(f"\n✅ Processing tube: {tube_name}")
    
        # ---------------------------------------------------
        # Load data
        # ---------------------------------------------------
    
    
        # ---------------------------------------------------
        # Discriminant direction
        # ---------------------------------------------------
        mu0, v_hat,mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)
        # proj_train = (Xtr - mu_all) @ v_hat
        # sigma_parallel = np.std(proj_train, ddof=1)
    
            # ---------------------------------------------------
        # Projections
        # ---------------------------------------------------
        proj_train = (Xtr - mu_all) @ v_hat
        proj_test  = (Xte - mu_all) @ v_hat
    
        sigma_parallel = np.std(proj_train, ddof=1)
    
        # ✅ normalize using TRAIN std only
        proj_train /= sigma_parallel
        proj_test  /= sigma_parallel
    
        print(f"σ∥ (train) = {sigma_parallel:.4f}")
    
        # ---------------------------------------------------
        # Figure (Nature double-column)
        # ---------------------------------------------------
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 8),
            sharex=True,
            sharey=True
        )
    
        panels = [
            ("Train projection", proj_train, Ytr),
            ("Test projection",  proj_test,  Yte)
        ]
    
        class_info = {
            0: {"label": "NBM",     "color": "#377eb8"},
            1: {"label": "AML_Dx",  "color": "#e41a1c"}
        }
    
        for ax, (title, proj, labels) in zip(axes, panels):
    
            for cls, info in class_info.items():
    
                proj_c = proj[labels == cls]
    
                # Histogram
                ax.hist(
                    proj_c,
                    bins=12,
                    range=(-2,2),
                    density=True,
                    alpha=0.5,
                    color=info["color"],
                    edgecolor="black",
                    linewidth=0.5,
                    label=info["label"]
                )
    
                # # KDE
                # kde = gaussian_kde(proj_c)
                # xs = np.linspace(proj.min(), proj.max(), 400)
                # ax.plot(
                #     xs,
                #     kde(xs),
                #     color=info["color"],
                #     linewidth=2
                # )
    
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel(
                r"Projection along discriminant direction",
                fontsize=8
            )
            ax.tick_params(labelsize=7)
    
            # Clean scientific formatting
            # formatter = ScalarFormatter(useMathText=True)
            # formatter.set_scientific(True)
            # formatter.set_powerlimits((-2, 3))
            # ax.xaxis.set_major_formatter(formatter)
            # ax.yaxis.set_major_formatter(formatter)
    
        axes[0].set_ylabel("Density", fontsize=8)
    
        # ---------------------------------------------------
        # Legend (single, clean)
        # ---------------------------------------------------
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.92)
        )
    
        # ---------------------------------------------------
        # Title
        # ---------------------------------------------------
        fig.suptitle(
            "LOT geometry along discriminant axis",
            fontsize=10,
            fontweight="bold",
            y=0.98
        )
    
        plt.tight_layout(rect=[0, 0, 1, 0.88])
    
        # ---------------------------------------------------
        # Save (publication ready)
        # ---------------------------------------------------
        fig.savefig(
            "lot_projection_train_test_by_class.pdf",
            format="pdf",
            dpi=600,
            bbox_inches="tight"
        )
    
        plt.show()
    @staticmethod
    def get_marker_names_from_tube(
        tube_name,
        h5_data_path=None,
        preprocess_id="1",
        dataset_name=None,
        subsampled_cell_count="1000",
    ):
        # Dynamically check Stage 2 files
        candidates = []
        if h5_data_path:
            candidates.append(h5_data_path)
        candidates.extend([
            "data/stage2/stage2_analytics.h5",
            "../data/stage2/stage2_analytics.h5",
        ])

        for cpath in candidates:
            if os.path.exists(cpath):
                try:
                    with h5py.File(cpath, "r") as h5:
                        dname = dataset_name
                        if dname is None:
                            for c in ["BLAST110", "FLOWCAPII", "LAIP29"]:
                                if c in h5 and str(subsampled_cell_count) in h5[c]:
                                    if tube_name in h5[c][str(subsampled_cell_count)]:
                                        dname = c
                                        break
                        if dname and dname in h5 and str(subsampled_cell_count) in h5[dname]:
                            t_grp = h5[dname][str(subsampled_cell_count)][tube_name]
                            prep_key = f"preprocess_{preprocess_id}"
                            if prep_key in t_grp and "marker_subset" in t_grp[prep_key]:
                                return [m.decode() if isinstance(m, bytes) else str(m) for m in t_grp[prep_key]["marker_subset"][:]]
                            if "metadata" in t_grp and "marker_descriptions" in t_grp["metadata"]:
                                raw_markers = [m.decode() if isinstance(m, bytes) else str(m) for m in t_grp["metadata"]["marker_descriptions"][:]]
                                return [m for m in raw_markers if m not in ["Time", "event_ID"]]
                except Exception:
                    pass

        # Common mapping
        common_map = {
            'FSC-A': 'FSC-A',
            'FSC-H': 'FSC-H',
            'SSC-A': 'SSC-A',
            'SSC-H': 'SSC-H',
            'PerCP-A': 'CD34',
            'PC7-A': 'CD117',
            'APC-H7-A': 'HLA-DR',
            'Horizon V450-A': 'CD13',
            'Horizon V500-A': 'CD45'
        }
    
        # Tube-specific mapping
        tube_map = {
            "P1": {
                'FITC-A': 'CD7',
                'PE-A': 'CD56',
                'APC-A': 'CD33'
            },
            "P2": {
                'FITC-A': 'CD15',
                'PE-A': 'CD22',
                'APC-A': 'CD19'
            },
            "P3": {
                'FITC-A': 'CD36',
                'PE-A': 'CD14',
                'APC-A': 'CD11B'
            },
            "P4": {
                'FITC-A': 'CD2',
                'PE-A': 'CD133',
                'APC-A': 'CD33'
            }
        }
    
        channels = [
            'FSC-A', 'FSC-H',
            'SSC-A', 'SSC-H',
            'FITC-A', 'PE-A',
            'PerCP-A', 'PC7-A',
            'APC-A', 'APC-H7-A',
            'Horizon V450-A', 'Horizon V500-A'
        ]
    
        markers = []
        for ch in channels:
            if ch in common_map:
                markers.append(common_map[ch])
            elif ch in tube_map.get(tube_name, {}):
                markers.append(tube_map[tube_name][ch])
            else:
                markers.append(ch)  # fallback
    
        return markers

    @staticmethod
    def plot_org_marginals_by_class(
        X,
        Y,
        marker_names,
        tube_name="P1"
    ):

    
        print(f"\n✅ Plotting ORIGINAL marginals for {tube_name}")
    
        n_markers = len(marker_names)
    
        n_cols = 3
        n_rows = int(np.ceil(n_markers / n_cols))
    
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 2.5 * n_rows),
            squeeze=False
        )
    
        for m in range(n_markers):
    
            row = m // n_cols
            col = m % n_cols
            ax = axes[row, col]
    
            vals_0 = X[Y == 0, m]
            vals_1 = X[Y == 1, m]
    
            # ✅ percentile-based limits (robust)
            vmin, vmax = np.percentile(
                np.concatenate([vals_0, vals_1]),
                [1, 99]
            )
    
            xs = np.linspace(vmin, vmax, 300)
    
            # KDE for class 0
            if len(vals_0) > 10:
                kde0 = gaussian_kde(vals_0)
                ax.plot(xs, kde0(xs), color="#4C72B0", lw=2, label="NBM")
    
            # KDE for class 1
            if len(vals_1) > 10:
                kde1 = gaussian_kde(vals_1)
                ax.plot(xs, kde1(xs), color="#DD8452", lw=2, label="AML")
    
            ax.set_title(marker_names[m], fontsize=9, fontweight="bold")
            ax.set_xlim(vmin, vmax)
            ax.tick_params(labelsize=7)
    
        # remove empty axes
        for idx in range(n_markers, n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols, idx % n_cols])
    
        # legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=2,
            fontsize=9
        )
    
        fig.suptitle(
            f"Marker marginal distributions (original data) – {tube_name}",
            fontsize=12,
            fontweight="bold"
        )
    
        plt.subplots_adjust(
            top=0.9,
            wspace=0.25,
            hspace=0.25
        )
    
        plt.show()
    @staticmethod
    def plot_lot_marginals_all_markers(
        Xtr, Ytr, Xte, Yte,
        tube_name="P1",
        K_VALUES=None,
        marker_names=None,
        palette="clinical",
        colors=None,
        cmap=None,
        save_path="lot_marginals_all_markers_final.pdf",
    ):
        if marker_names is None:
            marker_names = visualization_utils.get_marker_names_from_tube(tube_name)
        if K_VALUES is None:
            K_VALUES = np.arange(-1.5, 1.75, 0.75)

        print(f"\n✅ Marginal plots for: {tube_name} (palette: {palette})")

        mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)
        proj_train = (Xtr - mu_all) @ v_hat
        sigma_parallel = np.std(proj_train, ddof=1)

        n_markers = len(marker_names)
        target_indices = [idx for idx in [0, 2, 6, 8] if idx < n_markers]
        if not target_indices:
            target_indices = list(range(min(4, n_markers)))

        n_cols = len(target_indices)
        n_rows = 1

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            squeeze=False
        )

        from matplotlib.colors import LinearSegmentedColormap

        # Resolve color palette: ensure sigma = -1.5 (Normal) is Blue and sigma = +1.5 (AML) is Red
        if colors is not None:
            pass
        elif cmap is not None:
            colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            colors = [colormap(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]
        elif palette == "clinical":
            # High-contrast 5-color clinical divergent palette (Deep Navy -> Sky Blue -> Slate -> Coral -> Crimson)
            if len(K_VALUES) == 5:
                colors = ["#08519c", "#4292c6", "#525252", "#ef6548", "#b30000"]
            else:
                c_map = LinearSegmentedColormap.from_list(
                    "clinical_div", ["#08519c", "#4292c6", "#525252", "#ef6548", "#b30000"]
                )
                colors = [c_map(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]
        elif palette in {"nature_rdbu", "rdbu"}:
            if len(K_VALUES) == 5:
                colors = ["#2166ac", "#67a9cf", "#4d4d4d", "#f46d43", "#b2182b"]
            else:
                c_map = LinearSegmentedColormap.from_list(
                    "rdbu_div", ["#2166ac", "#67a9cf", "#4d4d4d", "#f46d43", "#b2182b"]
                )
                colors = [c_map(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]
        elif palette == "coolwarm":
            colormap = plt.get_cmap("coolwarm")
            colors = [colormap(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]
        elif palette == "set1" or palette == "corrected":
            # Set1 Blue (#377eb8, Normal) -> Set1 Red (#e41a1c, AML), directly matching 1D projection figure
            c_map = LinearSegmentedColormap.from_list("set1_blue_red", ["#377eb8", "#e41a1c"])
            colors = [c_map(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]
        else:
            # Fallback to Set1 Blue -> Red
            c_map = LinearSegmentedColormap.from_list("blue_red_fallback", ["#377eb8", "#e41a1c"])
            colors = [c_map(i / (len(K_VALUES) - 1)) for i in range(len(K_VALUES))]

        for m, ii in enumerate(target_indices):
            row = m // n_cols
            col = m % n_cols
            ax = axes[row, col]

            # -------- GLOBAL LIMIT for KDE ----------
            all_vals = []
            for k in K_VALUES:
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(x_k, n_markers)
                all_vals.append(V[ii, :])

            all_vals = np.concatenate(all_vals)
            vmin, vmax = np.percentile(all_vals / 1e5, [0, 99])
            xs = np.linspace(vmin, vmax, 300)
            # ---------------------------------------

            for k_idx, k in enumerate(K_VALUES):
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(x_k, n_markers)
                vals = V[ii, :] / 1e5

                if len(vals) > 10:
                    kde = gaussian_kde(vals)
                    lbl = rf"$\sigma={k}$"
                    if k == K_VALUES[0]:
                        lbl += " (NBM)"
                    elif k == K_VALUES[-1]:
                        lbl += " (AML)"
                    ax.plot(
                        xs,
                        kde(xs),
                        color=colors[k_idx],
                        lw=2.0,
                        label=lbl,
                    )

            ax.set_xlim(vmin, vmax)
            ax.set_title(marker_names[ii], fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.set_xlabel(r"LOT coordinate $(\times 10^5)$", fontsize=8)
            if col == 0:
                ax.set_ylabel("Density", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

        # Remove empty panels
        for idx in range(n_markers, n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols, idx % n_cols])

        # Legend
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.82),
            ncol=len(K_VALUES),
            fontsize=8.5,
            frameon=True,
            framealpha=0.9,
        )

        fig.suptitle(
            f"Marginal distributions across σ (Tube {tube_name})",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.64,
            bottom=0.18,
            wspace=0.25,
        )
        if save_path:
            fig.savefig(
                save_path,
                dpi=600,
                bbox_inches="tight",
            )

        plt.show()
        return fig, axes

    @staticmethod
    # def plot_composite_figure4(
    #     Xtr,
    #     Ytr,
    #     Xte,
    #     Yte,
    #     tube_name="P1",
    #     K_VALUES=np.array([-1.5, -0.75, 0.0, 0.75, 1.5]),
    #     marker_names=None,
    #     pairs=((0, 2), (6, 7)),
    #     target_markers=(0, 2, 6, 7),
    #     palette="clinical",
    #     save_png="figure4_nature_combined.png",
    #     save_pdf="figure4_nature_combined.pdf",
    #     show=True,
    # ):
    #     """
    #     Generate publication-grade composite Figure 4 (Nature Medicine style).
        
    #     Combines:
    #       Panel (a):
    #         - 1D Train & Test projection histograms along discriminant direction
    #         - Trapezoid callout connectors
    #         - 2x5 grid of pairwise LOT evolution scatter plots with column background tints
    #       Panel (b):
    #         - Marginal KDE distributions for selected markers across sigma values
        
    #     Colors transition semantically from Normal/NBM (Blue) to AML (Crimson Red).
    #     """
    #     from matplotlib.patches import Polygon, Rectangle
    #     from matplotlib.ticker import MaxNLocator
    #     from scipy.stats import gaussian_kde

    #     n_markers = len(marker_names) if marker_names is not None else 8
    #     display_marker_names = list(marker_names) if marker_names is not None else [f"Marker {i}" for i in range(n_markers)]
    #     if len(display_marker_names) > 6 and display_marker_names[6] in ("CD117", "PerCP-A", "Marker 6"):
    #         display_marker_names[6] = "CD34"
    #     if len(display_marker_names) > 7 and display_marker_names[7] in ("APC-A", "Marker 7"):
    #         display_marker_names[7] = "CD33"

    #     # 1. Discriminant Direction & Projections
    #     mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(Xtr, Ytr)
    #     proj_train = (Xtr - mu_all) @ v_hat
    #     proj_test = (Xte - mu_all) @ v_hat
    #     sigma_parallel = np.std(proj_train, ddof=1)
    #     proj_train /= sigma_parallel
    #     proj_test /= sigma_parallel

    #     # Palette selection
    #     if palette == "clinical":
    #         palette_curves = ["#08519c", "#3182bd", "#525252", "#e6550d", "#b30000"]
    #         palette_dots = ["#1a5fb4", "#3584e4", "#475569", "#d9534f", "#c01c28"]
    #         col_bg_tints = ["#edf4fb", "#f6f9fd", "#ffffff", "#fdf7f6", "#fbeeed"]
    #     elif palette == "nature_rdbu":
    #         palette_curves = ["#2166ac", "#67a9cf", "#4d4d4d", "#ef8a62", "#b2182b"]
    #         palette_dots = ["#2b7bba", "#529dc7", "#525252", "#e27952", "#c42236"]
    #         col_bg_tints = ["#edf3f9", "#f5f8fc", "#ffffff", "#fdf6f3", "#fceedd"]
    #     else:  # set1
    #         palette_curves = ["#377eb8", "#6baed6", "#737373", "#fb6a4a", "#e41a1c"]
    #         palette_dots = ["#377eb8", "#5aa0d8", "#666666", "#f05b43", "#e41a1c"]
    #         col_bg_tints = ["#f0f5fa", "#f7fafe", "#ffffff", "#fef6f4", "#fdf0ef"]

    #     fig = plt.figure(figsize=(10.5, 13.0), dpi=300)

    #     # =========================================================================
    #     # PANEL A: LOT geometry along discriminant axis
    #     # =========================================================================
    #     fig.text(0.50, 0.972, "LOT geometry along discriminant axis", ha="center", va="center", fontsize=11, fontweight="bold")

    #     ax_train = fig.add_axes([0.10, 0.865, 0.84, 0.055])
    #     ax_test = fig.add_axes([0.10, 0.790, 0.84, 0.055], sharex=ax_train)

    #     # Train histogram
    #     for cls, col, lbl in [(0, "#4C72B0", "NBM"), (1, "#DD8452", "AML_Dx")]:
    #         p = proj_train[Ytr == cls]
    #         ax_train.hist(
    #             p, bins=12, range=(-2.2, 2.2), density=True,
    #             color=col, alpha=0.55, edgecolor="black", linewidth=0.6, label=lbl
    #         )
    #     ax_train.set_title("Train projection", fontsize=8.5, fontweight="bold", pad=3)
    #     ax_train.set_ylabel("Density", fontsize=7.5)
    #     ax_train.set_xlim(-2.3, 2.3)
    #     ax_train.tick_params(labelsize=7)
    #     ax_train.xaxis.set_ticklabels([])
    #     ax_train.grid(True, linestyle="--", alpha=0.25)

    #     # Test histogram
    #     for cls, col, lbl in [(0, "#4C72B0", "NBM"), (1, "#DD8452", "AML_Dx")]:
    #         p = proj_test[Yte == cls]
    #         ax_test.hist(
    #             p, bins=12, range=(-2.2, 2.2), density=True,
    #             color=col, alpha=0.55, edgecolor="black", linewidth=0.6, label=lbl
    #         )
    #     ax_test.set_title("Test projection", fontsize=8.5, fontweight="bold", pad=3)
    #     ax_test.set_ylabel("Density", fontsize=7.5)
    #     ax_test.set_xlabel(r"Projection along $\hat{v}$ ($\sigma_\parallel$)", fontsize=7.5, labelpad=12)
    #     ax_test.set_xlim(-2.3, 2.3)
    #     ax_test.set_xticks(K_VALUES)
    #     ax_test.set_xticklabels([f"{k}" for k in K_VALUES], fontsize=7, fontweight="bold")
    #     ax_test.tick_params(labelsize=7)
    #     ax_test.grid(True, linestyle="--", alpha=0.25)

    #     handles, labels = ax_train.get_legend_handles_labels()
    #     fig.legend(
    #         handles, labels,
    #         loc="center", bbox_to_anchor=(0.50, 0.952),
    #         ncol=2, fontsize=8, frameon=False
    #     )

    #     # Highlight boxes under ticks of test projection
    #     for k in K_VALUES:
    #         p_k = ax_test.transData.transform([k, 0])
    #         inv = fig.transFigure.inverted()
    #         fk = inv.transform(p_k)
    #         tag_box = Rectangle((fk[0] - 0.016, fk[1] - 0.016), 0.032, 0.011, facecolor="#eeeeee", edgecolor="#888888", linewidth=0.5, transform=fig.transFigure, zorder=5)
    #         fig.patches.append(tag_box)
    #         fig.text(fk[0], fk[1] - 0.0105, f"{k}", fontsize=5.8, fontweight="bold", ha="center", va="center", zorder=6)

    #     # Title for pairwise grid
    #     fig.text(0.50, 0.725, "Pairwise LOT evolution across σ", ha="center", va="center", fontsize=9.5, fontweight="bold")

    #     pw_left = 0.078
    #     col_w = 0.168
    #     gap_w = 0.012
    #     row_h = 0.128
    #     row1_y = 0.575
    #     row2_y = 0.428

    #     # Inner box around pairwise scatter plots
    #     pw_box_x = pw_left - 0.014
    #     pw_box_w = 5 * col_w + 4 * gap_w + 0.028
    #     pw_box_y = row2_y - 0.022
    #     pw_box_h = (row1_y + row_h + 0.016) - pw_box_y

    #     pw_outer_box = Rectangle(
    #         (pw_box_x, pw_box_y), pw_box_w, pw_box_h,
    #         fill=False, edgecolor="#333333", linewidth=1.0,
    #         transform=fig.transFigure, zorder=1
    #     )
    #     fig.patches.append(pw_outer_box)

    #     axes_pairwise = [[], []]

    #     # Precompute limits
    #     row_limits = []
    #     for (i, j) in pairs:
    #         all_x, all_y = [], []
    #         for k in K_VALUES:
    #             x_k = mu_all + k * sigma_parallel * v_hat
    #             V = visualization_utils.lot_vec_to_V(x_k, n_markers)
    #             all_x.append(V[i, :])
    #             all_y.append(V[j, :])
    #         all_x = np.concatenate(all_x) / 1e5
    #         all_y = np.concatenate(all_y) / 1e5
    #         x_min, x_max = np.percentile(all_x, [1, 99])
    #         y_min, y_max = np.percentile(all_y, [1, 99])
    #         x_lim = max(abs(x_min), abs(x_max)) * 1.05
    #         y_lim = max(abs(y_min), abs(y_max)) * 1.05
    #         row_limits.append((x_lim, y_lim))

    #     for col_idx, k in enumerate(K_VALUES):
    #         cur_x = pw_left + col_idx * (col_w + gap_w)
    #         x_k = mu_all + k * sigma_parallel * v_hat
    #         V = visualization_utils.lot_vec_to_V(x_k, n_markers)

    #         # Row 1
    #         ax_r1 = fig.add_axes([cur_x, row1_y, col_w, row_h])
    #         ax_r1.set_facecolor(col_bg_tints[col_idx])
    #         x1 = V[pairs[0][0], :] / 1e5
    #         y1 = V[pairs[0][1], :] / 1e5
    #         ax_r1.scatter(x1, y1, s=6, color=palette_dots[col_idx], alpha=0.55, edgecolors="none")
    #         ax_r1.set_xlim(0, row_limits[0][0])
    #         ax_r1.set_ylim(0, row_limits[0][1])
    #         ax_r1.set_title(rf"$\sigma = {k}$", fontsize=8.5, fontweight="bold", pad=3)
    #         ax_r1.tick_params(labelsize=6, pad=1)
    #         ax_r1.set_xticks([0.8, 1.6, 2.4])
    #         if col_idx == 0:
    #             ax_r1.set_ylabel(display_marker_names[pairs[0][1]], fontsize=7.5, fontweight="bold")
    #             ax_r1.set_yticks([0.8, 1.6, 2.4])
    #         else:
    #             ax_r1.yaxis.set_ticklabels([])
    #         ax_r1.set_xlabel(display_marker_names[pairs[0][0]], fontsize=7.5, fontweight="bold", labelpad=1)
    #         axes_pairwise[0].append(ax_r1)

    #         # Row 2
    #         ax_r2 = fig.add_axes([cur_x, row2_y, col_w, row_h])
    #         ax_r2.set_facecolor(col_bg_tints[col_idx])
    #         x2 = V[pairs[1][0], :] / 1e5
    #         y2 = V[pairs[1][1], :] / 1e5
    #         ax_r2.scatter(x2, y2, s=6, color=palette_dots[col_idx], alpha=0.55, edgecolors="none")
    #         ax_r2.set_xlim(0, row_limits[1][0])
    #         ax_r2.set_ylim(0, row_limits[1][1])
    #         ax_r2.tick_params(labelsize=6, pad=1)
    #         ax_r2.set_xticks([0.03, 0.07, 0.11])
    #         if col_idx == 0:
    #             ax_r2.set_ylabel(display_marker_names[pairs[1][1]], fontsize=7.5, fontweight="bold")
    #             ax_r2.set_yticks([0.05, 0.10, 0.15])
    #         else:
    #             ax_r2.yaxis.set_ticklabels([])
    #         ax_r2.set_xlabel(display_marker_names[pairs[1][0]], fontsize=7.5, fontweight="bold", labelpad=1)
    #         axes_pairwise[1].append(ax_r2)

    #     # Trapezoid connector
    #     p_k_left = ax_test.transData.transform([-1.5, 0])
    #     p_k_right = ax_test.transData.transform([1.5, 0])
    #     inv = fig.transFigure.inverted()
    #     fig_k_left = inv.transform(p_k_left)
    #     fig_k_right = inv.transform(p_k_right)

    #     poly_points = [
    #         (fig_k_left[0] - 0.016, fig_k_left[1] - 0.016),
    #         (pw_box_x, pw_box_y + pw_box_h),
    #         (pw_box_x + pw_box_w, pw_box_y + pw_box_h),
    #         (fig_k_right[0] + 0.016, fig_k_right[1] - 0.016)
    #     ]
    #     poly = Polygon(
    #         poly_points, closed=True,
    #         facecolor="#334155", alpha=0.03,
    #         edgecolor="#222222", linestyle="-", linewidth=0.9,
    #         transform=fig.transFigure, zorder=2
    #     )
    #     fig.patches.append(poly)

    #     # =========================================================================
    #     # PANEL B: Marginal distributions across σ (4 markers)
    #     # =========================================================================
    #     fig.text(0.50, 0.355, "Marginal distributions across σ", ha="center", va="center", fontsize=11, fontweight="bold")

    #     mb_left = 0.075
    #     mb_w = 0.205
    #     mb_gap = 0.025
    #     mb_y = 0.065
    #     mb_h = 0.235

    #     axes_marginals = []
    #     for m_idx, m_chan in enumerate(target_markers):
    #         cur_x = mb_left + m_idx * (mb_w + mb_gap)
    #         ax_m = fig.add_axes([cur_x, mb_y, mb_w, mb_h])
    #         axes_marginals.append(ax_m)

    #         all_vals = []
    #         for k in K_VALUES:
    #             x_k = mu_all + k * sigma_parallel * v_hat
    #             V = visualization_utils.lot_vec_to_V(x_k, n_markers)
    #             all_vals.append(V[m_chan, :])
    #         all_vals = np.concatenate(all_vals)
    #         vmin, vmax = np.percentile(all_vals / 1e5, [0, 99])
    #         xs = np.linspace(vmin, vmax, 300)

    #         for k_idx, k in enumerate(K_VALUES):
    #             x_k = mu_all + k * sigma_parallel * v_hat
    #             V = visualization_utils.lot_vec_to_V(x_k, n_markers)
    #             vals = V[m_chan, :] / 1e5

    #             if len(vals) > 10:
    #                 kde = gaussian_kde(vals)
    #                 lbl = rf"$\sigma = {k}$"
    #                 if k == K_VALUES[0]:
    #                     lbl += " (NBM)"
    #                 elif k == K_VALUES[-1]:
    #                     lbl += " (AML)"
    #                 ax_m.plot(
    #                     xs, kde(xs),
    #                     color=palette_curves[k_idx],
    #                     lw=2.2,
    #                     label=lbl
    #                 )

    #         ax_m.set_xlim(vmin, vmax)
    #         ax_m.set_title(display_marker_names[m_chan], fontsize=9.5, fontweight="bold", pad=4)
    #         ax_m.tick_params(labelsize=7)
    #         ax_m.xaxis.set_major_locator(MaxNLocator(4))
    #         if m_idx == 0:
    #             ax_m.set_ylabel("Density", fontsize=8)
    #         ax_m.grid(True, linestyle="--", alpha=0.25)

    #     handles_m, labels_m = axes_marginals[0].get_legend_handles_labels()
    #     fig.legend(
    #         handles_m, labels_m,
    #         loc="center", bbox_to_anchor=(0.50, 0.330),
    #         ncol=len(K_VALUES), fontsize=8, frameon=True, framealpha=0.95
    #     )

    #     # Outer border boxes
    #     box_a = Rectangle((0.022, 0.400), 0.956, 0.588, fill=False, edgecolor="#222222", linewidth=1.2, transform=fig.transFigure, zorder=1)
    #     fig.patches.append(box_a)
    #     fig.text(0.035, 0.976, "a)", fontsize=14, fontweight="bold", ha="left", va="top")

    #     box_b = Rectangle((0.022, 0.025), 0.956, 0.355, fill=False, edgecolor="#222222", linewidth=1.2, transform=fig.transFigure, zorder=1)
    #     fig.patches.append(box_b)
    #     fig.text(0.035, 0.368, "b)", fontsize=14, fontweight="bold", ha="left", va="top")

    #     if save_png:
    #         fig.savefig(save_png, dpi=300, bbox_inches="tight")
    #     if save_pdf:
    #         fig.savefig(save_pdf, dpi=600, bbox_inches="tight")

    #     if show:
    #         plt.show()
    #     else:
    #         plt.close(fig)

    #     return fig

    def plot_composite_figure4(
        Xtr,
        Ytr,
        Xte,
        Yte,
        tube_name="P1",
        K_VALUES=np.array([-1.5, -0.75, 0.0, 0.75, 1.5]),
        marker_names=None,
        pairs=((0, 2), (6, 7)),
        target_markers=(0, 2, 6, 7),
        palette="clinical",
        save_png="figure4_nature_combined.png",
        save_pdf="figure4_nature_combined.pdf",
        show=True,
    ):
        """
        Composite Figure 4

        Panel (a):
            Train/Test projections + LOT geometry

        Panel (b):
            Sin(x)

        Panel (c):
            Marginal marker distributions
        """

        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.patches import Polygon, Rectangle
        from matplotlib.ticker import MaxNLocator
        from scipy.stats import gaussian_kde

        n_markers = len(marker_names) if marker_names is not None else 8

        display_marker_names = (
            list(marker_names)
            if marker_names is not None
            else [f"Marker {i}" for i in range(n_markers)]
        )

        if len(display_marker_names) > 6 and display_marker_names[6] in (
            "CD117",
            "PerCP-A",
            "Marker 6",
        ):
            display_marker_names[6] = "CD34"

        if len(display_marker_names) > 7 and display_marker_names[7] in (
            "APC-A",
            "Marker 7",
        ):
            display_marker_names[7] = "CD33"

        # ============================================================
        # Discriminant direction
        # ============================================================

        mu0, v_hat, mu_all = visualization_utils.compute_discriminant_direction(
            Xtr, Ytr
        )

        proj_train = (Xtr - mu_all) @ v_hat
        proj_test = (Xte - mu_all) @ v_hat

        sigma_parallel = np.std(proj_train, ddof=1)

        proj_train /= sigma_parallel
        proj_test /= sigma_parallel

        # ============================================================
        # Colors
        # ============================================================

        if palette == "clinical":
            palette_curves = [
                "#08519c",
                "#3182bd",
                "#525252",
                "#e6550d",
                "#b30000",
            ]

            palette_dots = [
                "#1a5fb4",
                "#3584e4",
                "#475569",
                "#d9534f",
                "#c01c28",
            ]

            col_bg_tints = [
                "#edf4fb",
                "#f6f9fd",
                "#ffffff",
                "#fdf7f6",
                "#fbeeed",
            ]

        elif palette == "nature_rdbu":
            palette_curves = [
                "#2166ac",
                "#67a9cf",
                "#4d4d4d",
                "#ef8a62",
                "#b2182b",
            ]

            palette_dots = [
                "#2b7bba",
                "#529dc7",
                "#525252",
                "#e27952",
                "#c42236",
            ]

            col_bg_tints = [
                "#edf3f9",
                "#f5f8fc",
                "#ffffff",
                "#fdf6f3",
                "#fceedd",
            ]

        else:
            palette_curves = [
                "#377eb8",
                "#6baed6",
                "#737373",
                "#fb6a4a",
                "#e41a1c",
            ]

            palette_dots = [
                "#377eb8",
                "#5aa0d8",
                "#666666",
                "#f05b43",
                "#e41a1c",
            ]

            col_bg_tints = [
                "#f0f5fa",
                "#f7fafe",
                "#ffffff",
                "#fef6f4",
                "#fdf0ef",
            ]

        # ============================================================
        # Figure
        # ============================================================

        fig = plt.figure(figsize=(11, 14.5), dpi=300)

        # ============================================================
        # PANEL A TITLE
        # ============================================================

        fig.text(
            0.50,
            0.972,
            "LOT geometry along discriminant axis",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        # ============================================================
        # PROJECTIONS (larger vertical, smaller horizontal)
        # ============================================================

        proj_x = 0.12
        proj_w = 0.76
        proj_h = 0.075

        ax_train = fig.add_axes(
            [proj_x, 0.875, proj_w, proj_h]
        )

        ax_test = fig.add_axes(
            [proj_x, 0.785, proj_w, proj_h],
            sharex=ax_train,
        )

        # ------------------------------------------------------------
        # Train projection
        # ------------------------------------------------------------

        for cls, col, lbl in [
            (0, "#4C72B0", "NBM"),
            (1, "#DD8452", "AML_Dx"),
        ]:
            p = proj_train[Ytr == cls]

            ax_train.hist(
                p,
                bins=12,
                range=(-2.2, 2.2),
                density=True,
                color=col,
                alpha=0.55,
                edgecolor="black",
                linewidth=0.6,
                label=lbl,
            )

        ax_train.set_title(
            "Train projection",
            fontsize=9,
            fontweight="bold",
            pad=6,
        )

        ax_train.set_ylabel("Density", fontsize=8)

        ax_train.set_xlim(-2.3, 2.3)

        ax_train.grid(
            True,
            linestyle="--",
            alpha=0.25,
        )

        ax_train.tick_params(labelsize=7)

        ax_train.xaxis.set_ticklabels([])

        # ------------------------------------------------------------
        # Test projection
        # ------------------------------------------------------------

        for cls, col, lbl in [
            (0, "#4C72B0", "NBM"),
            (1, "#DD8452", "AML_Dx"),
        ]:
            p = proj_test[Yte == cls]

            ax_test.hist(
                p,
                bins=12,
                range=(-2.2, 2.2),
                density=True,
                color=col,
                alpha=0.55,
                edgecolor="black",
                linewidth=0.6,
            )

        ax_test.set_title(
            "Test projection",
            fontsize=9,
            fontweight="bold",
            pad=6,
        )

        ax_test.set_ylabel(
            "Density",
            fontsize=8,
        )

        ax_test.set_xlabel(
            r"Projection along $\hat{v}$ ($\sigma_\parallel$)",
            fontsize=8,
            labelpad=6,
        )

        ax_test.set_xlim(-2.3, 2.3)

        ax_test.set_xticks(K_VALUES)

        ax_test.set_xticklabels(
            [f"{k}" for k in K_VALUES],
            fontsize=7,
            fontweight="bold",
        )

        ax_test.grid(
            True,
            linestyle="--",
            alpha=0.25,
        )

        ax_test.tick_params(labelsize=7)

        # legend
        handles, labels = ax_train.get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.50, 0.955),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        # ============================================================
        # Tick callout boxes
        # ============================================================

        for k in K_VALUES:

            p_k = ax_test.transData.transform([k, 0])

            inv = fig.transFigure.inverted()

            fk = inv.transform(p_k)

            tag_box = Rectangle(
                (fk[0] - 0.016, fk[1] - 0.016),
                0.032,
                0.011,
                facecolor="#eeeeee",
                edgecolor="#888888",
                linewidth=0.5,
                transform=fig.transFigure,
                zorder=5,
            )

            fig.patches.append(tag_box)

            fig.text(
                fk[0],
                fk[1] - 0.0105,
                f"{k}",
                fontsize=5.8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )

        # ============================================================
        # Pairwise title
        # ============================================================

        fig.text(
            0.50,
            0.710,
            "Pairwise LOT evolution across σ",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

        # ============================================================
        # KEEP YOUR EXISTING PAIRWISE SCATTER SECTION HERE
        # ============================================================
        pw_left = 0.078
        col_w = 0.168
        gap_w = 0.012
        row_h = 0.128
        row1_y = 0.575
        row2_y = 0.428

        # Inner box around pairwise scatter plots
        pw_box_x = pw_left - 0.014
        pw_box_w = 5 * col_w + 4 * gap_w + 0.028
        pw_box_y = row2_y - 0.022
        pw_box_h = (row1_y + row_h + 0.016) - pw_box_y

        pw_outer_box = Rectangle(
            (pw_box_x, pw_box_y), pw_box_w, pw_box_h,
            fill=False, edgecolor="#333333", linewidth=1.0,
            transform=fig.transFigure, zorder=1
        )
        fig.patches.append(pw_outer_box)

        axes_pairwise = [[], []]

        # Precompute limits
        row_limits = []
        for (i, j) in pairs:
            all_x, all_y = [], []
            for k in K_VALUES:
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(x_k, n_markers)
                all_x.append(V[i, :])
                all_y.append(V[j, :])
            all_x = np.concatenate(all_x) / 1e5
            all_y = np.concatenate(all_y) / 1e5
            x_min, x_max = np.percentile(all_x, [1, 99])
            y_min, y_max = np.percentile(all_y, [1, 99])
            x_lim = max(abs(x_min), abs(x_max)) * 1.05
            y_lim = max(abs(y_min), abs(y_max)) * 1.05
            row_limits.append((x_lim, y_lim))

        for col_idx, k in enumerate(K_VALUES):
            cur_x = pw_left + col_idx * (col_w + gap_w)
            x_k = mu_all + k * sigma_parallel * v_hat
            V = visualization_utils.lot_vec_to_V(x_k, n_markers)

            # Row 1
            ax_r1 = fig.add_axes([cur_x, row1_y, col_w, row_h])
            ax_r1.set_facecolor(col_bg_tints[col_idx])
            x1 = V[pairs[0][0], :] / 1e5
            y1 = V[pairs[0][1], :] / 1e5
            ax_r1.scatter(x1, y1, s=6, color=palette_dots[col_idx], alpha=0.55, edgecolors="none")
            ax_r1.set_xlim(0, row_limits[0][0])
            ax_r1.set_ylim(0, row_limits[0][1])
            ax_r1.set_title(rf"$\sigma = {k}$", fontsize=8.5, fontweight="bold", pad=3)
            ax_r1.tick_params(labelsize=6, pad=1)
            ax_r1.set_xticks([0.8, 1.6, 2.4])
            if col_idx == 0:
                ax_r1.set_ylabel(display_marker_names[pairs[0][1]], fontsize=7.5, fontweight="bold")
                ax_r1.set_yticks([0.8, 1.6, 2.4])
            else:
                ax_r1.yaxis.set_ticklabels([])
            ax_r1.set_xlabel(display_marker_names[pairs[0][0]], fontsize=7.5, fontweight="bold", labelpad=1)
            axes_pairwise[0].append(ax_r1)

            # Row 2
            ax_r2 = fig.add_axes([cur_x, row2_y, col_w, row_h])
            ax_r2.set_facecolor(col_bg_tints[col_idx])
            x2 = V[pairs[1][0], :] / 1e5
            y2 = V[pairs[1][1], :] / 1e5
            ax_r2.scatter(x2, y2, s=6, color=palette_dots[col_idx], alpha=0.55, edgecolors="none")
            ax_r2.set_xlim(0, row_limits[1][0])
            ax_r2.set_ylim(0, row_limits[1][1])
            ax_r2.tick_params(labelsize=6, pad=1)
            ax_r2.set_xticks([0.03, 0.07, 0.11])
            if col_idx == 0:
                ax_r2.set_ylabel(display_marker_names[pairs[1][1]], fontsize=7.5, fontweight="bold")
                ax_r2.set_yticks([0.05, 0.10, 0.15])
            else:
                ax_r2.yaxis.set_ticklabels([])
            ax_r2.set_xlabel(display_marker_names[pairs[1][0]], fontsize=7.5, fontweight="bold", labelpad=1)
            axes_pairwise[1].append(ax_r2)

        # Trapezoid connector
        p_k_left = ax_test.transData.transform([-1.5, 0])
        p_k_right = ax_test.transData.transform([1.5, 0])
        inv = fig.transFigure.inverted()
        fig_k_left = inv.transform(p_k_left)
        fig_k_right = inv.transform(p_k_right)

        poly_points = [
            (fig_k_left[0] - 0.016, fig_k_left[1] - 0.016),
            (pw_box_x, pw_box_y + pw_box_h),
            (pw_box_x + pw_box_w, pw_box_y + pw_box_h),
            (fig_k_right[0] + 0.016, fig_k_right[1] - 0.016)
        ]
        poly = Polygon(
            poly_points, closed=True,
            facecolor="#334155", alpha=0.03,
            edgecolor="#222222", linestyle="-", linewidth=0.9,
            transform=fig.transFigure, zorder=2
        )
        fig.patches.append(poly)
        # ============================================================

        # ============================================================
        # PANEL B : SIN CURVE
        # ============================================================

        fig.text(
            0.50,
            0.385,
            "Reference sine function",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax_sin = fig.add_axes(
            [0.12, 0.275, 0.76, 0.085]
        )

        xs = np.linspace(
            0,
            2 * np.pi,
            500,
        )

        ys = np.sin(xs)

        ax_sin.plot(
            xs,
            ys,
            lw=2.5,
            color="#2166ac",
        )

        ax_sin.set_xlabel(
            "x",
            fontsize=8,
        )

        ax_sin.set_ylabel(
            "sin(x)",
            fontsize=8,
        )

        ax_sin.grid(
            True,
            linestyle="--",
            alpha=0.25,
        )

        ax_sin.tick_params(
            labelsize=7
        )

        # ============================================================
        # PANEL C
        # ============================================================

        fig.text(
            0.50,
            0.235,
            "Marginal distributions across σ",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        mb_left = 0.075
        mb_w = 0.205
        mb_gap = 0.025

        mb_y = 0.050
        mb_h = 0.145

        axes_marginals = []

        for m_idx, m_chan in enumerate(target_markers):

            cur_x = mb_left + m_idx * (mb_w + mb_gap)

            ax_m = fig.add_axes(
                [cur_x, mb_y, mb_w, mb_h]
            )

            axes_marginals.append(ax_m)

            all_vals = []

            for k in K_VALUES:
                x_k = mu_all + k * sigma_parallel * v_hat
                V = visualization_utils.lot_vec_to_V(
                    x_k,
                    n_markers,
                )
                all_vals.append(V[m_chan, :])

            all_vals = np.concatenate(all_vals)

            vmin, vmax = np.percentile(
                all_vals / 1e5,
                [0, 99],
            )

            grid = np.linspace(vmin, vmax, 300)

            for k_idx, k in enumerate(K_VALUES):

                x_k = mu_all + k * sigma_parallel * v_hat

                V = visualization_utils.lot_vec_to_V(
                    x_k,
                    n_markers,
                )

                vals = V[m_chan, :] / 1e5

                if len(vals) > 10:

                    kde = gaussian_kde(vals)

                    lbl = rf"$\sigma = {k}$"

                    if k == K_VALUES[0]:
                        lbl    
                    elif k == K_VALUES[-1]:
                        lbl += " (AML)"

                    ax_m.plot(
                        grid,
                        kde(grid),
                        color=palette_curves[k_idx],
                        lw=2.2,
                        label=lbl,
                    )

            ax_m.set_xlim(vmin, vmax)

            ax_m.set_title(
                display_marker_names[m_chan],
                fontsize=9,
                fontweight="bold",
                pad=4,
            )

            if m_idx == 0:
                ax_m.set_ylabel(
                    "Density",
                    fontsize=8,
                )

            ax_m.tick_params(
                labelsize=7
            )

            ax_m.xaxis.set_major_locator(
                MaxNLocator(4)
            )

            ax_m.grid(
                True,
                linestyle="--",
                alpha=0.25,
            )

        handles_m, labels_m = (
            axes_marginals[0].get_legend_handles_labels()
        )

        fig.legend(
            handles_m,
            labels_m,
            loc="center",
            bbox_to_anchor=(0.50, 0.215),
            ncol=len(K_VALUES),
            fontsize=7.5,
            frameon=True,
            framealpha=0.95,
            handlelength=2.2,
            columnspacing=1.5,
        )

        # ============================================================
        # PANEL BORDERS
        # ============================================================

        box_a = Rectangle(
            (0.022, 0.400),
            0.956,
            0.588,
            fill=False,
            edgecolor="#222222",
            linewidth=1.2,
            transform=fig.transFigure,
        )

        fig.patches.append(box_a)

        fig.text(
            0.035,
            0.976,
            "a)",
            fontsize=14,
            fontweight="bold",
        )

        box_b = Rectangle(
            (0.022, 0.250),
            0.956,
            0.125,
            fill=False,
            edgecolor="#222222",
            linewidth=1.2,
            transform=fig.transFigure,
        )

        fig.patches.append(box_b)

        fig.text(
            0.035,
            0.370,
            "b)",
            fontsize=14,
            fontweight="bold",
        )

        box_c = Rectangle(
            (0.022, 0.020),
            0.956,
            0.210,
            fill=False,
            edgecolor="#222222",
            linewidth=1.2,
            transform=fig.transFigure,
        )

        fig.patches.append(box_c)

        fig.text(
            0.035,
            0.225,
            "c)",
            fontsize=14,
            fontweight="bold",
        )

        if save_png:
            fig.savefig(
                save_png,
                dpi=300,
                bbox_inches="tight",
            )

        if save_pdf:
            fig.savefig(
                save_pdf,
                dpi=600,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig  


