import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import seaborn as sns
from BTrees.OOBTree import OOBTree
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})


def extract_embeddings(inputs: List, embed_model: torch.nn.Module, model_path: str, batch_size: int = 32):
    """
        Extract embeddings for all images in the inputs

        Args:
            inputs: List of sample data in format:
                [sample_id, shape_mask, mz_array, intensity_array]
            model_path: Path to saved model state dict
            batch_size: Batch size for inference
            embed_model: Embedding model of model_path

        Returns:
            outputs: [['sample id', m/z array (length: # ion images), embedding array (shape: # ion images, embedding dimension)], ...]
        """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = embed_model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    outputs = []
    for s_input in inputs:
        sample_id, shape_mask, mz_array, intensity_array = s_input
        n_images = len(intensity_array)
        all_embeddings = []
        for i in range(0, n_images, batch_size):
            batch_images = intensity_array[i: i + batch_size]
            batch_tensor = torch.FloatTensor(batch_images)
            batch_tensor = batch_tensor.to(device)
            batch_tensor = batch_tensor.unsqueeze(1)  # 在第1维度增加一个维度

            with torch.no_grad():
                embeddings = model(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
        all_embeddings = np.concatenate(all_embeddings, axis=0,
                                        dtype=np.float32)  # shape: (# ion images, embedding dimension)
        outputs.append([sample_id, mz_array, all_embeddings])
    return outputs


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def two_files_comparison(inputs_original, inputs_align, outputs_final, file1, file2):
    """

    :param inputs_original:
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :param outputs_final:
    :param file1:
    :param file2:
    :return:
    """
    for s_input, s_input_align, s_output in zip(inputs_original, inputs_align, outputs_final):
        if s_output[0] == file1:
            input_file1 = s_input
            input_a_file1 = s_input_align
            output_file1 = s_output
        elif s_output[0] == file2:
            input_file2 = s_input
            input_a_file2 = s_input_align
            output_file2 = s_output

    cos_sim = np.zeros((output_file1[2].shape[0], output_file2[2].shape[0]), dtype=np.float32)
    print(cos_sim.shape)

    for i in range(output_file1[2].shape[0]):
        for j in range(output_file2[2].shape[0]):
            embed_i = output_file1[2][i]
            embed_j = output_file2[2][j]
            cos_sim[i, j] = cosine_similarity(embed_i, embed_j)

    # top5_indices = np.argsort(cos_sim.flatten())[-5:][::-1]  # 降序排列
    top5_indices = np.argsort(cos_sim.flatten())[:5]
    # top5_indices = np.argsort(cos_sim.flatten())[-10005:-10000]
    top5_pairs = [(idx // cos_sim.shape[1], idx % cos_sim.shape[1]) for idx in top5_indices]

    # 绘制5对图像
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))  # 5行，2列

    for row, (i, j) in enumerate(top5_pairs):
        # 获取原始图像数据
        img1 = input_file1[3][i]  # file1 的第i个离子图像
        img2 = input_file2[3][j]  # file2 的第j个离子图像

        img1_prealign = input_a_file1[3][i].flatten()
        img2_prealign = input_a_file2[3][j].flatten()

        # 绘制file1的图像
        axes[row, 0].imshow(img1, cmap='viridis')
        axes[row, 0].set_title(
            f"{file1} - m/z={input_file1[2][i]:.4f}(Sim: {cos_sim[i, j]:.2f}, OriSim: {cosine_similarity(img1_prealign, img2_prealign):.2f})")
        axes[row, 0].axis('off')

        # 绘制file2的图像
        axes[row, 1].imshow(img2, cmap='viridis')
        axes[row, 1].set_title(f"{file2} - m/z={input_file2[2][j]:.4f}")
        axes[row, 1].axis('off')

    # plt.tight_layout()
    plt.show()

    return cos_sim, top5_pairs


# import umap error: LLVM ERROR: Symbol not found: __svml_sqrtf8
def multi_files_umap(outputs_final, file_list=None, output_path=None):
    """

    :param outputs_final:
    [['sample id', m/z array (length: # ion images), embedding array (shape: # ion images, embedding dim)], ...]
    :param file_list:
    :return:
    """
    import umap
    from mpl_toolkits.mplot3d import Axes3D
    temp_outputs = []
    if file_list is None:
        temp_outputs = outputs_final
    else:
        for s_output in outputs_final:
            if s_output[0] in file_list:
                temp_outputs.append(s_output)

    all_embeddings = []
    sample_ids = []
    mz_values = []

    for s_output in temp_outputs:
        sample_id = s_output[0]
        embeddings = s_output[2]  # shape: (n_ions, embed_dim)
        mz = s_output[1]  # m/z array

        all_embeddings.append(embeddings)
        sample_ids.extend([sample_id] * len(embeddings))
        mz_values.extend(mz)

    all_embeddings = np.vstack(all_embeddings)  # shape: (total_ions, embed_dim)

    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_results = reducer.fit_transform(all_embeddings)  # shape: (total_ions, 2)

    unique_samples = list(set(sample_ids))
    palette = sns.color_palette("husl", len(unique_samples))  # 使用seaborn调色板
    color_map = {sample: palette[i] for i, sample in enumerate(unique_samples)}
    colors = [color_map[sample] for sample in sample_ids]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        umap_results[:, 0],
        umap_results[:, 1],
        umap_results[:, 2],
        c=colors,
        alpha=0.6,
        s=10  # 点大小
    )

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=sample,
                   markerfacecolor=color_map[sample], markersize=10)
        for sample in unique_samples
    ]
    ax.legend(handles=legend_elements, title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return umap_results, sample_ids, mz_values  # 使用返回结果进行后续分析，file_list必须为空，否则会报错


def dbscan_clustering(umap_results, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(umap_results)

    cluster_labels = labels.tolist()
    return cluster_labels


def multi_files_pca_kmeans(outputs_final, file_list=None, n_clusters=8, output_path=None):
    if file_list is None:
        file_list = [i[0] for i in outputs_final]
        temp_outputs = outputs_final
    else:
        temp_outputs = [s_output for s_output in outputs_final if s_output[0] in file_list]

    # 2. 合并所有样本的嵌入向量和标签
    all_embeddings = []
    sample_ids = []
    mz_values = []

    for s_output in temp_outputs:
        sample_id = s_output[0]
        embeddings = s_output[2]  # shape: (n_ions, embed_dim)
        mz = s_output[1]  # m/z array

        all_embeddings.append(embeddings)
        sample_ids.extend([sample_id] * len(embeddings))
        mz_values.extend(mz)

    all_embeddings = np.vstack(all_embeddings)  # shape: (total_ions, embed_dim)

    # 3. 运行PCA降维（降至2D）
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(all_embeddings)  # shape: (total_ions, 2)

    # k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_results)  # shape: (total_ions,)

    # 4. Plots for samples and clusters
    plt.figure(figsize=(4, 7.5))

    # Plot 1:
    plt.subplot(2, 1, 1)
    unique_samples = list(set(sample_ids))
    palette = sns.color_palette("husl", len(file_list))
    color_map = {sample: palette[i] for i, sample in enumerate(file_list)}
    colors = [color_map[sample] for sample in sample_ids]
    # color_map = {
    #     'pos': 'blue',
    #     'neg': 'red'
    # }
    # colors = [
    #     color_map['pos'] if 'pos' in sample.lower() else
    #     color_map['neg'] if 'neg' in sample.lower() else
    #     'gray'  # fallback color if neither 'pos' nor 'neg' is found
    #     for sample in sample_ids
    # ]

    scatter = plt.scatter(
        pca_results[:, 0],
        pca_results[:, 1],
        c=colors,
        # s=7
        s=2
    )

    # 添加图例和标签
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=sample,
                   markerfacecolor=color_map[sample], markersize=7)
        for sample in file_list
    ]
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', label='Positive Mode',
    #                markerfacecolor=color_map['pos'], markersize=10),
    #     plt.Line2D([0], [0], marker='o', color='w', label='Negative Mode',
    #                markerfacecolor=color_map['neg'], markersize=10)
    # ]
    # plt.legend(handles=legend_elements, title="Sample", bbox_to_anchor=(0.76, 1), loc='upper left')  # aging samples
    plt.legend(handles=legend_elements, title="Sample", bbox_to_anchor=(0.69, 1), loc='upper left')  # multi resolution samples

    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2f})")

    # Plot 2:
    plt.subplot(2, 1, 2)
    cluster_palette = sns.color_palette("husl", n_clusters)
    cluster_colors = [cluster_palette[label] for label in cluster_labels]

    plt.scatter(
        pca_results[:, 0],
        pca_results[:, 1],
        c=cluster_colors,
        # s=7
        s=2
    )

    # 添加聚类中心的标记
    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8, marker='X')
    for i, center in enumerate(centers):
        plt.text(center[0], center[1], str(i), fontsize=12, ha='center', va='center', color='black')

    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2f})")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return pca_results, sample_ids, mz_values, cluster_labels


def clusters_visualization(original_inputs, sample_ids, cluster_labels, output_folder):
    """

    :param original_inputs:
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :param sample_ids: long lists of all ion images samples
    :param cluster_labels: long lists of all ion images clusters id
    :param output_folder: folders to save representative images
    :return:
    """
    unique_clusters = np.unique(cluster_labels)
    unique_samples = np.unique(sample_ids)

    cluster_sample_indices = {}
    for cluster in unique_clusters:
        cluster_sample_indices[cluster] = {}
        for sample in unique_samples:
            cluster_sample_indices[cluster][sample] = []
    for idx, (sample, cluster) in enumerate(zip(sample_ids, cluster_labels)):
        cluster_sample_indices[cluster][sample].append(idx)

    image_pos = []
    for sample_ind, s_input in enumerate(original_inputs):
        for image_data_ind in range(s_input[3].shape[0]):
            image_pos.append([sample_ind, image_data_ind])

    for cluster in unique_clusters:
        for sample in unique_samples:
            indices = cluster_sample_indices[cluster][sample]

            if not indices:
                continue

            cluster_images = []
            for i in indices:
                sample_ind, image_data_ind = image_pos[i]
                arr = original_inputs[sample_ind][3][image_data_ind]
                normalized = arr / np.max(arr)
                cluster_images.append(normalized)
            cluster_images = np.stack(cluster_images, axis=0)
            cluster_images = np.mean(cluster_images, axis=0)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            img = ax.imshow(cluster_images, cmap='magma')
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Cluster {cluster} - Sample {sample}")
            ax.axis('off')
            import os
            filename = os.path.join(output_folder, f"cluster_{cluster}_sample_{sample}.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()


def single_cluster_visualization(original_inputs, sample_ids, cluster_labels, display_cluster, output_path=None):
    """

    :param original_inputs:
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :param sample_ids: long lists of all ion images samples
    :param cluster_labels: long lists of all ion images clusters id
    :return:
    """
    unique_clusters = np.unique(cluster_labels)
    unique_samples = np.unique(sample_ids)

    cluster_sample_indices = {}
    for sample in unique_samples:
        cluster_sample_indices[sample] = []
    for idx, (sample, cluster) in enumerate(zip(sample_ids, cluster_labels)):
        if cluster == display_cluster:
            cluster_sample_indices[sample].append(idx)

    image_pos = []
    for sample_ind, s_input in enumerate(original_inputs):
        for image_data_ind in range(s_input[3].shape[0]):
            image_pos.append([sample_ind, image_data_ind])

    n_cols = 3
    valid_samples = [s for s in unique_samples if cluster_sample_indices[s]]
    n_plots = len(valid_samples)
    import math
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.5 * n_rows))
    axes = axes.flatten()  # 变成 1D 列表，方便索引

    for idx, sample in enumerate(valid_samples):
        indices = cluster_sample_indices[sample]

        cluster_images = []
        for i in indices:
            sample_ind, image_data_ind = image_pos[i]
            arr = original_inputs[sample_ind][3][image_data_ind]
            normalized = arr / np.max(arr)
            cluster_images.append(normalized)
        cluster_images = np.stack(cluster_images, axis=0)
        cluster_images = np.mean(cluster_images, axis=0)

        ax = axes[idx]
        img = ax.imshow(cluster_images, cmap='magma')
        # img = ax.imshow(np.rot90(cluster_images, k=1), cmap='magma')
        ax.set_title(f"Cluster {display_cluster} - {sample}")
        ax.axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def find_mz_groups(mz_values, sample_ids, cluster_labels, output_folder, mz_tolerance=10):
    """
    找出在指定 ppm 容差范围内的 m/z 值组，并打印它们的样本名和簇名

    :param mz_values: m/z 值列表
    :param sample_ids: 样本ID列表
    :param cluster_labels: 簇标签列表
    :param mz_tolerance: m/z 容差（ppm）
    :return: None
    """
    from collections import defaultdict
    import os
    import csv
    combined = sorted(zip(mz_values, sample_ids, cluster_labels), key=lambda x: x[0])

    groups = []
    current_group = []

    for i in range(len(combined)):
        if not current_group:
            current_group.append(combined[i])
            continue

        # 计算当前m/z与前一个m/z的相对差异（ppm）
        mz_current = combined[i][0]
        mz_prev = current_group[-1][0]
        ppm_diff = abs(mz_current - mz_prev) / mz_prev * 1e6

        if ppm_diff <= mz_tolerance:
            current_group.append(combined[i])
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [combined[i]]

    # 添加最后一组
    if len(current_group) > 1:
        groups.append(current_group)

    # 准备CSV数据
    all_sample_ids = sorted(list(set(sample_ids)))

    # Create a dictionary to store the cluster labels for each sample in each group
    output_data = []

    for group in groups:
        # Calculate mean m/z for the group
        mean_mz = np.mean([item[0] for item in group])

        # Create a dictionary for this group's clusters by sample ID
        group_dict = defaultdict(str)
        for mz, sample, cluster in group:
            group_dict[sample] = cluster

        # Create a row with mean m/z followed by cluster labels for each sample
        row = {'mean_mz': mean_mz}
        for sample in all_sample_ids:
            row[sample] = group_dict.get(sample, '')

        output_data.append(row)

    # Write to CSV file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'mz_groups_clusters.csv')

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['mean_mz'] + all_sample_ids
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in output_data:
            writer.writerow(row)

    print(f"Results written to {output_path}")


def visualize_mz_embedding(outputs_final, file_list, interest_mz, output_path=None):
    if file_list is None:
        temp_outputs = outputs_final
    else:
        temp_outputs = [s_output for s_output in outputs_final if s_output[0] in file_list]

    # 2. 合并所有样本的嵌入向量和标签
    all_embeddings = []
    sample_ids = []
    mz_values = []

    for s_output in temp_outputs:
        sample_id = s_output[0]
        embeddings = s_output[2]  # shape: (n_ions, embed_dim)
        mz = s_output[1]  # m/z array

        all_embeddings.append(embeddings)
        sample_ids.extend([sample_id] * len(embeddings))
        mz_values.extend(mz)

    all_embeddings = np.vstack(all_embeddings)  # shape: (total_ions, embed_dim)
    mz_values = np.array(mz_values)

    # 3. 运行PCA降维（降至2D）
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(all_embeddings)  # shape: (total_ions, 2)

    def is_in_interest(mz_val, interest_mz, ppm=10):
        tolerance = interest_mz * ppm / 1e6
        if abs(mz_val - interest_mz) <= tolerance:
            return True
        return False

    highlight_mask = np.array([is_in_interest(mz, interest_mz) for mz in mz_values])

    # Plotting
    plt.figure(figsize=(4, 3))

    # Plot all points in gray
    plt.scatter(
        pca_results[:, 0],
        pca_results[:, 1],
        c='lightgray',
        s=7,
        label='All Ions'
    )

    # Highlight points in red
    highlighted_points = pca_results[highlight_mask]
    highlighted_ids = np.array(sample_ids)[highlight_mask]

    plt.scatter(
        highlighted_points[:, 0],
        highlighted_points[:, 1],
        c='red',
        s=8,
        label=f'{interest_mz:.4f}'
    )

    # Add text labels for highlighted points
    for (x, y), label in zip(highlighted_points, highlighted_ids):
        plt.text(x, y, label, fontsize=9, color='red')

    # Labels and formatting
    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2f})")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def find_existing_mz(mz_tree, target_mz, ppm=10):
    """在树中查找是否有落入ppm范围内的m/z"""
    tolerance = target_mz * ppm * 1e-6
    lower = target_mz - tolerance
    upper = target_mz + tolerance
    for mz_key in mz_tree.keys(lower, upper):
        return mz_key
    return None


def two_classes_distance(outputs_final, class1_list, class2_list, class1_name='Class1', class2_name='Class2'):
    """
    比较两个类别中相同m/z下离子图嵌入向量的余弦相似度

    :param outputs_final: [['sample id', m/z array(length:  # ion images), embedding array (shape: # ion images, embedding dimension)], ...]
    :param class1_list: 样本名列表，属于类1
    :param class2_list: 样本名列表，属于类2
    :return: list of (mz, cosine_similarity)
    """
    mz_tree = OOBTree()

    for sample_output in outputs_final:
        sample_id = sample_output[0]
        mz_array = sample_output[1]
        embedding_array = sample_output[2]

        if sample_id in class1_list or sample_id in class2_list:
            for i in range(len(mz_array)):
                mz = mz_array[i]
                embedding = embedding_array[i]

                matched_mz = find_existing_mz(mz_tree, mz)
                if matched_mz is None:
                    # 新建 key
                    mz_tree[mz] = {
                        class1_name: [],
                        class2_name: []
                    }
                    matched_mz = mz  # 用新mz作为键

                if sample_id in class1_list:
                    mz_tree[matched_mz][class1_name].append(embedding)
                elif sample_id in class2_list:
                    mz_tree[matched_mz][class2_name].append(embedding)

    # 计算每个 m/z 的类间余弦相似度
    result = []
    for mz, class_dict in mz_tree.items():
        vecs1 = class_dict[class1_name]
        vecs2 = class_dict[class2_name]

        # 两个类别都必须有数据才能比较
        if len(vecs1) == 0 or len(vecs2) == 0:
            continue

        # 分别求中心点
        centroid1 = np.mean(vecs1, axis=0)
        centroid2 = np.mean(vecs2, axis=0)

        # 计算余弦相似度（注意：scipy 的 cosine 距离 = 1 - cosine similarity）
        similarity = 1 - cosine(centroid1, centroid2)
        result.append((mz, similarity))

    return result


def two_classes_davies_bouldin(
    outputs_final,
    class1_list,
    class2_list,
    class1_name='Class1',
    class2_name='Class2'
):
    """
    对每个 m/z 计算两个类别 embedding 的 Davies–Bouldin Index (DBI)

    DBI 越小，表示该 m/z 下：
    - 类内距离更小
    - 类间分离度更大

    :param outputs_final: [
        ['sample id',
         m/z array (length: # ion images),
         embedding array (shape: # ion images, embedding dimension)
        ], ...
    ]
    :param class1_list: 类 1 的样本名列表
    :param class2_list: 类 2 的样本名列表
    :return: list of (mz, db_index)
    """

    mz_tree = OOBTree()

    for sample_output in outputs_final:
        sample_id = sample_output[0]
        mz_array = sample_output[1]
        embedding_array = sample_output[2]

        if sample_id in class1_list or sample_id in class2_list:
            for i in range(len(mz_array)):
                mz = mz_array[i]
                embedding = embedding_array[i]

                matched_mz = find_existing_mz(mz_tree, mz)
                if matched_mz is None:
                    mz_tree[mz] = {
                        class1_name: [],
                        class2_name: []
                    }
                    matched_mz = mz

                if sample_id in class1_list:
                    mz_tree[matched_mz][class1_name].append(embedding)
                elif sample_id in class2_list:
                    mz_tree[matched_mz][class2_name].append(embedding)

    # -------- 计算 Davies–Bouldin Index --------
    result = []

    for mz, class_dict in mz_tree.items():
        vecs1 = np.array(class_dict[class1_name])
        vecs2 = np.array(class_dict[class2_name])

        # 两类都至少需要 2 个样本，DBI 才有意义
        if vecs1.shape[0] < 2 or vecs2.shape[0] < 2:
            continue

        # 质心
        centroid1 = np.mean(vecs1, axis=0)
        centroid2 = np.mean(vecs2, axis=0)

        # 类内散度（平均欧氏距离）
        S1 = np.mean(np.linalg.norm(vecs1 - centroid1, axis=1))
        S2 = np.mean(np.linalg.norm(vecs2 - centroid2, axis=1))

        # 类间距离
        M12 = np.linalg.norm(centroid1 - centroid2)

        # 避免数值问题
        if M12 == 0:
            continue

        db_index = (S1 + S2) / M12
        result.append((mz, db_index))

    return result



def plot_similarity_bar(sorted_similarity_results, top_k=None, figsize=(12, 6)):
    """
    可视化 m/z 与 cosine similarity 的柱状图

    :param sorted_similarity_results: List of (mz, similarity)，已经排序好
    :param top_k: 可选参数，是否只画前 top_k 个
    :param figsize: 图像大小
    """
    # 截取前 top_k（如果指定）
    if top_k:
        sorted_similarity_results = sorted_similarity_results[:top_k]

    mz_values = [round(mz, 4) for mz, sim in sorted_similarity_results]
    similarities = [sim for mz, sim in sorted_similarity_results]

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(mz_values)), similarities, color='skyblue', edgecolor='gray')

    # 设置 x 轴标签为 m/z 值（可能很多，需要旋转）
    plt.xticks(ticks=range(len(mz_values)), labels=mz_values, rotation=90, fontsize=8)

    plt.xlabel("m/z")
    # plt.ylabel("Cosine Similarity")
    # plt.title("Cosine Similarity Between Two Classes (Sorted by Similarity)")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.show()


def single_mz_visualization(original_inputs, sample_list, plot_mz):
    """

    :param original_inputs:
    [['sample id', shape_mask (bool array, shape: height, width), m/z array (length: # ion images), intensity array (shape: # ion images, height, width)], ...]
    :return:
    """
    n_cols = 3
    n_plots = len(sample_list)
    import math
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # 变成 1D 列表，方便索引

    ppm = 10
    tolerance = plot_mz * ppm * 1e-6

    idx = 0
    for s_input in original_inputs:
        if s_input[0] in sample_list:
            for idj in range(len(s_input[2])):
                mz = s_input[2][idj]
                if plot_mz - tolerance < mz < plot_mz + tolerance:
                    ax = axes[idx]
                    img = ax.imshow(s_input[3][idj], cmap='magma')
                    ax.set_title(f"Sample: {s_input[0]} m/z: {mz}")
                    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
                    idx += 1
                    break

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def find_metabolite_mz_pairs(outputs_final, s_sample, l_sample, mass_dev=2.0145):
    candidate_mz_pairs = []
    ppm = 10
    for s_output in outputs_final:
        if s_output[0] == s_sample:
            for ids in range(len(s_output[1])):
                s_mz = s_output[1][ids]
                tolerance = s_mz * ppm * 1e-6
                for l_output in outputs_final:
                    if l_output[0] == l_sample:
                        for idl in range(len(l_output[1])):
                            l_mz = l_output[1][idl]
                            if s_mz + mass_dev - tolerance < l_mz < s_mz + mass_dev + tolerance:
                                s_embedding = s_output[2][ids]
                                l_embedding = l_output[2][idl]
                                similarity = 1 - cosine(s_embedding, l_embedding)
                                candidate_mz_pairs.append((s_mz, l_mz, similarity))
    candidate_mz_pairs = sorted(candidate_mz_pairs, key=lambda x: x[2], reverse=True)
    return candidate_mz_pairs


def find_colocalized_mz_pairs(outputs_final, s_sample, l_sample, loc_mz):
    candidate_mz_pairs = []
    ppm = 10
    tolerance = loc_mz * ppm * 1e-6
    for s_output in outputs_final:
        if s_output[0] == s_sample:
            for ids in range(len(s_output[1])):
                s_mz = s_output[1][ids]
                s_embedding = s_output[2][ids]
                if loc_mz - tolerance < s_mz < loc_mz + tolerance:
                    for l_output in outputs_final:
                        if l_output[0] == l_sample:
                            for idl in range(len(l_output[1])):
                                l_mz = l_output[1][idl]
                                l_embedding = l_output[2][idl]
                                similarity = 1 - cosine(s_embedding, l_embedding)
                                candidate_mz_pairs.append((s_mz, l_mz, similarity))
    candidate_mz_pairs = sorted(candidate_mz_pairs, key=lambda x: x[2], reverse=True)
    return candidate_mz_pairs


def multi_class_distance(outputs_final, class_dict, ppm=10):
    """
    计算多个类别之间在相同 m/z 下嵌入向量的类间相似度（余弦相似度）

    :param outputs_final: [['sample id', m/z array, embedding array], ...]
    :param class_dict: dict，格式为 {'ClassA': [sample_id1, sample_id2], 'ClassB': [...], ...}
    :param ppm: ppm 容差
    :return: list of (mz, {(class1, class2): similarity, ...})
    """
    import itertools
    mz_tree = OOBTree()
    for sample_output in outputs_final:
        sample_id, mz_array, embedding_array = sample_output

        for class_name, sample_ids in class_dict.items():
            if sample_id in sample_ids:
                for i in range(len(mz_array)):
                    mz = mz_array[i]
                    embedding = embedding_array[i]

                    matched_mz = find_existing_mz(mz_tree, mz, ppm)
                    if matched_mz is None:
                        # 初始化所有类的列表
                        mz_tree[mz] = {k: [] for k in class_dict.keys()}
                        matched_mz = mz

                    mz_tree[matched_mz][class_name].append(embedding)
                break

    # 计算每个 m/z 下的类间相似度
    result = []
    for mz, class_embeddings in mz_tree.items():
        # 找出有数据的类别
        available_classes = [cls for cls, vecs in class_embeddings.items() if len(vecs) > 0]

        if len(available_classes) < 2:
            continue  # 至少两个类别才计算

        similarities = {}
        for cls1, cls2 in itertools.combinations(available_classes, 2):
            centroid1 = np.mean(class_embeddings[cls1], axis=0)
            centroid2 = np.mean(class_embeddings[cls2], axis=0)
            sim = 1 - cosine(centroid1, centroid2)
            similarities[(cls1, cls2)] = sim
        avg_sim = np.mean(list(similarities.values()))

        result.append((mz, avg_sim))

    return result


if __name__ == '__main__':
    """class Args:
        seed = 42
        lr = 1e-4
        epochs = 30
        batch_size = 200
        train_pairs_per_sample = 10000
        test_pairs_per_sample = 200
        margin = 0.1
        embedding_dim = 8
        output_path = './test_mice_brain_aging'


    import pickle

    with open('./test_mice_brain_aging/input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i', 'i', 'i'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    from embedding_models import ResNetEmbedding, EfficientNetEmbedding

    # model = ResNetEmbedding(embedding_dim=Args.embedding_dim)
    model = EfficientNetEmbedding(embedding_dim=Args.embedding_dim)

    # outputs = extract_embeddings(inputs_after, model, './test_mice_brain_aging/best_model.pth',
    #                              batch_size=200)
    outputs = extract_embeddings(inputs_after, model, './test_mice_brain_aging/temp_resnet.pth',
                                 batch_size=200)

    # two_files_comparison(inputs, inputs_after, outputs,  '3m_pos', '3m_neg')
    # multi_files_umap(outputs)
    # multi_files_pca(outputs, file_list=['3m_pos', '3m_neg'])
    multi_files_pca(outputs, file_list=['3m_pos', '14d_pos'])"""

    # Test in five samples
    r"""import pickle
    from image_preprocessing import *
    from embedding_models import SimpleCNNEmbedding, ResNetEmbedding, EfficientNetEmbedding

    with open(r'E:\yangjun\msi\MSI_IIE_article\test_five_samples\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'f', 'i', 'i', 'f'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    model = SimpleCNNEmbedding(embedding_dim=16)
    outputs = extract_embeddings(inputs_after, model,
                                 r'E:\yangjun\msi\MSI_IIE_article\test_five_samples\simplecnn_16d.pth',
                                 batch_size=200)

    # two_files_comparison(inputs, inputs_after, outputs, '14d_pos', '5m_pos')
    pca_results, sample_ids, mz_values, cluster_labels = multi_files_pca_kmeans(outputs, n_clusters=4)
    clusters_visualization(inputs, sample_ids, cluster_labels, r'E:\yangjun\msi\MSI_IIE_article\test_five_samples\kmeans_clustering_4')
    find_mz_groups(mz_values, sample_ids, cluster_labels, output_folder=r'E:\yangjun\msi\MSI_IIE_article\test_five_samples\kmeans_clustering_4')"""

    # Test in mcf-pos-neg
    r"""import pickle
    from image_preprocessing import *
    from embedding_models import MultiscaleEmbedding

    with open(r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'i'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    model = MultiscaleEmbedding(embedding_dim=32)
    outputs = extract_embeddings(inputs_after, model,
                                 r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\multiscale_cnn_32d.pth',
                                 batch_size=200)

    # two_files_comparison(inputs, inputs_after, outputs, 'mcf-pos', 'mcf-neg')
    pca_results, sample_ids, mz_values, cluster_labels = multi_files_pca_kmeans(outputs, n_clusters=8)
    clusters_visualization(inputs, sample_ids, cluster_labels, r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\kmeans_clustering_8')
    find_mz_groups(mz_values, sample_ids, cluster_labels, output_folder=r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\kmeans_clustering_8')"""

    # Application on ad study
    r"""import os
    import pickle
    from embedding_models import MultiscaleEmbedding

    with open(r'E:\yangjun\msi\MSI_IIE_article\ad_study\input_data.pkl', 'rb') as f:
        inputs = pickle.load(f)


    class Args:
        seed = 42
        lr = 1e-4
        epochs = 40
        batch_size = 200
        train_pairs_per_sample = 8000
        test_pairs_per_sample = 1000
        embedding_dim = 32
        output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_study'
        model_data_file = 'multiscale_cnn_32d_ad.pth'


    args = Args()

    from image_preprocessing import *

    inputs_after = pre_alignment(inputs, sample_transform=['i', 'f', 'i', 'i', 'f',
                                                           'i', 'i', 'i', 'i', 'f',
                                                           'i', 'f', 'i', 'i', 'f',
                                                           'i', 'i', 'i', 'i', 'f'])
    inputs_after = input_normalization(inputs_after)
    ih, iw = get_input_size(inputs_after)
    inputs_after = resize_images(inputs_after, ih, iw)
    print('Preprocessing finishing.')

    model = MultiscaleEmbedding(embedding_dim=args.embedding_dim)
    outputs = extract_embeddings(inputs_after, model,
                                 os.path.join(args.output_path, args.model_data_file),
                                 batch_size=200)

    outputs_output_path = r'E:\yangjun\msi\MSI_IIE_article\ad_study\output_data.pkl'
    with open(outputs_output_path, 'wb') as f:
        pickle.dump(outputs, f)

    two_files_comparison(inputs, inputs_after, outputs, 'M_14d_neg', 'P_14d_neg')
    pca_results, sample_ids, mz_values, cluster_labels = multi_files_pca_kmeans(outputs, n_clusters=4)
    clusters_visualization(inputs, sample_ids, cluster_labels,
                           r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_4')
    find_mz_groups(mz_values, sample_ids, cluster_labels,
                   output_folder=r'E:\yangjun\msi\MSI_IIE_article\ad_study\kmeans_clustering_4')"""
    import pickle

    with open(r'E:\yangjun\msi\MSI_IIE_article\ad_study\output_data.pkl', 'rb') as f:
        outputs = pickle.load(f)

    pos_id_list = ['M_14d_pos', 'M_1m_pos', 'M_2m_pos', 'M_3m_pos', 'M_5m_pos',
                   'P_14d_pos', 'P_1m_pos', 'P_2m_pos', 'P_3m_pos', 'P_5m_pos']
    neg_id_list = ['M_14d_neg', 'M_1m_neg', 'M_2m_neg', 'M_3m_neg', 'M_5m_neg',
                   'P_14d_neg', 'P_1m_neg', 'P_2m_neg', 'P_3m_neg', 'P_5m_neg']
    # visualize_mz_embedding(outputs, pos_id_list, 169.98393)
    visualize_mz_embedding(outputs, neg_id_list, 733.4954)
