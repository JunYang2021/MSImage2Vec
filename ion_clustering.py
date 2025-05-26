import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Tuple
import seaborn as sns


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

    top5_indices = np.argsort(cos_sim.flatten())[-5:][::-1]  # 降序排列
    # top5_indices = np.argsort(cos_sim.flatten())[:5]
    # top5_indices = np.argsort(cos_sim.flatten())[-10005:-10000]
    top5_pairs = [(idx // cos_sim.shape[1], idx % cos_sim.shape[1]) for idx in top5_indices]

    # 绘制5对图像
    import matplotlib.pyplot as plt
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
def multi_files_umap(outputs_final, file_list=None):
    """

    :param outputs_final:
    [['sample id', m/z array (length: # ion images), embedding array (shape: # ion images, embedding dim)], ...]
    :param file_list:
    :return:
    """
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

    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(all_embeddings)  # shape: (total_ions, 2)

    unique_samples = list(set(sample_ids))
    palette = sns.color_palette("husl", len(unique_samples))  # 使用seaborn调色板
    color_map = {sample: palette[i] for i, sample in enumerate(unique_samples)}
    colors = [color_map[sample] for sample in sample_ids]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        umap_results[:, 0],
        umap_results[:, 1],
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
    plt.legend(handles=legend_elements, title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("UMAP Projection of Ion Images (Colored by Sample ID)", fontsize=14)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    return umap_results, sample_ids, mz_values


def multi_files_pca_kmeans(outputs_final, file_list=None, n_clusters=8):
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
    plt.figure(figsize=(18, 7))

    # Plot 1:
    plt.subplot(1, 2, 1)
    unique_samples = list(set(sample_ids))
    palette = sns.color_palette("husl", len(unique_samples))
    color_map = {sample: palette[i] for i, sample in enumerate(unique_samples)}
    colors = [color_map[sample] for sample in sample_ids]

    scatter = plt.scatter(
        pca_results[:, 0],
        pca_results[:, 1],
        c=colors,
        alpha=0.6,
        s=10
    )

    # 添加图例和标签
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=sample,
                   markerfacecolor=color_map[sample], markersize=10)
        for sample in unique_samples
    ]
    plt.legend(handles=legend_elements, title="Sample ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("PCA Projection of Ion Images (Colored by Sample ID)", fontsize=14)
    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True, alpha=0.2)

    # Plot 2:
    plt.subplot(1, 2, 2)
    cluster_palette = sns.color_palette("husl", n_clusters)
    cluster_colors = [cluster_palette[label] for label in cluster_labels]

    plt.scatter(
        pca_results[:, 0],
        pca_results[:, 1],
        c=cluster_colors,
        alpha=0.6,
        s=10
    )

    # 添加聚类中心的标记
    centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.8, marker='X')
    for i, center in enumerate(centers):
        plt.text(center[0], center[1], str(i), fontsize=12, ha='center', va='center', color='black')

    plt.title(f"PCA Projection with K-means Clustering (k={n_clusters})", fontsize=14)
    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
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
    """import pickle
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
    import pickle
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
    find_mz_groups(mz_values, sample_ids, cluster_labels, output_folder=r'E:\yangjun\msi\MSI_IIE_article\test_mcf_pos_neg\kmeans_clustering_8')



