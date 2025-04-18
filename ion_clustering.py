import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Tuple


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
    # top5_indices = np.argsort(cos_sim.flatten())[:5]
    top5_indices = np.argsort(cos_sim.flatten())[-10005:-10000]
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
        axes[row, 0].set_title(f"{file1} - m/z={input_file1[2][i]:.4f}(Sim: {cos_sim[i, j]:.2f}, OriSim: {cosine_similarity(img1_prealign, img2_prealign):.2f})")
        axes[row, 0].axis('off')

        # 绘制file2的图像
        axes[row, 1].imshow(img2, cmap='viridis')
        axes[row, 1].set_title(f"{file2} - m/z={input_file2[2][j]:.4f}")
        axes[row, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return cos_sim, top5_pairs



if __name__ == '__main__':
    class Args:
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

    from train_model import ResNetEmbedding

    model = ResNetEmbedding(embedding_dim=Args.embedding_dim)

    outputs = extract_embeddings(inputs_after, model, './test_mice_brain_aging/best_model.pth',
                                 batch_size=200)

    two_files_comparison(inputs, inputs_after, outputs,  '14d_pos', '3m_pos')

