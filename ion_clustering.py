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

            with torch.no_grad():
                embeddings = model(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        print(all_embeddings.shape)
        outputs.append([sample_id, mz_array, all_embeddings])
    return outputs

