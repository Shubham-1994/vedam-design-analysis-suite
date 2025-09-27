"""Embedding utilities using UIClip model from Hugging Face."""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union, Optional, Tuple
import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

# UIClip configuration constants
IMG_SIZE = 224
LOGIT_SCALE = 100  # based on OpenAI's CLIP example code
NORMALIZE_SCORING = True


class UIClipEmbeddings:
    """UIClip embeddings for UI/UX design analysis."""
    
    def __init__(self, huggingface_token: Optional[str] = None):
        self.model_name = settings.embedding_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.huggingface_token = huggingface_token or settings.huggingface_api_token
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the UIClip model and processor."""
        try:
            logger.info(f"Loading UIClip model: {self.model_name}")
            
            # Try to load the UIClip model with proper model and processor paths
            model_path = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
            processor_path = "openai/clip-vit-base-patch32"
            
            # Load model with token if available
            kwargs = {}
            if self.huggingface_token:
                kwargs['token'] = self.huggingface_token
            
            self.model = CLIPModel.from_pretrained(model_path, **kwargs)
            self.model = self.model.eval()
            self.model = self.model.to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(processor_path, **kwargs)
            
            logger.info("UIClip model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load UIClip model: {e}")
            logger.info("Falling back to standard CLIP model...")
            try:
                # Fallback to standard CLIP model
                fallback_model = "openai/clip-vit-base-patch32"
                logger.info(f"Loading fallback CLIP model: {fallback_model}")
                self.model = CLIPModel.from_pretrained(fallback_model)
                self.model = self.model.eval()
                self.model = self.model.to(self.device)
                self.processor = CLIPProcessor.from_pretrained(fallback_model)
                logger.info("Fallback CLIP model loaded successfully")
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback CLIP model: {fallback_e}")
                raise
    
    def compute_quality_scores(self, input_list: List[Tuple[str, Image.Image]]) -> torch.Tensor:
        """
        Compute quality scores for UI designs.
        
        Args:
            input_list: List of tuples where first element is description and second is PIL image
            
        Returns:
            Tensor of quality scores
        """
        # input_list is a list of types where the first element is a description and the second is a PIL image
        description_list = ["ui screenshot. well-designed. " + input_item[0] for input_item in input_list]
        img_list = [input_item[1] for input_item in input_list]
        text_embeddings_tensor = self.compute_description_embeddings(description_list)  # B x H
        img_embeddings_tensor = self.compute_image_embeddings(img_list)  # B x H

        # normalize tensors
        text_embeddings_tensor /= text_embeddings_tensor.norm(dim=-1, keepdim=True)
        img_embeddings_tensor /= img_embeddings_tensor.norm(dim=-1, keepdim=True)

        if NORMALIZE_SCORING:
            text_embeddings_tensor_poor = self.compute_description_embeddings([d.replace("well-designed. ", "poor design. ") for d in description_list])  # B x H
            text_embeddings_tensor_poor /= text_embeddings_tensor_poor.norm(dim=-1, keepdim=True)
            text_embeddings_tensor_all = torch.stack((text_embeddings_tensor, text_embeddings_tensor_poor), dim=1)  # B x 2 x H
        else:
            text_embeddings_tensor_all = text_embeddings_tensor.unsqueeze(1)

        img_embeddings_tensor = img_embeddings_tensor.unsqueeze(1)  # B x 1 x H

        scores = (LOGIT_SCALE * img_embeddings_tensor @ text_embeddings_tensor_all.permute(0, 2, 1)).squeeze(1)

        if NORMALIZE_SCORING:
            scores = scores.softmax(dim=-1)

        return scores[:, 0]

    def compute_description_embeddings(self, descriptions: List[str]) -> torch.Tensor:
        """Compute embeddings for text descriptions."""
        inputs = self.processor(text=descriptions, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        return text_embedding

    def compute_image_embeddings(self, image_list: List[Image.Image]) -> torch.Tensor:
        """Compute embeddings for images using sliding window approach."""
        windowed_batch = [self.slide_window_over_image(img, IMG_SIZE) for img in image_list]
        inds = []
        for imgi in range(len(windowed_batch)):
            inds.append([imgi for _ in windowed_batch[imgi]])

        processed_batch = [item for sublist in windowed_batch for item in sublist]
        inputs = self.processor(images=processed_batch, return_tensors="pt")
        # run all sub windows of all images in batch through the model
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # output contains all subwindows, need to mask for each image
        processed_batch_inds = torch.tensor([item for sublist in inds for item in sublist]).long().to(image_features.device)
        embed_list = []
        for i in range(len(image_list)):
            mask = processed_batch_inds == i
            embed_list.append(image_features[mask].mean(dim=0))
        image_embedding = torch.stack(embed_list, dim=0)
        return image_embedding

    def preresize_image(self, image: Image.Image, image_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            image = image.resize((int(aspect_ratio * image_size), image_size))
        else:
            image = image.resize((image_size, int(image_size / aspect_ratio)))
        return image

    def slide_window_over_image(self, input_image: Image.Image, img_size: int) -> List[Image.Image]:
        """Apply sliding window over image to create crops."""
        input_image = self.preresize_image(input_image, img_size)
        width, height = input_image.size
        square_size = min(width, height)
        longer_dimension = max(width, height)
        num_steps = (longer_dimension + square_size - 1) // square_size

        if num_steps > 1:
            step_size = (longer_dimension - square_size) // (num_steps - 1)
        else:
            step_size = square_size

        cropped_images = []

        for y in range(0, height - square_size + 1, step_size if height > width else square_size):
            for x in range(0, width - square_size + 1, step_size if width > height else square_size):
                left = x
                upper = y
                right = x + square_size
                lower = y + square_size
                cropped_image = input_image.crop((left, upper, right, lower))
                cropped_images.append(cropped_image)

        return cropped_images
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Encode an image to embeddings."""
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be PIL Image, file path, or string path")
            
            # Use the sliding window approach for consistency
            image_embeddings = self.compute_image_embeddings([image])
            return image_embeddings[0].detach().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings."""
        try:
            text_embeddings = self.compute_description_embeddings([text])
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings[0].detach().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def batch_encode_images(self, images: List[Union[str, Path, Image.Image]]) -> List[np.ndarray]:
        """Encode multiple images in batch."""
        embeddings = []
        for image in images:
            try:
                embedding = self.encode_image(image)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to encode image {image}: {e}")
                # Add zero embedding as placeholder
                embeddings.append(np.zeros(512))  # Assuming 512-dim embeddings
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # Create a dummy embedding to get dimension
        dummy_text = "test"
        embedding = self.encode_text(dummy_text)
        return embedding.shape[0]


# Global embeddings instance
_embeddings_instance: Optional[UIClipEmbeddings] = None


def get_embeddings(huggingface_token: Optional[str] = None) -> UIClipEmbeddings:
    """Get or create the global embeddings instance."""
    global _embeddings_instance
    # Create new instance if token is provided or if no instance exists
    if _embeddings_instance is None or huggingface_token:
        _embeddings_instance = UIClipEmbeddings(huggingface_token)
    return _embeddings_instance