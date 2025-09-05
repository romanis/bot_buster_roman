"""Machine learning processing for comment analysis."""

import torch
import numpy as np
import polars as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from typing import List, Tuple, Optional, Dict, Any
import logging
from tqdm import tqdm
from dataclasses import dataclass
from config import model_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result container for embedding processing."""
    embeddings: np.ndarray
    reduced_embeddings: np.ndarray
    cluster_labels: np.ndarray
    tsne_projection: np.ndarray
    pca_explained_variance: float


class CommentEmbeddingProcessor:
    """Processes comment embeddings for clustering analysis."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding processor."""
        self.model_name = model_name or model_config.model_name
        self.batch_size = model_config.batch_size
        self.max_length = model_config.max_length
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of sentences.
        Uses mean pooling with attention mask for better sentence representations.
        """
        if not sentences:
            raise ValueError("No sentences provided")
        
        sentence_batches = [
            sentences[i:i + self.batch_size] 
            for i in range(0, len(sentences), self.batch_size)
        ]
        
        all_embeddings = []
        
        try:
            for batch in tqdm(sentence_batches, desc="Processing batches"):
                # Tokenize batch
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length
                )
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    token_embeddings = outputs.last_hidden_state
                    
                    # Mean pooling with attention mask
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    
                    all_embeddings.append(batch_embeddings)
            
            # Stack all embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def reduce_dimensions(
        self, 
        embeddings: np.ndarray, 
        n_components: int = None
    ) -> Tuple[np.ndarray, float]:
        """Reduce dimensionality using PCA."""
        n_components = n_components or model_config.pca_components
        
        try:
            logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}")
            pca = PCA(n_components=n_components, random_state=model_config.random_state)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_variance:.3f}")
            
            return reduced_embeddings, explained_variance
            
        except Exception as e:
            logger.error(f"Error in dimension reduction: {e}")
            raise
    
    def cluster_embeddings(
        self, 
        embeddings: np.ndarray, 
        n_clusters: int = None
    ) -> np.ndarray:
        """Cluster embeddings using K-means."""
        n_clusters = n_clusters or model_config.n_clusters
        
        try:
            logger.info(f"Clustering with {n_clusters} clusters")
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=model_config.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)
            
            # Log cluster distribution
            unique, counts = np.unique(labels, return_counts=True)
            cluster_dist = dict(zip(unique, counts))
            logger.info(f"Cluster distribution: {cluster_dist}")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            raise
    
    def create_tsne_projection(
        self, 
        embeddings: np.ndarray, 
        perplexity: int = None
    ) -> np.ndarray:
        """Create t-SNE projection for visualization."""
        perplexity = perplexity or model_config.tsne_perplexity
        
        try:
            logger.info(f"Creating t-SNE projection with perplexity {perplexity}")
            tsne = TSNE(
                n_components=2, 
                perplexity=perplexity,
                random_state=model_config.random_state,
                n_iter=1000
            )
            projection = tsne.fit_transform(embeddings)
            logger.info("t-SNE projection completed")
            return projection
            
        except Exception as e:
            logger.error(f"Error creating t-SNE projection: {e}")
            raise
    
    def process_comments(
        self, 
        comments: List[str],
        n_components: int = None,
        n_clusters: int = None
    ) -> EmbeddingResult:
        """
        Complete processing pipeline: embeddings -> PCA -> clustering -> t-SNE.
        """
        if not comments:
            raise ValueError("No comments provided for processing")
        
        logger.info(f"Processing {len(comments)} comments")
        
        try:
            # Generate embeddings
            embeddings_tensor = self._get_sentence_embeddings(comments)
            embeddings = embeddings_tensor.numpy()
            
            # Reduce dimensions
            reduced_embeddings, explained_variance = self.reduce_dimensions(
                embeddings, n_components
            )
            
            # Cluster
            cluster_labels = self.cluster_embeddings(reduced_embeddings, n_clusters)
            
            # Create t-SNE projection
            tsne_projection = self.create_tsne_projection(reduced_embeddings)
            
            result = EmbeddingResult(
                embeddings=embeddings,
                reduced_embeddings=reduced_embeddings,
                cluster_labels=cluster_labels,
                tsne_projection=tsne_projection,
                pca_explained_variance=explained_variance
            )
            
            logger.info("Comment processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in comment processing pipeline: {e}")
            raise


class CommentAnalyzer:
    """High-level interface for comment analysis."""
    
    def __init__(self):
        self.processor = CommentEmbeddingProcessor()
    
    def analyze_comment_clusters(
        self, 
        comments_df: pl.DataFrame,
        text_column: str = 'text'
    ) -> pl.DataFrame:
        """
        Analyze comments and add cluster labels to DataFrame.
        """
        if text_column not in comments_df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Extract text
        comments_text = comments_df.select(pl.col(text_column)).to_numpy().flatten().tolist()
        
        # Process comments
        result = self.processor.process_comments(comments_text)
        
        # Add results to DataFrame
        enhanced_df = comments_df.with_columns([
            pl.Series('cluster_label', result.cluster_labels),
            pl.Series('tsne_x', result.tsne_projection[:, 0]),
            pl.Series('tsne_y', result.tsne_projection[:, 1])
        ])
        
        return enhanced_df, result
    
    def get_cluster_samples(
        self, 
        comments_df: pl.DataFrame, 
        cluster_id: int,
        text_column: str = 'text',
        n_samples: int = 10
    ) -> List[str]:
        """Get sample comments from a specific cluster."""
        cluster_comments = comments_df.filter(
            pl.col('cluster_label') == cluster_id
        ).select(pl.col(text_column)).limit(n_samples)
        
        return cluster_comments.to_numpy().flatten().tolist()


def validate_comments_data(comments: List[str]) -> List[str]:
    """Validate and clean comments data."""
    if not comments:
        raise ValueError("No comments provided")
    
    # Filter out empty or very short comments
    valid_comments = [
        comment.strip() for comment in comments 
        if comment and len(comment.strip()) > 5
    ]
    
    if not valid_comments:
        raise ValueError("No valid comments after filtering")
    
    logger.info(f"Validated {len(valid_comments)} comments from {len(comments)} total")
    return valid_comments