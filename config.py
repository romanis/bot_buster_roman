"""Configuration management for bot detection system."""

import os
from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    user: str = "root"
    password: str = ""
    database: str = "bot_buster"
    port: int = 3306
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'bot_buster'),
            port=int(os.getenv('DB_PORT', '3306'))
        )


@dataclass
class ModelConfig:
    """ML model configuration."""
    model_name: str = "DeepPavlov/rubert-base-cased-sentence"
    batch_size: int = 32
    max_length: int = 128
    pca_components: int = 50
    n_clusters: int = 10
    tsne_perplexity: int = 30
    random_state: int = 42


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    default_start_date: str = '2023-10-31'
    default_end_date: str = '2024-10-31'
    interval_format: str = '%Y-%m-%d %H:%i'
    suspicious_threshold: int = 10
    user_id_mod: int = 100


# Global configuration
db_config = DatabaseConfig.from_env()
model_config = ModelConfig()
analysis_config = AnalysisConfig()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bot_detection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)