"""Configuration settings for the multimodal design analysis suite."""

import os
import re
from pathlib import Path
from typing import Optional, Union
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Hugging Face Configuration
    huggingface_api_token: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Model Configuration
    embedding_model_name: str = "biglab/uiclip_jitteredwebsites-2-224-paraphrased_webpairs_humanpairs"
    llm_model_name: str = "openai/gpt-4-turbo-preview"
    
    # File and Storage Configuration
    upload_dir: Path = Path("uploads")
    vector_db_path: Path = Path("vector_db")
    max_file_size: str = "50MB"  # Will be converted to int by validator
    
    # Application Configuration
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))  # Use Railway's PORT env var
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @field_validator('max_file_size')
    @classmethod
    def parse_file_size(cls, v):
        """Parse file size from string format like '50MB' to bytes."""
        if isinstance(v, int):
            return str(v)  # Keep as string for now
        if isinstance(v, str):
            # Parse formats like "50MB", "10GB", "1024KB"
            match = re.match(r'^(\d+(?:\.\d+)?)\s*(MB|GB|KB|B)?$', v.upper().strip())
            if match:
                size, unit = match.groups()
                size = float(size)
                
                if unit == 'GB':
                    bytes_value = int(size * 1024 * 1024 * 1024)
                elif unit == 'MB' or unit is None:  # Default to MB if no unit
                    bytes_value = int(size * 1024 * 1024)
                elif unit == 'KB':
                    bytes_value = int(size * 1024)
                elif unit == 'B':
                    bytes_value = int(size)
                else:
                    bytes_value = int(size * 1024 * 1024)  # Default to MB
                
                return str(bytes_value)  # Return as string to avoid type issues
            
            # If it's just a number as string, treat as bytes
            try:
                return str(int(v))
            except ValueError:
                pass
        
        raise ValueError(f"Invalid file size format: {v}. Use formats like '50MB', '1GB', or integer bytes.")
    
    def get_max_file_size_bytes(self) -> int:
        """Get max_file_size as integer bytes."""
        return int(self.max_file_size)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(exist_ok=True)
settings.vector_db_path.mkdir(exist_ok=True)
