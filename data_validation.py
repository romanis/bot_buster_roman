"""Data validation utilities for bot detection system."""

import polars as pl
from typing import List, Optional, Dict, Any, Union
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError(Exception):
    """Custom exception for data validation errors."""
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class DataValidator:
    """Validates data quality for bot detection analysis."""
    
    def __init__(self):
        """Initialize data validator."""
        self.required_comment_fields = [
            'username', 'user_id', 'number_comments_all_time', 
            'max_CPP_this_CH', 'channel_title'
        ]
        self.required_text_fields = [
            'comment_id', 'user_id', 'text', 'published_at'
        ]
    
    def validate_dataframe_structure(
        self, 
        df: pl.DataFrame, 
        required_fields: List[str],
        df_name: str = "DataFrame"
    ) -> ValidationResult:
        """Validate basic DataFrame structure."""
        errors = []
        warnings = []
        
        try:
            # Check if DataFrame is empty
            if len(df) == 0:
                errors.append(f"{df_name} is empty")
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                errors.append(f"{df_name} missing required fields: {missing_fields}")
            
            # Check for null values in critical fields
            for field in required_fields:
                if field in df.columns:
                    null_count = df.select(pl.col(field).is_null().sum()).item()
                    if null_count > 0:
                        warnings.append(f"{df_name}.{field} has {null_count} null values")
            
            # Check data types
            numeric_fields = ['user_id', 'number_comments_all_time', 'max_CPP_this_CH']
            for field in numeric_fields:
                if field in df.columns:
                    if not df.select(pl.col(field).dtype).item().is_numeric():
                        warnings.append(f"{df_name}.{field} should be numeric")
            
            summary = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_fields': missing_fields,
                'columns_with_nulls': [
                    col for col in df.columns 
                    if df.select(pl.col(col).is_null().sum()).item() > 0
                ]
            }
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error validating DataFrame structure: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                summary={}
            )
    
    def validate_comment_data(self, df: pl.DataFrame) -> ValidationResult:
        """Validate comment analysis data."""
        try:
            # Basic structure validation
            result = self.validate_dataframe_structure(
                df, self.required_comment_fields, "Comment data"
            )
            
            if not result.is_valid:
                return result
            
            # Additional comment-specific validations
            additional_errors = []
            additional_warnings = []
            
            # Check for suspicious data patterns
            if 'number_comments_all_time' in df.columns:
                max_comments = df.select(pl.col('number_comments_all_time').max()).item()
                if max_comments > 100000:  # Extremely high comment count
                    additional_warnings.append(
                        f"Unusually high comment count detected: {max_comments}"
                    )
                
                # Check for negative values
                negative_count = df.filter(
                    pl.col('number_comments_all_time') < 0
                ).shape[0]
                if negative_count > 0:
                    additional_errors.append(
                        f"{negative_count} records have negative comment counts"
                    )
            
            # Validate user_id consistency
            if 'user_id' in df.columns and 'username' in df.columns:
                user_id_username_pairs = df.select(['user_id', 'username']).unique()
                duplicate_usernames = (
                    user_id_username_pairs.group_by('username')
                    .agg(pl.col('user_id').n_unique().alias('unique_ids'))
                    .filter(pl.col('unique_ids') > 1)
                )
                
                if len(duplicate_usernames) > 0:
                    additional_warnings.append(
                        f"{len(duplicate_usernames)} usernames map to multiple user_ids"
                    )
            
            # Check date ranges if available
            if 'month' in df.columns:
                date_range = df.select([
                    pl.col('month').min().alias('min_date'),
                    pl.col('month').max().alias('max_date')
                ]).to_dicts()[0]
                
                result.summary.update({
                    'date_range': date_range,
                    'unique_users': df.select(pl.col('user_id').n_unique()).item() if 'user_id' in df.columns else None,
                    'unique_channels': df.select(pl.col('channel_title').n_unique()).item() if 'channel_title' in df.columns else None
                })
            
            # Combine results
            result.errors.extend(additional_errors)
            result.warnings.extend(additional_warnings)
            result.is_valid = result.is_valid and len(additional_errors) == 0
            
            # Log validation results
            if result.errors:
                logger.error(f"Comment data validation errors: {result.errors}")
            if result.warnings:
                logger.warning(f"Comment data validation warnings: {result.warnings}")
            
            logger.info(f"Comment data validation completed. Valid: {result.is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating comment data: {e}")
            raise ValidationError(f"Comment data validation failed: {str(e)}")
    
    def validate_comment_text_data(self, df: pl.DataFrame) -> ValidationResult:
        """Validate comment text data for ML analysis."""
        try:
            # Basic structure validation
            result = self.validate_dataframe_structure(
                df, self.required_text_fields, "Comment text data"
            )
            
            if not result.is_valid:
                return result
            
            additional_errors = []
            additional_warnings = []
            
            # Check text quality
            if 'text' in df.columns:
                # Count empty or very short texts
                short_text_count = df.filter(
                    (pl.col('text').is_null()) | 
                    (pl.col('text').str.len_chars() < 5)
                ).shape[0]
                
                if short_text_count > 0:
                    additional_warnings.append(
                        f"{short_text_count} comments have very short or empty text"
                    )
                
                # Check for duplicate text
                total_comments = len(df)
                unique_texts = df.select(pl.col('text').n_unique()).item()
                duplicate_ratio = 1 - (unique_texts / total_comments)
                
                if duplicate_ratio > 0.1:  # More than 10% duplicates
                    additional_warnings.append(
                        f"High duplicate text ratio: {duplicate_ratio:.2%}"
                    )
                
                # Text length statistics
                text_stats = df.select([
                    pl.col('text').str.len_chars().mean().alias('avg_length'),
                    pl.col('text').str.len_chars().median().alias('median_length'),
                    pl.col('text').str.len_chars().max().alias('max_length'),
                    pl.col('text').str.len_chars().min().alias('min_length')
                ]).to_dicts()[0]
                
                result.summary.update({
                    'text_statistics': text_stats,
                    'unique_text_ratio': unique_texts / total_comments,
                    'short_text_count': short_text_count
                })
            
            # Validate date format if available
            if 'published_at' in df.columns:
                try:
                    # Try to parse dates
                    date_sample = df.select(pl.col('published_at')).limit(100)
                    # Basic date format check - this is simplified
                    non_null_dates = date_sample.filter(pl.col('published_at').is_not_null())
                    if len(non_null_dates) == 0:
                        additional_warnings.append("All published_at values are null")
                except Exception as e:
                    additional_warnings.append(f"Date validation issue: {str(e)}")
            
            # Combine results
            result.errors.extend(additional_errors)
            result.warnings.extend(additional_warnings)
            result.is_valid = result.is_valid and len(additional_errors) == 0
            
            # Log results
            if result.errors:
                logger.error(f"Comment text validation errors: {result.errors}")
            if result.warnings:
                logger.warning(f"Comment text validation warnings: {result.warnings}")
            
            logger.info(f"Comment text validation completed. Valid: {result.is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating comment text data: {e}")
            raise ValidationError(f"Comment text validation failed: {str(e)}")
    
    def validate_clustering_results(
        self, 
        df: pl.DataFrame, 
        expected_clusters: int = None
    ) -> ValidationResult:
        """Validate clustering results."""
        try:
            errors = []
            warnings = []
            
            required_cluster_fields = ['cluster_label', 'tsne_x', 'tsne_y']
            missing_fields = [field for field in required_cluster_fields if field not in df.columns]
            
            if missing_fields:
                errors.append(f"Missing clustering fields: {missing_fields}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    summary={}
                )
            
            # Check cluster distribution
            cluster_counts = (
                df.group_by('cluster_label')
                .agg(pl.len().alias('count'))
                .sort('cluster_label')
            )
            
            unique_clusters = len(cluster_counts)
            if expected_clusters and unique_clusters != expected_clusters:
                warnings.append(
                    f"Expected {expected_clusters} clusters, found {unique_clusters}"
                )
            
            # Check for empty clusters or very small clusters
            min_cluster_size = cluster_counts.select(pl.col('count').min()).item()
            if min_cluster_size < 5:
                warnings.append(f"Some clusters have very few members (min: {min_cluster_size})")
            
            # Check t-SNE coordinates
            tsne_stats = df.select([
                pl.col('tsne_x').is_null().sum().alias('null_x'),
                pl.col('tsne_y').is_null().sum().alias('null_y'),
                pl.col('tsne_x').is_infinite().sum().alias('inf_x'),
                pl.col('tsne_y').is_infinite().sum().alias('inf_y')
            ]).to_dicts()[0]
            
            if any(tsne_stats.values()):
                warnings.append(f"t-SNE coordinates have null/infinite values: {tsne_stats}")
            
            summary = {
                'unique_clusters': unique_clusters,
                'cluster_distribution': cluster_counts.to_dicts(),
                'tsne_coordinate_stats': tsne_stats
            }
            
            is_valid = len(errors) == 0
            
            logger.info(f"Clustering validation completed. Valid: {is_valid}")
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error validating clustering results: {e}")
            raise ValidationError(f"Clustering validation failed: {str(e)}")


def validate_input_parameters(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    threshold: Optional[int] = None
) -> ValidationResult:
    """Validate input parameters for analysis."""
    errors = []
    warnings = []
    
    try:
        # Validate date format
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                errors.append(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")
        
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                errors.append(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD")
        
        # Validate date order
        if start_date and end_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start_dt >= end_dt:
                    errors.append("start_date must be before end_date")
                
                # Check if date range is too large
                date_diff = (end_dt - start_dt).days
                if date_diff > 365:
                    warnings.append(f"Large date range ({date_diff} days) may affect performance")
                    
            except ValueError as e:
                errors.append(f"Date validation error: {str(e)}")
        
        # Validate threshold
        if threshold is not None:
            if not isinstance(threshold, int) or threshold < 1:
                errors.append("threshold must be a positive integer")
            elif threshold > 1000:
                warnings.append("Very high threshold may result in few suspicious users")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            summary={
                'start_date': start_date,
                'end_date': end_date,
                'threshold': threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Parameter validation error: {e}")
        return ValidationResult(
            is_valid=False,
            errors=[f"Parameter validation failed: {str(e)}"],
            warnings=[],
            summary={}
        )