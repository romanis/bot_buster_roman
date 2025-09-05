"""Main bot detection orchestrator that ties together all components."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import polars as pl
import sys

from config import setup_logging, analysis_config, db_config
from database import DatabaseManager, get_data_from_db
from ml_processing import CommentAnalyzer, validate_comments_data
from visualization import save_visualization_report
from data_validation import DataValidator, ValidationError

logger = setup_logging()


class BotDetector:
    """Main class for detecting bot behavior in YouTube comments."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize the bot detector with optional configuration override."""
        self.config = analysis_config
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        self.db_manager = DatabaseManager()
        self.comment_analyzer = CommentAnalyzer()
        self.validator = DataValidator()
        
        logger.info("Bot detector initialized successfully")
    
    def validate_database_connection(self) -> bool:
        """Validate database connection and required tables."""
        try:
            if not self.db_manager.validate_connection():
                logger.error("Database connection failed")
                return False
            
            # Check required tables exist
            required_tables = ['youtube_comments', 'youtube_users', 'youtube_videos', 'youtube_channels']
            for table in required_tables:
                try:
                    info = self.db_manager.get_table_info(table)
                    logger.info(f"Table {table}: {len(info)} columns")
                except Exception as e:
                    logger.error(f"Table {table} not accessible: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False
    
    def load_comment_data(
        self, 
        start_date: str = None, 
        end_date: str = None,
        use_optimized_query: bool = True
    ) -> pl.DataFrame:
        """Load comment data from database."""
        start_date = start_date or self.config.default_start_date
        end_date = end_date or self.config.default_end_date
        
        query_template = (
            "queries/minutely_comments_optimized.sql" if use_optimized_query 
            else "queries/minutely_comments.sql"
        )
        
        try:
            logger.info(f"Loading comment data from {start_date} to {end_date}")
            
            df = get_data_from_db(
                query_template,
                start_date=start_date,
                end_date=end_date,
                interval_format=self.config.interval_format,
                user_id_mod=self.config.user_id_mod,
                min_comments=1
            )
            
            # Validate loaded data
            self.validator.validate_comment_data(df)
            
            logger.info(f"Loaded {len(df)} comment records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading comment data: {e}")
            raise
    
    def identify_suspicious_commenters(
        self, 
        df: pl.DataFrame, 
        threshold: int = None
    ) -> pl.DataFrame:
        """Identify suspicious commenters based on activity patterns."""
        threshold = threshold or self.config.suspicious_threshold
        
        try:
            suspicious_df = df.filter(pl.col('max_CPP_this_CH') >= threshold)
            
            logger.info(
                f"Identified {len(suspicious_df)} suspicious commenters "
                f"out of {df.select(pl.col('user_id').n_unique()).item()} total users"
            )
            
            return suspicious_df
            
        except Exception as e:
            logger.error(f"Error identifying suspicious commenters: {e}")
            raise
    
    def load_suspicious_comments_text(
        self, 
        suspicious_users: pl.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> pl.DataFrame:
        """Load actual comment text for suspicious users."""
        start_date = start_date or '2024-10-01'  # Default to recent month
        end_date = end_date or '2024-10-31'
        
        try:
            user_ids = suspicious_users.select(pl.col('user_id').unique()).to_numpy().flatten()
            user_ids_str = ','.join(map(str, user_ids))
            
            logger.info(f"Loading comment text for {len(user_ids)} suspicious users")
            
            comments_df = get_data_from_db(
                "queries/comments_by_commenter.sql",
                start_date=start_date,
                end_date=end_date,
                user_ids=user_ids_str
            )
            
            # Validate comment text data
            self.validator.validate_comment_text_data(comments_df)
            
            logger.info(f"Loaded {len(comments_df)} comments for analysis")
            return comments_df
            
        except Exception as e:
            logger.error(f"Error loading comment text: {e}")
            raise
    
    def analyze_comment_patterns(self, comments_df: pl.DataFrame) -> pl.DataFrame:
        """Analyze comment patterns using ML clustering."""
        try:
            if len(comments_df) == 0:
                logger.warning("No comments to analyze")
                return comments_df
            
            logger.info("Starting comment pattern analysis")
            
            # Analyze comments and add cluster information
            enhanced_df, embedding_result = self.comment_analyzer.analyze_comment_clusters(
                comments_df, text_column='text'
            )
            
            logger.info(f"Comment analysis completed. PCA explained variance: {embedding_result.pca_explained_variance:.3f}")
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error in comment pattern analysis: {e}")
            raise
    
    def generate_report(
        self, 
        df: pl.DataFrame, 
        comments_df: Optional[pl.DataFrame] = None,
        output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            logger.info("Generating analysis report")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Generate visualizations
            visualization_files = save_visualization_report(df, output_dir)
            
            # Save suspicious commenter list
            suspicious_df = df.filter(pl.col('max_CPP_this_CH') >= self.config.suspicious_threshold)
            if len(suspicious_df) > 0:
                suspicious_users_file = output_path / "suspicious_commenters.csv"
                suspicious_df.select(['username', 'user_id', 'max_CPP_this_CH']).write_csv(
                    suspicious_users_file
                )
                visualization_files['suspicious_users'] = "suspicious_commenters.csv"
            
            # Save clustered comments if available
            if comments_df is not None and 'cluster_label' in comments_df.columns:
                clustered_comments_file = output_path / "clustered_comments.csv"
                comments_df.write_csv(clustered_comments_file)
                visualization_files['clustered_comments'] = "clustered_comments.csv"
                
                # Save cluster samples
                cluster_samples = {}
                for cluster_id in range(self.comment_analyzer.processor.n_clusters or 10):
                    samples = self.comment_analyzer.get_cluster_samples(
                        comments_df, cluster_id, n_samples=5
                    )
                    if samples:
                        cluster_samples[f"cluster_{cluster_id}"] = samples
                
                # Save cluster samples as JSON
                import json
                cluster_samples_file = output_path / "cluster_samples.json"
                with open(cluster_samples_file, 'w', encoding='utf-8') as f:
                    json.dump(cluster_samples, f, ensure_ascii=False, indent=2)
                visualization_files['cluster_samples'] = "cluster_samples.json"
            
            # Create execution log
            log_file = output_path / "execution_log.txt"
            with open(log_file, 'w') as f:
                f.write(f"Bot Detection Analysis Report\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Configuration: {self.config}\n")
                f.write(f"Database: {db_config.host}:{db_config.port}/{db_config.database}\n")
                f.write(f"Files generated: {list(visualization_files.keys())}\n")
            
            logger.info(f"Analysis report generated in {output_dir}")
            return {
                'output_directory': output_dir,
                'files_generated': visualization_files,
                'analysis_summary': {
                    'total_records': len(df),
                    'suspicious_users': len(suspicious_df) if len(suspicious_df) > 0 else 0,
                    'comments_analyzed': len(comments_df) if comments_df is not None else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def run_full_analysis(
        self,
        start_date: str = None,
        end_date: str = None,
        include_text_analysis: bool = True,
        output_dir: str = "analysis_output"
    ) -> Dict[str, Any]:
        """Run the complete bot detection analysis pipeline."""
        try:
            logger.info("Starting full bot detection analysis")
            
            # Validate database
            if not self.validate_database_connection():
                raise RuntimeError("Database validation failed")
            
            # Load comment data
            comment_data = self.load_comment_data(start_date, end_date)
            
            # Identify suspicious commenters
            suspicious_commenters = self.identify_suspicious_commenters(comment_data)
            
            comments_with_text = None
            if include_text_analysis and len(suspicious_commenters) > 0:
                # Load comment text for suspicious users
                comments_with_text = self.load_suspicious_comments_text(suspicious_commenters)
                
                # Analyze comment patterns
                if len(comments_with_text) > 0:
                    comments_with_text = self.analyze_comment_patterns(comments_with_text)
            
            # Generate report
            report = self.generate_report(comment_data, comments_with_text, output_dir)
            
            logger.info("Full analysis completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
            raise


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Comment Bot Detection")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    parser.add_argument("--no-text-analysis", action="store_true", help="Skip text analysis")
    parser.add_argument("--threshold", type=int, default=10, help="Suspicious activity threshold")
    
    args = parser.parse_args()
    
    try:
        # Create bot detector with custom threshold if provided
        config_override = {}
        if args.threshold != 10:
            config_override['suspicious_threshold'] = args.threshold
        
        detector = BotDetector(config_override)
        
        # Run analysis
        result = detector.run_full_analysis(
            start_date=args.start_date,
            end_date=args.end_date,
            include_text_analysis=not args.no_text_analysis,
            output_dir=args.output_dir
        )
        
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {result['output_directory']}")
        print(f"Files generated: {len(result['files_generated'])}")
        print(f"Summary: {result['analysis_summary']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()