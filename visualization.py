"""Visualization utilities for bot detection analysis."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BotDetectionVisualizer:
    """Creates visualizations for bot detection analysis."""
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default styling
        self.default_width = 800
        self.default_height = 400
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_cumulative_comments_distribution(
        self, 
        df: pl.DataFrame, 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot cumulative distribution of comments by user activity level.
        """
        try:
            # Prepare data for cumulative distribution
            plot_data = (
                df.sort('number_comments_all_time', descending=False)
                .with_columns(
                    (pl.col('number_comments_all_time').cum_sum() / 
                     pl.col('number_comments_all_time').sum()).alias('cumshare')
                )
                .group_by('number_comments_all_time')
                .agg(pl.col('cumshare').last())
            )
            
            fig = px.scatter(
                plot_data.to_pandas(),
                x='number_comments_all_time', 
                y='cumshare',
                width=self.default_width, 
                height=self.default_height,
                title="Cumulative Share of Comments by Number of Posted Comments",
                labels={
                    'number_comments_all_time': 'Number of Comments Posted',
                    'cumshare': 'Cumulative Share'
                }
            )
            
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(
                xaxis_title="Number of Comments Posted",
                yaxis_title="Cumulative Share",
                showlegend=False
            )
            
            if save_path:
                fig.write_html(self.output_dir / save_path)
                logger.info(f"Saved cumulative distribution plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cumulative distribution plot: {e}")
            raise
    
    def plot_suspicious_commenter_heatmap(
        self, 
        df: pl.DataFrame, 
        max_periods: int = 10,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create density heatmap of suspicious commenters.
        """
        try:
            # Prepare data for heatmap
            heatmap_data = (
                df.group_by(['number_comments_all_time', 'number_periods_with_comments'])
                .agg(pl.len().alias('number_commenters'))
                .sort('number_periods_with_comments')
                .filter(pl.col('number_periods_with_comments') < max_periods)
            )
            
            fig = px.density_heatmap(
                heatmap_data.to_pandas(),
                x="number_comments_all_time", 
                y="number_periods_with_comments", 
                z="number_commenters",
                title="Distribution of Suspicious Commenters",
                labels={
                    'number_comments_all_time': 'Number of Comments (All Time)',
                    'number_periods_with_comments': 'Number of Active Periods',
                    'number_commenters': 'Number of Commenters'
                }
            )
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height
            )
            
            if save_path:
                fig.write_html(self.output_dir / save_path)
                logger.info(f"Saved heatmap to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise
    
    def plot_cluster_tsne(
        self, 
        df: pl.DataFrame, 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot t-SNE visualization of comment clusters.
        """
        try:
            fig = px.scatter(
                df.to_pandas(),
                x='tsne_x', 
                y='tsne_y', 
                color='cluster_label',
                title="t-SNE Visualization of Comment Clusters",
                labels={
                    'tsne_x': 't-SNE Dimension 1',
                    'tsne_y': 't-SNE Dimension 2',
                    'cluster_label': 'Cluster'
                },
                width=900,
                height=600
            )
            
            fig.update_traces(marker=dict(size=3, opacity=0.7))
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            if save_path:
                fig.write_html(self.output_dir / save_path)
                logger.info(f"Saved t-SNE plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating t-SNE plot: {e}")
            raise
    
    def plot_cluster_statistics(
        self, 
        df: pl.DataFrame, 
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot statistics for each cluster.
        """
        try:
            # Calculate cluster statistics
            cluster_stats = (
                df.group_by('cluster_label')
                .agg([
                    pl.len().alias('count'),
                    pl.col('number_comments_all_time').mean().alias('avg_comments'),
                    pl.col('max_CPP_this_CH').mean().alias('avg_max_cpp'),
                    pl.col('mean_likes_per_comment').mean().alias('avg_likes_per_comment')
                ])
                .sort('cluster_label')
            )
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Comments Count by Cluster',
                    'Average Comments per User',
                    'Average Max Comments per Period',
                    'Average Likes per Comment'
                ]
            )
            
            cluster_data = cluster_stats.to_pandas()
            
            # Add traces
            fig.add_trace(
                go.Bar(x=cluster_data['cluster_label'], y=cluster_data['count'],
                       name='Count', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=cluster_data['cluster_label'], y=cluster_data['avg_comments'],
                       name='Avg Comments', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=cluster_data['cluster_label'], y=cluster_data['avg_max_cpp'],
                       name='Avg Max CPP', marker_color='lightcoral'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=cluster_data['cluster_label'], y=cluster_data['avg_likes_per_comment'],
                       name='Avg Likes/Comment', marker_color='lightyellow'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                width=1000,
                title_text="Cluster Statistics Overview",
                showlegend=False
            )
            
            if save_path:
                fig.write_html(self.output_dir / save_path)
                logger.info(f"Saved cluster statistics plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster statistics plot: {e}")
            raise
    
    def plot_time_series_activity(
        self, 
        df: pl.DataFrame,
        date_column: str = 'period',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot time series of comment activity.
        """
        try:
            # Aggregate by time period
            time_series = (
                df.group_by(date_column)
                .agg([
                    pl.len().alias('total_comments'),
                    pl.col('user_id').n_unique().alias('unique_users')
                ])
                .sort(date_column)
            )
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Total Comments Over Time', 'Unique Users Over Time'],
                shared_xaxes=True
            )
            
            time_data = time_series.to_pandas()
            
            fig.add_trace(
                go.Scatter(
                    x=time_data[date_column], 
                    y=time_data['total_comments'],
                    mode='lines+markers',
                    name='Total Comments',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_data[date_column], 
                    y=time_data['unique_users'],
                    mode='lines+markers',
                    name='Unique Users',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                width=1000,
                title_text="Comment Activity Time Series"
            )
            
            if save_path:
                fig.write_html(self.output_dir / save_path)
                logger.info(f"Saved time series plot to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            raise
    
    def create_summary_report(
        self, 
        df: pl.DataFrame,
        suspicious_threshold: int = 10
    ) -> Dict[str, Any]:
        """
        Create a summary report of the analysis.
        """
        try:
            total_users = df.select(pl.col('user_id').n_unique()).item()
            total_comments = df.select(pl.col('number_comments_all_time').sum()).item()
            
            suspicious_users = df.filter(
                pl.col('max_CPP_this_CH') >= suspicious_threshold
            )
            n_suspicious = len(suspicious_users)
            
            # Cluster distribution if available
            cluster_dist = {}
            if 'cluster_label' in df.columns:
                cluster_counts = (
                    df.group_by('cluster_label')
                    .agg(pl.len().alias('count'))
                    .sort('cluster_label')
                )
                cluster_dist = dict(zip(
                    cluster_counts.select('cluster_label').to_numpy().flatten(),
                    cluster_counts.select('count').to_numpy().flatten()
                ))
            
            summary = {
                'total_users': total_users,
                'total_comments': total_comments,
                'suspicious_users': n_suspicious,
                'suspicious_percentage': (n_suspicious / total_users) * 100 if total_users > 0 else 0,
                'avg_comments_per_user': total_comments / total_users if total_users > 0 else 0,
                'cluster_distribution': cluster_dist,
                'top_commenters': (
                    df.sort('number_comments_all_time', descending=True)
                    .head(10)
                    .select(['username', 'number_comments_all_time', 'max_CPP_this_CH'])
                    .to_pandas()
                    .to_dict('records')
                )
            }
            
            logger.info("Summary report generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            raise


def save_visualization_report(
    df: pl.DataFrame, 
    output_dir: str = "analysis_output"
) -> Dict[str, str]:
    """
    Generate and save a complete visualization report.
    """
    visualizer = BotDetectionVisualizer(output_dir)
    saved_files = {}
    
    try:
        # Cumulative distribution
        fig1 = visualizer.plot_cumulative_comments_distribution(
            df, save_path="cumulative_distribution.html"
        )
        saved_files['cumulative_distribution'] = "cumulative_distribution.html"
        
        # Suspicious commenter heatmap
        suspicious_df = df.filter(pl.col('max_CPP_this_CH') >= 10)
        if len(suspicious_df) > 0:
            fig2 = visualizer.plot_suspicious_commenter_heatmap(
                suspicious_df, save_path="suspicious_heatmap.html"
            )
            saved_files['suspicious_heatmap'] = "suspicious_heatmap.html"
        
        # Cluster visualizations if cluster data exists
        if 'cluster_label' in df.columns:
            fig3 = visualizer.plot_cluster_tsne(
                df, save_path="cluster_tsne.html"
            )
            saved_files['cluster_tsne'] = "cluster_tsne.html"
            
            fig4 = visualizer.plot_cluster_statistics(
                df, save_path="cluster_statistics.html"
            )
            saved_files['cluster_statistics'] = "cluster_statistics.html"
        
        # Summary report
        summary = visualizer.create_summary_report(df)
        
        # Save summary as JSON
        import json
        summary_path = visualizer.output_dir / "summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['summary'] = "summary_report.json"
        
        logger.info(f"Visualization report saved to {output_dir}")
        return saved_files
        
    except Exception as e:
        logger.error(f"Error generating visualization report: {e}")
        raise