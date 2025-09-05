# YouTube Comment Bot Detection System

A comprehensive system for detecting bot behavior in YouTube comments using machine learning and statistical analysis.

## Features

- **Statistical Analysis**: Identifies suspicious commenting patterns based on frequency and timing
- **ML Clustering**: Uses transformer-based embeddings to cluster similar comments
- **Visualization**: Interactive plots and reports for analysis results
- **Scalable Architecture**: Modular design with proper separation of concerns
- **Database Optimization**: Efficient SQL queries with connection pooling
- **Comprehensive Logging**: Full audit trail of analysis processes

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd bot_buster

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

Ensure you have a MySQL database with the required tables:
- `youtube_comments`
- `youtube_users` 
- `youtube_videos`
- `youtube_channels`

Create indexes for optimal performance:
```sql
CREATE INDEX idx_comments_date_user ON youtube_comments(published_at, user_id);
CREATE INDEX idx_comments_video ON youtube_comments(video_id);
CREATE INDEX idx_videos_channel ON youtube_videos(id, channel_id);
CREATE INDEX idx_users_lookup ON youtube_users(id);
```

### 3. Configuration

Copy the environment template and configure your settings:
```bash
cp .env.template .env
# Edit .env with your database credentials
```

### 4. Run Analysis

#### Command Line
```bash
# Basic analysis
python bot_detector.py --start-date 2024-01-01 --end-date 2024-12-31

# With custom settings
python bot_detector.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --threshold 15 \
  --output-dir results_2024 \
  --no-text-analysis
```

#### Python API
```python
from bot_detector import BotDetector

# Initialize detector
detector = BotDetector()

# Run full analysis
results = detector.run_full_analysis(
    start_date='2024-01-01',
    end_date='2024-12-31',
    include_text_analysis=True,
    output_dir='analysis_results'
)

print(f"Analysis completed: {results['analysis_summary']}")
```

#### Jupyter Notebook
See [`improved_analysis.ipynb`](improved_analysis.ipynb) for an interactive analysis workflow.

## Architecture

### Core Components

- **[`config.py`](config.py)**: Configuration management with environment variable support
- **[`database.py`](database.py)**: Database operations with connection pooling and security
- **[`ml_processing.py`](ml_processing.py)**: Machine learning pipeline for comment analysis
- **[`visualization.py`](visualization.py)**: Comprehensive visualization utilities
- **[`data_validation.py`](data_validation.py)**: Data quality validation and error handling
- **[`bot_detector.py`](bot_detector.py)**: Main orchestrator tying everything together

### SQL Queries

- **[`queries/minutely_comments_optimized.sql`](queries/minutely_comments_optimized.sql)**: Optimized query for user activity analysis
- **[`queries/comments_by_commenter.sql`](queries/comments_by_commenter.sql)**: Query for retrieving comment text

## Configuration

### Environment Variables

Create a `.env` file with your database configuration:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=bot_buster
DB_PORT=3306
```

### Analysis Parameters

Customize analysis in [`config.py`](config.py):

```python
@dataclass
class AnalysisConfig:
    default_start_date: str = '2023-10-31'
    default_end_date: str = '2024-10-31'
    suspicious_threshold: int = 10  # Comments per period threshold
    user_id_mod: int = 100  # Sampling parameter
```

## Output

The system generates:

1. **HTML Visualizations**:
   - Cumulative comment distribution
   - Suspicious user heatmaps
   - t-SNE cluster visualizations
   - Time series analysis

2. **Data Files**:
   - `suspicious_commenters.csv`: List of identified suspicious users
   - `clustered_comments.csv`: Comments with cluster assignments
   - `cluster_samples.json`: Sample comments from each cluster

3. **Analysis Reports**:
   - `summary_report.json`: Comprehensive analysis summary
   - `execution_log.txt`: Detailed execution log

## Bot Detection Methodology

### Statistical Indicators
- **Comments Per Period (CPP)**: Unusual comment frequency in short time windows
- **Activity Patterns**: Concentration of activity in specific channels/periods
- **Engagement Metrics**: Like-to-comment and reply-to-comment ratios

### ML Analysis
- **Text Embeddings**: Russian BERT model for semantic analysis
- **Dimensionality Reduction**: PCA for efficient processing
- **Clustering**: K-means clustering to identify comment patterns
- **Visualization**: t-SNE for 2D visualization of comment clusters

## Performance Optimizations

1. **Database**:
   - Connection pooling
   - Optimized SQL queries with proper indexes
   - Batch processing for large datasets

2. **Machine Learning**:
   - Efficient batch processing
   - Memory-optimized embedding generation
   - Configurable model parameters

3. **Data Processing**:
   - Polars for fast DataFrame operations
   - Streaming processing for large datasets
   - Comprehensive data validation

## Development

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy .

# Run tests
pytest
```

### Extending the System

1. **Add New Detection Methods**: Implement new analysis methods in `ml_processing.py`
2. **Custom Visualizations**: Extend `visualization.py` with new plot types
3. **Database Support**: Add new database backends in `database.py`
4. **Configuration Options**: Extend configuration classes in `config.py`

## Troubleshooting

### Common Issues

1. **Database Connection**: Verify credentials in `.env` file
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Memory Issues**: Reduce batch size in model configuration
4. **Performance**: Ensure database indexes are created

### Logging

All operations are logged to:
- Console output (configurable level)
- `bot_detection.log` file
- Analysis-specific execution logs

## Security Considerations

- Database credentials via environment variables
- SQL injection prevention
- Input parameter validation
- Secure connection handling

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]