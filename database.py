"""Database operations for bot detection system."""

import mysql.connector
from mysql.connector import Error, pooling
import polars as pl
from jinja2 import Template
from typing import Any, Dict, List, Optional, Union
import logging
from contextlib import contextmanager
from config import db_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database manager with connection pooling."""
        self.config = config or {
            'host': db_config.host,
            'user': db_config.user,
            'password': db_config.password,
            'database': db_config.database,
            'port': db_config.port
        }
        
        # Create connection pool
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="bot_detection_pool",
                pool_size=5,
                pool_reset_session=True,
                **self.config
            )
            logger.info("Database connection pool created successfully")
        except Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        connection = None
        try:
            connection = self.pool.get_connection()
            if connection.is_connected():
                yield connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute a query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if cursor.description:  # SELECT query
                    results = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return results, columns
                else:  # INSERT/UPDATE/DELETE
                    conn.commit()
                    return cursor.rowcount, []
                    
            except Error as e:
                logger.error(f"Query execution error: {e}")
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def execute_template_query(self, template_path: str, **kwargs) -> pl.DataFrame:
        """Execute a templated SQL query and return Polars DataFrame."""
        try:
            # Load and render template
            with open(template_path, 'r', encoding='utf-8') as f:
                template = Template(f.read())
            
            sql = template.render(**kwargs)
            logger.info(f"Executing query from template: {template_path}")
            logger.debug(f"Rendered SQL: {sql}")
            
            # Execute query
            results, columns = self.execute_query(sql)
            
            # Convert to Polars DataFrame
            if results:
                df = pl.DataFrame(results, schema=columns)
                logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns")
                return df
            else:
                logger.warning("Query returned no results")
                return pl.DataFrame()
                
        except Exception as e:
            logger.error(f"Error executing template query {template_path}: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate database connection."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                return result is not None
        except Error:
            return False
    
    def get_table_info(self, table_name: str) -> pl.DataFrame:
        """Get information about a table structure."""
        try:
            query = f"DESCRIBE {table_name}"
            results, columns = self.execute_query(query)
            return pl.DataFrame(results, schema=columns)
        except Error as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


def get_data_from_db(sql_template: str, **kwargs) -> pl.DataFrame:
    """
    Convenience function for backwards compatibility.
    Execute a templated SQL query and return Polars DataFrame.
    """
    return db_manager.execute_template_query(sql_template, **kwargs)


def validate_sql_injection(query: str) -> bool:
    """Basic SQL injection prevention check."""
    dangerous_keywords = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
        '--', '/*', '*/', ';', 'UNION', 'EXEC', 'EXECUTE'
    ]
    
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
            return False
    return True


def safe_execute_query(template_path: str, **kwargs) -> pl.DataFrame:
    """
    Safely execute a query with additional validation.
    """
    # Validate template path
    if not template_path.endswith('.sql'):
        raise ValueError("Template path must end with .sql")
    
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Basic validation
        if not validate_sql_injection(template_content):
            raise ValueError("Template contains potentially dangerous SQL")
        
        return db_manager.execute_template_query(template_path, **kwargs)
    
    except Exception as e:
        logger.error(f"Safe query execution failed: {e}")
        raise