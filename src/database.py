"""
Database utilities for PostgreSQL.
Handles image storage and retrieval for retraining.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
import io
from dotenv import load_dotenv

# Database connection pool
_connection_pool = None


def get_db_connection_string() -> str:
    """
    Get database connection string from environment variable.
    
    Returns:
        Database connection string
    """
    # read it from the .env file
    load_dotenv()
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Please provide PostgreSQL connection string."
        )
    return db_url


def init_db_pool():
    """Initialize database connection pool."""
    global _connection_pool
    if _connection_pool is None:
        conn_string = get_db_connection_string()
        _connection_pool = SimpleConnectionPool(1, 20, conn_string)
    return _connection_pool


def get_db_connection():
    """Get a database connection from the pool."""
    pool = init_db_pool()
    return pool.getconn()


def return_db_connection(conn):
    """Return a connection to the pool."""
    pool = init_db_pool()
    pool.putconn(conn)


def create_tables():
    """Create database tables if they don't exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_images (
                id SERIAL PRIMARY KEY,
                image_data BYTEA NOT NULL,
                class_name VARCHAR(100) NOT NULL,
                filename VARCHAR(255),
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_for_training BOOLEAN DEFAULT FALSE,
                training_run_id INTEGER
            )
        """)
        
        # Create training runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status VARCHAR(50),
                metrics JSONB,
                model_path VARCHAR(500)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_class_name 
            ON training_images(class_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_used_for_training 
            ON training_images(used_for_training)
        """)
        
        conn.commit()
        print("Database tables created successfully")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def save_image_to_db(image_bytes: bytes, class_name: str, filename: str) -> int:
    """
    Save an image to the database.
    
    Args:
        image_bytes: Image file bytes
        class_name: Waste category class name
        filename: Original filename
    
    Returns:
        ID of the inserted record
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_images (image_data, class_name, filename)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (psycopg2.Binary(image_bytes), class_name, filename))
        
        image_id = cursor.fetchone()[0]
        conn.commit()
        return image_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def save_images_batch(images_data: List[Tuple[bytes, str, str]]) -> List[int]:
    """
    Save multiple images to the database in a batch.
    
    Args:
        images_data: List of (image_bytes, class_name, filename) tuples
    
    Returns:
        List of inserted record IDs
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Prepare data for batch insert
        values = [
            (psycopg2.Binary(img_bytes), class_name, filename)
            for img_bytes, class_name, filename in images_data
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO training_images (image_data, class_name, filename)
            VALUES %s
            RETURNING id
            """,
            values
        )
        
        ids = [row[0] for row in cursor.fetchall()]
        conn.commit()
        return ids
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def get_example_image_for_class(class_name: str, limit: int = 1) -> Optional[bytes]:
    """
    Get an example image for a specific class.
    
    Args:
        class_name: Class name to get example for
        limit: Number of examples to return (default: 1)
    
    Returns:
        Image bytes if found, None otherwise
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_data
            FROM training_images
            WHERE class_name = %s
            ORDER BY uploaded_at DESC
            LIMIT %s
        """, (class_name, limit))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        return None
    except Exception as e:
        raise e
    finally:
        return_db_connection(conn)


def get_images_for_training(class_name: Optional[str] = None) -> List[Tuple[int, bytes, str]]:
    """
    Retrieve images from database for training.
    
    Args:
        class_name: Optional filter by class name
    
    Returns:
        List of (id, image_bytes, class_name) tuples
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        if class_name:
            cursor.execute("""
                SELECT id, image_data, class_name
                FROM training_images
                WHERE class_name = %s
                ORDER BY uploaded_at DESC
            """, (class_name,))
        else:
            cursor.execute("""
                SELECT id, image_data, class_name
                FROM training_images
                ORDER BY uploaded_at DESC
            """)
        
        results = cursor.fetchall()
        return [(row[0], bytes(row[1]), row[2]) for row in results]
    finally:
        return_db_connection(conn)


def mark_images_as_used(image_ids: List[int], training_run_id: int):
    """
    Mark images as used for a specific training run.
    
    Args:
        image_ids: List of image IDs
        training_run_id: Training run ID
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE training_images
            SET used_for_training = TRUE, training_run_id = %s
            WHERE id = ANY(%s)
        """, (training_run_id, image_ids))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def create_training_run() -> int:
    """
    Create a new training run record.
    
    Returns:
        Training run ID
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_runs (status)
            VALUES ('started')
            RETURNING id
        """)
        
        run_id = cursor.fetchone()[0]
        conn.commit()
        return run_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def update_training_run(run_id: int, status: str, metrics: dict = None, model_path: str = None):
    """
    Update training run with completion status and metrics.
    
    Args:
        run_id: Training run ID
        status: Status ('completed', 'failed', etc.)
        metrics: Training metrics dictionary
        model_path: Path to saved model
    """
    conn = get_db_connection()
    try:
        import json
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute("""
                UPDATE training_runs
                SET status = %s, completed_at = CURRENT_TIMESTAMP,
                    metrics = %s, model_path = %s
                WHERE id = %s
            """, (status, json.dumps(metrics), model_path, run_id))
        else:
            cursor.execute("""
                UPDATE training_runs
                SET status = %s, completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (status, run_id))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)


def get_training_runs(limit: int = 10) -> List[dict]:
    """
    Get list of training runs, ordered by most recent first.
    
    Args:
        limit: Maximum number of runs to return (default: 10)
    
    Returns:
        List of training run dictionaries with id, started_at, completed_at, 
        status, metrics, and model_path
    """
    conn = get_db_connection()
    try:
        import json
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, started_at, completed_at, status, metrics, model_path
            FROM training_runs
            ORDER BY started_at DESC
            LIMIT %s
        """, (limit,))
        
        runs = []
        for row in cursor.fetchall():
            # Handle metrics - JSONB is already deserialized by psycopg2 to a dict
            metrics = row[4]
            if metrics is not None and isinstance(metrics, str):
                # If it's a string (unlikely with JSONB), parse it
                metrics = json.loads(metrics)
            # If it's already a dict (normal case), use it directly
            
            run_dict = {
                'id': row[0],
                'started_at': row[1].isoformat() if row[1] else None,
                'completed_at': row[2].isoformat() if row[2] else None,
                'status': row[3],
                'metrics': metrics,
                'model_path': row[5]
            }
            runs.append(run_dict)
        
        return runs
    except Exception as e:
        raise e
    finally:
        return_db_connection(conn)

