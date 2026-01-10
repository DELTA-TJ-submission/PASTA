#!/usr/bin/env python3
"""Task manager for asynchronous prediction pipeline."""

import threading
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import traceback


@dataclass
class PredictionTask:
    """Represents a prediction task with its current state."""
    task_id: str
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    current_step: str = ""
    result_paths: Dict[str, str] = field(default_factory=dict)
    error_message: str = ""
    wsi_filename: str = ""
    config: Dict = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class TaskManager:
    """Manages asynchronous prediction tasks."""
    
    def __init__(self, max_workers: int = 2):
        """
        Initialize task manager.
        
        Args:
            max_workers: Maximum number of concurrent tasks
        """
        self.tasks: Dict[str, PredictionTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
    
    def create_task(self, wsi_filename: str, config: Dict) -> str:
        """
        Create a new prediction task.
        
        Args:
            wsi_filename: Name of the WSI file
            config: Configuration dictionary
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())[:8]
        
        with self.lock:
            task = PredictionTask(
                task_id=task_id,
                wsi_filename=wsi_filename,
                config=config,
                status="pending"
            )
            self.tasks[task_id] = task
        
        return task_id
    
    def submit_task(self, task_id: str, pipeline_func: Callable, *args, **kwargs):
        """
        Submit a task for asynchronous execution.
        
        Args:
            task_id: Task identifier
            pipeline_func: Function to execute
            *args, **kwargs: Arguments to pass to pipeline_func
        """
        def wrapped_execution():
            try:
                with self.lock:
                    self.tasks[task_id].status = "processing"
                    self.tasks[task_id].current_step = "Initializing..."
                
                # Execute the pipeline
                result = pipeline_func(task_id, *args, **kwargs)
                
                with self.lock:
                    self.tasks[task_id].status = "completed"
                    self.tasks[task_id].progress = 1.0
                    self.tasks[task_id].current_step = "Completed"
                    self.tasks[task_id].end_time = time.time()
                    if result:
                        self.tasks[task_id].result_paths = result
                        
            except Exception as e:
                with self.lock:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].error_message = str(e)
                    self.tasks[task_id].current_step = "Failed"
                    self.tasks[task_id].end_time = time.time()
                print(f"Task {task_id} failed: {e}")
                traceback.print_exc()
        
        self.executor.submit(wrapped_execution)
    
    def update_progress(self, task_id: str, progress: float, step: str):
        """
        Update task progress.
        
        Args:
            task_id: Task identifier
            progress: Progress value (0.0 to 1.0)
            step: Current step description
        """
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = progress
                self.tasks[task_id].current_step = step
    
    def get_task(self, task_id: str) -> Optional[PredictionTask]:
        """
        Get task by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            PredictionTask or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, PredictionTask]:
        """Get all tasks."""
        with self.lock:
            return dict(self.tasks)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Clean up old completed/failed tasks.
        
        Args:
            max_age_hours: Maximum age in hours for completed tasks
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        with self.lock:
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task.end_time and (current_time - task.end_time) > max_age_seconds:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
        
        return len(tasks_to_remove)
    
    def shutdown(self):
        """Shutdown the task manager and wait for all tasks to complete."""
        self.executor.shutdown(wait=True)


# Global task manager instance
_global_task_manager = None


def get_task_manager(max_workers: int = 2) -> TaskManager:
    """Get or create global task manager instance."""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = TaskManager(max_workers=max_workers)
    return _global_task_manager

