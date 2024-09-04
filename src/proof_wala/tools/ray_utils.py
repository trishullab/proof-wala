import time
import os
import tempfile
import logging


def ignore_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return None
    return wrapper


def list_ray_logs(user_ray_dir: str, logger: logging.Logger = None):
    logger = logger if logger else logging.getLogger(__name__)
    logs = []
    if os.path.exists(user_ray_dir):
        dirs = ignore_exception(os.listdir)(user_ray_dir)
        if dirs is None:
            return logs
        for f in dirs:
            if f.startswith('session_'):
                log_dir = os.path.join(user_ray_dir, f, 'logs')
                if os.path.exists(log_dir):
                    log_dirs = ignore_exception(os.listdir)(log_dir)
                    if log_dirs is None:
                        continue
                    for log_file in log_dirs:
                        if "worker-" in log_file:
                            log_file_path = os.path.join(log_dir, log_file)
                            is_file = ignore_exception(os.path.isfile)(log_file_path)
                            if is_file is None:
                                continue
                            if is_file:
                                logs.append(log_file_path)
    return logs

def clean_ray_logs(logger: logging.Logger = None):
    logger = logger if logger else logging.getLogger(__name__)
    user_ray_dir = os.path.join(tempfile.gettempdir(), 'ray')
    size_cleaned = 0
    if os.path.exists(user_ray_dir):
        logger.info(f"Cleaning up Ray logs in {user_ray_dir}")
        all_logs = list_ray_logs(user_ray_dir)
        for log_file_path in all_logs:
            is_file = ignore_exception(os.path.isfile)(log_file_path)
            if is_file is None:
                continue
            if is_file:
                # Delete the log file if it is older than 16 minutes
                file_time = ignore_exception(os.path.getmtime)(log_file_path)
                if file_time is None:
                    continue
                if time.time() - file_time > 16 * 60:
                    file_size = ignore_exception(os.path.getsize)(log_file_path)
                    if file_size is None:
                        continue
                    ignore_exception(os.remove)(log_file_path)
                    size_cleaned += file_size
        logger.info(f"Cleaned up {size_cleaned/1024/1024} MB of Ray logs")

if __name__ == "__main__":
    # log to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    clean_ray_logs()