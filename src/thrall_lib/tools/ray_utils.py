import time
import os
import tempfile
import logging

def clean_ray_logs(logger: logging.Logger = None):
    logger = logger if logger else logging.getLogger(__name__)
    user_ray_dir = os.path.join(tempfile.gettempdir(), 'ray')
    size_cleaned = 0
    if os.path.exists(user_ray_dir):
        logger.info(f"Cleaning up Ray logs in {user_ray_dir}")
        for f in os.listdir(user_ray_dir):
            if f.startswith('session_'):
                # Delete all the log files which are older than 30 minutes
                log_dir = os.path.join(user_ray_dir, f, 'logs')
                if os.path.exists(log_dir):
                    for log_file in os.listdir(log_dir):
                        try:
                            if "worker-" in log_file:
                                log_file_path = os.path.join(log_dir, log_file)
                                if os.path.isfile(log_file_path):
                                    # Delete the log file if it is older than 30 minutes
                                    if time.time() - os.path.getmtime(log_file_path) > 30 * 60:
                                        file_size = os.path.getsize(log_file_path)
                                        os.remove(log_file_path)
                                        size_cleaned += file_size
                        except Exception as e:
                            pass
        logger.info(f"Cleaned up {size_cleaned/1024/1024} MB of Ray logs")

if __name__ == "__main__":
    # log to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    clean_ray_logs()