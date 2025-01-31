if __name__ == "__main__":
    # Start the ray cluster
    from filelock import FileLock
    import json
    import os
    import ray
    import logging
    import time
    import sys
    root_dir = f"{__file__.split('proof_wala')[0]}"
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.makedirs(".log/locks", exist_ok=True)
    ray_was_started = False
    pid = os.getpid()
    print("Initializing Ray")
    print("PID: ", pid)
    with FileLock(".log/locks/ray.lock"):
        if os.path.exists(".log/ray/session_latest"):
            with open(".log/ray/session_latest", "r") as f:
                ray_session = f.read()
                ray_session = json.loads(ray_session)
            ray_address = ray_session["address"]
            ray.init(address=ray_address)
            print("Ray was already started")
            print("Ray session: ", ray_session)
        else:
            ray_session = ray.init(
                num_cpus=100, 
                object_store_memory=150*2**30, 
                _memory=300*2**30, 
                logging_level=logging.CRITICAL, 
                ignore_reinit_error=False, 
                log_to_driver=False, 
                configure_logging=False,
                _system_config={"metrics_report_interval_ms": 3*10**8})
            ray_session = dict(ray_session)
            ray_session["main_pid"] = pid
            print("Ray session: ", ray_session)
            with open(".log/ray/session_latest", "w") as f:
                f.write(json.dumps(ray_session))
            ray_was_started = True
            print("Ray was started")
            print("Ray session: ", ray_session)
    # Flush the stdout buffer
    sys.stdout.flush()
    while ray_was_started:
        # Keep the ray cluster alive till killed
        time.sleep(10000)