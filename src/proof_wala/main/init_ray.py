def main():
    # Start the ray cluster
    from filelock import FileLock
    import json
    import os
    import ray
    import logging
    import time
    import sys
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--num_cpus", type=int, default=100)
    argument_parser.add_argument("--object_store_memory", type=int, default=150*2**30)
    argument_parser.add_argument("--memory", type=int, default=300*2**30)
    argument_parser.add_argument("--metrics_report_interval_ms", type=int, default=3*10**8)
    args = argument_parser.parse_args()
    root_dir = f"{os.path.abspath(__file__).split('proof_wala')[-2]}"
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    os.makedirs(".log/locks", exist_ok=True)
    os.makedirs(".log/ray", exist_ok=True)
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
                num_cpus=args.num_cpus, 
                object_store_memory=args.object_store_memory, 
                _memory=args.memory, 
                logging_level=logging.CRITICAL, 
                ignore_reinit_error=False, 
                log_to_driver=False, 
                configure_logging=False,
                _system_config={"metrics_report_interval_ms": args.metrics_report_interval_ms})
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

if __name__ == "__main__":
    main()