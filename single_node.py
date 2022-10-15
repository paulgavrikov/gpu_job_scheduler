import multiprocessing as mp
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

def gpu_worker(gpu, q):
    while not q.empty():
        cmd = q.get()
        logging.info(f"Executing {cmd} on GPU {gpu}")
        status = os.system(cmd)
        if status < 0:
            logging.error(f"{cmd} FAILED with status {status}")
        else:
            logging.info(f"{cmd} finished with status {status}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Schedule parallel processes on different GPUs.")

    parser.add_argument("command_file", type=str, help="File containing commands to run.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma separated list of GPUs to use.")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of workers per GPU.")

    args = parser.parse_args()

    que = mp.Queue()
    commands = 0

    with open(args.command_file, "r") as f:
        for line in f.readlines():
            que.put(line.strip())
            commands += 1

    logging.info(f"Queued {commands} commands from {args.command_file}")

    processes = []
    for gpu in args.gpus.split(",") * args.workers_per_gpu:
        logging.info("Starting worker on GPU %s", gpu)
        p = mp.Process(target=gpu_worker, args=(gpu, que), daemon=True)
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except:
        for p in processes:
            p.kill()
    logging.info("Done")
