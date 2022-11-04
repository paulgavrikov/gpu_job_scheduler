import multiprocessing as mp
import os
import argparse
import logging
import coloredlogs
coloredlogs.install(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def gpu_worker(i, gpu, q):
    logger = logging.getLogger(f"Worker-{i}@cuda:{gpu}")
    while not q.empty():
        cmd = q.get().replace("%gpu%", str(gpu))
        logger.info(f"Executing {cmd} on GPU {gpu}")
        status = os.system(cmd)
        if status < 0:
            logger.error(f"{cmd} FAILED with status {status}")
        else:
            logger.info(f"{cmd} finished with status {status}")


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
    for i, gpu in enumerate(args.gpus.split(",") * args.workers_per_gpu):
        logging.info("Starting worker on GPU %s", gpu)
        p = mp.Process(target=gpu_worker, args=(i, gpu, que), daemon=True)
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except:
        for p in processes:
            p.kill()
    logging.info("Done")
