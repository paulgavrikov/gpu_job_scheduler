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
        logger.info(f"{cmd}")
        status = os.system(cmd)
        if status < 0:
            logger.error(f"FAILED (status: {status})")
        else:
            logger.info(f"OKAY (status: {status})")
    logger.warning("shutting down - queue empty")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Schedule parallel processes on different GPUs.")

    parser.add_argument("command_file", type=str, help="File containing commands to run.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma separated list of GPUs to use.")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of workers per GPU.")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Shuffle commands.")

    args = parser.parse_args()

    que = mp.Queue()
    num_commands = 0
    commands = []

    with open(args.command_file, "r") as f:
        for line in f.readlines():
            commands.append(line.strip())
            num_commands += 1
            
    if args.shuffle:
        import random
        random.shuffle(commands)

    list(map(que.put, commands))
    logging.info(f"Queued {num_commands} commands from {args.command_file}")

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
