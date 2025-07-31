from polysim import GrpcSimulationClient
from polymetis.robot_client.metadata import RobotClientMetadata
from utils.environment import Environment

import hydra

import os
import time
import logging
import subprocess
import atexit
import sys
import signal

import torchcontrol as toco

from polymetis.utils.grpc_utils import check_server_exists
from polymetis.utils.data_dir import get_full_path_to_urdf
from polymetis.utils.data_dir import BUILD_DIR, which

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="launch_robot")
def main(cfg):
    # Initialize simulation server
    log.info(f"Adding {BUILD_DIR} to $PATH")
    os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ["PATH"]

    # Check if another server is alive on address
    assert not check_server_exists(
        cfg.ip, cfg.port
    ), "Port unavailable; possibly another server found on designated address. To prevent undefined behavior, start the service on a different port or kill stale servers with 'pkill -9 run_server'"

    # Parse server address
    ip = str(cfg.ip)
    port = str(cfg.port)

    # Start server
    log.info(f"Starting server")
    server_exec_path = which(cfg.server_exec)
    server_cmd = [server_exec_path]
    server_cmd = server_cmd + ["-s", ip, "-p", port]

    if cfg.use_real_time:
        log.info(f"Acquiring sudo...")
        subprocess.run(["sudo", "echo", '"Acquired sudo."'], check=True)

        server_cmd = ["sudo", "-s", "env", '"PATH=$PATH"'] + server_cmd + ["-r"]
    server_output = subprocess.Popen(
        server_cmd, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=os.setpgrp
    )
    pgid = os.getpgid(server_output.pid)

    # Kill process at the end
    if cfg.use_real_time:

        def cleanup():
            log.info(
                f"Using sudo to kill subprocess with pid {server_output.pid}, pgid {pgid}..."
            )
            # send NEGATIVE of process group ID to kill process tree
            subprocess.check_call(["sudo", "kill", "-9", f"-{pgid}"])

    else:

        def cleanup():
            log.info(f"Killing subprocess with pid {server_output.pid}, pgid {pgid}...")
            subprocess.check_call(["kill", "-9", f"-{pgid}"])

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda signal_number, stack_frame: cleanup())

    # Get data for simulation environment
    object_centers = {"HUMAN_CENTER": [0.5, -0.55, 0.9], "LAPTOP_CENTER": [-0.7929, -0.1, 0.0]}
    robot_model_cfg = cfg.robot_model

    robot_description_path = get_full_path_to_urdf(
            robot_model_cfg.robot_description_path
        )

    robot_model = toco.models.RobotModelPinocchio(
        urdf_filename=robot_description_path,
        ee_link_name=robot_model_cfg.ee_link_name
    )

    if cfg.robot_client:
        t0 = time.time()
        while not check_server_exists(cfg.ip, cfg.port):
            time.sleep(0.1)
            if time.time() - t0 > cfg.timeout:
                raise ConnectionError("Robot client: Unable to locate server.")

    try:
        gui = cfg.gui
        use_grav_comp = cfg.use_grav_comp
        env = Environment(robot_model_cfg=robot_model_cfg, object_centers=object_centers, robot_model=robot_model, gui=gui, use_grav_comp=use_grav_comp)
        metadata_cfg = cfg.robot_client.metadata_cfg

        # Start simulation client
        log.info(f"Simulation client is being used...")
        sim = GrpcSimulationClient(
            env=env,
            metadata_cfg=metadata_cfg,
            ip="localhost"
        )
        sim.run()

    except:
        # Start hardware client
        log.info("Hardware client is being used...")
        client = hydra.utils.instantiate(cfg.robot_client)
        client.run()
    
if __name__ == "__main__":
    main()