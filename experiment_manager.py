import sys
import os
import io
import json
import subprocess
from datetime import datetime
from fabric import Connection
from patchwork import transfers
#import spython.main as singularity
from pathlib import Path

class system():
    def __init__(self, ssh_target, hostname, username, workdir, environment, use_slurm, ramdisk_path):
        self.ssh_target = ssh_target
        self.hostname = hostname
        self.username = username
        self.workdir = workdir
        self.environment = environment
        self.use_slurm = use_slurm
        self.ramdisk_path = ramdisk_path

    # add system-specific first-time setup
        # setup (passwordless) ssh (to server)
        # setup any required Github deploy keys + corresponding ssh config (from server to Github)
        # create workdir

class directory():
    def __init__(self, path, exclude=[]):
        self.path = path
        self.exclude = exclude

    def deploy(self, system, rel_path):
        print("Deploying dependency directory {} to target..".format(self.path))
        with Connection(system.ssh_target) as c:
            # manually pass ProxyCommand option to patchwork.transfers as it derives only user, host, and port from fabric.Connection
            proxy_cmd = c.ssh_config.get('proxycommand')
            ssh_opts = '-o \"ProxyCommand {}\"'.format(proxy_cmd) if proxy_cmd else ""
            transfers.rsync(c, ssh_opts = ssh_opts, rsync_opts="--progress", source = self.path, target = os.path.join(system.workdir, rel_path, ""), exclude=self.exclude)

class repository():
    def __init__(self, dir, remote, branch_commit):
        self.dir = dir
        self.remote = remote
        self.branch_commit = branch_commit

    def deploy(self, system, rel_path):
        print("Deploying dependency repo {} to target..".format(self.remote))
        with Connection(system.ssh_target) as c:
            result = c.run("cd {target_dir} && git clone {remote} {dir} && cd {dir} && git checkout {branch_commit}".format(
                            target_dir = os.path.join(system.workdir, rel_path), 
                            remote = self.remote, 
                            dir = self.dir, 
                            branch_commit=self.branch_commit))
            if not result.ok:
                print("Error")

    def deploy_local(self, local_path):
        # clone to a path on the local machine
        result = subprocess.run("cd {local_path} && git clone {remote} {dir} && cd {dir} && git checkout {branch_commit}".format(
                            local_path = local_path, 
                            remote = self.remote, 
                            dir = self.dir, 
                            branch_commit=self.branch_commit),
                            shell=True)
    

class experiment():
    def __init__(self, system, tag="experiment", dependencies=None, environment={}, cmd=None, slurm_options=None, use_ramdisk=False, copy_back=True):
        self.system = system
        self.tag = tag
        self.dependencies = dependencies
        self.environment = environment
        self.cmd = cmd
        self.slurm_options = slurm_options
        self.use_ramdisk = use_ramdisk
        self.copy_back = copy_back

    def prepare_experiment_dir(self):
        time = datetime.now()
        self.experiment_name = time.strftime("%y_%m_%d-%H_%M-") + self.tag
        self.experiment_path = os.path.join(self.system.workdir, self.experiment_name)
        with Connection(self.system.ssh_target) as c:
            result = c.run("mkdir {}".format(self.experiment_path))
            if not result.ok:
                print("Error")


    def deploy_dependencies(self):
        for dependency in self.dependencies:
            dependency.deploy(self.system, self.experiment_name)

    def prepare_environment(self):
        script = ""
        for var in self.system.environment:
            script += "export {var}={value}\n".format(var=var, value=self.system.environment[var])
        script_system = io.StringIO(script)
    
        script = ""
        for var in self.environment:
            script += "export {var}={value}\n".format(var=var, value=self.environment[var])
        script += "export EXPERIMENT_DIR={}\n".format(self.experiment_path)
        script_experiment = io.StringIO(script)

        if self.system.use_slurm:
            script = "#!/bin/bash\n"

            if self.use_ramdisk:
                script += "cp -dfR . {ramdisk_path}\n".format(ramdisk_path=self.system.ramdisk_path)
                script += "cd {ramdisk_path}\n".format(ramdisk_path=self.system.ramdisk_path)
                script += "rm slurm-*.out\n" # delete copy of .out file so it is not copied back afterwards

            for command in self.slurm_options["pre_commands"]:
                script += "{}\n".format(command)

            script += "{}\n".format(self.cmd) # user/payload command

            for command in self.slurm_options["post_commands"]:
                script += "{}\n".format(command)

            if self.use_ramdisk and self.copy_back:
                script += "cp -dfR {ramdisk_path}/. {experiment_path}\n".format(ramdisk_path=self.system.ramdisk_path, experiment_path=self.experiment_path)

            script_slurm = io.StringIO(script)

        with Connection(self.system.ssh_target) as c:
            result = c.put(script_system, os.path.join(self.experiment_path, "env_system.sh"))
            result = c.put(script_experiment, os.path.join(self.experiment_path, "env_experiment.sh"))

            if self.system.use_slurm:
                result = c.put(script_slurm, os.path.join(self.experiment_path, "slurm_script.sh"))
                result = c.put(io.StringIO(json.dumps(self.slurm_options)), os.path.join(self.experiment_path, "slurm_options.json")) # just for logging purposes


    def execute_cmd(self):
        with Connection(self.system.ssh_target) as c:
            if self.system.use_slurm:
                command = "sbatch "
                for option in self.slurm_options:
                    if option not in ["pre_commands", "post_commands"]:
                        command += "{option}={value} ".format(option=option, value=self.slurm_options[option])
                command += "--job-name={} ".format(self.experiment_name)
                command += "slurm_script.sh"
            else:
                command = self.cmd

            print("Running command: {}".format(command))
            result = c.run("cd {experiment_path} && source env_system.sh && source env_experiment.sh && {cmd}".format(experiment_path = self.experiment_path, cmd = command))
            if not result.ok:
                print("Error")

    def launch(self):
        self.prepare_experiment_dir()
        self.deploy_dependencies()
        self.prepare_environment()
        self.execute_cmd()

def main():
    # parse arguments
    args = sys.argv[1:]

    # TODO: implement CLI with pre-set default options instead of configuring runs in a Python script
    pass

if __name__ == "__main__":
    main()
