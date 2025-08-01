import redis
import time
import hashlib
import subprocess
import pickle
import datetime
import os
import pwd

EXP_REDIS_PORT = 9876
KVREDIS_INIT_POLLING_INTERVAL = 0.5
KVREDIS_CONNECT_TIMEOUT = 30
KVREDIS_POLLING_INTERVAL = 0.5


# redis client to track experiment progress between different nodes
class RedisKVStore(object):
    # a blocking local redis client
    def __init__(self, host_ip, node_rank, num_nodes, name_key):
        self.node_rank = node_rank
        self.is_master = node_rank == 0
        self.host = host_ip
        self.port = (
            int(
                hashlib.sha1(name_key.encode("utf-8")).hexdigest(),
                16,
            )
            % 62535
            + 3000
        )
        self.n_processes = num_nodes
        self.barrier_cnt = 0
        self.gather_cnt = 0
        if self.is_master:
            self.server = self._run_redis_server()
        # wait for redis server to start
        t = time.time()
        print(
            "Connecting to KV Server at {}:{}, {} processes in total.".format(
                self.host, self.port, self.n_processes
            )
        )
        while True:
            try:
                self.client = redis.Redis(host=self.host, port=self.port, db=0)
                self.client.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(KVREDIS_INIT_POLLING_INTERVAL)
                if time.time() - t > KVREDIS_CONNECT_TIMEOUT:
                    raise RuntimeError(
                        f"WARNING: Cannot connect to KV Server at host {self.host}"
                        f" and port {self.port}. "
                    )
                continue
        print(
            "Connected to KV Server at {}:{}, {} processes in total.".format(
                self.host, self.port, self.n_processes
            )
        )

    def __del__(self):
        if self.is_master:
            if self.server.poll() is not None:
                return
            self.server.send_signal(subprocess.signal.SIGINT)
            self.server.wait()

    def _run_redis_server(self):
        # run a redis server
        p = subprocess.Popen(
            [
                "redis-server",
                "--save",
                "",
                "--port",
                str(self.port),
                "--bind",
                str(self.host),
                "--protected-mode",
                "no",
            ],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        # wait 0.5s and check if the server is running
        time.sleep(0.5)
        if p.poll() is not None:
            # try to get the error message
            err_msg = "Failed to get error message."
            try:
                err_msg = p.stderr.read().decode("utf-8")
                if err_msg:
                    err_msg = f"Error: {err_msg}"
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to start redis server at {self.host}:{self.port}. "
                f"Error message: {err_msg}"
            )
        return p

    def wait(self, keys, timeout=None):
        # wait for a key to be set
        time_start = datetime.datetime.now()
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        while True:
            if self.client.exists(*keys):
                break
            if (
                timeout is not None
                and datetime.datetime.now() - time_start > timeout
            ):
                # match torch kvstore behavior
                raise RuntimeError("Timeout")
            time.sleep(KVREDIS_POLLING_INTERVAL)

    def barrier(self):
        if self.check_abort_signal():
            raise RuntimeError("Abort signal received")
        key = "barrier_{}".format(self.barrier_cnt)
        self.client.incr(key)
        while True:
            count = int(self.client.get(key).decode())
            if count == self.n_processes:
                break
            time.sleep(KVREDIS_POLLING_INTERVAL)
        self.barrier_cnt += 1

    def blocking_get(self, key):
        self.wait(key)
        return self.client.get(key)

    def set(self, key, value):
        # match torch kvstore behavior
        self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)

    def add(self, key, value: int):
        # match torch kvstore behavior
        return self.client.incr(key, value)

    def delete_key(self, key):
        return self.client.delete(key)

    def gather(self, obj):
        if self.check_abort_signal():
            raise RuntimeError("Abort signal received")
        # synchronous gather
        ack_key = f"gather_ack_{self.gather_cnt}"
        if self.node_rank == 0:
            recved_objs = [obj]
            # read from all keys
            for i in range(1, self.n_processes):
                key = "gather_{}_r{}".format(self.gather_cnt, i)
                self.wait(key)
                recved_objs.append(pickle.loads(self.client.get(key)))
                self.delete_key(key)
            # set ack key
            self.set(ack_key, "1")
            self.gather_cnt += 1
            return recved_objs
        else:
            # delete ack key
            self.delete_key(ack_key)
            key = "gather_{}_r{}".format(self.gather_cnt, self.node_rank)
            self.client.set(key, pickle.dumps(obj))
            # wait for ack key before returning
            self.wait(ack_key)
            self.gather_cnt += 1
            return

    def allgather(self, obj):
        if self.check_abort_signal():
            raise RuntimeError("Abort signal received")
        # synchronous allgather
        ack_key = f"allgather_ack_{self.gather_cnt}_r{self.node_rank}"
        recved_objs = [None] * self.n_processes
        # delete self ack key if exists
        if self.get(ack_key) is not None:
            self.delete_key(ack_key)
        # set allgather key
        key = f"allgather_{self.gather_cnt}_r{self.node_rank}"
        self.set(key, pickle.dumps(obj))
        # wait for all allgather keys
        for i in range(self.n_processes):
            if i == self.node_rank:
                continue
            key = f"allgather_{self.gather_cnt}_r{i}"
            self.wait(key)
            recved_objs[i] = pickle.loads(self.client.get(key))
        # set self ack key
        self.set(ack_key, "1")
        # wait for all ack keys
        for i in range(self.n_processes):
            if i == self.node_rank:
                continue
            ack_key = f"allgather_ack_{self.gather_cnt}_r{i}"
            self.wait(ack_key)
        recved_objs[self.node_rank] = obj
        self.gather_cnt += 1
        return recved_objs

    def send_abort_signal(self):
        self.client.set("abort", 1)

    def check_abort_signal(self):
        signal = self.client.get("abort")
        if signal is not None and int(signal.decode()) == 1:
            return True
        return False


def kill_redis_servers(node_rank, kv_store, include_controller=False):
    if node_rank != 0 or include_controller:
        server_pid = None
    else:
        kv: RedisKVStore = kv_store
        server_pid = kv.server.pid if kv.server else None
    redis_pids = list(
        map(int, os.popen("pgrep redis").read().strip().splitlines())
    )
    for pid in redis_pids:
        if pid != server_pid:
            proc_stat_file = os.stat(f"/proc/{pid}")
            uid = proc_stat_file.st_uid
            try:
                username = pwd.getpwuid(uid)[0]
            except KeyError:
                # user not found, maybe we're in docker
                username = None
            if username is not None and username != "redis":
                # we only kill our own redis servers
                os.system(f"kill -9 {pid}")
