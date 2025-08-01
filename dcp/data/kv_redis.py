# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import datetime
import os
import subprocess
import time
import traceback

import redis

REDIS_CMD = os.environ.get("DCP_REDIS_CMD", "redis-server")
KVREDIS_POLLING_INTERVAL = float(
    os.environ.get("DCP_KVREDIS_POLLING_INTERVAL", "0.05")
)
KVREDIS_CONNECT_TIMEOUT = float(
    os.environ.get("DCP_KVREDIS_CONNECT_TIMEOUT", 30)
)


class RedisServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = self._run_redis_server()
        # register cleanup
        atexit.register(self.__del__)

    def __del__(self):
        if self.server.poll() is not None:
            return
        self.server.send_signal(subprocess.signal.SIGINT)
        self.server.wait()

    def _run_redis_server(self):
        # run a redis server
        p = subprocess.Popen(
            [
                REDIS_CMD,
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


class RedisKVStore(object):
    # a blocking redis client
    def __init__(self, host, port):
        self.host = host
        self.port = port
        # wait for redis server to start
        t = time.time()
        while True:
            try:
                self.client = redis.Redis(host=host, port=port, db=0)
                self.client.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(KVREDIS_POLLING_INTERVAL)
                if time.time() - t > KVREDIS_CONNECT_TIMEOUT:
                    raise RuntimeError(
                        f"WARNING: Cannot connect to KV Server at host {host}"
                        f" and port {port}. "
                        "Is DCP_KV_HOST and "
                        "DCP_KV_PORT set correctly?"
                    )
                continue

    def wait(self, keys, timeout=None):
        # wait for a key to be set
        time_start = datetime.datetime.now()
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        while True:
            # print("waiting for keys", keys, "timeout", timeout)
            # traceback.print_stack()
            if self.client.exists(*keys):
                break
            if (
                timeout is not None
                and datetime.datetime.now() - time_start > timeout
            ):
                # match torch kvstore behavior
                raise RuntimeError("Timeout")
            time.sleep(KVREDIS_POLLING_INTERVAL)

    def get(self, key, wait=True):
        if wait:
            self.wait(key)
        return self.client.get(key)

    def set(self, key, value: str, logger=None):
        # match torch kvstore behavior
        value_bytes = value.encode()
        self.client.set(key, value_bytes)
        if logger:
            if value.isprintable():
                logger.debug("KVStore: set {} to {}".format(key, value))
            else:
                logger.debug(
                    "KVStore: set {} to non-printable value".format(key)
                )

    def add(self, key, value: int):
        # match torch kvstore behavior
        return self.client.incr(key, value)

    def delete_key(self, key):
        return self.client.delete(key)
