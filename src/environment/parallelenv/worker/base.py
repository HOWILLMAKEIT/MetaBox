import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                           np.ndarray]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """Send action signal to low-level worker.

        When action is None, it indicates sending "reset" signal; otherwise
        it indicates "step" signal. The paired return value from "recv"
        function is determined by such kind of different signal.
        """
        if hasattr(self, "send_action"):
            warnings.warn(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)  # type: ignore

    def send_reset(self) -> None:
        if hasattr(self, "send_action"):
            warnings.warn(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            self.is_reset = True
            self.result = self.reset()

    def recv(
        self
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Receive result from low-level worker.

        If the last "send" function sends a NULL action, it only returns a
        single observation; otherwise it returns a tuple of (obs, rew, done,
        info).
        """
        if hasattr(self, "get_result"):
            warnings.warn(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()  # type: ignore
        return self.result

    def reset(self) -> np.ndarray:
        self.send(None)
        return self.recv()  # type: ignore

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform one timestep of the environment's dynamic.

        "send" and "recv" are coupled in sync simulation, so users only call
        "step" function. But they can be called separately in async
        simulation, i.e. someone calls "send" first, and calls "recv" later.
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    # def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
    #     return self.action_space.seed(int(seed))  # issue 299

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()
