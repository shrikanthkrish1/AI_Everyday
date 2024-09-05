import numpy as np
import typing
import json
import os
import importlib


class CustomerBehaviorEnv(gym.Env):
    def __init__(
            self,
            data_feeder: typing.List[CustomerState],
            max_episode_steps: int = None,
            window_size: int = 50,
            reward_function: typing.Callable = RewardCalculator(),
            action_space: ActionSpace = ActionSpace.DISCRETE,
            metrics: typing.List[typing.Callable] = []
    ) -> None:

        default_state_args = {
            'recency': 0,
            'history': 0,
            'used_discount': 0,
            'used_bogo': 0,
            'zip_code': 0,
            'is_referral': 0,  # Changed to int
            'channel': 0,
            'offer': 0,
            'conversion': 0  # Changed to int
        }

        self._data_feeder = data_feeder
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else len(data_feeder)
        self._window_size = window_size
        self._reward_function = reward_function
        self._metrics = metrics

        self._observations = Observations(window_size=window_size,
                                          observations=[CustomerState(**default_state_args)] * window_size)
        self._observation_space = np.zeros(self.reset()[0].shape)
        self._action_space = action_space

    @property
    def action_space(self):
        return self._action_space.value

    @property
    def observation_space(self):
        return self._observation_space

    def _get_obs(self, index: int) -> CustomerState:
        next_state = self._data_feeder[index]
        return next_state

    def _get_terminated(self):
        return False

    def _take_action(self, action: int) -> typing.Tuple[int, float]:
        assert action in list(
            range(self._action_space.value)), f'action must be in range {self._action_space.value}, received: {action}'

        last_state, next_state = self._observations[-2:]

        # Implement action logic: e.g., update offers and channels
        if action == 0:  # Example action logic
            next_state.add_offer(1, 10)  # Add discount offer with cost
            next_state.add_channel(1, 5)  # Use a specific channel with cost
        elif action == 1:
            next_state.add_offer(2, 15)  # Another discount offer with cost
            next_state.add_channel(2, 10)  # Use another channel with cost
        elif action == 2:
            next_state.add_offer(0, 0)  # No discount
            next_state.add_channel(0, 0)  # Use default channel with cost

        return action, 1.0

    @property
    def metrics(self):
        return self._metrics

    def _metricsHandler(self, observation: CustomerState):
        metrics = {}
        for metric in self._metrics:
            metric.update(observation)
            metrics[metric.name] = metric.result

        return metrics

    def step(self, action: int) -> typing.Tuple[np.ndarray, float, bool, bool, dict]:
        index = self._env_step_indexes.pop(0)

        observation = self._get_obs(index)
        self._observations.append(observation)

        action, _ = self._take_action(action)
        reward = self._reward_function.calculate_reward(self._observations[-1], action)
        terminated = self._get_terminated()
        truncated = False if self._env_step_indexes else True
        info = {
            "states": [observation],
            "metrics": self._metricsHandler(observation)
        }

        # Convert observations to a numerical format (e.g., list of floats)
        transformed_obs = np.array([obs.to_array() for obs in self._observations.observations])

        if np.isnan(transformed_obs).any():
            raise ValueError("transformed_obs contains nan values, check your data")

        return transformed_obs, reward, terminated, truncated, info

    def reset(self) -> typing.Tuple[np.ndarray, dict]:
        size = len(self._data_feeder) - self._max_episode_steps
        self._env_start_index = np.random.randint(0, size) if size > 0 else 0
        self._env_step_indexes = list(range(self._env_start_index, self._env_start_index + self._max_episode_steps))

        self._observations.reset()
        while not self._observations.full:
            obs = self._get_obs(self._env_step_indexes.pop(0))
            self._observations.append(obs)

        info = {
            "states": self._observations.observations,
            "metrics": {}
        }

        for metric in self._metrics:
            metric.reset(self._observations.observations[-1])

        # Convert observations to a numerical format (e.g., list of floats)
        transformed_obs = np.array([obs.to_array() for obs in self._observations.observations])
        if np.isnan(transformed_obs).any():
            raise ValueError("transformed_obs contains nan values, check your data")

        return transformed_obs, info

    def render(self, mode='human', interval=500):
        """
        Renders the environment's current state visually with animation.

        Parameters:
        - mode (str): The mode in which to render the environment. Options: 'human', 'json', 'animated'.
        - interval (int): The interval between frames in milliseconds for the animation.
        """
        if mode == 'human' or mode == 'json':
            # Existing render code for human and json modes
            self._render_text_or_json(mode)
        elif mode == 'animated':
            # Visual rendering using animated bar chart
            self._render_animated(interval)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def _render_text_or_json(self, mode):
        # Handle human and JSON rendering (from the previous example)
        if mode == 'human':
            current_state = self._observations.observations[-1]
            print(f"Current Customer State:")
            print(f"  Recency: {current_state.recency}")
            print(f"  History: {current_state.history}")
            print(f"  Used Discounts: {current_state.used_discount}")
            print(f"  Used BOGO: {current_state.used_bogo}")
            print(f"  Zip Code: {current_state.zip_code}")
            print(f"  Is Referral: {current_state.is_referral}")
            print(f"  Channel: {current_state.channel}")
            print(f"  Offer: {current_state.offer}")
            print(f"  Conversion: {current_state.conversion}")
            print(f"  Budget: {current_state.budget}")
            print(f"  Engagement Score: {current_state.engagement_score}")
        elif mode == 'json':
            current_state = self._observations.observations[-1]
            state_dict = {
                "recency": current_state.recency,
                "history": current_state.history,
                "used_discount": current_state.used_discount,
                "used_bogo": current_state.used_bogo,
                "zip_code": current_state.zip_code,
                "is_referral": current_state.is_referral,
                "channel": current_state.channel,
                "offer": current_state.offer,
                "conversion": current_state.conversion,
                "budget": current_state.budget,
                "engagement_score": current_state.engagement_score
            }
            print(json.dumps(state_dict, indent=4))

    def _render_animated(self, interval):
        # Define the animation function
        def animate(i):
            current_state = self._observations.observations[i % len(self._observations.observations)]
            data = [
                current_state.recency,
                current_state.history,
                current_state.used_discount,
                current_state.used_bogo,
                current_state.zip_code,
                current_state.is_referral,
                current_state.channel,
                current_state.offer,
                current_state.conversion,
                current_state.budget,
                current_state.engagement_score
            ]
            for bar, value in zip(bars, data):
                bar.set_height(value)
            return bars

        # Set up the figure and axis
        fig, ax = plt.subplots()
        labels = [
            'Recency', 'History', 'Used Discount', 'Used BOGO',
            'Zip Code', 'Is Referral', 'Channel', 'Offer',
            'Conversion', 'Budget', 'Engagement Score'
        ]
        bars = ax.bar(labels, np.zeros(len(labels)))

        # Set up the animation
        ani = FuncAnimation(fig, animate, frames=len(self._observations.observations), interval=interval, blit=False,
                            repeat=True)

        # Display the animation
        plt.show()

    def close(self):
        pass

    def config(self):
        return {
            "data_feeder": "CustomerStateList",
            "max_episode_steps": self._max_episode_steps,
            "window_size": self._window_size,
            "reward_function": self._reward_function.__class__.__name__,
            "metrics": [metric.__class__.__name__ for metric in self._metrics],
            "observation_space_shape": tuple(self.observation_space.shape),
            "action_space": self._action_space.name,
        }

    def save_config(self, path: str = ""):
        output_path = os.path.join(path, "CustomerBehaviorEnv.json")
        with open(output_path, "w") as f:
            json.dump(self.config(), f, indent=4)

    @staticmethod
    def load_config(data_feeder: typing.List[CustomerState], path: str = "", **kwargs):
        input_path = os.path.join(path, "CustomerBehaviorEnv.json")
        if not os.path.exists(input_path):
            raise Exception(f"CustomerBehaviorEnv Config file not found in {path}")
        with open(input_path, "r") as f:
            config = json.load(f)

        environment = CustomerBehaviorEnv(
            data_feeder=data_feeder,
            max_episode_steps=kwargs.get("max_episode_steps") or config["max_episode_steps"],
            window_size=kwargs.get("window_size") or config["window_size"],
            reward_function=getattr(importlib.import_module(".reward", package=__package__),
                                    config["reward_function"])(),
            action_space=ActionSpace[config["action_space"]],
            metrics=[getattr(importlib.import_module(".metrics", package=__package__), metric)() for metric in
                     config["metrics"]]
        )

        return environment



