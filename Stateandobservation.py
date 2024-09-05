import typing
import numpy as np
import pandas as pd


class CustomerState:
    def __init__(
            self,
            recency: float,
            history: float,
            used_discount: int,
            used_bogo: int,
            zip_code: int,
            is_referral: int,
            channel: int,
            offer: int,
            conversion: int,
            recency_range: tuple = (0, 13),  # Example range for recency
            history_range: tuple = (0, 3350),
            initial_budget: float = 1000.0  # Initial budget
    ):
        self.recency = self.normalize(recency, recency_range)
        self.history = self.normalize(history, history_range)
        self.used_discount = used_discount
        self.used_bogo = used_bogo
        self.zip_code = zip_code
        self.is_referral = is_referral
        self.budget = initial_budget
        self.channel = channel
        self.offer = offer
        self.conversion = conversion

        self.conversion_count = 0
        self.offers_used = []
        self.channels_used = []
        self.budget = initial_budget  # Budget management
        self.engagement_score = self.calculate_engagement_score()

    def normalize(self, value: float, value_range: tuple) -> float:
        """Normalize a value to the range [0, 1]."""
        min_val, max_val = value_range
        if max_val > min_val:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0.0  # If range is invalid, return 0.0

    def update_conversion(self):
        self.conversion_count += 1
        self.engagement_score = self.calculate_engagement_score()

    def add_offer(self, offer: int, cost: float):
        self.offers_used.append(offer)
        self.budget -= cost  # Subtract the cost of the offer from the budget
        self.engagement_score = self.calculate_engagement_score()

    def add_channel(self, channel: int, cost: float):
        self.channels_used.append(channel)
        self.budget -= cost  # Subtract the cost of the channel from the budget
        self.engagement_score = self.calculate_engagement_score()

    def calculate_engagement_score(self) -> float:
        # Example weights
        w_recency = 0.15
        w_history = 0.2
        w_discount = 0.1
        w_bogo = 0.1
        w_conversion = 0.2
        w_offers = 0.15
        w_channels = 0.1
        w_budget = 0.1  # Adding budget as a factor in engagement

        score = (
                w_recency * self.recency +
                w_history * self.history +
                w_discount * self.used_discount +
                w_bogo * self.used_bogo +
                w_conversion * self.conversion_count +
                w_offers * len(self.offers_used) +
                w_channels * len(self.channels_used) +
                w_budget * self.budget / 1000.0  # Normalizing budget impact
        )

        return score

    def to_array(self):
        """Convert the CustomerState to a NumPy array."""
        return np.array([
            self.recency,
            self.history,
            self.used_discount,
            self.used_bogo,
            self.zip_code,
            self.is_referral,
            self.channel,
            self.offer,
            self.conversion,
            self.budget
        ], dtype=float)


class Observations:
    def __init__(
            self,
            window_size: int,
            observations: typing.List[CustomerState] = [],
    ):
        self._observations = observations
        self._window_size = window_size

        assert isinstance(self._observations, list), "observations must be a list"
        assert len(
            self._observations) <= self._window_size, f'observations length must be <= window_size, received: {len(self._observations)}'
        assert all(isinstance(observation, CustomerState) for observation in
                   self._observations), "observations must be a list of CustomerState objects"

    def __len__(self) -> int:
        return len(self._observations)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def observations(self) -> typing.List[CustomerState]:
        return self._observations

    @property
    def full(self) -> bool:
        return len(self._observations) == self._window_size

    def __getitem__(self, idx: int) -> CustomerState:
        try:
            return self._observations[idx]
        except IndexError:
            raise IndexError(f'index out of range: {idx}, observations length: {len(self._observations)}')

    def __iter__(self) -> CustomerState:
        """ Create a generator that iterates over the Sequence."""
        for index in range(len(self)):
            yield self[index]

    def reset(self) -> None:
        self._observations = []

    def append(self, state: CustomerState) -> None:
        # state should be CustomerState object or None
        assert isinstance(state, CustomerState) or state is None, "state must be a CustomerState object or None"
        self._observations.append(state)

        if len(self._observations) > self._window_size:
            self._observations.pop(0)

    @property
    def recency(self) -> np.ndarray:
        return np.array([state.recency for state in self._observations])

    @property
    def history(self) -> np.ndarray:
        return np.array([state.history for state in self._observations])

    @property
    def used_discount(self) -> np.ndarray:
        return np.array([state.used_discount for state in self._observations])

    @property
    def used_bogo(self) -> np.ndarray:
        return np.array([state.used_bogo for state in self._observations])

    @property
    def zip_code(self) -> np.ndarray:
        return np.array([state.zip_code for state in self._observations])

    @property
    def is_referral(self) -> np.ndarray:
        return np.array([state.is_referral for state in self._observations])

    @property
    def channel(self) -> np.ndarray:
        return np.array([state.channel for state in self._observations])

    @property
    def offer(self) -> np.ndarray:
        return np.array([state.offer for state in self._observations])

    @property
    def conversion(self) -> np.ndarray:
        return np.array([state.conversion for state in self._observations])

    @property
    def conversion_count(self) -> np.ndarray:
        return np.array([state.conversion_count for state in self._observations])

    @property
    def offers_used(self) -> typing.List[list]:
        return [state.offers_used for state in self._observations]

    @property
    def channels_used(self) -> typing.List[list]:
        return [state.channels_used for state in self._observations]

    @property
    def engagement_score(self) -> np.ndarray:
        return np.array([state.engagement_score for state in self._observations])

    @property
    def budget(self) -> np.ndarray:
        """Returns the budget of each CustomerState in the observation window."""
        return np.array([state.budget for state in self._observations])
