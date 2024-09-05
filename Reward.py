from Stateandobservation import CustomerState,Observations




class RewardCalculator():
    def calculate_reward(self, state, action):

        recency = state.recency
        history = state.history
        used_discount = state.used_discount
        used_bogo = state.used_bogo
        zip_code = state.zip_code
        is_referral = state.is_referral
        channel = state.channel
        offer = state.offer
        conversion = state.conversion
        budget = state.budget

        # Initialize reward based on conversion
        reward = 1 if conversion else -1

        # Recency-based reward (normalized)
        normalized_recency = recency / 100  # Assuming max recency is 100
        if normalized_recency > 0.1 and conversion == 1:
            reward += 0.5 * normalized_recency

        # History-based reward (normalized)
        normalized_history = history / 1000  # Assuming max history is 1000
        if normalized_history > 0.5 and action == 0 and conversion == 1:
            reward += 0.7 * normalized_history

        # Discount usage reward
        if used_discount == 0 and action == 1 and conversion == 1:
            reward += 0.3

        # Bogo usage reward
        if used_bogo == 0 and action == 2 and conversion == 1:
            reward += 0.3

        # Referral-based reward
        if is_referral == 1 and conversion == 1:
            reward += 0.4

        # Channel-based reward
        if channel == 0 and conversion == 1:
            reward += 0.1  # Lesser reward for Channel 0
        elif channel == 1 and conversion == 1:
            reward += 0.2  # Higher reward for Channel 1
        elif channel == 2 and conversion == 1:
            reward += 0.15  # Medium reward for Channel 2

        # Budget consideration
        cost_of_action = self.get_action_cost(action, channel, offer)
        reward += (1 - cost_of_action / budget) * 0.5  # Reward for cost-efficient action

        # Cost efficiency penalty
        if action != 0 and conversion == 0:
            reward -= 0.3

        # Additional penalty for repeated failed actions
        if action != 0 and conversion == 0:
            reward -= 0.4 * (normalized_recency + normalized_history)

        return reward

    def get_action_cost(self, action, channel, offer):
        """Calculate the cost of the action based on the channel and offer used."""
        # Hypothetical costs for channels and offers
        channel_costs = {0: 0.1, 1: 0.2, 2: 0.15}  # Example costs for channels
        offer_costs = {0: 0.1, 1: 0.2, 2: 0.15}  # Example costs for offers

        # Cost associated with the action
        return channel_costs.get(channel, 0) + offer_costs.get(offer, 0)

    def calculate_engagement_score(self, state):
        recency, history, used_discount, used_bogo, zip_code, is_referral, channel, offer, conversion, budget = state

        # Hypothetical engagement score calculation
        score = 0.3 * (1 - recency / 100) + 0.3 * (history / 1000) + 0.2 * is_referral + 0.2 * conversion
        return score
