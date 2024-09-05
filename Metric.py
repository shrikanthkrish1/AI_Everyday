from Stateandobservation import CustomerState

class Metric:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def update(self, state: CustomerState):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def result(self):
        raise NotImplementedError

class ConversionRate(Metric):
    def __init__(self):
        super().__init__("ConversionRate")
        self.total_conversions = 0
        self.total_customers = 0

    def update(self, state: CustomerState):
        if state.conversion:
            self.total_conversions += 1
        self.total_customers += 1

    def reset(self):
        self.total_conversions = 0
        self.total_customers = 0

    @property
    def result(self):
        return self.total_conversions / self.total_customers if self.total_customers > 0 else 0

class OfferEffectiveness(Metric):
    def __init__(self):
        super().__init__("OfferEffectiveness")
        self.offer_stats = {}

    def update(self, state: CustomerState):
        if state.offer not in self.offer_stats:
            self.offer_stats[state.offer] = {'conversions': 0, 'total': 0}
        self.offer_stats[state.offer]['total'] += 1
        if state.conversion:
            self.offer_stats[state.offer]['conversions'] += 1

    def reset(self):
        self.offer_stats = {}

    @property
    def result(self):
        effectiveness = {}
        for offer, stats in self.offer_stats.items():
            effectiveness[offer] = stats['conversions'] / stats['total'] if stats['total'] > 0 else 0
        return effectiveness

class ChannelEffectiveness(Metric):
    def __init__(self):
        super().__init__("ChannelEffectiveness")
        self.channel_stats = {}

    def update(self, state: CustomerState):
        if state.channel not in self.channel_stats:
            self.channel_stats[state.channel] = {'conversions': 0, 'total': 0}
        self.channel_stats[state.channel]['total'] += 1
        if state.conversion:
            self.channel_stats[state.channel]['conversions'] += 1

    def reset(self):
        self.channel_stats = {}

    @property
    def result(self):
        effectiveness = {}
        for channel, stats in self.channel_stats.items():
            effectiveness[channel] = stats['conversions'] / stats['total'] if stats['total'] > 0 else 0
        return effectiveness
