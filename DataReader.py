import numpy as np
from Stateandobservation import CustomerState

class PdDataFeeder:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")

        row = self.data.iloc[idx]
        return CustomerState(
            recency=row['recency'],
            history=row['history'],
            used_discount=row['used_discount'],
            used_bogo=row['used_bogo'],
            zip_code=row['zip_code'],
            is_referral=row['is_referral'],
            channel=row['channel'],
            offer=row['offer'],
            conversion=row['conversion']
        )
