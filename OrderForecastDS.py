import torch
from torch.utils.data import Dataset
#from torch import nn
import numpy as np
from datetime import timedelta

def mean_std_norm(X, mean=None, std=None):

    if mean is None and std is None:
        # Compute mean and std along time dimensions (dim=(0))
        mean = X.mean(dim=(0), keepdim=True)
        std = X.std(dim=(0), keepdim=True)

        # Avoid division by zero
        std[std == 0] = 1e-8

    # Normalize
    Xn = (X - mean) / std
    return Xn, mean, std

def mean_std_denorm(Xn, mean, std):
    # DeNormalize
    X = Xn * std + mean 
    return X

class OFDataset(Dataset):
    def __init__(self, _spark, _pdc, _sDate, _eDate):
        self.pdc = _pdc
        self.sDate = _sDate
        self.eDate = _eDate

        self.predictorLenW = 26
        self.responseLenW = 1

        df = _spark.sql(f"""
            with date_counts as (
                select date(CREATEDT) as created_date, count(*) as row_count
                from snowflake_datalake_prod.aftersales.dv_order
                where I_DEPOT = {_pdc}
                    and CREATEDT >= '{_sDate}'
                    and CREATEDT <= '{_eDate}'
                group by date(CREATEDT)
            ),
            date_range as (
                select sequence(
                    min(date(CREATEDT)),
                    max(date(CREATEDT))
                ) as all_dates
                from snowflake_datalake_prod.aftersales.dv_order
                where I_DEPOT = {_pdc}
                    and CREATEDT >= '{_sDate}'
                    and CREATEDT <= '{_eDate}'
            ),
            expanded_dates as (
                select explode(all_dates) as created_date
                from date_range
            )
            select
                ed.created_date,
                coalesce(dc.row_count, 0) as row_count
            from expanded_dates ed
            left join date_counts dc
                on ed.created_date = dc.created_date
            order by ed.created_date
        """)

        #Snowflake
        #df = _spark.sql(f"""
        #    with date_counts as (
        #        select date(CREATEDT) as created_date, count(*) as row_count
        #        from snowflake_datalake_prod.aftersales.dv_order
        #        where I_DEPOT = {_pdc}
        #            and CREATEDT >= '{_sDate}'
        #            and CREATEDT <= '{_eDate}'
        #        group by date(CREATEDT)
        #    ),
        #    date_range as (
        #        select distinct date(CREATEDT) as all_dates
        #        from snowflake_datalake_prod.aftersales.dv_order
        #        where I_DEPOT = {_pdc}
        #            and CREATEDT >= '{_sDate}'
        #            and CREATEDT <= '{_eDate}'
        #        order by all_dates
        #    ),
        #    expanded_dates as (
        #        select all_dates as created_date
        #        from date_range
        #    )
        #    select
        #        ed.created_date,
        #        coalesce(dc.row_count, 0) as row_count
        #    from expanded_dates ed
        #    left join date_counts dc
        #        on ed.created_date = dc.created_date
        #    order by ed.created_date
        #""")

        self.predictorLenMonthOH = 12
        self.predictorLenDayOfMonthOH = 31
        self.predictorLenDayOfWeekOH = 7
        self.predictorLenOH = self.predictorLenMonthOH + self.predictorLenDayOfMonthOH + self.predictorLenDayOfWeekOH
        self.predictorLenD = self.predictorLenW * 7
        self.predictorLen = self.predictorLenD + self.predictorLenOH

        self.responseLenD = self.responseLenW * 7

        self.windowLen = self.predictorLenD + self.responseLenD

        # Collect all data once - much faster than multiple iterations
        all_row_counts = np.array([row.row_count for row in df.toLocalIterator()], dtype=np.float32)
        total_rows = len(all_row_counts)

        created_dateOH = np.array([self.createOH(row.created_date + timedelta(days=self.predictorLenD-1)) for row in df.toLocalIterator()], dtype=np.float32)
        
        self.len = total_rows - self.windowLen + 1

        # Pre-allocate tensors
        self.X = torch.zeros((self.predictorLen, self.len), dtype=torch.float32)
        self.Y = torch.zeros((self.responseLenD, self.len), dtype=torch.float32)

        # Vectorized window creation - much faster than loops
        for i in range(self.len):
            countsDateOH = np.zeros(self.predictorLen, dtype=np.float32)
            countsDateOH[:self.predictorLenD] = all_row_counts[i:i + self.predictorLenD]
            countsDateOH[self.predictorLenD:] = created_dateOH[i]
            self.X[:, i] = torch.from_numpy(countsDateOH)
            self.Y[:, i] = torch.from_numpy(all_row_counts[i + self.predictorLenD:i + self.windowLen])

        # Normalize
        self.Xn = self.X
        self.Xn[:self.predictorLenD, :], self.mean, self.std = mean_std_norm(self.X[:self.predictorLenD, :])
        self.Yn, _, _ = mean_std_norm(self.Y, self.mean, self.std)
        self.normFl = 1

    def createOH(self, created_date):
        # Create one-hot encoding for month, day of month, and day of week
        Xoh = np.zeros(self.predictorLenOH, dtype=np.float32)
        
        # Extract date components
        month = created_date.month - 1  # 0-indexed (0-11)
        day_of_month = created_date.day - 1  # 0-indexed (0-30)
        day_of_week = created_date.weekday()  # Monday=0, Sunday=6
        
        # Set one-hot encodings
        Xoh[month] = 1  # Month encoding (positions 0-11)
        Xoh[self.predictorLenMonthOH + day_of_month] = 1  # Day of month (positions 12-42)
        Xoh[self.predictorLenMonthOH + self.predictorLenDayOfMonthOH + day_of_week] = 1  # Day of week (positions 43-49)
        
        return Xoh

    def __len__(self):
        return self.len

    def __getitem__(self, _idx):
        if self.normFl == 1:
            return self.Xn[:, _idx], self.Yn[:, _idx], self.mean[:,_idx], self.std[:,_idx]
        else:
            return self.X[:, _idx], self.Y[:, _idx], self.mean[:,_idx], self.std[:,_idx]
        
    def deNorm(self):
        self.normFl = 0
