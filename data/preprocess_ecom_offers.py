import gzip
import os
import zipfile
import numpy as np
import polars as pl
import kaggle
from loguru import logger
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'ecom-offers'
TMP_DATA_PATH = Path(__file__).parent / 'tmp' / 'ecom-offers'
TMP_DATA_PATH.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

def unzip(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(file_path.parent)

def main():
    logger.info('Downloading kaggle dataset acquire-valued-shoppers-challenge')
    kaggle.api.competition_download_files('acquire-valued-shoppers-challenge', path=TMP_DATA_PATH)
    if not (TMP_DATA_PATH / 'transactions.csv.gz').exists():
        logger.info('Unzipping dataset')
        unzip(TMP_DATA_PATH / 'acquire-valued-shoppers-challenge.zip')
    else:
        logger.info('Dataset already unzipped. Skipping unzip.')

    # ======================================================================================
    # >>> Memory-Efficient Preprocessing & Downsampling <<<
    # ======================================================================================

    logger.info('Loading small files')
    data_offers = pl.read_csv(TMP_DATA_PATH/'offers.csv.gz')
    data_train_history = pl.read_csv(TMP_DATA_PATH/'trainHistory.csv.gz').with_columns(
        pl.col('offerdate').str.strptime(pl.Date)
    )

    # Downsample users to fit in 8GB RAM (e.g., take 15% of users)
    fraction = 0.15
    logger.info(f'Downsampling users to {fraction*100}% to save RAM')
    sampled_ids = data_train_history.sample(fraction=fraction, seed=42)['id'].to_list()
    sampled_ids_set = set(sampled_ids)
    
    data_train_history = data_train_history.filter(pl.col('id').is_in(sampled_ids))

    logger.info('Filtering transactions.csv.gz line-by-line (O(1) memory)...')
    filtered_transactions_path = TMP_DATA_PATH / 'transactions_filtered.csv'
    
    if not filtered_transactions_path.exists():
        with gzip.open(TMP_DATA_PATH / 'transactions.csv.gz', 'rt') as f_in, \
             open(filtered_transactions_path, 'w') as f_out:
            
            header = next(f_in)
            f_out.write(header)
            
            # Determine which column is 'id'. Usually it's the first column (index 0).
            header_cols = header.strip().split(',')
            id_idx = header_cols.index('id')
            
            written = 0
            for line in f_in:
                cols = line.strip().split(',')
                # Check if this row's id is in our sampled set
                # The id in transactions might be an integer or string, match exactly
                if cols[id_idx] in sampled_ids_set or int(cols[id_idx]) in sampled_ids_set:
                    f_out.write(line)
                    written += 1
                    
        logger.info(f'Filtered transactions: wrote {written} rows.')
    else:
        logger.info('Filtered file already exists. Skipping filtering step.')

    logger.info('Loading filtered transactions with Polars Lazy API')
    data_transactions_lazy = pl.scan_csv(filtered_transactions_path).with_columns(
        pl.col('date').str.strptime(pl.Date)
    )

    data_train_offer = (
        data_train_history
        .join(data_offers, on='offer')
        .with_columns(pl.col('repeater').eq('t').cast(pl.Int32).alias('target'))
        .drop('repeater')
    )

    data_transactions_lazy = (
        data_transactions_lazy
        .join(data_train_offer.lazy(), on='id')
        .with_columns((pl.col('offerdate') - pl.col('date')).dt.total_days().alias('date_diff'))
    )

    filters = {
        'bought_company': pl.col('company').eq(pl.col('company_right')),
        'bought_category': pl.col('category').eq(pl.col('category_right')),
        'bought_brand': pl.col('brand').eq(pl.col('brand_right')),
    }

    date_diffs = [
        pl.col('date_diff').lt(d).alias(f'{d}') for d in [1, 3, 7, 14, 21, 28, 60, 90, 120, 150, 180]
    ]

    # Expressions used in aggregation of transaction histories
    exprs = [
        pl.col('purchaseamount').cast(pl.Float64).sum().alias('total_spend'),
        pl.col('target').first(),
        pl.col('offervalue').first(),
        pl.col('offerdate').first(),
        pl.col('offerdate').first().dt.weekday().alias('day_of_week'),
        pl.col('offerdate').first().dt.day().alias('day_of_month'),
        pl.col('offerdate').first().dt.ordinal_day().alias('day_of_year'),
    ]

    exprs += sum([
        [
            fv.sum().alias(f'has_{fn}'),
            pl.col('purchasequantity').cast(pl.Float64).filter(fv).alias(f'has_{fn}_q').sum(),
            pl.col('purchaseamount').cast(pl.Float64).filter(fv).sum().alias(f'has_{fn}_a')
        ]
        for fn,fv in filters.items()
    ], [])

    exprs += sum([
        [
            fv.and_(d).sum().alias(f'has_{fn}_{d.meta.output_name()}'),
            pl.col('purchasequantity').cast(pl.Float64).filter(fv.and_(d)).alias(f'has_{fn}_q_{d.meta.output_name()}').sum(),
            pl.col('purchaseamount').cast(pl.Float64).filter(fv.and_(d)).alias(f'has_{fn}_a_{d.meta.output_name()}').sum() 
        ]
        for d in date_diffs for fn,fv in filters.items()
    ], [])


    logger.info('Computing aggregations using Polars Streaming API...')
    data = (
        data_transactions_lazy
        .group_by('id')
        .agg(*exprs)
        .sort(by='offerdate')
        .collect(streaming=True)
    )

    X_num_data = data.select(
        *[pl.col(c) for c in data.columns if c not in ['id', 'target', 'offerdate']]
    ).cast(pl.Float32)

    X_bin_data = data.select(
        *[pl.col(f'has_bought_{c}').eq(0).alias(f'never_bought_{c}') for c in ['company', 'category', 'brand']],
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_category').ne(0)).and_(pl.col('has_bought_company').ne(0)).alias('has_bought_brand_company_category'),
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_category').ne(0)).alias('has_bought_brand_category'),
        pl.col('has_bought_brand').ne(0).and_(pl.col('has_bought_company').ne(0)).alias('has_bought_brand_company'),
    ).cast(pl.Float32)

    X_data = pl.concat([X_num_data, X_bin_data], how='horizontal')
    Y_data = data.select(pl.col('target')).cast(pl.Int64).to_numpy().ravel()
    periods = data.select(pl.col('offerdate').alias('timestamp')).cast(pl.Int64).to_numpy().ravel()

    # ======================================================================================
    # >>> Default task split <<<
    # ======================================================================================

    logger.info('Creating standard splits')
    # Last 5 days of offers
    test_mask = data.select(pl.col('offerdate').ge(pl.datetime(2013, 4, 25))).to_series().to_numpy()
    
    # Second to last 5 days of offers
    val_mask = data.select(
        pl.col('offerdate').ge(pl.datetime(2013, 4, 20)) &
        pl.col('offerdate').lt(pl.datetime(2013, 4, 25))
    ).to_series().to_numpy()
    
    # The rest is train
    train_mask = data.select(pl.col('offerdate').lt(pl.datetime(2013, 4, 20))).to_series().to_numpy()

    logger.info('Writing data to disk (Parquet and Npy)')

    def save_split(X, y, p, mask, prefix):
        X_split = X.filter(pl.Series(mask))
        X_split.write_parquet(DATA_DIR / f'X_{prefix}.parquet')
        np.save(DATA_DIR / f'y_{prefix}.npy', y[mask])
        np.save(DATA_DIR / f'periods_{prefix}.npy', p[mask])
        logger.info(f'Saved {prefix} split: {len(X_split)} samples')

    save_split(X_data, Y_data, periods, train_mask, 'train')
    save_split(X_data, Y_data, periods, val_mask, 'val')
    save_split(X_data, Y_data, periods, test_mask, 'test')

    logger.info('Done!')

if __name__ == "__main__":
    main()
