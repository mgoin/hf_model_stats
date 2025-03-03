import pandas as pd
import argparse
import os

def calculate_deltas(df):
    """
    Calculate daily deltas for likes and downloads_all_time for each model
    """
    # Sort by model_id and date to ensure proper delta calculations
    df = df.sort_values(['model_id', 'date'])
    
    # Initialize columns for deltas
    df['likes_delta'] = 0
    df['downloads_delta'] = 0
    
    # Group by model_id and calculate deltas within each group
    for model_id, group in df.groupby('model_id'):
        # Calculate deltas compared to previous day's values
        df.loc[group.index, 'likes_delta'] = group['likes'].diff()
        df.loc[group.index, 'downloads_delta'] = group['downloads_all_time'].diff()
    
    # Replace NaN values (from first entries) with the actual values
    # This assumes the first entry represents the initial count
    df['likes_delta'] = df['likes_delta'].fillna(df['likes'])
    df['downloads_delta'] = df['downloads_delta'].fillna(df['downloads_all_time'])
    
    # Ensure delta columns are integers
    df['likes_delta'] = df['likes_delta'].astype(int)
    df['downloads_delta'] = df['downloads_delta'].astype(int)
    
    return df

def analyze_stats(input_file):
    """
    Read the stats file, calculate deltas, and return analyzed data
    """
    df = pd.read_csv(input_file)
    
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate deltas
    df_with_deltas = calculate_deltas(df)
    
    return df_with_deltas

def main():
    parser = argparse.ArgumentParser(description="Calculate day-to-day deltas for model stats")
    parser.add_argument("--file", type=str, required=True, help="Input CSV file with model stats")
    
    args = parser.parse_args()
    
    # Get the base filename without extension
    base_name = os.path.splitext(args.file)[0]
    
    # Analyze the data
    df_full = analyze_stats(args.file)
    
    # Save the results
    output_file_full = f"{base_name}_with_deltas.csv"
    
    df_full.to_csv(output_file_full, index=False)
    
    print(f"Full data with deltas saved to: {output_file_full}")
    
    # Display summary statistics
    print("\nSummary of daily download deltas by model:")
    model_summary = df_full.groupby('model_id')['downloads_delta'].agg(['sum', 'mean', 'max']).sort_values('sum', ascending=False)
    print(model_summary.to_string())
    
    print("\nTotal downloads by date:")
    date_summary = df_full.groupby('date')['downloads_delta'].sum().reset_index()
    print(date_summary.to_string(index=False))

if __name__ == "__main__":
    main()