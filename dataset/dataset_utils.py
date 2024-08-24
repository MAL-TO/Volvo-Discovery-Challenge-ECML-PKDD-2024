import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

def wavelet_spectrogram(df, n, wavelet='db1'):
    """
    Perform wavelet transformation on each column of the input DataFrame and return a DataFrame with the wavelet spectrogram.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data.
    n (int): Number of scales to use for the wavelet transformation.
    wavelet (str): Type of wavelet to use (default is 'db1').
    
    Returns:
    pd.DataFrame: DataFrame with wavelet spectrograms, columns as signal_x_scale_y.
    """
    wavelet = pywt.Wavelet(wavelet)
    
    # Check if n is valid
    if n < 1:
        raise ValueError("Number of scales (n) must be at least 1.")
    
    num_samples = df.shape[0]
    num_features = df.shape[1]
    
    # Initialize a list to store the column names
    columns = []
    
    # Initialize a 2D array to store the wavelet spectrograms
    num_decompositions = (n+1)
    spectrogram = np.zeros((num_samples, num_features * num_decompositions))
    
    # Calculate the maximum level of decomposition for the given data length
    max_level = min(n, pywt.dwt_max_level(num_samples, wavelet.dec_len))

    # Loop through each column in the DataFrame
    for i, col in enumerate(df.columns):
        series = df[col].values
        columns.extend([f'{col}_wavelet_{wavelet}_scale_{j}' for j in range(num_decompositions)])
        
        
        # Perform discrete wavelet transform up to the specified number of scales
        coeffs = pywt.wavedec(series, wavelet, level=max_level)
        
        mycoeff = np.zeros((num_decompositions, num_samples))

        coeff = coeffs
        num_repeats = np.power(2, len(coeff)-1)
        repeated = np.repeat(coeff[0], num_repeats)
        mycoeff[0] = repeated[:num_samples]
        for j in range(1, len(coeff)):
            num_repeats = np.power(2, len(coeff)-j)
            repeated = np.repeat(coeff[j], num_repeats)
            mycoeff[j] = repeated[:num_samples]

        spectrogram[:, i * num_decompositions : (i+1) * num_decompositions] = mycoeff.T

    # Create the DataFrame
    spectrogram_df = pd.DataFrame(spectrogram, columns=columns)
    
    return spectrogram_df


from scipy.signal import stft

def my_stft(signal, window=5):
        # Parameters for STFT
        nperseg = window  # Length of each segment
        noverlap = window-1  # Number of points to overlap between segments

        # Compute the STFT
        f, t, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap, padded=True)

        # Extract features for each timestep
        # Feature extraction: Use the magnitude of the STFT coefficients as features
        stft_magnitude = np.abs(Zxx)
        return stft_magnitude

def stft_spectogram(df, window=5):
    
    num_samples = df.shape[0]
    num_features = df.shape[1]
    
    # Initialize a list to store the column names
    columns = []
    
    # Initialize a 2D array to store the wavelet spectrograms
    num_frequencies = window//2 + 1
    spectrogram = np.zeros((num_samples, num_features*num_frequencies))
    
    # Loop through each column in the DataFrame
    for i, col in enumerate(df.columns):
        series = df[col].values
        columns.extend([f'{col}_stft_{j}' for j in range(num_frequencies)])

        freq_extracted = my_stft(series, window=window)
        spectrogram[:, i * num_frequencies : (i+1) * num_frequencies] = freq_extracted.T
        
    
    spectrogram_df = pd.DataFrame(spectrogram, columns=columns)

    return spectrogram_df

def classify_columns(df, ratio=0.1):
    classification = []

    for column in df.columns:
        unique_values = df[column].nunique()
        total_values = len(df[column])
        unique_ratio = unique_values / total_values
        
        # Heuristic: If unique ratio is high, it's likely numerical
        if df[column].dtype in [int, float] and unique_ratio > ratio:
            classification.append("numerical")
        # Heuristic: If unique ratio is low, it's likely categorical
        elif df[column].dtype == object or unique_ratio <= ratio:
            classification.append("categorical")
        # Heuristic: For integer columns, check the number of unique values
        elif df[column].dtype == int:
            if unique_values < ratio * total_values:  # Arbitrary threshold for uniqueness
                classification.append("categorical")
            else:
                classification.append("numerical")
        # Fallback: Classify as numerical if not sure
        else:
            classification.append("numerical")

    return classification