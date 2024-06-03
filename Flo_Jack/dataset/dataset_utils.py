import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd

def wavelet_spectrogram(df, n, wavelet='coif1'):
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
        columns.extend([f'{col}_wavelet_scale_{j}' for j in range(num_decompositions)])
        
        
        # Perform discrete wavelet transform up to the specified number of scales
        coeffs = pywt.wavedec(series, wavelet, level=max_level)
        
        mycoeff = np.zeros((num_decompositions, num_samples))

        # coeff = coeffs[::-1]
        # for j in range(len(coeff)-1):
        #     num_repeats = np.power(2, j+1)
        #     repeated = np.repeat(coeff[j], num_repeats)
        #     mycoeff[j] = repeated[:num_samples]
        # num_repeats = np.power(2, len(coeff)-1)
        # repeated = np.repeat(coeff[-1], num_repeats)
        # mycoeff[len(coeff)-1] = repeated[:num_samples]

        coeff = coeffs
        num_repeats = np.power(2, len(coeff)-1)
        repeated = np.repeat(coeff[0], num_repeats)
        mycoeff[0] = repeated[:num_samples]
        for j in range(1, len(coeff)):
            num_repeats = np.power(2, len(coeff)-j)
            repeated = np.repeat(coeff[j], num_repeats)
            mycoeff[j] = repeated[:num_samples]

        # print(mycoeff.T.shape)
        # print(i * n, (i+1) * n)
        # print(spectrogram[:, i * n : (i+1) * n].shape)
        spectrogram[:, i * num_decompositions : (i+1) * num_decompositions] = mycoeff.T
        # Add each level of detailed coefficients to the spectrogram
        # for j in range(1, n):
        #     if j <= max_level:
        #         # Resize the coefficients to match the original series length
        #         cD = pywt.upcoef('d', coeffs[j], wavelet, take=num_samples)
        #         if len(cD) < num_samples:
        #             cD = np.pad(cD, (0, num_samples - len(cD)), 'constant')

    # Create the DataFrame
    spectrogram_df = pd.DataFrame(spectrogram, columns=columns)
    
    return spectrogram_df