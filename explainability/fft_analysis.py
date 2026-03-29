import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_fft_profile(face_img):
    # FFT helps us look at the image entirely in terms of spatial frequency rather than color pixels.
    # We drop the color channels because structure/textures are fully captured in grayscale.
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    
    # Compute the 2D transform 
    f = np.fft.fft2(gray)
    
    # Normally the lowest frequencies are clustered at the corners of the array,
    # but we shift them to the center here so the plot looks like a traditional bullseye.
    fshift = np.fft.fftshift(f)
    
    # Take the log of the magnitude since differences in frequency power are exponential
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # To distill the entire 2D image into a simple 1D line chart, we basically just average
    # out the values in concentric rings moving out from the center (low to high freq).
    h, w = gray.shape
    y, x = np.indices((h, w))
    center = (x.max() / 2, y.max() / 2)
    r = np.hypot(x - center[0], y - center[1]).astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(radialprofile, color='#00ffcc', linewidth=2)
    
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.set_title("1D Frequency Power Spectrum", color='white', pad=15)
    ax.set_xlabel("Spatial Frequency (Low to High)", color='#cccccc')
    ax.set_ylabel("Magnitude (Log Scale)", color='#cccccc')
    ax.tick_params(colors='#cccccc')
    
    ax.axvspan(len(radialprofile)*0.7, len(radialprofile), color='red', alpha=0.1, label='Artifact Zone')
    ax.legend(loc='upper right', facecolor='#0e1117', labelcolor='white')
    
    plt.grid(color='#333333', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig