try:
    import numpy as np
    import pandas as pd
    import librosa
    import tensorflow as tf
    print("Tüm kütüphaneler başarıyla yüklendi.")
except ImportError as e:
    print("Bir kütüphane eksik:", e)
