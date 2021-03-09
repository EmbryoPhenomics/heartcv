.. _quantification:


For some section below:

HeartCV provides a thin wrapper around the primary peak analysis method supplied by scipy (scipy.signal.find_peaks). This wrapper permits the extraction of both peaks and troughs, whereby troughs are identified from the inverted representation of the vector supplied. Additionally, any number of parameters permitted by scipy can be passed to this wrapper, enabling the fine tuning of key arguments such as peak width, prominence and distance between peaks (heartcv.find_events).