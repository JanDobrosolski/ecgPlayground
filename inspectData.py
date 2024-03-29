import matplotlib.pyplot as plt
import numpy as np 
import json
import scipy.signal as signal
import pandas as pd
import heartpy as hp
import os

SAMPLING_FREQUENCY = 125

if __name__ == "__main__":
    os.makedirs('analysisResults', exist_ok=True)

    dataPaths = []
    for curDir, _, files in os.walk('dataset'):
        dataPaths.extend([os.path.join(curDir, file) for file in files])

    for dataPath in dataPaths:
        subdirPath = os.path.join('analysisResults', dataPath.split(os.sep)[-1][:-4])
        os.makedirs(subdirPath, exist_ok=True)

        # Load the data
        df = pd.read_csv(dataPath, delimiter=',', nrows=1000)

        sampled_rows = df.sample(n=32)

        # Create 4x4 subplots
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))

        # Flatten the axis array for easy iteration
        axs = axs.flatten()

        measuresDict = {}
        processed = 0

        # Iterate over the sampled rows and plot each one in a subplot
        for i, (idx, row) in enumerate(sampled_rows.iterrows()):
            rowPlot = row.to_list()
            rowPlot = rowPlot[:-1]
            
            rowData = row.to_numpy()
            
            rowLabel = rowData[-1]
            rowData = rowData[:-1]
            rowData = rowData[np.where(rowData != 0.)[0]]

            try:
                wd, m = hp.process(rowData, SAMPLING_FREQUENCY)
            except:
                continue

            xLabels = [i for i in range(len(rowPlot))]
            measuresDict[idx] = {}

            rPeakIndices = wd['peaklist']
            rPeakValues = [rowData[j] for j in rPeakIndices]

            # other measures are also available, saving simples ones
            measuresDict[idx]['bpm'] = m['bpm'] #beats per minute
            measuresDict[idx]['ibi'] = m['ibi'] #inter-beat interval
            measuresDict[idx]['sdnn'] = m['sdnn']  #standard deviation of normal-to-normal intervals
            measuresDict[idx]['RR_std'] = wd['rrsd'] #standard deviation of RR intervals
            measuresDict[idx]['RR_list'] = list(wd['RR_list']) #standard deviation of successive differences
            measuresDict[idx]['peakList'] = [int(x) for x in wd['peaklist']] #list of detected peaks
            measuresDict[idx]['peakValues'] = list(rPeakValues) #list of values of detected peaks

            axs[processed].plot(xLabels, rowPlot)
            axs[processed].plot(rPeakIndices, rPeakValues, 'ro')
            axs[processed].set_title(f"Sample {idx}")

            processed += 1

            if processed == 16:
                break

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(subdirPath, "samples.png"))
        plt.close()

        with open(os.path.join(subdirPath, "measures.json"), "w") as f:
            json.dump(measuresDict, f, indent=4)

    # longer sample to showcase how classic analysis is suboptimal for short samples consisting of a single beat
    data, timer = hp.load_exampledata(1)
    sample_rate = hp.get_samplerate_mstimer(timer)

    wd, m = hp.process(data, sample_rate = sample_rate)

    rPeakIndices = wd['peaklist']
    rPeakValues = [data[j] for j in rPeakIndices]

    longSamplePath = os.path.join("analysisResults", "longSample")
    os.makedirs(longSamplePath, exist_ok=True)

    xLabels = [i for i in range(len(data))]
    plt.plot(xLabels, data)
    plt.plot(rPeakIndices, rPeakValues, 'ro')
    plt.savefig(os.path.join(longSamplePath, "longerSample.png"))
    plt.close()

    measuresDict = {}

    measuresDict['bpm'] = m['bpm'] #beats per minute
    measuresDict['ibi'] = m['ibi'] #inter-beat interval
    measuresDict['sdnn'] = m['sdnn']  #standard deviation of normal-to-normal intervals
    measuresDict['RR_std'] = wd['rrsd'] #standard deviation of RR intervals
    measuresDict['RR_list'] = list(wd['RR_list']) #standard deviation of successive differences
    measuresDict['peakList'] = [int(x) for x in wd['peaklist']] #list of detected peaks
    measuresDict['peakValues'] = [int(x) for x in rPeakValues] #list of values of detected peaks

    with open(os.path.join(longSamplePath, "measures.json"), "w") as f:
        json.dump(measuresDict, f, indent=4)
