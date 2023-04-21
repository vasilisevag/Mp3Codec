import numpy as np
import scipy.fft as sp
#from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from frame import *
from nothing import *
import time
from heapq import heappush, heappop, heapify
from collections import Counter
from itertools import groupby


def Hz2Barks(f):
    """
    Converting Hertz to Barks scale
    Params:
        f: Hertz
    Returns:
        bark
    """
    bark = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan(np.power(f / 7500, 2))
    return bark


# LEVEL 3.1 FUNCTIONS
def make_mp3_analysisfb(h, M):
    """
    Analyse the filterbank to M bands
    Params:
        h: Base function with L length of low pass filter
        M: bands to split
    Returns:
        H matrix LxM analysis filters per band(M collumns)
    """
    L = h.shape[0]
    H = np.ndarray([L, M])
    for rowIdx in range(L):
        for colIdx in range(M):
            H[rowIdx][colIdx] = h[rowIdx] * np.cos(
                ((2 * colIdx + 1) * np.pi * rowIdx) / (2 * M) + ((2 * colIdx + 1) * np.pi) / 4)
    return H


def make_mp3_synthesisfb(h, M):
    """
    Compute the synthesis functions of filterbank for M bands
    Params:
        h: Base function with L length of low pass filter
        M: bands to split
    Returns:
        G matrix LxM synthesis filters per band(col)
    """
    H = make_mp3_analysisfb(h, M)
    G = np.flipud(H)

    return G


def get_column_fourier(H):
    """
    Compute the Fast Fourier transformation of H matrix per each band
    Params:
        H: matrix LxM analysis filters per band(M collumns)
    Returns:
        Hf Fourier transformation of H matrix
    """
    Hf = np.ndarray(H.shape, dtype=np.cdouble)
    for i in range(Hf.shape[1]):
        Hf[:, i] = sp.fft(H[:, i])
    return Hf


def plot_in_hz_in_db_units(Hf):
    """
    Plotting M filters of Hf matrix based on Frequency x-axis
    Params:
        Hf: Fourier transformation of H matrix
    Returns:
        -
    """
    fs = 44100
    fstep = fs / 512
    frequencyXaxis = [x * fstep for x in range(0, 255)]
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0: 255]]
        imagPart = [x.imag for x in Hf[:, i][0: 255]]

        plt.plot(frequencyXaxis, [10 * np.log10(x ** 2 + y ** 2) for x, y in zip(realPart, imagPart)])
    plt.show()

def plot_in_barks_in_db_units(Hf):
    """
    Plotting M filters of Hf matrix based on Barks x-axis
    Params:
        Hf: Fourier transformation of H matrix
    Returns:
        -
    """
    fs = 44100
    fstep = fs / 512
    frequencyXaxis = [x * fstep for x in range(0, 255)]
    barksXaxis = [Hz2Barks(x) for x in frequencyXaxis]
    for i in range(Hf.shape[1]):
        realPart = [x.real for x in Hf[:, i][0: 255]]
        imagPart = [x.imag for x in Hf[:, i][0: 255]]

        plt.plot(barksXaxis, [10 * np.log10(x ** 2 + y ** 2) for x, y in zip(realPart, imagPart)])
    plt.show()


def coder0(wavin, h, M, N):
    subwavinsTotal = wavin.shape[0] // (M * N)
    Ytot = np.ndarray([N * subwavinsTotal, M])
    H = make_mp3_analysisfb(h, M)
    wavin = np.append(wavin, [0 for _ in range(512)])  # padded

    for i in range(subwavinsTotal):
        subwav = wavin[i * (M * N):i * M * N + M * (N - 1) + 512]
        Y = frame_sub_analysis(subwav, H, N)
        Yc = donothing(Y)
        Ytot[i * N:(i + 1) * N, :] = Yc

    return Ytot


def decoder0(Ytot, h, M, N):
    G = make_mp3_synthesisfb(h, M)
    buffSize = M * N
    totalSize = Ytot.shape[0] * Ytot.shape[1]
    xhat = np.ndarray([totalSize])
    for i in range(Ytot.shape[0] // N):
        Yc = Ytot[i * N:(i + 1) * N + h.shape[0] // M, :]
        Yh = idonothing(Yc)
        xhat[i * buffSize: (i + 1) * buffSize] = frame_sub_synthesis(Yh, G)

    return xhat


def codec0(wavin, h, M, N):
    # 4 early steps
    Ytot = coder0(wavin, h, M, N)

    # 2 last steps
    xhat = decoder0(Ytot, h, M, N)

    return xhat, Ytot


##################
# LEVEL 3.2 FUNCTIONS DCT-IV
# https://docs.scipy.org/doc/scipy/tutorial/fft.html#type-iv-dct
def frameDCT(Y):
    # 36x32
    tempC = np.ndarray(Y.shape)
    for i in range(Y.shape[1]):
        tempC[:, i] = sp.dct(Y[:, i], type=4)
    c = tempC.flatten('F')

    return c


def iframeDCT(c):
    M = 32
    N = 36
    tempC = np.reshape(c, (N, M), 'F')
    Yh = np.ndarray((N, M))
    for i in range(M):
        Yh[:, i] = sp.idct(tempC[:, i], type=4)

    return Yh


# LEVEL 3.3 PSYCHOACOUSTIC MODEL

def DCTpower(c):
    # sxesh 10
    P = 10 * np.log10(np.power(c, 2))
    return P


def Dksparse(Kmax):
    matrix = np.zeros([Kmax, Kmax])
    for k in range(Kmax):
        if 2 <= k and k < 282:
            matrix[k][k - 2] = 1
            matrix[k][k + 2] = 1
        elif 282 <= k and k < 570:
            for n in range(2, 14):
                matrix[k][k - n] = 1
                matrix[k][k + n] = 1
        elif 570 <= k and k < Kmax:
            for n in range(2, 28):
                matrix[k][k - n] = 1
                if k + n < Kmax:
                    matrix[k][k + n] = 1

    D = csr_matrix(matrix)
    return D


def ST_init(c, D):
    P = DCTpower(c)
    ST = np.zeros(c.shape[0])
    for i in range(2, c.shape[0] - 1):
        sparserow = D.getrow(i).nonzero()
        _, indices = sparserow
        isTonalComponent = True
        if P[i] <= P[i - 1] or P[i] <= P[i + 1]: 
            isTonalComponent = False
        else:
            for idx in indices:
                if P[i] <= P[idx] + 7: 
                    isTonalComponent = False
                    break
        
        if isTonalComponent:
            ST[i] = 1

    ST = np.where(ST == 1)
    return ST[0]

def MaskPower(c, ST):
    P = DCTpower(c)
    maskerspower = np.ndarray([ST.shape[0]])
    for idx in range(ST.shape[0]):
        val = 0
        for n in range(-1, 2):
            val = val + 10**(0.1*P[ST[idx]+n])
        val = 10*np.log10(val)
        maskerspower[idx] = val
    return maskerspower

def STreduction(ST, c, Tq):
    Pm = MaskPower(c, ST)
    currentMaskers = np.array([])
    for i in range(ST.shape[0]):
        if Pm[i] >= Tq[ST[i]]:
            currentMaskers = np.append(currentMaskers, ST[i])

    fs = 44100
    Tq_scale = fs / 2
    coeffsTotal = Tq.shape[0]
    STr = np.array([])
    leftIdx = 0
    rightIdx = 1

    while(leftIdx < currentMaskers.shape[0] and rightIdx < currentMaskers.shape[0]):
        freqLeftMasker = currentMaskers[leftIdx] * Tq_scale / coeffsTotal
        freqRightMasker = currentMaskers[rightIdx] * Tq_scale / coeffsTotal
        freqDistance = freqRightMasker - freqLeftMasker
        barksDistance = Hz2Barks(freqDistance)

        if barksDistance >= 0.5:
            STr = np.append(STr, currentMaskers[leftIdx])
            leftIdx = rightIdx
            rightIdx = rightIdx + 1
        else:
            if Pm[leftIdx] < Pm[rightIdx]:
                leftIdx = rightIdx
                rightIdx = rightIdx + 1
            else: 
                rightIdx = rightIdx + 1

    STr = STr.astype(int)
    PMr = MaskPower(c, STr)
    return STr, PMr

def SpreadFunc(ST, PM,Kmax):
    # δίνει στην έξοδο τον πίνακα Sf διάστασης
    # (max + 1) × length(ST) έτσι ώστε η j στήλη του να περιέχει τις τιμές του spreading function
    # για το σύνολο των διακριτών συχνοτήτων i = 0, . . . ,Kmax.
    Sf = np.zeros([Kmax+1, ST.shape[0]])
    fs = 44100
    for colIdx in range(ST.shape[0]):
        zfk = Hz2Barks(ST[colIdx] * fs / (2 * (Kmax + 1)))
        for rowIdx in range(Kmax+1):
            zfi = Hz2Barks(rowIdx* fs / (2 * (Kmax + 1)))
            dz = zfi - zfk
            if -3 <= dz and dz < -1:
                Sf[rowIdx][colIdx] = 17*dz - 0.4*PM[colIdx] + 11
            elif -1 <= dz and dz < 0:
                Sf[rowIdx][colIdx] = (0.4*PM[colIdx] + 6)*dz
            elif 0 <= dz and dz < 1:
                Sf[rowIdx][colIdx] = -17*dz
            elif 1 <= dz and dz < 8:
                Sf[rowIdx][colIdx] = (0.15*PM[colIdx]-17)*dz - 0.15*PM[colIdx]

    return Sf

def Masking_Thresholds(ST, PM, Kmax):
    Sf = SpreadFunc(ST, PM, Kmax)
    TM = np.zeros([Kmax+1, ST.shape[0]])

    for colIdx in range(ST.shape[0]):
        zfk = Hz2Barks(ST[colIdx] * fs / (2 * (Kmax + 1)))
        for rowIdx in range(Kmax+1):
            TM[rowIdx][colIdx] = PM[colIdx] - 0.275*zfk + Sf[rowIdx][colIdx] - 6.025

    return TM

def Global_Masking_Thresholds(Ti, Tq):
    Tg = np.ndarray(Tq.shape)
    for i in range(Tq.shape[0]):
        val = 0
        for maskerIdx in range(Ti.shape[1]):
            val = val + 10**(0.1*Ti[i][maskerIdx])
        val = 10*np.log10(10**(0.1*Tq[i]) + val)
        Tg[i] = val

    return Tg

def psycho(c, D):
    # xronobora st init kai masking thresholds . eidika st init
    Tq = np.load('Tq.npy', allow_pickle=True)
    Tq = Tq.flatten()
    for j in range(Tq.shape[0]):
        if np.isnan(Tq[j]):
            Tq[j] = 0

    Kmax = Tq.shape[0] - 1


    st = ST_init(c, D)
    STr, PMr = STreduction(st, c, Tq)
    Ti = Masking_Thresholds(STr, PMr, Kmax)
    Tg = Global_Masking_Thresholds(Ti, Tq)
    return Tg

# LEVEL 3.4 QUANTIZER FUNCTIONS
def critical_bands(K):
    fs = 44100
    cb = np.ndarray([K])
    for idx in range(K):
        fidx = idx * 44100 / (2*K)
        if 0 <= fidx and fidx < 100:
            cb[idx] = 0
        elif 100 <= fidx and fidx < 200:
            cb[idx] = 1
        elif 200 <= fidx and fidx < 300:
            cb[idx] = 2
        elif 300 <= fidx and fidx < 400:
            cb[idx] = 3
        elif 400 <= fidx and fidx < 510:
            cb[idx] = 4
        elif 510 <= fidx and fidx < 630:
            cb[idx] = 5
        elif 630 <= fidx and fidx < 770:
            cb[idx] = 6
        elif 770 <= fidx and fidx < 920:
            cb[idx] = 7
        elif 920 <= fidx and fidx < 1080:
            cb[idx] = 8
        elif 1080 <= fidx and fidx < 1270:
            cb[idx] = 9
        elif 1270 <= fidx and fidx < 1480:
            cb[idx] = 10
        elif 1480 <= fidx and fidx < 1720:
            cb[idx] = 11
        elif 1720 <= fidx and fidx < 2000:
            cb[idx] = 12
        elif 2000 <= fidx and fidx < 2320:
            cb[idx] = 13
        elif 2320 <= fidx and fidx < 2700:
            cb[idx] = 14
        elif 2700 <= fidx and fidx < 3150:
            cb[idx] = 15
        elif 3150 <= fidx and fidx < 3700:
            cb[idx] = 16
        elif 3700 <= fidx and fidx < 4400:
            cb[idx] = 17
        elif 4400 <= fidx and fidx < 5300:
            cb[idx] = 18
        elif 5300 <= fidx and fidx < 6400:
            cb[idx] = 19
        elif 6400 <= fidx and fidx < 7700:
            cb[idx] = 20
        elif 7700 <= fidx and fidx < 9500:
            cb[idx] = 21
        elif 9500 <= fidx and fidx < 12000:
            cb[idx] = 22
        elif 12000 <= fidx and fidx < 15500:
            cb[idx] = 23
        elif 15500 <= fidx:
            cb[idx] = 24

    return cb.astype(int)

def DCT_band_scale(c):
    #Να κατασκευάσετε τη συνάρτηση cs, sc = DCT_band_scale(c) που δέχεται σαν είσοδο τους
    #συντελεστές DCT ενός frame και παράγει τους κανονικοποιημένους συντελεστές ˜c(i) και τα scale
    #factors, ένα για κάθε critical band.
    bandsTotal = 25
    cb = critical_bands(c.shape[0])
    cs = np.ndarray(c.shape[0])
    sc = np.ndarray([bandsTotal])
    bandIdx = 0
    idx = 0
    while bandIdx < bandsTotal:
        maxval = -np.inf
        while idx < 1152 and cb[idx] == bandIdx:
            if maxval < np.power(np.abs(c[idx]), 3/4): maxval = np.power(np.abs(c[idx]), 3/4)
            idx = idx + 1

        sc[bandIdx] = maxval
        bandIdx = bandIdx + 1

    for idx in range(cs.shape[0]):
        cs[idx] = np.sign(c[idx])*(np.power(np.abs(c[idx]), 3/4))/(sc[cb[idx]])

    return cs, sc

def quantizer(x, b):
    symb_index = np.ndarray(x.shape[0])
    levelsTotal = 2**b - 1
    wb = 1 / levelsTotal
    for symbidx in range(x.shape[0]):
        for i in range(levelsTotal):
            if(i*wb <= np.abs(x[symbidx]) and np.abs(x[symbidx]) <= (i+1)*wb):
                if(x[symbidx] > 0): symb_index[symbidx] = i
                else: symb_index[symbidx] = -i

    return symb_index

def dequantizer(symb_index, b):
    # Να κατασκευάσετε τη συνάρτηση xh = dequantizer(symb_index, b) που αντιστρέφει την προηγούμενη.
    xh = np.ndarray(symb_index.shape[0])
    levelsTotal = 2**b - 1
    wb = 1 / levelsTotal
    center_wb = wb/2
    for symbIdx in range(symb_index.shape[0]):
        for i in range(levelsTotal):
            if(symb_index[symbIdx] == 0):
                xh[symbIdx] = 0
            elif symb_index[symbIdx] > 0:
                xh[symbIdx] = symb_index[symbIdx]*wb + center_wb
            elif symb_index[symbIdx] < 0:
                xh[symbIdx] = symb_index[symbIdx]*wb - center_wb    
    
    return xh

def all_bands_quantizer(c, Tg):
    # (α) ένα διάνυσμα
    # με τα αντίστοιχα σύμβολα κβαντισμού (ακέραιους όπως αυτούς της quantizer()), (β) τους scale
    # factors SF (που παράγονται από τηνDCT_band_scale()) και (γ) τον αριθμό των bits που χρησιμοποίησε
    # ο κβαντιστής σε κάθε critical band.
    # c 1152
    # Tg 1152
    cb = critical_bands(c.shape[0])

    cs, sc = DCT_band_scale(c)
    b = 1
    symb_index_all = np.ndarray(c.shape[0])

    c_all = np.ndarray(c.shape[0])

    B = np.ndarray(sc.shape[0])

    for bandId in range(sc.shape[0]):
        bandsDisbanded = np.where(cb == bandId)
        bandsDisbanded = bandsDisbanded[0]
        b = 1
        while True:
            
            symb_index = quantizer(cs[bandsDisbanded], b)
            c_tone = dequantizer(symb_index, b)
            cs_tone = np.sign(c_tone)*(np.power(np.abs(c_tone) * sc[bandId], 4/3))
            err = np.abs(c[bandsDisbanded] - cs_tone)
            Pb = 10 * np.log10(np.power(err, 2))
            if (Pb <= Tg[bandsDisbanded]).all():
                symb_index_all[bandsDisbanded] = symb_index  
                break
            b = b + 1

        B[bandId] = b
    return symb_index_all, sc, B.astype(int)

def all_bands_dequantizer(symb_index,B, SF):
    bandsTotal = SF.shape[0]
    cb = critical_bands(symb_index.shape[0])
    xh = np.ndarray(symb_index.shape[0])

    for bandId in range(bandsTotal):
        bandsDisbanded = np.where(cb == bandId)
        bandsDisbanded = bandsDisbanded[0].astype(int)
        c_tone = dequantizer(symb_index[bandsDisbanded], B[bandId])
        xh[bandsDisbanded] = np.sign(c_tone)*(np.power(np.abs(c_tone) * SF[bandId], 4/3))
    # Denormalized c = xh
    return xh

# LEVEL 3.5 RUN LENGTH ENCODING
# xwrista gia ka8e frame ~ component : 1152 DCT syntelestes 

def RLE(symb_index, K):
    # R x 2 , 1h sthlh symbolo, 2h sthlh epanalhpsh autou sta epomena 
    R = [[0, 0]]
    i = 0
    while (i <= K - 1):
        count = 1
        ch = symb_index[i]
        j = i
        while (j < K - 1):   
            if (symb_index[j] == symb_index[j + 1]): 
                count = count + 1
                j = j + 1
            else: 
                break
        R = np.append(R, [[ch, count]], axis=0)
        i = j + 1

    run_symbols = R[1:R.shape[0], :]
    return run_symbols.astype(int)

def RLEreverse(run_symbols, K):
    symb_index = []
    i = 0
    for i in range(run_symbols.shape[0]):
        symb_index = np.append(symb_index, [run_symbols[i][0] for _ in range(run_symbols[i][1])])
                               
    return symb_index.astype(int)

# LEVEL 3.6 HUFFMAN CODING
def rle_to_string(rle_data):
    string = ""
    for symbol, count in rle_data:
        string += str(symbol)+str(count) #symbol * count
    return string

#huffman_encoded, huffman_codes = huffman_encode(rle_string)
# Επισήμανση: είναι πιθανόν το δημιουργούμενο bitstream που προκύπτει από την αλληλουχία των
# frame_stream να είναι υπερβολικά μεγάλο για τη μνήμη του υπολογιστή σας. Σ’ αυτή την περίπτωση
# φροντίστε να γράφετε τα αποτελέσματα κάθε κλήσης της huff() σε ένα αρχείο ascii το οποίο θα
# διαβάζετε στη συνέχεια κατά την αποκωδικοποίηση.
def huffman_encode(data):
    freq = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapify(heap)
    while len(heap) > 1:
        low = heappop(heap)
        high = heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    codes = dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
    encoded = ''.join([codes[symbol] for symbol in data])
    return encoded, codes


def huffman_decode(encoded, codes):
    decoded = []
    code = ''
    for bit in encoded:
        code += bit
        for symbol, value in codes.items():
            if value == code:
                decoded.append(symbol)
                code = ''
                break
    return ''.join(decoded)

def run_length_encode(data):
    """Run-length encode a 1D numpy array"""
    return [(k, len(list(g))) for k, g in groupby(data)]

def run_length_decode(runs):
    """Run-length decode a list of (symbol, run-length) tuples"""
    data = []
    for symbol, run_length in runs:
        data.extend([symbol] * run_length)
    return np.array(data)


def MP3cod(wavin, h, M, N):
    subwavinsTotal = wavin.shape[0] // (M * N)
    Ytot = np.ndarray([N * subwavinsTotal, M])
    H = make_mp3_analysisfb(h, M)

    wavin = np.append(wavin, [0 for _ in range(512)])  # padded

    #cb = critical_bands(M*N)


    D = Dksparse(M * N - 1)

    for i in range(subwavinsTotal):
        subwav = wavin[i * (M * N):i * M * N + M * (N - 1) + 512]
        Y = frame_sub_analysis(subwav, H, N)

        # ########################################################################
        c = frameDCT(Y)
        Tg = psycho(c, D)
        Tg = Tg - 15
        symb_index, SF,B = all_bands_quantizer(c, Tg)
        run_symbols = RLE(symb_index, symb_index.shape[0])
        stringRLE = rle_to_string(run_symbols)
        huffman_encoded, huffman_codes = huffman_encode(stringRLE)


        

        # ascii_data = binary_to_ascii(huffman_encoded)
        # with open("encoded_data.txt", "wb") as f:
        #     f.write(ascii_data)


        # # Read the encoded data back from the file and decode it
        # with open("encoded_data.txt", "rb") as f:
        #     ascii_data = f.read()
        # binary_data = ascii_to_binary(ascii_data)

        frame_stream = huffman_encoded
        decoded_data = huffman_decode(frame_stream, huffman_codes)

        decoded_symbols = [int(s) for s in decoded_data]

        decoded_rle = [(c, len(list(g))) for c, g in groupby(decoded_symbols)]
        run_symbolsREV = np.array(decoded_rle)
        print(run_symbolsREV)
        print("auta pou gyrisan panw")
        #print(run_symbolsREV)
        #print(f"Decoded RLE-encoded data: {decoded_rle}")
        # run_symbols = ihuff(frame_stream, frame_symbol_prob)
        symb_index = RLEreverse(run_symbolsREV, symb_index.shape[0])

        xh = all_bands_dequantizer(symb_index, B, SF)
        tempY = iframeDCT(xh)
        Y = tempY
        # #######################################################
        Yc = donothing(Y)
        Ytot[i * N:(i + 1) * N, :] = Yc

    return Ytot


def MP3decod(Ytot, h, M, N):
    G = make_mp3_synthesisfb(h, M)
    buffSize = M * N
    totalSize = Ytot.shape[0] * Ytot.shape[1]
    xhat = np.ndarray([totalSize])
    for i in range(Ytot.shape[0] // N):
        Yc = Ytot[i * N:(i + 1) * N + h.shape[0] // M, :]
        # if i == 100:
        #     print(Yc.shape)
        #     xTemp = frame_sub_synthesis(Yc, G)
        #     print(xTemp.shape)
        #     print(type(xTemp[1]))

        Yh = idonothing(Yc)
        xhat[i * buffSize: (i + 1) * buffSize] = frame_sub_synthesis(Yh, G)

    return xhat


def MP3codec(wavin, h, M, N):
    # 4 early steps
    Ytot = MP3cod(wavin, h, M, N)

    # 2 last steps
    xhat = MP3decod(Ytot, h, M, N)

    return xhat, Ytot



# UTILITIES 
def zerosPrinter(arr):
    nz_arr = np.count_nonzero(arr)
    z_arr = arr.size - nz_arr
    print(f"Total size: {arr.size}")
    print(f"number of non-zero: {nz_arr}" )
    print(f"number of zeros: {z_arr}")  
    return z_arr

def plotterV1(wavin, shifted_xhat, start, end):
    error = wavin - shifted_xhat
    # error = wavin - xhatscaled
    powS = np.mean(np.power(shifted_xhat, 2))
    powN = np.mean(np.power(error, 2))
    snr = 10 * np.log10((powS - powN) / powN)
    # error projection

    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    plt.subplot(2, 1, 1)
    plt.plot(wavin[start:end])
    plt.title("MyFile Wavin")
    plt.subplot(2, 1, 2)
    plt.plot(shifted_xhat[start:end])
    plt.title("Decoded Shifted Wavin")
    plt.show()

    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    plt.title("Error between input and decoded wavin file(SNR = %1.5f dB)" % snr)
    plt.plot(error[start:end])
    plt.show()

def writerV1(xhat, Ytot, wavin, start, end):
    xhatscaled = np.int16(xhat * 32767 / np.max(np.abs(xhat)))
    write("testDec2.wav", fs, xhatscaled)

    xhatscaled = read("testDec2.wav")
    xhatscaled = np.array(xhatscaled[1], dtype=float)

    xhatscaled = xhatscaled * np.max(np.abs(wavin)) / np.max(np.abs(xhatscaled))

    # prin thn sysxetish
    # plotterV1(wavin, xhatscaled, start, end)

    # meta thn sysxetish
    allRhos = np.ndarray((1000, 1))

    for i in range(1000):
        shifted_xhat = np.roll(xhatscaled, i)
        rho = np.corrcoef(wavin, shifted_xhat)
        allRhos[i] = rho[0][1]

        # rho[0][1]


    maxRhoIdx = np.argmax(allRhos)
    shifted_xhat = np.roll(xhatscaled, maxRhoIdx)

    plotterV1(wavin, shifted_xhat, start, end)
    shifted_xhat = np.int16(shifted_xhat * 32767 / np.max(np.abs(shifted_xhat)))
    write("testDec3.wav", fs, shifted_xhat)


def writerV2(xhat2, Ytot2, wavin, start, end):
    xhatscaled2 = np.int16(xhat2 * 32767 / np.max(np.abs(xhat2)))
    write("testMP3Dec2.wav", fs, xhatscaled2)

    xhatscaled2 = read("testMP3Dec2.wav")
    xhatscaled2 = np.array(xhatscaled2[1], dtype=float)

    xhatscaled2 = xhatscaled2 * np.max(np.abs(wavin)) / np.max(np.abs(xhatscaled2))

    # prin thn sysxetish
    # plotterV1(wavin, xhatscaled, start, end)
    # meta thn sysxetish
    allRhos = np.ndarray((1000, 1))

    for i in range(1000):
        shifted_xhat = np.roll(xhatscaled2, i)
        rho = np.corrcoef(wavin, shifted_xhat)
        # print(rho)
        allRhos[i] = rho[0][1]

        # rho[0][1]


    maxRhoIdx = np.argmax(allRhos)
    shifted_xhat = np.roll(xhatscaled2, maxRhoIdx)

    plotterV1(wavin, shifted_xhat, start, end)
    shifted_xhat = np.int16(shifted_xhat * 32767 / np.max(np.abs(shifted_xhat)))
    write("testMP3Dec3.wav", fs, shifted_xhat)

# LEVEL 3.1 FILTERBANK EXECUTION
# ANSWER TO 1-3
fs = 44100
M = 32
N = 36
# ((N-1) + L/M)*M = 51x32 = 1632
start = 500
end = 800

data_h = np.load('h.npy', allow_pickle=True)
h = data_h[()]['h']
H = make_mp3_analysisfb(h, M)
Hf = get_column_fourier(H)
# plot_in_hz_in_db_units(Hf)
# plot_in_barks_in_db_units(Hf)

# ANSWER TO 4
wavin = read("myfile.wav")
wavin = np.array(wavin[1], dtype=float)
# print(type(wavin[2]))

# xhat, Ytot = codec0(wavin, h, M, N)
# writerV1(xhat, Ytot, wavin, start, end)

# FINAL MP3 COMPOSITION AND EXECUTION OF MODELS
xhat2, Ytot2 = MP3codec(wavin, h, M, N)
writerV2(xhat2, Ytot2, wavin, start, end)

# 2 * 10^6 to max se float




