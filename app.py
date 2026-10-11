import sys
import numpy as np
import heapq
from collections import Counter
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, 
    QPushButton, QPlainTextEdit, QFormLayout, QSizePolicy, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QFileDialog,
    QStatusBar, QToolBox
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtMultimedia import QSoundEffect

import scipy.io.wavfile as wavfile
from scipy.signal import firwin, lfilter, freqz, group_delay
import tempfile
import os

# ==========================================================
# Matplotlib Canvas Wrapper (Dark Theme)
# ==========================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        self.axes.tick_params(colors='#cccccc')
        for spine in self.axes.spines.values():
            spine.set_color('#cccccc')
        self.axes.grid(True, color='#3e3e3e', linestyle='--')
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

# ==========================================================
# DSP & Coding Helper Functions
# ==========================================================
def hamming_encode(bits):
    G = np.array([[1, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0, 1]], dtype=int)
    return np.dot(bits, G) % 2

def hamming_decode(bits):
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=int)
    syndrome = np.dot(bits, H.T) % 2
    s_dec = syndrome[:, 0]*4 + syndrome[:, 1]*2 + syndrome[:, 2]
    corrected = bits.copy()
    for i in range(1, 8):
        mask = (s_dec == i)
        corrected[mask, i-1] = (corrected[mask, i-1] + 1) % 2
    return corrected[:, 3:7]

# Rate 1/2 Convolutional Code (K=3)
# Generators: g1 = [1,1,1] (7), g2 = [1,0,1] (5)
def conv_encode(bits):
    state = np.array([0, 0], dtype=int)
    coded = []
    for b in bits:
        c1 = b ^ state[0] ^ state[1]
        c2 = b ^ state[1]
        coded.extend([c1, c2])
        state = np.array([b, state[0]], dtype=int)
    return np.array(coded, dtype=int)

def viterbi_decode(coded_bits):
    # Trellis: {state: {input: (next_state, output_dec)}}
    # States: 00(0), 01(1), 10(2), 11(3)
    trellis = {
        0: {0: (0, 0), 1: (2, 7)},
        1: {0: (0, 2), 1: (2, 5)},
        2: {0: (1, 3), 1: (3, 4)},
        3: {0: (1, 1), 1: (3, 6)}
    }
    num_steps = len(coded_bits) // 2
    pm = np.full(4, np.inf)
    pm[0] = 0
    paths = np.zeros((num_steps, 4), dtype=int)
    
    for step in range(num_steps):
        r = coded_bits[step*2 : step*2+2]
        r_dec = r[0]*2 + r[1]
        new_pm = np.full(4, np.inf)
        new_paths = np.zeros(4, dtype=int)
        for s in range(4):
            if pm[s] == np.inf: continue
            for u in [0, 1]:
                next_s, out_dec = trellis[s][u]
                dist = bin(r_dec ^ out_dec).count('1')
                if pm[s] + dist < new_pm[next_s]:
                    new_pm[next_s] = pm[s] + dist
                    new_paths[next_s] = s
        pm = new_pm
        paths[step] = new_paths
        
    best_state = np.argmin(pm)
    decoded = []
    for step in range(num_steps-1, -1, -1):
        prev_state = paths[step, best_state]
        for u in [0, 1]:
            if trellis[prev_state][u][0] == best_state:
                decoded.append(u)
                break
        best_state = prev_state
        
    return np.array(decoded[::-1], dtype=int)

def huffman_decode(encoded_str, codes):
    if not encoded_str: return ""
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_text = ""
    for bit in encoded_str:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""
    return decoded_text

def get_raised_cosine(sps, alpha, length=6):
    t_max = length * sps
    t = np.arange(-t_max, t_max+1) / sps
    h = np.zeros_like(t)
    for i, x in enumerate(t):
        if x == 0: h[i] = 1.0
        elif alpha != 0 and abs(x) == 1.0/(2*alpha): h[i] = np.pi / (4 * alpha)
        else:
            num = np.sin(np.pi * x * (1 - alpha)) + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha))
            den = np.pi * x * (1 - (4 * alpha * x)**2)
            h[i] = num / den
    return h / np.sum(h) * sps

def calculate_lzw_bits(text):
    if not text: return 0
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []
    for c in text:
        wc = w + c
        if wc in dictionary: w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    if w: result.append(dictionary[w])
        
    bits = 0
    current_bits = 8
    max_codes = 256
    for _ in result:
        bits += current_bits
        max_codes += 1
        if max_codes > 2**current_bits: current_bits += 1
    return bits

class Node:
    def __init__(self, freq, char=None, left=None, right=None):
        self.freq = freq; self.char = char; self.left = left; self.right = right

def build_huffman_tree_node(probabilities):
    heap = [[weight, Node(weight, char)] for char, weight in probabilities.items()]
    heapq.heapify(heap)
    if not heap: return None
    while len(heap) > 1:
        lo = heapq.heappop(heap)[1]; hi = heapq.heappop(heap)[1]
        merged = Node(lo.freq + hi.freq, left=lo, right=hi)
        heapq.heappush(heap, [merged.freq, merged])
    return heapq.heappop(heap)[1]

# ==========================================================
# Tab 1: Advanced Signal Analysis
# ==========================================================
class SignalAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.sound_effect = QSoundEffect()
        self.last_signal = np.array([])
        self.last_fs = 44100
        self.I_baseband = np.array([])
        self.Q_baseband = np.array([])
        
        self.toolbox = QToolBox()
        
        # 1. Signal Generator
        gen_widget = QWidget()
        gen_layout = QFormLayout(gen_widget)
        self.signal_type = QComboBox()
        self.signal_type.addItems(["Sine Wave", "Square Wave", "Sum of Sines (3f)", "Linear Chirp", "BPSK", "QPSK", "16-QAM"])
        gen_layout.addRow("Type:", self.signal_type)
        self.amp_spin = QDoubleSpinBox(); self.amp_spin.setRange(0.1, 10.0); self.amp_spin.setValue(1.0)
        gen_layout.addRow("Amplitude:", self.amp_spin)
        self.freq_spin = QDoubleSpinBox(); self.freq_spin.setRange(0.1, 5000.0); self.freq_spin.setValue(440.0)
        gen_layout.addRow("Freq / Sym Rate (Hz):", self.freq_spin)
        self.phase_spin = QDoubleSpinBox(); self.phase_spin.setRange(0.0, 360.0); self.phase_spin.setValue(0.0)
        gen_layout.addRow("Phase (°):", self.phase_spin)
        self.pulse_combo = QComboBox(); self.pulse_combo.addItems(["Rectangular", "Raised Cosine (α=0.35)"])
        gen_layout.addRow("Pulse Shape:", self.pulse_combo)
        self.toolbox.addItem(gen_widget, "📡 Signal Generator")
        
        # 2. Environment & Modulation
        env_widget = QWidget()
        env_layout = QFormLayout(env_widget)
        self.sample_rate_spin = QDoubleSpinBox(); self.sample_rate_spin.setRange(1.0, 44100.0); self.sample_rate_spin.setValue(5000.0)
        env_layout.addRow("Sample Rate (Hz):", self.sample_rate_spin)
        self.duration_spin = QDoubleSpinBox(); self.duration_spin.setRange(0.1, 5.0); self.duration_spin.setValue(1.0)
        env_layout.addRow("Duration (s):", self.duration_spin)
        self.snr_spin = QDoubleSpinBox(); self.snr_spin.setRange(-20.0, 100.0); self.snr_spin.setValue(100.0)
        env_layout.addRow("SNR (dB):", self.snr_spin)
        self.impulse_spin = QDoubleSpinBox(); self.impulse_spin.setRange(0.0, 1.0); self.impulse_spin.setValue(0.0); self.impulse_spin.setSingleStep(0.01)
        env_layout.addRow("Impulse Noise Prob:", self.impulse_spin)
        self.bits_spin = QSpinBox(); self.bits_spin.setRange(1, 16); self.bits_spin.setValue(16)
        env_layout.addRow("Quantization (Bits):", self.bits_spin)
        self.downsample_spin = QSpinBox(); self.downsample_spin.setRange(1, 10); self.downsample_spin.setValue(1)
        env_layout.addRow("Downsample Factor (M):", self.downsample_spin)
        self.toolbox.addItem(env_widget, "🌍 Environment & AWGN")
        
        # 3. DSP & Filters
        dsp_widget = QWidget()
        dsp_layout = QVBoxLayout(dsp_widget)
        mod_layout = QVBoxLayout()
        self.am_check = QCheckBox("AM Modulate (Carrier: 1000 Hz)")
        mod_layout.addWidget(self.am_check)
        fm_layout = QHBoxLayout()
        self.fm_check = QCheckBox("FM Modulate (Dev: ")
        self.fm_check.toggled.connect(self.toggle_fm)
        self.kf_spin = QDoubleSpinBox(); self.kf_spin.setRange(1.0, 1000.0); self.kf_spin.setValue(500.0); self.kf_spin.setEnabled(False)
        fm_layout.addWidget(self.fm_check); fm_layout.addWidget(self.kf_spin); fm_layout.addWidget(QLabel("Hz)"))
        mod_layout.addLayout(fm_layout)
        dsp_layout.addLayout(mod_layout)
        
        sys_layout = QHBoxLayout(); sys_layout.addWidget(QLabel("LTI System:"))
        self.sys_combo = QComboBox(); self.sys_combo.addItems(["None", "Moving Average (LPF)", "Echo (Delay)", "Envelope Detector"])
        sys_layout.addWidget(self.sys_combo); dsp_layout.addLayout(sys_layout)
        
        fir_group = QGroupBox("Custom FIR Filter (Window-Sinc)")
        fir_layout = QFormLayout(fir_group)
        self.fir_check = QCheckBox("Enable FIR Filter")
        fir_layout.addRow(self.fir_check)
        self.fir_type_combo = QComboBox()
        self.fir_type_combo.addItems(["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"])
        fir_layout.addRow("Filter Type:", self.fir_type_combo)
        self.fir_taps_spin = QSpinBox(); self.fir_taps_spin.setRange(11, 101); self.fir_taps_spin.setValue(31); self.fir_taps_spin.setSingleStep(2)
        fir_layout.addRow("Filter Taps (Order):", self.fir_taps_spin)
        self.fir_cutoff1_spin = QDoubleSpinBox(); self.fir_cutoff1_spin.setRange(1.0, 20000.0); self.fir_cutoff1_spin.setValue(500.0)
        fir_layout.addRow("Cutoff 1 (Hz):", self.fir_cutoff1_spin)
        self.fir_cutoff2_spin = QDoubleSpinBox(); self.fir_cutoff2_spin.setRange(1.0, 20000.0); self.fir_cutoff2_spin.setValue(1000.0); self.fir_cutoff2_spin.setEnabled(False)
        fir_layout.addRow("Cutoff 2 (Hz):", self.fir_cutoff2_spin)
        self.fir_type_combo.currentIndexChanged.connect(self.toggle_fir_cutoffs)
        dsp_layout.addWidget(fir_group)
        
        win_layout = QHBoxLayout(); win_layout.addWidget(QLabel("FFT Window:"))
        self.window_combo = QComboBox(); self.window_combo.addItems(["Rectangular", "Hann", "Hamming", "Blackman"])
        win_layout.addWidget(self.window_combo); dsp_layout.addLayout(win_layout)
        self.db_check = QCheckBox("Show FFT Magnitude in dB"); dsp_layout.addWidget(self.db_check)
        self.toolbox.addItem(dsp_widget, "🎛️ DSP & Filters")
        
        btn_layout = QVBoxLayout()
        self.update_btn = QPushButton("Generate & Analyze")
        self.update_btn.setStyleSheet("font-weight: bold; background-color: #007acc; color: white; padding: 10px;")
        self.update_btn.clicked.connect(self.update_plot)
        self.play_btn = QPushButton("▶ Play Signal Audio")
        self.play_btn.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 8px;")
        self.play_btn.clicked.connect(self.play_audio)
        self.export_btn = QPushButton("💾 Export to WAV")
        self.export_btn.setStyleSheet("font-weight: bold; background-color: #6c757d; color: white; padding: 8px;")
        self.export_btn.clicked.connect(self.export_audio)
        btn_layout.addWidget(self.update_btn); btn_layout.addWidget(self.play_btn); btn_layout.addWidget(self.export_btn)
        
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.toolbox, 1)
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()
        
        plot_tabs = QTabWidget()
        tf_tab = QWidget(); tf_layout = QVBoxLayout(tf_tab)
        self.time_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.freq_canvas = MplCanvas(self, width=5, height=3, dpi=100)
        tf_layout.addWidget(QLabel("<b>Time Domain</b>")); tf_layout.addWidget(self.time_canvas)
        tf_layout.addWidget(QLabel("<b>Frequency Domain (FFT)</b>")); tf_layout.addWidget(self.freq_canvas)
        plot_tabs.addTab(tf_tab, "Time / Freq")
        
        spec_tab = QWidget(); spec_layout = QVBoxLayout(spec_tab)
        self.spec_canvas = MplCanvas(self, width=5, height=6, dpi=100); spec_layout.addWidget(self.spec_canvas)
        plot_tabs.addTab(spec_tab, "Spectrogram")
        
        const_tab = QWidget(); const_layout = QVBoxLayout(const_tab)
        self.const_canvas = MplCanvas(self, width=5, height=6, dpi=100); const_layout.addWidget(self.const_canvas)
        plot_tabs.addTab(const_tab, "Constellation (IQ)")
        
        eye_tab = QWidget(); eye_layout = QVBoxLayout(eye_tab)
        self.eye_canvas = MplCanvas(self, width=5, height=6, dpi=100); eye_layout.addWidget(self.eye_canvas)
        plot_tabs.addTab(eye_tab, "Eye Diagram")
        
        fir_tab = QWidget(); fir_layout = QVBoxLayout(fir_tab)
        self.fir_canvas = MplCanvas(self, width=5, height=6, dpi=100); fir_layout.addWidget(self.fir_canvas)
        plot_tabs.addTab(fir_tab, "Filter Response")

        self.layout.addLayout(controls_layout, 1)
        self.layout.addWidget(plot_tabs, 3)
        self.update_plot()

    def toggle_fm(self, checked): self.kf_spin.setEnabled(checked)
    def toggle_fir_cutoffs(self, index):
        is_band = "Band" in self.fir_type_combo.currentText()
        self.fir_cutoff2_spin.setEnabled(is_band)

    def play_audio(self):
        if len(self.last_signal) == 0: return
        y_norm = self.last_signal / (np.max(np.abs(self.last_signal)) + 1e-6)
        y_pcm = np.int16(y_norm * 32767)
        temp_path = os.path.join(tempfile.gettempdir(), "dsp_sig.wav")
        try:
            wavfile.write(temp_path, int(self.last_fs), y_pcm)
            self.sound_effect.setSource(QUrl.fromLocalFile(temp_path))
            self.sound_effect.setVolume(0.8); self.sound_effect.play()
        except Exception as e: print(f"Audio playback error: {e}")

    def export_audio(self):
        if len(self.last_signal) == 0: return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", "WAV Files (*.wav)")
        if file_name:
            try:
                y_norm = self.last_signal / (np.max(np.abs(self.last_signal)) + 1e-6)
                y_pcm = np.int16(y_norm * 32767)
                wavfile.write(file_name, int(self.last_fs), y_pcm)
            except Exception as e: print(f"Export error: {e}")

    def update_plot(self):
        sig_type = self.signal_type.currentText()
        amp = self.amp_spin.value(); freq = self.freq_spin.value(); phase = np.deg2rad(self.phase_spin.value())
        fs = self.sample_rate_spin.value(); dur = self.duration_spin.value(); snr_db = self.snr_spin.value()
        imp_prob = self.impulse_spin.value(); bits = self.bits_spin.value(); m_down = self.downsample_spin.value()

        t = np.linspace(0, dur, int(fs * dur), endpoint=False)
        y_clean = np.zeros_like(t); self.I_baseband = np.zeros_like(t); self.Q_baseband = np.zeros_like(t)
        
        if sig_type == "Sine Wave": y_clean = amp * np.sin(2 * np.pi * freq * t + phase)
        elif sig_type == "Square Wave": y_clean = amp * np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif sig_type == "Sum of Sines (3f)": y_clean = amp * np.sin(2 * np.pi * freq * t + phase) + (amp/3) * np.sin(2 * np.pi * (freq*3) * t + phase)
        elif sig_type == "Linear Chirp":
            k = (freq * 5 - freq) / dur; y_clean = amp * np.sin(2 * np.pi * (freq * t + 0.5 * k * t**2) + phase)
        elif sig_type in ["BPSK", "QPSK", "16-QAM"]:
            sps = int(fs / freq); num_sym = int(dur * freq)
            if sig_type == "BPSK":
                I = np.where(np.random.randint(0, 2, num_sym) == 0, -1, 1) * amp; Q = np.zeros_like(I)
            elif sig_type == "QPSK":
                b = np.random.randint(0, 4, num_sym); I = np.cos(np.pi/4 + b * np.pi/2) * amp; Q = np.sin(np.pi/4 + b * np.pi/2) * amp
            elif sig_type == "16-QAM":
                # 4 bits per symbol. I and Q take values from [-3, -1, 1, 3] * amp/3
                I_levels = np.array([-3, -1, 1, 3]) * amp/3
                Q_levels = np.array([-3, -1, 1, 3]) * amp/3
                I = I_levels[np.random.randint(0, 4, num_sym)]
                Q = Q_levels[np.random.randint(0, 4, num_sym)]
                
            I_up = np.zeros(num_sym * sps); Q_up = np.zeros(num_sym * sps)
            I_up[::sps] = I; Q_up[::sps] = Q
            h = get_raised_cosine(sps, 0.35) if self.pulse_combo.currentText() == "Raised Cosine (α=0.35)" else np.ones(sps)
            I_shaped = np.convolve(I_up, h, mode='same'); Q_shaped = np.convolve(Q_up, h, mode='same')
            self.I_baseband = I_shaped[:len(t)]; self.Q_baseband = Q_shaped[:len(t)]
            y_clean = I_shaped * np.cos(2 * np.pi * (freq * 2) * t) - Q_shaped * np.sin(2 * np.pi * (freq * 2) * t)
            
        if self.am_check.isChecked(): y_clean = (1 + y_clean / (np.max(np.abs(y_clean))+1e-6)) * np.cos(2 * np.pi * 1000.0 * t)
        if self.fm_check.isChecked():
            integral_msg = np.cumsum(y_clean) / fs
            y_clean = amp * np.cos(2 * np.pi * 1000.0 * t + 2 * np.pi * self.kf_spin.value() * integral_msg)

        # AWGN
        if snr_db < 100.0:
            noise_power = np.mean(y_clean**2) / (10**(snr_db / 10))
            y = y_clean + np.sqrt(noise_power) * np.random.randn(len(t))
        else: y = y_clean.copy()
            
        # Impulse Noise (Bernoulli-Gaussian)
        if imp_prob > 0:
            mask = np.random.rand(len(t)) < imp_prob
            y[mask] += np.random.randn(np.sum(mask)) * (np.max(np.abs(y_clean)) * 2.0)
            
        max_val = max(np.max(np.abs(y)), 1e-6)
        if bits < 16: y = ((2 * max_val) / 2**bits) * np.round(y / ((2 * max_val) / 2**bits))

        sys_type = self.sys_combo.currentText()
        if sys_type != "None":
            if sys_type == "Moving Average (LPF)": y = np.convolve(y, np.array([1/3, 1/3, 1/3]), mode='same')
            elif sys_type == "Echo (Delay)":
                ds = int(0.05 * fs); k = np.zeros(ds + 1); k[0] = 1.0; k[-1] = 0.6; y = np.convolve(y, k, mode='full')[:len(t)]
            elif sys_type == "Envelope Detector":
                mk = np.ones(int(fs/50)) / int(fs/50); y = np.convolve(np.abs(y), mk, mode='same')

        y_filtered = None; fir_taps = None
        if self.fir_check.isChecked():
            numtaps = self.fir_taps_spin.value(); ftype = self.fir_type_combo.currentText(); nyq = fs / 2.0
            c1 = self.fir_cutoff1_spin.value() / nyq; c2 = self.fir_cutoff2_spin.value() / nyq
            try:
                if ftype == "Low-Pass": fir_taps = firwin(numtaps, c1, pass_zero=True)
                elif ftype == "High-Pass": fir_taps = firwin(numtaps, c1, pass_zero=False)
                elif ftype == "Band-Pass": fir_taps = firwin(numtaps, [c1, c2], pass_zero=False)
                elif ftype == "Band-Stop": fir_taps = firwin(numtaps, [c1, c2], pass_zero=True)
                y_filtered = lfilter(fir_taps, 1.0, y); y = y_filtered
            except Exception as e: print(f"FIR Error: {e}")

        self.last_signal = y; self.last_fs = fs
        if m_down > 1: y = y[::m_down]; t = t[::m_down]; fs = fs / m_down

        n = len(y); win_type = self.window_combo.currentText()
        if win_type == "Hann": window = np.hanning(n)
        elif win_type == "Hamming": window = np.hamming(n)
        elif win_type == "Blackman": window = np.blackman(n)
        else: window = np.ones_like(n)
            
        yf = np.fft.fft(y * window); xf = np.fft.fftfreq(n, 1/fs)[:n//2]
        yf_mag = 2.0/n * np.abs(yf[0:n//2])

        self.time_canvas.axes.clear()
        draw_style = 'steps-mid' if bits < 8 else 'default'
        if y_filtered is not None:
            self.time_canvas.axes.plot(t, self.last_signal[:len(t)] if m_down==1 else self.last_signal[::m_down], 'b-', alpha=0.3, linewidth=1, label="Pre-Filter")
            self.time_canvas.axes.plot(t, y, 'r-', linewidth=1.5, drawstyle=draw_style, label="FIR Filtered")
        else:
            self.time_canvas.axes.plot(t, y, 'b-', linewidth=1.5, drawstyle=draw_style, label="Signal")
            if snr_db < 100 or bits < 16 or imp_prob > 0:
                 self.time_canvas.axes.plot(t, y_clean[::m_down], 'g--', alpha=0.5, linewidth=1, label="Clean Analog")
        if bits <= 4:
            for lvl in np.linspace(-max_val, max_val, 2**bits): self.time_canvas.axes.axhline(lvl, color='cyan', linestyle=':', alpha=0.3)
        self.time_canvas.axes.set_title(f"Signal (fs={fs:.1f}Hz, SNR={snr_db}dB)", color='white')
        self.time_canvas.axes.set_xlabel("Time (s)", color='white'); self.time_canvas.axes.set_ylabel("Amplitude", color='white')
        self.time_canvas.axes.set_xlim(0, min(dur, 0.2)); self.time_canvas.axes.legend(facecolor='#2b2b2b', edgecolor='#cccccc', labelcolor='white')
        self.time_canvas.fig.tight_layout(); self.time_canvas.draw()

        self.freq_canvas.axes.clear()
        if self.db_check.isChecked():
            mag_db = 20 * np.log10(yf_mag + 1e-12)
            self.freq_canvas.axes.plot(xf, mag_db, 'r-', linewidth=1.5)
            self.freq_canvas.axes.set_ylabel("Magnitude (dB)", color='white'); self.freq_canvas.axes.set_ylim(np.max(mag_db) - 80, np.max(mag_db) + 5)
        else:
            self.freq_canvas.axes.plot(xf, yf_mag, 'r-', linewidth=1.5); self.freq_canvas.axes.set_ylabel("Magnitude", color='white')
        self.freq_canvas.axes.set_title(f"FFT Spectrum ({win_type})", color='white')
        self.freq_canvas.axes.axvline(fs/2, color='yellow', linestyle='--', label=f"Nyquist ({fs/2:.1f}Hz)")
        self.freq_canvas.axes.set_xlabel("Frequency (Hz)", color='white'); self.freq_canvas.axes.set_xlim(0, fs/2)
        self.freq_canvas.axes.legend(facecolor='#2b2b2b', edgecolor='#cccccc', labelcolor='white')
        self.freq_canvas.fig.tight_layout(); self.freq_canvas.draw()

        self.spec_canvas.axes.clear()
        nfft = min(256, len(y) // 2) if len(y) > 64 else 64
        if nfft > 0:
            Pxx, freqs, bins, im = self.spec_canvas.axes.specgram(y, NFFT=nfft, Fs=fs, noverlap=nfft//2, cmap='viridis')
            self.spec_canvas.axes.set_title("Spectrogram", color='white')
            self.spec_canvas.axes.set_xlabel("Time (s)", color='white'); self.spec_canvas.axes.set_ylabel("Frequency (Hz)", color='white')
            cbar = self.spec_canvas.fig.colorbar(im, ax=self.spec_canvas.axes)
            cbar.ax.yaxis.set_tick_params(color='#cccccc'); plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')
        self.spec_canvas.fig.tight_layout(); self.spec_canvas.draw()

        self.const_canvas.axes.clear()
        if sig_type in ["BPSK", "QPSK", "16-QAM"]:
            sps = int(self.sample_rate_spin.value() / freq)
            centers = np.arange(sps//2, len(self.last_signal), sps)[:int(freq * dur)]
            I_s = self.last_signal[centers] * np.cos(2 * np.pi * (freq * 2) * centers / self.last_fs)
            Q_s = self.last_signal[centers] * (-np.sin(2 * np.pi * (freq * 2) * centers / self.last_fs))
            self.const_canvas.axes.scatter(I_s, Q_s, c='red', alpha=0.5, s=20)
            self.const_canvas.axes.set_title(f"{sig_type} Constellation (Noisy)", color='white')
            self.const_canvas.axes.set_xlabel("In-Phase (I)", color='white'); self.const_canvas.axes.set_ylabel("Quadrature (Q)", color='white')
            self.const_canvas.axes.axhline(0, color='gray', linestyle='--'); self.const_canvas.axes.axvline(0, color='gray', linestyle='--')
            lim = max_val * 1.5; self.const_canvas.axes.set_xlim(-lim, lim); self.const_canvas.axes.set_ylim(-lim, lim); self.const_canvas.axes.set_aspect('equal', 'box')
        else:
            self.const_canvas.axes.set_title("Constellation (Select Digital Mod)", color='white'); self.const_canvas.axes.axis('off')
        self.const_canvas.fig.tight_layout(); self.const_canvas.draw()

        self.eye_canvas.axes.clear()
        if sig_type in ["BPSK", "QPSK", "16-QAM"]:
            sps = int(self.sample_rate_spin.value() / freq); sig = self.I_baseband; t_eye = np.linspace(-1, 1, 2*sps)
            for i in range(0, len(sig) - 2*sps, sps//2): self.eye_canvas.axes.plot(t_eye, sig[i:i+2*sps], 'b-', alpha=0.1)
            self.eye_canvas.axes.set_title(f"Baseband Eye Diagram ({self.pulse_combo.currentText()})", color='white')
            self.eye_canvas.axes.set_xlabel("Symbol Periods (T)", color='white'); self.eye_canvas.axes.set_ylabel("Amplitude", color='white')
        else:
            self.eye_canvas.axes.set_title("Eye Diagram (Select Digital Mod)", color='white'); self.eye_canvas.axes.axis('off')
        self.eye_canvas.fig.tight_layout(); self.eye_canvas.draw()

        self.fir_canvas.axes.clear()
        if fir_taps is not None:
            # Split into two subplots for Mag and Group Delay
            ax1 = self.fir_canvas.fig.add_subplot(211)
            ax2 = self.fir_canvas.fig.add_subplot(212)
            
            w, h = freqz(fir_taps, 1, worN=2000, fs=self.sample_rate_spin.value())
            ax1.plot(w, 20 * np.log10(np.abs(h) + 1e-12), 'b-', linewidth=2)
            ax1.set_title("FIR Magnitude Response", color='white'); ax1.set_ylabel("Mag (dB)", color='white')
            ax1.set_ylim(-80, 5); ax1.grid(True, color='#3e3e3e', linestyle='--')
            ax1.tick_params(colors='#cccccc')
            for spine in ax1.spines.values(): spine.set_color('#cccccc')
            
            w_gd, gd = group_delay((fir_taps, 1), w=2000, fs=self.sample_rate_spin.value())
            ax2.plot(w_gd, gd, 'r-', linewidth=2)
            ax2.set_title("FIR Group Delay", color='white'); ax2.set_xlabel("Frequency (Hz)", color='white'); ax2.set_ylabel("Delay (samples)", color='white')
            ax2.grid(True, color='#3e3e3e', linestyle='--')
            ax2.tick_params(colors='#cccccc')
            for spine in ax2.spines.values(): spine.set_color('#cccccc')
        else:
            self.fir_canvas.axes.set_title("FIR Response (Enable Filter)", color='white'); self.fir_canvas.axes.axis('off')
        self.fir_canvas.fig.tight_layout(); self.fir_canvas.draw()

# ==========================================================
# Tab 2: Advanced Information Theory & Coding
# ==========================================================
class InformationTheoryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        
        input_group = QGroupBox("Source Data")
        input_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load .txt"); self.load_btn.clicked.connect(self.load_file)
        self.gen_uniform_btn = QPushButton("Gen Uniform"); self.gen_uniform_btn.clicked.connect(lambda: self.text_input.setPlainText("abcdefghijklmnopqrstuvwxyz"))
        self.gen_skew_btn = QPushButton("Gen Skew"); self.gen_skew_btn.clicked.connect(lambda: self.text_input.setPlainText("aaaaaaaaaaaaaaaaaaabbbccde"))
        self.gen_rep_btn = QPushButton("Gen Repetitive"); self.gen_rep_btn.clicked.connect(lambda: self.text_input.setPlainText("aaaaabbbbbcccccaaaaabbbbbccccc"))
        self.calc_btn = QPushButton("Calculate & Encode"); self.calc_btn.setStyleSheet("font-weight: bold; background-color: #007acc; color: white; padding: 8px;")
        self.calc_btn.clicked.connect(self.update_calculation)
        btn_layout.addWidget(self.load_btn); btn_layout.addWidget(self.gen_uniform_btn); btn_layout.addWidget(self.gen_skew_btn); btn_layout.addWidget(self.gen_rep_btn)
        
        self.text_input = QPlainTextEdit(); self.text_input.setPlainText("the quick brown fox jumps over the lazy dog")
        input_layout.addLayout(btn_layout); input_layout.addWidget(self.text_input)
        input_group.setLayout(input_layout)
        
        res_group = QGroupBox("Information Metrics")
        res_layout = QVBoxLayout()
        self.lbl_entropy0 = QLabel("0-Order Entropy (H): 0.0000 bits/symbol")
        self.lbl_entropy1 = QLabel("1-Order Entropy (H|ctx): 0.0000 bits/symbol")
        self.lbl_max_entropy = QLabel("Max Possible Entropy: 0.0000 bits/symbol")
        self.lbl_redundancy = QLabel("Relative Redundancy: 0.00 %")
        self.lbl_huffman_len = QLabel("Avg Huffman Length (L): 0.0000 bits/symbol")
        self.lbl_efficiency = QLabel("Coding Efficiency: 0.00 %")
        self.lbl_kraft = QLabel("Kraft-McMillan Sum: 0.0000")
        for lbl in [self.lbl_entropy0, self.lbl_entropy1, self.lbl_max_entropy, self.lbl_redundancy, self.lbl_huffman_len, self.lbl_efficiency, self.lbl_kraft]:
            lbl.setStyleSheet("font-size: 13px; padding: 2px;"); res_layout.addWidget(lbl)
        res_group.setLayout(res_layout)
        
        chan_group = QGroupBox("Channel & Compression Simulation")
        chan_layout = QVBoxLayout()
        bsc_layout = QFormLayout()
        self.bsc_spin = QDoubleSpinBox(); self.bsc_spin.setRange(0.0, 1.0); self.bsc_spin.setSingleStep(0.01); self.bsc_spin.setValue(0.05)
        bsc_layout.addRow("BSC Flip Prob (p):", self.bsc_spin)
        self.lbl_channel_cap = QLabel("Max Channel Capacity (C): 1.0000 bits/use")
        self.lbl_channel_cap.setStyleSheet("font-size: 13px; padding: 2px; font-weight: bold;")
        bsc_layout.addRow(self.lbl_channel_cap); chan_layout.addLayout(bsc_layout)
        
        self.ber_table = QTableWidget(); self.ber_table.setColumnCount(2)
        self.ber_table.setHorizontalHeaderLabels(["Scheme", "Bit Error Rate (BER)"])
        self.ber_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ber_table.setRowCount(4)
        self.ber_table.setItem(0, 0, QTableWidgetItem("Uncoded")); self.ber_table.setItem(1, 0, QTableWidgetItem("Repetition (1/3)"))
        self.ber_table.setItem(2, 0, QTableWidgetItem("Hamming (7,4)")); self.ber_table.setItem(3, 0, QTableWidgetItem("Convolutional (1/2)"))
        chan_layout.addWidget(self.ber_table)
        
        self.comp_table = QTableWidget(); self.comp_table.setColumnCount(3)
        self.comp_table.setHorizontalHeaderLabels(["Algorithm", "Size (bits)", "Savings vs ASCII"])
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comp_table.setRowCount(4)
        self.comp_table.setItem(0, 0, QTableWidgetItem("ASCII (8-bit)")); self.comp_table.setItem(1, 0, QTableWidgetItem("Huffman Coding"))
        self.comp_table.setItem(2, 0, QTableWidgetItem("RLE (12-bit)")); self.comp_table.setItem(3, 0, QTableWidgetItem("LZW Dictionary"))
        chan_layout.addWidget(self.comp_table)
        chan_group.setLayout(chan_layout)

        controls_layout.addWidget(input_group); controls_layout.addWidget(res_group); controls_layout.addWidget(chan_group); controls_layout.addStretch()

        table_group = QGroupBox("Huffman Codebook")
        table_layout = QVBoxLayout()
        self.table = QTableWidget(); self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Symbol", "Probability", "Info (bits)", "Code"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.table); table_group.setLayout(table_layout)

        plot_tabs = QTabWidget()
        prob_tab = QWidget(); prob_layout = QVBoxLayout(prob_tab)
        prob_controls = QHBoxLayout(); prob_controls.addWidget(QLabel("View:"))
        self.prob_combo = QComboBox(); self.prob_combo.addItems(["PMF (Probability)", "CDF (Cumulative)", "Info Content (-log2 p)"])
        self.prob_combo.currentIndexChanged.connect(self.update_prob_plot); prob_controls.addWidget(self.prob_combo); prob_controls.addStretch()
        prob_layout.addLayout(prob_controls)
        self.prob_canvas = MplCanvas(self, width=5, height=5, dpi=100); prob_layout.addWidget(self.prob_canvas)
        plot_tabs.addTab(prob_tab, "Probability")
        
        comp_tab = QWidget(); comp_layout = QVBoxLayout(comp_tab)
        self.comp_canvas = MplCanvas(self, width=5, height=5, dpi=100); comp_layout.addWidget(self.comp_canvas)
        plot_tabs.addTab(comp_tab, "Compression Stats")
        
        tree_tab = QWidget(); tree_layout = QVBoxLayout(tree_tab)
        self.tree_canvas = MplCanvas(self, width=5, height=5, dpi=100); tree_layout.addWidget(self.tree_canvas)
        plot_tabs.addTab(tree_tab, "Huffman Tree")
        
        markov_tab = QWidget(); markov_layout = QVBoxLayout(markov_tab)
        self.markov_canvas = MplCanvas(self, width=5, height=5, dpi=100); markov_layout.addWidget(self.markov_canvas)
        plot_tabs.addTab(markov_tab, "Markov Matrix")
        
        cap_tab = QWidget(); cap_layout = QVBoxLayout(cap_tab)
        self.cap_canvas = MplCanvas(self, width=5, height=5, dpi=100); cap_layout.addWidget(self.cap_canvas)
        plot_tabs.addTab(cap_tab, "Channel Capacity")
        
        ber_tab = QWidget(); ber_layout = QVBoxLayout(ber_tab)
        self.ber_canvas = MplCanvas(self, width=5, height=5, dpi=100); ber_layout.addWidget(self.ber_canvas)
        plot_tabs.addTab(ber_tab, "BER vs SNR Curve")
        
        rd_tab = QWidget(); rd_layout = QVBoxLayout(rd_tab)
        self.rd_canvas = MplCanvas(self, width=5, height=5, dpi=100); rd_layout.addWidget(self.rd_canvas)
        plot_tabs.addTab(rd_tab, "Rate-Distortion")

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(QLabel("<b>Received Text (After Channel + Decoding):</b>"))
        self.decoded_text = QPlainTextEdit(); self.decoded_text.setReadOnly(True); self.decoded_text.setMaximumHeight(100)
        bottom_layout.addWidget(self.decoded_text)

        right_layout = QVBoxLayout()
        right_layout.addWidget(plot_tabs, 3)
        right_layout.addLayout(bottom_layout, 1)

        self.layout.addLayout(controls_layout, 1); self.layout.addWidget(table_group, 1); self.layout.addLayout(right_layout, 2)
        self.current_probs = {}; self.current_chars = []
        self.update_calculation()

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as f: self.text_input.setPlainText(f.read())
            except Exception as e: self.text_input.setPlainText(f"Error: {e}")

    def build_huffman_tree(self, probabilities):
        if not probabilities: return None
        heap = [[weight, [char, ""]] for char, weight in probabilities.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap); hi = heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p[0]))
        return {char: code for char, code in codes}

    def draw_huffman_tree(self, node, x=0, y=0, dx=1.0):
        if node is None: return
        if node.char is not None:
            color = plt.cm.viridis(node.freq)
            display_char = node.char if node.char.isprintable() and node.char != ' ' else f"[{repr(node.char)[1:-1]}]"
            self.tree_canvas.axes.scatter(x, y, s=100, color=color, zorder=5)
            self.tree_canvas.axes.text(x, y, display_char, color='white', fontsize=9, ha='center', va='center', zorder=6)
        else:
            self.tree_canvas.axes.scatter(x, y, s=30, color='gray', zorder=5)
        if node.left:
            nx, ny = x - dx, y - 1
            self.tree_canvas.axes.plot([x, nx], [y, ny], color='#cccccc', zorder=1)
            self.tree_canvas.axes.text((x+nx)/2, (y+ny)/2, '0', color='cyan', fontsize=8, ha='center')
            self.draw_huffman_tree(node.left, nx, ny, dx/2)
        if node.right:
            nx, ny = x + dx, y - 1
            self.tree_canvas.axes.plot([x, nx], [y, ny], color='#cccccc', zorder=1)
            self.tree_canvas.axes.text((x+nx)/2, (y+ny)/2, '1', color='yellow', fontsize=8, ha='center')
            self.draw_huffman_tree(node.right, nx, ny, dx/2)

    def update_prob_plot(self):
        self.prob_canvas.axes.clear()
        if not self.current_chars: return
        view_mode = self.prob_combo.currentText(); probs = [self.current_probs[c] for c in self.current_chars]
        if view_mode == "CDF (Cumulative)":
            self.prob_canvas.axes.plot(self.current_chars, np.cumsum(probs), 'b-', linewidth=2, marker='o')
            self.prob_canvas.axes.set_ylabel("Cumulative Prob", color='white'); self.prob_canvas.axes.set_title("Cumulative Distribution Function (CDF)", color='white')
        elif view_mode == "Info Content (-log2 p)":
            info = [-np.log2(p) if p > 0 else 0 for p in probs]
            self.prob_canvas.axes.bar(self.current_chars, info, color='#ff5722')
            self.prob_canvas.axes.set_ylabel("Information (bits)", color='white'); self.prob_canvas.axes.set_title("Shannon Information Content", color='white')
        else:
            bars = self.prob_canvas.axes.bar(self.current_chars, probs, color='#4caf50')
            self.prob_canvas.axes.set_ylabel("Probability P(x)", color='white'); self.prob_canvas.axes.set_title("Probability Mass Function (PMF)", color='white')
            for bar in bars:
                yval = bar.get_height()
                self.prob_canvas.axes.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', color='white', fontsize=8)
        self.prob_canvas.axes.set_ylim(0, 1.1); self.prob_canvas.axes.set_xlabel("Character", color='white')
        self.prob_canvas.axes.grid(True, color='#3e3e3e', linestyle='--'); self.prob_canvas.fig.tight_layout(); self.prob_canvas.draw()

    def simulate_ber_curve(self):
        ebn0_db = np.arange(0, 11, 1); ebn0_lin = 10**(ebn0_db / 10)
        from scipy.special import erfc
        ber_uncoded = 0.5 * erfc(np.sqrt(ebn0_lin))
        ber_rep = 0.5 * erfc(np.sqrt(3 * ebn0_lin))
        ber_hamming = 0.5 * erfc(np.sqrt((7/4) * ebn0_lin))
        # Convolutional (K=3) soft bound approx
        ber_conv = 0.5 * erfc(np.sqrt(2 * ebn0_lin)) 
        
        self.ber_canvas.axes.clear()
        self.ber_canvas.axes.semilogy(ebn0_db, ber_uncoded, 'b-o', label="Uncoded BPSK")
        self.ber_canvas.axes.semilogy(ebn0_db, ber_rep, 'g-s', label="Repetition (1/3)")
        self.ber_canvas.axes.semilogy(ebn0_db, ber_hamming, 'r-^', label="Hamming (7,4) Approx")
        self.ber_canvas.axes.semilogy(ebn0_db, ber_conv, 'm-d', label="Convolutional (1/2) Bound")
        self.ber_canvas.axes.set_title("BER vs Eb/N0 (Theoretical Limits)", color='white')
        self.ber_canvas.axes.set_xlabel("Eb/N0 (dB)", color='white'); self.ber_canvas.axes.set_ylabel("Bit Error Rate (BER)", color='white')
        self.ber_canvas.axes.set_ylim(1e-6, 1); self.ber_canvas.axes.set_xlim(0, 10)
        self.ber_canvas.axes.grid(True, which='both', color='#3e3e3e', linestyle='--')
        self.ber_canvas.axes.legend(facecolor='#2b2b2b', edgecolor='#cccccc', labelcolor='white')
        self.ber_canvas.fig.tight_layout(); self.ber_canvas.draw()

    def update_calculation(self):
        text = self.text_input.toPlainText()
        if not text:
            self.table.setRowCount(0)
            for canvas in [self.prob_canvas, self.comp_canvas, self.tree_canvas, self.markov_canvas, self.cap_canvas, self.ber_canvas, self.rd_canvas]:
                canvas.axes.clear(); canvas.draw()
            return

        counts = Counter(text); total_chars = sum(counts.values())
        probabilities = {char: count / total_chars for char, count in counts.items()}
        entropy0 = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        pairs = Counter(zip(text[:-1], text[1:])); single_counts = Counter(text[:-1]); entropy1 = 0.0
        for (x_prev, x_curr), count in pairs.items():
            p_joint = count / (len(text) - 1); p_cond = count / single_counts[x_prev]
            entropy1 -= p_joint * np.log2(p_cond)
            
        alphabet_size = len(probabilities); max_entropy = np.log2(alphabet_size) if alphabet_size > 0 else 0
        redundancy = (1 - (entropy0 / max_entropy)) * 100 if max_entropy > 0 else 0
        
        huffman_codes = self.build_huffman_tree(probabilities)
        avg_code_len = sum(probabilities[char] * len(huffman_codes[char]) for char in probabilities)
        efficiency = (entropy0 / avg_code_len * 100) if avg_code_len > 0 else 0
        kraft_sum = sum(2**(-len(code)) for code in huffman_codes.values())

        p = self.bsc_spin.value()
        H_p = -p * np.log2(p) - (1-p) * np.log2(1-p) if 0 < p < 1 else 0
        channel_cap = 1 - H_p

        full_encoded = "".join(huffman_codes[c] for c in text)
        orig_bits = np.array([int(b) for b in full_encoded], dtype=int)
        noise_mask = (np.random.rand(len(orig_bits)) < p).astype(int)
        
        uncoded_rx = (orig_bits + noise_mask) % 2; uncoded_ber = np.mean(orig_bits != uncoded_rx) if len(orig_bits) > 0 else 0
        
        rep_coded = np.repeat(orig_bits, 3); rep_noise = (np.random.rand(len(rep_coded)) < p).astype(int)
        rep_rx = (rep_coded + rep_noise) % 2; rep_decoded = (np.sum(rep_rx.reshape(-1, 3), axis=1) > 1).astype(int)
        rep_ber = np.mean(orig_bits != rep_decoded) if len(orig_bits) > 0 else 0
        
        hamming_ber = 0.0; hamming_decoded = orig_bits
        if len(orig_bits) >= 4:
            pad_len = (4 - len(orig_bits) % 4) % 4; padded_bits = np.pad(orig_bits, (0, pad_len), 'constant')
            coded_bits = hamming_encode(padded_bits.reshape(-1, 4))
            coded_noise = (np.random.rand(len(coded_bits)) < p).astype(int); coded_rx = (coded_bits + coded_noise) % 2
            hamming_decoded = hamming_decode(coded_rx).flatten()[:len(orig_bits)]
            hamming_ber = np.mean(orig_bits != hamming_decoded)
            
        # Convolutional Code (limit to 2000 bits for Viterbi speed)
        conv_ber = 0.0; conv_decoded = orig_bits
        sim_len = min(len(orig_bits), 2000)
        if sim_len > 0:
            sim_bits = orig_bits[:sim_len]
            conv_coded = conv_encode(sim_bits)
            conv_noise = (np.random.rand(len(conv_coded)) < p).astype(int)
            conv_rx = (conv_coded + conv_noise) % 2
            conv_decoded = viterbi_decode(conv_rx)
            conv_ber = np.mean(sim_bits != conv_decoded)
        
        # Decode Text (Use best available)
        best_decoded_bits = conv_decoded if conv_ber == 0 else (hamming_decoded if hamming_ber == 0 else (rep_decoded if rep_ber < uncoded_ber else uncoded_rx))
        decoded_str = "".join(str(b) for b in best_decoded_bits)
        decoded_text = huffman_decode(decoded_str, huffman_codes)
        if len(decoded_text) > 500: decoded_text = decoded_text[:500] + "..."
        self.decoded_text.setPlainText(decoded_text)

        self.lbl_entropy0.setText(f"0-Order Entropy (H): {entropy0:.4f} bits/symbol")
        self.lbl_entropy1.setText(f"1-Order Entropy (H|ctx): {entropy1:.4f} bits/symbol")
        self.lbl_max_entropy.setText(f"Max Possible Entropy: {max_entropy:.4f} bits/symbol")
        self.lbl_redundancy.setText(f"Relative Redundancy: {redundancy:.2f} %")
        self.lbl_huffman_len.setText(f"Avg Huffman Length (L): {avg_code_len:.4f} bits/symbol")
        self.lbl_efficiency.setText(f"Coding Efficiency: {efficiency:.2f} %")
        self.lbl_kraft.setText(f"Kraft-McMillan Sum: {kraft_sum:.4f} (Should be 1.0)")
        self.lbl_channel_cap.setText(f"Max Channel Capacity (C): {channel_cap:.4f} bits/use")
        
        self.ber_table.setItem(0, 1, QTableWidgetItem(f"{uncoded_ber:.4f} ({uncoded_ber*100:.2f}%)"))
        self.ber_table.setItem(1, 1, QTableWidgetItem(f"{rep_ber:.4f} ({rep_ber*100:.2f}%)"))
        self.ber_table.setItem(2, 1, QTableWidgetItem(f"{hamming_ber:.4f} ({hamming_ber*100:.2f}%)"))
        self.ber_table.setItem(3, 1, QTableWidgetItem(f"{conv_ber:.4f} ({conv_ber*100:.2f}%)"))
        
        ascii_bits = len(text) * 8; huff_bits = len(full_encoded); rle_runs = 1
        for i in range(1, len(text)):
            if text[i] != text[i-1]: rle_runs += 1
        rle_bits = rle_runs * 12; lzw_bits = calculate_lzw_bits(text)
        
        self.comp_table.setItem(0, 1, QTableWidgetItem(f"{ascii_bits} bits")); self.comp_table.setItem(1, 1, QTableWidgetItem(f"{huff_bits} bits"))
        self.comp_table.setItem(2, 1, QTableWidgetItem(f"{rle_bits} bits")); self.comp_table.setItem(3, 1, QTableWidgetItem(f"{lzw_bits} bits"))
        self.comp_table.setItem(0, 2, QTableWidgetItem("0.00 %"))
        self.comp_table.setItem(1, 2, QTableWidgetItem(f"{(1 - huff_bits/ascii_bits)*100:.2f} %" if ascii_bits > 0 else "0%"))
        self.comp_table.setItem(2, 2, QTableWidgetItem(f"{(1 - rle_bits/ascii_bits)*100:.2f} %" if ascii_bits > 0 else "0%"))
        self.comp_table.setItem(3, 2, QTableWidgetItem(f"{(1 - lzw_bits/ascii_bits)*100:.2f} %" if ascii_bits > 0 else "0%"))
        
        self.table.setRowCount(alphabet_size)
        for i, (char, p) in enumerate(sorted(probabilities.items(), key=lambda item: item[1], reverse=True)):
            info = -np.log2(p) if p > 0 else 0
            display_char = char if char.isprintable() and char != ' ' else repr(char)[1:-1]
            self.table.setItem(i, 0, QTableWidgetItem(display_char)); self.table.setItem(i, 1, QTableWidgetItem(f"{p:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{info:.4f}")); self.table.setItem(i, 3, QTableWidgetItem(huffman_codes[char]))
        
        sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        self.current_chars = [item[0] if item[0].isprintable() and item[0] != ' ' else f"[{repr(item[0])[1:-1]}]" for item in sorted_items]
        self.current_probs = {self.current_chars[i]: item[1] for i, item in enumerate(sorted_items)}
        self.update_prob_plot()

        # Compression Bar Chart
        self.comp_canvas.axes.clear()
        algos = ["ASCII", "Huffman", "RLE", "LZW"]
        sizes = [ascii_bits, huff_bits, rle_bits, lzw_bits]
        colors = ['#dc3545', '#007acc', '#ffc107', '#28a745']
        bars = self.comp_canvas.axes.bar(algos, sizes, color=colors)
        self.comp_canvas.axes.set_title("Compression Algorithm Comparison", color='white')
        self.comp_canvas.axes.set_ylabel("Size (bits)", color='white')
        for bar in bars:
            yval = bar.get_height()
            self.comp_canvas.axes.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval}", ha='center', va='bottom', color='white', fontsize=10)
        self.comp_canvas.axes.grid(True, color='#3e3e3e', linestyle='--', axis='y')
        self.comp_canvas.fig.tight_layout(); self.comp_canvas.draw()

        self.tree_canvas.axes.clear()
        self.draw_huffman_tree(build_huffman_tree_node(probabilities), x=0, y=0, dx=alphabet_size * 0.5)
        self.tree_canvas.axes.set_title("Huffman Binary Tree", color='white'); self.tree_canvas.axes.axis('off')
        self.tree_canvas.fig.tight_layout(); self.tree_canvas.draw()

        self.markov_canvas.axes.clear()
        sorted_chars = sorted(probabilities.keys()); matrix = np.zeros((alphabet_size, alphabet_size))
        char_to_idx = {c: i for i, c in enumerate(sorted_chars)}
        for (x_prev, x_curr), count in pairs.items():
            if x_prev in char_to_idx and x_curr in char_to_idx:
                matrix[char_to_idx[x_prev], char_to_idx[x_curr]] = count / single_counts[x_prev]
        im = self.markov_canvas.axes.imshow(matrix, cmap='magma', interpolation='nearest')
        self.markov_canvas.axes.set_xticks(range(alphabet_size)); self.markov_canvas.axes.set_yticks(range(alphabet_size))
        disp_chars = [c if c.isprintable() and c != ' ' else f"[{repr(c)[1:-1]}]" for c in sorted_chars]
        self.markov_canvas.axes.set_xticklabels(disp_chars, rotation=45); self.markov_canvas.axes.set_yticklabels(disp_chars)
        self.markov_canvas.axes.set_title("1st-Order Markov Transition Matrix", color='white')
        self.markov_canvas.axes.set_xlabel("Next Character", color='white'); self.markov_canvas.axes.set_ylabel("Current Character", color='white')
        cbar = self.markov_canvas.fig.colorbar(im, ax=self.markov_canvas.axes)
        cbar.ax.yaxis.set_tick_params(color='#cccccc'); plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cccccc')
        self.markov_canvas.fig.tight_layout(); self.markov_canvas.draw()

        self.cap_canvas.axes.clear()
        p_range = np.linspace(0.001, 0.999, 100)
        I_xy_range = 1 - (-p_range * np.log2(p_range) - (1-p_range) * np.log2(1-p_range))
        self.cap_canvas.axes.plot(p_range, I_xy_range, 'b-', linewidth=2, label="Mutual Info I(X;Y)")
        self.cap_canvas.axes.axvline(p, color='red', linestyle='--', label=f"Current p = {p:.2f}")
        self.cap_canvas.axes.scatter([p], [channel_cap], color='red', s=50, zorder=5)
        self.cap_canvas.axes.set_title("BSC Mutual Information (Channel Capacity)", color='white')
        self.cap_canvas.axes.set_xlabel("Crossover Probability (p)", color='white'); self.cap_canvas.axes.set_ylabel("Mutual Information (bits/use)", color='white')
        self.cap_canvas.axes.set_xlim(0, 1); self.cap_canvas.axes.set_ylim(0, 1.05)
        self.cap_canvas.axes.grid(True, color='#3e3e3e', linestyle='--')
        self.cap_canvas.axes.legend(facecolor='#2b2b2b', edgecolor='#cccccc', labelcolor='white')
        self.cap_canvas.fig.tight_layout(); self.cap_canvas.draw()
        
        self.simulate_ber_curve()
        
        self.rd_canvas.axes.clear()
        sigma2 = 1.0; D = np.linspace(0.01, sigma2, 100); R_D = 0.5 * np.log2(sigma2 / D)
        self.rd_canvas.axes.plot(D, R_D, 'm-', linewidth=2, label="R(D) for Gaussian Source")
        self.rd_canvas.axes.set_title("Rate-Distortion Theory", color='white')
        self.rd_canvas.axes.set_xlabel("Distortion (MSE)", color='white'); self.rd_canvas.axes.set_ylabel("Rate (bits/sample)", color='white')
        self.rd_canvas.axes.set_xlim(0, sigma2); self.rd_canvas.axes.set_ylim(0, np.max(R_D)*1.1)
        self.rd_canvas.axes.grid(True, color='#3e3e3e', linestyle='--')
        self.rd_canvas.axes.legend(facecolor='#2b2b2b', edgecolor='#cccccc', labelcolor='white')
        self.rd_canvas.fig.tight_layout(); self.rd_canvas.draw()

# ==========================================================
# Main Application Window
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP & Information Theory Studio v13.0")
        self.resize(1500, 950)
        tabs = QTabWidget()
        tabs.addTab(SignalAnalysisTab(), "Signal Analysis (DSP)")
        tabs.addTab(InformationTheoryTab(), "Information Theory & Coding")
        self.setCentralWidget(tabs)
        self.apply_dark_theme()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Try 16-QAM and Convolutional Coding!")

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 45)); dark_palette.setColor(QPalette.WindowText, QColor(208, 208, 208))
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30)); dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(208, 208, 208)); dark_palette.setColor(QPalette.ToolTipText, QColor(208, 208, 208))
        dark_palette.setColor(QPalette.Text, QColor(208, 208, 208)); dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ButtonText, QColor(208, 208, 208)); dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218)); dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(dark_palette); app.setStyle("Fusion")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())