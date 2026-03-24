from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

app = Flask(__name__)
CORS(app)

N = 512

def make_t():
    return np.linspace(0, 1, N)

@app.route('/message', methods=['POST'])
def message():
    data = request.json
    msg = data.get('message', 'HELLO')[:10]
    bits = []
    for ch in msg:
        bits.extend([int(x) for x in format(ord(ch), '08b')])
    t = make_t()
    spb = N // max(len(bits), 1)
    baseband = np.zeros(N)
    for i, b in enumerate(bits):
        s = i * spb
        e = min(s + spb, N)
        baseband[s:e] = b
    return jsonify({
        't': t.tolist(),
        'bits': bits,
        'baseband': baseband.tolist(),
        'binary_str': ' '.join(''.join(format(ord(c),'08b')) for c in msg),
        'ones': int(sum(bits)),
        'zeros': int(len(bits) - sum(bits)),
        'total': len(bits)
    })

@app.route('/modulate', methods=['POST'])
def modulate():
    data = request.json
    msg        = data.get('message', 'HELLO')[:10]
    mod_type   = data.get('mod_type', 'AM')
    f_carrier  = float(data.get('f_carrier', 10))
    amplitude  = float(data.get('amplitude', 1.0))
    noise      = float(data.get('noise', 0.0))
    t = make_t()
    bits = []
    for ch in msg:
        bits.extend([int(x) for x in format(ord(ch), '08b')])
    spb = N // max(len(bits), 1)
    baseband = np.zeros(N)
    for i, b in enumerate(bits):
        s = i * spb; e = min(s + spb, N)
        baseband[s:e] = b
    carrier = amplitude * np.sin(2 * np.pi * f_carrier * t)
    if mod_type == 'AM':
        modulated = (1 + baseband) * carrier
    elif mod_type == 'FM':
        phase = 2*np.pi*f_carrier*t + 2*np.pi*3*np.cumsum(baseband)/N
        modulated = amplitude * np.sin(phase)
    else:
        modulated = amplitude * np.sin(2*np.pi*f_carrier*t + np.pi*baseband)
    noisy = modulated + np.random.normal(0, noise, N) if noise > 0 else modulated
    snr = round(float(10*np.log10(np.mean(modulated**2)/max(noise**2,1e-10))), 2) if noise > 0 else 99.0
    return jsonify({
        't': t.tolist(),
        'baseband': baseband.tolist(),
        'carrier': carrier.tolist(),
        'modulated': noisy.tolist(),
        'modulated_clean': modulated.tolist(),
        'snr': snr,
        'bits': bits
    })

@app.route('/demodulate', methods=['POST'])
def demodulate():
    data = request.json
    modulated   = np.array(data.get('modulated', []))
    f_signal    = float(data.get('f_signal', 10))
    f_offset    = float(data.get('f_offset', 1))
    phase_diff  = float(data.get('phase_diff', 0))
    if len(modulated) == 0:
        return jsonify({'error': 'no signal'})
    t = np.linspace(0, 1, len(modulated))
    lo_h = np.sin(2*np.pi*f_signal*t + phase_diff)
    lo_e = np.sin(2*np.pi*(f_signal+f_offset)*t)
    mix_h = modulated * lo_h
    mix_e = modulated * lo_e
    # LPF
    alpha = 0.04
    lpf = np.zeros(len(modulated))
    lpf[0] = mix_h[0]
    for i in range(1, len(modulated)):
        lpf[i] = alpha*mix_h[i] + (1-alpha)*lpf[i-1]
    # BPF envelope
    win = 15
    bpf = np.zeros(len(modulated))
    for i in range(len(modulated)):
        seg = mix_e[max(0,i-win):i+1]
        env = np.sqrt(np.mean(seg**2))*np.sqrt(2)
        bpf[i] = env * np.sin(2*np.pi*f_offset*t[i])
    I_h = float(np.mean(lpf[-80:]))
    Q_h = float(np.mean(lpf[-80:]*np.cos(np.pi/2)))
    I_e = float(np.mean(bpf[-80:]*np.cos(2*np.pi*f_offset*t[-80:])))
    Q_e = float(np.mean(bpf[-80:]*np.sin(2*np.pi*f_offset*t[-80:])))
    return jsonify({
        't': t.tolist(),
        'lo_homo': lo_h.tolist(),
        'lo_hetero': lo_e.tolist(),
        'mixed_homo': mix_h.tolist(),
        'mixed_hetero': mix_e.tolist(),
        'lpf': lpf.tolist(),
        'bpf': bpf.tolist(),
        'I_homo': round(I_h,4), 'Q_homo': round(Q_h,4),
        'I_hetero': round(I_e,4), 'Q_hetero': round(Q_e,4)
    })

@app.route('/qubit', methods=['POST'])
def qubit():
    data   = request.json
    theta  = float(data.get('theta', 90))
    phi    = float(data.get('phi', 0))
    shots  = int(data.get('shots', 100))
    tr, pr = np.radians(theta), np.radians(phi)
    qc = QuantumCircuit(1)
    qc.ry(tr, 0); qc.rz(pr, 0)
    state = Statevector(qc)
    probs = state.probabilities()
    p0, p1 = float(probs[0]), float(probs[1])
    results = np.random.choice([0,1], size=shots, p=[p0,p1])
    bx = float(np.sin(tr)*np.cos(pr))
    by = float(np.sin(tr)*np.sin(pr))
    bz = float(np.cos(tr))
    iq0 = np.random.normal([0.5, 0.1], 0.07, (50,2)).tolist()
    iq1 = np.random.normal([-0.5,-0.1], 0.07, (50,2)).tolist()
    return jsonify({
        'p0': round(p0,4), 'p1': round(p1,4),
        'c0': int(np.sum(results==0)), 'c1': int(np.sum(results==1)),
        'bx': round(bx,4), 'by': round(by,4), 'bz': round(bz,4),
        'iq0': iq0, 'iq1': iq1,
        'I_meas': round(bz*0.5+np.random.normal(0,0.04),4),
        'Q_meas': round(bx*0.2+np.random.normal(0,0.04),4),
        'shots': results.tolist()
    })

@app.route('/collapse', methods=['POST'])
def collapse():
    data = request.json
    p0 = float(data.get('p0', 0.5))
    p1 = float(data.get('p1', 0.5))
    r  = int(np.random.choice([0,1], p=[p0,p1]))
    return jsonify({'result': r, 'state': f'|{r}⟩'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
