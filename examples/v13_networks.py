import numpy as np
import genny

sr=48000
clar=genny.MultiHoleClarinet(220,sr=sr)
fing=np.zeros((sr,len(clar.bore.holes)))
fing[sr//2:,-2:]=1.0
audio, reed, bore=clar.note(1.0,fingering=fing)
print('clarinet', audio.shape, reed.effort_unit, bore.flow_unit)

strings=genny.CoupledStringBank((440.0,440.7,439.3),sr=sr)
print('coupled strings', strings.render(1.0).shape)

cavity=genny.AcousticCavityNetwork(volume_m3=.025,sr=sr)
print('cavity', cavity.render_impulse(4096).shape)
