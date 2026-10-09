import genny

sr = 48000
r = genny.CommonPhysicalSolver.reed_bore(220.0, 1.0, breath=.7, sr=sr)
print('samples:', len(r.audio))
print('mouth pressure unit:', r.drive.effort_unit)
print('flow unit:', r.drive.flow_unit)
print('solver:', r.metadata['solver'])
